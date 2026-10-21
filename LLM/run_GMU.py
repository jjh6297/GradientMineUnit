import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HOME"] = "/mnt/disk1/MineGradient/.cache"
os.environ["PYTHONUNBUFFERED"] = "1"

import gc
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


# ============================================================
# Configuration
# ============================================================
MODEL_ID = "Qwen/Qwen3-1.7B"
TASKS = ["scienceqa"]
TRIALS = [1]
BASE_SEED = 43

EPSILON = 1e-20
OMEGA = 1e7
K = 8
NUM_LAYERS = 4
GMU_SELECTION = "low_norm"  # paper: low-norm or random native slots
CAMOUFLAGE_SCALE = 1e7
CAMOUFLAGE_NUM_CHANNELS = K
CAMOUFLAGE_KIND = "pseudo_ffn"

NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03


TASK_MAX_LENGTH = {
    "sst2": 256,
    "scienceqa": 512,
    "logiqa": 1024,
}


TASK_BATCH = {
    "sst2": (4, 8),       # micro-batch, grad accumulation
    "scienceqa": (2, 16),
    "logiqa": (1, 32),
}

EVAL_ALL_TASKS_AFTER_FT = False
SAVE_MODELS = False


# ============================================================
# Logging
# ============================================================
class IterationLossCallback(TrainerCallback):
    def __init__(self, task_name: str, trial_idx: int):
        self.losses = []
        self.save_name = f"loss_GMU_{task_name}_trial{trial_idx}.json"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append({
                "step": int(state.global_step),
                "loss": float(logs["loss"]),
            })
            with open(self.save_name, "w") as f:
                json.dump(self.losses, f, indent=2)


# ============================================================
# Paper-aligned GMU: reuse original SwiGLU neuron slots
# ============================================================


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    if hasattr(base, "layers"):
        return base.layers
    raise ValueError("Could not find Qwen3 decoder layers")


def _last_n_indices(model: nn.Module, count: int) -> list[int]:
    layers = _get_decoder_layers(model)
    if not 0 < count <= len(layers):
        raise ValueError(f"NUM_LAYERS={count} invalid for {len(layers)} layers")
    return list(range(len(layers) - count, len(layers)))


@torch.no_grad()
def inject_gmu_reuse(
    model: nn.Module,
    target_layers: list[int],
    k: int = K,
    omega: float = OMEGA,
    epsilon: float = EPSILON,
    selection: str = GMU_SELECTION,
) -> dict[int, list[int]]:
    """In-place K-channel reuse; pairwise excitation/readout; zero gate.

    For each pair (i, i+K/2):
        gate_proj.weight[i] = gate_proj.weight[i+K/2] = 0
        up_proj.weight[i]   = up_proj.weight[i+K/2]   = omega * P[i]
        down_proj.weight[:,i] = -down_proj.weight[:,i+K/2] = epsilon * A[i]
    No additional neurons, dimensions, forward wrappers, or Parameters.
    """
    if k < 2 or k % 2:
        raise ValueError("GMU K must be an even integer >= 2")
    if not omega > 0 or not epsilon > 0:
        raise ValueError("omega and epsilon must be positive")

    records: dict[int, list[int]] = {}
    layers = _get_decoder_layers(model)
    for idx in target_layers:
        mlp = layers[idx].mlp
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        dim = up.out_features
        if gate.out_features != dim or down.in_features != dim or k > dim:
            raise ValueError(f"Layer {idx}: incompatible SwiGLU shapes or K={k}")
        if gate.in_features != up.in_features:
            raise ValueError(f"Layer {idx}: gate/up input dimensions differ")

        if selection == "low_norm":
            scores = down.weight.detach().float().norm(dim=0)
            selected = torch.topk(scores, k, largest=False).indices
        elif selection == "random":
            selected = torch.randperm(dim, device=up.weight.device)[:k]
        else:
            raise ValueError(f"Unknown GMU_SELECTION={selection}")
        selected = selected.to(device=up.weight.device, dtype=torch.long)
        first, second = selected[: k // 2], selected[k // 2 :]

        templates = torch.rand(
            k // 2, up.in_features,
            device=up.weight.device, dtype=up.weight.dtype,
        ) * omega
        readout = torch.rand(
            down.out_features, k // 2,
            device=down.weight.device, dtype=down.weight.dtype,
        ) * epsilon
        gate.weight[selected, :] = 0
        if gate.bias is not None:
            gate.bias[selected] = 0
        up.weight[first, :] = templates
        up.weight[second, :] = templates
        if up.bias is not None:
            up.bias[selected] = 0
        down.weight[:, first] = readout
        down.weight[:, second] = -readout

        assert torch.equal(up.weight[first], up.weight[second])
        assert torch.equal(down.weight[:, first], -down.weight[:, second])
        records[int(idx)] = selected.detach().cpu().tolist()
        print(f"[GMU:reuse] block={idx}, K={k}, omega={omega:g}, "
              f"epsilon={epsilon:g}, channels={records[idx]}")
    return records


@torch.no_grad()
def apply_pseudo_equiv_ffn_camouflage(
    model: nn.Module,
    target_layers: list[int],
    gmu_indices: dict[int, list[int]],
    num_channels: int,
    scale: float,
    seed: int,
) -> dict[int, list[int]]:
    """Paper Eq. (12) applied to native SwiGLU up/down channels.

    Co-located in GMU blocks, strictly excludes all replaced GMU channels.
    For SwiGLU (gate stays fixed) this is exactly function preserving in
    real arithmetic, despite the generic paper term 'pseudo-equivalent'.
    """
    if not scale > 0:
        raise ValueError("scale must be positive")
    records: dict[int, list[int]] = {}
    for idx in target_layers:
        mlp = _get_decoder_layers(model)[idx].mlp
        up, down = mlp.up_proj, mlp.down_proj
        blocked = set(gmu_indices[idx])
        allowed = [i for i in range(up.out_features) if i not in blocked]
        if num_channels > len(allowed) or num_channels < 0:
            raise ValueError("Invalid number of native camouflage channels")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + idx * 100003)
        perm = torch.randperm(len(allowed), generator=generator)[:num_channels]
        selected = torch.tensor(
            [allowed[int(i)] for i in perm],
            device=up.weight.device, dtype=torch.long,
        )
        if len(blocked.intersection(selected.tolist())):
            raise AssertionError("GMU/native camouflage overlap")
        up.weight[selected, :] *= scale
        if up.bias is not None:
            up.bias[selected] *= scale
        down.weight[:, selected] /= scale
        records[int(idx)] = selected.detach().cpu().tolist()
        print(f"[PseudoEquiv:FFN] block={idx}, s={scale:g}, "
              f"native_channels={records[idx]}")
    return records


@torch.no_grad()
def apply_equiv_qk_camouflage(
    model: nn.Module,
    target_layers: list[int],
    scale: float,
) -> dict[int, dict]:
    """Q/K reciprocal reparameterization adapted to Qwen3's QK RMSNorm.

    In Qwen3, Q/K projections are followed by q_norm/k_norm (RMSNorm)
    and RoPE. Scaling ONLY q_proj/k_proj as in a plain ViT would NOT
    preserve attention, especially at s=1e7. We therefore also apply:
      q_norm.weight *= s; k_norm.weight /= s;
      q_norm.eps *= s**2; k_norm.eps /= s**2.
    This produces (effective Q, K) -> (s Q, K/s) AFTER RMSNorm,
    preserving QK^T (including RoPE and grouped-query attention) up to
    numerical precision. No extra trainable Parameters are added.

    IMPORTANT: variance_epsilon is a runtime attribute in Transformers;
    preserve it when saving/reloading a camouflaged checkpoint. See README.

    Because each RMSNorm is shared by all heads of the same attention block,
    this method transforms *all* Q/K channels in a selected decoder block.
    Partial per-channel high-scale scaling is not inference-equivalent in
    standard Qwen3 without replacing the normalization implementation.
    """
    if not scale > 0:
        raise ValueError("scale must be positive")
    records: dict[int, dict] = {}
    for idx in target_layers:
        attn = _get_decoder_layers(model)[idx].self_attn
        for attr in ("q_proj", "k_proj", "q_norm", "k_norm"):
            if not hasattr(attn, attr):
                raise TypeError(f"Block {idx}: missing {attr} (expected Qwen3 attention)")
        q, k = attn.q_proj, attn.k_proj
        qn, kn = attn.q_norm, attn.k_norm
        if not hasattr(qn, "variance_epsilon") or not hasattr(kn, "variance_epsilon"):
            raise TypeError("Qwen3 q_norm/k_norm must expose variance_epsilon")
        qeps_old = float(qn.variance_epsilon)
        keps_old = float(kn.variance_epsilon)
        q.weight.mul_(scale)
        if q.bias is not None:
            q.bias.mul_(scale)
        k.weight.div_(scale)
        if k.bias is not None:
            k.bias.div_(scale)
        qn.weight.mul_(scale)
        kn.weight.div_(scale)
        qn.variance_epsilon = qeps_old * (scale ** 2)
        kn.variance_epsilon = keps_old / (scale ** 2)
        records[int(idx)] = {
            "q_norm_eps": float(qn.variance_epsilon),
            "k_norm_eps": float(kn.variance_epsilon),
            "q_proj_rows": int(q.out_features),
            "k_proj_rows": int(k.out_features),
            "scale": float(scale),
        }
        print(f"[Equiv:QK] block={idx}, s={scale:g}, "
              f"q_rows={q.out_features}, k_rows={k.out_features}, "
              f"q_eps={qn.variance_epsilon:g}, k_eps={kn.variance_epsilon:g}")
    return records


def save_qk_runtime_metadata(save_dir: str, qk_records: dict[int, dict]):
    """Save non-parameter Q/K normalization epsilons along with weights."""
    with open(os.path.join(save_dir, "qk_rmsnorm_runtime.json"), "w") as f:
        json.dump({str(k): v for k, v in qk_records.items()}, f, indent=2)


def restore_qk_runtime_metadata(model: nn.Module, save_dir: str):
    """Call after from_pretrained(save_dir) to restore layer-specific eps."""
    with open(os.path.join(save_dir, "qk_rmsnorm_runtime.json")) as f:
        records = json.load(f)
    layers = _get_decoder_layers(model)
    for key, entry in records.items():
        attn = layers[int(key)].self_attn
        attn.q_norm.variance_epsilon = float(entry["q_norm_eps"])
        attn.k_norm.variance_epsilon = float(entry["k_norm_eps"])
    return model

# ============================================================
# Dataset loading / schema normalization
# ============================================================
def load_task_dataset(task_name: str, split: str) -> Dataset:
    if task_name == "sst2":
        return load_dataset("glue", "sst2", split=split)

    if task_name == "scienceqa":
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        return ds.filter(lambda x: x["image"] is None)

    if task_name == "logiqa":
        try:
            return load_dataset("lucasmccabe/logiqa", split=split)
        except (RuntimeError, ValueError) as e:
            print(f"[LogiQA] Standard loader failed ({type(e).__name__}); using parquet fallback.")
            parquet_uri = (
                "hf://datasets/lucasmccabe/logiqa@refs/convert/parquet/"
                f"default/{split}/0000.parquet"
            )
            return load_dataset("parquet", data_files=parquet_uri, split="train")

    raise ValueError(f"Unknown task: {task_name}")


def _logiqa_question(ex: Dict) -> str:
    if "query" in ex and ex["query"] is not None:
        return str(ex["query"])
    if "question" in ex and ex["question"] is not None:
        return str(ex["question"])
    raise KeyError("LogiQA example has neither 'query' nor 'question'.")


def _normalize_letter(value, num_choices: int = 4) -> str:
    letters = [chr(ord("A") + i) for i in range(num_choices)]

    if isinstance(value, (int, np.integer)):
        idx = int(value)
        if 0 <= idx < num_choices:
            return letters[idx]

    if isinstance(value, str):
        v = value.strip()
        if v.upper() in letters:
            return v.upper()
        if v.isdigit():
            idx = int(v)
            if 0 <= idx < num_choices:
                return letters[idx]
            if 1 <= idx <= num_choices:
                return letters[idx - 1]

    raise ValueError(f"Cannot convert answer value {value!r} to a choice letter.")


def _logiqa_answer(ex: Dict) -> str:
    for key in ("correct_option", "label", "answer"):
        if key in ex and ex[key] is not None:
            return _normalize_letter(ex[key], num_choices=len(ex["options"]))
    raise KeyError("LogiQA example has no recognized answer field.")


def format_prompt_and_answer(task_name: str, ex: Dict) -> Tuple[str, str]:
    """Return prompt WITHOUT the answer and the gold answer separately."""
    if task_name == "sst2":
        prompt = (
            "Classify the sentiment of the following sentence as 0 (negative) "
            "or 1 (positive).\n"
            f"Sentence: {ex['sentence']}\n"
            "Answer:"
        )
        answer = str(int(ex["label"]))
        return prompt, answer

    if task_name == "scienceqa":
        choices = list(ex["choices"])
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        prompt = (
            "Answer the following science question by choosing the correct option letter.\n"
            f"Question: {ex['question']}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )
        answer = letters[int(ex["answer"])]
        return prompt, answer

    if task_name == "logiqa":
        choices = list(ex["options"])
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        prompt = (
            "Read the context and question, then choose the correct option letter.\n"
            f"Context: {ex['context']}\n"
            f"Question: {_logiqa_question(ex)}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )
        answer = _logiqa_answer(ex)
        return prompt, answer

    raise ValueError(f"Unknown task: {task_name}")


# ============================================================
# Answer-only supervised fine-tuning
# ============================================================
def encode_sft_example(ex: Dict, tokenizer, task_name: str) -> Dict[str, List[int]]:
    prompt, answer = format_prompt_and_answer(task_name, ex)

    answer_text = " " + answer
    if tokenizer.eos_token is not None:
        answer_text += tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    max_length = TASK_MAX_LENGTH[task_name]
    if len(answer_ids) >= max_length:
        raise ValueError(f"Answer itself exceeds max_length for {task_name}.")

    max_prompt_len = max_length - len(answer_ids)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + answer_ids
    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(prompt_ids) + answer_ids.copy()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class AnswerOnlyDataCollator:
    tokenizer: object

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )

        seq_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            pad_len = seq_len - len(f["labels"])
            if self.tokenizer.padding_side == "right":
                padded = f["labels"] + [-100] * pad_len
            else:
                padded = [-100] * pad_len + f["labels"]
            labels.append(padded)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# ============================================================
# Multiple-choice evaluation: score answer tokens only
# ============================================================
def get_candidates_and_label(task_name: str, ex: Dict) -> Tuple[List[str], str]:
    if task_name == "sst2":
        return ["0", "1"], str(int(ex["label"]))

    if task_name == "scienceqa":
        choices = list(ex["choices"])
        candidates = [chr(ord("A") + i) for i in range(len(choices))]
        return candidates, candidates[int(ex["answer"])]

    if task_name == "logiqa":
        candidates = [chr(ord("A") + i) for i in range(len(ex["options"]))]
        return candidates, _logiqa_answer(ex)

    raise ValueError(task_name)


@torch.no_grad()
def candidate_mean_nll(model, tokenizer, prompt: str, candidate: str) -> float:
    """Mean NLL over candidate answer tokens only; prompt tokens are ignored."""
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]
    answer_ids = tokenizer(
        " " + candidate,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    prompt_len = prompt_ids.shape[1]
    logits = outputs.logits[:, prompt_len - 1 : -1, :].contiguous()
    targets = input_ids[:, prompt_len:].contiguous()

    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="mean",
    )
    return float(loss.item())


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, task_name: str, split: str) -> Dict[str, float]:
    ds = load_task_dataset(task_name, split)
    model.eval()

    correct = 0
    for ex in ds:
        prompt, _ = format_prompt_and_answer(task_name, ex)
        candidates, gold = get_candidates_and_label(task_name, ex)

        losses = [
            candidate_mean_nll(model, tokenizer, prompt, candidate)
            for candidate in candidates
        ]
        pred = candidates[int(np.argmin(losses))]
        correct += int(pred == gold)

    return {
        "accuracy": 100.0 * correct / len(ds),
        "correct": int(correct),
        "total": int(len(ds)),
    }


def eval_split(task_name: str) -> str:
    return "validation" if task_name == "sst2" else "test"


# ============================================================
# Model helpers
# ============================================================

def load_paper_aligned_model(seed: int):
    """Fresh Qwen3; reuse K channels in LAST FOUR MLP blocks; camouflage."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    target_layers = _last_n_indices(model, NUM_LAYERS)

    gmu_indices = inject_gmu_reuse(
        model, target_layers, k=K, omega=OMEGA,
        epsilon=EPSILON, selection=GMU_SELECTION,
    )

    if CAMOUFLAGE_KIND == "equiv_qk":
        camouflage_indices = apply_equiv_qk_camouflage(
            model, target_layers=target_layers, scale=CAMOUFLAGE_SCALE,
        )
    elif CAMOUFLAGE_KIND == "pseudo_ffn":
        camouflage_indices = apply_pseudo_equiv_ffn_camouflage(
            model, target_layers=target_layers, gmu_indices=gmu_indices,
            num_channels=CAMOUFLAGE_NUM_CHANNELS,
            scale=CAMOUFLAGE_SCALE, seed=seed,
        )
    else:
        raise ValueError(f"Unsupported CAMOUFLAGE_KIND={CAMOUFLAGE_KIND}")

    model._gmu_paper_meta = {
        "camouflage_kind": CAMOUFLAGE_KIND,
        "gmu_layers": target_layers,
        "gmu_indices": gmu_indices,
        "camouflage_indices": camouflage_indices,
        "gmu_reuses_native_channels": True,
        "gmu_selection": GMU_SELECTION,
        "camouflage_in_same_decoder_blocks": True,
    }
    return model


def release_model(model=None, trainer=None):
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# Main experiment: same SFT/evaluation protocol as clean baseline
# ============================================================

def run_trial(trial_idx: int):
    seed = BASE_SEED + trial_idx
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results_file = f"results_GMU_trial{trial_idx}.json"
    results = {
        "trial": trial_idx,
        "seed": seed,
        "model_id": MODEL_ID,
        "method": "GMU",
        "gmu_camouflage": {
            "camouflage_kind": CAMOUFLAGE_KIND,
            "epsilon": EPSILON,
            "omega": OMEGA,
            "k": K,
            "num_gmu_layers": NUM_LAYERS,
            "gmu_target": "last_four_decoder_MLP_blocks",
            "gmu_selection": GMU_SELECTION,
            "camouflage_scale": CAMOUFLAGE_SCALE,
            "camouflage_num_channels": (
                CAMOUFLAGE_NUM_CHANNELS if CAMOUFLAGE_KIND == "pseudo_ffn"
                else "all_QK_rows_per_selected_block_due_to_Qwen3_RMSNorm"
            ),
            "same_decoder_blocks": True,
            "no_extra_parameters": True,
        },
        "training": {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "answer_only_loss": True,
            "dynamic_padding": True,
            "independent_task_finetuning": True,
        },
        "pre_FT": {},
        "post_FT": {},
    }

    print(f"\n>>> TRIAL {trial_idx}: PRE-FT GMU {CAMOUFLAGE_KIND} <<<")
    pre_model = load_paper_aligned_model(seed)
    results["gmu_camouflage"]["placement"] = pre_model._gmu_paper_meta
    for task in TASKS:
        results["pre_FT"][task] = evaluate_accuracy(
            pre_model, tokenizer, task, eval_split(task)
        )
        print(f"[Pre-FT] {task}: {results['pre_FT'][task]['accuracy']:.2f}%")
    release_model(model=pre_model)

    for train_task in TASKS:
        set_seed(seed)
        model = load_paper_aligned_model(seed)
        train_ds = load_task_dataset(train_task, "train")
        tokenized_train = train_ds.map(
            lambda ex: encode_sft_example(ex, tokenizer, train_task),
            batched=False,
            remove_columns=train_ds.column_names,
            desc=f"Tokenizing {train_task}",
        )
        micro_batch, grad_accum = TASK_BATCH[train_task]
        output_dir = f"./chkpt_GMU_{train_task}_trial{trial_idx}"
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=micro_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=True,
            seed=seed,
            data_seed=seed,
            remove_unused_columns=False,
        )
        callback = IterationLossCallback(train_task, trial_idx)
        trainer = Trainer(
            model=model, args=args, train_dataset=tokenized_train,
            data_collator=AnswerOnlyDataCollator(tokenizer),
            callbacks=[callback],
        )
        trainer.train()

        if SAVE_MODELS:
            save_dir = f"./weights_gmu_{prefix}_reuse_qwen1.7b_{train_task}_trial{trial_idx}"
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            if CAMOUFLAGE_KIND == "equiv_qk":
                save_qk_runtime_metadata(
                    save_dir, model._gmu_paper_meta["camouflage_indices"]
                )

        eval_tasks = TASKS if EVAL_ALL_TASKS_AFTER_FT else [train_task]
        results["post_FT"][train_task] = {}
        for eval_task in eval_tasks:
            metrics = evaluate_accuracy(
                model, tokenizer, eval_task, eval_split(eval_task)
            )
            results["post_FT"][train_task][eval_task] = metrics
            print(f"[Post-{train_task}] {eval_task}: {metrics['accuracy']:.2f}%")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        release_model(model=model, trainer=trainer)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    for trial in TRIALS:
        run_trial(trial)
