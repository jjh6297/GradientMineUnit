import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
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
    "sst2": (4, 8),       
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
        self.save_name = f"loss_Clean_{task_name}_trial{trial_idx}.json"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append({
                "step": int(state.global_step),
                "loss": float(logs["loss"]),
            })
            with open(self.save_name, "w") as f:
                json.dump(self.losses, f, indent=2)


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

    # The leading space is part of the answer segment in both training and eval.
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
def load_clean_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    return model


def release_model(model=None, trainer=None):
    if trainer is not None:
        del trainer
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# Main experiment
# ============================================================
def run_trial(trial_idx: int):
    seed = BASE_SEED + trial_idx
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results_file = f"results_Clean_trial{trial_idx}.json"
    results = {
        "trial": trial_idx,
        "seed": seed,
        "model_id": MODEL_ID,
        "method": "Clean",
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

    # --------------------------------------------------------
    # Pre-FT evaluation: one untouched pretrained model
    # --------------------------------------------------------
    print(f"\n>>> TRIAL {trial_idx}: PRE-FT CLEAN EVALUATION <<<")
    pre_model = load_clean_model()
    for task in TASKS:
        print(f"[Pre-FT] Evaluating {task}...")
        results["pre_FT"][task] = evaluate_accuracy(
            pre_model, tokenizer, task, eval_split(task)
        )
        print(f"  {task}: {results['pre_FT'][task]['accuracy']:.2f}%")
    release_model(model=pre_model)

    # --------------------------------------------------------
    # Each downstream task starts from a FRESH pretrained model.
    # No SST2 -> ScienceQA -> LogiQA chaining.
    # --------------------------------------------------------
    for train_task in TASKS:
        print(f"\n>>> TRIAL {trial_idx}: CLEAN FT ON {train_task.upper()} <<<")
        set_seed(seed)

        model = load_clean_model()
        train_ds = load_task_dataset(train_task, "train")

        tokenized_train = train_ds.map(
            lambda ex: encode_sft_example(ex, tokenizer, train_task),
            batched=False,
            remove_columns=train_ds.column_names,
            desc=f"Tokenizing {train_task}",
        )

        micro_batch, grad_accum = TASK_BATCH[train_task]
        output_dir = f"./chkpt_Clean_{train_task}_trial{trial_idx}"

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
        collator = AnswerOnlyDataCollator(tokenizer)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            data_collator=collator,
            callbacks=[callback],
        )

        trainer.train()

        if SAVE_MODELS:
            save_dir = f"./weights_clean_qwen1.7b_{train_task}_trial{trial_idx}"
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)

        eval_tasks = TASKS if EVAL_ALL_TASKS_AFTER_FT else [train_task]
        results["post_FT"][train_task] = {}

        for eval_task in eval_tasks:
            print(f"[Post-{train_task}] Evaluating {eval_task}...")
            metrics = evaluate_accuracy(
                model, tokenizer, eval_task, eval_split(eval_task)
            )
            results["post_FT"][train_task][eval_task] = metrics
            print(f"  {eval_task}: {metrics['accuracy']:.2f}%")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        release_model(model=model, trainer=trainer)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    for trial in TRIALS:
        run_trial(trial)
