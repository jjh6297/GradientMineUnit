import os
import argparse
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import json
import time
import random
import gc

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/vit-base-patch16-224"
BATCH_SIZE = 128
EPOCHS = 30
SEEDS = [1, 12, 123, 1234, 12345]

OMEGA_VAL = 10**7
SHADOW_SIZE = 8
EPSILON = 1e-20
GMU_TARGET_LAYERS = list(range(2, 6))

# Q/K reciprocal camouflage
# Taken from the first code.
QK_CAMO_RATIO = 1.0
QK_CAMO_SCALE = 10**7
QK_CAMO_MODE = "random"  # {"random", "top_norm"}
QK_CAMO_TARGET_LAYERS = list(range(12))


NATIVE_CAMO_RATIO = 0.05
NATIVE_CAMO_SCALE = 10**7
NATIVE_CAMO_MODE = "random"  # {"random", "top_norm"}
NATIVE_CAMO_TARGET_LAYERS = [
    i for i in range(12) if i not in GMU_TARGET_LAYERS
]

# Taken from the first code.
GRAD_CLIP_NORM = None

# Taken from the second code, but now all four datasets are enabled.
DATASETS = ["cifar10", "food101"]

OUTPUT_DIR = "GMU_PseudoEquiv_Camouflage"
LOG_FILE = os.path.join(OUTPUT_DIR, "GMU_PseudoEquiv_Camouflage.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Utils
# ============================================================

def log_print(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    formatted_msg = f"{ts} {msg}"
    print(formatted_msg)

    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ============================================================
# GMU Injection
# ============================================================

@torch.no_grad()
def inject_gmu(
    model,
    scale,
    shadow_size=8,
    epsilon=1e-20,
    target_layers=None,
):
    """
    Inject GMUs into the final shadow_size rows of MLP intermediate.dense.

    GMU location:
        vit.encoder.layer.{i}.intermediate.dense.weight[-shadow_size:, :]

    Symmetry lock:
        output.dense.weight[:, -shadow_size:-K] = +epsilon-scale temp
        output.dense.weight[:, -K:]             = -epsilon-scale temp
    """
    if target_layers is None:
        target_layers = GMU_TARGET_LAYERS

    K = shadow_size // 2

    for layer_idx in target_layers:
        intermediate = model.vit.encoder.layer[layer_idx].intermediate.dense
        output = model.vit.encoder.layer[layer_idx].output.dense

        out_dim = output.out_features

        intermediate.weight[-shadow_size:, :] = torch.rand(shadow_size, intermediate.weight.size(1), device=model.device, dtype=model.dtype) * scale
        intermediate.bias[-shadow_size:] = 0.0

        temp = torch.rand(
            out_dim,
            K,
            device=output.weight.device,
            dtype=output.weight.dtype,
        ) * epsilon

        output.weight[:, -shadow_size:-K] = temp
        output.weight[:, -K:] = -temp

        log_print(
            f"[GMU] layer={layer_idx:02d} | "
            f"K={shadow_size} | Omega={scale:.2e} | epsilon={epsilon:.2e}"
        )

    return model


# ============================================================
# Camouflage 1: Native MLP Camouflage
# ============================================================

@torch.no_grad()
def apply_native_scale_camouflage(
    model,
    scale=10**7,
    ratio=0.05,
    target_layers=None,
    mode="random",
    seed=123,
):
    if ratio <= 0:
        return model, {}, {}

    if target_layers is None:
        target_layers = NATIVE_CAMO_TARGET_LAYERS

    rng = np.random.default_rng(seed)
    native_camo_indices = {}
    native_camo_stats = {}

    for layer_idx in target_layers:
        if layer_idx in GMU_TARGET_LAYERS:
            raise ValueError(
                f"Native MLP camouflage target layer {layer_idx} overlaps with GMU_TARGET_LAYERS."
            )

        layer = model.vit.encoder.layer[layer_idx]
        inter = layer.intermediate.dense
        out = layer.output.dense

        hidden_dim = inter.out_features
        num_select = max(1, int(round(hidden_dim * ratio)))
        num_select = min(num_select, hidden_dim)

        weight_before_norm = inter.weight.detach().float().norm(dim=1)
        out_before_norm = out.weight.detach().float().norm(dim=0)

        if mode == "random":
            selected_np = rng.choice(hidden_dim, size=num_select, replace=False)
            selected = torch.tensor(
                selected_np,
                device=inter.weight.device,
                dtype=torch.long,
            )
        elif mode == "top_norm":
            selected = torch.topk(
                weight_before_norm.to(inter.weight.device),
                k=num_select,
                largest=True,
            ).indices
        else:
            raise ValueError(f"Unknown native camouflage mode: {mode}")

        inter.weight[selected, :] *= scale
        if inter.bias is not None:
            inter.bias[selected] *= scale

        out.weight[:, selected] /= scale

        weight_after_norm = inter.weight.detach().float().norm(dim=1)
        out_after_norm = out.weight.detach().float().norm(dim=0)

        selected_list = selected.detach().cpu().tolist()
        native_camo_indices[layer_idx] = selected_list

        native_camo_stats[layer_idx] = {
            "module": f"vit.encoder.layer.{layer_idx}.intermediate.dense",
            "num_hidden": int(hidden_dim),
            "num_selected": int(num_select),
            "ratio": float(num_select / hidden_dim),
            "scale": float(scale),
            "mode": mode,
            "excluded_gmu_layers": GMU_TARGET_LAYERS,
            "encoder_norm_selected_before_mean": float(weight_before_norm[selected].mean().item()),
            "encoder_norm_selected_after_mean": float(weight_after_norm[selected].mean().item()),
            "readout_norm_selected_before_mean": float(out_before_norm[selected].mean().item()),
            "readout_norm_selected_after_mean": float(out_after_norm[selected].mean().item()),
            "encoder_norm_all_after_max": float(weight_after_norm.max().item()),
            "encoder_norm_all_after_median": float(weight_after_norm.median().item()),
        }

        log_print(
            f"[Native-MLP-Camo] layer={layer_idx:02d} | "
            f"selected={num_select}/{hidden_dim} ({num_select / hidden_dim:.2%}) | "
            f"scale={scale:.2e} | mode={mode}"
        )

    return model, native_camo_indices, native_camo_stats


# ============================================================
# Camouflage 2: Q/K Reciprocal Camouflage
# ============================================================

@torch.no_grad()
def apply_qk_camouflage(
    model,
    scale=10**7,
    ratio=1.00,
    target_layers=None,
    mode="random",
    seed=123,
):
    if ratio <= 0:
        return model, {}, {}

    if target_layers is None:
        target_layers = list(range(len(model.vit.encoder.layer)))

    rng = np.random.default_rng(seed)
    qk_camo_indices = {}
    qk_camo_stats = {}

    for layer_idx in target_layers:
        self_attn = model.vit.encoder.layer[layer_idx].attention.attention
        q_proj = self_attn.query
        k_proj = self_attn.key

        hidden_dim = q_proj.out_features
        num_select = max(1, int(round(hidden_dim * ratio)))
        num_select = min(num_select, hidden_dim)

        q_before_norm = q_proj.weight.detach().float().norm(dim=1)
        k_before_norm = k_proj.weight.detach().float().norm(dim=1)

        if mode == "random":
            selected_np = rng.choice(hidden_dim, size=num_select, replace=False)
            selected = torch.tensor(
                selected_np,
                device=q_proj.weight.device,
                dtype=torch.long,
            )
        elif mode == "top_norm":
            selected = torch.topk(
                q_before_norm.to(q_proj.weight.device),
                k=num_select,
                largest=True,
            ).indices
        else:
            raise ValueError(f"Unknown Q/K camouflage mode: {mode}")

        q_proj.weight[selected, :] *= scale
        if q_proj.bias is not None:
            q_proj.bias[selected] *= scale

        k_proj.weight[selected, :] /= scale
        if k_proj.bias is not None:
            k_proj.bias[selected] /= scale

        q_after_norm = q_proj.weight.detach().float().norm(dim=1)
        k_after_norm = k_proj.weight.detach().float().norm(dim=1)

        selected_list = selected.detach().cpu().tolist()
        qk_camo_indices[layer_idx] = selected_list

        qk_camo_stats[layer_idx] = {
            "module_q": f"vit.encoder.layer.{layer_idx}.attention.attention.query",
            "module_k": f"vit.encoder.layer.{layer_idx}.attention.attention.key",
            "hidden_dim": int(hidden_dim),
            "num_selected": int(num_select),
            "ratio": float(num_select / hidden_dim),
            "scale": float(scale),
            "mode": mode,
            "q_norm_selected_before_mean": float(q_before_norm[selected].mean().item()),
            "q_norm_selected_after_mean": float(q_after_norm[selected].mean().item()),
            "k_norm_selected_before_mean": float(k_before_norm[selected].mean().item()),
            "k_norm_selected_after_mean": float(k_after_norm[selected].mean().item()),
            "q_norm_all_after_max": float(q_after_norm.max().item()),
            "q_norm_all_after_median": float(q_after_norm.median().item()),
            "k_norm_all_after_max": float(k_after_norm.max().item()),
            "k_norm_all_after_median": float(k_after_norm.median().item()),
        }

        log_print(
            f"[QK-Camo] layer={layer_idx:02d} | "
            f"selected={num_select}/{hidden_dim} ({num_select / hidden_dim:.2%}) | "
            f"scale={scale:.2e} | mode={mode}"
        )

    return model, qk_camo_indices, qk_camo_stats




# ============================================================
# Dataset Loading / Transforms
# ============================================================

def prepare_data(dataset_name, processor):
    """
    Dataset and transform protocol are taken from the second code.

    Supported datasets:
        - cifar10
        - food101
        - cub200
        - cars196

    Train transform:
        RandomResizedCrop(size)
        RandomHorizontalFlip()
        ToTensor()
        Normalize(mean, std)

    Validation/Test transform:
        Resize(size)
        CenterCrop(size)
        ToTensor()
        Normalize(mean, std)
    """
    log_print(f"[*] Preparing dataset: {dataset_name}")

    mean, std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    t_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    v_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if dataset_name == "cifar10":
        ds = load_dataset("cifar10")
        num_labels = 10

        def t_fn(ex):
            ex["pixel_values"] = [
                t_transform(img.convert("RGB")) for img in ex["img"]
            ]
            return ex

        def v_fn(ex):
            ex["pixel_values"] = [
                v_transform(img.convert("RGB")) for img in ex["img"]
            ]
            return ex

    elif dataset_name == "food101":
        ds = load_dataset("food101")
        num_labels = 101

        def t_fn(ex):
            ex["pixel_values"] = [
                t_transform(img.convert("RGB")) for img in ex["image"]
            ]
            return ex

        def v_fn(ex):
            ex["pixel_values"] = [
                v_transform(img.convert("RGB")) for img in ex["image"]
            ]
            return ex

    elif dataset_name == "cub200":
        ds = load_dataset("cassiekang/cub200_dataset")
        num_labels = 200

        if "label" not in ds["train"].features:
            classes = sorted(list(set(ds["train"]["text"] + ds["test"]["text"])))
            c2idx = {name: i for i, name in enumerate(classes)}
            ds = ds.map(lambda x: {"label": c2idx[x["text"]]}, num_proc=4)

        def t_fn(ex):
            ex["pixel_values"] = [
                t_transform(img.convert("RGB")) for img in ex["image"]
            ]
            return ex

        def v_fn(ex):
            ex["pixel_values"] = [
                v_transform(img.convert("RGB")) for img in ex["image"]
            ]
            return ex

    elif dataset_name == "cars196":
        ds = load_dataset("tanganke/stanford_cars", token=True)
        num_labels = 196

        l_key = "label" if "label" in ds["train"].features else "class_id"

        if l_key != "label":
            ds = ds.map(lambda x: {"label": x[l_key]}, num_proc=4)

        def t_fn(ex):
            ex["pixel_values"] = [
                t_transform(img.convert("RGB")) for img in ex["image"]
            ]
            return ex

        def v_fn(ex):
            ex["pixel_values"] = [
                v_transform(img.convert("RGB")) for img in ex["image"]
            ]
            return ex

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_ds = ds["train"].with_transform(t_fn)
    val_ds = (ds["test"] if "test" in ds else ds["validation"]).with_transform(v_fn)

    def collate_fn(exs):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in exs]),
            "labels": torch.tensor([x["label"] for x in exs], dtype=torch.long),
        }

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_dl, val_dl, num_labels


# ============================================================
# Train / Eval
# ============================================================

@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()

    total_correct = 0
    total_samples = 0

    for batch in val_dl:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(pixel_values=inputs).logits
        pred = logits.argmax(dim=-1)

        total_correct += (pred == labels).sum().item()
        total_samples += labels.numel()

    return total_correct / max(total_samples, 1)


def train_one_epoch(model, train_dl, optimizer, criterion, device, dataset_name, seed, epoch):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        train_dl,
        desc=f"{dataset_name} S{seed} Ep {epoch + 1}/{EPOCHS}",
    )

    for batch in progress:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(pixel_values=inputs).logits
        loss = criterion(logits, labels)

        loss.backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += batch_size

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss = total_loss / max(total_samples, 1)
    train_acc = total_correct / max(total_samples, 1)

    return train_loss, train_acc


# ============================================================
# Trial
# ============================================================

def run_trial(dataset_name, seed, trial_idx, gpu_id):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join(
        OUTPUT_DIR,
        f"results_bothcamo_mlp_exclude_gmulayers_{dataset_name}_seed{seed}.json",
    )

    if os.path.exists(save_path):
        log_print(f"[SKIP] Existing result found: {save_path}")
        return

    log_print(
        f"\n>>> [START] BothCamouflage | {dataset_name} | "
        f"Trial={trial_idx} | Seed={seed} | Device={device}"
    )

    set_seed(seed)

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    train_dl, val_dl, num_labels = prepare_data(dataset_name, processor)

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 1. Native MLP camouflage.
    log_print("\n[*] Applying Native MLP camouflage outside GMU layers...")
    model, native_camo_indices, native_camo_stats = apply_native_scale_camouflage(
        model=model,
        scale=NATIVE_CAMO_SCALE,
        ratio=NATIVE_CAMO_RATIO,
        target_layers=NATIVE_CAMO_TARGET_LAYERS,
        mode=NATIVE_CAMO_MODE,
        seed=seed,
    )

    # 2. Q/K reciprocal camouflage.
    log_print("\n[*] Applying Q/K reciprocal camouflage...")
    model, qk_camo_indices, qk_camo_stats = apply_qk_camouflage(
        model=model,
        scale=QK_CAMO_SCALE,
        ratio=QK_CAMO_RATIO,
        target_layers=QK_CAMO_TARGET_LAYERS,
        mode=QK_CAMO_MODE,
        seed=seed,
    )

    # 3. GMU injection.
    log_print("\n[*] Injecting GMUs...")
    model = inject_gmu(
        model=model,
        scale=OMEGA_VAL,
        shadow_size=SHADOW_SIZE,
        epsilon=EPSILON,
        target_layers=GMU_TARGET_LAYERS,
    )


    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    initial_val_acc = evaluate(model, val_dl, device)
    log_print(f"[Before FT] {dataset_name} Acc={initial_val_acc:.4f}")

    history = {
        "model_type": "GMU_with_qk_camouflage_and_native_mlp_camouflage_outside_gmu_layers",
        "dataset": dataset_name,
        "model_id": MODEL_ID,
        "trial": trial_idx,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "epochs_total": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": 5e-5,
        "grad_clip": GRAD_CLIP_NORM,
        "omega": OMEGA_VAL,
        "shadow_size": SHADOW_SIZE,
        "epsilon": EPSILON,
        "gmu_target_layers": GMU_TARGET_LAYERS,
        "native_camouflage": {
            "enabled": True,
            "scale": NATIVE_CAMO_SCALE,
            "ratio": NATIVE_CAMO_RATIO,
            "mode": NATIVE_CAMO_MODE,
            "target_layers": NATIVE_CAMO_TARGET_LAYERS,
            "explicitly_excludes_gmu_layers": True,
            "indices": native_camo_indices,
            "stats": native_camo_stats,
        },
        "qk_camouflage": {
            "enabled": True,
            "scale": QK_CAMO_SCALE,
            "ratio": QK_CAMO_RATIO,
            "mode": QK_CAMO_MODE,
            "target_layers": QK_CAMO_TARGET_LAYERS,
            "indices": qk_camo_indices,
            "stats": qk_camo_stats,
        },
        "initial_val_acc_before_finetuning": initial_val_acc,
        "epochs": [],
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            dataset_name=dataset_name,
            seed=seed,
            epoch=epoch,
        )

        val_acc = evaluate(model, val_dl, device)

        stats = {
            "epoch": epoch + 1,
            "loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

        history["epochs"].append(stats)

        log_print(
            f"  E{epoch + 1}: "
            f"Loss={stats['loss']:.4f}, "
            f"TrainAcc={stats['train_acc']:.4f}, "
            f"ValAcc={stats['val_acc']:.4f}"
        )

        with open(save_path, "w") as f:
            json.dump(to_jsonable(history), f, indent=2)

    log_print(f"[DONE] Saved result to: {save_path}")

    del model, optimizer, train_dl, val_dl
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DATASETS,
        choices=["cifar10", "food101", "cub200", "cars196", ],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="Start seed index, inclusive.",
    )
    parser.add_argument(
        "--seed_end",
        type=int,
        default=len(SEEDS),
        help="End seed index, exclusive.",
    )
    args = parser.parse_args()

    with open(LOG_FILE, "w") as f:
        f.write(
            f"--- Four-Dataset Both-Camouflage GMU Fine-tuning "
            f"(first-code camouflage settings + second-code transforms) "
            f"Started: {time.ctime()} ---\n"
        )

    for dataset_name in args.datasets:
        for idx in range(args.seed_start, args.seed_end):
            seed = SEEDS[idx]
            run_trial(dataset_name, seed, idx, args.gpu)