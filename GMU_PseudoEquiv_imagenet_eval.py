import os
import argparse
import glob
import json
import time
import random
import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

print("[Debug] Imports finished.")


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/vit-base-patch16-224"
BATCH_SIZE = 128
SEED = 123

CAMO_RATIO_qk = 1.0
CAMO_SCALE = 10**7  
OMEGA_VAL = 10**7   
CAMO_TARGET_LAYERS = list(range(12))
GMU_TARGET_LAYERS = list(range(2, 6))
NATIVE_CAMO_TARGET_LAYERS = [
    i for i in range(12) if i not in GMU_TARGET_LAYERS
]
SHADOW_SIZE = 8


CAMO_RATIO_native = 0.05              # 10% native hidden neurons per selected layer. If CAMO_RATIO=0.0, it is same to Equiv. GMU
CAMO_MODE = "random"           # {"random", "top_norm"}

# ImageNet validation path
# If this path does not exist, the code falls back to load_dataset("imagenet-1k").
IMAGENET_VAL_PARQUET = "/mnt/disk1/datasets/imagenet-1k-validation/*.parquet"
OUTPUT_DIR = "PseudoEquivGMU_ImageNetEval"
LOG_FILE = os.path.join(OUTPUT_DIR, "PseudoEquivGMU_imagenet_eval.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Utils
# ============================================================

def log_print(msg, log_path=None):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    formatted_msg = f"{ts} {msg}"
    print(formatted_msg)
    target_log = log_path if log_path is not None else LOG_FILE
    with open(target_log, "a") as f:
        f.write(formatted_msg + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


# ============================================================
# Native Weight Camouflage
# ============================================================

@torch.no_grad()
def apply_native_scale_camouflage(
    model,
    scale=10**7,
    ratio=0.10,
    target_layers=None,
    mode="top_norm",
    seed=123,
    log_path=None,
):
    """
    Apply native MLP camouflage to selected non-GMU ViT blocks.

    For selected native neuron j:
        intermediate.weight[j, :] *= scale
        intermediate.bias[j]      *= scale
        output.weight[:, j]       /= scale

    In this version, target_layers should exclude GMU_TARGET_LAYERS.
    Therefore, MLP camouflage is not applied to layers where GMUs are inserted.
    """
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

        if log_path is not None:
            log_print(
                f"[Native-MLP-Camo] layer={layer_idx:02d} | "
                f"selected={num_select}/{hidden_dim} ({num_select / hidden_dim:.2%}) | "
                f"scale={scale:.2e} | mode={mode}",
                log_path,
            )
            log_print(
                f"  encoder norm selected mean: "
                f"{native_camo_stats[layer_idx]['encoder_norm_selected_before_mean']:.4e} -> "
                f"{native_camo_stats[layer_idx]['encoder_norm_selected_after_mean']:.4e}",
                log_path,
            )
            log_print(
                f"  readout norm selected mean: "
                f"{native_camo_stats[layer_idx]['readout_norm_selected_before_mean']:.4e} -> "
                f"{native_camo_stats[layer_idx]['readout_norm_selected_after_mean']:.4e}",
                log_path,
            )

    return model, native_camo_indices, native_camo_stats

@torch.no_grad()
def apply_camouflage(model, scale, ratio=0.10, target_layers=None, seed=123):
    if ratio <= 0:
        return model, {}

    if target_layers is None:
        target_layers = list(range(len(model.vit.encoder.layer)))

    rng = np.random.default_rng(seed)
    camo_indices = {}

    for layer_idx in target_layers:
        self_attn = model.vit.encoder.layer[layer_idx].attention.attention
        q_proj = self_attn.query
        k_proj = self_attn.key

        hidden_dim = q_proj.out_features
        num_select = max(1, int(round(hidden_dim * ratio)))

        selected_np = rng.choice(hidden_dim, size=num_select, replace=False)
        selected = torch.tensor(
            selected_np,
            device=q_proj.weight.device,
            dtype=torch.long,
        )

        camo_indices[layer_idx] = selected.detach().cpu().tolist()

        q_proj.weight[selected, :] *= scale
        if q_proj.bias is not None:
            q_proj.bias[selected] *= scale

        k_proj.weight[selected, :] /= scale
        if k_proj.bias is not None:
            k_proj.bias[selected] /= scale

    return model, camo_indices
    
@torch.no_grad()
def apply_mrt(model, scale, shadow_size=8, epsilon=1e-20):
    target_layers = range(2, 6)

    for i in target_layers:
        intermediate = model.vit.encoder.layer[i].intermediate.dense
        output = model.vit.encoder.layer[i].output.dense

        out_dim = output.out_features

        # Preserved from the uploaded code:
        # random high-value injection instead of constant scale.
        intermediate.weight[-shadow_size:, :] = (
            torch.rand(
                shadow_size,
                intermediate.weight.size(1),
                device=model.device,
                dtype=model.dtype,
            ) * scale
        )
        intermediate.bias[-shadow_size:] = 0.0

        temp = torch.rand(
            out_dim,
            shadow_size // 2,
            device=model.device,
            dtype=model.dtype,
        ) * epsilon

        output.weight[:, -shadow_size:-(shadow_size // 2)] = temp
        output.weight[:, -(shadow_size // 2):] = -temp

    return model

def verify_native_camo_excludes_gmu_layers(native_camo_indices, log_path=None):
    for layer_idx in native_camo_indices:
        if layer_idx in GMU_TARGET_LAYERS:
            msg = f"CRITICAL ERROR: Native MLP camouflage applied to GMU layer {layer_idx}!"
            if log_path:
                log_print(msg)
            else:
                print(msg)
            raise ValueError(msg)
    if log_path:
        log_print("[Verification] Native MLP camouflage successfully excluded all GMU layers.")
    else:
        print("[Verification] Native MLP camouflage successfully excluded all GMU layers.")
# ============================================================
# ImageNet Validation
# ============================================================

def build_imagenet_val_loader(processor, batch_size, parquet_path=None, max_examples=None):
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"] if isinstance(processor.size, dict) else processor.size

    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    if parquet_path is not None and len(glob.glob(parquet_path)) > 0:
        log_print(f"[*] Loading ImageNet validation from parquet: {parquet_path}")
        ds = load_dataset("parquet", data_files=parquet_path, split="train")
    else:
        log_print("[*] Loading ImageNet validation from Hugging Face: imagenet-1k")
        ds = load_dataset(
            "imagenet-1k",
            split="validation",
            trust_remote_code=True,
        )

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def collate_fn(examples):
        images = []
        labels = []

        for ex in examples:
            if "image" in ex:
                img = ex["image"]
            elif "img" in ex:
                img = ex["img"]
            else:
                raise KeyError(f"Cannot find image key. Available keys: {ex.keys()}")

            if "label" in ex:
                label = ex["label"]
            elif "labels" in ex:
                label = ex["labels"]
            else:
                raise KeyError(f"Cannot find label key. Available keys: {ex.keys()}")

            images.append(val_transform(img))
            labels.append(label)

        return {
            "pixel_values": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return loader, len(ds)


@torch.no_grad()
def evaluate_imagenet(model, dataloader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="ImageNet Val"):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(pixel_values=inputs).logits
        loss = criterion(logits, labels)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item()
        num_batches += 1

    acc = 100.0 * correct / max(total, 1)
    avg_loss = loss_sum / max(num_batches, 1)

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "num_examples": int(total),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("[Debug] main() started.")
    global OUTPUT_DIR, LOG_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--scale", type=float, default=CAMO_SCALE)
    parser.add_argument("--ratio", type=float, default=CAMO_RATIO_native)
    parser.add_argument("--omega", type=float, default=OMEGA_VAL)
    parser.add_argument("--mode", type=str, default=CAMO_MODE, choices=["random", "top_norm"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--imagenet_parquet", type=str, default=IMAGENET_VAL_PARQUET)
    parser.add_argument("--target_layers", type=int, nargs="+", default=CAMO_TARGET_LAYERS)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "native_camouflage_imagenet_eval.log")

    with open(LOG_FILE, "w") as f:
        f.write(f"--- Native Camouflage ImageNet Evaluation Started: {time.ctime()} ---\n")

    set_seed(args.seed)
    device = get_device(args.gpu)

    log_print("=" * 80)
    log_print("[Start] Native Weight Camouflage ImageNet Evaluation")
    log_print(f"model_id      = {MODEL_ID}")
    log_print(f"device        = {device}")
    log_print(f"seed          = {args.seed}")
    log_print(f"scale         = {args.scale:.2e}")
    log_print(f"ratio         = {args.ratio:.2%}")
    log_print(f"mode          = {args.mode}")
    log_print(f"target_layers = {args.target_layers}")
    log_print("=" * 80)

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    val_loader, num_val = build_imagenet_val_loader(
        processor=processor,
        batch_size=args.batch_size,
        parquet_path=args.imagenet_parquet,
        max_examples=args.max_examples,
    )

    log_print("\n[*] Loading ImageNet-pretrained ViT-B/16...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=1000,
        ignore_mismatched_sizes=False,
    ).to(device)

    log_print("\n[*] Evaluating clean pretrained model...")
    clean_metrics = evaluate_imagenet(model, val_loader, device)
    log_print(
        f"[Clean] ImageNet Val Loss={clean_metrics['loss']:.4f} | "
        f"Acc={clean_metrics['accuracy']:.3f}% | "
        f"N={clean_metrics['num_examples']}"
    )

    log_print("\n[*] Applying native scale camouflage...")
    model, native_camo_indices, native_camo_stats = apply_native_scale_camouflage(
        model=model,
        scale=CAMO_SCALE,
        ratio=CAMO_RATIO_native,
        target_layers=NATIVE_CAMO_TARGET_LAYERS,
        mode=args.mode,
        seed=args.seed,
        log_path=LOG_FILE,
    )

    model, indices = apply_camouflage(
        model,
        scale=CAMO_SCALE,
        ratio=CAMO_RATIO_qk,
        target_layers=CAMO_TARGET_LAYERS,
        seed=args.seed,
    )

    model = apply_mrt(
        model,
        scale=CAMO_SCALE,
        shadow_size=SHADOW_SIZE,
    )
    log_print("\n[*] Evaluating camouflaged pretrained model...")
    camo_metrics = evaluate_imagenet(model, val_loader, device)
    log_print(
        f"[Pseudo_GMU] ImageNet Val Loss={camo_metrics['loss']:.4f} | "
        f"Acc={camo_metrics['accuracy']:.3f}% | "
        f"N={camo_metrics['num_examples']}"
    )

    result = {
        "model_id": MODEL_ID,
        "seed": args.seed,
        "scale": args.scale,
        "ratio": args.ratio,
        "mode": args.mode,
        "target_layers": args.target_layers,
        "num_val_examples": num_val,
        "clean_metrics": clean_metrics,
        "camouflaged_metrics": camo_metrics,
        "accuracy_drop": clean_metrics["accuracy"] - camo_metrics["accuracy"],
        "loss_increase": camo_metrics["loss"] - clean_metrics["loss"],
        "camouflage_indices": native_camo_indices,
        "camouflage_stats": native_camo_stats,
        "finished_at": time.ctime(),
    }

    result_path = os.path.join(
        OUTPUT_DIR,
        f"imagenet_native_camouflage_scale{args.scale:.0e}_ratio{args.ratio:.2f}_seed{args.seed}.json",
    )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    log_print("\n[Done]")
    log_print(f"Result JSON: {result_path}")
    log_print(
        f"Accuracy drop: {result['accuracy_drop']:.3f}% | "
        f"Loss increase: {result['loss_increase']:.4f}"
    )

    del model, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()