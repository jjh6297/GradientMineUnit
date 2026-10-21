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

# Supported datasets
DATASETS = ["cifar10", "food101"]

OUTPUT_DIR = "CleanViT_Finetuning"
LOG_FILE = os.path.join(OUTPUT_DIR, "CleanViT_Finetuning.log")
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
# Dataset Helpers
# ============================================================

def prepare_data(dataset_name, processor, batch_size=BATCH_SIZE):
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]

    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    if dataset_name == "cifar10":
        ds_train = load_dataset("cifar10", split="train")
        ds_val = load_dataset("cifar10", split="test")
        img_key = "img"
        label_key = "label"
        num_labels = 10

    elif dataset_name == "food101":
        ds_train = load_dataset("food101", split="train")
        ds_val = load_dataset("food101", split="validation")
        img_key = "image"
        label_key = "label"
        num_labels = 101

    elif dataset_name == "cub200":
        ds = load_dataset("cassiekang/cub200_dataset")
        num_labels = 200
        if "label" not in ds["train"].features:
            classes = sorted(list(set(ds["train"]["text"]).union(set(ds["test"]["text"]))))
            c2idx = {name: i for i, name in enumerate(classes)}
            ds = ds.map(lambda x: {"label": c2idx[x["text"]]}, num_proc=4)
        ds_train = ds["train"]
        ds_val = ds["test"] if "test" in ds else ds["validation"]
        img_key = "image"
        label_key = "label"

    elif dataset_name == "cars196":
        ds_train = load_dataset("tanganke/stanford_cars", split="train")
        ds_val = load_dataset("tanganke/stanford_cars", split="test")
        img_key = "image"
        label_key = "label"
        num_labels = 196

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def collate_train(batch):
        pixel_values = torch.stack([train_transform(x[img_key]) for x in batch])
        labels = torch.tensor([x[label_key] for x in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    def collate_val(batch):
        pixel_values = torch.stack([val_transform(x[img_key]) for x in batch])
        labels = torch.tensor([x[label_key] for x in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

    train_dl = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_train,
    )
    val_dl = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_val,
    )

    return train_dl, val_dl, num_labels


# ============================================================
# Train & Evaluate
# ============================================================

def evaluate(model, val_dl, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dl:
            inputs = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(inputs)
            preds = outputs.logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model,
    train_dl,
    optimizer,
    criterion,
    device,
    dataset_name,
    seed,
    epoch,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_dl,
        desc=f"Clean ViT {dataset_name} S{seed} Ep {epoch + 1}/{EPOCHS}",
        leave=False,
    )

    for batch in pbar:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss = total_loss / total
    train_acc = correct / total

    return train_loss, train_acc


# ============================================================
# Trial
# ============================================================

def run_trial(dataset_name, seed, trial_idx, gpu_id, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=5e-5):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join(
        OUTPUT_DIR,
        f"results_clean_{dataset_name}_seed{seed}.json",
    )

    if os.path.exists(save_path):
        log_print(f"[SKIP] Existing result found: {save_path}")
        return

    log_print(
        f"\n>>> [START] Clean Pretrained ViT Fine-tuning | {dataset_name} | "
        f"Trial={trial_idx} | Seed={seed} | Device={device}"
    )

    set_seed(seed)

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    train_dl, val_dl, num_labels = prepare_data(dataset_name, processor, batch_size=batch_size)

    log_print("\n[*] Loading clean ImageNet-pretrained ViT-B/16...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    initial_val_acc = evaluate(model, val_dl, device)
    log_print(f"[Before FT] {dataset_name} Acc={initial_val_acc:.4f}")

    history = {
        "model_type": "Clean_Pretrained_ViT",
        "dataset": dataset_name,
        "model_id": MODEL_ID,
        "trial": trial_idx,
        "seed": seed,
        "batch_size": batch_size,
        "epochs_total": epochs,
        "optimizer": "AdamW",
        "learning_rate": lr,
        "initial_val_acc_before_finetuning": initial_val_acc,
        "epochs": [],
    }

    for epoch in range(epochs):
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
    parser = argparse.ArgumentParser(description="Clean Pretrained ViT Fine-tuning (Baseline)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DATASETS,
        choices=["cifar10", "food101", "cub200", "cars196"],
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    args = parser.parse_args()

    with open(LOG_FILE, "a") as f:
        f.write(
            f"--- Clean Pretrained ViT Fine-tuning Started: {time.ctime()} ---\n"
        )

    for dataset_name in args.datasets:
        for idx in range(args.seed_start, args.seed_end):
            seed = SEEDS[idx]
            run_trial(
                dataset_name=dataset_name,
                seed=seed,
                trial_idx=idx,
                gpu_id=args.gpu,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
