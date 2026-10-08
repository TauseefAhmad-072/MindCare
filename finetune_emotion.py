"""
finetune_emotion.py — Fine-tune the emotion classifier on your own frames.

Reads from ./uploads/raw dataset/<class>/ with folders:
  angry, sad, neutral, surprised

Builds a 4-class classifier by reinitializing the head of the base model.
Freezes the ViT encoder; trains only the classifier. Works with 30+ frames
per class. Runtime: under 2 minutes on RTX 3060.

Output: ./models/face_finetuned/ ready to be swapped into the pipeline.
"""

import os
import shutil
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_ID  = "dima806/facial_emotions_image_detection"
RAW_DIR   = Path("./uploads/dataset_raw")
SPLIT_DIR = Path("./dataset_split")
OUT_DIR   = Path("./models/face_finetuned")

# Map folder names to canonical labels. Handles your 'surprised' folder.
FOLDER_TO_LABEL = {
    "angry":     "angry",
    "happy":       "happy",
    "neutral":   "neutral",
    "surprised": "surprise",
}

VAL_FRACTION = 0.2
SEED = 42
random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Verify raw data
# ──────────────────────────────────────────────────────────────────────────────

if not RAW_DIR.exists():
    raise SystemExit(f"Raw dataset folder not found: {RAW_DIR.resolve()}")

print(f"Scanning {RAW_DIR.resolve()}")
class_frames = {}
for sub in sorted(RAW_DIR.iterdir()):
    if not sub.is_dir():
        continue
    if sub.name not in FOLDER_TO_LABEL:
        print(f"  [skip] {sub.name} (no mapping)")
        continue
    frames = [f for f in sub.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    class_frames[sub.name] = frames
    print(f"  {sub.name:<12} → {len(frames):>4} frames")

if not class_frames:
    raise SystemExit("No usable class folders found.")

min_count = min(len(v) for v in class_frames.values())
print(f"\nSmallest class has {min_count} frames. Fine-tuning will cap every "
      f"class at {min_count} frames for balance.\n")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Split into train/val
# ──────────────────────────────────────────────────────────────────────────────

if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

for raw_name, frames in class_frames.items():
    label = FOLDER_TO_LABEL[raw_name]

    # Cap to smallest class so training is balanced
    shuffled = frames[:]
    random.shuffle(shuffled)
    shuffled = shuffled[:min_count]

    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_frames   = shuffled[:n_val]
    train_frames = shuffled[n_val:]

    for split, flist in (("train", train_frames), ("val", val_frames)):
        out_dir = SPLIT_DIR / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in flist:
            shutil.copy(f, out_dir / f.name)

    print(f"  {raw_name:<12} → {len(train_frames):>3} train / {len(val_frames):>2} val  (label='{label}')")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build 4-class model
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = sorted({FOLDER_TO_LABEL[n] for n in class_frames})
print(f"\nFinal classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

config = AutoConfig.from_pretrained(MODEL_ID)
config.num_labels = len(CLASS_NAMES)
config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

processor = AutoImageProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    config=config,
    ignore_mismatched_sizes=True,   # rebuilds classifier head for 4 classes
)

# Freeze everything except the classifier head
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Training {trainable:,} / {total:,} params ({100*trainable/total:.3f}%)\n")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Dataset class with augmentation
# ──────────────────────────────────────────────────────────────────────────────

train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
])


class FrameDataset(Dataset):
    def __init__(self, root: Path, augment: bool):
        self.samples = []
        self.augment = augment
        for label in CLASS_NAMES:
            label_dir = root / label
            if not label_dir.exists():
                continue
            for f in label_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((f, label))
        self.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.augment:
            img = train_aug(img)
        inputs = processor(img, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),
        }


train_ds = FrameDataset(SPLIT_DIR / "train", augment=True)
val_ds   = FrameDataset(SPLIT_DIR / "val",   augment=False)
print(f"Train: {len(train_ds)} frames | Val: {len(val_ds)} frames\n")


def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train
# ──────────────────────────────────────────────────────────────────────────────

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=1e-3,
    warmup_steps=20,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=5,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

OUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT_DIR))
processor.save_pretrained(str(OUT_DIR))
print(f"\n✅ Saved fine-tuned model to {OUT_DIR.resolve()}")