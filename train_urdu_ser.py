"""
train_urdu_ser.py — Fine-tune wav2vec2-xls-r-300m on an Urdu speech-emotion corpus.

Expected dataset layout (recursive discovery, so any of these work):
    uploads/UrduSER/Angry/*.wav
    uploads/UrduSER/Happy/*.wav
    uploads/UrduSER/Neutral/*.wav
    uploads/UrduSER/Sad/*.wav

    OR with train/test split:
    uploads/UrduSER/train/Angry/*.wav
    uploads/UrduSER/test/Angry/*.wav

    OR any other layout with class folders containing audio files.

Class names are auto-discovered from folder names, so:
  Angry / Anger / ANGRY           → "anger"
  Happy / Happiness / Joy          → "happy"
  Neutral / Calm                   → "neutral"
  Sad / Sadness                    → "sad"
  Fear / Fearful / Scared          → "fear"
  Surprise / Surprised             → "surprise"
  Disgust                          → "disgust"

Output:
    models/audio/urdu_ser/          ← HF-format model, drop-in for modules/audio_emotion

Usage:
    python train_urdu_ser.py
    python train_urdu_ser.py --data-dir uploads/UrduSER --epochs 10 --batch-size 8
    python train_urdu_ser.py --smoke-test         # 2-minute sanity run on a tiny subset
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger("mindcare.train_urdu_ser")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Label normalization — map diverse folder names to a canonical set
# ─────────────────────────────────────────────────────────────────────────────

# Canonical set the model will output. Must match what your pipeline
# downstream expects in AUDIO classes. We map a broad set of folder names
# into these 7.
CANONICAL_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

_FOLDER_TO_CANONICAL = {
    # anger
    "angry": "anger", "anger": "anger", "annoyed": "anger", "rage": "anger",
    # disgust
    "disgust": "disgust", "disgusted": "disgust",
    # fear
    "fear": "fear", "fearful": "fear", "scared": "fear", "afraid": "fear",
    "anxious": "fear", "anxiety": "fear", "nervous": "fear",
    # happy
    "happy": "happy", "happiness": "happy", "joy": "happy", "joyful": "happy",
    "excited": "happy",
    # neutral
    "neutral": "neutral", "calm": "neutral", "normal": "neutral", "plain": "neutral",
    # sad
    "sad": "sad", "sadness": "sad", "depressed": "sad", "depression": "sad",
    "unhappy": "sad", "melancholy": "sad",
    # surprise
    "surprise": "surprise", "surprised": "surprise", "surprising": "surprise",
}


def _canonical(folder_name: str) -> Optional[str]:
    key = folder_name.lower().strip().replace("_", "").replace("-", "").replace(" ", "")
    return _FOLDER_TO_CANONICAL.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class Sample:
    path: str
    label: str        # canonical label
    folder: str       # original folder name (for logging)
    split: str        # "train" or "test" (auto-detected from parent folder if present)


def discover(data_root: str) -> List[Sample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    samples: List[Sample] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip __pycache__ and similar
        dirnames[:] = [d for d in dirnames if not d.startswith("__") and not d.startswith(".")]

        folder_name = Path(dirpath).name
        canon = _canonical(folder_name)
        if canon is None:
            continue

        # Detect train/test split from the folder hierarchy
        rel_parts = Path(dirpath).relative_to(root).parts
        split = "train"
        for part in rel_parts[:-1]:
            p = part.lower()
            if p in ("train", "training"):
                split = "train"
                break
            if p in ("test", "valid", "val", "validation", "dev"):
                split = "test"
                break

        for fn in sorted(filenames):
            ext = Path(fn).suffix.lower()
            if ext not in AUDIO_EXTS:
                continue
            samples.append(Sample(
                path=str(Path(dirpath) / fn),
                label=canon,
                folder=folder_name,
                split=split,
            ))

    return samples


def split_train_val(samples: List[Sample], val_frac: float, seed: int
                     ) -> Tuple[List[Sample], List[Sample]]:
    """Use existing train/test split if present, else random stratified split."""
    has_explicit_test = any(s.split == "test" for s in samples)

    if has_explicit_test:
        train = [s for s in samples if s.split == "train"]
        val   = [s for s in samples if s.split == "test"]
        logger.info("Using dataset's own train/test split.")
        return train, val

    # Random stratified split
    rng = np.random.default_rng(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    train, val = [], []
    for label, group in by_label.items():
        idx = rng.permutation(len(group))
        n_val = max(1, int(len(group) * val_frac))
        val_idx = set(idx[:n_val].tolist())
        for i, s in enumerate(group):
            (val if i in val_idx else train).append(s)

    logger.info("Stratified random split (val_frac=%.2f).", val_frac)
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — reads audio, resamples to 16kHz, crops/pads to max_seconds
# ─────────────────────────────────────────────────────────────────────────────

LABEL_TO_IDX = {label: i for i, label in enumerate(CANONICAL_LABELS)}


class UrduSERDataset(Dataset):
    def __init__(self, samples: List[Sample], target_sr: int = 16000,
                 max_seconds: float = 5.0, augment: bool = False):
        self.samples = samples
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_seconds)
        self.augment = augment

        # Optional augmentation deps
        if self.augment:
            import random
            self._rng = random.Random(1337)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_wave(self, path: str) -> np.ndarray:
        import librosa
        try:
            wav, sr = librosa.load(path, sr=self.target_sr, mono=True)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return np.zeros(self.max_len, dtype=np.float32)

        if wav.size == 0:
            return np.zeros(self.max_len, dtype=np.float32)

        # Peak normalize
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-6:
            wav = wav / peak

        # Pad or crop
        if len(wav) < self.max_len:
            pad = self.max_len - len(wav)
            wav = np.pad(wav, (0, pad))
        else:
            # Random crop during training, center crop during val
            if self.augment:
                start = self._rng.randint(0, len(wav) - self.max_len)
            else:
                start = (len(wav) - self.max_len) // 2
            wav = wav[start:start + self.max_len]

        return wav.astype(np.float32)

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        """Light augmentation — gain + noise + slight time shift.
        We deliberately do NOT pitch-shift, because pitch carries emotion
        signal that the model needs to learn."""
        rng = self._rng
        # Gain
        gain = rng.uniform(0.7, 1.3)
        wav = wav * gain
        # Additive noise at ~25-40 dB SNR
        if rng.random() < 0.5:
            snr_db = rng.uniform(25.0, 40.0)
            p = float(np.mean(wav ** 2)) + 1e-12
            noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(p / (10 ** (snr_db / 10)))
            wav = wav + noise
        # Clip
        wav = np.clip(wav, -1.0, 1.0)
        return wav.astype(np.float32)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        wav = self._load_wave(s.path)
        if self.augment:
            wav = self._augment(wav)
        return wav, LABEL_TO_IDX[s.label]


def collate(batch):
    wavs, labels = zip(*batch)
    max_len = max(len(w) for w in wavs)
    padded = np.zeros((len(wavs), max_len), dtype=np.float32)
    for i, w in enumerate(wavs):
        padded[i, :len(w)] = w
    return torch.from_numpy(padded), torch.tensor(labels, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Model — wav2vec2 with a fresh classification head
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_labels: int, base_id: str = "facebook/wav2vec2-xls-r-300m",
                device: str = "cuda"):
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config

    # Try to load pretrained weights. If the base model is huge and network
    # is slow, we fall back to a config-only init (still trains fine with
    # enough data).
    try:
        logger.info("Loading base model: %s", base_id)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            base_id,
            num_labels=num_labels,
            label2id={l: i for i, l in enumerate(CANONICAL_LABELS)},
            id2label={i: l for i, l in enumerate(CANONICAL_LABELS)},
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        logger.warning("Could not load pretrained weights (%s); initializing fresh.", exc)
        cfg = Wav2Vec2Config.from_pretrained(base_id) if False else Wav2Vec2Config(
            num_labels=num_labels,
            id2label={i: l for i, l in enumerate(CANONICAL_LABELS)},
            label2id={l: i for i, l in enumerate(CANONICAL_LABELS)},
        )
        model = Wav2Vec2ForSequenceClassification(cfg)

    # Freeze the CNN feature extractor (standard wav2vec2 fine-tune practice)
    model.freeze_feature_encoder()

    # Optional: freeze the bottom transformer layers to speed up training.
    # XLS-R has 24 layers; freezing the first 12 halves compute.
    n_layers = len(model.wav2vec2.encoder.layers)
    for layer_idx in range(n_layers // 2):
        for p in model.wav2vec2.encoder.layers[layer_idx].parameters():
            p.requires_grad = False
    logger.info("Froze bottom %d of %d transformer layers.", n_layers // 2, n_layers)

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (manual — HF Trainer has Windows multiprocessing issues)
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Device: %s", device)

    # 1. Discover
    samples = discover(args.data_dir)
    logger.info("Discovered %d audio files.", len(samples))
    if not samples:
        logger.error("No audio found. Expected class folders under %s", args.data_dir)
        return 1

    per_class = Counter(s.label for s in samples)
    per_folder = Counter(s.folder for s in samples)
    logger.info("Files per canonical class: %s", dict(per_class))
    logger.info("Original folder names: %s", dict(per_folder))

    # Warn about missing classes
    missing = [c for c in CANONICAL_LABELS if per_class.get(c, 0) == 0]
    if missing:
        logger.warning("No training data for classes: %s. "
                        "The model will not learn them.", missing)

    train_s, val_s = split_train_val(samples, args.val_frac, args.seed)
    logger.info("Train: %d  Val: %d", len(train_s), len(val_s))

    if args.smoke_test:
        logger.info("[smoke-test] Truncating to 20 train, 10 val samples.")
        train_s = train_s[:20]
        val_s = val_s[:10]

    # 2. Datasets
    train_ds = UrduSERDataset(train_s, target_sr=16000,
                              max_seconds=args.max_seconds, augment=True)
    val_ds   = UrduSERDataset(val_s, target_sr=16000,
                              max_seconds=args.max_seconds, augment=False)

    # Class-balanced sampler — keeps minority classes visible
    class_counts = Counter(s.label for s in train_s)
    weights = [1.0 / class_counts[s.label] for s in train_s]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_s),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # 3. Model
    model = build_model(num_labels=len(CANONICAL_LABELS), device=device.type)

    # 4. Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * max(1, len(train_loader))
    warmup = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Training loop
    os.makedirs(args.output, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_n, tr_correct = 0.0, 0, 0
        for step, (wavs, y) in enumerate(train_loader, 1):
            wavs = wavs.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(wavs).logits
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_loss += float(loss.item()) * len(y)
            tr_n += len(y)
            tr_correct += int((logits.argmax(1) == y).sum().item())

            if step % 10 == 0 or step == len(train_loader):
                print(f"  epoch {epoch:2d} step {step:3d}/{len(train_loader)}  "
                      f"loss={tr_loss/max(1,tr_n):.4f}  acc={tr_correct/max(1,tr_n):.3f}",
                      flush=True)

        # Validation
        model.eval()
        val_loss, val_n, val_correct = 0.0, 0, 0
        per_class_c = Counter()
        per_class_t = Counter()
        with torch.no_grad():
            for wavs, y in val_loader:
                wavs = wavs.to(device); y = y.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(wavs).logits
                    loss = F.cross_entropy(logits, y)
                val_loss += float(loss.item()) * len(y)
                val_n += len(y)
                preds = logits.argmax(1)
                val_correct += int((preds == y).sum().item())
                for t, p in zip(y.tolist(), preds.tolist()):
                    per_class_t[t] += 1
                    if t == p:
                        per_class_c[t] += 1

        val_acc = val_correct / max(1, val_n)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.4f",
            epoch, args.epochs, tr_loss/max(1,tr_n), tr_correct/max(1,tr_n),
            val_loss/max(1,val_n), val_acc,
        )
        pc = {CANONICAL_LABELS[i]: round(per_class_c[i] / max(1, per_class_t[i]), 3)
              for i in range(len(CANONICAL_LABELS)) if per_class_t[i] > 0}
        logger.info("  per-class val acc: %s", pc)

        # Save checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "canonical_labels": CANONICAL_LABELS,
            "val_acc": val_acc,
        }
        torch.save(ckpt, os.path.join(args.output, "checkpoint_last.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.output, "checkpoint_best.pt"))
            logger.info("  -> New best val_acc=%.4f", best_val_acc)

        # Early stop if val_acc is not improving and we're past the plateau
        if epoch >= 3 and val_acc < best_val_acc - 0.05:
            logger.info("Val acc dropping sharply; stopping early at epoch %d.", epoch)
            break

    # 6. Save final HF-format model
    logger.info("Loading best checkpoint for final save.")
    best = torch.load(os.path.join(args.output, "checkpoint_best.pt"), map_location="cpu")
    model.load_state_dict(best["model_state"])

    final_dir = args.output
    model.save_pretrained(final_dir)
    logger.info("Saved HF-format model to %s", final_dir)

    # Also save a feature extractor so from_pretrained works cleanly downstream
    try:
        from transformers import Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=False,
        )
        fe.save_pretrained(final_dir)
    except Exception as exc:
        logger.warning("Could not save feature extractor: %s", exc)

    logger.info("Done. Best val_acc=%.4f  Total time=%.1f min",
                best_val_acc, (time.time()-t0)/60)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description="Fine-tune wav2vec2 for Urdu SER.")
    p.add_argument("--data-dir", default="./uploads/UrduSER",
                   help="Root folder containing class subfolders with audio files.")
    p.add_argument("--output", default="./models/audio/urdu_ser",
                   help="Where to save the trained model.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-device batch. 4 fits in ~5 GB VRAM on wav2vec2-xls-r-300m.")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-seconds", type=float, default=5.0,
                   help="Crop/pad every clip to this length.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU training (slow — only for tiny smoke tests).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run on 20 train / 10 val samples for a quick sanity check.")
    args = p.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.batch_size = 2

    sys.exit(train(args))


if __name__ == "__main__":
    _cli()