"""
modules/image_emotion/train_ensemble.py — Cross-dataset FER training
pipeline for AffectNet (YOLO format) + FER2013.

PURPOSE
-------
Train a single facial-expression classifier on the union of two
in-the-wild datasets so the resulting model generalizes to unseen
faces / cameras / demographics.

DATASETS
--------
AffectNet (YOLO format), expected at ./uploads/affectnet/YOLO_format/:
    data.yaml           (names: [Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise])
    train/images/*.jpg  +  train/labels/*.txt
    test/images/*.jpg   +  test/labels/*.txt
    valid/images/*.jpg  +  valid/labels/*.txt
Class 1 ("Contempt") is dropped — not in the canonical 7-class space.

FER2013, expected at ./uploads/fer2013/:
    train/<emotion>/*.jpg   and   test/<emotion>/*.jpg
    OR fer2013.csv (emotion, pixels, Usage)

GENERALIZATION TECHNIQUES
-------------------------
1. Unified 7-class label space (Ekman).
2. Class-balanced WeightedRandomSampler.
3. Domain-randomized augmentation (rotation, shear, brightness, blur,
   noise, grayscale, JPEG, resized crop).
4. Focal loss + label smoothing.
5. MixUp.
6. Domain-adversarial training (gradient reversal).

USAGE
-----
    python -m modules.image_emotion.train_ensemble ^
        --data-root ./uploads ^
        --output ./models/face_ensemble ^
        --epochs 20 --batch-size 64 --lr 3e-5
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("mindcare.image_emotion.train_ensemble")


# ── Canonical label space ──────────────────────────────────────────────────

# Order MATCHES trpakov/vit-face-expression's id2label exactly:
#   {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
#    4: 'neutral', 5: 'sad', 6: 'surprise'}
# Any change here must be mirrored by the base model, or training
# will teach the head the wrong class-to-index mapping.
CANONICAL_CLASSES = [
    "anger",       # 0  <- base: 'angry'
    "disgust",     # 1
    "fear",        # 2
    "happy",       # 3  <- base: 'happy' (we accept 'happiness' as synonym)
    "neutral",     # 4
    "sad",         # 5  <- base: 'sad' (we accept 'sadness' as synonym)
    "surprise",    # 6
]
# Explicit mapping that accepts BOTH the long names our scanners emit
# ("happiness", "sadness") AND the short names the base model uses
# ("happy", "sad"). Indices match trpakov/vit-face-expression's head:
#   {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
#    4: 'neutral', 5: 'sad', 6: 'surprise'}
CLASS_TO_IDX = {
    "anger": 0, "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3, "happiness": 3,
    "neutral": 4,
    "sad": 5, "sadness": 5,
    "surprise": 6,
}

# AffectNet raw label -> canonical
_AFFECTNET_MAP = {
    "anger": "anger", "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness", "happiness": "happiness",
    "sad": "sadness", "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
    # "contempt" -> DROPPED
}

# FER2013 raw label -> canonical
_FER2013_MAP = {
    "angry": "anger", "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness", "happiness": "happiness",
    "sad": "sadness", "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

# FER2013 CSV numeric labels (0-6) -> canonical
_FER2013_CSV_IDX = {
    0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
    4: "sadness", 5: "surprise", 6: "neutral",
}


# ── Sample container ───────────────────────────────────────────────────────

@dataclass
class Sample:
    path: str
    label: str
    dataset: str
    subject: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # normalized x1,y1,x2,y2

    def __repr__(self):
        return f"Sample({self.dataset}, {self.label}, {os.path.basename(self.path)})"


# ── Dataset scanners ───────────────────────────────────────────────────────

def _parse_data_yaml(yaml_path: Path) -> Dict[int, str]:
    """Parse YOLO data.yaml -> {class_id: class_name}. Uses PyYAML if
    available, otherwise a tolerant regex fallback."""
    if not yaml_path.exists():
        return {}
    text = yaml_path.read_text(encoding="utf-8", errors="replace")

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("PyYAML parse failed (%s); falling back to regex.", exc)

    import re
    m = re.search(r"names\s*:\s*\[([^\]]*)\]", text)
    if m:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        return {i: v for i, v in enumerate(items)}
    m = re.search(r"names\s*:\s*\n((?:[ \t]+\d+\s*:\s*.+\n?)+)", text)
    if m:
        out: Dict[int, str] = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[int(k.strip())] = v.strip().strip("'\"")
                except ValueError:
                    continue
        return out
    return {}


def _read_yolo_label(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO label file -> [(class_id, x1, y1, x2, y2), ...] normalized."""
    out: List[Tuple[int, float, float, float, float]] = []
    if not txt_path.exists():
        return out
    try:
        for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            except (ValueError, IndexError):
                continue
            x1 = max(0.0, cx - w / 2.0)
            y1 = max(0.0, cy - h / 2.0)
            x2 = min(1.0, cx + w / 2.0)
            y2 = min(1.0, cy + h / 2.0)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((cid, x1, y1, x2, y2))
    except Exception as exc:
        logger.debug("YOLO label read failed for %s: %s", txt_path, exc)
    return out


def _scan_affectnet(root: Path) -> List[Sample]:
    """AffectNet YOLO layout:
        root/YOLO_format/data.yaml
        root/YOLO_format/{train,test,valid}/images/*.jpg
        root/YOLO_format/{train,test,valid}/labels/*.txt
    Also falls back to class-folder layout: root/Train/<emotion>/*.jpg."""
    if root is None or not root.exists():
        logger.warning("AffectNet root not found: %s", root)
        return []

    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    samples: List[Sample] = []

    # YOLO: find data.yaml up to 3 levels deep
    yaml_candidates: List[Path] = []
    for depth in range(0, 4):
        pattern = "/".join(["*"] * depth) + ("/data.yaml" if depth else "data.yaml")
        yaml_candidates.extend(root.glob(pattern))
    yaml_path = next((p for p in yaml_candidates if p.is_file()), None)

    if yaml_path is not None:
        class_names = _parse_data_yaml(yaml_path)
        logger.info("AffectNet YOLO: data.yaml at %s, classes=%s", yaml_path, class_names)
        yolo_root = yaml_path.parent

        id_to_canonical: Dict[int, Optional[str]] = {}
        for cid, raw_name in class_names.items():
            id_to_canonical[cid] = _AFFECTNET_MAP.get(raw_name.lower().strip())

        def _walk_yolo_split(split_dir: Path):
            for images_dir in split_dir.rglob("images"):
                if not images_dir.is_dir():
                    continue
                labels_dir = images_dir.parent / "labels"
                for img in images_dir.iterdir():
                    if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                        continue
                    lbl = labels_dir / (img.stem + ".txt")
                    boxes = _read_yolo_label(lbl)
                    if not boxes:
                        continue
                    cid, x1, y1, x2, y2 = max(
                        boxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2])
                    )
                    canon = id_to_canonical.get(cid)
                    if canon is None:
                        continue
                    samples.append(Sample(
                        path=str(img.resolve()),
                        label=canon,
                        dataset="affectnet",
                        subject=None,
                        bbox=(x1, y1, x2, y2),
                    ))

        found_split = False
        for name in ("train", "test", "valid", "val", "validation", "training"):
            split_dir = yolo_root / name
            if split_dir.exists() and split_dir.is_dir():
                _walk_yolo_split(split_dir)
                found_split = True

        if found_split and samples:
            per_class = Counter(s.label for s in samples)
            logger.info("AffectNet (YOLO layout): %d samples across %d classes: %s",
                        len(samples), len(per_class), dict(per_class))
            return samples

    # Fallback: classic class-folder layout
    for split_name in ("Train", "Training", "train", "Val", "Validation",
                       "val", "Test", "test"):
        split_dir = root / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            canon = _AFFECTNET_MAP.get(cls_dir.name.lower().strip())
            if canon is None:
                continue
            for img in cls_dir.iterdir():
                if img.is_file() and img.suffix.lower() in IMG_EXTS:
                    samples.append(Sample(
                        path=str(img.resolve()),
                        label=canon,
                        dataset="affectnet",
                        subject=None,
                    ))

    if not samples:
        logger.warning("AffectNet: no usable layout under %s", root)
    else:
        logger.info("AffectNet (class-folder layout): %d samples", len(samples))
    return samples


def _scan_fer2013(root: Path) -> List[Sample]:
    """FER2013 in either folder-per-class layout or CSV layout."""
    samples: List[Sample] = []
    if root is None or not root.exists():
        logger.warning("FER2013 root not found: %s", root)
        return []

    # Folder-per-class layout
    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.exists() and train_dir.is_dir():
        for split in [train_dir, test_dir]:
            if not split.exists():
                continue
            for cls_dir in sorted(split.iterdir()):
                if not cls_dir.is_dir():
                    continue
                canon = _FER2013_MAP.get(cls_dir.name.lower().strip())
                if canon is None:
                    continue
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        samples.append(Sample(
                            path=str(p.resolve()),
                            label=canon,
                            dataset="fer2013",
                            subject=None,
                        ))
        logger.info("FER2013 (folder layout): %d samples", len(samples))
        return samples

    # CSV layout
    csv_path = root / "fer2013.csv"
    if not csv_path.exists():
        for cand in root.glob("*.csv"):
            if "fer" in cand.name.lower():
                csv_path = cand
                break

    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row["emotion"])
                    usage = row.get("Usage", "Training").strip().lower()
                except (KeyError, ValueError):
                    continue
                canon = _FER2013_CSV_IDX.get(idx)
                if canon is None:
                    continue
                samples.append(Sample(
                    path=f"mem://{usage}/{len(samples)}",
                    label=canon,
                    dataset="fer2013",
                    subject=None,
                ))
        logger.info("FER2013 (CSV layout): %d samples", len(samples))
        return samples

    logger.warning("FER2013: no recognised layout under %s", root)
    return samples


# ── FER2013 CSV in-memory store ────────────────────────────────────────────

class Fer2013CsvStore:
    def __init__(self, csv_path: str):
        self.rows: Dict[int, np.ndarray] = {}
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    idx = int(row["emotion"])
                    pixels = row["pixels"]
                except (KeyError, ValueError):
                    continue
                if _FER2013_CSV_IDX.get(idx) is None:
                    continue
                arr = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
                self.rows[i] = arr
        logger.info("Fer2013CsvStore: cached %d rows.", len(self.rows))

    def get(self, idx: int) -> np.ndarray:
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)


# ── Augmentation ───────────────────────────────────────────────────────────

class DomainRandomizedAugment:
    """Heavy augmentation spanning the axis of variation between
    AffectNet (in-the-wild photos) and FER2013 (grayscale low-res)."""

    def __init__(self, size: int = 224, seed: int = 0):
        self.size = size
        self.rng = random.Random(seed)

    def _u(self, lo, hi):
        return self.rng.uniform(lo, hi)

    def __call__(self, pil: Image.Image) -> Image.Image:
        img = np.array(pil.convert("RGB"))
        h, w = img.shape[:2]

        if self.rng.random() < 0.5:
            img = cv2.flip(img, 1)
        if self.rng.random() < 0.7:
            angle = self._u(-20.0, 20.0)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if self.rng.random() < 0.4:
            shear = self._u(-0.18, 0.18)
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if self.rng.random() < 0.8:
            alpha = self._u(0.7, 1.3)
            beta = self._u(-30, 30)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.35:
            k = self.rng.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), sigmaX=self._u(0.4, 2.0))
        if self.rng.random() < 0.25:
            sigma = self._u(3.0, 12.0)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.2:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

        crop_scale = self._u(0.78, 1.0)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        x0 = self.rng.randint(0, max(0, w - cw))
        y0 = self.rng.randint(0, max(0, h - ch))
        img = img[y0:y0 + ch, x0:x0 + cw]
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)

        if self.rng.random() < 0.3:
            ok, enc = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.rng.randint(35, 85)])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return Image.fromarray(img)


# ── Dataset ────────────────────────────────────────────────────────────────

class UnifiedFERDataset:
    def __init__(self, samples: List[Sample], processor, augment: bool,
                 size: int, fer_csv_store: Optional[Fer2013CsvStore] = None):
        self.samples = samples
        self.processor = processor
        self.augment = augment
        self.aug = DomainRandomizedAugment(size=size, seed=hash(str(samples)) & 0xFFFF)
        self.fer_csv = fer_csv_store

    def __len__(self):
        return len(self.samples)

    def _load_image(self, s: Sample) -> Image.Image:
        if s.path.startswith("mem://"):
            row_idx = int(s.path.rsplit("/", 1)[-1])
            arr = self.fer_csv.get(row_idx)
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = Image.open(s.path).convert("RGB")

        if s.bbox is not None:
            w, h = img.size
            x1n, y1n, x2n, y2n = s.bbox
            bw, bh = (x2n - x1n), (y2n - y1n)
            pad = 0.15
            x1n = max(0.0, x1n - pad * bw)
            y1n = max(0.0, y1n - pad * bh)
            x2n = min(1.0, x2n + pad * bw)
            y2n = min(1.0, y2n + pad * bh)
            box_px = (int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h))
            if box_px[2] > box_px[0] and box_px[3] > box_px[1]:
                img = img.crop(box_px)
        return img

    def __getitem__(self, i):
        import torch
        s = self.samples[i]
        img = self._load_image(s)
        if self.augment:
            img = self.aug(img)
        else:
            size = self.processor.size.get("height", 224) if hasattr(self.processor, "size") else 224
            img = img.resize((size, size), Image.BILINEAR)
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        y = torch.tensor(CLASS_TO_IDX[s.label], dtype=torch.long)
        d = torch.tensor({"affectnet": 0, "fer2013": 1}[s.dataset], dtype=torch.long)
        return px, y, d


# ── Domain-adversarial components ──────────────────────────────────────────

def _build_grad_reverse():
    import torch
    from torch.autograd import Function

    class _GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = lambd
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambd * grad_output, None

    class GradReverse(torch.nn.Module):
        def __init__(self, lambd: float = 1.0):
            super().__init__()
            self.lambd = lambd

        def forward(self, x):
            return _GradReverse.apply(x, self.lambd)

    return GradReverse


def _build_domain_classifier(hidden_size: int, n_domains: int = 2, grl_lambda: float = 1.0):
    import torch
    import torch.nn as nn
    GRL = _build_grad_reverse()

    class DomainHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.grl = GRL(grl_lambda)
            self.net = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, n_domains),
            )

        def forward(self, features):
            return self.net(self.grl(features))

    return DomainHead()


# ── Losses ─────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, gamma: float = 2.0):
    import torch
    import torch.nn.functional as F
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    targets_oh = F.one_hot(targets, num_classes=logits.size(-1)).float()
    focal_weight = (1.0 - probs).pow(gamma)
    loss = -(targets_oh * focal_weight * log_probs).sum(dim=-1)
    return loss.mean()


def mixup_batch(px, y, alpha: float = 0.2):
    import torch
    if alpha <= 0:
        return px, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(px.size(0), device=px.device)
    mixed = lam * px + (1 - lam) * px[idx]
    return mixed, y, y[idx], lam


# ── Training routine ───────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from .model import get_model_and_processor, IMAGE_EMOTION_MODEL_ID

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    def _split_other(samples, val_frac=0.10):
        local_rng = random.Random(args.seed)
        idx = list(range(len(samples)))
        local_rng.shuffle(idx)
        n_val = max(1, int(len(samples) * val_frac))
        val_idx = set(idx[:n_val])
        return ([s for i, s in enumerate(samples) if i not in val_idx],
                [s for i, s in enumerate(samples) if i in val_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    data_root = Path(args.data_root).resolve()
    logger.info("Searching for datasets under: %s", data_root)

    # ── 1. Discover dataset folders ────────────────────────────────────
    def _find_dir(root: Path, aliases: list, max_depth: int = 5):
        alias_set = {a.lower() for a in aliases}
        if not root.exists():
            return None
        queue = [(root, 0)]
        seen = set()
        while queue:
            d, depth = queue.pop(0)
            try:
                d_resolved = d.resolve()
            except Exception:
                continue
            if d_resolved in seen:
                continue
            seen.add(d_resolved)
            if depth > max_depth:
                continue
            try:
                entries = sorted(d.iterdir())
            except (PermissionError, OSError):
                continue
            for entry in entries:
                if not entry.is_dir():
                    continue
                if entry.name.lower() in alias_set:
                    return entry
                queue.append((entry, depth + 1))
        return None

    an_dir  = _find_dir(data_root, ["affectnet", "affect_net"])
    fer_dir = _find_dir(data_root, ["fer2013", "fer-2013", "fer_2013", "fer2013_dataset"])

    logger.info("Resolved dataset directories:")
    logger.info("  affectnet  -> %s", an_dir)
    logger.info("  fer2013    -> %s", fer_dir)

    an  = _scan_affectnet(an_dir) if an_dir else []
    fer = _scan_fer2013(fer_dir) if fer_dir else []

    fer_csv_store = None
    if fer and fer[0].path.startswith("mem://"):
        csv_path = fer_dir / "fer2013.csv"
        if not csv_path.exists():
            for cand in fer_dir.glob("*.csv"):
                if "fer" in cand.name.lower():
                    csv_path = cand
                    break
        fer_csv_store = Fer2013CsvStore(str(csv_path))

    all_samples = an + fer
    if not all_samples:
        try:
            present = sorted(p.name for p in data_root.iterdir() if p.is_dir())
        except Exception:
            present = []
        raise RuntimeError(
            f"No samples found under {data_root}.\n"
            f"  Folders present: {present or '(none)'}\n"
            f"  affectnet -> {an_dir}\n"
            f"  fer2013   -> {fer_dir}"
        )

    per_ds = Counter(s.dataset for s in all_samples)
    per_cls = Counter(s.label for s in all_samples)
    logger.info("Total samples: %d", len(all_samples))
    logger.info("  Per dataset: %s", dict(per_ds))
    logger.info("  Per class:   %s", dict(per_cls))

    # ── 2. Train / val split ───────────────────────────────────────────
    an_train, an_val = _split_other(an) if an else ([], [])
    fer_train, fer_val = _split_other(fer) if fer else ([], [])

    train_samples = an_train + fer_train
    val_samples   = an_val + fer_val
    logger.info("Train=%d  Val=%d", len(train_samples), len(val_samples))

    # ── 3. Class-balanced sampler ──────────────────────────────────────
    ds_counts  = Counter(s.dataset for s in train_samples)
    cls_counts = Counter(s.label for s in train_samples)
    weights = [
        (1.0 / ds_counts[s.dataset]) * (1.0 / cls_counts[s.label])
        for s in train_samples
    ]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_samples),
        replacement=True,
    )

    # ── 4. Model ───────────────────────────────────────────────────────
    model, processor, _ = get_model_and_processor()
      
    model = model.to(device)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(pixel_values=dummy, output_hidden_states=True)
        hidden_size = out.hidden_states[-1].shape[-1]
    logger.info("Backbone hidden size: %d", hidden_size)

    domain_head = _build_domain_classifier(
        hidden_size, n_domains=2, grl_lambda=args.grl_lambda
    ).to(device)

    # ── 5. Optimizer / scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr},
        {"params": domain_head.parameters(), "lr": args.lr * 5.0},
    ], weight_decay=0.02)

    total_steps  = args.epochs * max(1, len(train_samples) // args.batch_size)
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 6. DataLoaders ─────────────────────────────────────────────────
    proc_size = processor.size.get("height", 224) if hasattr(processor, "size") else 224

    train_ds = UnifiedFERDataset(train_samples, processor, augment=True,
                                 size=proc_size, fer_csv_store=fer_csv_store)
    val_ds = UnifiedFERDataset(val_samples, processor, augment=False,
                               size=proc_size, fer_csv_store=fer_csv_store) if val_samples else None

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True) if val_ds else None

    # ── 7. Training loop ───────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        domain_head.train()
        running_cls, running_dom, n_batches = 0.0, 0.0, 0

        for px, y, d in train_dl:
            px = px.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)

            use_mixup = args.mixup_alpha > 0 and random.random() < 0.5
            if use_mixup:
                px_m, y_a, y_b, lam = mixup_batch(px, y, alpha=args.mixup_alpha)
            else:
                px_m, y_a, y_b, lam = px, y, y, 1.0

            optimizer.zero_grad()
            out = model(pixel_values=px_m, output_hidden_states=True)
            logits = out.logits
            feats = out.hidden_states[-1].mean(dim=1)

            if use_mixup:
                loss_cls = (lam * focal_loss(logits, y_a, gamma=args.focal_gamma)
                            + (1 - lam) * focal_loss(logits, y_b, gamma=args.focal_gamma))
            else:
                loss_cls = focal_loss(logits, y, gamma=args.focal_gamma)

            if args.lambda_domain > 0 and not use_mixup:
                dom_logits = domain_head(feats)
                loss_dom = F.cross_entropy(dom_logits, d)
            else:
                loss_dom = torch.tensor(0.0, device=device)

            loss = loss_cls + args.lambda_domain * loss_dom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_cls += float(loss_cls.item())
            running_dom += float(loss_dom.item())
            n_batches += 1

        train_cls = running_cls / max(1, n_batches)
        train_dom = running_dom / max(1, n_batches)

        # Validation
        val_acc, val_per_class = None, None
        if val_dl is not None:
            model.eval()
            correct, total = 0, 0
            per_cls_correct = defaultdict(int)
            per_cls_total = defaultdict(int)
            with torch.no_grad():
                for px, y, _ in val_dl:
                    px = px.to(device); y = y.to(device)
                    logits = model(pixel_values=px).logits
                    pred = logits.argmax(dim=-1)
                    correct += int((pred == y).sum().item())
                    total += int(y.numel())
                    for t, p in zip(y.tolist(), pred.tolist()):
                        per_cls_total[t] += 1
                        if t == p:
                            per_cls_correct[t] += 1
            val_acc = correct / max(1, total)
            val_per_class = {
                CANONICAL_CLASSES[i]: round(per_cls_correct[i] / max(1, per_cls_total[i]), 3)
                for i in range(len(CANONICAL_CLASSES))
                if per_cls_total[i] > 0
            }
            logger.info(
                "Epoch %d/%d | cls_loss=%.4f dom_loss=%.4f | val_acc=%.4f | per_class=%s",
                epoch, args.epochs, train_cls, train_dom, val_acc, val_per_class,
            )
        else:
            logger.info("Epoch %d/%d | cls_loss=%.4f dom_loss=%.4f",
                        epoch, args.epochs, train_cls, train_dom)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "domain_head_state": domain_head.state_dict(),
            "canonical_classes": CANONICAL_CLASSES,
            "base_model_id": IMAGE_EMOTION_MODEL_ID,
        }
        torch.save(ckpt, os.path.join(args.output, "checkpoint_last.pt"))
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.output, "checkpoint_best.pt"))
            logger.info("  -> New best (val_acc=%.4f)", best_val_acc)

    # ── 8. Save HF-compatible model ────────────────────────────────────
    logger.info("Saving HF-compatible model to %s", args.output)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    import json
    with open(os.path.join(args.output, "canonical_classes.json"), "w") as f:
        json.dump({"classes": CANONICAL_CLASSES, "class_to_idx": CLASS_TO_IDX}, f, indent=2)
    logger.info("Done. Best val_acc=%.4f", best_val_acc)


# ── CLI ────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(
        description="Train a cross-dataset FER model on AffectNet + FER2013."
    )
    p.add_argument("--data-root",
                   default=os.environ.get("MINDCARE_UPLOAD_DIR", "./uploads"),
                   help="Root to search for dataset folders. Defaults to ./uploads.")
    p.add_argument("--output", default="./models/face_ensemble",
                   help="Where to write the fine-tuned HF model.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--lambda-domain", type=float, default=0.3)
    p.add_argument("--grl-lambda", type=float, default=1.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train(args)


if __name__ == "__main__":
    _cli()