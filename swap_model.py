"""
swap_model.py — Swap the pipeline's active emotion model folder.

Usage:
    python swap_model.py list             # show available model folders
    python swap_model.py use <name>       # swap ./models/face to ./models/<name>
    python swap_model.py current          # show which model is active
"""
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path("./models")
ACTIVE     = MODELS_DIR / "face"


def list_models():
    if not MODELS_DIR.exists():
        print("No ./models folder found.")
        return
    print(f"\n{ACTIVE.resolve()}")
    active_target = None
    if ACTIVE.is_symlink():
        active_target = ACTIVE.resolve().name
        print(f"  → currently points at: {active_target}")
    elif ACTIVE.exists():
        print("  → currently a regular folder (not a link)")

    print(f"\nAvailable models in {MODELS_DIR.resolve()}:")
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or d.name == "face":
            continue
        marker = "  ← ACTIVE" if d.name == active_target else ""
        # Check for the files a HF model folder needs
        has_config = (d / "config.json").exists()
        has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
        status = "✓" if (has_config and has_weights) else "✗ (incomplete)"
        print(f"  {status}  {d.name}{marker}")


def use(name):
    target = MODELS_DIR / name
    if not target.exists():
        print(f"Model folder not found: {target.resolve()}")
        sys.exit(1)

    if not (target / "config.json").exists():
        print(f"Warning: {target.name} has no config.json — may fail to load.")

    # Move current ./models/face aside if it exists and is not a symlink
    if ACTIVE.exists() and not ACTIVE.is_symlink():
        backup = MODELS_DIR / "face_base_original"
        if backup.exists():
            shutil.rmtree(backup)
        ACTIVE.rename(backup)
        print(f"Moved previous ./models/face → {backup.name}")

    # Remove any previous symlink
    if ACTIVE.is_symlink():
        ACTIVE.unlink()

    # Create a symlink (Windows: requires admin OR developer mode)
    try:
        ACTIVE.symlink_to(target.resolve(), target_is_directory=True)
        print(f"✅ ./models/face → {target.name}")
    except OSError as e:
        # Symlink failed (permissions) — fall back to copying
        print(f"Symlink failed ({e}), copying instead...")
        shutil.copytree(target, ACTIVE)
        print(f"✅ Copied {target.name} → ./models/face")


def current():
    if not ACTIVE.exists():
        print("./models/face does not exist.")
        return
    if ACTIVE.is_symlink():
        print(f"./models/face → {ACTIVE.resolve()}")
    else:
        # Print the classifier label map so you know which model this is
        import json
        cfg = ACTIVE / "config.json"
        if cfg.exists():
            data = json.loads(cfg.read_text())
            labels = data.get("id2label", {})
            print(f"./models/face is a real folder")
            print(f"  labels: {labels}")
        else:
            print(f"./models/face is a real folder (no config.json)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        list_models()
    elif cmd == "use" and len(sys.argv) == 3:
        use(sys.argv[2])
    elif cmd == "current":
        current()
    else:
        print(__doc__)