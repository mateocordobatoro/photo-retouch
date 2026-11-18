"""Project, model, and record locations."""
from pathlib import Path

# src/photo_retouch/paths.py -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not (PROJECT_ROOT / "requirements.txt").is_file():
    raise FileNotFoundError(
        f"{PROJECT_ROOT} is not the photo-retouch project. Install the package "
        "in editable mode: python -m pip install --no-deps -e ."
    )

MODEL_DIR = PROJECT_ROOT / "models"
RECORDS_DIR = PROJECT_ROOT / "records"


def latest_record_dir():
    """Most recent snapshot that contains the fixed validation masks."""
    candidates = sorted(
        path.parent for path in RECORDS_DIR.glob("*/validation_mixed_masks.npz")
    )
    if not candidates:
        raise FileNotFoundError(f"No validation snapshot in {RECORDS_DIR}.")
    return candidates[-1]
