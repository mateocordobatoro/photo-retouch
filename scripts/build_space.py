"""Assemble the Hugging Face Space in outputs/space/ (ignored by Git).

The Space gets the app, the package modules it imports, the two stage-1
models, its own requirements, and its README (the Space card). Upload the
folder as it is; see README.md, "Publish the app".

    python scripts/build_space.py
"""
import shutil
from pathlib import Path

from photo_retouch.models import COARSE_MODEL_FILE, SAVED_REFINERS
from photo_retouch.paths import MODEL_DIR, PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "space"
PACKAGE = PROJECT_ROOT / "src" / "photo_retouch"
PACKAGE_FILES = ["__init__.py", "paths.py", "models.py", "inpaint.py"]
MODEL_FILES = [COARSE_MODEL_FILE, SAVED_REFINERS["Low-chroma refiner"][0]]
APP_FILES = {
    "app.py": "app.py",
    "requirements-space.txt": "requirements.txt",
    "SPACE_README.md": "README.md",
}


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "src" / "photo_retouch").mkdir(parents=True)
    (OUT / "models").mkdir()
    for source, target in APP_FILES.items():
        shutil.copyfile(PROJECT_ROOT / "app" / source, OUT / target)
    for name in PACKAGE_FILES:
        shutil.copyfile(PACKAGE / name, OUT / "src" / "photo_retouch" / name)
    for name in MODEL_FILES:
        shutil.copyfile(MODEL_DIR / name, OUT / "models" / name)

    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        print(f"{path.stat().st_size / 1e6:8.2f} MB  {path.relative_to(OUT)}")


if __name__ == "__main__":
    main()
