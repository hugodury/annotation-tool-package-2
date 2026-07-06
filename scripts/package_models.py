#!/usr/bin/env python3
"""Create vldbench-models-v1.tar.gz for GitHub Release (maintainer only)."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT = ROOT / "vldbench-models-v1.tar.gz"
MANIFEST = ROOT / "models.manifest.json"


def main() -> None:
    if not MODELS_DIR.is_dir():
        raise SystemExit(f"Missing {MODELS_DIR}. Place models/ before packaging.")

    required = json.loads(MANIFEST.read_text())["required_files"]
    for rel in required:
        if not (ROOT / rel).is_file():
            raise SystemExit(f"Missing required file: {rel}")

    if OUT.exists():
        OUT.unlink()

    with tarfile.open(OUT, "w:gz") as tar:
        for item in MODELS_DIR.rglob("*"):
            if item.is_file():
                arcname = Path("models") / item.relative_to(MODELS_DIR)
                tar.add(item, arcname=str(arcname))

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Created {OUT} ({size_mb:.1f} Mo)")
    print("Upload to GitHub Release tag v1.0.0 as: vldbench-models-v1.tar.gz")


if __name__ == "__main__":
    main()
