#!/usr/bin/env python3
"""Publish Cascade V8 assets to GitHub Release (maintainer, multi-OS).

Crée les assets (si besoin), crée/upload la release tag v8.0.0, asset par asset
pour limiter l'espace disque.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "cascade-v8"
TAG = "v8.0.0"
REPO = "hugodury/annotation-tool-package-2"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    if not OUT.is_dir() or not any(OUT.iterdir()):
        run([sys.executable, str(ROOT / "scripts" / "package_models_v8.py")])

    assets = sorted(
        p
        for p in OUT.iterdir()
        if p.is_file() and p.name != "models-v8.manifest.json"
    )
    if not assets:
        print("Aucun asset dans", OUT, file=sys.stderr)
        return 1

    # Release exists?
    exists = subprocess.call(
        ["gh", "release", "view", TAG, "--repo", REPO],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    notes = (
        "## Cascade V8 models (multi-OS / multi-PC)\n\n"
        "Poids **exclus de Git** (comme `vldbench-models-v1`).\n\n"
        "| Asset | Contenu |\n"
        "|-------|--------|\n"
        "| `vldbench-cascade-v8-meta.tar.gz` | `config.json` + tokenizers |\n"
        "| `vldbench-cascade-v8-minilm.safetensors` | MiniLM v7 |\n"
        "| `vldbench-cascade-v8-deberta.safetensors` | DeBERTa Large v8.1 |\n"
        "| `*.reranker.safetensors.partXX` | Reranker découpé (< 2 Go / fichier GitHub) |\n\n"
        "Installation auto via `./start.sh` / `start.ps1` / `start.bat` "
        "(`scripts/download_models_v8.py`).\n\n"
        "Voir `DEPLOYMENT.md` et `models-v8.manifest.json`."
    )

    if exists != 0:
        # create empty release then upload
        run(
            [
                "gh",
                "release",
                "create",
                TAG,
                "--repo",
                REPO,
                "--title",
                "Cascade V8 models",
                "--notes",
                notes,
            ]
        )
    else:
        print(f"Release {TAG} existe déjà — upload des assets manquants.")

    for asset in assets:
        size_gb = asset.stat().st_size / (1024**3)
        print(f"Uploading {asset.name} ({size_gb:.2f} GiB)…")
        run(
            [
                "gh",
                "release",
                "upload",
                TAG,
                str(asset),
                "--repo",
                REPO,
                "--clobber",
            ]
        )

    print(f"OK — https://github.com/{REPO}/releases/tag/{TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
