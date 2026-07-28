#!/usr/bin/env python3
"""Create Cascade V8 release assets for GitHub (maintainer only).

GitHub Release limit = 2 GiB / asset. Le Reranker (~2.1 Go) est découpé en parts.
Les autres poids sont uploadés en fichiers bruts ou petites archives.

Sortie dans dist/cascade-v8/ :
  - vldbench-cascade-v8-meta.tar.gz          (config + tokenizers, sans poids)
  - vldbench-cascade-v8-minilm.safetensors
  - vldbench-cascade-v8-deberta.safetensors
  - vldbench-cascade-v8-reranker.safetensors.part00
  - vldbench-cascade-v8-reranker.safetensors.part01
  - models-v8.manifest.json (checksums remplis)

Usage:
  python scripts/package_models_v8.py
  python scripts/package_models_v8.py --source /path/to/cascade_annotation_v8_complete
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = (
    ROOT.parent
    / "AI_annotation"
    / "cascade_annotation_v8_complete"
    / "cascade_annotation_v8_complete"
)
OUT_DIR = ROOT / "dist" / "cascade-v8"
PART_SIZE = 1100 * 1024 * 1024  # ~1.1 GiB — sous la limite GitHub 2 GiB
GITHUB_TAG = "v8.0.0"
REPO = "hugodury/annotation-tool-package-2"

DEST_MINILM = "models/cascade_v8/models/minilm_full_v7/model.safetensors"
DEST_DEBERTA = "models/cascade_v8/models/deberta_large_v8.1/model.safetensors"
DEST_RERANKER = "models/cascade_v8/models/reranker_undetermined_v8/model.safetensors"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_file(src: Path, dest_prefix: Path, part_size: int) -> list[Path]:
    parts: list[Path] = []
    with open(src, "rb") as f:
        idx = 0
        while True:
            data = f.read(part_size)
            if not data:
                break
            part = Path(f"{dest_prefix}.part{idx:02d}")
            part.write_bytes(data)
            parts.append(part)
            idx += 1
            print(f"  part {part.name}: {len(data) / (1024**3):.2f} GiB")
    return parts


def _release_url(filename: str) -> str:
    return (
        f"https://github.com/{REPO}/releases/download/{GITHUB_TAG}/{filename}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Cascade V8 models for GitHub Release")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Racine package collègue (config.json + models/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Dossier de sortie des assets",
    )
    args = parser.parse_args()
    source: Path = args.source.resolve()
    out: Path = args.out.resolve()

    if not (source / "config.json").is_file():
        raise SystemExit(f"config.json manquant dans {source}")
    for name in ("minilm_full_v7", "deberta_large_v8.1", "reranker_undetermined_v8"):
        weight = source / "models" / name / "model.safetensors"
        if not weight.is_file():
            raise SystemExit(f"Poids manquant: {weight}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # --- meta archive: config + configs/tokenizers (no safetensors) ---
    meta_name = "vldbench-cascade-v8-meta.tar.gz"
    meta_path = out / meta_name
    print(f"Packing {meta_name}…")
    with tarfile.open(meta_path, "w:gz") as tar:
        tar.add(source / "config.json", arcname="models/cascade_v8/config.json")
        rules = source / "CASCADE_RULES.md"
        if rules.is_file():
            tar.add(rules, arcname="models/cascade_v8/CASCADE_RULES.md")
        for name in ("minilm_full_v7", "deberta_large_v8.1", "reranker_undetermined_v8"):
            model_dir = source / "models" / name
            for f in model_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix == ".safetensors" or f.name.startswith("._"):
                    continue
                arc = Path("models/cascade_v8/models") / name / f.relative_to(model_dir)
                tar.add(f, arcname=str(arc))
    print(f"  {meta_path.name}: {meta_path.stat().st_size / (1024**2):.1f} MiB")

    # --- raw weight copies (hardlink if possible to save disk) ---
    minilm_src = source / "models" / "minilm_full_v7" / "model.safetensors"
    deberta_src = source / "models" / "deberta_large_v8.1" / "model.safetensors"
    reranker_src = source / "models" / "reranker_undetermined_v8" / "model.safetensors"

    minilm_name = "vldbench-cascade-v8-minilm.safetensors"
    deberta_name = "vldbench-cascade-v8-deberta.safetensors"
    minilm_out = out / minilm_name
    deberta_out = out / deberta_name

    print(f"Copying {minilm_name}…")
    try:
        minilm_out.hardlink_to(minilm_src)
    except OSError:
        shutil.copy2(minilm_src, minilm_out)
    print(f"  {minilm_out.stat().st_size / (1024**2):.1f} MiB")

    print(f"Copying {deberta_name}…")
    try:
        deberta_out.hardlink_to(deberta_src)
    except OSError:
        shutil.copy2(deberta_src, deberta_out)
    print(f"  {deberta_out.stat().st_size / (1024**3):.2f} GiB")

    print("Splitting reranker (GitHub asset limit 2 GiB)…")
    reranker_parts = _split_file(
        reranker_src, out / "vldbench-cascade-v8-reranker.safetensors", PART_SIZE
    )

    # checksums of final assembled files (source weights)
    checksums = {
        DEST_MINILM: _sha256(minilm_src),
        DEST_DEBERTA: _sha256(deberta_src),
        DEST_RERANKER: _sha256(reranker_src),
    }

    archives = [
        {
            "filename": meta_name,
            "kind": "tar.gz",
            "urls": [_release_url(meta_name), "${CASCADE_V8_DOWNLOAD_URL_META}"],
        },
        {
            "filename": minilm_name,
            "kind": "raw",
            "dest": DEST_MINILM,
            "urls": [_release_url(minilm_name), "${CASCADE_V8_DOWNLOAD_URL_MINILM}"],
        },
        {
            "filename": deberta_name,
            "kind": "raw",
            "dest": DEST_DEBERTA,
            "urls": [_release_url(deberta_name), "${CASCADE_V8_DOWNLOAD_URL_DEBERTA}"],
        },
        {
            "filename": "vldbench-cascade-v8-reranker.safetensors",
            "kind": "parts",
            "dest": DEST_RERANKER,
            "parts": [p.name for p in reranker_parts],
            "urls_template": _release_url("{part}"),
            "urls_env_prefix": "CASCADE_V8_DOWNLOAD_URL_RERANKER_PART",
        },
    ]

    manifest = {
        "version": "8.0.0",
        "description": (
            "Cascade V8 multi-OS: MiniLM + DeBERTa Large + Reranker undetermined. "
            "Poids exclus de Git — distribués via GitHub Release."
        ),
        "github_release_tag": GITHUB_TAG,
        "required_free_gb": 7.0,
        "required_files": [
            "models/cascade_v8/config.json",
            DEST_MINILM,
            DEST_DEBERTA,
            DEST_RERANKER,
        ],
        "checksums": checksums,
        "archives": archives,
    }

    # Write manifest into dist AND repo root (download reads ROOT/models-v8.manifest.json)
    manifest_path = out / "models-v8.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "models-v8.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nAssets prêts dans", out)
    for p in sorted(out.iterdir()):
        print(f"  {p.name}: {p.stat().st_size / (1024**2):.1f} MiB")
    print(
        "\nPublier:\n"
        f"  gh release create {GITHUB_TAG} {out}/* \\\n"
        f"    --repo {REPO} \\\n"
        '    --title "Cascade V8 models" \\\n'
        '    --notes "MiniLM + DeBERTa Large + Reranker (multi-OS). Reranker split en parts <2Go."\n'
        "Ou: python scripts/publish_models_v8.py"
    )


if __name__ == "__main__":
    main()
