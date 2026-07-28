#!/usr/bin/env python3
"""Download and verify fine-tuned ML models for offline use."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "models.manifest.json"

sys.path.insert(0, str(ROOT / "scripts"))
from disk_check import require_disk_for_models  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def models_complete(manifest: dict) -> bool:
    for rel in manifest.get("required_files", []):
        if not (ROOT / rel).is_file():
            return False
    checksums = manifest.get("checksums", {})
    for rel, expected in checksums.items():
        path = ROOT / rel
        if not path.is_file():
            return False
        if _sha256(path) != expected:
            print(f"Checksum mismatch: {rel}", file=sys.stderr)
            return False
    return True


def sbert_complete(manifest: dict | None = None) -> bool:
    """SBERT alone is required for Run Model similarity scores."""
    rel = "models/fine_tuned_sbert/model.safetensors"
    path = ROOT / rel
    if not path.is_file():
        return False
    if manifest is None and MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = (manifest or {}).get("checksums", {}).get(rel)
    if expected and _sha256(path) != expected:
        print(f"Checksum mismatch: {rel}", file=sys.stderr)
        return False
    return True


def _download(url: str, dest: Path) -> None:
    print(f"Downloading models from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "vldbench-setup/1.0"})
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Models not found at {url}\n"
                "The maintainer must publish vldbench-models-v1.tar.gz on GitHub Releases (tag v1.0.0),\n"
                "or set MODELS_DOWNLOAD_URL to a valid archive URL."
            ) from e
        raise
    with resp_ctx as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 1024
        with open(dest, "wb") as out:
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                out.write(data)
                done += len(data)
                if total:
                    pct = min(100, int(done * 100 / total))
                    print(f"\r  {pct}% ({done // (1024 * 1024)} Mo)", end="", flush=True)
        print()


def _resolve_urls(manifest: dict) -> list[str]:
    urls: list[str] = []
    env_url = os.environ.get("MODELS_DOWNLOAD_URL", "").strip()
    for raw in manifest.get("archive", {}).get("urls", []):
        if raw == "${MODELS_DOWNLOAD_URL}":
            if env_url:
                urls.append(env_url)
        elif raw and raw not in urls:
            urls.append(raw)
    return urls


def download_models(force: bool = False) -> bool:
    if not MANIFEST_PATH.is_file():
        print(f"Manifest missing: {MANIFEST_PATH}", file=sys.stderr)
        return False

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # Run Model needs SBERT (similarity). Archive v1 also ships legacy DeBERTa-base / CE.
    if not force and sbert_complete(manifest):
        print("SBERT (similarity) already present and verified.")
        base_ok = True
    else:
        base_ok = False
        require_disk_for_models(ROOT)

        archive_name = manifest.get("archive", {}).get("filename", "vldbench-models-v1.tar.gz")
        urls = _resolve_urls(manifest)
        if not urls:
            print(
                "No download URL configured. Set MODELS_DOWNLOAD_URL or publish a GitHub Release.\n"
                "See DEPLOYMENT.md for maintainer instructions.",
                file=sys.stderr,
            )
            return False

        print("Downloading Release v1.0.0 (SBERT + legacy weights)…")
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / archive_name
            last_err: Exception | None = None
            for url in urls:
                try:
                    _download(url, archive_path)
                    last_err = None
                    break
                except (urllib.error.URLError, OSError, RuntimeError) as e:
                    last_err = e
                    print(f"  Failed: {e}", file=sys.stderr)
            if last_err is not None:
                return False

            print("Extracting archive...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=ROOT)

        if sbert_complete(manifest):
            print("SBERT installed successfully.")
            base_ok = True
            if not models_complete(manifest):
                print(
                    "Note: full v1 package incomplete, but SBERT (required) is OK.",
                    file=sys.stderr,
                )
        else:
            print("Download finished but SBERT verification failed.", file=sys.stderr)
            return False

    # Cascade V8 (MiniLM + DeBERTa Large + Reranker) — même flux Release multi-OS
    try:
        from download_models_v8 import download_v8_models
    except ImportError:
        sys.path.insert(0, str(ROOT / "scripts"))
        from download_models_v8 import download_v8_models  # type: ignore

    v8_ok = download_v8_models(force=force)
    if not v8_ok:
        print(
            "Avertissement: modèles Cascade V8 non installés — "
            "les modes « Cascade V8 + Qwen » / Compare seront indisponibles.\n"
            "Relancer: python scripts/download_models_v8.py\n"
            "Doc: DEPLOYMENT.md (release v8.0.0).",
            file=sys.stderr,
        )
    return base_ok and v8_ok


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download VLDBench models (SBERT + Cascade V8) — multi-OS / fresh git clone"
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    ok = download_models(force=args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
