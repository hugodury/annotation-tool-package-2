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
    if not force and models_complete(manifest):
        print("ML models already present and verified.")
        return True

    archive_name = manifest.get("archive", {}).get("filename", "vldbench-models-v1.tar.gz")
    urls = _resolve_urls(manifest)
    if not urls:
        print(
            "No download URL configured. Set MODELS_DOWNLOAD_URL or publish a GitHub Release.\n"
            "See DEPLOYMENT.md for maintainer instructions.",
            file=sys.stderr,
        )
        return False

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

    if models_complete(manifest):
        print("Models installed successfully.")
        return True

    print("Download finished but model verification failed.", file=sys.stderr)
    return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download VLDBench fine-tuned models")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    ok = download_models(force=args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
