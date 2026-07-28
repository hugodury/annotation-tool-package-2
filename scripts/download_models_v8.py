#!/usr/bin/env python3
"""Download Cascade V8 models from GitHub Release (multi-OS / multi-PC)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "models-v8.manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def v8_models_complete(manifest: dict | None = None) -> bool:
    if manifest is None:
        if not MANIFEST_PATH.is_file():
            return False
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for rel in manifest.get("required_files", []):
        if not (ROOT / rel).is_file():
            return False
    for rel, expected in (manifest.get("checksums") or {}).items():
        path = ROOT / rel
        if not path.is_file():
            return False
        if _sha256(path) != expected:
            print(f"Checksum mismatch (V8): {rel}", file=sys.stderr)
            return False
    return True


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "vldbench-setup-v8/1.0"})
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=1200)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Cascade V8 asset not found: {url}\n"
                "Publish release tag v8.0.0 (see DEPLOYMENT.md) or set CASCADE_V8_DOWNLOAD_URL_*."
            ) from e
        raise
    with resp_ctx as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 1024
        dest.parent.mkdir(parents=True, exist_ok=True)
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


def _resolve_url(raw: str, *, part_name: str | None = None, part_idx: int | None = None) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("${") and raw.endswith("}"):
        env_key = raw[2:-1]
        if part_idx is not None and env_key.endswith("_PART"):
            # CASCADE_V8_DOWNLOAD_URL_RERANKER_PART0 / PART1
            val = os.environ.get(f"{env_key}{part_idx}", "").strip()
            return val or None
        val = os.environ.get(env_key, "").strip()
        return val or None
    if "{part}" in raw and part_name:
        return raw.replace("{part}", part_name)
    return raw


def _try_urls(urls: list[str], dest: Path) -> None:
    last_err: Exception | None = None
    for url in urls:
        if not url:
            continue
        try:
            _download(url, dest)
            return
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            last_err = e
            print(f"  Failed: {e}", file=sys.stderr)
    raise RuntimeError(str(last_err) if last_err else "No URL worked")


def _require_disk(manifest: dict) -> None:
    import shutil

    need = float(manifest.get("required_free_gb", 7.0))
    free = shutil.disk_usage(ROOT).free / (1024**3)
    # Si déjà partiellement installé, on laisse passer avec un warning bas
    if free > 0 and free < need and not any(
        (ROOT / rel).is_file() for rel in manifest.get("required_files", [])
    ):
        raise RuntimeError(
            f"Espace disque insuffisant pour Cascade V8.\n"
            f"  Libre : {free:.1f} Go\n"
            f"  Requis : ~{need:.1f} Go (poids + extraction)\n"
            f"  Libérez de l'espace puis relancez ./start.sh"
        )
    if free > 0:
        print(f"  Espace disque OK (V8) : {free:.1f} Go libres (besoin ~{need:.1f} Go)")


def download_v8_models(force: bool = False) -> bool:
    if not MANIFEST_PATH.is_file():
        print(f"Manifest V8 missing: {MANIFEST_PATH}", file=sys.stderr)
        return False

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not force and v8_models_complete(manifest):
        print("Cascade V8 models already present and verified.")
        return True

    _require_disk(manifest)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for arch in manifest.get("archives", []):
            kind = arch.get("kind", "tar.gz")
            filename = arch["filename"]

            if kind == "tar.gz":
                urls = [
                    u
                    for u in (_resolve_url(x) for x in arch.get("urls", []))
                    if u
                ]
                dest = tmp_path / filename
                _try_urls(urls, dest)
                print(f"Extracting {filename}…")
                with tarfile.open(dest, "r:gz") as tar:
                    tar.extractall(path=ROOT)

            elif kind == "raw":
                dest_rel = arch["dest"]
                dest = ROOT / dest_rel
                if dest.is_file() and not force:
                    print(f"  Skip existing {dest_rel}")
                    continue
                urls = [
                    u
                    for u in (_resolve_url(x) for x in arch.get("urls", []))
                    if u
                ]
                _try_urls(urls, dest)

            elif kind == "parts":
                dest_rel = arch["dest"]
                dest = ROOT / dest_rel
                if dest.is_file() and not force:
                    print(f"  Skip existing {dest_rel}")
                    continue
                parts = arch.get("parts") or []
                part_files: list[Path] = []
                for i, part_name in enumerate(parts):
                    urls: list[str] = []
                    template = arch.get("urls_template")
                    if template:
                        u = _resolve_url(template, part_name=part_name)
                        if u:
                            urls.append(u)
                    env_prefix = arch.get("urls_env_prefix")
                    if env_prefix:
                        env_u = os.environ.get(f"{env_prefix}{i}", "").strip()
                        if env_u:
                            urls.append(env_u)
                    part_path = tmp_path / part_name
                    _try_urls(urls, part_path)
                    part_files.append(part_path)
                print(f"Assembling {dest_rel} from {len(part_files)} parts…")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as out:
                    for part_path in part_files:
                        with open(part_path, "rb") as inp:
                            shutil_copyfileobj = __import__("shutil").copyfileobj
                            shutil_copyfileobj(inp, out)
            else:
                print(f"Unknown archive kind: {kind}", file=sys.stderr)
                return False

    if v8_models_complete(manifest):
        print("Cascade V8 models installed successfully.")
        return True

    print("V8 download finished but verification failed.", file=sys.stderr)
    return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download Cascade V8 models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return 0 if download_v8_models(force=args.force) else 1


if __name__ == "__main__":
    raise SystemExit(main())
