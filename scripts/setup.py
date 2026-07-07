#!/usr/bin/env python3
"""Cross-platform setup: venv, PyTorch, ML models, Ollama LLM, optional app launch."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / "venv"
IS_WINDOWS = platform.system() == "Windows"

sys.path.insert(0, str(ROOT / "scripts"))
from disk_check import estimate_disk_need  # noqa: E402
from ollama_service import ensure_ollama_ready  # noqa: E402
from prerequisites import ensure_ollama_binary, ensure_python  # noqa: E402
from system_check import build_report, print_report, ram_gb  # noqa: E402


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def ensure_venv(python_cmd: str) -> None:
    if venv_python().is_file():
        print("Virtual environment: OK")
        return
    print("Creating virtual environment...")
    run([python_cmd, "-m", "venv", str(VENV_DIR)])


def torch_install_args() -> list[str]:
    """Pick a PyTorch wheel suited to the host (CPU/CUDA/MPS)."""
    if os.environ.get("PYTORCH_INDEX_URL"):
        return ["--index-url", os.environ["PYTORCH_INDEX_URL"]]

    system = platform.system()
    try:
        import torch  # noqa: F401 — only if already installed in venv

        return []
    except ImportError:
        pass

    if system == "Darwin":
        return []  # default pip torch supports MPS on Apple Silicon
    if system == "Linux" and shutil.which("nvidia-smi"):
        return ["--index-url", "https://download.pytorch.org/whl/cu124"]
    if system == "Windows" and os.environ.get("CUDA_PATH"):
        return ["--index-url", "https://download.pytorch.org/whl/cu124"]
    return ["--index-url", "https://download.pytorch.org/whl/cpu"]


def install_dependencies() -> None:
    py = str(venv_python())
    print("Installing Python dependencies...")
    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    torch_args = torch_install_args()
    if torch_args:
        run([py, "-m", "pip", "install", "torch", *torch_args])
    run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])


def ensure_models(python_cmd: str | None = None) -> None:
    print("Checking ML models (DeBERTa, SBERT, cross-encoder)...")
    py = python_cmd or str(venv_python())
    run([py, str(ROOT / "scripts" / "download_models.py")])


def load_cascade_config() -> dict:
    path = ROOT / "cascade" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_ollama_llm(cfg: dict) -> None:
    print("Préparation Ollama (service + LLM)...")
    ensure_ollama_ready(cfg, ROOT / "cascade" / "config.json", install_binary=True, pull_llm=True)


def open_browser(url: str) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", url])
        elif system == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", url])
    except OSError:
        pass


def start_app(host: str, port: int, open_tab: bool) -> None:
    py = str(venv_python())
    env = os.environ.copy()
    env.setdefault("FLASK_HOST", host)
    env.setdefault("FLASK_PORT", str(port))
    if open_tab:
        import threading

        threading.Timer(2.0, lambda: open_browser(f"http://{host}:{port}")).start()
    print(f"\nStarting app at http://{host}:{port}\n")
    os.chdir(ROOT)
    run([py, str(ROOT / "app.py")])


def main() -> int:
    parser = argparse.ArgumentParser(description="ISIALAB Annotation Interface setup")
    parser.add_argument("--start", action="store_true", help="Launch Flask after setup")
    parser.add_argument("--skip-models", action="store_true", help="Skip ML model download")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Ollama LLM setup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("=== ISIALAB Annotation Interface — Setup ===")

    try:
        python_cmd = ensure_python()

        if not args.skip_llm:
            ensure_ollama_binary()

        cfg = load_cascade_config()
        disk = estimate_disk_need(ROOT, cfg, ram_gb_val=ram_gb())
        if not disk.get("disk_ok"):
            print(
                f"⚠ Espace disque limité : {disk['disk_free_gb']} Go libres, "
                f"~{disk['disk_required_gb']} Go recommandés."
            )
        else:
            print(
                f"Espace disque : {disk['disk_free_gb']} Go libres "
                f"(besoin estimé ~{disk['disk_required_gb']} Go)"
            )

        if not args.skip_models:
            ensure_models(python_cmd)

        ensure_venv(python_cmd)
        install_dependencies()
        py = str(venv_python())

        cfg = load_cascade_config()
        if not args.skip_llm:
            ensure_ollama_llm(cfg)

        report = build_report(ROOT, cfg)
        print_report(report)
        if report["errors"]:
            print("\n⚠ Avertissements détectés — l'application démarre quand même.")

        print("\nSetup complete — vous pouvez utiliser Run Model (avertissements possibles).")

        if args.start:
            start_app(args.host, args.port, not args.no_browser)
        else:
            print(f"Run: {py} app.py")
            print(f"Or:  ./start.sh   (Mac/Linux)  |  start.bat / start.ps1 (Windows)")

        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nSetup failed (exit {e.returncode})", file=sys.stderr)
        return e.returncode or 1
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
