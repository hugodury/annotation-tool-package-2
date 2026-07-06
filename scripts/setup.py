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
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / "venv"
IS_WINDOWS = platform.system() == "Windows"


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def find_python() -> str:
    for cmd in ("python3", "python", "py"):
        if shutil.which(cmd):
            try:
                subprocess.run([cmd, "--version"], check=True, capture_output=True)
                return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    raise RuntimeError("Python 3.9+ not found. Install from https://www.python.org/downloads/")


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


def ollama_available(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ollama_installed_models(host: str) -> set[str]:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
        names: set[str] = set()
        for m in data.get("models", []):
            names.add(m.get("name", "").split(":")[0])
            names.add(m.get("name", ""))
        return names
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return set()


def system_ram_gb() -> float:
    try:
        if IS_WINDOWS:
            ps = (
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out.isdigit():
                return int(out) / (1024**3)
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(out) / (1024**3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return 8.0


def load_cascade_config() -> dict:
    path = ROOT / "cascade" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_active_llm(cfg: dict, profile: dict) -> None:
    cfg["llm"] = {
        "label": profile["label"],
        "ollama": profile["ollama"],
        "prompt": profile.get("prompt", "P3"),
        "profile_id": profile["id"],
    }
    path = ROOT / "cascade" / "config.json"
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_installed(tag: str, installed: set[str]) -> bool:
    return tag in installed or tag.split(":")[0] in installed


def select_llm_profile(cfg: dict, installed: set[str]) -> dict | None:
    override = os.environ.get("OLLAMA_LLM_MODEL", "").strip()
    profiles: list[dict] = cfg.get("llm_profiles", [])
    ram = system_ram_gb()

    if override:
        for p in profiles:
            if p["ollama"] == override or p["id"] == override:
                return p
        return {"id": "custom", "label": override, "ollama": override, "prompt": "P3", "min_ram_gb": 0}

    if not cfg.get("llm_auto_select", True):
        return cfg.get("llm")

    candidates = sorted(profiles, key=lambda p: p.get("priority", 99))

    # Prefer an already-pulled model that fits this machine.
    for p in candidates:
        if _model_installed(p["ollama"], installed) and ram >= p.get("min_ram_gb", 0) - 1:
            return p

    # Nothing installed yet: pick the best profile for available RAM (smallest fallback last).
    for p in candidates:
        if ram >= p.get("min_ram_gb", 0) - 1:
            return p
    return candidates[-1] if candidates else None


def ollama_pull(model: str) -> bool:
    if not shutil.which("ollama"):
        return False
    print(f"Pulling Ollama model: {model} (may take several minutes)...")
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_ollama_llm(cfg: dict) -> None:
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")

    if not shutil.which("ollama"):
        print(
            "\nERROR: Ollama is required but not installed.\n"
            "  1. Install from https://ollama.com/download\n"
            "  2. Re-run: ./start.sh  (or start.bat on Windows)\n"
            "\nDeBERTa-only annotation works without Ollama, but LLM arbitration will fail."
        )
        if os.environ.get("REQUIRE_OLLAMA", "0") == "1":
            raise RuntimeError("Ollama not installed")
        return

    if not ollama_available(host):
        print("Ollama not responding — attempting to start ollama serve...")
        if IS_WINDOWS:
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        for _ in range(15):
            time.sleep(1)
            if ollama_available(host):
                break
        else:
            print(
                f"WARNING: Ollama not reachable at {host}.\n"
                "  Start it manually: ollama serve"
            )
            return

    installed = ollama_installed_models(host)
    profile = select_llm_profile(cfg, installed)
    if not profile:
        print("WARNING: No LLM profile configured in cascade/config.json")
        return

    tag = profile["ollama"]
    if not _model_installed(tag, installed):
        if not ollama_pull(tag):
            fallbacks = sorted(
                [p for p in cfg.get("llm_profiles", []) if p["ollama"] != tag],
                key=lambda p: p.get("priority", 99),
            )
            for fb in fallbacks:
                print(f"Trying fallback LLM: {fb['ollama']}")
                if ollama_pull(fb["ollama"]):
                    profile = fb
                    break
            else:
                print("WARNING: Could not pull any LLM model.")
                return

    save_active_llm(cfg, profile)
    print(f"Active LLM: {profile['label']} ({profile['ollama']})")


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
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"RAM: ~{system_ram_gb():.1f} Go")

    try:
        python_cmd = find_python()

        if not args.skip_models:
            ensure_models(python_cmd)

        ensure_venv(python_cmd)
        install_dependencies()
        py = str(venv_python())

        cfg = load_cascade_config()
        if not args.skip_llm:
            ensure_ollama_llm(cfg)

        print("\nSetup complete.")

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
