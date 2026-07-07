#!/usr/bin/env python3
"""Installation automatique de Python 3.9+ et Ollama si absents."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
AUTO_INSTALL = os.environ.get("AUTO_INSTALL_PREREQS", "1") != "0"
PYTHON_TARGET = "3.12"


def _run(cmd: list[str], check: bool = True) -> bool:
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=check)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_python_cmd() -> str | None:
    for cmd in ("python3", "python", "py"):
        if shutil.which(cmd):
            try:
                out = subprocess.check_output([cmd, "-c", "import sys; print(sys.version_info[:2])"], text=True)
                major, minor = map(int, out.strip().strip("()").split(", "))
                if major > 3 or (major == 3 and minor >= 9):
                    return cmd
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                continue
    return None


def install_python() -> str | None:
    if not AUTO_INSTALL:
        return None
    print(f"Python {PYTHON_TARGET} (ou 3.9+) introuvable — tentative d'installation automatique...")

    if IS_WINDOWS and shutil.which("winget"):
        if _run(
            [
                "winget", "install", "-e",
                "--id", "Python.Python.3.12",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            check=False,
        ):
            return find_python_cmd()

    if IS_MAC and shutil.which("brew"):
        if _run(["brew", "install", "python@3.12"], check=False):
            for cmd in ("python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3"):
                if shutil.which(cmd):
                    return cmd

    if IS_LINUX:
        if shutil.which("apt-get"):
            if _run(["sudo", "apt-get", "update"], check=False):
                _run(
                    ["sudo", "apt-get", "install", "-y", "python3", "python3-venv", "python3-pip"],
                    check=False,
                )
                return find_python_cmd()
        if shutil.which("dnf"):
            _run(["sudo", "dnf", "install", "-y", "python3", "python3-pip"], check=False)
            return find_python_cmd()

    return find_python_cmd()


def ensure_python() -> str:
    cmd = find_python_cmd()
    if cmd:
        print(f"Python : OK ({cmd})")
        return cmd
    cmd = install_python()
    if cmd:
        print(f"Python {PYTHON_TARGET} installé : {cmd}")
        return cmd
    raise RuntimeError(
        f"Python {PYTHON_TARGET} (ou 3.9+) requis mais introuvable.\n"
        "  Installez-le : https://www.python.org/downloads/\n"
        "  Windows : cochez « Add Python to PATH » pendant l'installation."
    )


def ollama_version() -> str | None:
    if not shutil.which("ollama"):
        return None
    try:
        out = subprocess.check_output(
            ["ollama", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        return out.splitlines()[0] if out else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def install_ollama() -> bool:
    if not AUTO_INSTALL:
        return False
    if shutil.which("ollama"):
        return True

    print("Ollama introuvable — tentative d'installation automatique...")

    if IS_WINDOWS:
        if shutil.which("winget"):
            return _run(
                [
                    "winget", "install", "-e",
                    "--id", "Ollama.Ollama",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ],
                check=False,
            )
        print(
            "  Installez Ollama manuellement : https://ollama.com/download\n"
            "  Puis relancez start.bat"
        )
        return False

    if IS_MAC or IS_LINUX:
        if not shutil.which("curl"):
            print("  curl requis pour installer Ollama.")
            return False
        return _run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], check=False)

    return False


def ensure_ollama_binary() -> None:
    if shutil.which("ollama"):
        ver = ollama_version()
        print(f"Ollama : OK{f' ({ver})' if ver else ''}")
        return
    if install_ollama():
        # winget / install script may need a fresh PATH
        for _ in range(5):
            time.sleep(2)
            if shutil.which("ollama"):
                print("Ollama installé.")
                return
    if not shutil.which("ollama"):
        raise RuntimeError(
            "Ollama est obligatoire pour l'annotation assistée par LLM.\n"
            "  Installez-le : https://ollama.com/download\n"
            "  Puis relancez l'application."
        )


def main() -> int:
    try:
        ensure_python()
        ensure_ollama_binary()
        print("Prérequis OK.")
        return 0
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
