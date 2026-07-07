#!/usr/bin/env python3
"""Démarrage automatique d'Ollama et installation du LLM (ollama pull)."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

_ensure_lock = threading.Lock()
_ensure_state: dict[str, Any] = {
    "status": "idle",  # idle | running | ready | error
    "message": "",
    "last_error": None,
}


def ensure_state() -> dict[str, Any]:
    return dict(_ensure_state)


def _set_state(status: str, message: str = "", error: str | None = None) -> None:
    _ensure_state["status"] = status
    _ensure_state["message"] = message
    _ensure_state["last_error"] = error


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
            name = m.get("name", "")
            if name:
                names.add(name)
                names.add(name.split(":")[0])
        return names
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return set()


def _model_installed(tag: str, installed: set[str]) -> bool:
    return tag in installed or tag.split(":")[0] in installed


def _run_quiet(cmd: list[str], **kwargs) -> bool:
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def _try_systemd_ollama() -> bool:
    if not IS_LINUX or not shutil.which("systemctl"):
        return False
    for scope in ("--user", ""):
        args = ["systemctl"]
        if scope:
            args.append(scope)
        args.extend(["start", "ollama"])
        if _run_quiet(args):
            time.sleep(2)
            if ollama_available("http://127.0.0.1:11434"):
                return True
    return False


def _try_brew_services_ollama() -> bool:
    if not IS_MAC or not shutil.which("brew"):
        return False
    if _run_quiet(["brew", "services", "start", "ollama"]):
        time.sleep(2)
        return ollama_available("http://127.0.0.1:11434")
    return False


def _spawn_ollama_serve() -> None:
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


def try_start_ollama(host: str = "http://127.0.0.1:11434", *, wait_seconds: int = 20) -> bool:
    """Démarre Ollama si installé mais inactif."""
    if ollama_available(host):
        return True
    if not shutil.which("ollama"):
        return False

    if _try_systemd_ollama() or _try_brew_services_ollama():
        return True

    try:
        _spawn_ollama_serve()
        for _ in range(wait_seconds):
            time.sleep(1)
            if ollama_available(host):
                return True
    except OSError:
        pass
    return ollama_available(host)


def ram_gb() -> float:
    try:
        if IS_WINDOWS:
            ps = "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out.isdigit():
                return int(out) / (1024**3)
        elif IS_MAC:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            return int(out) / (1024**3)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return 0.0


def select_llm_profile(cfg: dict, installed: set[str]) -> dict | None:
    override = os.environ.get("OLLAMA_LLM_MODEL", "").strip()
    profiles: list[dict] = cfg.get("llm_profiles", [])
    mem = ram_gb()

    if override:
        for p in profiles:
            if p["ollama"] == override or p["id"] == override:
                return p
        return {"id": "custom", "label": override, "ollama": override, "prompt": "P3", "min_ram_gb": 0}

    if not cfg.get("llm_auto_select", True) and cfg.get("llm"):
        return cfg["llm"]

    candidates = sorted(profiles, key=lambda p: p.get("priority", 99))
    for p in candidates:
        if _model_installed(p["ollama"], installed) and mem >= p.get("min_ram_gb", 0) - 1:
            return p
    for p in candidates:
        if mem >= p.get("min_ram_gb", 0) - 1:
            return p
    return candidates[-1] if candidates else None


def save_active_llm(cfg: dict, profile: dict, config_path: Path) -> None:
    cfg["llm"] = {
        "label": profile["label"],
        "ollama": profile["ollama"],
        "prompt": profile.get("prompt", "P3"),
        "profile_id": profile.get("id", "custom"),
    }
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ollama_pull(model: str) -> bool:
    if not shutil.which("ollama"):
        return False
    print(f"Téléchargement du LLM Ollama : {model} (peut prendre plusieurs minutes)...")
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_llm_pulled(cfg: dict, config_path: Path, *, quiet: bool = False) -> dict | None:
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    installed = ollama_installed_models(host)
    profile = select_llm_profile(cfg, installed)
    if not profile:
        if not quiet:
            print("Aucun profil LLM configuré dans cascade/config.json")
        return None

    tag = profile["ollama"]
    if not _model_installed(tag, installed):
        if not quiet:
            print(f"LLM « {tag} » absent — téléchargement en cours...")
        if not ollama_pull(tag):
            fallbacks = sorted(
                [p for p in cfg.get("llm_profiles", []) if p["ollama"] != tag],
                key=lambda p: p.get("priority", 99),
            )
            for fb in fallbacks:
                if not quiet:
                    print(f"Essai du LLM de secours : {fb['ollama']}")
                if ollama_pull(fb["ollama"]):
                    profile = fb
                    break
            else:
                raise RuntimeError(
                    "Impossible de télécharger un LLM Ollama. "
                    "Vérifiez votre connexion et l'espace disque."
                )

    save_active_llm(cfg, profile, config_path)
    if not quiet:
        print(f"LLM actif : {profile['label']} ({profile['ollama']})")
    return profile


def ensure_ollama_ready(
    cfg: dict,
    config_path: Path,
    *,
    install_binary: bool = True,
    pull_llm: bool = True,
    quiet: bool = False,
) -> bool:
    """
    Installe Ollama si besoin, démarre le service, télécharge le LLM si absent.
    Retourne True si Ollama répond (LLM pullé si pull_llm=True).
    """
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")

    if install_binary:
        from prerequisites import ensure_ollama_binary

        ensure_ollama_binary()

    if not try_start_ollama(host):
        raise RuntimeError(
            f"Ollama installé mais inaccessible ({host}).\n"
            "  Relancez l'application ou exécutez : ollama serve"
        )

    if pull_llm:
        ensure_llm_pulled(cfg, config_path, quiet=quiet)

    return True


def ensure_ollama_ready_async(
    cfg: dict,
    config_path: Path,
    *,
    install_binary: bool = True,
    pull_llm: bool = True,
) -> None:
    """Lance ensure_ollama_ready en arrière-plan (idempotent)."""

    def _worker() -> None:
        if not _ensure_lock.acquire(blocking=False):
            return
        try:
            if _ensure_state["status"] == "ready":
                return
            _set_state("running", "Démarrage d'Ollama et préparation du LLM...")
            ensure_ollama_ready(
                cfg,
                config_path,
                install_binary=install_binary,
                pull_llm=pull_llm,
                quiet=True,
            )
            _set_state("ready", "Ollama actif — LLM prêt")
        except Exception as e:
            _set_state("error", str(e), error=str(e))
        finally:
            _ensure_lock.release()

    if _ensure_state["status"] == "running":
        return
    threading.Thread(target=_worker, daemon=True, name="ollama-ensure").start()
