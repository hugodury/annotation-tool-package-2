#!/usr/bin/env python3
"""Analyse la configuration machine et produit garde-fous pour le setup et l'app web."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

IS_WINDOWS = platform.system() == "Windows"
OLLAMA_DOWNLOAD = "https://ollama.com/download"
PYTHON_DOWNLOAD = "https://www.python.org/downloads/"
PYTHON_MIN = (3, 9)
PYTHON_RECOMMENDED = (3, 12)


def _disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        return 0.0


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
        elif platform.system() == "Darwin":
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


def python_ok() -> tuple[bool, str, bool]:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    ok = v.major > 3 or (v.major == 3 and v.minor >= PYTHON_MIN[1])
    recommended = v.major > 3 or (v.major == 3 and v.minor >= PYTHON_RECOMMENDED[1])
    return ok, version, recommended


def gpu_info() -> dict[str, Any]:
    device = "cpu"
    label = "CPU uniquement"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            label = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            label = "Apple Silicon (MPS)"
    except ImportError:
        if shutil.which("nvidia-smi"):
            device = "cuda"
            label = "NVIDIA (CUDA probable, PyTorch pas encore installé)"
    return {"device": device, "label": label}


from ollama_service import ollama_available, try_start_ollama  # noqa: E402


def ollama_state(host: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    installed = shutil.which("ollama") is not None
    running = ollama_available(host) if installed else False
    models: list[str] = []
    if running:
        try:
            with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            pass
    return {"installed": installed, "running": running, "host": host, "models": models}


def ml_models_ok(root: Path) -> bool:
    required = [
        "models/fine_tuned_deberta_base_expanded/model.safetensors",
        "models/fine_tuned_sbert/model.safetensors",
        "models/fine_tuned_cross_encoder/model.safetensors",
    ]
    return all((root / p).is_file() for p in required)


def recommended_llm(ram: float, profiles: list[dict]) -> dict | None:
    ordered = sorted(profiles, key=lambda p: p.get("priority", 99))
    for p in ordered:
        if ram >= p.get("min_ram_gb", 0) - 1:
            return p
    return ordered[-1] if ordered else None


def build_report(root: Path | None = None, cfg: dict | None = None) -> dict[str, Any]:
    root = root or Path.cwd()
    cfg = cfg or {}
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    profiles: list[dict] = cfg.get("llm_profiles", [])

    py_ok, py_ver, py_recommended = python_ok()
    mem = ram_gb()
    disk = _disk_free_gb(root)
    ollama = ollama_state(host)
    gpu = gpu_info()
    models_ok = ml_models_ok(root)
    rec_llm = recommended_llm(mem, profiles) if profiles else None

    disk_info: dict[str, Any] = {}
    try:
        from disk_check import estimate_disk_need

        disk_info = estimate_disk_need(
            root, cfg, models_ok=models_ok, llm_profile=rec_llm, ram_gb_val=mem
        )
    except Exception:
        disk_info = {"disk_free_gb": disk, "disk_required_gb": None, "disk_ok": True}

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not py_ok:
        errors.append(
            f"Python {py_ver} détecté — Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ requis. "
            f"Téléchargement : {PYTHON_DOWNLOAD}"
        )
    elif not py_recommended:
        warnings.append(
            f"Python {py_ver} fonctionne, mais Python 3.12+ est recommandé pour de meilleures performances."
        )

    if not ollama["installed"]:
        warnings.append(
            "Ollama n'est pas installé — la partie LLM de la cascade échouera. "
            f"Téléchargement : {OLLAMA_DOWNLOAD}"
        )
    elif not ollama["running"]:
        warnings.append(
            "Ollama est installé mais le service ne répond pas encore. "
            "Démarrage automatique en cours (ollama serve)…"
        )

    if not models_ok:
        warnings.append(
            "Modèles ML manquants dans models/ (DeBERTa / SBERT / cross-encoder). "
            "Relancez ./start.sh pour les télécharger."
        )

    if mem > 0 and mem < 6:
        warnings.append(
            f"RAM faible (~{mem:.1f} Go). L'annotation sera très lente ou pourra échouer."
        )
    elif 0 < mem < 10:
        warnings.append(
            f"RAM limitée (~{mem:.1f} Go). Qwen 7B peut échouer ou être très lent."
        )

    if disk_info.get("disk_required_gb") and not disk_info.get("disk_ok"):
        warnings.append(
            f"Espace disque limité : {disk_info['disk_free_gb']} Go libres, "
            f"~{disk_info['disk_required_gb']} Go recommandés pour une installation complète."
        )
    elif disk > 0 and disk < 5:
        warnings.append(
            f"Espace disque faible (~{disk:.1f} Go libres). "
            "Prévoyez ~8–12 Go pour une installation complète."
        )

    if gpu["device"] == "cpu":
        warnings.append(
            "Pas de GPU détecté. DeBERTa et le LLM tourneront sur CPU — "
            "l'annotation automatique sera nettement plus lente."
        )

    if ollama["installed"] and ollama["running"] and rec_llm:
        active = cfg.get("llm", {}).get("ollama", rec_llm["ollama"])
        if not any(active in m or active.split(":")[0] in m for m in ollama["models"]):
            warnings.append(
                f"LLM « {active} » pas encore téléchargé — "
                "téléchargement automatique en cours (plusieurs minutes)."
            )
        notes.append(f"LLM recommandé pour cette machine : {rec_llm['label']} ({rec_llm['ollama']})")

    if mem >= 10 and gpu["device"] != "cpu":
        notes.append("Configuration confortable pour DeBERTa + Qwen 7B.")
    elif mem >= 6:
        notes.append("Configuration limitée — Qwen 7B sera lent sur cette machine.")

    notes.append(
        "Vous pouvez toujours cliquer Run Model — une machine limitée sera simplement plus lente."
    )

    if disk_info.get("disk_breakdown_gb"):
        parts = ", ".join(f"{k} ~{v} Go" for k, v in disk_info["disk_breakdown_gb"].items())
        notes.append(f"Espace estimé nécessaire : {disk_info['disk_required_gb']} Go ({parts}).")

    perf_labels = {
        "good": "Bonne — annotation rapide attendue",
        "acceptable": "Correcte — Qwen 7B possible mais lent",
        "slow": "Lente — CPU uniquement, soyez patient",
        "insufficient": "Limitée — vous pouvez quand même lancer Run Model",
    }

    can_manual = py_ok
    can_auto_deberta = py_ok and models_ok
    can_auto_full = (
        can_auto_deberta and ollama["installed"] and ollama["running"]
    )

    if not can_auto_full:
        performance = "insufficient"
    elif gpu["device"] == "cpu":
        performance = "slow"
    elif mem < 10:
        performance = "acceptable"
    else:
        performance = "good"

    return {
        "platform": {
            "os": platform.system(),
            "arch": platform.machine(),
            "python": py_ver,
        },
        "hardware": {
            "ram_gb": round(mem, 1) if mem else None,
            "disk_free_gb": round(disk, 1) if disk else None,
            "disk_required_gb": disk_info.get("disk_required_gb"),
            "disk_ok": disk_info.get("disk_ok", True),
            "disk_breakdown_gb": disk_info.get("disk_breakdown_gb"),
            "gpu": gpu,
        },
        "ollama": ollama,
        "ml_models": models_ok,
        "recommended_llm": rec_llm,
        "active_llm": cfg.get("llm"),
        "performance": performance,
        "performance_label": perf_labels.get(performance, performance),
        "can_manual_annotate": can_manual,
        "can_auto_annotate_deberta": can_auto_deberta,
        "can_auto_annotate_full": can_auto_full,
        "run_model_allowed": True,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "ollama_download_url": OLLAMA_DOWNLOAD,
        "python_download_url": PYTHON_DOWNLOAD,
    }


def print_report(report: dict[str, Any]) -> None:
    hw = report["hardware"]
    print("\n=== Vérification système ===")
    print(f"  OS        : {report['platform']['os']} {report['platform']['arch']}")
    print(f"  Python    : {report['platform']['python']}")
    if hw["ram_gb"]:
        print(f"  RAM       : {hw['ram_gb']} Go")
    if hw["disk_free_gb"]:
        print(f"  Disque    : {hw['disk_free_gb']} Go libres")
    print(f"  Accélération : {hw['gpu']['label']}")
    print(f"  Ollama    : {'installé' if report['ollama']['installed'] else 'ABSENT'} / "
          f"{'actif' if report['ollama']['running'] else 'inactif'}")
    print(f"  Modèles ML: {'OK' if report['ml_models'] else 'MANQUANTS'}")
    print(f"  Performance estimée : {report.get('performance_label', report['performance'])}")
    if hw.get("disk_required_gb"):
        print(f"  Disque requis : ~{hw['disk_required_gb']} Go")

    for msg in report["notes"]:
        print(f"  ℹ {msg}")
    for msg in report["warnings"]:
        print(f"  ⚠ {msg}")
    for msg in report["errors"]:
        print(f"  ✗ {msg}")

    if report["errors"]:
        print("\nErreurs bloquantes (setup) : corrigez avant de continuer.")
    elif report["warnings"]:
        print("\nAvertissements : Run Model reste possible, mais peut être lent ou partiel.")
    else:
        print("\nConfiguration compatible avec l'annotation automatique complète.")


def require_ready(report: dict[str, Any], *, require_ollama: bool = True) -> None:
    blockers = list(report["errors"])
    if require_ollama and not report["ollama"]["installed"]:
        if not any("Ollama" in e for e in blockers):
            blockers.append(f"Ollama requis — {OLLAMA_DOWNLOAD}")
    if blockers:
        print_report(report)
        raise RuntimeError(
            "Configuration incompatible. Corrigez les erreurs ci-dessus puis relancez."
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Vérification configuration machine")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    parser.add_argument("--root", default=".", help="Racine du projet")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg_path = root / "cascade" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    report = build_report(root, cfg)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
