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


def llm_model_pulled(ollama: dict[str, Any], cfg: dict) -> tuple[bool, str]:
    active = cfg.get("llm", {}).get("ollama", "") or "qwen2.5:7b-instruct"
    if not ollama.get("running"):
        return False, active
    models = ollama.get("models") or []
    pulled = any(active in m or active.split(":")[0] in m for m in models)
    return pulled, active


def deps_ok() -> dict[str, tuple[bool, str]]:
    out: dict[str, tuple[bool, str]] = {}
    try:
        import torch

        out["pytorch"] = (True, torch.__version__)
    except ImportError:
        out["pytorch"] = (False, "Relancez ./start.sh")
    try:
        import sentence_transformers as st

        ver = st.__version__
        parts = [int(p) for p in ver.split(".")[:2]]
        ok = parts[0] > 5 or (parts[0] == 5 and parts[1] >= 5)
        out["sentence_transformers"] = (ok, ver)
    except ImportError:
        out["sentence_transformers"] = (False, "Relancez ./start.sh")
    return out


def installation_complete(
    *,
    py_ok: bool,
    pt_ok: bool,
    st_ok: bool,
    models_ok: bool,
    ollama: dict[str, Any],
    llm_ok: bool,
) -> bool:
    return bool(
        py_ok
        and pt_ok
        and st_ok
        and models_ok
        and ollama.get("installed")
        and ollama.get("running")
        and llm_ok
    )


def build_checklist(
    *,
    py_ok: bool,
    py_ver: str,
    models_ok: bool,
    ollama: dict[str, Any],
    cfg: dict,
    mem: float,
    disk_info: dict[str, Any],
    gpu: dict[str, Any],
    platform: dict[str, Any],
    performance: str,
    performance_label: str,
    deps: dict[str, tuple[bool, str]],
) -> list[dict[str, Any]]:
    llm_ok, llm_tag = llm_model_pulled(ollama, cfg)
    disk_ok = disk_info.get("disk_ok", True)
    disk_free = disk_info.get("disk_free_gb")
    fresh_gb = disk_info.get("fresh_install_required_gb")
    install_done = disk_info.get("install_complete", False)

    ram_ok: bool | None
    if mem <= 0:
        ram_ok = None
    elif mem >= 10:
        ram_ok = True
    else:
        ram_ok = False

    pt_ok, pt_ver = deps.get("pytorch", (False, "?"))
    st_ok, st_ver = deps.get("sentence_transformers", (False, "?"))

    perf_ok: bool | None
    if performance in ("good", "acceptable"):
        perf_ok = True
    elif performance == "insufficient":
        perf_ok = False
    else:
        perf_ok = None

    os_label = platform.get("os", "?")
    arch = platform.get("arch", "?")

    return [
        {
            "id": "platform",
            "label": "Systeme",
            "ok": True,
            "required": False,
            "detail": f"{os_label} {arch}",
        },
        {
            "id": "python",
            "label": "Python 3.9+",
            "ok": py_ok,
            "required": True,
            "detail": f"v{py_ver}" if py_ok else "Installez Python 3.9+",
            "action": PYTHON_DOWNLOAD if not py_ok else None,
        },
        {
            "id": "pytorch",
            "label": "PyTorch",
            "ok": pt_ok,
            "required": True,
            "detail": f"v{pt_ver}" if pt_ok else pt_ver,
            "action": "./start.sh" if not pt_ok else None,
        },
        {
            "id": "sentence_transformers",
            "label": "sentence-transformers >= 5.5",
            "ok": st_ok,
            "required": True,
            "detail": f"v{st_ver}" if st_ok else f"v{st_ver} — modeles incompatibles",
            "action": "pip install 'sentence-transformers>=5.5.1'" if not st_ok else None,
        },
        {
            "id": "ml_models",
            "label": "Modeles ML (DeBERTa, SBERT, cross-encoder)",
            "ok": models_ok,
            "required": True,
            "detail": "Presents dans models/" if models_ok else "Manquants",
            "action": "./start.sh" if not models_ok else None,
        },
        {
            "id": "ollama_installed",
            "label": "Ollama installe",
            "ok": bool(ollama.get("installed")),
            "required": True,
            "detail": "OK" if ollama.get("installed") else "Non installe",
            "action": OLLAMA_DOWNLOAD if not ollama.get("installed") else None,
        },
        {
            "id": "ollama_running",
            "label": "Ollama actif (serve)",
            "ok": bool(ollama.get("running")),
            "required": True,
            "detail": "Service en cours" if ollama.get("running") else "ollama serve",
            "action": "ollama serve" if not ollama.get("running") else None,
        },
        {
            "id": "llm_qwen",
            "label": f"LLM {llm_tag}",
            "ok": llm_ok,
            "required": True,
            "detail": "Telecharge" if llm_ok else "En cours ou manquant",
            "action": f"ollama pull {llm_tag}" if not llm_ok else None,
        },
        {
            "id": "ram",
            "label": "RAM >= 10 Go",
            "ok": ram_ok,
            "required": False,
            "detail": f"{mem:.1f} Go" if mem > 0 else "Non detectee",
        },
        {
            "id": "disk",
            "label": "Espace disque",
            "ok": True if install_done else (disk_ok if fresh_gb else None),
            "required": False,
            "detail": (
                f"{disk_free} Go libres"
                if disk_free is not None
                else ""
            ),
            "action": (
                None
                if install_done
                else (
                    f"~{fresh_gb} Go libres requis pour terminer l'installation"
                    if fresh_gb and not disk_ok
                    else None
                )
            ),
        },
        {
            "id": "gpu",
            "label": "GPU (acceleration)",
            "ok": True if gpu.get("device") != "cpu" else None,
            "required": False,
            "detail": gpu.get("label", "CPU"),
        },
        {
            "id": "performance",
            "label": "Performance estimee",
            "ok": perf_ok,
            "required": False,
            "detail": performance_label,
        },
    ]


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
    llm_ok, _llm_tag = llm_model_pulled(ollama, cfg)
    deps = deps_ok()
    pt_ok, _ = deps.get("pytorch", (False, ""))
    st_ok, _ = deps.get("sentence_transformers", (False, ""))
    install_done = installation_complete(
        py_ok=py_ok,
        pt_ok=pt_ok,
        st_ok=st_ok,
        models_ok=models_ok,
        ollama=ollama,
        llm_ok=llm_ok,
    )

    disk_info: dict[str, Any] = {}
    try:
        from disk_check import estimate_disk_need

        disk_info = estimate_disk_need(
            root,
            cfg,
            install_complete=install_done,
            ram_gb_val=mem,
        )
    except Exception:
        disk_info = {"disk_free_gb": disk, "disk_ok": True, "install_complete": install_done}

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not py_ok:
        errors.append(
            f"Python {py_ver} détecté — Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ requis. "
            f"Téléchargement : {PYTHON_DOWNLOAD}"
        )
    elif not py_recommended and not install_done:
        warnings.append(
            f"Python {py_ver} fonctionne, mais Python 3.12+ est recommandé pour de meilleures performances."
        )

    if not install_done:
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
        if ollama["installed"] and ollama["running"] and rec_llm and not llm_ok:
            active = cfg.get("llm", {}).get("ollama", rec_llm["ollama"])
            warnings.append(
                f"LLM « {active} » pas encore téléchargé — "
                "téléchargement automatique en cours (plusieurs minutes)."
            )

        if mem > 0 and mem < 6:
            warnings.append(
                f"RAM faible (~{mem:.1f} Go). L'annotation sera très lente ou pourra échouer."
            )
        elif 0 < mem < 10:
            warnings.append(
                f"RAM limitée (~{mem:.1f} Go). Qwen 7B peut échouer ou être très lent."
            )

        if (
            disk_info.get("fresh_install_required_gb")
            and not disk_info.get("disk_ok")
        ):
            warnings.append(
                f"Espace disque insuffisant pour terminer l'installation : "
                f"{disk_info['disk_free_gb']} Go libres, "
                f"~{disk_info['fresh_install_required_gb']} Go recommandes."
            )

        if gpu["device"] == "cpu":
            warnings.append(
                "Pas de GPU détecté. DeBERTa et le LLM tourneront sur CPU — "
                "l'annotation automatique sera nettement plus lente."
            )

        if rec_llm:
            notes.append(f"LLM recommandé pour cette machine : {rec_llm['label']} ({rec_llm['ollama']})")

        notes.append("Completez la checklist obligatoire avant Run Model.")
    elif disk_info.get("disk_free_gb", 0) > 0 and disk_info.get("disk_free_gb", 0) < 0.3:
        warnings.append(
            f"Espace disque critique : {disk_info['disk_free_gb']} Go libres."
        )

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

    checklist = build_checklist(
        py_ok=py_ok,
        py_ver=py_ver,
        models_ok=models_ok,
        ollama=ollama,
        cfg=cfg,
        mem=mem,
        disk_info=disk_info,
        gpu=gpu,
        platform={"os": platform.system(), "arch": platform.machine()},
        performance=performance,
        performance_label=perf_labels.get(performance, performance),
        deps=deps,
    )
    ready_for_run_model = all(item["ok"] is True for item in checklist if item["required"])
    missing_required = [item["label"] for item in checklist if item["required"] and item["ok"] is not True]

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
            "fresh_install_disk_gb": disk_info.get("fresh_install_required_gb"),
            "disk_ok": disk_info.get("disk_ok", True),
            "disk_breakdown_gb": disk_info.get("disk_breakdown_gb"),
            "install_complete": install_done,
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
        "run_model_allowed": ready_for_run_model,
        "ready_for_run_model": ready_for_run_model,
        "install_complete": install_done,
        "missing_required": missing_required,
        "checklist": checklist,
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
