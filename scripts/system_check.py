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
# Qwen 7B alone ≈ 10 GB. Cascade (Duo+Reranker in RAM) + Qwen → prefer ≥16 GB.
RAM_QWEN_MIN_GB = 10.0
RAM_CASCADE_RECOMMENDED_GB = 16.0


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
    label = "CPU only"
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
            label = "NVIDIA (CUDA likely, PyTorch not installed yet)"
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


def ml_models_sbert_ok(root: Path) -> bool:
    """SBERT v2 (Release v8.0.0) — scores de similarité."""
    return (root / "models/fine_tuned_sbert/model.safetensors").is_file()


def ml_models_base_ok(root: Path) -> bool:
    """Legacy check — full old v1 package (unused by Run Model)."""
    required = [
        "models/fine_tuned_deberta_base_expanded/model.safetensors",
        "models/fine_tuned_sbert/model.safetensors",
        "models/fine_tuned_cross_encoder/model.safetensors",
    ]
    return all((root / p).is_file() for p in required)


def cascade_v8_models_ok(root: Path) -> bool:
    """Release v8.0.0 — MiniLM + DeBERTa Large + Reranker undetermined."""
    required = [
        "models/cascade_v8/config.json",
        "models/cascade_v8/models/minilm_full_v7/model.safetensors",
        "models/cascade_v8/models/deberta_large_v8.1/model.safetensors",
        "models/cascade_v8/models/reranker_undetermined_v8/model.safetensors",
    ]
    return all((root / p).is_file() for p in required)


def ml_models_ok(root: Path) -> bool:
    """Poids requis : Cascade V8 (labels) + SBERT (similarité)."""
    return cascade_v8_models_ok(root) and ml_models_sbert_ok(root)


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
        out["pytorch"] = (False, "Re-run ./start.sh")
    try:
        import sentence_transformers as st

        ver = st.__version__
        parts = [int(p) for p in ver.split(".")[:2]]
        ok = parts[0] > 5 or (parts[0] == 5 and parts[1] >= 5)
        out["sentence_transformers"] = (ok, ver)
    except ImportError:
        out["sentence_transformers"] = (False, "Re-run ./start.sh")
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
    models_base_ok: bool,
    models_v8_ok: bool,
    models_sbert_ok: bool,
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
    elif mem >= RAM_CASCADE_RECOMMENDED_GB:
        ram_ok = True
    elif mem >= RAM_QWEN_MIN_GB:
        ram_ok = None  # OK for Qwen only; tight for Cascade + Qwen
    else:
        ram_ok = False

    if mem <= 0:
        ram_detail = "Not detected"
    else:
        ram_detail = (
            f"{mem:.1f} GB total — Cascade OK (≥{RAM_CASCADE_RECOMMENDED_GB:.0f} GB)"
            if mem >= RAM_CASCADE_RECOMMENDED_GB
            else (
                f"{mem:.1f} GB total — Cascade prefers ≥{RAM_CASCADE_RECOMMENDED_GB:.0f} GB "
                f"(Qwen only OK from ≥{RAM_QWEN_MIN_GB:.0f} GB)"
                if mem >= RAM_QWEN_MIN_GB
                else (
                    f"{mem:.1f} GB total — below Qwen minimum "
                    f"(≥{RAM_QWEN_MIN_GB:.0f} GB; Cascade ≥{RAM_CASCADE_RECOMMENDED_GB:.0f} GB)"
                )
            )
        )

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
            "label": "System",
            "ok": True,
            "required": False,
            "detail": f"{os_label} {arch}",
        },
        {
            "id": "python",
            "label": "Python 3.9+",
            "ok": py_ok,
            "required": True,
            "detail": f"v{py_ver}" if py_ok else "Install Python 3.9+",
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
            "detail": f"v{st_ver}" if st_ok else f"v{st_ver} — incompatible models",
            "action": "pip install 'sentence-transformers>=5.5.1'" if not st_ok else None,
        },
        {
            "id": "ml_models_v8",
            "label": "Cascade V8 models (Release v8)",
            "ok": models_v8_ok,
            "required": True,
            "detail": (
                "MiniLM + DeBERTa Large + Reranker — Annotate Cascade & Compare (Qwen = last stage via Ollama)"
                if models_v8_ok
                else "Missing — run ./start.sh (downloads GitHub Release v8.0.0); Qwen still required via Ollama"
            ),
            "action": "python scripts/download_models_v8.py" if not models_v8_ok else None,
        },
        {
            "id": "ml_models_sbert",
            "label": "SBERT model v2 (Release v8)",
            "ok": models_sbert_ok,
            "required": True,
            "detail": (
                "Installed — fine-tuned SBERT v2 → similarity_annotation (cosine)"
                if models_sbert_ok
                else "Missing — ./start.sh downloads it with Cascade V8 (same Release v8.0.0)"
            ),
            "action": "./start.sh  (or: python scripts/download_models.py)" if not models_sbert_ok else None,
        },
        {
            "id": "ollama_installed",
            "label": "Ollama installed",
            "ok": bool(ollama.get("installed")),
            "required": True,
            "detail": "OK" if ollama.get("installed") else "Not installed",
            "action": OLLAMA_DOWNLOAD if not ollama.get("installed") else None,
        },
        {
            "id": "ollama_running",
            "label": "Ollama running (serve)",
            "ok": bool(ollama.get("running")),
            "required": True,
            "detail": "Service running" if ollama.get("running") else "ollama serve",
            "action": "ollama serve" if not ollama.get("running") else None,
        },
        {
            "id": "llm_qwen",
            "label": f"LLM {llm_tag}",
            "ok": llm_ok,
            "required": True,
            "detail": (
                "Downloaded — used by Qwen only, Cascade (last stage), and Compare"
                if llm_ok
                else "Pending or missing — required by all three Run Model modes"
            ),
            "action": f"ollama pull {llm_tag}" if not llm_ok else None,
        },
        {
            "id": "ram",
            "label": f"RAM ≥ {RAM_CASCADE_RECOMMENDED_GB:.0f} GB (Cascade)",
            "ok": ram_ok,
            "required": False,
            "detail": ram_detail,
        },
        {
            "id": "disk",
            "label": "Disk space",
            "ok": True if install_done else (disk_ok if fresh_gb else None),
            "required": False,
            "detail": (
                f"{disk_free} GB free — install complete"
                if install_done and disk_free is not None
                else (
                    f"{disk_free} GB free"
                    if disk_free is not None
                    else ""
                )
            ),
            "action": (
                None
                if install_done
                else (
                    f"~{fresh_gb} GB free required (venv + Cascade V8 + Qwen)"
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
            "label": "Estimated performance",
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
    models_base_ok = ml_models_base_ok(root)
    models_v8_ok = cascade_v8_models_ok(root)
    models_sbert_ok = ml_models_sbert_ok(root)
    models_ok = ml_models_ok(root)  # Cascade V8 + SBERT
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
            f"Python {py_ver} detected — Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ required. "
            f"Download: {PYTHON_DOWNLOAD}"
        )
    elif not py_recommended and not install_done:
        warnings.append(
            f"Python {py_ver} works, but Python 3.12+ is recommended for better performance."
        )

    if not install_done:
        if not ollama["installed"]:
            warnings.append(
                "Ollama is not installed — the LLM part of the cascade will fail. "
                f"Download: {OLLAMA_DOWNLOAD}"
            )
        elif not ollama["running"]:
            warnings.append(
                "Ollama is installed but the service is not responding yet. "
                "Auto-start in progress (ollama serve)…"
            )
        if not models_v8_ok:
            warnings.append(
                "Cascade V8 models missing (MiniLM / DeBERTa Large / Reranker). "
                "Run ./start.sh or: python scripts/download_models_v8.py (Release v8.0.0). "
                "Cascade V8 still needs Qwen via Ollama for the last stage."
            )
        if not models_sbert_ok:
            warnings.append(
                "SBERT missing (similarity scores). "
                "Run ./start.sh or: python scripts/download_models.py (Release v8.0.0)."
            )
        if ollama["installed"] and ollama["running"] and rec_llm and not llm_ok:
            active = cfg.get("llm", {}).get("ollama", rec_llm["ollama"])
            warnings.append(
                f"LLM « {active} » not downloaded yet — "
                "automatic download in progress (may take several minutes)."
            )

        if mem > 0 and mem < 6:
            warnings.append(
                f"Low RAM (~{mem:.1f} GB). Annotation may be very slow or fail."
            )
        elif 0 < mem < 10:
            warnings.append(
                f"Limited RAM (~{mem:.1f} GB). Qwen 7B may fail or be very slow."
            )

        if (
            disk_info.get("fresh_install_required_gb")
            and not disk_info.get("disk_ok")
        ):
            warnings.append(
                f"Insufficient disk space to finish installation: "
                f"{disk_info['disk_free_gb']} GB free, "
                f"~{disk_info['fresh_install_required_gb']} GB recommended."
            )

        if gpu["device"] == "cpu":
            warnings.append(
                "No GPU detected. DeBERTa and the LLM will run on CPU — "
                "automatic annotation will be significantly slower."
            )

        if rec_llm:
            notes.append(f"Recommended LLM for this machine: {rec_llm['label']} ({rec_llm['ollama']})")

        notes.append("Complete the required checklist before Run Model.")
    elif disk_info.get("disk_free_gb", 0) > 0 and disk_info.get("disk_free_gb", 0) < 0.3:
        warnings.append(
            f"Critical disk space: {disk_info['disk_free_gb']} GB free."
        )

    perf_labels = {
        "good": "Good — fast annotation expected",
        "acceptable": "Fair — Qwen 7B possible but slow",
        "slow": "Slow — CPU only, be patient",
        "insufficient": "Limited — you can still run Run Model",
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
        models_base_ok=models_base_ok,
        models_v8_ok=models_v8_ok,
        models_sbert_ok=models_sbert_ok,
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
        "ml_models_base": models_base_ok,
        "ml_models_v8": models_v8_ok,
        "ml_models_sbert": models_sbert_ok,
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
