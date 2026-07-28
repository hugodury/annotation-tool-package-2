#!/usr/bin/env python3
"""Estimation espace disque requis avant téléchargement des modèles."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

VENV_AND_DEPS_GB = 2.5
SAFETY_MARGIN_GB = 1.0


def disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return 0.0


def load_manifest(root: Path) -> dict:
    p = root / "models.manifest.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def ml_models_present(root: Path) -> bool:
    required = [
        "models/fine_tuned_deberta_base_expanded/model.safetensors",
        "models/fine_tuned_sbert/model.safetensors",
        "models/fine_tuned_cross_encoder/model.safetensors",
    ]
    return all((root / rel).is_file() for rel in required)


def cascade_v8_present(root: Path) -> bool:
    required = [
        "models/cascade_v8/config.json",
        "models/cascade_v8/models/minilm_full_v7/model.safetensors",
        "models/cascade_v8/models/deberta_large_v8.1/model.safetensors",
        "models/cascade_v8/models/reranker_undetermined_v8/model.safetensors",
    ]
    return all((root / rel).is_file() for rel in required)


def full_install_disk_gb(
    root: Path,
    cfg: dict | None = None,
    *,
    ram_gb_val: float = 0,
) -> tuple[float, dict[str, float]]:
    """Espace libre recommande pour une installation complete depuis zero (informatif)."""
    cfg = cfg or {}
    breakdown: dict[str, float] = {}

    # Release v1 (DeBERTa-base / SBERT) is unused by current Run Model modes — not counted.
    v8_manifest_path = root / "models-v8.manifest.json"
    if v8_manifest_path.is_file():
        v8 = json.loads(v8_manifest_path.read_text(encoding="utf-8"))
        breakdown["modeles_cascade_v8"] = float(v8.get("required_free_gb", 7.0))
    else:
        breakdown["modeles_cascade_v8"] = 7.0

    breakdown["venv_et_dependances"] = VENV_AND_DEPS_GB

    llm_profile = None
    profiles = cfg.get("llm_profiles", [])
    if profiles:
        ordered = sorted(profiles, key=lambda p: p.get("priority", 99))
        mem = ram_gb_val if ram_gb_val > 0 else 8.0
        for p in ordered:
            if mem >= p.get("min_ram_gb", 0) - 1:
                llm_profile = p
                break
        if llm_profile is None:
            llm_profile = ordered[-1]
    if llm_profile:
        breakdown["llm_ollama"] = float(llm_profile.get("min_disk_gb", 6))
    else:
        breakdown["llm_ollama"] = 6.0

    breakdown["marge_securite"] = SAFETY_MARGIN_GB
    return round(sum(breakdown.values()), 1), breakdown


def estimate_disk_need(
    root: Path,
    cfg: dict | None = None,
    *,
    install_complete: bool = False,
    ram_gb_val: float = 0,
) -> dict[str, Any]:
    cfg = cfg or {}
    free = round(disk_free_gb(root), 1)
    fresh_gb, fresh_breakdown = full_install_disk_gb(root, cfg, ram_gb_val=ram_gb_val)

    if install_complete:
        critical = free > 0 and free < 0.3
        return {
            "disk_free_gb": free,
            "fresh_install_required_gb": fresh_gb,
            "fresh_install_breakdown_gb": fresh_breakdown,
            "disk_required_gb": None,
            "disk_ok": not critical,
            "disk_breakdown_gb": {},
            "install_complete": True,
        }

    return {
        "disk_free_gb": free,
        "fresh_install_required_gb": fresh_gb,
        "fresh_install_breakdown_gb": fresh_breakdown,
        "disk_required_gb": fresh_gb,
        "disk_ok": free >= fresh_gb if free > 0 else True,
        "disk_breakdown_gb": fresh_breakdown,
        "install_complete": False,
    }


def require_disk_for_models(root: Path) -> None:
    if ml_models_present(root):
        return
    manifest = load_manifest(root)
    need = float(manifest.get("archive", {}).get("required_free_gb", 3.5))
    free = disk_free_gb(root)
    if free > 0 and free < need:
        raise RuntimeError(
            f"Espace disque insuffisant pour télécharger les modèles ML.\n"
            f"  Libre : {free:.1f} Go\n"
            f"  Requis : {need:.1f} Go (archive + extraction)\n"
            f"  Libérez de l'espace disque puis relancez ./start.sh"
        )
    if free > 0:
        print(f"  Espace disque OK pour modèles ML : {free:.1f} Go libres (besoin ~{need:.1f} Go)")
