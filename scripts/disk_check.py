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


def estimate_disk_need(
    root: Path,
    cfg: dict | None = None,
    *,
    models_ok: bool | None = None,
    llm_profile: dict | None = None,
    ram_gb_val: float = 0,
) -> dict[str, Any]:
    cfg = cfg or {}
    manifest = load_manifest(root)
    models_ok = ml_models_present(root) if models_ok is None else models_ok

    breakdown: dict[str, float] = {}
    if not models_ok:
        archive_gb = manifest.get("archive", {}).get("required_free_gb")
        if archive_gb is None:
            size = manifest.get("archive", {}).get("size_bytes", 1_500_000_000)
            archive_gb = round(size / (1024**3) * 2 + 0.5, 1)
        breakdown["modeles_ml"] = float(archive_gb)

    breakdown["venv_et_dependances"] = VENV_AND_DEPS_GB

    if llm_profile is None and cfg.get("llm_profiles"):
        profiles = sorted(cfg["llm_profiles"], key=lambda p: p.get("priority", 99))
        mem = ram_gb_val if ram_gb_val > 0 else 8.0
        for p in profiles:
            if mem >= p.get("min_ram_gb", 0) - 1:
                llm_profile = p
                break
        if llm_profile is None and profiles:
            llm_profile = profiles[-1]
    if llm_profile:
        breakdown["llm_ollama"] = float(llm_profile.get("min_disk_gb", 3))

    breakdown["marge_securite"] = SAFETY_MARGIN_GB
    total = round(sum(breakdown.values()), 1)
    free = round(disk_free_gb(root), 1)

    return {
        "disk_free_gb": free,
        "disk_required_gb": total,
        "disk_ok": free >= total if free > 0 else True,
        "disk_breakdown_gb": breakdown,
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
