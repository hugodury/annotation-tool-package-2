"""Utilitaires de stockage pour les fichiers JSON annotés.

Fournit :
- Verrous par fichier (thread-safe) pour éviter les écritures concurrentes.
- Sauvegarde horodatée (backups/) avant chaque écrasement.
- Validation des plages d'indices pour les requêtes Run Model.
"""
from __future__ import annotations

import shutil
import threading
from datetime import datetime
from pathlib import Path

_file_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def file_lock_for(path: Path | str) -> threading.Lock:
    """Retourne un verrou threading unique par chemin de fichier (résolu).

    Les verrous sont créés à la demande et réutilisés pour les appels
    ultérieurs au même fichier. Thread-safe via un méta-verrou interne.

    Args:
        path: Chemin du fichier (absolu ou relatif).

    Returns:
        Le verrou ``threading.Lock`` associé à ce fichier.
    """
    key = str(Path(path).resolve())
    with _guard:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


def backup_json_file(file_path: Path) -> Path | None:
    """Copie horodatée d'un fichier JSON dans le sous-dossier ``backups/``.

    Ne fait rien si le fichier source n'existe pas.

    Args:
        file_path: Chemin du fichier JSON à sauvegarder.

    Returns:
        Le chemin de la copie créée, ou ``None`` si le source est absent.
    """
    if not file_path.is_file():
        return None
    backup_dir = file_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{file_path.stem}_{ts}{file_path.suffix}"
    shutil.copy2(file_path, dest)
    return dest


def parse_index_range(req: dict, data_len: int) -> tuple[int, int]:
    """Valide et extrait la plage [start_index, end_index] d'une requête.

    Les indices sont 0-based et inclusifs des deux côtés.

    Args:
        req: Dictionnaire de requête contenant ``start_index`` et ``end_index``.
        data_len: Longueur totale du dataset (borne supérieure exclusive).

    Returns:
        Tuple ``(start, end)`` validé.

    Raises:
        ValueError: Si l'un des indices est absent, non entier, ou hors bornes.
    """
    if req.get("start_index") is None or req.get("end_index") is None:
        raise ValueError("start_index and end_index are required.")
    try:
        start = int(req["start_index"])
        end = int(req["end_index"])
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid indices (integer expected).") from e
    if start < 0 or end < 0 or start > end or end >= data_len:
        raise ValueError(f"Invalid index range (0–{data_len - 1}).")
    return start, end
