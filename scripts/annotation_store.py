"""Verrous fichier JSON, sauvegardes et validation des plages d'index."""
from __future__ import annotations

import shutil
import threading
from datetime import datetime
from pathlib import Path

_file_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def file_lock_for(path: Path | str) -> threading.Lock:
    key = str(Path(path).resolve())
    with _guard:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


def backup_json_file(file_path: Path) -> Path | None:
    if not file_path.is_file():
        return None
    backup_dir = file_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{file_path.stem}_{ts}{file_path.suffix}"
    shutil.copy2(file_path, dest)
    return dest


def parse_index_range(req: dict, data_len: int) -> tuple[int, int]:
    if req.get("start_index") is None or req.get("end_index") is None:
        raise ValueError("start_index et end_index sont requis.")
    try:
        start = int(req["start_index"])
        end = int(req["end_index"])
    except (TypeError, ValueError) as e:
        raise ValueError("Indices invalides (nombre entier attendu).") from e
    if start < 0 or end < 0 or start > end or end >= data_len:
        raise ValueError(f"Plage d'index invalide (0–{data_len - 1}).")
    return start, end
