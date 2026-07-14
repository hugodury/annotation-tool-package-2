"""Discover annotation JSON sessions across the local machine."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

CACHE_TTL_SEC = 180
MAX_DEPTH = 14
MAX_SESSIONS = 500

SKIP_DIR_NAMES = frozenset({
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".cache",
    "site-packages",
    "models",
    ".cursor",
    "Library",
    "AppData",
    "backups",
    "logs",
    "instance",
})


def _scan_roots() -> list[Path]:
    roots = [Path.home().resolve()]
    system = platform.system()
    if system == "Linux":
        for base in (Path("/media"), Path("/mnt")):
            if not base.is_dir():
                continue
            try:
                for child in base.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        roots.append(child.resolve())
            except OSError:
                continue
    elif system == "Darwin":
        volumes = Path("/Volumes")
        if volumes.is_dir():
            try:
                for child in volumes.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        roots.append(child.resolve())
            except OSError:
                pass
    elif system == "Windows":
        try:
            import string
            from ctypes import windll

            bitmask = windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    drive = Path(f"{letter}:/")
                    if drive.is_dir():
                        roots.append(drive.resolve())
                bitmask >>= 1
        except OSError:
            pass
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _looks_like_annotation_json(path: Path) -> bool:
    try:
        if path.stat().st_size > 200 * 1024 * 1024:
            return False
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data:
            return False
        item = data[0]
        return isinstance(item, dict) and "news_id" in item and "database" in item
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _session_entry(path: Path) -> dict | None:
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        return None
    if not _looks_like_annotation_json(resolved):
        return None
    return {
        "filename": resolved.name,
        "path": str(resolved),
        "parent": str(resolved.parent),
        "mtime": stat.st_mtime,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }


def _add_session(found: dict[str, dict], path: Path) -> None:
    if len(found) >= MAX_SESSIONS:
        return
    item = _session_entry(path)
    if item:
        found[item["path"]] = item


def _shallow_scan_folder(folder: Path, found: dict[str, dict]) -> None:
    if not folder.is_dir():
        return
    try:
        for entry in folder.iterdir():
            if len(found) >= MAX_SESSIONS:
                return
            if entry.is_file() and entry.suffix.lower() == ".json":
                _add_session(found, entry)
    except OSError:
        return


def _walk_json_files(root: Path, found: dict[str, dict], depth: int) -> None:
    if depth > MAX_DEPTH or len(found) >= MAX_SESSIONS:
        return
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        if len(found) >= MAX_SESSIONS:
            return
        try:
            if entry.is_dir():
                if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                    continue
                _walk_json_files(entry, found, depth + 1)
            elif entry.is_file() and entry.suffix.lower() == ".json":
                _add_session(found, entry)
        except OSError:
            continue


def _full_computer_scan() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for root in _scan_roots():
        if root.is_file() and root.suffix.lower() == ".json":
            _add_session(found, root)
        elif root.is_dir():
            _walk_json_files(root, found, 0)
    return found


def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sessions"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _save_cache(cache_path: Path, sessions: list[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cached_at": time.time(), "sessions": sessions[:MAX_SESSIONS]}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def invalidate_sessions_cache(cache_path: Path) -> None:
    try:
        if cache_path.is_file():
            cache_path.unlink()
    except OSError:
        pass


def discover_sessions(
    *,
    extra_paths: list[str] | None = None,
    storage_folders: list[str] | None = None,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Merge storage-folder scans, known paths, and cached/full computer scan."""
    found: dict[str, dict] = {}

    for raw in storage_folders or []:
        _shallow_scan_folder(Path(raw).expanduser(), found)

    for raw in extra_paths or []:
        if len(found) >= MAX_SESSIONS:
            break
        path = Path(raw).expanduser()
        if path.is_file():
            _add_session(found, path)

    cache_file = cache_path
    cache = None if force_refresh else (_load_cache(cache_file) if cache_file else None)
    cache_fresh = (
        cache is not None
        and (time.time() - float(cache.get("cached_at", 0))) < CACHE_TTL_SEC
    )

    if cache_fresh:
        for session in cache["sessions"]:
            if isinstance(session, dict) and session.get("path"):
                found[session["path"]] = session
    else:
        found.update(_full_computer_scan())
        if cache_file:
            _save_cache(cache_file, list(found.values()))

    sessions = list(found.values())
    sessions.sort(key=lambda s: s.get("mtime", 0), reverse=True)
    return sessions[:MAX_SESSIONS]
