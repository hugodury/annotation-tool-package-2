"""Run Model background job: progress, logs, incremental save."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import statistics
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

SESSION_LOG_FILENAME = "run_model_session.log"

_lock = threading.Lock()
_cancel_requested = False
_session_log_path: Path | None = None
_state: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "progress": {},
    "result": None,
    "error": None,
}


def _format_elapsed(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{m} min {rem} s" if rem else f"{m} min"
    h, rem_m = divmod(m, 60)
    tail = f" {rem_m} min" if rem_m else ""
    return f"{h} h{tail}" + (f" {rem} s" if rem else "")


def get_job_status() -> dict[str, Any]:
    with _lock:
        out = {
            "running": _state["running"],
            "started_at": _state["started_at"],
            "progress": deepcopy(_state.get("progress") or {}),
            "result": deepcopy(_state.get("result")),
            "error": _state.get("error"),
            "cancel_requested": _cancel_requested,
        }
    return out


def _set_progress(**kwargs: Any) -> None:
    with _lock:
        _state["progress"].update(kwargs)


def _finish(result: dict | None = None, error: str | None = None) -> None:
    global _cancel_requested
    with _lock:
        _state["running"] = False
        _state["result"] = result
        _state["error"] = error
        _cancel_requested = False


def prepare_session_logs(logs_dir: Path) -> Path:
    """Delete previous Run Model logs and open a fresh log for this server session."""
    global _session_log_path
    logs_dir.mkdir(parents=True, exist_ok=True)
    for old in logs_dir.glob("run_model_*.log"):
        try:
            old.unlink()
        except OSError:
            pass
    _session_log_path = logs_dir / SESSION_LOG_FILENAME
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = os.environ.get("FLASK_PORT", "5000")
    header = "\n".join(
        [
            "=" * 72,
            f"SESSION START  {datetime.now().isoformat(timespec='seconds')}",
            f"PID            {os.getpid()}",
            f"Platform       {platform.system()} {platform.release()} ({platform.machine()})",
            f"Python         {sys.version.split()[0]}",
            f"App URL        http://{host}:{port}",
            f"Log file       {SESSION_LOG_FILENAME}",
            "=" * 72,
            "",
        ]
    )
    _session_log_path.write_text(header, encoding="utf-8")
    return _session_log_path


def get_session_log_path() -> Path | None:
    return _session_log_path


def _setup_logger(logs_dir: Path) -> logging.Logger:
    global _session_log_path
    if _session_log_path is None:
        prepare_session_logs(logs_dir)
    log_path = _session_log_path
    logger = logging.getLogger("run_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    if not logger.handlers or not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        pass  # file only
    return logger


def effective_run_config(cfg: dict, is_cpu: bool) -> dict:
    cfg = deepcopy(cfg)
    run_cfg = cfg.setdefault("run_model", {})
    cpu_slow = run_cfg.get("cpu_slow") or {}
    if is_cpu and cpu_slow.get("enabled", True):
        timeout = int(cpu_slow.get("inference_timeout_sec", 900))
        cfg.setdefault("inference", {})["timeout"] = timeout
    return cfg


KNOWN_ROUTES = (
    "llm_auto",
    "deberta_auto",
    "deberta_ambiguous",
    "consensus",
    "compare_agree",
    "compare_disagree",
    "v8_duo",
    "v8_reranker",
    "v8_qwen",
    "human",
    "rejected",
)

# Fractions theoriques par mode (estimation avant calibration fichier).
DEFAULT_ROUTE_FRACTIONS: dict[str, float] = {
    "llm_auto": 0.9,
    "deberta_auto": 0.0,
    "deberta_ambiguous": 0.0,
    "consensus": 0.0,
    "human": 0.1,
    "rejected": 0.0,
}
CASCADE_ROUTE_FRACTIONS: dict[str, float] = {
    "deberta_auto": 0.75,
    "deberta_ambiguous": 0.12,
    "consensus": 0.05,
    "human": 0.06,
    "rejected": 0.02,
    "llm_auto": 0.0,
}

_session_models_warm = False
_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_TARGET_DONE = re.compile(
    r"TARGET DONE ref_index=\d+ ref=\d+/\d+ target=\d+/\d+ route=(\w+) duration=([\d.]+)s"
)
_BATCH_START_FILE = re.compile(r"BATCH START file=(.+?) indices=")
_BATCH_START_MODE = re.compile(
    r"BATCH START file=(.+?) indices=.*?cascade_mode=(\w+)"
)
LLM_ROUTES = ("llm_auto", "consensus", "v8_qwen", "human", "rejected")
DEBERTA_ROUTES = ("deberta_auto", "deberta_ambiguous", "v8_duo", "v8_reranker")
ESTIMATE_SAMPLE_MAX = 100
ESTIMATE_MIN_SAMPLES = 1
ESTIMATE_HINT_FIRST_RUN = (
    "No time estimate yet — finish a first Run Model with this mode "
    "(Qwen only or Cascade) so timings can be calibrated from a real sample."
)
ESTIMATE_HINT_COMPARE = (
    "Compare has no single-time estimate (runs Qwen only + Cascade). "
    "Estimate each mode separately after a first calibrated run."
)
CALIBRATION_VERSION = 2
TPR_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)
REFS_BUCKETS = (10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000)


def models_warm_in_session() -> bool:
    return _session_models_warm


def _bucket_nearest(value: float, buckets: tuple[int, ...]) -> int:
    if value <= buckets[0]:
        return buckets[0]
    best = buckets[0]
    best_dist = abs(value - best)
    for b in buckets[1:]:
        dist = abs(value - b)
        if dist < best_dist:
            best, best_dist = b, dist
    return best


def compute_file_profile(data: list) -> dict[str, Any]:
    """Profil structurel d'un JSON VLDBench (taille, densite de cibles)."""
    refs = len(data)
    tprs = [len(item.get("database") or []) for item in data]
    targets_total = sum(tprs)
    tpr_median = float(statistics.median(tprs)) if tprs else 0.0
    tpr_bucket = _bucket_nearest(tpr_median, TPR_BUCKETS)
    refs_bucket = _bucket_nearest(float(refs), REFS_BUCKETS)
    profile_key = f"tpr{tpr_bucket}_r{refs_bucket}"
    return {
        "refs": refs,
        "refs_bucket": refs_bucket,
        "targets_per_ref_median": round(tpr_median, 2),
        "targets_per_ref_bucket": tpr_bucket,
        "targets_total": targets_total,
        "profile_key": profile_key,
    }


def _calibration_path(base_dir: Path | None) -> Path | None:
    if not base_dir:
        return None
    return base_dir / "instance" / "estimate_calibration.json"


def _empty_calibration_store(device_type: str) -> dict[str, Any]:
    return {
        "version": CALIBRATION_VERSION,
        "device": device_type,
        "device_global": {"startup_sec": None, "routes": {}},
        "by_file": {},
        "by_profile": {},
    }


def _migrate_calibration_v1(data: dict[str, Any], device_type: str) -> dict[str, Any]:
    store = _empty_calibration_store(device_type)
    if data.get("startup_sec"):
        store["device_global"]["startup_sec"] = data["startup_sec"]
    if data.get("routes"):
        store["device_global"]["routes"] = dict(data["routes"])
    return store


def _load_calibration_store(base_dir: Path | None, device_type: str) -> dict[str, Any]:
    path = _calibration_path(base_dir)
    if not path or not path.is_file():
        return _empty_calibration_store(device_type)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_calibration_store(device_type)
        if data.get("device") and data.get("device") != device_type:
            return _empty_calibration_store(device_type)
        if int(data.get("version", 1)) < CALIBRATION_VERSION:
            return _migrate_calibration_v1(data, device_type)
        return data
    except (OSError, json.JSONDecodeError):
        return _empty_calibration_store(device_type)


def _save_calibration_store(base_dir: Path | None, store: dict[str, Any]) -> None:
    path = _calibration_path(base_dir)
    if not path:
        return
    store["updated_at"] = datetime.utcnow().isoformat() + "Z"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def _merge_route_entry(
    prev_route: dict[str, Any] | None,
    new_vals: list[float],
    *,
    max_n: int = 500,
) -> dict[str, float | int]:
    prev_route = prev_route or {}
    prev_n = int(prev_route.get("n", 0))
    prev_mean = float(prev_route.get("mean", new_vals[0]))
    merged: list[float] = []
    if prev_n > 0:
        merged.extend([prev_mean] * min(prev_n, 40))
    merged.extend(new_vals[-80:])
    return {
        "n": min(prev_n + len(new_vals), max_n),
        "mean": round(statistics.mean(merged), 3),
        "p50": round(statistics.median(merged), 3),
    }


def _blend_startup(prev: float | None, measured: float | None) -> float | None:
    if measured is None:
        return prev
    if prev is None:
        return round(measured, 2)
    return round(prev * 0.35 + measured * 0.65, 2)


def _blend_sec_per_target(prev: float | None, measured: float | None) -> float | None:
    if measured is None or measured <= 0:
        return prev
    if prev is None:
        return round(measured, 3)
    return round(prev * 0.25 + measured * 0.75, 3)


def _parse_log_timestamp(line: str) -> float | None:
    m = _LOG_TS.match(line)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        return dt.timestamp() + int(m.group(2)) / 1000.0
    except ValueError:
        return None


def _parse_session_log_stats(
    log_path: Path,
    *,
    filename: str | None = None,
) -> tuple[dict[str, list[float]], list[float]]:
    route_buckets: dict[str, list[float]] = {r: [] for r in KNOWN_ROUTES}
    startup_delays: list[float] = []
    if not log_path.is_file():
        return route_buckets, startup_delays
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return route_buckets, startup_delays
    active_file: str | None = None
    batch_start_ts: float | None = None
    seen_target_in_batch = False
    for line in lines:
        if "BATCH START" in line:
            m = _BATCH_START_FILE.search(line)
            active_file = m.group(1) if m else None
            if filename is None or active_file == filename:
                batch_start_ts = _parse_log_timestamp(line)
                seen_target_in_batch = False
            else:
                batch_start_ts = None
                seen_target_in_batch = False
            continue
        if "BATCH END" in line:
            active_file = None
            batch_start_ts = None
            seen_target_in_batch = False
            continue
        if filename and active_file != filename:
            continue
        m = _TARGET_DONE.search(line)
        if not m or batch_start_ts is None:
            continue
        ts = _parse_log_timestamp(line)
        route, raw = m.group(1), float(m.group(2))
        if route in route_buckets and 0.02 <= raw <= 900:
            route_buckets[route].append(raw)
            if not seen_target_in_batch and ts is not None:
                startup_delays.append(max(0.0, ts - batch_start_ts))
                seen_target_in_batch = True
    return route_buckets, startup_delays


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def refresh_estimate_calibration(
    *,
    base_dir: Path | None,
    logs_dir: Path | None,
    device_type: str,
    filename: str,
    file_profile: dict[str, Any],
    routing_stats: dict[str, int] | None = None,
    batch_elapsed_sec: float | None = None,
    targets_processed: int = 0,
    cpu_slow: dict | None = None,
    est_cfg: dict | None = None,
    cascade_mode: str = "qwen_only",
) -> None:
    if not base_dir or not logs_dir or not filename:
        return
    from cascade.core import CASCADE_MODE_COMPARE, normalize_cascade_mode

    mode = normalize_cascade_mode(cascade_mode)
    if mode == CASCADE_MODE_COMPARE:
        # Compare = deux pipelines ; on ne calibrait pas un temps unique.
        return
    session = logs_dir / SESSION_LOG_FILENAME
    route_buckets, startup_delays = _parse_session_log_stats(session, filename=filename)
    if not any(route_buckets.values()) and not routing_stats:
        return

    store = _load_calibration_store(base_dir, device_type)
    profile_key = file_profile.get("profile_key", "")
    device_global = store.setdefault("device_global", {"startup_sec": None, "routes": {}, "by_mode": {}})
    by_file = store.setdefault("by_file", {})
    by_profile = store.setdefault("by_profile", {})

    file_entry = dict(by_file.get(filename) or {})
    file_entry["profile_key"] = profile_key
    file_entry["profile"] = file_profile
    file_entry["n_batches"] = int(file_entry.get("n_batches", 0)) + 1

    profile_entry = dict(by_profile.get(profile_key) or {})
    profile_entry["profile"] = file_profile
    profile_entry["n_batches"] = int(profile_entry.get("n_batches", 0)) + 1
    files_seen = set(profile_entry.get("files_seen") or [])
    files_seen.add(filename)
    profile_entry["files_seen"] = sorted(files_seen)
    profile_entry["file_count"] = len(files_seen)

    startup = _median_or_none(startup_delays)
    device_global["startup_sec"] = _blend_startup(device_global.get("startup_sec"), startup)
    file_entry["startup_sec"] = _blend_startup(file_entry.get("startup_sec"), startup)
    profile_entry["startup_sec"] = _blend_startup(profile_entry.get("startup_sec"), startup)

    if routing_stats:
        total_routes = sum(int(routing_stats.get(r, 0)) for r in KNOWN_ROUTES)
        if total_routes > 0:
            fracs = _normalize_fractions(
                {r: float(routing_stats.get(r, 0)) for r in KNOWN_ROUTES}
            )
            file_entry["route_fractions"] = fracs
            file_entry["llm_fraction_observed"] = round(_llm_fraction(fracs), 4)
            prev_pf = profile_entry.get("route_fractions")
            if prev_pf:
                blended = {
                    r: round(prev_pf.get(r, fracs[r]) * 0.3 + fracs[r] * 0.7, 4)
                    for r in KNOWN_ROUTES
                }
                profile_entry["route_fractions"] = _normalize_fractions(blended)
            else:
                profile_entry["route_fractions"] = fracs
            profile_entry["llm_fraction_observed"] = round(
                _llm_fraction(profile_entry["route_fractions"]), 4
            )

    weighted_from_logs = _weighted_sec_per_from_routes(route_buckets)
    if batch_elapsed_sec and targets_processed > 0:
        startup_used = float(file_entry.get("startup_sec") or device_global.get("startup_sec") or 0)
        measured_spt = max(0.05, (batch_elapsed_sec - startup_used) / targets_processed)
        llm_obs = float(file_entry.get("llm_fraction_observed", 1.0))
        if llm_obs < 0.02 and weighted_from_logs is not None:
            measured_spt = weighted_from_logs
        elif llm_obs < 0.02:
            measured_spt = min(measured_spt, 0.45)
        file_entry["sec_per_target"] = _blend_sec_per_target(
            file_entry.get("sec_per_target"), measured_spt
        )
        profile_entry["sec_per_target"] = _blend_sec_per_target(
            profile_entry.get("sec_per_target"), measured_spt
        )
    elif weighted_from_logs is not None:
        file_entry["sec_per_target"] = _blend_sec_per_target(
            file_entry.get("sec_per_target"), weighted_from_logs
        )

    for layer_key, layer in (
        ("device_global", device_global),
        ("file", file_entry),
        ("profile", profile_entry),
    ):
        routes_out: dict[str, dict[str, float | int]] = dict(layer.get("routes") or {})
        for route, vals in route_buckets.items():
            if not vals:
                continue
            routes_out[route] = _merge_route_entry(routes_out.get(route), vals)
        layer["routes"] = routes_out
        if layer_key == "file":
            by_file[filename] = file_entry
        elif layer_key == "profile" and profile_key:
            by_profile[profile_key] = profile_entry

    snapshot = _compute_two_bucket_snapshot(
        route_buckets,
        routing_stats,
        device_routes=device_global.get("routes") or {},
        device_type=device_type,
        cpu_slow=cpu_slow or {},
        est_cfg=est_cfg or {},
    )
    if snapshot["sample_size"] > 0:
        file_entry["two_bucket"] = _merge_two_bucket(file_entry.get("two_bucket"), snapshot)
        profile_entry["two_bucket"] = _merge_two_bucket(
            profile_entry.get("two_bucket"), snapshot
        )
        by_file[filename] = file_entry
        if profile_key:
            by_profile[profile_key] = profile_entry

    # Calibration par mode (Qwen only vs Cascade) — obligatoire pour estimer.
    for layer in (file_entry, profile_entry, device_global):
        by_mode = dict(layer.get("by_mode") or {})
        slot = dict(by_mode.get(mode) or {})
        slot["n_batches"] = int(slot.get("n_batches", 0)) + 1
        if snapshot["sample_size"] > 0:
            slot["two_bucket"] = _merge_two_bucket(slot.get("two_bucket"), snapshot)
            slot["sample_size"] = int((slot.get("two_bucket") or {}).get("sample_size", 0))
        if routing_stats:
            total_routes = sum(int(routing_stats.get(r, 0)) for r in KNOWN_ROUTES)
            if total_routes > 0:
                slot["route_fractions"] = _normalize_fractions(
                    {r: float(routing_stats.get(r, 0)) for r in KNOWN_ROUTES}
                )
                slot["llm_fraction_observed"] = round(
                    _llm_fraction(slot["route_fractions"]), 4
                )
        if file_entry.get("sec_per_target") is not None and layer is file_entry:
            slot["sec_per_target"] = file_entry.get("sec_per_target")
        by_mode[mode] = slot
        layer["by_mode"] = by_mode

    by_file[filename] = file_entry
    if profile_key:
        by_profile[profile_key] = profile_entry
    store["device_global"] = device_global

    _save_calibration_store(base_dir, store)


def _mode_calibration_slot(
    parent: dict[str, Any] | None, cascade_mode: str
) -> dict[str, Any]:
    if not parent:
        return {}
    return dict((parent.get("by_mode") or {}).get(cascade_mode) or {})


def _has_recorded_estimate_timing(
    *,
    device_cal: dict[str, Any] | None,
    file_cal: dict[str, Any] | None,
    logs_dir: Path | None,
    filename: str | None,
    cascade_mode: str = "qwen_only",
) -> bool:
    """True seulement si ce mode a déjà un échantillon chronométré.

    Premier run d'un mode → pas d'estimation théorique.
    Après un run Qwen only → estimation Qwen only.
    Après un run Cascade → estimation Cascade (indépendant).
    """
    from cascade.core import CASCADE_MODE_COMPARE, normalize_cascade_mode

    mode = normalize_cascade_mode(cascade_mode)
    if mode == CASCADE_MODE_COMPARE:
        return False

    for parent in (file_cal, device_cal):
        slot = _mode_calibration_slot(parent, mode)
        tb = slot.get("two_bucket") or {}
        if int(tb.get("sample_size", 0)) >= ESTIMATE_MIN_SAMPLES:
            return True
        if int(slot.get("sample_size", 0)) >= ESTIMATE_MIN_SAMPLES:
            return True
        if int(slot.get("n_batches", 0)) >= 1:
            return True

    return _session_has_mode_prior(logs_dir, filename, mode)


def _session_has_mode_prior(
    logs_dir: Path | None, filename: str | None, cascade_mode: str
) -> bool:
    """True si la session courante a déjà chronométré ce mode (même après reboot partiel)."""
    if not logs_dir:
        return False
    session = logs_dir / SESSION_LOG_FILENAME
    if not session.is_file():
        return False
    try:
        text = session.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    found_mode_batch = False
    for match in _BATCH_START_MODE.finditer(text):
        file_name = match.group(1).strip()
        mode = match.group(2).strip().lower()
        if mode != cascade_mode:
            continue
        if filename and file_name != filename:
            continue
        found_mode_batch = True
        break
    if not found_mode_batch:
        return False
    # Au moins une cible terminée dans la session pour ce fichier / mode
    if filename:
        route_buckets, _ = _parse_session_log_stats(session, filename=filename)
        return any(route_buckets.values())
    return "TARGET DONE" in text


def _session_has_prior_targets(logs_dir: Path | None, filename: str | None = None) -> bool:
    if not logs_dir:
        return False
    session = logs_dir / SESSION_LOG_FILENAME
    if not session.is_file():
        return False
    try:
        text = session.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if filename:
        if not _BATCH_START_FILE.search(text) or not any(
            m.group(1) == filename for m in _BATCH_START_FILE.finditer(text)
        ):
            return False
        route_buckets, _ = _parse_session_log_stats(session, filename=filename)
        return any(route_buckets.values())
    return "TARGET DONE" in text


def _device_label(device_type: str) -> str:
    return {"cuda": "GPU CUDA", "mps": "GPU Apple", "cpu": "CPU"}.get(device_type, device_type.upper())


def _normalize_fractions(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.get(r, 0.0) for r in KNOWN_ROUTES)
    if total <= 0:
        return dict(DEFAULT_ROUTE_FRACTIONS)
    return {r: raw.get(r, 0.0) / total for r in KNOWN_ROUTES}


def _route_fractions_from_data(
    data: list,
    *,
    target_is_annotated_fn: Callable[[dict], bool] | None = None,
    min_samples: int = 12,
) -> dict[str, float] | None:
    counts: dict[str, int] = {r: 0 for r in KNOWN_ROUTES}
    total = 0
    for item in data:
        for target in item.get("database") or []:
            route = target.get("cascade_route")
            if route not in counts:
                continue
            if target_is_annotated_fn and not target_is_annotated_fn(target):
                continue
            counts[route] += 1
            total += 1
    if total < min_samples:
        return None
    return _normalize_fractions({r: float(counts[r]) for r in KNOWN_ROUTES})


def _route_fractions_from_logs(
    logs_dir: Path,
    *,
    filename: str | None = None,
    min_samples: int = 8,
) -> dict[str, float] | None:
    if not logs_dir.is_dir():
        return None
    pattern = re.compile(r"route=(\w+)")
    counts: dict[str, int] = {r: 0 for r in KNOWN_ROUTES}
    total = 0
    log_files = sorted(
        logs_dir.glob("run_model_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        return None
    session = logs_dir / SESSION_LOG_FILENAME
    if session in log_files:
        log_files = [session] + [p for p in log_files if p != session]
    active_file: str | None = None
    for path in log_files[:2]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines[-2500:]:
            if "BATCH START" in line:
                m = _BATCH_START_FILE.search(line)
                active_file = m.group(1) if m else None
                continue
            if "BATCH END" in line:
                active_file = None
                continue
            if filename and active_file != filename:
                continue
            m = pattern.search(line)
            if not m or m.group(1) not in counts:
                continue
            counts[m.group(1)] += 1
            total += 1
    if total < min_samples:
        return None
    return _normalize_fractions({r: float(counts[r]) for r in KNOWN_ROUTES})


def _default_sec_per_route(device_type: str, est_cfg: dict) -> dict[str, float]:
    is_cpu = device_type == "cpu"
    key = "route_sec_cpu" if is_cpu else "route_sec_gpu"
    cfg_map = est_cfg.get(key) or {}
    fallback_cpu = {
        "llm_auto": 14.0,
        "deberta_auto": 0.30,
        "deberta_ambiguous": 0.35,
        "consensus": 16.0,
        "compare_agree": 14.5,
        "compare_disagree": 14.5,
        "human": 14.0,
        "rejected": 15.0,
    }
    fallback_gpu = {
        "llm_auto": 7.0,
        "deberta_auto": 0.20,
        "deberta_ambiguous": 0.25,
        "consensus": 8.0,
        "compare_agree": 7.5,
        "compare_disagree": 7.5,
        "human": 7.0,
        "rejected": 8.0,
    }
    base = fallback_cpu if is_cpu else fallback_gpu
    if device_type == "mps":
        base = {r: v * 0.85 for r, v in fallback_gpu.items()}
    return {
        r: float(cfg_map.get(r, base.get(r, 10.0 if is_cpu else 5.0)))
        for r in KNOWN_ROUTES
    }


def _llm_fraction(fracs: dict[str, float]) -> float:
    return sum(float(fracs.get(r, 0.0)) for r in LLM_ROUTES)


def _config_llm_fraction(est_cfg: dict) -> float:
    cfg_frac = est_cfg.get("route_fractions") or DEFAULT_ROUTE_FRACTIONS
    return _llm_fraction(cfg_frac)


def _weighted_sec_per_from_routes(route_buckets: dict[str, list[float]]) -> float | None:
    total = sum(len(v) for v in route_buckets.values())
    if total <= 0:
        return None
    return round(
        sum(statistics.median(v) * len(v) for v in route_buckets.values() if v) / total,
        3,
    )


def _default_bucket_secs(
    device_type: str,
    cpu_slow: dict,
    est_cfg: dict,
    device_routes: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Retourne (sec_deberta, sec_llm) par defaut ou depuis calibration machine.

    Les p50 par route sont ponderes par le nombre d'echantillons (n) afin
    qu'une route rare (ex. un unique timeout aberrant) ne domine pas la
    moyenne LLM."""
    routes = device_routes or {}

    def _weighted_p50(route_names: tuple[str, ...]) -> float | None:
        num = 0.0
        den = 0.0
        for r in route_names:
            ent = routes.get(r)
            if not ent or not ent.get("p50"):
                continue
            n = max(1, int(ent.get("n", 1)))
            num += float(ent["p50"]) * n
            den += n
        return num / den if den else None

    deberta_secs = _weighted_p50(DEBERTA_ROUTES)
    llm_secs = _weighted_p50(LLM_ROUTES)
    if deberta_secs is not None:
        sec_deberta = deberta_secs
    elif device_type == "cpu" and cpu_slow:
        sec_deberta = float(cpu_slow.get("sec_deberta", 1.0))
    else:
        sec_deberta = float(
            (_default_sec_per_route(device_type, est_cfg).get("deberta_auto", 0.25))
        )
    if llm_secs is not None:
        sec_llm = llm_secs
    elif device_type == "cpu" and cpu_slow:
        sec_llm = float(cpu_slow.get("sec_llm", 45.0))
    else:
        base = _default_sec_per_route(device_type, est_cfg)
        sec_llm = statistics.mean([base[r] for r in LLM_ROUTES])
    return round(sec_deberta, 3), round(sec_llm, 2)


def _compute_two_bucket_snapshot(
    route_buckets: dict[str, list[float]],
    routing_stats: dict[str, int] | None,
    *,
    device_routes: dict[str, Any],
    device_type: str,
    cpu_slow: dict,
    est_cfg: dict,
) -> dict[str, Any]:
    if routing_stats:
        counts = {r: int(routing_stats.get(r, 0)) for r in KNOWN_ROUTES}
    else:
        counts = {r: len(route_buckets.get(r, [])) for r in KNOWN_ROUTES}
    total = sum(counts.values())
    llm_n = sum(counts.get(r, 0) for r in LLM_ROUTES)
    deberta_n = sum(counts.get(r, 0) for r in DEBERTA_ROUTES)
    if total <= 0:
        llm_frac = _config_llm_fraction(est_cfg) if est_cfg else 0.12
        sec_deberta, sec_llm = _default_bucket_secs(
            device_type, cpu_slow, est_cfg, device_routes
        )
        return {
            "sample_size": 0,
            "llm_fraction": round(llm_frac, 4),
            "deberta_fraction": round(1.0 - llm_frac, 4),
            "sec_deberta": sec_deberta,
            "sec_llm": sec_llm,
            "llm_calls_observed": 0,
            "deberta_calls_observed": 0,
        }
    llm_frac = llm_n / total
    deberta_frac = deberta_n / total
    deberta_vals = [v for r in DEBERTA_ROUTES for v in route_buckets.get(r, [])]
    llm_vals = [v for r in LLM_ROUTES for v in route_buckets.get(r, [])]
    sec_deberta, sec_llm = _default_bucket_secs(
        device_type, cpu_slow, est_cfg, device_routes
    )
    if deberta_vals:
        sec_deberta = round(float(statistics.median(deberta_vals)), 3)
    if llm_vals:
        sec_llm = round(float(statistics.median(llm_vals)), 2)
    return {
        "sample_size": total,
        "llm_fraction": round(llm_frac, 4),
        "deberta_fraction": round(deberta_frac, 4),
        "sec_deberta": sec_deberta,
        "sec_llm": sec_llm,
        "llm_calls_observed": llm_n,
        "deberta_calls_observed": deberta_n,
    }


def _merge_two_bucket(
    prev: dict[str, Any] | None,
    new: dict[str, Any],
) -> dict[str, Any]:
    if not prev or int(prev.get("sample_size", 0)) <= 0:
        return dict(new)
    prev_n = int(prev.get("sample_size", 0))
    new_n = int(new.get("sample_size", 0))
    total_n = min(prev_n + new_n, ESTIMATE_SAMPLE_MAX * 2)
    w_prev = prev_n / (prev_n + new_n)
    w_new = new_n / (prev_n + new_n)
    return {
        "sample_size": total_n,
        "llm_fraction": round(
            float(prev.get("llm_fraction", 0)) * w_prev
            + float(new.get("llm_fraction", 0)) * w_new,
            4,
        ),
        "deberta_fraction": round(
            float(prev.get("deberta_fraction", 0)) * w_prev
            + float(new.get("deberta_fraction", 0)) * w_new,
            4,
        ),
        "sec_deberta": round(
            float(prev.get("sec_deberta", 0.3)) * w_prev
            + float(new.get("sec_deberta", 0.3)) * w_new,
            3,
        ),
        "sec_llm": round(
            float(prev.get("sec_llm", 45)) * w_prev + float(new.get("sec_llm", 45)) * w_new,
            2,
        ),
        "llm_calls_observed": int(prev.get("llm_calls_observed", 0))
        + int(new.get("llm_calls_observed", 0)),
        "deberta_calls_observed": int(prev.get("deberta_calls_observed", 0))
        + int(new.get("deberta_calls_observed", 0)),
    }


def _two_bucket_from_file_batches(
    file_cal: dict[str, Any],
    *,
    device_routes: dict[str, Any],
    device_type: str,
    cpu_slow: dict,
    est_cfg: dict,
) -> dict[str, Any] | None:
    """Reconstitue le modele deux buckets depuis les batches deja termines sur ce fichier."""
    n_batches = int(file_cal.get("n_batches", 0))
    if n_batches < 1:
        return None
    fracs = file_cal.get("route_fractions")
    if not fracs:
        return None
    llm_frac = _llm_fraction(fracs)
    if llm_frac < 0.02:
        return None
    deberta_frac = 1.0 - llm_frac
    sec_deberta, sec_llm = _default_bucket_secs(
        device_type, cpu_slow, est_cfg, device_routes
    )
    file_routes = file_cal.get("routes") or {}
    deberta_p50 = [
        float(file_routes[r]["p50"])
        for r in DEBERTA_ROUTES
        if file_routes.get(r, {}).get("p50")
    ]
    llm_p50 = [
        float(file_routes[r]["p50"])
        for r in LLM_ROUTES
        if file_routes.get(r, {}).get("p50")
    ]
    if deberta_p50:
        sec_deberta = float(statistics.median(deberta_p50))
    if llm_p50:
        sec_llm = float(statistics.median(llm_p50))
    return {
        "llm_fraction": llm_frac,
        "deberta_fraction": deberta_frac,
        "sec_deberta": sec_deberta,
        "sec_llm": sec_llm,
        "sample_size": max(10, n_batches * 15),
        "source": f"batches reels fichier ({n_batches} batch(es))",
        "tier": "fichier",
    }


def _resolve_two_bucket_estimate(
    *,
    n_remaining: int,
    device_type: str,
    cpu_slow: dict,
    est_cfg: dict,
    data: list,
    target_is_annotated_fn: Callable[[dict], bool] | None,
    logs_dir: Path | None,
    filename: str | None,
    file_cal: dict[str, Any] | None,
    profile_cal: dict[str, Any] | None,
    device_cal: dict[str, Any] | None,
    cascade_mode: str = "qwen_only",
) -> dict[str, Any]:
    """Modele: n_llm = N * p_llm, n_deberta = N * p_deberta, temps = somme ponderee."""
    device_routes = (device_cal or {}).get("routes") or {}
    from cascade.core import (
        CASCADE_MODE_V8_QWEN,
        load_config,
        normalize_cascade_mode,
    )

    mode = normalize_cascade_mode(cascade_mode)
    sec_deberta, sec_llm = _default_bucket_secs(
        device_type, cpu_slow, est_cfg, device_routes
    )
    if mode == CASCADE_MODE_V8_QWEN:
        try:
            llm_frac = float(
                (load_config().get("cascade_v8") or {}).get("llm_fraction_estimate", 0.45)
            )
        except Exception:
            llm_frac = 0.45
    else:
        llm_frac = 1.0
    deberta_frac = 1.0 - llm_frac
    has_device_times = any(
        (device_routes.get(r) or {}).get("p50")
        for r in (*DEBERTA_ROUTES, *LLM_ROUTES)
    )
    source = (
        "repartition theorique (config) + temps machine calibres"
        if has_device_times
        else f"repartition et temps theoriques (config {device_type})"
    )
    sample_size = 0
    tier = "theorique"

    # Prefer timings recorded for this exact mode (Qwen only vs Cascade).
    mode_file = _mode_calibration_slot(file_cal, mode)
    mode_device = _mode_calibration_slot(device_cal, mode)
    mode_profile = _mode_calibration_slot(profile_cal, mode)
    file_tb = mode_file.get("two_bucket") or None
    if not file_tb:
        file_tb = mode_device.get("two_bucket") or mode_profile.get("two_bucket")
    legacy_tb = None
    if not file_tb:
        # Ancienne calibration sans by_mode : ne pas l'utiliser pour estimer
        # (sinon un run Qwen débloquerait Cascade). On garde le fallback
        # theorique uniquement pour le calcul interne ; estimate_available
        # reste False sans by_mode.
        legacy_tb = _two_bucket_from_file_batches(
            file_cal or {},
            device_routes=device_routes,
            device_type=device_type,
            cpu_slow=cpu_slow,
            est_cfg=est_cfg,
        )

    file_tb_usable = (
        file_tb
        and int(file_tb.get("sample_size", 0)) >= ESTIMATE_MIN_SAMPLES
        and (
            mode != CASCADE_MODE_V8_QWEN
            or float(file_tb.get("llm_fraction", 0)) >= 0.0
        )
    )

    if file_tb_usable:
        llm_frac = float(file_tb["llm_fraction"])
        deberta_frac = float(file_tb.get("deberta_fraction", 1.0 - llm_frac))
        sec_deberta = float(file_tb.get("sec_deberta", sec_deberta))
        sec_llm = float(file_tb.get("sec_llm", sec_llm))
        sample_size = int(file_tb["sample_size"])
        source = f"batches reels fichier ({sample_size} cibles)"
        tier = "fichier"
    elif legacy_tb:
        llm_frac = float(legacy_tb["llm_fraction"])
        deberta_frac = float(legacy_tb["deberta_fraction"])
        sec_deberta = float(legacy_tb["sec_deberta"])
        sec_llm = float(legacy_tb["sec_llm"])
        sample_size = int(legacy_tb["sample_size"])
        source = legacy_tb["source"]
        tier = legacy_tb["tier"]

    # Qwen seul : 100 % LLM. Cascade V8 : repartition calibree ou theorique.
    if mode == CASCADE_MODE_V8_QWEN:
        n_llm = int(round(n_remaining * llm_frac))
        n_deberta = max(0, n_remaining - n_llm)
        v8_overhead = 1.8
        processing_sec = n_remaining * sec_deberta * v8_overhead + n_llm * sec_llm
        formula = (
            f"{n_remaining} V8-duo x {sec_deberta * v8_overhead:.1f}s "
            f"+ {n_llm} Qwen x {sec_llm:.0f}s"
        )
    else:
        llm_frac = 1.0
        deberta_frac = 0.0
        n_llm = n_remaining
        n_deberta = 0
        processing_sec = n_llm * sec_llm
        formula = f"{n_llm} Qwen x {sec_llm:.0f}s"
    sec_per = processing_sec / max(1, n_remaining)
    return {
        "llm_fraction": llm_frac,
        "deberta_fraction": deberta_frac,
        "sec_deberta": sec_deberta,
        "sec_llm": sec_llm,
        "n_llm": n_llm,
        "n_deberta": n_deberta,
        "processing_sec": processing_sec,
        "sec_per_target": round(sec_per, 3),
        "sample_size": sample_size,
        "source": source,
        "tier": tier,
        "formula": formula,
    }


def _first_run_sec_per_target(device_type: str, cpu_slow: dict, est_cfg: dict) -> float | None:
    """Heuristique materiel pour le tout premier batch (aucune calibration)."""
    sec_d, sec_l = _default_bucket_secs(device_type, cpu_slow, est_cfg)
    llm_frac = float(cpu_slow.get("llm_fraction", 0.18)) if cpu_slow else _config_llm_fraction(est_cfg)
    return round((1.0 - llm_frac) * sec_d + llm_frac * sec_l, 2)


def _route_entry_usable(entry: dict[str, Any] | None, *, min_n: int) -> bool:
    if not entry:
        return False
    return int(entry.get("n", 0)) >= min_n


def _pick_route_calibration(
    route: str,
    *,
    file_cal: dict[str, Any] | None,
    profile_cal: dict[str, Any] | None,
    device_cal: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    file_routes = (file_cal or {}).get("routes") or {}
    profile_routes = (profile_cal or {}).get("routes") or {}
    device_routes = (device_cal or {}).get("routes") or {}
    if _route_entry_usable(file_routes.get(route), min_n=2):
        return file_routes[route], "fichier"
    if _route_entry_usable(profile_routes.get(route), min_n=4):
        return profile_routes[route], "profil structurel"
    if _route_entry_usable(device_routes.get(route), min_n=8):
        return device_routes[route], "machine"
    return None, "defaut"


def _blend_route_seconds(
    default_sec: float,
    *,
    empirical: float | None = None,
    calibrated: dict[str, float | int] | None = None,
    measured_sec_per_target: float | None = None,
    route_fraction: float = 0.0,
) -> float:
    values: list[float] = []
    if measured_sec_per_target is not None and route_fraction > 0:
        values.append(measured_sec_per_target * route_fraction * 2.5)
    if empirical is not None:
        values.append(empirical)
    if calibrated:
        for key in ("p50", "mean"):
            raw = calibrated.get(key)
            if raw is not None:
                values.append(float(raw))
                break
    if not values:
        return default_sec
    blended = statistics.mean(values)
    cap = default_sec * 2.5 if measured_sec_per_target else default_sec * 1.25
    return round(min(max(blended, default_sec * 0.5), cap), 3)


def _resolve_sec_per_route(
    device_type: str,
    est_cfg: dict,
    logs_dir: Path | None,
    *,
    filename: str | None,
    file_cal: dict[str, Any] | None,
    profile_cal: dict[str, Any] | None,
    device_cal: dict[str, Any] | None,
    fractions: dict[str, float],
    measured_sec_per_target: float | None,
) -> tuple[dict[str, float], str]:
    base = _default_sec_per_route(device_type, est_cfg)
    empirical = (
        _empirical_route_stats(logs_dir, filename=filename, min_samples=3)
        if logs_dir and filename
        else None
    )
    merged = dict(base)
    sources: set[str] = set()
    for route in KNOWN_ROUTES:
        cal_entry, src = _pick_route_calibration(
            route,
            file_cal=file_cal,
            profile_cal=profile_cal,
            device_cal=device_cal,
        )
        merged[route] = _blend_route_seconds(
            base[route],
            empirical=(empirical or {}).get(route),
            calibrated=cal_entry,
            measured_sec_per_target=measured_sec_per_target,
            route_fraction=fractions.get(route, 0.0),
        )
        if route in (empirical or {}):
            sources.add("session fichier")
        if cal_entry:
            sources.add(src)
    if measured_sec_per_target and file_cal:
        sources.add("mesure fichier")
    if "fichier" in sources or "mesure fichier" in sources:
        label = "calibration fichier"
    elif "profil structurel" in sources:
        label = "profil structurel"
    elif "machine" in sources:
        label = "calibration machine"
    elif "session fichier" in sources:
        label = "session fichier"
    else:
        label = f"defaut {_device_label(device_type).lower()}"
    if len(sources) > 1:
        label += " + " + ", ".join(sorted(s for s in sources if s not in label))
    return merged, label


def _resolve_model_load_sec(
    *,
    device_type: str,
    est_cfg: dict,
    logs_dir: Path | None,
    filename: str | None,
    file_cal: dict[str, Any] | None,
    profile_cal: dict[str, Any] | None,
    device_cal: dict[str, Any] | None,
    models_warm: bool,
) -> tuple[float, str]:
    if models_warm:
        warm = float(est_cfg.get("model_load_sec_warm", 2))
        return warm, "modeles deja charges"
    startup_values: list[float] = []
    for layer, label in (
        (file_cal, "fichier"),
        (profile_cal, "profil"),
        (device_cal, "machine"),
    ):
        if layer and layer.get("startup_sec"):
            startup_values.append(float(layer["startup_sec"]))
    if logs_dir and filename:
        _, delays = _parse_session_log_stats(
            logs_dir / SESSION_LOG_FILENAME, filename=filename
        )
        if delays:
            startup_values.append(statistics.median(delays[-3:]))
    if startup_values:
        return round(statistics.mean(startup_values), 1), "demarrage mesure"
    if device_type == "cpu":
        return float(est_cfg.get("model_load_sec_cpu", 18)), "premier batch CPU"
    return float(est_cfg.get("model_load_sec_gpu", 6)), f"premier batch {device_type}"


def _resolve_route_fractions(
    data: list,
    logs_dir: Path | None,
    est_cfg: dict,
    *,
    target_is_annotated_fn: Callable[[dict], bool] | None,
    filename: str | None,
    file_cal: dict[str, Any] | None,
    profile_cal: dict[str, Any] | None,
) -> tuple[dict[str, float], str]:
    from_data = _route_fractions_from_data(data, target_is_annotated_fn=target_is_annotated_fn)
    if from_data:
        return from_data, "corpus annote (fichier)"
    if file_cal and file_cal.get("route_fractions") and int(file_cal.get("n_batches", 0)) >= 1:
        return _normalize_fractions(dict(file_cal["route_fractions"])), "historique fichier"
    if profile_cal and profile_cal.get("route_fractions") and int(profile_cal.get("n_batches", 0)) >= 2:
        return _normalize_fractions(dict(profile_cal["route_fractions"])), "profil structurel"
    if logs_dir and filename:
        from_logs = _route_fractions_from_logs(logs_dir, filename=filename)
        if from_logs:
            return from_logs, "logs fichier"
    cfg_frac = est_cfg.get("route_fractions") or {}
    if cfg_frac:
        return _normalize_fractions({r: float(cfg_frac.get(r, 0)) for r in KNOWN_ROUTES}), "config"
    return dict(DEFAULT_ROUTE_FRACTIONS), "defaut"


def _format_route_breakdown(
    fractions: dict[str, float],
    sec_per_route: dict[str, float],
) -> str:
    parts: list[str] = []
    labels = {
        "deberta_auto": "DeBERTa auto",
        "deberta_ambiguous": "DeBERTa ambigu",
        "consensus": "consensus LLM",
        "human": "revue humaine",
        "rejected": "rejet",
    }
    for route in KNOWN_ROUTES:
        pct = int(round(fractions.get(route, 0) * 100))
        if pct <= 0:
            continue
        sec = sec_per_route.get(route, 0)
        parts.append(f"{pct}% {labels.get(route, route)} (~{sec:.0f}s)")
    return ", ".join(parts)


def _empirical_route_stats(
    logs_dir: Path,
    *,
    filename: str | None = None,
    min_samples: int = 5,
) -> dict[str, float] | None:
    """Durees medianes par route depuis les logs (filtre par fichier si precise)."""
    if not logs_dir.is_dir():
        return None
    pattern = re.compile(r"route=(\w+).*(?:duration|duree)=([\d.]+)s")
    buckets: dict[str, list[float]] = {}
    log_files = sorted(
        logs_dir.glob("run_model_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        return None
    session = logs_dir / SESSION_LOG_FILENAME
    if session in log_files:
        log_files = [session] + [p for p in log_files if p != session]
    active_file: str | None = None
    for path in log_files[:2]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines[-1500:]:
            if "BATCH START" in line:
                m = _BATCH_START_FILE.search(line)
                active_file = m.group(1) if m else None
                continue
            if "BATCH END" in line:
                active_file = None
                continue
            if filename and active_file != filename:
                continue
            m = pattern.search(line)
            if not m:
                continue
            route, raw = m.group(1), float(m.group(2))
            if 0.05 <= raw <= 900:
                buckets.setdefault(route, []).append(raw)
    if sum(len(v) for v in buckets.values()) < min_samples:
        return None
    out: dict[str, float] = {}
    for route, vals in buckets.items():
        if len(vals) < 2:
            continue
        vals.sort()
        out[route] = vals[len(vals) // 2]
    return out or None


def count_range_annotation_stats(
    data: list,
    start_index: int,
    end_index: int,
    *,
    ref_status_fn: Callable[[dict], str] | None = None,
    target_is_annotated_fn: Callable[[dict], bool] | None = None,
) -> dict[str, int | bool]:
    """Compte annotations existantes sur une plage d'indices."""
    refs_total = max(0, end_index - start_index + 1)
    targets_total = 0
    targets_annotated = 0
    refs_complete = 0
    refs_partial = 0
    for idx in range(start_index, end_index + 1):
        item = data[idx]
        if ref_status_fn:
            status = ref_status_fn(item)
            if status == "complete":
                refs_complete += 1
            elif status == "partial":
                refs_partial += 1
        targets = item.get("database") or []
        targets_total += len(targets)
        for target in targets:
            if target_is_annotated_fn and target_is_annotated_fn(target):
                targets_annotated += 1
    return {
        "refs_total": refs_total,
        "refs_complete": refs_complete,
        "refs_partial": refs_partial,
        "targets_total": targets_total,
        "targets_annotated": targets_annotated,
        "targets_remaining": targets_total - targets_annotated,
        "has_existing_annotations": targets_annotated > 0,
        "all_annotated": targets_total > 0 and targets_annotated == targets_total,
    }


def clear_target_for_reannotate(target: dict) -> None:
    """Efface les champs d'annotation avant une re-annotation forcee."""
    for key in (
        "related",
        "similarity_annotation",
        "cascade_route",
        "model_confidence",
        "llm_error",
        "llm_pred",
        "llm_confidence",
        "pipeline_compare",
        "cascade_v8",
        "annotated_by",
        "annotated_by_label",
    ):
        target.pop(key, None)


def _count_batch_targets(
    data: list,
    start_index: int,
    end_index: int,
    *,
    ref_status_fn: Callable[[dict], str] | None = None,
    target_is_annotated_fn: Callable[[dict], bool] | None = None,
    force_reannotate: bool = False,
) -> tuple[int, int, int]:
    """Retourne (refs, cibles totales, cibles restantes a traiter)."""
    n_refs = max(0, end_index - start_index + 1)
    n_targets = 0
    n_remaining = 0
    for idx in range(start_index, end_index + 1):
        item = data[idx]
        if not force_reannotate and ref_status_fn and ref_status_fn(item) == "complete":
            continue
        targets = item.get("database") or []
        n_targets += len(targets)
        for target in targets:
            if (
                not force_reannotate
                and target_is_annotated_fn
                and target_is_annotated_fn(target)
            ):
                continue
            n_remaining += 1
    return n_refs, n_targets, n_remaining


def estimate_batch(
    data: list,
    start_index: int,
    end_index: int,
    *,
    is_cpu: bool,
    cfg: dict,
    ref_status_fn: Callable[[dict], str] | None = None,
    target_is_annotated_fn: Callable[[dict], bool] | None = None,
    logs_dir: Path | None = None,
    force_reannotate: bool = False,
    device_type: str | None = None,
    models_warm: bool = False,
    base_dir: Path | None = None,
    filename: str | None = None,
    cascade_mode: str = "qwen_only",
) -> dict[str, Any]:
    from cascade.core import CASCADE_MODE_COMPARE, normalize_cascade_mode

    mode = normalize_cascade_mode(cascade_mode)
    run_cfg = cfg.get("run_model", {})
    cpu_slow = run_cfg.get("cpu_slow") or {}
    est_cfg = run_cfg.get("estimate") or {}
    device = device_type or ("cpu" if is_cpu else "cuda")
    is_cpu = device == "cpu"
    file_profile = compute_file_profile(data)
    store = _load_calibration_store(base_dir, device)
    file_cal = (store.get("by_file") or {}).get(filename) if filename else None
    profile_cal = (store.get("by_profile") or {}).get(file_profile["profile_key"])
    device_cal = store.get("device_global") or {}

    file_warm = (
        models_warm
        or models_warm_in_session()
        or _session_has_prior_targets(logs_dir, filename)
    )
    warm = file_warm or (
        not filename and _session_has_prior_targets(logs_dir)
    )

    range_stats = count_range_annotation_stats(
        data,
        start_index,
        end_index,
        ref_status_fn=ref_status_fn,
        target_is_annotated_fn=target_is_annotated_fn,
    )

    n_refs, n_targets, n_remaining = _count_batch_targets(
        data,
        start_index,
        end_index,
        ref_status_fn=ref_status_fn,
        target_is_annotated_fn=target_is_annotated_fn,
        force_reannotate=force_reannotate,
    )

    fractions, frac_source = _resolve_route_fractions(
        data,
        logs_dir,
        est_cfg,
        target_is_annotated_fn=target_is_annotated_fn,
        filename=filename,
        file_cal=file_cal,
        profile_cal=profile_cal,
    )

    bucket = _resolve_two_bucket_estimate(
        n_remaining=n_remaining,
        device_type=device,
        cpu_slow=cpu_slow,
        est_cfg=est_cfg,
        data=data,
        target_is_annotated_fn=target_is_annotated_fn,
        logs_dir=logs_dir,
        filename=filename,
        file_cal=file_cal,
        profile_cal=profile_cal,
        device_cal=device_cal,
        cascade_mode=mode,
    )
    calibration_tier = bucket["tier"]
    sec_per = bucket["sec_per_target"]
    sec_source = bucket["source"]

    model_load, load_source = _resolve_model_load_sec(
        device_type=device,
        est_cfg=est_cfg,
        logs_dir=logs_dir,
        filename=filename,
        file_cal=file_cal,
        profile_cal=profile_cal,
        device_cal=device_cal,
        models_warm=warm,
    )

    ref_overhead = float(est_cfg.get("sec_per_ref_overhead", 0.15))
    total_sec = model_load + bucket["processing_sec"] + n_refs * ref_overhead

    sec_per_route = _default_sec_per_route(device, est_cfg)
    frac_display = {
        r: bucket["deberta_fraction"] / len(DEBERTA_ROUTES)
        for r in DEBERTA_ROUTES
    }
    for r in LLM_ROUTES:
        frac_display[r] = bucket["llm_fraction"] / len(LLM_ROUTES)
    breakdown = _format_route_breakdown(frac_display, sec_per_route)
    hw = _device_label(device)
    method_label = (
        f"{hw} — {calibration_tier} — {bucket['formula']} ; "
        f"echantillon: {bucket['source']} ; "
        f"demarrage ~{model_load:.0f}s ({load_source})"
    )

    max_refs = int(cpu_slow.get("max_refs_suggested", 50)) if is_cpu else None
    warning = None
    if is_cpu and max_refs and n_refs > max_refs:
        warning = (
            f"On CPU, recommended batch size is at most {max_refs} references "
            f"(you selected {n_refs})."
        )
    if (
        not force_reannotate
        and n_remaining == 0
        and range_stats["has_existing_annotations"]
    ):
        warning = (warning + " " if warning else "") + (
            "All targets in this range are already annotated."
        )
    # Pas d'estimation theorique : il faut un premier run chronometre de CE mode.
    estimate_available = _has_recorded_estimate_timing(
        device_cal=device_cal,
        file_cal=file_cal,
        logs_dir=logs_dir,
        filename=filename,
        cascade_mode=mode,
    )
    if mode == CASCADE_MODE_COMPARE:
        estimate_hint = ESTIMATE_HINT_COMPARE
    elif not estimate_available:
        estimate_hint = ESTIMATE_HINT_FIRST_RUN
    else:
        estimate_hint = None

    route_fractions_pct = {r: round(frac_display.get(r, 0) * 100, 1) for r in KNOWN_ROUTES}
    route_seconds = {r: round(sec_per_route[r], 2) for r in KNOWN_ROUTES}
    llm_fraction = round(bucket["llm_fraction"] * 100, 1)
    deberta_fraction = round(bucket["deberta_fraction"] * 100, 1)

    return {
        "refs": n_refs,
        "targets": n_targets,
        "targets_remaining": n_remaining,
        "estimate_available": estimate_available,
        "estimate_hint": estimate_hint,
        "estimated_seconds": round(total_sec) if estimate_available else None,
        "estimated_label": _format_elapsed(total_sec) if estimate_available else None,
        "sec_per_target": round(sec_per, 2) if estimate_available else None,
        "estimate_method": (
            f"{calibration_tier}+{bucket['source']}+{load_source}"
            if estimate_available
            else None
        ),
        "estimate_detail": method_label if estimate_available else None,
        "estimate_formula": bucket["formula"] if estimate_available else None,
        "estimate_llm_calls": bucket["n_llm"] if estimate_available else None,
        "estimate_deberta_calls": bucket["n_deberta"] if estimate_available else None,
        "sec_deberta": bucket["sec_deberta"] if estimate_available else None,
        "sec_llm": bucket["sec_llm"] if estimate_available else None,
        "estimate_sample_size": (
            bucket["sample_size"] if estimate_available else 0
        ),
        "calibration_tier": calibration_tier if estimate_available else None,
        "file_profile": file_profile,
        "filename": filename,
        "route_fractions_pct": route_fractions_pct if estimate_available else {},
        "route_seconds": route_seconds if estimate_available else {},
        "fraction_source": frac_source if estimate_available else None,
        "model_load_seconds": round(model_load) if estimate_available else None,
        "model_load_source": load_source if estimate_available else None,
        "is_cpu": is_cpu,
        "device_type": device,
        "device_label": hw,
        "models_warm": warm,
        "llm_fraction_pct": llm_fraction if estimate_available else None,
        "deberta_fraction_pct": deberta_fraction if estimate_available else None,
        "max_refs_suggested": max_refs,
        "warning": warning,
        "has_existing_annotations": range_stats["has_existing_annotations"],
        "all_annotated": range_stats["all_annotated"],
        "targets_annotated": range_stats["targets_annotated"],
        "targets_total_in_range": range_stats["targets_total"],
        "refs_complete": range_stats["refs_complete"],
        "refs_partial": range_stats["refs_partial"],
        "force_reannotate": force_reannotate,
        "cascade_mode": mode,
    }


def find_resume_index(
    data: list,
    start_index: int,
    end_index: int,
    ref_status_fn,
) -> dict[str, Any]:
    for idx in range(start_index, end_index + 1):
        if ref_status_fn(data[idx]) != "complete":
            return {
                "resume_index": idx,
                "all_complete": False,
                "remaining_refs": end_index - idx + 1,
            }
    return {
        "resume_index": None,
        "all_complete": True,
        "remaining_refs": 0,
    }


def _infer_cascade_source_from_side(cascade_side: dict) -> tuple[str, str]:
    """Retombe sur la route stockée si decision_source absente (anciennes annos)."""
    source = cascade_side.get("decision_source")
    label = cascade_side.get("decision_source_label")
    if source in {"deberta", "qwen"}:
        return source, label or ("DeBERTa" if source == "deberta" else "Qwen")
    route = cascade_side.get("route")
    if route in {"deberta_auto", "deberta_ambiguous"}:
        return "deberta", "DeBERTa"
    if route == "consensus":
        return "qwen", "Qwen"
    if cascade_side.get("llm_pred") is not None and route not in {
        "deberta_auto",
        "deberta_ambiguous",
    }:
        return "qwen", "Qwen"
    if cascade_side.get("deberta_pred") is not None:
        return "deberta", "DeBERTa"
    return "unknown", "?"


def build_compare_disagreement_report(
    data: list,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    """Liste et agrège les désaccords Qwen vs Cascade sur une plage."""
    from collections import Counter

    items: list[dict[str, Any]] = []
    by_pair: Counter[str] = Counter()
    qwen_labels: Counter[str] = Counter()
    cascade_labels: Counter[str] = Counter()
    cascade_sources: Counter[str] = Counter()

    for ref_idx in range(start_index, end_index + 1):
        if ref_idx < 0 or ref_idx >= len(data):
            continue
        item = data[ref_idx]
        targets = item.get("database") or []
        for t_idx, target in enumerate(targets):
            cmp = target.get("pipeline_compare") or {}
            route = target.get("cascade_route")
            is_disagree = route == "compare_disagree" or (
                bool(cmp) and cmp.get("agree") is False
            )
            if not is_disagree:
                continue
            qwen_side = cmp.get("qwen_only") or {}
            cascade_side = cmp.get("deberta_qwen") or {}
            qwen_label = (
                qwen_side.get("related")
                or qwen_side.get("llm_pred")
                or target.get("llm_pred")
                or "?"
            )
            # Preférer llm_pred à deberta_pred : sinon cascade→human affiche
            # l'undetermined DeBERTa alors que Qwen-cascade a le même label que Qwen-only.
            cascade_label = (
                cascade_side.get("related")
                or cascade_side.get("llm_pred")
                or cascade_side.get("deberta_pred")
                or target.get("deberta_pred")
                or "?"
            )
            # Faux désaccord déjà stocké (agree=False car related cascade était None).
            if qwen_label != "?" and qwen_label == cascade_label:
                continue
            source, source_label = _infer_cascade_source_from_side(cascade_side)
            if cmp.get("cascade_source") in {"deberta", "qwen"}:
                source = cmp["cascade_source"]
                source_label = cmp.get("cascade_source_label") or source_label
            cascade_display = f"{cascade_label} ({source_label})"
            pair = f"{qwen_label} ≠ {cascade_display}"
            by_pair[pair] += 1
            qwen_labels[str(qwen_label)] += 1
            cascade_labels[str(cascade_display)] += 1
            cascade_sources[source_label] += 1
            news = str(target.get("news") or "").strip()
            preview = (news[:72] + "…") if len(news) > 72 else news
            items.append(
                {
                    "ref_index": ref_idx,
                    "target_index": t_idx,
                    "target_num": t_idx + 1,
                    "target_id": f"T{t_idx + 1}",
                    "label": f"ref {ref_idx} · T{t_idx + 1}",
                    "qwen": qwen_label,
                    "cascade": cascade_label,
                    "cascade_source": source,
                    "cascade_source_label": source_label,
                    "cascade_display": cascade_display,
                    "pair": pair,
                    "news_preview": preview,
                }
            )

    top_pairs = [
        {"pair": pair, "count": count}
        for pair, count in by_pair.most_common()
    ]
    insight = None
    if items:
        top_q = qwen_labels.most_common(1)[0]
        top_c = cascade_labels.most_common(1)[0]
        top_p = top_pairs[0] if top_pairs else None
        source_bits = ", ".join(
            f"{name} {n}" for name, n in cascade_sources.most_common()
        )
        insight = (
            f"{len(items)} désaccord(s). "
            f"Qwen penche surtout vers « {top_q[0]} » ({top_q[1]}) ; "
            f"Cascade vers « {top_c[0]} » ({top_c[1]})"
        )
        if source_bits:
            insight += f". Décision Cascade via : {source_bits}"
        if top_p:
            insight += f". Motif le plus fréquent : {top_p['pair']} ({top_p['count']})"
        insight += "."

    return {
        "count": len(items),
        "items": items,
        "by_pair": top_pairs,
        "qwen_labels": dict(qwen_labels),
        "cascade_labels": dict(cascade_labels),
        "cascade_sources": dict(cascade_sources),
        "insight": insight,
    }


def build_batch_summary(
    *,
    targets_annotated: int,
    total_targets: int,
    routing_stats: dict[str, int],
    elapsed_label: str,
    first_review_index: int | None,
    references_in_batch: int,
    compare_durations: dict[str, float | int] | None = None,
    compare_disagreements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    human = routing_stats.get("human", 0)
    rejected = routing_stats.get("rejected", 0)
    needs_manual = human + rejected
    summary: dict[str, Any] = {
        "title": "Automatic annotation complete",
        "prefilled_label": f"{targets_annotated} / {total_targets} targets pre-filled",
        "deberta_auto": routing_stats.get("deberta_auto", 0),
        "deberta_ambiguous": routing_stats.get("deberta_ambiguous", 0),
        "consensus": routing_stats.get("consensus", 0),
        "human_review": human,
        "rejected": rejected,
        "needs_manual": needs_manual,
        "duration_label": elapsed_label,
        "references_in_batch": references_in_batch,
        "first_review_index": first_review_index,
        "lines": [
            f"{targets_annotated} targets pre-filled out of {total_targets}",
            f"{routing_stats.get('deberta_auto', 0)} DeBERTa auto, "
            f"{routing_stats.get('deberta_ambiguous', 0)} DeBERTa ambiguous, "
            f"{routing_stats.get('consensus', 0)} LLM consensus",
            f"{human} legacy-human / {rejected} rejected (retry)",
            f"Duration: {elapsed_label}",
        ],
    }
    if compare_durations and int(compare_durations.get("pairs") or 0) > 0:
        qwen_s = float(compare_durations.get("qwen_s") or 0.0)
        cascade_s = float(compare_durations.get("cascade_s") or 0.0)
        deberta_s = float(compare_durations.get("cascade_deberta_s") or 0.0)
        cascade_llm_s = float(compare_durations.get("cascade_llm_s") or 0.0)
        n_deberta_only = int(compare_durations.get("n_deberta_only") or 0)
        n_cascade_llm = int(compare_durations.get("n_cascade_llm") or 0)
        delta_s = abs(qwen_s - cascade_s)
        if abs(qwen_s - cascade_s) < 1e-9:
            faster = "tie"
            faster_label = "égalité"
        elif qwen_s < cascade_s:
            faster = "qwen"
            faster_label = f"Qwen seul plus rapide (−{_format_elapsed(delta_s)})"
        else:
            faster = "cascade"
            faster_label = f"Cascade plus rapide (−{_format_elapsed(delta_s)})"
        qwen_label = _format_elapsed(qwen_s)
        cascade_label = _format_elapsed(cascade_s)
        delta_label = _format_elapsed(delta_s)
        deberta_label = _format_elapsed(deberta_s)
        cascade_llm_label = _format_elapsed(cascade_llm_s)
        summary["compare_time"] = {
            "qwen_s": round(qwen_s, 2),
            "cascade_s": round(cascade_s, 2),
            "cascade_deberta_s": round(deberta_s, 2),
            "cascade_llm_s": round(cascade_llm_s, 2),
            "n_deberta_only": n_deberta_only,
            "n_cascade_llm": n_cascade_llm,
            "delta_s": round(delta_s, 2),
            "faster": faster,
            "qwen_label": qwen_label,
            "cascade_label": cascade_label,
            "cascade_deberta_label": deberta_label,
            "cascade_llm_label": cascade_llm_label,
            "delta_label": delta_label,
            "faster_label": faster_label,
            "pairs": int(compare_durations["pairs"]),
        }
        summary["compare_time_label"] = (
            f"Qwen seul {qwen_label} − Cascade (DeBERTa {deberta_label} "
            f"+ Qwen-cascade {cascade_llm_label} = {cascade_label}) "
            f"→ Diff {delta_label} ({faster_label})"
        )
        summary["lines"].append(f"Compare timing: {summary['compare_time_label']}")
    if compare_disagreements and int(compare_disagreements.get("count") or 0) > 0:
        summary["compare_disagreements"] = compare_disagreements
        summary["lines"].append(
            compare_disagreements.get("insight")
            or f"{compare_disagreements['count']} compare disagreement(s)"
        )
    elif compare_disagreements is not None:
        summary["compare_disagreements"] = compare_disagreements
    return summary


build_french_summary = build_batch_summary


def _find_first_review_index(data: list, start: int, end: int) -> int | None:
    for idx in range(start, end + 1):
        for target in data[idx].get("database") or []:
            route = target.get("cascade_route")
            if route in ("human", "rejected", "compare_disagree"):
                return idx
            if route and not target.get("related"):
                return idx
    return None


def is_job_running() -> bool:
    with _lock:
        return bool(_state["running"])


def request_cancel_batch() -> bool:
    global _cancel_requested
    with _lock:
        if not _state["running"]:
            return False
        _cancel_requested = True
        _state["progress"]["message"] = "Cancellation requested…"
    try:
        logging.getLogger("run_model").info("CANCEL requested by user — stopping batch")
    except Exception:
        pass
    return True


def _cancelled() -> bool:
    with _lock:
        return _cancel_requested


def _run_cancellable(work: Callable[[], Any], should_cancel: Callable[[], bool], poll: float = 0.2) -> Any:
    """Run blocking ML work in a side thread; raise when cancel is requested."""
    from cascade.core import BatchCancelledError

    if not should_cancel:
        return work()

    holder: dict[str, Any] = {}
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            holder["result"] = work()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while thread.is_alive():
        if should_cancel():
            raise BatchCancelledError("Batch cancelled by user.")
        thread.join(poll)
    if errors:
        raise errors[0]
    return holder["result"]


def start_batch(
    flask_app,
    *,
    start_index: int,
    end_index: int,
    threshold: float,
    original_filename: str,
    file_path: Path,
    base_dir: Path,
    targets_total: int | None = None,
    force_reannotate: bool = False,
    backup_path: str | None = None,
    app_module=None,
    cascade_mode: str = "qwen_only",
) -> bool:
    if app_module is None:
        app_module = sys.modules.get("__main__")
    refs_total = max(0, end_index - start_index + 1)
    with _lock:
        if _state["running"]:
            return False
        _cancel_requested = False
        _state["running"] = True
        _state["started_at"] = datetime.utcnow().isoformat() + "Z"
        _state["progress"] = {
            "refs_done": 0,
            "refs_total": refs_total,
            "targets_done": 0,
            "targets_total": targets_total if targets_total is not None else 0,
            "current_ref_index": start_index,
            "current_ref_num": 0,
            "message": "Loading ML models…",
        }
        _state["result"] = None
        _state["error"] = None

    thread = threading.Thread(
        target=_run_batch,
        args=(
            flask_app,
            app_module,
            start_index,
            end_index,
            threshold,
            original_filename,
            file_path,
            base_dir,
            force_reannotate,
            backup_path,
            cascade_mode,
        ),
        daemon=True,
    )
    thread.start()
    return True


def _run_batch(
    flask_app,
    app_mod,
    start_index: int,
    end_index: int,
    threshold: float,
    original_filename: str,
    file_path: Path,
    base_dir: Path,
    force_reannotate: bool = False,
    backup_path: str | None = None,
    cascade_mode: str = "qwen_only",
) -> None:
    try:
        with flask_app.app_context():
            _run_batch_inner(
                app_mod,
                start_index,
                end_index,
                threshold,
                original_filename,
                file_path,
                base_dir,
                force_reannotate=force_reannotate,
                backup_path=backup_path,
                cascade_mode=cascade_mode,
            )
    except Exception as e:
        logging.getLogger("run_model").exception("Erreur fatale batch: %s", e)
        _finish(error=str(e))


def _run_batch_inner(
    app_mod,
    start_index: int,
    end_index: int,
    threshold: float,
    original_filename: str,
    file_path: Path,
    base_dir: Path,
    *,
    force_reannotate: bool = False,
    backup_path: str | None = None,
    cascade_mode: str = "qwen_only",
) -> None:
    from annotation_store import file_lock_for

    file_lock = file_lock_for(file_path)
    if not file_lock.acquire(blocking=False):
        _finish(error="File locked — another process is using this JSON.")
        return

    logger = _setup_logger(base_dir / "logs")
    try:
        from cascade.core import (
            BatchCancelledError,
            CASCADE_MODE_COMPARE,
            CASCADE_MODE_V8_QWEN,
            normalize_cascade_mode,
        )

        mode = normalize_cascade_mode(cascade_mode)

        batch_start = time.monotonic()
        _set_progress(message="Loading ML models (may take 1–3 min on CPU)…")
        cfg = app_mod._load_cascade_config()
        run_cfg = cfg.get("run_model", {})
        cpu_slow = run_cfg.get("cpu_slow") or {}
        save_every = max(1, int(run_cfg.get("save_every_n_refs", 3)))

        report = app_mod.build_report(base_dir, cfg)
        is_cpu = report.get("hardware", {}).get("gpu", {}).get("device") == "cpu"
        hw_label = report.get("hardware", {}).get("gpu", {}).get("label", "CPU")
        cfg = effective_run_config(cfg, is_cpu)

        no_annotation_timeout = int(run_cfg.get("no_annotation_timeout", 360))
        if is_cpu:
            no_annotation_timeout = max(
                no_annotation_timeout,
                int(cpu_slow.get("no_annotation_timeout_sec", 900)),
            )

        if is_cpu:
            app_mod.cascade_engine = None

        load_box: dict[str, Any] = {}
        load_errors: list[BaseException] = []

        def _load_models() -> None:
            try:
                eng = app_mod.get_cascade_engine()
                if is_cpu:
                    eng.cfg = cfg
                if mode == CASCADE_MODE_V8_QWEN or mode == CASCADE_MODE_COMPARE:
                    eng.ensure_v8()
                load_box["engine"] = eng
                # Trained SBERT v2 (Release v8.0.0) — similarity scores only
                load_box["sbert"] = app_mod.get_sbert_model()
            except BaseException as e:
                load_errors.append(e)

        load_start = time.monotonic()
        loader = threading.Thread(target=_load_models, daemon=True)
        loader.start()
        while loader.is_alive():
            if _cancelled():
                logger.info("Batch cancelled during model loading")
                _finish(error="Batch cancelled.")
                return
            loader.join(0.2)

        if load_errors:
            logger.error("Model loading failed: %s", load_errors[0])
            _finish(error=str(load_errors[0]))
            return

        load_sec = round(time.monotonic() - load_start, 1)
        logger.info("Models loaded in %ss (%s)", load_sec, hw_label)

        engine = load_box["engine"]
        sbert = load_box["sbert"]
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        targets_total = sum(
            len(data[idx].get("database") or [])
            for idx in range(start_index, end_index + 1)
        )
        _set_progress(targets_total=targets_total, message="Annotation in progress…")
        logger.info(
            "BATCH START file=%s indices=%s-%s refs=%s targets=%s threshold=%.3f "
            "cascade_mode=%s force_reannotate=%s backup=%s",
            original_filename,
            start_index,
            end_index,
            end_index - start_index + 1,
            targets_total,
            threshold,
            mode,
            force_reannotate,
            backup_path or "none",
        )

        targets_annotated_count = 0
        pairs_evaluated = 0
        total_targets_evaluated = 0
        references_fully_annotated_count = 0
        refs_processed_in_batch = 0
        routing_stats = {
            "llm_auto": 0,
            "deberta_auto": 0,
            "deberta_ambiguous": 0,
            "consensus": 0,
            "compare_agree": 0,
            "compare_disagree": 0,
            "v8_duo": 0,
            "v8_reranker": 0,
            "v8_qwen": 0,
            "rejected": 0,
            "human": 0,
        }
        compare_durations: dict[str, float | int] = {
            "qwen_s": 0.0,
            "cascade_s": 0.0,
            "cascade_deberta_s": 0.0,
            "cascade_llm_s": 0.0,
            "n_deberta_only": 0,
            "n_cascade_llm": 0,
            "pairs": 0,
        }

        def save_checkpoint() -> None:
            app_mod.db.session.commit()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        def _abort_cancelled() -> bool:
            if not _cancelled():
                return False
            save_checkpoint()
            elapsed = _format_elapsed(time.monotonic() - batch_start)
            _finish(
                result=_partial_result(
                    app_mod,
                    data,
                    original_filename,
                    targets_annotated_count,
                    total_targets_evaluated,
                    references_fully_annotated_count,
                    routing_stats,
                    elapsed,
                    partial_error="Batch cancelled by user.",
                    start_index=start_index,
                    end_index=end_index,
                    compare_durations=compare_durations,
                ),
                error="Batch cancelled.",
            )
            return True

        for idx in range(start_index, end_index + 1):
            if _abort_cancelled():
                return

            item = data[idx]
            ref_num = idx - start_index + 1

            _set_progress(
                current_ref_index=idx,
                current_ref_num=ref_num,
                message=f"Reference {ref_num} / {end_index - start_index + 1} (index {idx})",
            )

            if (
                not force_reannotate
                and app_mod.reference_status_from_item(item) == "complete"
            ):
                targets_skip = item.get("database") or []
                refs_processed_in_batch += 1
                pairs_evaluated += len(targets_skip)
                _set_progress(
                    refs_done=refs_processed_in_batch,
                    targets_done=pairs_evaluated,
                    message=f"Reference {ref_num} already complete (skipped)",
                )
                continue

            anchor_news = item.get("news", "")
            anchor_topic = item.get("topic", "N/A") or "N/A"
            anchor_date_raw = item.get("metadata", {}).get("date")
            anchor_date = str(anchor_date_raw)[:10] if anchor_date_raw else "N/A"
            anchor_text = f"[Topic: {anchor_topic}] [Date: {anchor_date}] {anchor_news}"
            targets = item.get("database") or []
            if not targets:
                refs_processed_in_batch += 1
                _set_progress(refs_done=refs_processed_in_batch)
                continue

            total_targets_evaluated += len(targets)
            all_above_threshold = True

            for i, target in enumerate(targets):
                if _abort_cancelled():
                    return

                if (
                    not force_reannotate
                    and app_mod._target_is_annotated(target)
                ):
                    pairs_evaluated += 1
                    _set_progress(targets_done=pairs_evaluated)
                    logger.info(
                        "SKIP ref_index=%s target=%s/%s (already annotated)",
                        idx,
                        i + 1,
                        len(targets),
                    )
                    continue

                if force_reannotate:
                    clear_target_for_reannotate(target)

                # Timeout seulement si aucune cible terminee (y compris human/rejected)
                if (
                    pairs_evaluated == 0
                    and time.monotonic() - batch_start > no_annotation_timeout
                ):
                    save_checkpoint()
                    elapsed = _format_elapsed(time.monotonic() - batch_start)
                    logger.error("Timeout: zero targets annotated after %ss", no_annotation_timeout)
                    _finish(
                        result=_partial_result(
                            app_mod,
                            data,
                            original_filename,
                            targets_annotated_count,
                            total_targets_evaluated,
                            references_fully_annotated_count,
                            routing_stats,
                            elapsed,
                            partial_error=(
                                f"No target annotated after {no_annotation_timeout}s."
                            ),
                            start_index=start_index,
                            end_index=end_index,
                            compare_durations=compare_durations,
                        ),
                        error=f"No target annotated after {no_annotation_timeout}s.",
                    )
                    return

                _set_progress(
                    targets_done=pairs_evaluated,
                    refs_done=refs_processed_in_batch,
                    current_ref_num=ref_num,
                    current_target_num=i + 1,
                    current_target_total=len(targets),
                    message=(
                        f"Reference {ref_num}/{end_index - start_index + 1} — "
                        f"target {i + 1}/{len(targets)} ("
                        f"{'Compare Qwen↔Cascade' if mode == CASCADE_MODE_COMPARE else 'Cascade (incl. Qwen)' if mode == CASCADE_MODE_V8_QWEN else 'Qwen only'} running…)"
                    ),
                )
                logger.info(
                    "TARGET START ref_index=%s ref=%s/%s target=%s/%s",
                    idx,
                    ref_num,
                    end_index - start_index + 1,
                    i + 1,
                    len(targets),
                )

                pair_start = time.monotonic()
                target_news = target.get("news", "")
                target_topic = target.get("topic", "N/A") or "N/A"
                target_date_raw = target.get("metadata", {}).get("date")
                target_date = str(target_date_raw)[:10] if target_date_raw else "N/A"
                target_text = f"[Topic: {target_topic}] [Date: {target_date}] {target_news}"

                try:
                    out = engine.route(
                        anchor_text,
                        target_text,
                        tau_auto=threshold,
                        should_cancel=_cancelled,
                        cascade_mode=mode,
                    )
                except BatchCancelledError:
                    logger.info(
                        "Batch cancelled at ref_index=%s target=%s/%s",
                        idx,
                        i + 1,
                        len(targets),
                    )
                    _abort_cancelled()
                    return

                route_name = out["route"]
                target["cascade_route"] = route_name
                if out.get("v8_stage") or out.get("v8_rule"):
                    target["cascade_v8"] = {
                        "status": out.get("v8_status"),
                        "stage": out.get("v8_stage"),
                        "rule": out.get("v8_rule"),
                        "label": out.get("v8_label"),
                        "confidence": out.get("v8_conf"),
                        "minilm": out.get("v8_minilm_label"),
                        "deberta": out.get("v8_deberta_label"),
                        "reranker_undet": out.get("v8_reranker_undet_prob"),
                    }
                # Who decided the label (visible in UI + JSON)
                if route_name == "v8_duo":
                    target["annotated_by"] = "duo"
                    target["annotated_by_label"] = "Duo (MiniLM + DeBERTa Large)"
                elif route_name == "v8_reranker":
                    target["annotated_by"] = "reranker"
                    target["annotated_by_label"] = "Reranker (undetermined)"
                elif route_name == "v8_qwen":
                    target["annotated_by"] = "qwen"
                    target["annotated_by_label"] = "Qwen (Cascade fallback)"
                elif route_name in {"llm_auto", "consensus"}:
                    target["annotated_by"] = "qwen"
                    target["annotated_by_label"] = "Qwen"
                elif route_name == "compare_agree":
                    c_side = (out.get("pipeline_compare") or {}).get("deberta_qwen") or {}
                    c_route = c_side.get("route") or ""
                    if c_route == "v8_duo":
                        target["annotated_by"] = "duo"
                        target["annotated_by_label"] = "Compare agree · Cascade Duo"
                    elif c_route == "v8_reranker":
                        target["annotated_by"] = "reranker"
                        target["annotated_by_label"] = "Compare agree · Cascade Reranker"
                    elif c_route == "v8_qwen":
                        target["annotated_by"] = "qwen"
                        target["annotated_by_label"] = "Compare agree · Cascade Qwen"
                    else:
                        target["annotated_by"] = "compare"
                        target["annotated_by_label"] = "Compare agree (Qwen = Cascade)"
                elif route_name == "compare_disagree":
                    target["annotated_by"] = "compare_disagree"
                    target["annotated_by_label"] = "Compare disagree — needs review"
                elif route_name in {"human", "rejected"}:
                    target["annotated_by"] = "retry"
                    target["annotated_by_label"] = "Not final — retry next run"
                if out.get("pipeline_compare"):
                    target["pipeline_compare"] = out["pipeline_compare"]
                    cmp = out["pipeline_compare"]
                    q_dur = (cmp.get("qwen_only") or {}).get("duration_s")
                    c_side = cmp.get("deberta_qwen") or {}
                    c_dur = c_side.get("duration_s")
                    if isinstance(q_dur, (int, float)) and isinstance(c_dur, (int, float)):
                        compare_durations["qwen_s"] = (
                            float(compare_durations["qwen_s"]) + float(q_dur)
                        )
                        compare_durations["cascade_s"] = (
                            float(compare_durations["cascade_s"]) + float(c_dur)
                        )
                        deb_dur = c_side.get("deberta_duration_s")
                        llm_dur = c_side.get("cascade_llm_duration_s")
                        if isinstance(deb_dur, (int, float)):
                            compare_durations["cascade_deberta_s"] = (
                                float(compare_durations["cascade_deberta_s"]) + float(deb_dur)
                            )
                        else:
                            # Anciennes annos sans split : tout le temps cascade
                            # hors LLM connu est attribué à DeBERTa.
                            llm_fallback = float(llm_dur) if isinstance(llm_dur, (int, float)) else 0.0
                            compare_durations["cascade_deberta_s"] = (
                                float(compare_durations["cascade_deberta_s"])
                                + max(0.0, float(c_dur) - llm_fallback)
                            )
                        if isinstance(llm_dur, (int, float)) and float(llm_dur) > 0:
                            compare_durations["cascade_llm_s"] = (
                                float(compare_durations["cascade_llm_s"]) + float(llm_dur)
                            )
                            compare_durations["n_cascade_llm"] = (
                                int(compare_durations["n_cascade_llm"]) + 1
                            )
                        else:
                            compare_durations["n_deberta_only"] = (
                                int(compare_durations["n_deberta_only"]) + 1
                            )
                        compare_durations["pairs"] = int(compare_durations["pairs"]) + 1
                if out.get("llm_error"):
                    target["llm_error"] = out["llm_error"]

                if route_name in {
                    "llm_auto",
                    "deberta_auto",
                    "deberta_ambiguous",
                    "consensus",
                    "compare_agree",
                    "v8_duo",
                    "v8_reranker",
                    "v8_qwen",
                }:
                    target["related"] = out["related"]
                    conf = (
                        out["llm_conf"]
                        if route_name in {"consensus", "llm_auto", "v8_qwen", "compare_agree"}
                        else out.get("v8_conf") or out["deberta_conf"]
                    )
                    target["model_confidence"] = (
                        round(conf, 4) if conf is not None else None
                    )
                    # Keep LLM / cascade default sim for reference; final score = SBERT.
                    if out.get("llm_sim") is not None:
                        target["llm_sim"] = round(float(out["llm_sim"]), 4)
                    elif out.get("similarity_annotation") is not None:
                        target["cascade_sim"] = round(
                            float(out["similarity_annotation"]), 4
                        )

                    if _abort_cancelled():
                        return
                    try:
                        embs = _run_cancellable(
                            lambda: sbert.encode(
                                [anchor_text, target_text],
                                show_progress_bar=False,
                            ),
                            _cancelled,
                        )
                    except BatchCancelledError:
                        logger.info(
                            "Batch cancelled during SBERT encoding ref_index=%s target=%s/%s",
                            idx,
                            i + 1,
                            len(targets),
                        )
                        _abort_cancelled()
                        return
                    emb_anchor, emb_target = embs[0], embs[1]
                    sim = float(
                        np.dot(emb_anchor, emb_target)
                        / (np.linalg.norm(emb_anchor) * np.linalg.norm(emb_target))
                    )
                    target["similarity_annotation"] = round(
                        max(0.0, min(1.0, sim)), 4
                    )
                    target["similarity_source"] = "sbert"
                    from cascade.core import enforce_label_sim_consistency

                    target["related"] = enforce_label_sim_consistency(
                        target.get("related"),
                        target["similarity_annotation"],
                        anchor=anchor_text,
                        target=target_text,
                    )
                    targets_annotated_count += 1
                    routing_stats[route_name] = routing_stats.get(route_name, 0) + 1
                elif route_name == "compare_disagree":
                    target["related"] = None
                    target["similarity_annotation"] = None
                    all_above_threshold = False
                    if out.get("llm_pred") is not None:
                        target["llm_pred"] = out["llm_pred"]
                        target["llm_confidence"] = round(out["llm_conf"], 4)
                    if out.get("deberta_pred") is not None:
                        target["deberta_pred"] = out["deberta_pred"]
                    routing_stats[route_name] = routing_stats.get(route_name, 0) + 1
                else:
                    all_above_threshold = False
                    conf = out.get("deberta_conf")
                    if conf is None:
                        conf = out.get("llm_conf")
                    if conf is not None:
                        target["model_confidence"] = round(conf, 4)
                    if out.get("llm_pred") is not None:
                        target["llm_pred"] = out["llm_pred"]
                        target["llm_confidence"] = round(out["llm_conf"], 4)
                    routing_stats[route_name] = routing_stats.get(route_name, 0) + 1

                pair_sec = round(time.monotonic() - pair_start, 2)
                pairs_evaluated += 1
                related = target.get("related")
                sim = target.get("similarity_annotation")
                extra = ""
                if out.get("llm_error"):
                    extra += f" llm_error={out.get('llm_error')}"
                if out.get("llm_pred") is not None:
                    extra += f" llm_pred={out.get('llm_pred')}"
                logger.info(
                    "TARGET DONE ref_index=%s ref=%s/%s target=%s/%s route=%s "
                    "duration=%ss related=%s sim=%s conf=%s%s",
                    idx,
                    ref_num,
                    end_index - start_index + 1,
                    i + 1,
                    len(targets),
                    route_name,
                    pair_sec,
                    related,
                    sim,
                    target.get("model_confidence"),
                    extra,
                )
                _set_progress(
                    targets_done=pairs_evaluated,
                    refs_done=refs_processed_in_batch,
                    current_ref_num=ref_num,
                )

            if all_above_threshold:
                references_fully_annotated_count += 1

            app_mod.sync_reference_to_db(original_filename, item, source="auto")
            refs_processed_in_batch += 1
            _set_progress(refs_done=refs_processed_in_batch, current_ref_num=ref_num)

            if refs_processed_in_batch % save_every == 0:
                save_checkpoint()
                logger.info("CHECKPOINT saved after ref_index=%s", idx)

        save_checkpoint()
        data = app_mod.enrich_data_with_status(original_filename, data)
        processed_ids = {
            str(item.get("news_id"))
            for item in data
            if app_mod.reference_status_from_item(item) == "complete"
        }
        elapsed = _format_elapsed(time.monotonic() - batch_start)
        first_review = _find_first_review_index(data, start_index, end_index)
        compare_disagreements = build_compare_disagreement_report(
            data, start_index, end_index
        )
        summary = build_batch_summary(
            targets_annotated=targets_annotated_count,
            total_targets=total_targets_evaluated,
            routing_stats=routing_stats,
            elapsed_label=elapsed,
            first_review_index=first_review,
            references_in_batch=end_index - start_index + 1,
            compare_durations=compare_durations,
            compare_disagreements=compare_disagreements,
        )
        logger.info(
            "BATCH END file=%s annotated=%s/%s refs_fully_done=%s duration=%s "
            "routes=%s compare_timing=%s disagreements=%s",
            original_filename,
            targets_annotated_count,
            total_targets_evaluated,
            references_fully_annotated_count,
            elapsed,
            routing_stats,
            compare_durations,
            compare_disagreements.get("count", 0),
        )
        global _session_models_warm
        device_type = report.get("hardware", {}).get("gpu", {}).get("device", "cpu")
        batch_elapsed = time.monotonic() - batch_start
        refresh_estimate_calibration(
            base_dir=base_dir,
            logs_dir=base_dir / "logs",
            device_type=device_type,
            filename=original_filename,
            file_profile=compute_file_profile(data),
            routing_stats=routing_stats,
            batch_elapsed_sec=batch_elapsed,
            targets_processed=targets_annotated_count,
            cpu_slow=cpu_slow,
            est_cfg=run_cfg.get("estimate") or {},
            cascade_mode=mode,
        )
        _session_models_warm = True
        _finish(
            result={
                "ok": True,
                "message": summary["lines"][0] + " — " + summary["duration_label"],
                "summary_en": summary,
                "summary_fr": summary,
                "compare_time": summary.get("compare_time"),
                "compare_time_label": summary.get("compare_time_label"),
                "compare_disagreements": summary.get("compare_disagreements"),
                "annotated_count": references_fully_annotated_count,
                "data": data,
                "processed_ids": list(processed_ids),
                "routing_stats": routing_stats,
                "targets_annotated": targets_annotated_count,
                "targets_total": total_targets_evaluated,
                "elapsed_seconds": round(time.monotonic() - batch_start, 1),
                "elapsed_label": elapsed,
                "first_review_index": first_review,
            }
        )
    except Exception as e:
        app_mod.db.session.rollback()
        try:
            save_checkpoint()
        except Exception:
            pass
        logger.exception("Erreur batch: %s", e)
        _finish(error=str(e))
    finally:
        file_lock.release()


def _partial_result(
    app_mod,
    data,
    original_filename,
    targets_annotated_count,
    total_targets_evaluated,
    references_fully_annotated_count,
    routing_stats,
    elapsed_label,
    partial_error,
    start_index: int = 0,
    end_index: int | None = None,
    compare_durations: dict[str, float | int] | None = None,
) -> dict:
    if end_index is None:
        end_index = len(data) - 1
    enriched = app_mod.enrich_data_with_status(original_filename, data)
    compare_disagreements = build_compare_disagreement_report(
        enriched, start_index, end_index
    )
    summary = build_batch_summary(
        targets_annotated=targets_annotated_count,
        total_targets=total_targets_evaluated,
        routing_stats=routing_stats,
        elapsed_label=elapsed_label,
        first_review_index=_find_first_review_index(data, start_index, end_index),
        references_in_batch=end_index - start_index + 1,
        compare_durations=compare_durations,
        compare_disagreements=compare_disagreements,
    )
    processed_ids = {
        str(item.get("news_id"))
        for item in data
        if app_mod.reference_status_from_item(item) == "complete"
    }
    return {
        "ok": False,
        "partial": True,
        "error": partial_error,
        "message": (
            f"Run Model interrupted — {targets_annotated_count} target(s) saved. "
            f"Duration: {elapsed_label}."
        ),
        "summary_en": summary,
        "summary_fr": summary,
        "compare_time": summary.get("compare_time"),
        "compare_time_label": summary.get("compare_time_label"),
        "compare_disagreements": summary.get("compare_disagreements"),
        "annotated_count": references_fully_annotated_count,
        "data": enriched,
        "processed_ids": list(processed_ids),
        "routing_stats": routing_stats,
        "elapsed_label": elapsed_label,
        "first_review_index": summary.get("first_review_index"),
    }
