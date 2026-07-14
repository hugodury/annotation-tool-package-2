"""Run Model background job: progress, logs, incremental save."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
    "deberta_auto",
    "deberta_ambiguous",
    "consensus",
    "human",
    "rejected",
)

DEFAULT_ROUTE_FRACTIONS: dict[str, float] = {
    "deberta_auto": 0.52,
    "deberta_ambiguous": 0.25,
    "consensus": 0.10,
    "human": 0.08,
    "rejected": 0.05,
}


def _normalize_fractions(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.get(r, 0.0) for r in KNOWN_ROUTES)
    if total <= 0:
        return dict(DEFAULT_ROUTE_FRACTIONS)
    return {r: raw.get(r, 0.0) / total for r in KNOWN_ROUTES}


def _route_fractions_from_data(
    data: list,
    *,
    target_is_annotated_fn: Callable[[dict], bool] | None = None,
    min_samples: int = 24,
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


def _route_fractions_from_logs(logs_dir: Path, *, min_samples: int = 24) -> dict[str, float] | None:
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
    # Current session uses a single log file; read it first.
    session = logs_dir / SESSION_LOG_FILENAME
    if session in log_files:
        log_files = [session] + [p for p in log_files if p != session]
    for path in log_files[:3]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines[-2000:]:
            m = pattern.search(line)
            if not m or m.group(1) not in counts:
                continue
            counts[m.group(1)] += 1
            total += 1
    if total < min_samples:
        return None
    return _normalize_fractions({r: float(counts[r]) for r in KNOWN_ROUTES})


def _default_sec_per_route(is_cpu: bool, est_cfg: dict) -> dict[str, float]:
    key = "route_sec_cpu" if is_cpu else "route_sec_gpu"
    cfg_map = est_cfg.get(key) or {}
    fallback_cpu = {
        "deberta_auto": 0.6,
        "deberta_ambiguous": 0.6,
        "consensus": 32.0,
        "human": 28.0,
        "rejected": 30.0,
    }
    fallback_gpu = {
        "deberta_auto": 0.35,
        "deberta_ambiguous": 0.35,
        "consensus": 14.0,
        "human": 12.0,
        "rejected": 14.0,
    }
    base = fallback_cpu if is_cpu else fallback_gpu
    return {r: float(cfg_map.get(r, base[r])) for r in KNOWN_ROUTES}


def _resolve_route_fractions(
    data: list,
    logs_dir: Path | None,
    est_cfg: dict,
    *,
    target_is_annotated_fn: Callable[[dict], bool] | None,
) -> tuple[dict[str, float], str]:
    from_data = _route_fractions_from_data(data, target_is_annotated_fn=target_is_annotated_fn)
    if from_data:
        return from_data, "corpus annote"
    from_logs = _route_fractions_from_logs(logs_dir) if logs_dir else None
    if from_logs:
        return from_logs, "logs locaux"
    cfg_frac = est_cfg.get("route_fractions") or {}
    if cfg_frac:
        return _normalize_fractions({r: float(cfg_frac.get(r, 0)) for r in KNOWN_ROUTES}), "config"
    return dict(DEFAULT_ROUTE_FRACTIONS), "defaut"


def _resolve_sec_per_route(
    is_cpu: bool,
    est_cfg: dict,
    logs_dir: Path | None,
) -> tuple[dict[str, float], str]:
    base = _default_sec_per_route(is_cpu, est_cfg)
    empirical = _empirical_route_stats(logs_dir) if logs_dir else None
    if not empirical:
        return base, "defaut materiel"
    merged = dict(base)
    for route, sec in empirical.items():
        if route in merged:
            merged[route] = sec
    return merged, "calibration logs + materiel"


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


def _empirical_route_stats(logs_dir: Path, *, min_samples: int = 5) -> dict[str, float] | None:
    """Durees medianes par route cascade depuis les logs locaux."""
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
    # Current session uses a single log file; read it first.
    session = logs_dir / SESSION_LOG_FILENAME
    if session in log_files:
        log_files = [session] + [p for p in log_files if p != session]
    for path in log_files[:3]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines[-1200:]:
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
        if len(vals) < 3:
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
) -> dict[str, Any]:
    run_cfg = cfg.get("run_model", {})
    cpu_slow = run_cfg.get("cpu_slow") or {}
    est_cfg = run_cfg.get("estimate") or {}

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
        data, logs_dir, est_cfg, target_is_annotated_fn=target_is_annotated_fn
    )
    sec_per_route, sec_source = _resolve_sec_per_route(is_cpu, est_cfg, logs_dir)

    sec_per = sum(fractions[r] * sec_per_route[r] for r in KNOWN_ROUTES)
    model_load = float(
        est_cfg.get("model_load_sec_cpu" if is_cpu else "model_load_sec_gpu", 90 if is_cpu else 20)
    )
    total_sec = model_load + n_remaining * sec_per

    breakdown = _format_route_breakdown(fractions, sec_per_route)
    hw = "CPU" if is_cpu else "GPU"
    method_label = (
        f"{hw} — repartition ({frac_source}) : {breakdown} ; "
        f"~{sec_per:.1f} s/cible ({sec_source})"
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

    route_fractions_pct = {r: round(fractions[r] * 100, 1) for r in KNOWN_ROUTES}
    route_seconds = {r: round(sec_per_route[r], 2) for r in KNOWN_ROUTES}

    return {
        "refs": n_refs,
        "targets": n_targets,
        "targets_remaining": n_remaining,
        "estimated_seconds": round(total_sec),
        "estimated_label": _format_elapsed(total_sec),
        "sec_per_target": round(sec_per, 2),
        "estimate_method": f"{frac_source}+{sec_source}",
        "estimate_detail": method_label,
        "route_fractions_pct": route_fractions_pct,
        "route_seconds": route_seconds,
        "fraction_source": frac_source,
        "model_load_seconds": round(model_load),
        "is_cpu": is_cpu,
        "max_refs_suggested": max_refs,
        "warning": warning,
        "has_existing_annotations": range_stats["has_existing_annotations"],
        "all_annotated": range_stats["all_annotated"],
        "targets_annotated": range_stats["targets_annotated"],
        "targets_total_in_range": range_stats["targets_total"],
        "refs_complete": range_stats["refs_complete"],
        "refs_partial": range_stats["refs_partial"],
        "force_reannotate": force_reannotate,
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


def build_batch_summary(
    *,
    targets_annotated: int,
    total_targets: int,
    routing_stats: dict[str, int],
    elapsed_label: str,
    first_review_index: int | None,
    references_in_batch: int,
) -> dict[str, Any]:
    human = routing_stats.get("human", 0)
    rejected = routing_stats.get("rejected", 0)
    needs_manual = human + rejected
    return {
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
            f"{human} human review, {rejected} rejected",
            f"Duration: {elapsed_label}",
        ],
    }


build_french_summary = build_batch_summary


def _find_first_review_index(data: list, start: int, end: int) -> int | None:
    for idx in range(start, end + 1):
        for target in data[idx].get("database") or []:
            route = target.get("cascade_route")
            if route in ("human", "rejected"):
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
) -> None:
    import numpy as np

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
                np,
                force_reannotate=force_reannotate,
                backup_path=backup_path,
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
    np,
    *,
    force_reannotate: bool = False,
    backup_path: str | None = None,
) -> None:
    from annotation_store import file_lock_for

    file_lock = file_lock_for(file_path)
    if not file_lock.acquire(blocking=False):
        _finish(error="File locked — another process is using this JSON.")
        return

    logger = _setup_logger(base_dir / "logs")
    try:
        from cascade.core import BatchCancelledError

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
                load_box["engine"] = eng
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
            "force_reannotate=%s backup=%s",
            original_filename,
            start_index,
            end_index,
            end_index - start_index + 1,
            targets_total,
            threshold,
            force_reannotate,
            backup_path or "none",
        )

        targets_annotated_count = 0
        pairs_evaluated = 0
        total_targets_evaluated = 0
        references_fully_annotated_count = 0
        refs_processed_in_batch = 0
        routing_stats = {
            "deberta_auto": 0,
            "deberta_ambiguous": 0,
            "consensus": 0,
            "rejected": 0,
            "human": 0,
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
                        f"target {i + 1}/{len(targets)} (DeBERTa / LLM running…)"
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
                if out.get("llm_error"):
                    target["llm_error"] = out["llm_error"]

                if route_name in {"deberta_auto", "deberta_ambiguous", "consensus"}:
                    target["related"] = out["related"]
                    target["model_confidence"] = round(
                        out["deberta_conf"] if route_name != "consensus" else out["llm_conf"],
                        4,
                    )
                    if route_name == "consensus":
                        target["similarity_annotation"] = round(out["llm_sim"], 4)
                    else:
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
                        sim = np.dot(emb_anchor, emb_target) / (
                            np.linalg.norm(emb_anchor) * np.linalg.norm(emb_target)
                        )
                        target["similarity_annotation"] = round(
                            max(0.0, min(1.0, float(sim))), 4
                        )
                    targets_annotated_count += 1
                    routing_stats[route_name] = routing_stats.get(route_name, 0) + 1
                else:
                    all_above_threshold = False
                    target["model_confidence"] = round(out["deberta_conf"], 4)
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
        summary = build_batch_summary(
            targets_annotated=targets_annotated_count,
            total_targets=total_targets_evaluated,
            routing_stats=routing_stats,
            elapsed_label=elapsed,
            first_review_index=first_review,
            references_in_batch=end_index - start_index + 1,
        )
        logger.info(
            "BATCH END file=%s annotated=%s/%s refs_fully_done=%s duration=%s "
            "routes=%s",
            original_filename,
            targets_annotated_count,
            total_targets_evaluated,
            references_fully_annotated_count,
            elapsed,
            routing_stats,
        )
        _finish(
            result={
                "ok": True,
                "message": summary["lines"][0] + " — " + summary["duration_label"],
                "summary_en": summary,
                "summary_fr": summary,
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
) -> dict:
    if end_index is None:
        end_index = len(data) - 1
    enriched = app_mod.enrich_data_with_status(original_filename, data)
    summary = build_batch_summary(
        targets_annotated=targets_annotated_count,
        total_targets=total_targets_evaluated,
        routing_stats=routing_stats,
        elapsed_label=elapsed_label,
        first_review_index=_find_first_review_index(data, start_index, end_index),
        references_in_batch=end_index - start_index + 1,
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
        "annotated_count": references_fully_annotated_count,
        "data": enriched,
        "processed_ids": list(processed_ids),
        "routing_stats": routing_stats,
        "elapsed_label": elapsed_label,
        "first_review_index": summary.get("first_review_index"),
    }
