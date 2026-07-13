"""Run Model en arrière-plan : progression, logs, sauvegarde incrémentale."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_cancel_requested = False
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


def _setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_model_{datetime.now().strftime('%Y%m%d')}.log"
    logger = logging.getLogger("run_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    return logger


def effective_run_config(cfg: dict, is_cpu: bool) -> dict:
    cfg = deepcopy(cfg)
    run_cfg = cfg.setdefault("run_model", {})
    cpu_slow = run_cfg.get("cpu_slow") or {}
    if is_cpu and cpu_slow.get("enabled", True):
        timeout = int(cpu_slow.get("inference_timeout_sec", 900))
        cfg.setdefault("inference", {})["timeout"] = timeout
    return cfg


def estimate_batch(
    data: list,
    start_index: int,
    end_index: int,
    *,
    is_cpu: bool,
    cfg: dict,
) -> dict[str, Any]:
    run_cfg = cfg.get("run_model", {})
    cpu_slow = run_cfg.get("cpu_slow") or {}
    n_refs = max(0, end_index - start_index + 1)
    n_targets = 0
    for idx in range(start_index, end_index + 1):
        n_targets += len(data[idx].get("database") or [])
    sec_per = float(cpu_slow.get("sec_per_target_estimate", 5 if is_cpu else 1.5))
    total_sec = n_targets * sec_per
    max_refs = int(cpu_slow.get("max_refs_suggested", 50)) if is_cpu else None
    warning = None
    if is_cpu and max_refs and n_refs > max_refs:
        warning = (
            f"Sur CPU, plage conseillee : {max_refs} references max par batch "
            f"(vous en avez {n_refs})."
        )
    return {
        "refs": n_refs,
        "targets": n_targets,
        "estimated_seconds": round(total_sec),
        "estimated_label": _format_elapsed(total_sec),
        "is_cpu": is_cpu,
        "max_refs_suggested": max_refs,
        "warning": warning,
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


def build_french_summary(
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
        "title": "Annotation automatique terminee",
        "prefilled_label": f"{targets_annotated} / {total_targets} cibles pre-remplies",
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
            f"{targets_annotated} cibles pre-remplies sur {total_targets}",
            f"{routing_stats.get('deberta_auto', 0)} DeBERTa auto, "
            f"{routing_stats.get('deberta_ambiguous', 0)} DeBERTa ambigu, "
            f"{routing_stats.get('consensus', 0)} consensus LLM",
            f"{human} revue humaine, {rejected} rejetees",
            f"Duree : {elapsed_label}",
        ],
    }


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
        _state["progress"]["message"] = "Annulation demandee…"
        return True


def _cancelled() -> bool:
    with _lock:
        return _cancel_requested


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
            "message": "Chargement des modeles ML…",
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
) -> None:
    from annotation_store import file_lock_for

    file_lock = file_lock_for(file_path)
    if not file_lock.acquire(blocking=False):
        _finish(error="Fichier verrouille — un autre traitement est en cours.")
        return

    logger = _setup_logger(base_dir / "logs")
    try:
        from cascade.core import BatchCancelledError

        batch_start = time.monotonic()
        _set_progress(message="Chargement des modeles ML (peut prendre 1-3 min sur CPU)…")
        cfg = app_mod._load_cascade_config()
        run_cfg = cfg.get("run_model", {})
        save_every = max(1, int(run_cfg.get("save_every_n_refs", 3)))
        no_annotation_timeout = int(run_cfg.get("no_annotation_timeout", 360))

        report = app_mod.build_report(base_dir, cfg)
        is_cpu = report.get("hardware", {}).get("gpu", {}).get("device") == "cpu"
        cfg = effective_run_config(cfg, is_cpu)

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

        loader = threading.Thread(target=_load_models, daemon=True)
        loader.start()
        while loader.is_alive():
            if _cancelled():
                logger.info("Batch annule pendant chargement des modeles")
                _finish(error="Batch annule.")
                return
            loader.join(0.5)

        if load_errors:
            logger.error("Echec chargement modeles: %s", load_errors[0])
            _finish(error=str(load_errors[0]))
            return

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
        _set_progress(targets_total=targets_total, message="Annotation en cours…")
        logger.info(
            "DEBUT batch %s indices %s-%s (%s refs, %s cibles)",
            original_filename,
            start_index,
            end_index,
            end_index - start_index + 1,
            targets_total,
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
                    partial_error="Batch annule par l'utilisateur.",
                    start_index=start_index,
                    end_index=end_index,
                ),
                error="Batch annule.",
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

            if app_mod.reference_status_from_item(item) == "complete":
                targets_skip = item.get("database") or []
                refs_processed_in_batch += 1
                pairs_evaluated += len(targets_skip)
                _set_progress(
                    refs_done=refs_processed_in_batch,
                    targets_done=pairs_evaluated,
                    message=f"Reference {ref_num} deja complete (ignoree)",
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

                if app_mod._target_is_annotated(target):
                    pairs_evaluated += 1
                    _set_progress(targets_done=pairs_evaluated)
                    continue

                if (
                    targets_annotated_count == 0
                    and time.monotonic() - batch_start > no_annotation_timeout
                ):
                    save_checkpoint()
                    elapsed = _format_elapsed(time.monotonic() - batch_start)
                    logger.error("Timeout zero annotation apres %ss", no_annotation_timeout)
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
                                f"Aucune annotation apres {no_annotation_timeout}s."
                            ),
                            start_index=start_index,
                            end_index=end_index,
                        ),
                        error=f"Aucune annotation apres {no_annotation_timeout}s.",
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
                        f"cible {i + 1}/{len(targets)} (DeBERTa / LLM en cours…)"
                    ),
                )
                logger.info(
                    "debut cible ref=%s pair=%s/%s",
                    idx,
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
                        "Batch annule pendant cible ref=%s pair=%s/%s",
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
                        emb_anchor = sbert.encode([anchor_text], show_progress_bar=False)[0]
                        emb_target = sbert.encode([target_text], show_progress_bar=False)[0]
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
                logger.info(
                    "ref=%s idx=%s/%s route=%s pair=%s/%s duree=%ss%s",
                    idx,
                    ref_num,
                    end_index - start_index + 1,
                    route_name,
                    i + 1,
                    len(targets),
                    pair_sec,
                    f" llm_error={out.get('llm_error')}" if out.get("llm_error") else "",
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
                logger.info("Checkpoint JSON+DB apres ref index %s", idx)

        save_checkpoint()
        data = app_mod.enrich_data_with_status(original_filename, data)
        processed_ids = {
            str(item.get("news_id"))
            for item in data
            if app_mod.reference_status_from_item(item) == "complete"
        }
        elapsed = _format_elapsed(time.monotonic() - batch_start)
        first_review = _find_first_review_index(data, start_index, end_index)
        summary = build_french_summary(
            targets_annotated=targets_annotated_count,
            total_targets=total_targets_evaluated,
            routing_stats=routing_stats,
            elapsed_label=elapsed,
            first_review_index=first_review,
            references_in_batch=end_index - start_index + 1,
        )
        logger.info(
            "FIN batch %s/%s cibles, duree %s",
            targets_annotated_count,
            total_targets_evaluated,
            elapsed,
        )
        _finish(
            result={
                "ok": True,
                "message": summary["lines"][0] + " — " + summary["duration_label"],
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
    summary = build_french_summary(
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
            f"Run Model interrompu — {targets_annotated_count} cible(s) sauvegardee(s). "
            f"Duree : {elapsed_label}."
        ),
        "summary_fr": summary,
        "annotated_count": references_fully_annotated_count,
        "data": enriched,
        "processed_ids": list(processed_ids),
        "routing_stats": routing_stats,
        "elapsed_label": elapsed_label,
        "first_review_index": summary.get("first_review_index"),
    }
