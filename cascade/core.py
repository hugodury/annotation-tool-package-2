"""Logique cascade DeBERTa-v3 -> LLM (P3) -> humain / rejet."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.special import softmax
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import accuracy_score, f1_score

CASCADE_DIR = Path(__file__).resolve().parent
REPO = CASCADE_DIR.parent
VALID_LABELS = frozenset({"supporting", "against", "undetermined", "not_related", "dismissed"})
LABEL_TO_ID = {"against": 0, "not_related": 1, "supporting": 2, "undetermined": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class BatchCancelledError(Exception):
    """Levee lorsque l'utilisateur annule un batch Run Model en cours."""


CancelCheck = Callable[[], bool] | None

CASCADE_MODE_QWEN_ONLY = "qwen_only"
CASCADE_MODE_DEBERTA_QWEN = "deberta_qwen"
CASCADE_MODE_COMPARE = "compare"
VALID_CASCADE_MODES = frozenset({
    CASCADE_MODE_QWEN_ONLY,
    CASCADE_MODE_DEBERTA_QWEN,
    CASCADE_MODE_COMPARE,
})

AUTO_ANNOTATE_ROUTES = frozenset({
    "llm_auto",
    "deberta_auto",
    "deberta_ambiguous",
    "consensus",
    "compare_agree",
})

# Intervalle de verification de l'annulation cote thread principal (s).
OLLAMA_CANCEL_POLL_SEC = 0.2


def effective_related(route_out: dict) -> str | None:
    """Label final auto-annote pour un resultat de pipeline, ou None."""
    route = route_out.get("route")
    if route in {
        "llm_auto",
        "deberta_auto",
        "deberta_ambiguous",
        "consensus",
        "compare_agree",
    }:
        return route_out.get("related")
    return None


def compare_related(route_out: dict) -> str | None:
    """Label comparable en mode Compare (inclut la proposition LLM si human/rejected).

    Sans cela, cascade→human compte toujours comme désaccord même quand
    Qwen-only et Qwen-cascade ont le même llm_pred.
    """
    auto = effective_related(route_out)
    if auto is not None:
        return auto
    llm_pred = route_out.get("llm_pred")
    if isinstance(llm_pred, str) and llm_pred.strip():
        return llm_pred.strip()
    return None


def cascade_decision_source(cascade_out: dict) -> str:
    """Qui a produit le label final côté cascade: 'deberta' | 'qwen' | 'unknown'."""
    route = cascade_out.get("route")
    if route in {"deberta_auto", "deberta_ambiguous"}:
        return "deberta"
    if route == "consensus":
        return "qwen"
    if route in {"human", "rejected"}:
        if cascade_out.get("llm_pred") is not None:
            return "qwen"
        if cascade_out.get("deberta_pred") is not None:
            return "deberta"
    if cascade_out.get("llm_pred") is not None:
        return "qwen"
    if cascade_out.get("deberta_pred") is not None:
        return "deberta"
    return "unknown"


def cascade_source_label(source: str | None) -> str:
    if source == "deberta":
        return "DeBERTa"
    if source == "qwen":
        return "Qwen"
    return "?"


def build_pipeline_compare(qwen_out: dict, cascade_out: dict) -> dict:
    qwen_rel = compare_related(qwen_out)
    cascade_rel = compare_related(cascade_out)
    agree = (
        qwen_rel is not None
        and cascade_rel is not None
        and qwen_rel == cascade_rel
    )
    qwen_dur = qwen_out.get("duration_s")
    cascade_dur = cascade_out.get("duration_s")
    faster = None
    delta_s = None
    if isinstance(qwen_dur, (int, float)) and isinstance(cascade_dur, (int, float)):
        delta_s = round(abs(float(qwen_dur) - float(cascade_dur)), 3)
        if abs(float(qwen_dur) - float(cascade_dur)) < 1e-9:
            faster = "tie"
        else:
            faster = "cascade" if float(qwen_dur) > float(cascade_dur) else "qwen"
    source = cascade_decision_source(cascade_out)
    source_label = cascade_source_label(source)
    return {
        "qwen_only": {
            "route": qwen_out.get("route"),
            "related": qwen_rel,
            "llm_pred": qwen_out.get("llm_pred"),
            "llm_sim": qwen_out.get("llm_sim"),
            "llm_error": qwen_out.get("llm_error"),
            "duration_s": qwen_dur,
        },
        "deberta_qwen": {
            "route": cascade_out.get("route"),
            "related": cascade_rel,
            "deberta_pred": cascade_out.get("deberta_pred"),
            "deberta_conf": cascade_out.get("deberta_conf"),
            "deberta_sim": cascade_out.get("deberta_sim"),
            "similarity_annotation": cascade_out.get("similarity_annotation"),
            "llm_pred": cascade_out.get("llm_pred"),
            "llm_sim": cascade_out.get("llm_sim"),
            "llm_error": cascade_out.get("llm_error"),
            "duration_s": cascade_dur,
            "deberta_duration_s": cascade_out.get("deberta_duration_s"),
            "cascade_llm_duration_s": cascade_out.get("cascade_llm_duration_s"),
            "decision_source": source,
            "decision_source_label": source_label,
        },
        "agree": agree,
        "faster": faster,
        "duration_delta_s": delta_s,
        "cascade_source": source,
        "cascade_source_label": source_label,
    }


def normalize_cascade_mode(mode: str | None) -> str:
    value = (mode or CASCADE_MODE_QWEN_ONLY).strip().lower()
    if value in VALID_CASCADE_MODES:
        return value
    return CASCADE_MODE_QWEN_ONLY


def load_config() -> dict:
    with open(CASCADE_DIR / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return apply_env_overrides(cfg)


def apply_env_overrides(cfg: dict) -> dict:
    """Surcharges documentées dans .env.example (sans modifier config.json)."""
    cfg = json.loads(json.dumps(cfg))
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        cfg["ollama_host"] = host.rstrip("/")
    llm_tag = os.environ.get("OLLAMA_LLM_MODEL", "").strip()
    if llm_tag:
        llm = dict(cfg.get("llm") or {})
        llm["ollama"] = llm_tag
        cfg["llm"] = llm
    return cfg


def load_protocol() -> str:
    """Unique protocole P3 : cascade/protocol.md (complet et concis)."""
    return (CASCADE_DIR / "protocol.md").read_text(encoding="utf-8")


def load_few_shot() -> list[dict]:
    p = Path.cwd() / "few_shot.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))["examples"]
    return json.load(open(CASCADE_DIR / "few_shot.json", encoding="utf-8"))["examples"]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        m = re.search(r"0?\.\d+|1\.0|0", str(value))
        return float(m.group()) if m else None


_BRIDGE_STOP = frozenset(
    """
    a an the and or but if in on at to for of from with by as is are was were be been being
    this that these those it its they them their we our you your he she his her not no nor
    vs versus over under into about after before between during than then so such also more
    most other out up down off too very can could may might will would shall should do does
    did doing done have has had having what which who whom when where why how new best top
    says say said amid
    """.split()
)
_BRIDGE_MEDIA = frozenset(
    """
    forbes reuters bloomberg bbc cnn cnbc abc cbs nbc npr ap associated press globe mail
    washington post york times nytimes financial ft guardian independent daily newsweek
    news yahoo msn fox foxnews toronto global
    """.split()
)
# Acronymes 2–3 lettres exclus (mots anglais courants en majuscules dans les titres)
_ACRONYM_STOP = frozenset("of to in is as or an at by on up it we my be am if no so".split())


def _headline_core(text: str) -> str:
    text = re.sub(r"\s+-\s+[A-Za-z].*$", "", text or "")
    text = re.sub(r"\[.*?\]", " ", text)
    return text.strip()


def _content_tokens(text: str) -> set[str]:
    """Tokens utiles pour un pont topical (hors media / stopwords)."""
    raw = text or ""
    core = _headline_core(raw)
    out: set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z'-]+", core):
        wl = w.lower().strip("'")
        if len(wl) < 4 or wl in _BRIDGE_STOP or wl in _BRIDGE_MEDIA:
            continue
        out.add(wl)
    # Acronymes courts (AI, EV, EU, DEI, ECB…) — sinon « AI » était filtré (len < 4)
    for m in re.findall(r"\b[A-Z]{2,5}\b", core):
        wl = m.lower()
        if wl in _ACRONYM_STOP or wl in _BRIDGE_STOP or wl in _BRIDGE_MEDIA:
            continue
        out.add(wl)
    return out


def _is_person_directory_headline(text: str) -> bool:
    """Fiche personne type « Ron Corio - Forbes » (nom court, pas d'accroche news)."""
    core = _headline_core(text)
    if not core or len(core) > 55 or ":" in core or "?" in core:
        return False
    words = re.findall(r"[A-Za-z']+", core)
    if not (2 <= len(words) <= 5):
        return False
    # Pas de verbe / mot « news » fréquent
    lower = {w.lower() for w in words}
    if lower & {"says", "said", "after", "against", "amid", "could", "will", "best", "review"}:
        return False
    return True


def has_topical_bridge(anchor: str, target: str) -> bool:
    """Vrai si T_ref et T_n partagent une entité / thème lexical fort.

    Filet anti-crush : Starbucks↔Starbucks, AI↔AI, tips+taxes, lawyers↔lawyers,
    fiches personnes Forbes↔Forbes. Mets↔Nats / météo↔fundraising restent sans pont.
    """
    if _is_person_directory_headline(anchor) and _is_person_directory_headline(target):
        return True
    a = _content_tokens(anchor)
    b = _content_tokens(target)
    if a & b:
        return True
    for x in a:
        if len(x) < 5:
            continue
        for y in b:
            if len(y) < 5:
                continue
            if x.startswith(y) or y.startswith(x):
                return True
    return False


def enforce_label_sim_consistency(
    label: str,
    score: float | None,
    anchor: str | None = None,
    target: str | None = None,
) -> str:
    """Aligne related et similarity (garde-fou post-LLM / post-SBERT).

    Logique metier :
    - pont topical lexical + not_related  => undetermined (anti-crush T1/T2)
    - sim <= 0.2 sans pont               => not_related
    - not_related + sim > 0.3            => undetermined

    Ne convertit PAS undetermined→not_related sur la bande 0.2–0.3.
    Ne convertit JAMAIS vers supporting.
    """
    cfg = load_config()
    cutoff = float(cfg["score_not_related_cutoff"])
    nr_max = float(cfg.get("score_not_related_max", 0.3))
    label = label if label in VALID_LABELS else "undetermined"
    s = _safe_float(score)
    bridge = bool(anchor and target and has_topical_bridge(anchor, target))
    if bridge and label == "not_related":
        return "undetermined"
    if s is None:
        return label
    if s <= cutoff:
        return "undetermined" if bridge else "not_related"
    if label == "not_related" and s > nr_max:
        return "undetermined"
    return label


def postprocess_prediction(
    label: str,
    score: Any,
    anchor: str | None = None,
    target: str | None = None,
) -> tuple[str, float]:
    cfg = load_config()
    score_max = cfg["score_max"]
    label = label if label in VALID_LABELS else "undetermined"
    s = _safe_float(score)
    if s is None:
        s = 0.5
    s = max(0.0, min(score_max, s))
    label = enforce_label_sim_consistency(label, s, anchor=anchor, target=target)
    if label == "undetermined" and anchor and target and has_topical_bridge(anchor, target):
        # Evite sim 0.1 + undetermined (incoherent) apres correction anti-crush.
        s = max(s, 0.35)
    return label, round(min(score_max, s), 3)


def parse_llm_json(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    # Bloc ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Premier objet JSON equilibre
    start = text.find("{")
    if start >= 0:
        depth = 0
        for pos in range(start, len(text)):
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : pos + 1])
                    except json.JSONDecodeError:
                        break
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def build_prompt(prompt_id: str, anchor: str, target: str, few_shot: list[dict]) -> str:
    pair = f"T_ref: {anchor}\nT_n: {target}"
    if prompt_id == "P0":
        return (
            "Annotate the relationship between these news headlines.\n"
            f"{pair}\n"
            'Output JSON only: {"related":"...","similarity_annotation":0.0,"confidence":0.0}'
        )
    examples = ""
    for ex in few_shot:
        examples += (
            f"\nT_ref: {ex['text_anchor']}\nT_n: {ex['text_target']}\n"
            f'-> {{"related":"{ex["related"]}","similarity_annotation":{ex["similarity_annotation"]}}}\n'
        )
    if prompt_id == "P2":
        return (
            "Training examples (apply the same logic):\n"
            f"{examples}\n---\nAnnotate:\n{pair}\nJSON only."
        )
    protocol = load_protocol()
    if prompt_id == "P1":
        return f"{protocol}\n\n---\nAnnotate:\n{pair}\nJSON only."
    # P3: protocole concis + few-shot (garde-fous A/B/C en tête des exemples)
    return (
        f"{protocol}\n\n"
        "Few-shot examples (mirror these patterns; first three = guardrails A/B/C):\n"
        f"{examples}\n---\n"
        "HARD: (1) Shared brand/acronym/topic (Starbucks, AI, tips tax) → NEVER not_related. "
        "(2) Forbes-style person directory pages (Name - Outlet) → undetermined, not not_related. "
        "(3) not_related ONLY if zero topical bridge (Mets vs Nats, fundraising vs weather). "
        "Same claim → supporting. sim > 0.3 → never not_related.\n"
        f"Annotate:\n{pair}\nJSON only."
    )


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    timeout: int = 600,
    keep_alive: str = "30m",
    should_cancel: CancelCheck = None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    parts: list[str] = []

    def _stream() -> None:
        # Lecture bloquante ligne par ligne. On NE fixe PAS de timeout de lecture
        # sous-seconde: sur CPU l'intervalle entre deux tokens depasse souvent la
        # seconde (surtout au 1er appel, chargement du modele), ce qui corrompait
        # le flux HTTP ("cannot read from timed out object") et faisait echouer
        # l'appel a tort. urlopen(timeout) borne toujours la duree totale.
        resp = urllib.request.urlopen(req, timeout=timeout)
        try:
            for raw_line in resp:
                if should_cancel and should_cancel():
                    break
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunk = obj.get("response")
                if chunk:
                    parts.append(chunk)
                if obj.get("done"):
                    break
        finally:
            try:
                resp.close()
            except OSError:
                pass

    # Sans annulation demandee: lecture directe (bloquante).
    if not should_cancel:
        _stream()
        return "".join(parts)

    # Avec annulation: lecture dans un thread worker, l'annulation est verifiee
    # cote thread principal toutes les OLLAMA_CANCEL_POLL_SEC secondes.
    err: list[BaseException] = []

    def _work() -> None:
        try:
            _stream()
        except BaseException as e:  # noqa: BLE001 - propage au thread principal
            err.append(e)

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    while worker.is_alive():
        if should_cancel():
            raise BatchCancelledError("Batch cancelled by user.")
        worker.join(OLLAMA_CANCEL_POLL_SEC)
    if err:
        raise err[0]
    return "".join(parts)


class CascadeEngine:
    def __init__(self):
        self.cfg = load_config()
        self.llm_cfg = self.cfg["llm"]
        self.few_shot = load_few_shot()
        # DeBERTa charge a la demande (mode deberta_qwen). SBERT reste utilise pour la similarite.
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.deberta = None

    def ensure_deberta(self) -> None:
        if self.deberta is None:
            self._load_deberta()

    def _load_deberta(self):
        model_path = REPO / self.cfg["deberta_model_path"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele DeBERTa manquant: {model_path}. "
                "Executez: python scripts/setup.py"
            )
        self.deberta = CrossEncoder(str(model_path), device=self.device)

    def deberta_predict(
        self,
        anchor: str,
        target: str,
        should_cancel: CancelCheck = None,
    ) -> tuple[str, float, float]:
        if should_cancel and should_cancel():
            raise BatchCancelledError("Batch cancelled by user.")
        self.ensure_deberta()

        if not should_cancel:
            return self._deberta_predict_impl(anchor, target)

        holder: dict[str, Any] = {}
        err: list[BaseException] = []

        def _work() -> None:
            try:
                holder["result"] = self._deberta_predict_impl(anchor, target)
            except BaseException as e:
                err.append(e)

        worker = threading.Thread(target=_work, daemon=True)
        worker.start()
        while worker.is_alive():
            if should_cancel():
                raise BatchCancelledError("Batch cancelled by user.")
            worker.join(0.15)
        if err:
            raise err[0]
        return holder["result"]

    def _deberta_predict_impl(self, anchor: str, target: str) -> tuple[str, float, float]:
        logits = self.deberta.predict([[anchor, target]], convert_to_numpy=True, show_progress_bar=False)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        probs = softmax(logits, axis=1)[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        label = ID_TO_LABEL[pred_idx]
        sim = float(self.cfg["similarity_defaults"].get(label, 0.5))
        return label, conf, sim

    def _active_llm_tag(self) -> str:
        return os.environ.get("OLLAMA_LLM_MODEL") or self.llm_cfg["ollama"]

    def llm_predict(
        self,
        anchor: str,
        target: str,
        should_cancel: CancelCheck = None,
    ) -> tuple[str, float, float, str | None]:
        prompt_id = self.llm_cfg["prompt"]
        prompt = build_prompt(prompt_id, anchor, target, self.few_shot)
        model = self._active_llm_tag()
        inf = self.cfg["inference"]
        retries = int(inf.get("llm_retries", 2))
        last_err: str | None = None
        for attempt in range(retries + 1):
            if should_cancel and should_cancel():
                raise BatchCancelledError("Batch cancelled by user.")
            try:
                raw = ollama_generate(
                    self.cfg["ollama_host"],
                    model,
                    prompt,
                    inf["temperature"],
                    inf["num_predict"],
                    timeout=int(inf.get("timeout", 600)),
                    keep_alive=str(inf.get("keep_alive", "30m")),
                    should_cancel=should_cancel,
                )
                parsed = parse_llm_json(raw) or {}
                label, score = postprocess_prediction(
                    parsed.get("related", "undetermined"),
                    parsed.get("similarity_annotation"),
                    anchor=anchor,
                    target=target,
                )
                llm_conf = float(_safe_float(parsed.get("confidence")) or 0.5)
                return label, score, llm_conf, None
            except BatchCancelledError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = str(e)
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 5))
        return "undetermined", 0.0, 0.5, last_err

    def route(
        self,
        anchor: str,
        target: str,
        tau_auto: float | None = None,
        should_cancel: CancelCheck = None,
        cascade_mode: str | None = None,
    ) -> dict:
        mode = normalize_cascade_mode(cascade_mode)
        if mode == CASCADE_MODE_DEBERTA_QWEN:
            return self._route_deberta_qwen(
                anchor, target, tau_auto=tau_auto, should_cancel=should_cancel
            )
        if mode == CASCADE_MODE_COMPARE:
            return self._route_compare(
                anchor, target, tau_auto=tau_auto, should_cancel=should_cancel
            )
        return self._route_qwen_only(anchor, target, should_cancel=should_cancel)

    def _route_compare(
        self,
        anchor: str,
        target: str,
        tau_auto: float | None = None,
        should_cancel: CancelCheck = None,
    ) -> dict:
        """Execute les deux pipelines et compare les labels auto-annotables."""
        qwen_start = time.monotonic()
        qwen_out = self._route_qwen_only(anchor, target, should_cancel=should_cancel)
        qwen_out["duration_s"] = round(time.monotonic() - qwen_start, 3)

        cascade_start = time.monotonic()
        cascade_out = self._route_deberta_qwen(
            anchor, target, tau_auto=tau_auto, should_cancel=should_cancel
        )
        cascade_out["duration_s"] = round(time.monotonic() - cascade_start, 3)

        compare = build_pipeline_compare(qwen_out, cascade_out)
        result = self._empty_route_result()
        result["pipeline_compare"] = compare
        result["qwen_only_result"] = qwen_out
        result["deberta_qwen_result"] = cascade_out

        if compare["agree"]:
            agreed = compare["qwen_only"]["related"]
            result.update(
                route="compare_agree",
                related=agreed,
                similarity_annotation=qwen_out.get("llm_sim"),
                llm_pred=qwen_out.get("llm_pred"),
                llm_conf=qwen_out.get("llm_conf"),
                llm_sim=qwen_out.get("llm_sim"),
                deberta_pred=cascade_out.get("deberta_pred"),
                deberta_conf=cascade_out.get("deberta_conf"),
            )
        else:
            result.update(
                route="compare_disagree",
                related=None,
                similarity_annotation=None,
                llm_pred=qwen_out.get("llm_pred"),
                llm_conf=qwen_out.get("llm_conf"),
                deberta_pred=cascade_out.get("deberta_pred"),
                deberta_conf=cascade_out.get("deberta_conf"),
                requires_human_review=True,
            )
        return result

    def _route_qwen_only(
        self,
        anchor: str,
        target: str,
        should_cancel: CancelCheck = None,
    ) -> dict:
        """Chaque cible passe par le LLM (Qwen P3)."""
        result = self._empty_route_result()

        llm_label, llm_score, llm_conf, err = self.llm_predict(
            anchor, target, should_cancel=should_cancel
        )
        result["llm_pred"] = llm_label
        result["llm_conf"] = round(llm_conf, 4)
        result["llm_sim"] = llm_score

        if err:
            result["llm_error"] = err
            result.update(
                route="human",
                requires_human_review=True,
            )
            return result

        result.update(
            route="llm_auto",
            related=llm_label,
            similarity_annotation=llm_score,
        )
        return result

    def _route_deberta_qwen(
        self,
        anchor: str,
        target: str,
        tau_auto: float | None = None,
        should_cancel: CancelCheck = None,
    ) -> dict:
        """Cascade DeBERTa -> Qwen (P3) -> humain / rejet."""
        if tau_auto is None:
            tau_auto = self.cfg["tau_deberta_auto"]
        tau_reject = self.cfg["tau_disagreement_reject"]

        deberta_start = time.monotonic()
        deberta_label, deberta_conf, deberta_sim = self.deberta_predict(
            anchor, target, should_cancel=should_cancel
        )
        deberta_duration_s = round(time.monotonic() - deberta_start, 3)

        result = self._empty_route_result()
        result.update(
            deberta_pred=deberta_label,
            deberta_conf=round(deberta_conf, 4),
            deberta_sim=deberta_sim,
            deberta_duration_s=deberta_duration_s,
            cascade_llm_duration_s=0.0,
        )

        if deberta_conf >= tau_auto:
            result.update(
                route="deberta_auto",
                related=deberta_label,
                similarity_annotation=deberta_sim,
            )
            return result

        llm_target_classes = frozenset(
            self.cfg.get("llm_target_classes", ["supporting", "undetermined"])
        )

        if deberta_label not in llm_target_classes:
            result.update(
                route="deberta_ambiguous",
                related=deberta_label,
                similarity_annotation=deberta_sim,
            )
            return result

        llm_start = time.monotonic()
        llm_label, llm_score, llm_conf, err = self.llm_predict(
            anchor, target, should_cancel=should_cancel
        )
        result["cascade_llm_duration_s"] = round(time.monotonic() - llm_start, 3)
        result["llm_pred"] = llm_label
        result["llm_conf"] = round(llm_conf, 4)
        result["llm_sim"] = llm_score

        if err:
            result["llm_error"] = err
            result.update(route="human", requires_human_review=True)
            return result

        if llm_label == deberta_label:
            result.update(
                route="consensus",
                related=llm_label,
                similarity_annotation=llm_score,
            )
            return result

        disagree_conf = max(deberta_conf, llm_conf)
        if disagree_conf >= tau_reject:
            result.update(route="rejected", rejected=True)
        else:
            result.update(
                route="human",
                related=llm_label,
                similarity_annotation=llm_score,
                requires_human_review=True,
            )
        return result

    def _empty_route_result(self) -> dict:
        return {
            "llm_model": self.llm_cfg["label"],
            "llm_prompt": self.llm_cfg["prompt"],
            "deberta_pred": None,
            "deberta_conf": None,
            "deberta_sim": None,
            "llm_pred": None,
            "llm_conf": None,
            "llm_sim": None,
            "route": None,
            "related": None,
            "similarity_annotation": None,
            "requires_human_review": False,
            "rejected": False,
            "error": None,
        }


def annotate_pairs(
    pairs: list[dict],
    anchor_key: str = "text_anchor",
    target_key: str = "text_target",
) -> tuple[list[dict], dict]:
    engine = CascadeEngine()
    routes = {
        "llm_auto": 0,
        "deberta_auto": 0,
        "deberta_ambiguous": 0,
        "consensus": 0,
        "rejected": 0,
        "human": 0,
    }
    y_true, y_pred = [], []

    for pair in pairs:
        anchor = pair[anchor_key]
        target = pair[target_key]
        gold = pair.get("related_gold", pair.get("related"))
        out = engine.route(anchor, target)
        pair["cascade"] = out
        if out["related"] is not None and not out.get("requires_human_review") and not out.get("rejected"):
            pair["related"] = out["related"]
            pair["similarity_annotation"] = out["similarity_annotation"]
        routes[out["route"]] = routes.get(out["route"], 0) + 1
        if gold and out["related"] and not out.get("requires_human_review") and not out.get("rejected"):
            y_true.append(gold)
            y_pred.append(out["related"])

    stats = {"routing": routes, "llm": engine.llm_cfg["label"]}
    if y_true:
        stats["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
        stats["f1_macro"] = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
    return pairs, stats
