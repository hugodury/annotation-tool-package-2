"""Logique cascade DeBERTa-v3 -> LLM (P3) -> humain / rejet."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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


def load_config() -> dict:
    with open(CASCADE_DIR / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_protocol() -> str:
    for name in ["protocole.md", "protocol.md"]:
        p = Path.cwd() / name
        if p.exists():
            return p.read_text(encoding="utf-8")
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


def postprocess_prediction(label: str, score: Any) -> tuple[str, float]:
    cfg = load_config()
    score_max = cfg["score_max"]
    cutoff = cfg["score_not_related_cutoff"]
    label = label if label in VALID_LABELS else "undetermined"
    s = _safe_float(score)
    if s is None:
        s = 0.5
    s = max(0.0, min(score_max, s))
    if s <= cutoff:
        label = "not_related"
    return label, round(s, 3)


def parse_llm_json(text: str) -> dict | None:
    text = text.strip()
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
    return (
        f"{protocol}\n\nFew-shot examples:\n{examples}\n---\nAnnotate:\n{pair}\nJSON only."
    )


def ollama_generate(host: str, model: str, prompt: str, temperature: float, num_predict: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["response"]


class CascadeEngine:
    def __init__(self):
        self.cfg = load_config()
        self.llm_cfg = self.cfg["llm"]
        self.few_shot = load_few_shot()
        self._load_deberta()

    def _load_deberta(self):
        model_path = REPO / self.cfg["deberta_model_path"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele DeBERTa manquant: {model_path}. "
                "Executez: python scripts/setup.py"
            )
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = device
        self.deberta = CrossEncoder(str(model_path), device=device)

    def deberta_predict(self, anchor: str, target: str) -> tuple[str, float, float]:
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

    def llm_predict(self, anchor: str, target: str) -> tuple[str, float, float, str | None]:
        prompt_id = self.llm_cfg["prompt"]
        prompt = build_prompt(prompt_id, anchor, target, self.few_shot)
        model = self._active_llm_tag()
        try:
            raw = ollama_generate(
                self.cfg["ollama_host"],
                model,
                prompt,
                self.cfg["inference"]["temperature"],
                self.cfg["inference"]["num_predict"],
            )
        except (urllib.error.URLError, TimeoutError) as e:
            return "undetermined", 0.0, 0.5, str(e)
        parsed = parse_llm_json(raw) or {}
        label, score = postprocess_prediction(
            parsed.get("related", "undetermined"),
            parsed.get("similarity_annotation"),
        )
        llm_conf = float(_safe_float(parsed.get("confidence")) or 0.5)
        return label, score, llm_conf, None

    def route(self, anchor: str, target: str, tau_auto: float | None = None) -> dict:
        if tau_auto is None:
            tau_auto = self.cfg["tau_deberta_auto"]
        tau_reject = self.cfg["tau_disagreement_reject"]

        deberta_label, deberta_conf, deberta_sim = self.deberta_predict(anchor, target)

        result = {
            "llm_model": self.llm_cfg["label"],
            "llm_prompt": self.llm_cfg["prompt"],
            "deberta_pred": deberta_label,
            "deberta_conf": round(deberta_conf, 4),
            "deberta_sim": deberta_sim,
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

        if deberta_conf >= tau_auto:
            result.update(
                route="deberta_auto",
                related=deberta_label,
                similarity_annotation=deberta_sim,
            )
            return result

        llm_target_classes = frozenset(self.cfg.get("llm_target_classes", ["supporting", "undetermined"]))

        if deberta_label not in llm_target_classes:
            result.update(
                route="deberta_ambiguous",
                related=deberta_label,
                similarity_annotation=deberta_sim,
            )
            return result

        llm_label, llm_score, llm_conf, err = self.llm_predict(anchor, target)
        result["llm_pred"] = llm_label
        result["llm_conf"] = round(llm_conf, 4)
        result["llm_sim"] = llm_score
        result["error"] = err

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


def annotate_pairs(
    pairs: list[dict],
    anchor_key: str = "text_anchor",
    target_key: str = "text_target",
) -> tuple[list[dict], dict]:
    engine = CascadeEngine()
    routes = {
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
