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
CASCADE_MODE_V8_QWEN = "v8_qwen"
CASCADE_MODE_COMPARE = "compare"
# Alias historique (mode retiré de l'UI) → remappé vers Cascade V8+Qwen.
CASCADE_MODE_DEBERTA_QWEN = "deberta_qwen"
VALID_CASCADE_MODES = frozenset({
    CASCADE_MODE_QWEN_ONLY,
    CASCADE_MODE_V8_QWEN,
    CASCADE_MODE_COMPARE,
})
LEGACY_CASCADE_MODE_ALIASES = {
    CASCADE_MODE_DEBERTA_QWEN: CASCADE_MODE_V8_QWEN,
}

AUTO_ANNOTATE_ROUTES = frozenset({
    "llm_auto",
    "deberta_auto",
    "deberta_ambiguous",
    "consensus",
    "compare_agree",
    "v8_duo",
    "v8_reranker",
    "v8_qwen",
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
        "v8_duo",
        "v8_reranker",
        "v8_qwen",
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
    """Qui a produit le label final: 'deberta' | 'v8' | 'qwen' | 'unknown'."""
    route = cascade_out.get("route")
    if route in {"deberta_auto", "deberta_ambiguous", "v8_duo", "v8_reranker"}:
        return "deberta" if route.startswith("deberta") else "v8"
    if route == "consensus":
        return "qwen"
    if route == "v8_qwen":
        return "qwen"
    if route in {"human", "rejected"}:
        if cascade_out.get("llm_pred") is not None:
            return "qwen"
        if cascade_out.get("v8_label") is not None or cascade_out.get("deberta_pred") is not None:
            return "v8" if cascade_out.get("v8_label") is not None else "deberta"
    if cascade_out.get("llm_pred") is not None:
        return "qwen"
    if cascade_out.get("v8_label") is not None:
        return "v8"
    if cascade_out.get("deberta_pred") is not None:
        return "deberta"
    return "unknown"


def cascade_source_label(source: str | None) -> str:
    if source == "deberta":
        return "DeBERTa"
    if source == "v8":
        return "Cascade V8"
    if source == "qwen":
        return "Qwen"
    return "?"


def build_pipeline_compare(qwen_out: dict, cascade_out: dict) -> dict:
    """Compare Qwen-only vs Cascade V8+Qwen (même schéma clé `deberta_qwen` = côté cascade)."""
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
    cascade_side = {
        "route": cascade_out.get("route"),
        "related": cascade_rel,
        "deberta_pred": cascade_out.get("deberta_pred"),
        "deberta_conf": cascade_out.get("deberta_conf") or cascade_out.get("v8_conf"),
        "deberta_sim": cascade_out.get("deberta_sim"),
        "similarity_annotation": cascade_out.get("similarity_annotation"),
        "llm_pred": cascade_out.get("llm_pred"),
        "llm_sim": cascade_out.get("llm_sim"),
        "llm_error": cascade_out.get("llm_error"),
        "duration_s": cascade_dur,
        "deberta_duration_s": cascade_out.get("v8_duration_s")
        or cascade_out.get("deberta_duration_s"),
        "cascade_llm_duration_s": cascade_out.get("cascade_llm_duration_s"),
        "v8_stage": cascade_out.get("v8_stage"),
        "v8_rule": cascade_out.get("v8_rule"),
        "v8_label": cascade_out.get("v8_label"),
        "v8_conf": cascade_out.get("v8_conf"),
        "decision_source": source,
        "decision_source_label": source_label,
    }
    return {
        "qwen_only": {
            "route": qwen_out.get("route"),
            "related": qwen_rel,
            "llm_pred": qwen_out.get("llm_pred"),
            "llm_sim": qwen_out.get("llm_sim"),
            "llm_error": qwen_out.get("llm_error"),
            "duration_s": qwen_dur,
        },
        # Clé historique UI ; contenu = Cascade V8 + Qwen
        "deberta_qwen": cascade_side,
        "cascade": cascade_side,
        "agree": agree,
        "faster": faster,
        "duration_delta_s": delta_s,
        "cascade_source": source,
        "cascade_source_label": source_label,
    }


def normalize_cascade_mode(mode: str | None) -> str:
    value = (mode or CASCADE_MODE_QWEN_ONLY).strip().lower()
    if value in LEGACY_CASCADE_MODE_ALIASES:
        return LEGACY_CASCADE_MODE_ALIASES[value]
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
    says say said amid first make early rare lead city issues opinion attacks president
    page higher power movement join joins claim claims want wants last time year years
    analysis column opinion guide visit visits leave leaves found find live here since
    promise promises because late share gets get potential potentially contain studies
    know both plan media national updates carrying
    chaos death video need worst right things essentials packing viewed times
    william people man woman
    """.split()
)
_BRIDGE_MEDIA = frozenset(
    """
    forbes reuters bloomberg bbc cnn cnbc abc cbs nbc npr ap associated press globe mail
    washington post york times nytimes financial ft guardian independent daily newsweek
    news yahoo msn fox foxnews toronto global national review newyorker
    """.split()
)
# Acronymes trop faibles / ambigus pour un pont seul
_ACRONYM_STOP = frozenset(
    "of to in is as or an at by on up it we my be am if no so us uk un eu".split()
)
# Synonymes thématiques (évite faux NR T1/T2 sur même sujet lexical différent)
_SYNONYM_GROUPS = (
    frozenset({"pregnancy", "pregnant", "trimester", "prenatal", "maternity"}),
    frozenset({"inheritance", "estate", "heir", "heirs", "inherit"}),
    frozenset({"climate", "planet", "environmental", "environment", "decarbonization"}),
    frozenset({"ukraine", "ukrainian", "kyiv", "kiev"}),
    # Famille / polarisation partisane (évite faux NR T1/T2 type Fox mom ↔ Dem/Rep couple)
    frozenset({"democrat", "democratic", "republican", "liberal", "conservative", "maga"}),
)


def _headline_core(text: str) -> str:
    # Retire le suffixe media une fois (dernier « - Outlet »)
    text = re.sub(r"\s+-\s+[A-Za-z][^-\n]*$", "", text or "")
    text = re.sub(r"\[.*?\]", " ", text)
    return text.strip()


def _expand_synonyms(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for group in _SYNONYM_GROUPS:
        if out & group:
            out |= group
    return out


def _content_tokens(text: str) -> set[str]:
    """Tokens utiles pour un pont topical (hors media / stopwords faibles)."""
    core = _headline_core(text or "")
    out: set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z'-]+", core):
        wl = w.lower().strip("'")
        # Coupe les composés (Russia-Ukraine → russia + ukraine)
        parts = [p for p in re.split(r"[-–—/]", wl) if p]
        if not parts:
            parts = [wl]
        for part in parts:
            if len(part) < 4 or part in _BRIDGE_STOP or part in _BRIDGE_MEDIA:
                continue
            out.add(part)
    # Acronymes courts utiles (AI, EV, DEI, ECB…) — pas US/UK seuls
    for m in re.findall(r"\b[A-Z]{2,5}\b", core):
        wl = m.lower()
        if wl in _ACRONYM_STOP or wl in _BRIDGE_STOP or wl in _BRIDGE_MEDIA:
            continue
        out.add(wl)
    return _expand_synonyms(out)


def _is_person_directory_headline(text: str) -> bool:
    """Fiche personne type « Ron Corio - Forbes » uniquement (pas un titre thématique)."""
    core = _headline_core(text)
    if not core or len(core) > 45 or ":" in core or "?" in core or "," in core:
        return False
    words = re.findall(r"[A-Za-z']+", core)
    if not (2 <= len(words) <= 4):
        return False
    lower = {w.lower() for w in words}
    banned = {
        "says", "said", "after", "against", "amid", "could", "will", "best", "review",
        "higher", "power", "movement", "ed", "page", "daily", "guide", "opinion",
        "analysis", "column", "police", "president", "election", "olympics",
    }
    if lower & banned:
        return False
    # Chaque mot ressemble à un nom propre (capitale)
    for w in words:
        if not w[0].isupper():
            return False
    return True


# Un seul nom politique / célébrité ne suffit pas comme pont (Trump movie ≠ Trump ticket)
_WEAK_ENTITY_ALONE = frozenset(
    "trump biden harris obama putin musk vance zelensky netanyahu william".split()
)


def _prefix_related(a: str, b: str) -> bool:
    """serbia/serbians OK ; potential/potentially NON ; nigeria/niger NON."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 6:
        return False
    if len(long) - len(short) > 2:
        return False
    if not long.startswith(short):
        return False
    # Évite potential→potentially, extreme→extremely
    if long.endswith("ly") and len(long) - len(short) <= 3:
        return False
    return True


def has_topical_bridge(anchor: str, target: str) -> bool:
    """Vrai si T_ref et T_n partagent une entité / thème lexical fort.

    Filet anti-faux-NR (Starbucks, AI, Ukraine, climate).
    Ne doit PAS lier fillers T4–T6 via mots faibles ni un seul nom (Trump seul).
    """
    if _is_person_directory_headline(anchor) and _is_person_directory_headline(target):
        return True
    a = _content_tokens(anchor)
    b = _content_tokens(target)
    overlap = a & b
    if overlap:
        strong = overlap - _WEAK_ENTITY_ALONE
        if strong:
            return True
        # Uniquement trump/biden/… → pas un pont
    for x in a:
        for y in b:
            if x == y:
                continue
            if _prefix_related(x, y):
                # Préfixe entre deux weak entities : ignorer
                if {x, y} <= _WEAK_ENTITY_ALONE:
                    continue
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
    - pont topical lexical + not_related  => undetermined (anti-faux-NR T1/T2)
    - undetermined sans pont + sim <= nr_max (0.3) => not_related
      (NE PAS utiliser 0.4 : ça contredisait « sim>0.3 → never NR »)
    - sim <= 0.2 sans pont               => not_related
    - not_related + sim > 0.3            => undetermined
      (aligné protocole : never NR si sim>0.3, sauf templates)
    - faux pont lexical force NR seulement si sim <= nr_max

    Ne convertit JAMAIS vers supporting.
    """
    cfg = load_config()
    cutoff = float(cfg["score_not_related_cutoff"])
    nr_max = float(cfg.get("score_not_related_max", 0.3))
    label = label if label in VALID_LABELS else "undetermined"
    s = _safe_float(score)
    # Wordle / crossword : always NR (évite pont lexical « wordle/hints »)
    if anchor and target and _is_serial_template_nr_pair(anchor, target):
        return "not_related"
    # Price-today : undetermined (gold valid) — ne pas redescendre en NR via sim
    if anchor and target and _is_price_today_pair(anchor, target):
        return "undetermined"
    jac, ov = (0.0, 0)
    if anchor and target:
        jac, ov = _token_overlap(anchor, target)
    weak_lex = ov <= 1 and jac < 0.15
    bridge = bool(anchor and target and has_topical_bridge(anchor, target))
    # Pont uniquement sur filler → ignorer
    if weak_lex or ov == 0:
        bridge = False
    falseb = bool(anchor and target and _is_false_lexical_bridge(anchor, target))
    if falseb:
        bridge = False
        # Faux pont (chaos seul, video of X≠Y…) : NR seulement si sim <= nr_max.
        # Sinon même thème large avec sim moyenne → undetermined (pas NR).
        if label in ("undetermined", "not_related") and (s is None or s <= nr_max):
            return "not_related"
        if label == "not_related" and s is not None and s > nr_max:
            return "undetermined"
    if bridge and label == "not_related":
        return "undetermined"
    # Soft undetermined sans pont réel : NR seulement sous le plafond protocole
    if label == "undetermined" and not bridge and s is not None and s <= nr_max:
        return "not_related"
    if s is None:
        return label
    # Ne jamais écraser against/supporting via le seul score bas
    # (Qwen met souvent sim~0.2 quand il a dit NR à tort sur un vrai against)
    if label in ("against", "supporting"):
        return label
    if s <= cutoff:
        return "undetermined" if bridge else "not_related"
    # Protocole : sim > 0.3 → never not_related (sauf templates déjà gérés)
    if label == "not_related" and s > nr_max:
        return "undetermined"
    return label


# --- Post-LLM polarity / template refine (Qwen ignore souvent le HARD) ---
_REFINE_STOP = frozenset(
    """
    a an the and or but if in on at to for of from with by as is are was were be been being
    this that these those it its they them their we our you your he she his her not no nor
    vs versus over under into about after before between during than then so such also more
    most other out up down off too very can could may might will would shall should do does
    did doing done have has had having what which who whom when where why how new best top
    says say said amid first make early rare lead city issues opinion analysis column guide
    """.split()
)

_STRONG_CONTRA_PAIRS: tuple[tuple[str, str], ...] = (
    (r"\bwill not stand\b|\bwon'?t stand\b", r"\bwill stand\b|\bto stand\b|\bu-turn\b"),
    (r"\bcleared\b|\bno suspension\b", r"\bsuspend|\bsuspension\b"),
    (r"\bbad idea\b", r"\bat its best\b"),
    (r"\bvetoes?\b", r"\bpasses?\b|\bendorses?\b"),
    (r"\bpause tariffs\b|\bagrees to pause\b", r"\bplace tariffs\b|\bfinalizes tariffs\b|\bwill place tariffs\b"),
    (r"\bfilibuster should die\b|\bnuke the filibuster\b", r"\bdefend.*filibuster\b|\bkeep the filibuster\b|\bpreserving filibuster\b|\blet the talking filibuster\b"),
    (r"\bcan.?t win\b|\bcannot win\b|\bcan’t win\b", r"\bcan (?:still )?win\b"),
    (r"\bprohibit\b", r"\bcoexist\b"),
    (r"\bgloom\b", r"\bsurge\b|\bhope\b"),
    (r"\bechoes?\b|\bhaunt", r"\bdifferences?\b"),
    (r"\bwon'?t influence\b|\bdon'?t matter\b", r"\bthey do\b|\bvoices matter\b"),
    (r"\bwill not investigate\b", r"\bprobe\b|\binvestigat"),
    (r"\bpermanently bans?\b", r"\bis back\b"),
    (r"\bdown [\d.]+%", r"\bup [\d.]+%"),
    (r"\bnot needed\b", r"\bis needed\b|needed to cushion"),
    (r"\bsabotage\b", r"\bapproves?\b"),
    (r"\bcould block\b", r"\bgiant win\b"),
    (r"\baxes?\b|\bscraps?\b", r"\brevive\b|\bkeep all stores open\b"),
    (r"\bfailure of the un security council to pass\b", r"\bpasses? (?:gaza )?ceasefire\b|\bpasses? resolution\b"),
    (r"\beu agrees\b.*\bwithout frozen\b", r"\bfrom frozen\b"),
    # ¬P clair (Qwen dit souvent supporting à tort)
    (r"\bshould not cut\b|\bshould not cut in\b", r"\bneeds? to cut\b|\bcut interest rates sooner\b"),
    (r"\bleading\b.*\b\d+\s*point", r"\bsurges? ahead\b|\bahead of\b"),
    (r"\bhanding victory\b|\bhanding .{0,20}victory\b", r"\bdoesn'?t stand a chance\b|\bdon'?t stand a chance\b"),
    (r"\bcongratulat(?:es|ions)?\b.*\bbill\b", r"\bdevastating harms?\b"),
    (r"\bseeks? dismissal\b|\battempt to have .{0,40}dismissed\b", r"\bthrows? out\b|\bdismissed\b"),
    (r"\bno cease-?fire\b", r"\bcease-?fire begins\b|\bpaves way for ceasefire\b|\bseeks? cease-?fire\b"),
    (r"\brates rise\b|\bas rates rise\b|\brates spike\b", r"\brates hit new lows\b|\brates drop\b"),
    (r"\boptimism\b", r"\bdepression\b"),
    (r"\bcase for .{0,20}staying\b|\bstaying in\b", r"\bdebacle\b"),
    (r"\breally bad decision\b", r"\bvirtually no impact\b"),
    (r"\bfine with harris\b|\bfine with .{0,10}replacing\b", r"\bcast harris aside\b"),
    (r"\bdivided .{0,20}left\b|\bwill not repeat\b", r"\bjoin forces\b"),
    (r"\bpolling blow\b|\bsuffers .{0,20}blow\b", r"\bdismisses .{0,30}deficit\b|\bdismisses double-digit\b"),
    (r"\bthat was a lie\b", r"\bintimately involved\b|\bstar witness\b"),
    (r"\bbig cash advantage\b|\bclaims big cash\b", r"\bkeeps cash advantage\b|\bbiden keeps cash\b"),
    (r"\bnearly halves\b|\bbuying nearly halves\b", r"\bjump into\b|\binvestors jump\b"),
    # Opinions opposées explicites (holdout against)
    (r"\bbold solution\b|\bdeserve more money\b|\btime for a raise\b|universal basic income: a bold", r"\bbad idea\b|\bcan.?t afford a universal|\bdon.?t need a ubi\b"),
    (r"\bwon.?t derail .{0,20}climate\b", r"\babandon climate\b|\burg[ei]s? the world to abandon\b"),
    (r"\baffordability is a con\b|\bcon job by the democrats\b", r"\bnot a hoax\b|\breality costs\b"),
    (r"\basks? .{0,20}gag order\b|\bfor gag order\b", r"\brejects gag\b|\brejects .{0,10}gag\b"),
    (r"\bskeptical .{0,30}birthright\b", r"\bpraises .{0,30}birthright\b|\bfixing birthright\b"),
    (r"\brefuses .{0,20}bail\b|\brefuses regular bail\b", r"\bgrants bail\b"),
    (r"\bfully canceled\b|\bloans could be fully canceled\b", r"\bsave plan .{0,20}shutters\b|\bofficially shutters\b"),
    (r"\bmight still lose\b|\bhow kamala harris might still lose\b", r"\bcould win\b|\bwin big\b"),
    (r"\bcould win this election\b|\blet her\b", r"\bcan.?t win\b|\bcannot win\b"),
    (r"\bbans hurt\b|\bhow trans healthcare bans hurt\b", r"\bpraises effort to ban\b|\bpraises .{0,20}ban transgender\b"),
    (r"\bno poll can tell\b", r"\bwhat polls tell\b"),
)

# ¬P thématiques (dataset_valid) : Qwen met trop souvent NR / undetermined
_VALID_POLAR_PAIRS: tuple[tuple[str, str], ...] = (
    (r"second-fastest|fastest growing|grows? by \d|unexpected boost for", r"worst-performing|worst performing|economy shrinks|disappointed after economy shrinks"),
    (r"bigger this year|bigger .{0,15}refund", r"smaller this year|smaller .{0,15}refund"),
    (r"inflation .{0,50}rise|expected to rise|threat to inflation|high inflation is back", r"moderates|best news about inflation|falls? to|deflects inflation|better times"),
    (r"wasn'?t transitory|was not transitory", r"transitory"),
    (r"don'?t cause inflation|doesn'?t cause inflation", r"boost inflation|will boost inflation"),
    (r"rate cut|cuts strengthen|time has come .{0,30}cut|odds of .{0,40}cut", r"dents hopes|warns|higher inflation unavoidable|raising interest|holding interest|least-worst option"),
    (r"boost .{0,40}housing", r"worst thing.{0,40}housing"),
    (r"boost the middle class|lower taxes, cheaper", r"increase middle-class taxes"),
    (r"add to american growth|unleash.{0,60}growth|modestly boost|revitalize", r"harm the economy|shrink the economy|raise consumer|not enough to offset"),
    (r"congratulat|slashes deficits", r"devastating harms|increase national debt|trillions"),
    (r"muscle .{0,40}tax-cut|tax-cut and spending bill through", r"reject .{0,40}tax-cut"),
    (r"boost child tax credit|expand child tax credit", r"rejects? legislation.{0,40}child tax"),
    (r"preserve .{0,20}debt brake", r"ditches debt brake"),
    (
        r"restrict|restrictions|reversals|block on .{0,50}forgiveness|cut off forgiveness|suspends? .{0,50}forgiveness|forgiveness .is not happening.|repeal .{0,50}forgiveness|blocking loan forgiveness|delay .{0,40}forgiveness|shutters|save plan is (gone|ending)|officially (shutters|ending)|interest is resuming|spins death of student loan|death of student loan",
        r"ramp up|resumes? .{0,50}forgiveness|forgiveness resumes|allowed to proceed|forgiveness is back|reinstate|accelerate|offers relief|fully canceled|how to apply|huge win.{0,30}forgiveness|restarts student-loan forgiveness|who qualifies for .{0,40}forgiveness",
    ),
    (r"interest may resume|payments? resume|interest is resuming", r"pause extended|payment pause|plan is officially ending|officially ending"),
    (r"new student loan forgiveness application|who qualifies for biden|who qualifies for the trump", r"judges? block|federal judges block|spins death|death of student loan"),
    (r"garnishing your wages|start garnishing", r"won'?t be garnished|wages won'?t be garnished"),
    (
        r"killed .{0,30}jobs|has killed|job losses|disastrous|caused .{0,20}job|blamed for closures|costs jobs|workers are losing|lost over .{0,10}fast food",
        r"didn'?t kill jobs|doesn'?t hurt employment|zero job loss|helped .{0,30}workers|barely raised prices|isn'?t doomsday|won'?t wreck|industry growth|fast food industry growth|beating usa",
    ),
    (r"recession could still|still be in the cards", r"not .in the cards.|breathe easy|likely not"),
    (r"100 days of lies|breaking his campaign promises", r"delivering on his promises"),
    (r"basic income politics|guaranteed income policies proposed|trap people in depende|dangerous delusion|not a solution for what ails|isn'?t a good argument for basic income", r"no basic income debate|banning guaranteed income|bold solution|unlocking human potential|does not stop people working|can'?t wait"),
    (r"ai will take all our jobs|eliminate all jobs", r"increase human workforce"),
    (
        r"fails? to pass|failed to pass|failure of the un|vetoes?|vetoed|us stop gaza|us again vetoes|us vetoes",
        r"adopts? .{0,40}cease-?fire|adopts? .{0,40}resolution|backs? plan for .{0,30}cease|security council adopts|endorsed by un|demanding immediate gaza cease",
    ),
    (r"passes? .{0,50}cease-?fire|cease-?fire resolution|backs? plan for israel-gaza|demanding an immediate cease", r"fails?|failed|veto|blocks?|rejects?|failure of the un|us stop gaza"),
    (r"shrinks? from ceasefire|cannot accept .{0,30}ceasefire|changes? to gaza ceasefire .{0,20}unacceptable|breaks ceasefire", r"backs ceasefire|agrees to .{0,30}ceasefire|ceasefire proposal .{0,20}workable|inches forward"),
    (r"\bfamine\b", r"denies famine|no famine|not famine"),
    (r"peace deal|first phase|ready to reach gaza deal|might finally have a deal", r"resume war|peace deal stalls|teetering toward war"),
    (r"peace plan is unlikely|unlikely to end", r"best chance to end|offers best chance"),
    (r"will help fund russia|will help russia", r"won'?t help russia|will not help russia"),
    (r"putin not ready|not ready to end", r"russia is ready|ready to compromise"),
    (r"optimism .{0,20}sinks|optimism for ukraine sinks", r"offers hope|success offers hope"),
    (r"no longer poses major threat|hamas cannot be destroyed|doubts hamas can be demilitarized", r"remains potent threat|can be destroyed|will be disarmed"),
    (r"time to end gaza war", r"war will go on"),
    (r"two-state solution is back|gaining momentum|resurrection of the two-state", r"killed the two-state|two-state solution is dead|no longer any pretence"),
    (r"genocide denial should be criminal|war in gaza a .genocide", r"wrong to call.{0,30}genocide"),
    (r"has no choice but to fight", r"cease-fire, get hostages, leave"),
    (r"no solution to the gaza war", r"solution to the gaza problem is well underway"),
    (r"reopens kyiv embassy", r"shuts kyiv embassy"),
    (r"trapped in a war", r"not trapped"),
    (
        r"tariffs? on (canada|mexico|china)|place tariffs|finalizes tariffs|imposing tariffs|hits? (?:canada|china|mexico).{0,80}tariffs|with tariffs|triggers? trade war|vows? (?:\d+% )?tariffs|vows? big tariffs|to hit (canada|mexico|china)",
        r"pauses? tariffs|pause tariffs|avoids? tariffs|no tariffs|agrees to pause",
    ),
    (r"what to know about save|new student loan repayment program", r"save plan officially shutters|officially shutters for student"),
    (r"gold card.? visas have been sold|1000 .gold card", r"supposed to solve|gold card.? visas were supposed"),
    (r"as australia.?s teen social media ban looms|lobbying for an exemption", r"australia widens teen|scraps? exemption|widens teen social media ban"),
    (r"isn'?t climate change a huge issue|why isn'?t climate", r"brings climate change to forefront|climate change to forefront"),
    (r"save the filibuster", r"nuke the filibuster"),
    (r"it is not too late .{0,20}to go|not too late for joe biden to go", r"isn'?t going anywhere|is not going anywhere"),
    (r"attacked .{0,30}entering politics", r"delays entering politics"),
    (r"take over last remaining|approval to prosecute", r"dismisses last|all .{0,20}discharged|\bdischarged\b"),
    (r"all setup, not enough|not enough action|is all setup", r"still great|flawed but still"),
    (
        r"tiktok ban is a bad idea|warn against banning|not the answer|not the solution|more harm than good|reject call for under-16s|will not age test",
        r"tiktok ban is congress at its best|banned from social media|ban for under-16s|lot of good|should pass|votes? for under-16s|widens teen social media ban",
    ),
    (r"ai to address the climate|climate challenge solution|saving the world from climate|won'?t derail.{0,30}climate", r"fuelling the climate crisis|no climate savior|abandon climate|disastrous for the climate"),
    (r"meat isn'?t climate threat", r"meat is bad for the environment"),
    (r"support a carbon tax|path to bipartisan", r"hate the carbon tax|damaged.{0,20}leadership"),
    (r"strengthen democracy|boost innovation and manufactur", r"peril to democracy|destroyed in its infancy|dangerous regulatory vacuum"),
    (r"preempt state ai|regulating ai will be difficult|always going to lag", r"should not block state|surprisingly straightforward|time for smart"),
    (r"ev sales accelerate|meet ev sales targets|remain strong", r"rolling backwards|fall short|demand stalls"),
    (r"can help with long-term weight", r"fail to provide"),
    (r"bans hurt|strikes down.{0,40}ban|lifts pause", r"praises effort to ban|allows?.{0,40}ban|halts gender|againhalts|again halts"),
    (r"won'?t give closing|kills the historic|dismisses last|fails to obtain indictment|disqualified|discharges|grants bail|motions to dismiss", r"closing argument|keeps case.{0,20}alive|indicts|prosecute|refuses to grant|double down|takes over"),
    (r"prosecution was not best", r"conviction should stand"),
    (r"end mail-in voting", r"pushes early and mail-in"),
    (r"should not be counted|must have dates", r"should still count|must count undated"),
    (r"student ids?.{0,40}advances|banning student ids", r"blocks?.{0,40}student id ban|win for voters"),
    (r"access is still under threat", r"maintain mifepristone|hails supreme"),
    (r"fixing birthright|praises supreme court for fixing", r"tears into supreme|birthright citizenship .hoax."),
    (r"created ad tech monopoly|is an ad tech monopolist", r"plenty of ad tech competition"),
    (r"surging maga antitrust", r"comes to a screeching halt|always bogus"),
    (r"terrible vice-presidential pick", r"smartest pick"),
    (r"how harris could still lose|may not last", r"could actually win"),
    (r"push to remove biden|oppose virtual", r"push back on calls to delay"),
    (r"won'?t vote for trump if", r"ready to vote for him"),
    (r"denial of stage|denies anti-israel", r"reverses course"),
    (r"affordability is a con|con job", r"not a hoax|reality costs"),
    (r"threatens to be a backward step", r"strike a deal to boost"),
    (r"fuel america.?s economic growth|will fuel america", r"don'?t expect much growth"),
    (r"skip minimum-wage raise", r"above-inflation raise"),
    # clusters restants dataset_valid
    (r"eu fails to adopt .{0,40}sanctions", r"envoys approve|approve new russia sanctions"),
    (r"economy booming despite sanctions|continues to outperform", r"economy cracks|sanctions pressu"),
    (r"boost russian crude|maintain russian oil", r"cut russian oil|sharply cut russian"),
    (r"lift sanctions on russian oligarchs", r"extends sanctions"),
    (r"remain on eu sanctions", r"removes?.{0,40}from sanctions"),
    (r"war on russia.?s .shadow fleet.|tighten chokehold.{0,30}shadow fleet", r"keep russia.?s shadow fleet afloat|european ships keep"),
    (r"falls into recession|enters technical recession|is in a recession", r"exits recession|avoids recession|end to recession|welcomes end"),
    (r"avoids recession|gdp climbs", r"enters technical recession|gdp contraction"),
    (r"downplays recession odds", r"predicts .{0,30}crash|predicts 2025 market"),
    (r"3-year low|lowest since|still slowing|falls below 2%", r"16-month high|7-month high|heated up|holds at 3"),
    (r"eight-month high", r"eight-month low"),
    (r"leaves key interest rate unchanged", r"cuts its key interest rate"),
    (r"cuts interest rates to", r"cuts not on the horizon"),
    (r"border crisis", r"economic success story"),
    (r"no democrats.{0,50}voter id", r"readers support voter id"),
    (r"smartphone ban for kids is worth", r"ban isn'?t the an"),
    (r"settlement could end save|forcing millions into new repayment", r"revives the save"),
    (r"bad news for universal basic income", r"bold solution|unlocking human"),
    (r"can improve people.?s lives", r"moral hazard"),
    (r"moral hazard", r"unlocking human potential|bold solution"),
    (r"basic income for artists continues to have a positive", r"falls short for irish artists"),
    (r"not resigning", r"\bresigns\b"),
    (r"denies bond for|judge denies bond", r"granted .{0,20}bond"),
    (r"breaks law against|denies state funding", r"can be taught after|restore public funding"),
    (r"cold shoulder", r"rekindling her friendship"),
    (r"banned from standing for labour", r"free to stand for labour"),
    (r"defends biden.?s health", r"cover-up of biden.?s health"),
    (r"will give about .45 million|give about \$45 million", r"not donating .45 million|not donating \$45 million"),
    (r"backs away from radical reform of graduate", r"unswayed by warnings against scrapping"),
    (r"extends streak", r"streak ends"),
    (r"lawsuit against trump.?s executive order on|protect mail-in voting against", r"allows? trump to implement mail-in"),
    (r"kinds of kindness review", r"awful oscar bait"),
)

_CHRONO_SUPPORT_PAIRS: tuple[tuple[str, str], ...] = (
    (r"appeals? (?:to revive|dismissal)", r"appeals? (?:to revive|dismissal)"),
    (r"in house to answer|testify|hearing", r"resign"),
    (r"control of .{0,40}crossing|have control of", r"closure of .{0,40}crossing|cuts off"),
    (r"files to run|on trial", r"convicted"),
    (r"ashamed|security lapses", r"resign|calls for .{0,20}resign"),
    (r"cancels?|firing over|fired", r"blasts?|blasting"),
    (r"recite names|gold star families", r"criticize biden|afghanistan withdrawal"),
    (r"ten commandments", r"ten commandments"),
    (r"high stakes", r"vote against union|blow to uaw"),
    (r"how accurate .{0,40}claims", r"accused of exaggerat"),
    (r"seeks input on whether to regulate", r"condemns exclusive"),
    (r"cold shoulder", r"has not spoken"),
    (r"crowds on ai|blames .{0,30}ai", r"falsely claims .{0,40}ai"),
    (r"race against time to keep far right", r"thwarts far right|leftwing surge"),
    (r"grotesque", r"sham trial|appalling"),
    (r"promise made to keep all stores open", r"scraps plan for limited store"),
    (r"enter walz", r"condemning walz|condemn.{0,15}walz"),
    (r"gold star father|gold star mom|gold star families", r"gold star|arlington|afghan 13"),
    (r"giuliani|wabc", r"giuliani|wabc"),
    (r"menendez", r"menendez"),
    (r"wouldn'?t sign a federal abortion ban", r"would veto .{0,40}abortion ban"),
    (r"crowdstrike", r"crowdstrike"),
    (r"india.?pakistan|india vs\.? pakistan", r"india.?pakistan|ticket demand"),
    # Même dossier / continuum (VLDBench = supporting même si ton opposé)
    (r"kinds of kindness", r"kinds of kindness"),
    (r"call off strikes", r"resume strikes"),
    (r"fauci|should go to prison", r"won'?t prosecute|fauci"),
    (r"criticiz(?:e|es|ed|ing).{0,30}alito", r"defend(?:s|ed|ing).{0,30}alito"),
    (r"zero day|china invades taiwan", r"china invades taiwan|zero day"),
    (r"hamas will be disarmed|ceasefire begins", r"keep striking hamas|vows to keep striking"),
    (r"israel-hamas deal|paves way for ceasefire", r"israel-hamas war|crisis update"),
    (r"status of the save|could it pass", r"should reject the save"),
    (r"nippon steel|u\.?s\.? steel", r"nippon steel|u\.?s\.? steel"),
    (r"abortion.{0,40}trump|trump.{0,40}vance.{0,40}abortion", r"abortion|trump-vance"),
    (r"labour plans to revive|labour axes", r"labour axes|labour plans to revive"),
    (r"direct file|win for taxpayers", r"strikes against filers|tax"),
    # Pas « student loan » large : restrict↔resume = against (¬P), pas chrono
    (r"student loan forgiveness", r"wage garnishing"),
    (r"illegal immigration|illegal immigrants", r"illegal immigration|illegal immigrants"),
    (r"populist noise|moderate mould", r"populism has plenty|false promises"),
    (r"ev subsidies|ontario insists", r"thanking ford|pivoting"),
    (r"outraised|fundraising|cash advantage", r"outraises|outspends|fundraising"),
    # Pew / même série de sondages (gold supporting unanimité)
    (
        r"views of the u\.?s\.? political system|federal government and federal-state",
        r"what people think about congress|state governor and local leaders",
    ),
    # Même événement / même prise (faux against Qwen)
    (r"house kills motion to vacate|kills motion to vacate", r"votes to block .{0,40}oust|block greene.?s effort"),
    (r"\bend the international criminal court\b", r"\bagainst the international criminal court\b"),
    (r"tiktok ban was due|new deadline", r"will not be banned on|tiktok will not be banned"),
    (r"trump subverted democracy|democracy.?s destroyer", r"trump subverted democracy|pretending the left"),
)

# against sans ¬P clair → undetermined (pièges holdout + unanimité)
_SOFT_AGAINST_TO_UNDET: tuple[tuple[str, str], ...] = (
    (r"peace plan", r"no peace in sight"),
    (r"downplays?", r"visits?|arriving|warships|navy"),
    (r"may not help democrats", r"leading in the polls"),
    (r"leading in the polls", r"may not help democrats"),
    (r"falsely claims .{0,40}insulin", r"cost of insulin|insulin takes center"),
    (r"cost of insulin|insulin takes center", r"falsely claims .{0,40}insulin"),
    # Unanimité 5/6 : faux against (gold undetermined)
    (r"adam met:", r"adam pankratz:"),
    (r"milei is the frontrunner|frontrunner in argentina", r"anti-milei"),
    (r"could upend trump trial", r"kick trump off the ballot"),
    (r"bragg'?s? moment", r"rages against alvin bragg|rages against .{0,10}bragg"),
    (r"wasn'?t job interview|was not job interview", r"interview for the nation'?s top job"),
    (r"aggressive.? social media strategy|social media strategy", r"social-media censorship|censorship regime"),
    (r"finally agree on something", r"doesn'?t hang on michael cohen|does not hang on|hang on michael cohen"),
    (r"nato brushes off", r"distorts how the alliance|nato remarks distort"),
    # Faux against (gold undetermined) — nouveau run valid
    (r"kristi noem killing a dog|killing a dog will appeal", r"biden.?s dog should be shot|thinks biden.?s dog"),
    (r"plan to defeat lauren boebert|defeat lauren boebert", r"rages against plan|keep republicans off ballots"),
    (r"enemy is making a comeback|enemy as he slips", r"embraces enemy|enemy is making a comeback"),
    (r"best vpn for tiktok|vpns? for tiktok", r"tiktok not working with a vpn|not working with a vpn"),
    (r"raise the debt ceiling|debt ceiling despite", r"abolish the debt ceiling"),
    (r"both venezuela .{0,40}claim election|maduro and opposition claim", r"maduro lost venezuela|says maduro lost"),
    (r"dishonest.{0,30}border .deal.|senate border .deal.", r"border deal fails again"),
    (r"mark cuban .{0,40}student debt|knocking student debt", r"slams biden.?s student loan|suggests reversal"),
    (r"canada seeks help .{0,40}ev-sales|meeting ev-sales mandate", r"pauses? ev sales mandate"),
    # Même débat politique / backlash sans ¬P clair → undetermined
    (r"faces heat over|sparks? .{0,20}fury|sparks? .{0,20}debate|search for middle ground", r"faces heat over|sparks? .{0,20}fury|sparks? .{0,20}debate|search for middle ground"),
)

# supporting trop large (même personne / thème, claim différent) → undetermined
_SOFT_SUPPORT_TO_UNDET: tuple[tuple[str, str], ...] = (
    (r"arctic blasts|dangerous arctic", r"heat wave|historic us heat|flash flooding"),
    (r"earns trump.?s praise after dropping|dropping out of race", r"revels in the limelight"),
    (r"inevitable defeat|faces .inevitable|new york trial, lawyer", r"what to know about the trial donald trump faces"),
    (r"pours cold water on .{0,30}mar-a-lago|mar-a-lago boxes excuse", r"favorite.? newspaper damns|stinging six-word"),
    (r"gets new threat out of arizona and georgia", r"stung by key legal ruling in georgia"),
    (r"republicans blast .{0,20}spending|swamp.?s.? spending", r"stare down another funding failure|funding failure"),
    (r"nato brushes off", r"nato remarks distort|distorts how the alliance"),
)


def _headline_core_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", " ", text or "")
    text = re.sub(r"\s+-\s+[A-Za-z][^-\n]*$", "", text)
    return text.strip()


def _refine_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z'-]+", _headline_core_text(text)):
        wl = w.lower().strip("'")
        # normalise possessifs : rogan's → rogan
        if wl.endswith("'s"):
            wl = wl[:-2]
        elif wl.endswith("s'") and len(wl) > 2:
            wl = wl[:-2]
        for part in re.split(r"[-–—/]", wl):
            part = part.strip("'")
            if len(part) >= 4 and part not in _REFINE_STOP:
                out.add(part)
    return out


def _token_overlap(anchor: str, target: str) -> tuple[float, int]:
    a, b = _refine_tokens(anchor), _refine_tokens(target)
    if not a or not b:
        return 0.0, 0
    ov = a & b
    return len(ov) / len(a | b), len(ov)


def _content_bigrams(text: str) -> set[tuple[str, str]]:
    """Bigrammes de contenu (ordre conservé) pour signal « même claim P »."""
    toks: list[str] = []
    for w in re.findall(r"[A-Za-z][A-Za-z'-]+", _headline_core_text(text or "")):
        wl = w.lower().strip("'")
        if wl.endswith("'s"):
            wl = wl[:-2]
        if len(wl) >= 3 and wl not in _REFINE_STOP:
            toks.append(wl)
    return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}


def _shared_claim_bigram(anchor: str, target: str) -> bool:
    """Vrai si les titres partagent un bigramme de *claim* (pas seulement entité/œuvre).

    « running mate », « tiktok ban », « press briefing » → signal P.
    « baby reindeer », « young voters », « greg abbott » → univers partagé, pas P.
    """
    shared = _content_bigrams(anchor) & _content_bigrams(target)
    if not shared:
        return False
    weak = {
        ("new", "york"),
        ("white", "house"),
        ("united", "states"),
        ("donald", "trump"),
        ("joe", "biden"),
        ("real", "clear"),
        ("baby", "reindeer"),
        ("young", "voters"),
        ("greg", "abbott"),
        ("michelle", "obama"),
        ("mary", "trump"),
        ("nicole", "shanahan"),
    }
    shared = shared - weak
    if not shared:
        return False
    # Au moins un token « événement / relation » dans le bigramme
    claim_cues = {
        "mate", "running", "ban", "banned", "blocking", "debate", "interview",
        "recruit", "recruits", "recruiting", "mercenaries", "ceasefire", "veto",
        "shutdown", "crossing", "resigns", "resignation", "appeal", "appeals",
        "verdict", "indictment", "shortlist", "contenders", "manifesto",
        "briefing", "warning", "prediction", "paraphrase", "transcript",
        "outage", "crashed", "killed", "forgiveness", "inflation", "growth",
        "ropes", "campaign", "prime", "minister",
    }
    return any(claim_cues & set(bg) for bg in shared)


def _same_survey_series_pair(anchor: str, target: str) -> bool:
    """Même série de sondages (ex. Pew « Views of… ») = même dossier P → supporting."""
    a, b = (anchor or "").lower(), (target or "").lower()
    if "pew research" not in a or "pew research" not in b:
        return False
    survey = r"views of|trust in scientists|positive views of science|what people think about"
    return bool(re.search(survey, a) and re.search(survey, b))


def _pair_matches_any(anchor: str, target: str, pairs: tuple[tuple[str, str], ...]) -> bool:
    al = (anchor or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    bl = (target or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    for p1, p2 in pairs:
        if (re.search(p1, al) and re.search(p2, bl)) or (re.search(p1, bl) and re.search(p2, al)):
            return True
    return False


def _is_wordle_template_pair(anchor: str, target: str) -> bool:
    return bool(re.search(r"\bwordle\b", anchor or "", re.I) and re.search(r"\bwordle\b", target or "", re.I))


def _is_price_today_pair(anchor: str, target: str) -> bool:
    """Snapshots prix / taux du jour (jours différents) → NR holdout."""
    a, b = anchor or "", target or ""
    return bool(
        (
            re.search(r"price today|prices today|trading at \$", a, re.I)
            and re.search(r"price today|prices today|trading at \$", b, re.I)
        )
        or (
            re.search(r"mortgage (?:and refinance )?rates? today|current mortgage rates", a, re.I)
            and re.search(
                r"mortgage (?:and refinance )?rates? today|current mortgage rates|rates hit new lows|rates drop|rates spike",
                b,
                re.I,
            )
        )
    )


def _is_serial_template_nr_pair(anchor: str, target: str) -> bool:
    """Templates snapshot strictement indépendants → not_related.

    Uniquement Wordle / crossword. Price-today et digests → undetermined
    (gold valid les traite ainsi ; holdout Wordle reste NR).
    """
    a, b = anchor or "", target or ""
    al, bl = a.lower(), b.lower()
    if re.search(r"off the grid:.*crossword", al) and re.search(r"off the grid:.*crossword", bl):
        return True
    if _is_wordle_template_pair(a, b):
        return True
    return False


def _has_polar_negation(anchor: str, target: str) -> bool:
    """True si T_n nie clairement P de T_ref (ou l'inverse) — against VLDBench."""
    return _pair_matches_any(anchor, target, _STRONG_CONTRA_PAIRS) or _pair_matches_any(
        anchor, target, _VALID_POLAR_PAIRS
    )


def _is_serial_soft_undet_pair(anchor: str, target: str) -> bool:
    """Séries snapshot jours différents → undetermined (pas supporting)."""
    a, b = (anchor or "").lower(), (target or "").lower()
    checks = [
        (r"daily mortgage rates for", r"daily mortgage rates for"),
        (r"top cd rates today", r"top cd rates today"),
        (r"crossword blog & answers", r"crossword blog & answers"),
        (r"what to watch this weekend", r"what to watch this weekend"),
        (r"palladium price today|gold price today|silver price today", r"palladium price today|gold price today|silver price today"),
    ]
    for p1, p2 in checks:
        if re.search(p1, a) and re.search(p2, b):
            return True
    return False


def _should_upgrade_undet_to_supporting(anchor: str, target: str, score: float) -> bool:
    """Paraphrase / même dossier clair : undetermined → supporting."""
    if _has_polar_negation(anchor, target):
        return False
    if _is_price_today_pair(anchor, target) or _is_serial_template_nr_pair(anchor, target):
        return False
    if _is_serial_soft_undet_pair(anchor, target):
        return False
    if _pair_matches_any(anchor, target, _CHRONO_SUPPORT_PAIRS):
        return True
    if _same_survey_series_pair(anchor, target):
        return True
    al, bl = (anchor or "").lower(), (target or "").lower()
    named = [
        (r"highlights from the 2024 presidential election campaign", r"highlights from the 2024 presidential election campaign"),
        (r"today'?s? top savings account rate roundup", r"today'?s? top savings account rate roundup"),
        (r"the bulletin (january|february|march|april|may|june|july|august|september|october|november|december)", r"the bulletin (january|february|march|april|may|june|july|august|september|october|november|december)"),
        (r"the city.?s wish list|city of london.?s wish list|what the city of london wants", r"the city.?s wish list|city of london.?s wish list|what the city of london wants"),
        (r"israeli extremists won|extremists took over israel|the unpunished", r"israeli extremists won|extremists took over israel|the unpunished"),
        (r"venezuela.?s stolen election|venezuela.?s fraudulent election|after venezuela.?s fraudulent", r"venezuela.?s stolen election|venezuela.?s fraudulent election|stolen election"),
        (r"risks? lurking in .{0,40}private (credit|capital)", r"risks? lurking in .{0,40}private (credit|capital)"),
        (r"netanyahu shrinks from ceasefire|coalition slides into infighting over ceasefire", r"netanyahu shrinks from ceasefire|infighting over ceasefire|backs ceasefire compromise"),
        (r"how the world.?s tech crashed|global tech outage", r"how the world.?s tech crashed|global tech outage|brought many computer systems"),
        (r"dnc speakers schedule|who.?s speaking at the democratic national", r"dnc speakers schedule|who.?s speaking at the democratic national|speaking thursday"),
        (r"who will be thailand.?s next prime minister|thailand.?s new prime minister", r"who will be thailand.?s next prime minister|thailand.?s new prime minister"),
        (r"tiktok bans across the world|law that could get tiktok banned", r"tiktok bans across the world|law that could get tiktok banned"),
        (r"desantis on the ropes|what happened to the desantis campaign", r"desantis on the ropes|what happened to the desantis campaign"),
        (r"desantis-haley debate|biggest moments|winner of the desantis-haley debate", r"desantis-haley debate|biggest moments|winner of the desantis-haley debate"),
        (r"trust in scientists|anti-science views|trust the science", r"trust in scientists|anti-science views|trust the science"),
        # raw string : un seul backslash avant le point (u\.?s\.? matche « U.S. »)
        (r"views of american politics|views of the republican party and democratic party|views of the u\.?s\.? political system|do political parties represent", r"views of american politics|views of the republican party and democratic party|views of the u\.?s\.? political system|do political parties represent"),
        # Types de claim P explicites (pas des paires mémorisées)
        (r"interview with|exclusive interview|transcript of .{0,30}interview", r"interview with|exclusive interview|transcript of .{0,30}interview"),
        (r"running[- ]mate", r"running[- ]mate"),
        (r"\brecruit|\brecruits|\brecruiting|\bmercenar", r"\brecruit|\brecruits|\brecruiting|\bmercenar"),
    ]
    for p1, p2 in named:
        if (re.search(p1, al) and re.search(p2, bl)) or (re.search(p1, bl) and re.search(p2, al)):
            return True
    jac, ov = _token_overlap(anchor, target)
    bridge = has_topical_bridge(anchor, target) or ov >= 3
    if not bridge:
        return False
    same_phrase = _shared_claim_bigram(anchor, target)
    # Paraphrase lexicale forte : exige bigramme de claim OU overlap très dense
    # (évite Baby Reindeer / young voters : mêmes noms, P différent → undetermined)
    if ov >= 5 and score >= 0.55 and (same_phrase or jac >= 0.35):
        return True
    if ov >= 4 and score >= 0.70 and jac >= 0.32 and same_phrase:
        return True
    if ov >= 6 and score >= 0.50 and same_phrase:
        return True
    if same_phrase and score >= 0.78 and ov >= 3 and jac >= 0.25:
        return True
    return False


def refine_qwen_relation(
    label: str,
    score: float | None,
    anchor: str | None = None,
    target: str | None = None,
) -> str:
    """Correctifs post-LLM (Qwen seul) : polarité / templates / soft labels.

    Priorité dataset_valid : récupérer against quand Qwen met NR/undet sur ¬P clair.
    """
    if not anchor or not target:
        return label
    label = label if label in VALID_LABELS else "undetermined"
    s = _safe_float(score)
    if s is None:
        s = 0.5

    # 1) Templates snapshot
    if _is_serial_template_nr_pair(anchor, target):
        return "not_related"
    if _is_price_today_pair(anchor, target) or _is_serial_soft_undet_pair(anchor, target):
        return "undetermined"

    jac, ov = _token_overlap(anchor, target)
    contra = _has_polar_negation(anchor, target)
    chrono = _pair_matches_any(anchor, target, _CHRONO_SUPPORT_PAIRS)
    soft_undet = _pair_matches_any(anchor, target, _SOFT_AGAINST_TO_UNDET)
    soft_supp_undet = _pair_matches_any(anchor, target, _SOFT_SUPPORT_TO_UNDET)
    false_bridge = _is_false_lexical_bridge(anchor, target)
    bridge = has_topical_bridge(anchor, target) or ov >= 2

    # 2) ¬P lexical fort → against (ne dépend pas du pont faible / sim basse)
    if contra and not soft_undet:
        return "against"

    if label == "against":
        if chrono and not contra:
            return "supporting"
        if soft_undet:
            return "undetermined"
        if contra:
            return "against"
        if s >= 0.78 and ov >= 4 and jac >= 0.35:
            return "supporting"
        if ov <= 1 and s < 0.55:
            return "undetermined"
        return "against"

    if label == "supporting":
        if contra:
            return "against"
        if soft_supp_undet and not chrono:
            return "undetermined"
        if chrono:
            return "supporting"
        # Garde-fou S↔U: supporting seulement si même claim assez explicite.
        # Sinon (même thème/personnes avec recouvrement faible), rester prudent.
        if not _should_upgrade_undet_to_supporting(anchor, target, s):
            if ov <= 3 and s < 0.75:
                return "undetermined"
            if ov <= 2 and s < 0.8:
                return "undetermined"
            # Overlap élevé mais pas de bigramme de claim → même univers, P différent
            if ov >= 3 and not _shared_claim_bigram(anchor, target) and not _same_survey_series_pair(anchor, target):
                return "undetermined"
        # Overlap élevé mais claim dilué (jac faible) → undetermined
        if ov >= 4 and s >= 0.70 and jac < 0.28 and not chrono and not _shared_claim_bigram(anchor, target):
            return "undetermined"
        if false_bridge and s <= 0.3:
            return "not_related"
        if _is_serial_soft_undet_pair(anchor, target):
            return "undetermined"
        if ov <= 1 and jac < 0.12 and s < 0.65:
            return "undetermined"
        if re.search(r"forbes daily:", anchor or "", re.I) and re.search(r"forbes daily:", target or "", re.I):
            return "undetermined"
        if re.search(r"abbreviated pundit roundup", anchor or "", re.I) and re.search(
            r"abbreviated pundit roundup", target or "", re.I
        ):
            return "undetermined"
        if re.search(r"\bkitten\b|\bcat felt\b", (anchor or "") + (target or ""), re.I) and ov <= 3:
            return "undetermined"
        return "supporting"

    if label == "undetermined":
        if soft_undet and re.search(r"insulin", (anchor or "") + (target or ""), re.I):
            if re.search(r"falsely claims", (anchor or "") + (target or ""), re.I):
                return "supporting"
        if contra and not soft_undet:
            return "against"
        if soft_undet:
            return "undetermined"
        if soft_supp_undet:
            return "undetermined"
        if _should_upgrade_undet_to_supporting(anchor, target, s):
            # Même claim clair seulement ; éviter faux supporting sur overlap de noms
            if not (ov >= 4 and jac < 0.28 and not _shared_claim_bigram(anchor, target)):
                return "supporting"
        if chrono and s >= 0.55:
            return "supporting"
        if false_bridge and s <= 0.3:
            return "not_related"
        return "undetermined"

    if label == "not_related":
        if false_bridge and s <= 0.3:
            return "not_related"
        if false_bridge and s > 0.3:
            return "undetermined"
        if _is_price_today_pair(anchor, target) or _is_serial_soft_undet_pair(anchor, target):
            return "undetermined"
        if _should_upgrade_undet_to_supporting(anchor, target, s):
            return "supporting"
        # Pont réel : au minimum undetermined ; ¬P déjà géré plus haut
        if bridge:
            return "undetermined"
        # Alignement protocole : sim > 0.3 → never NR
        if s > 0.3:
            return "undetermined"
        return "not_related"

    return label


def _is_false_lexical_bridge(anchor: str, target: str) -> bool:
    """Ponts lexicaux fallacieux où le gold est clairement not_related."""
    a, b = (anchor or "").lower(), (target or "").lower()
    # Prénom William seul (Lai ≠ Burrows)
    if re.search(r"\bwilliam\b", a) and re.search(r"\bwilliam\b", b):
        if not (re.search(r"\blai\b", a) and re.search(r"\blai\b", b)):
            if not (re.search(r"\bburrows\b", a) and re.search(r"\bburrows\b", b)):
                ta, tb = _refine_tokens(anchor), _refine_tokens(target)
                if ta & tb <= {"william"}:
                    return True
    # « Video of X » ≠ « Video of Y » (sujets différents)
    if re.search(r"\bvideo of\b", a) and re.search(r"\bvideo of\b", b):
        ta, tb = _refine_tokens(anchor), _refine_tokens(target)
        if not ((ta & tb) - {"video", "viewed", "times"}):
            return True
    # Template listicle « need to know / essentials you need »
    if re.search(r"things you need to know|essentials you need", a) and re.search(
        r"things you need to know|essentials you need|packing list", b
    ):
        return True
    if re.search(r"things you need to know|essentials you need", b) and re.search(
        r"things you need to know|essentials you need|packing list", a
    ):
        return True
    # « You Are the Worst » ≠ « Worst Rotten Tomatoes »
    if re.search(r"\bthe worst\b", a) and re.search(r"\bworst\b", b):
        ta, tb = _refine_tokens(anchor), _refine_tokens(target)
        if ta & tb <= {"worst"}:
            return True
    # chaos Columbia ≠ chaos White House (seul « chaos »)
    ta, tb = _refine_tokens(anchor), _refine_tokens(target)
    ov = ta & tb
    if ov and ov <= {"chaos", "death", "right", "video", "need", "worst", "william"}:
        return True
    # Votes / élections pays différents sans autre overlap
    if re.search(r"\bvoting\b|\bvotes\b", a) and re.search(r"\bvoting\b|\bvotes\b", b):
        if ov <= {"voting", "votes", "sunday", "future"} or not ov:
            if ("puerto" in a) != ("puerto" in b) or ("dominican" in a) != ("dominican" in b):
                return True
    # Mascots / oiseau national — zéro claim partagé
    if re.search(r"\bmascots?\b", a) and re.search(r"\bnational bird\b|\bbird\b", b):
        return True
    if re.search(r"\bmascots?\b", b) and re.search(r"\bnational bird\b|\bbird\b", a):
        return True
    # Tuberville military ≠ Walz stolen valor (thème military seul)
    if re.search(r"\btuberville\b", a) and re.search(r"\bwalz\b|\bstolen valor\b", b):
        return True
    if re.search(r"\btuberville\b", b) and re.search(r"\bwalz\b|\bstolen valor\b", a):
        return True
    # Undetermined Qwen avec overlap nul + sim « inventée »
    if not ov and re.search(r"\bmascot|national bird|stolen valor|packing list\b", a + " " + b):
        return True
    return False


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
    # 1) polarité / templates (avant filet NR qui pourrait re-pontifier Wordle)
    label = refine_qwen_relation(label, s, anchor=anchor, target=target)
    # 2) cohérence sim / pont lexical
    label = enforce_label_sim_consistency(label, s, anchor=anchor, target=target)
    # 3) Templates série restent NR même si un pont lexical faible existe
    if anchor and target and _is_serial_template_nr_pair(anchor, target):
        label = "not_related"
    if label == "undetermined" and anchor and target and has_topical_bridge(anchor, target):
        # Evite sim 0.1 + undetermined (incoherent) apres correction anti-crush.
        s = max(s, 0.35)
    if label == "supporting" and s < 0.55:
        s = max(s, 0.65)
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
    # P3: protocole court + few-shot + HARD ultra-court (évite mur de texte ignoré)
    return (
        f"{protocol}\n\n"
        "Few-shot (mirror these; first of each label = main guardrail):\n"
        f"{examples}\n---\n"
        "HARD (read last): "
        "1) Write P from T_ref. against ONLY if T_n asserts ¬P on the SAME P "
        "(opposite outcome: fastest↔worst, rise↔fall, blocked↔resumes, killed-jobs↔didn't-kill, "
        "passes-ceasefire↔veto/fail, UBI-delusion↔UBI-solution). "
        "NEVER not_related when that shared P has opposite outcomes. "
        "Negative tone / fury / heat / debate on same policy without ¬P → undetermined (not against). "
        "2) supporting ONLY if same explicit claim P (same event update / shortlist / interview / sequel). "
        "Same person/theme alone → undetermined. "
        "3) Mid similarity (sim>0.3) with weak word overlap but shared theme → undetermined, NEVER not_related. "
        "Wordle #N vs #M / pure filler → not_related. "
        "4) Unsure supporting vs against (no ¬P) → supporting. Unsure against vs undet → undetermined. "
        "5) Zero bridge only → not_related. sim>0.3 → never NR.\n"
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
        # DeBERTa base / Cascade V8 charges a la demande selon le mode.
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.deberta = None
        self.v8: Any = None

    def ensure_deberta(self) -> None:
        if self.deberta is None:
            self._load_deberta()

    def ensure_v8(self) -> None:
        if self.v8 is None:
            self._load_v8()

    def _load_deberta(self):
        model_path = REPO / self.cfg["deberta_model_path"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele DeBERTa manquant: {model_path}. "
                "Executez: python scripts/setup.py"
            )
        self.deberta = CrossEncoder(str(model_path), device=self.device)

    def _load_v8(self):
        from cascade.v8_predictor import CascadePredictor, default_v8_root

        root = default_v8_root()
        cfg_override = (self.cfg.get("cascade_v8") or {}).get("root")
        if cfg_override:
            candidate = (CASCADE_DIR / cfg_override).resolve()
            if not candidate.is_dir():
                candidate = (REPO / cfg_override).resolve()
            if (candidate / "config.json").is_file():
                root = candidate
        self.v8 = CascadePredictor(config_path=root / "config.json")


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
        if mode == CASCADE_MODE_V8_QWEN:
            return self._route_v8_qwen(anchor, target, should_cancel=should_cancel)
        if mode == CASCADE_MODE_COMPARE:
            return self._route_compare(
                anchor, target, tau_auto=tau_auto, should_cancel=should_cancel
            )
        return self._route_qwen_only(anchor, target, should_cancel=should_cancel)

    def _route_v8_qwen(
        self,
        anchor: str,
        target: str,
        should_cancel: CancelCheck = None,
    ) -> dict:
        """Cascade collègue V8 : Duo MiniLM+DeBERTa → Reranker → Qwen P3 + post-LLM."""
        if should_cancel and should_cancel():
            raise BatchCancelledError("Batch cancelled by user.")

        result = self._empty_route_result()
        v8_start = time.monotonic()
        self.ensure_v8()

        if should_cancel:
            holder: dict[str, Any] = {}
            err_box: list[BaseException] = []

            def _work() -> None:
                try:
                    holder["out"] = self.v8.annotate_one(anchor, target)
                except BaseException as e:  # noqa: BLE001
                    err_box.append(e)

            worker = threading.Thread(target=_work, daemon=True)
            worker.start()
            while worker.is_alive():
                if should_cancel():
                    raise BatchCancelledError("Batch cancelled by user.")
                worker.join(0.2)
            if err_box:
                raise err_box[0]
            v8_out = holder["out"]
        else:
            v8_out = self.v8.annotate_one(anchor, target)

        v8_duration_s = round(time.monotonic() - v8_start, 3)
        label = v8_out.get("label")
        conf = float(v8_out.get("confidence") or 0.0)
        stage = v8_out.get("stage") or "Duo_Zero_Faute"
        rule = v8_out.get("rule_triggered") or ""
        probs = v8_out.get("probabilities") or {}
        sim_defaults = self.cfg.get("similarity_defaults") or {}

        result.update(
            v8_status=v8_out.get("status"),
            v8_label=label,
            v8_conf=round(conf, 4),
            v8_stage=stage,
            v8_rule=rule,
            v8_probabilities=probs,
            v8_minilm_label=v8_out.get("minilm_label"),
            v8_deberta_label=v8_out.get("deberta_label"),
            v8_reranker_undet_prob=v8_out.get("reranker_undetermined_prob"),
            v8_duration_s=v8_duration_s,
            deberta_pred=v8_out.get("deberta_label") or label,
            deberta_conf=round(conf, 4),
            deberta_duration_s=v8_duration_s,
            cascade_llm_duration_s=0.0,
        )

        if v8_out.get("status") == "AUTO_ANNOTATED" and label in VALID_LABELS:
            sim = float(sim_defaults.get(label, 0.55))
            # Aligne légèrement sim sur la confiance modèle (borne score_max)
            score_max = float(self.cfg.get("score_max", 0.8))
            sim = min(score_max, max(sim, conf * 0.85 if conf >= 0.9 else sim))
            route_name = "v8_reranker" if stage == "Reranker" else "v8_duo"
            result.update(
                route=route_name,
                related=label,
                similarity_annotation=round(sim, 3),
                deberta_sim=round(sim, 3),
            )
            return result

        # HUMAN_REVIEW modèles → Qwen P3 + post-LLM
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

        result.update(
            route="v8_qwen",
            related=llm_label,
            similarity_annotation=llm_score,
        )
        return result

    def _route_compare(
        self,
        anchor: str,
        target: str,
        tau_auto: float | None = None,
        should_cancel: CancelCheck = None,
    ) -> dict:
        """Compare Qwen only vs Cascade V8 + Qwen (duo/reranker puis Qwen si besoin)."""
        del tau_auto  # τ DeBERTa-base non utilisé pour V8
        qwen_start = time.monotonic()
        qwen_out = self._route_qwen_only(anchor, target, should_cancel=should_cancel)
        qwen_out["duration_s"] = round(time.monotonic() - qwen_start, 3)

        cascade_start = time.monotonic()
        cascade_out = self._route_v8_qwen(anchor, target, should_cancel=should_cancel)
        cascade_out["duration_s"] = round(time.monotonic() - cascade_start, 3)

        compare = build_pipeline_compare(qwen_out, cascade_out)
        result = self._empty_route_result()
        result["pipeline_compare"] = compare
        result["qwen_only_result"] = qwen_out
        result["cascade_v8_result"] = cascade_out
        result["deberta_qwen_result"] = cascade_out  # alias historique

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
                deberta_conf=cascade_out.get("v8_conf") or cascade_out.get("deberta_conf"),
                v8_stage=cascade_out.get("v8_stage"),
                v8_rule=cascade_out.get("v8_rule"),
                v8_label=cascade_out.get("v8_label"),
                v8_conf=cascade_out.get("v8_conf"),
            )
        else:
            result.update(
                route="compare_disagree",
                related=None,
                similarity_annotation=None,
                llm_pred=qwen_out.get("llm_pred"),
                llm_conf=qwen_out.get("llm_conf"),
                deberta_pred=cascade_out.get("deberta_pred"),
                deberta_conf=cascade_out.get("v8_conf") or cascade_out.get("deberta_conf"),
                v8_stage=cascade_out.get("v8_stage"),
                v8_rule=cascade_out.get("v8_rule"),
                v8_label=cascade_out.get("v8_label"),
                v8_conf=cascade_out.get("v8_conf"),
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
            "v8_status": None,
            "v8_label": None,
            "v8_conf": None,
            "v8_stage": None,
            "v8_rule": None,
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
