"""Cascade V8 — Duo Zero Faute (MiniLM + DeBERTa) et Reranker undetermined.

Qwen (dernier étage) est géré en aval par ``cascade.core.CascadeEngine``.

Règles appliquées dans l'ordre par ``annotate_pairs`` :
    Rule 5 — Models Disagree    : argmax(MiniLM) ≠ argmax(DeBERTa) → rejet duo.
    Rule 1 — Against Specialist : P(against) > ``against_binary_threshold`` → AUTO against.
    Rule 2 — Undetermined Trap  : P(undetermined) > ``undetermined_rejection_threshold`` → rejet.
    Rule 3 — Global Confidence  : max(P_ensemble) ≥ ``global_tau`` et accord → AUTO argmax.
    Rule 4 — Low Confidence     : sinon → rejet.
    Reranker                    : si rejet duo et P(undetermined) ≥ ``reranker_undetermined_threshold``
                                  → AUTO undetermined ; sinon status HUMAN_REVIEW (→ Qwen).

Configuration :
    Lue depuis ``models/cascade_v8/config.json`` (chemin résolu par ``default_v8_root()``
    ou variable d'environnement ``CASCADE_V8_ROOT``).

Modèles chargés :
    - ``minilm_full_v7``       : MiniLM v7, 6 couches RoBERTa, 4 classes (fast device).
    - ``deberta_large_v8.1``   : DeBERTa-v3 Large, 24 couches, 4 classes (CPU forcé).
    - ``reranker_undetermined_v8`` : XLM-RoBERTa Large binaire (fast device).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit, softmax

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers requis pour la cascade V8. "
        "Installez-le dans le venv de l'app."
    ) from e


def default_v8_root() -> Path:
    """Résout la racine du package modèles Cascade V8.

    Cherche dans l'ordre :
        1. Variable d'environnement ``CASCADE_V8_ROOT`` (si définie).
        2. ``<repo>/models/cascade_v8/`` (lien symbolique ou dossier direct).
        3. Workspace ``AI_annotation/cascade_annotation_v8_complete/…``.

    Returns:
        Chemin résolu vers la racine V8 (peut ne pas exister si non installé).
    """
    here = Path(__file__).resolve().parent
    env = os.environ.get("CASCADE_V8_ROOT", "").strip()
    candidates = [
        Path(env) if env else None,
        # App : annotation-tool-package-2/models/cascade_v8 → symlink
        here.parent / "models" / "cascade_v8",
        # Workspace AI_annotation
        here.parent.parent
        / "AI_annotation"
        / "cascade_annotation_v8_complete"
        / "cascade_annotation_v8_complete",
        # Depuis AI_annotation/cascade/
        here.parent
        / "cascade_annotation_v8_complete"
        / "cascade_annotation_v8_complete",
        here.parent / "models" / "cascade_v8",
    ]
    for c in candidates:
        if c and (c / "config.json").is_file() and (c / "models").is_dir():
            return c.resolve()
    return (here.parent / "models" / "cascade_v8").resolve()


class CascadePredictor:
    """Duo Zero Faute (MiniLM + DeBERTa) suivi du Reranker undetermined.

    Qwen (dernier étage de la cascade) est délégué à ``CascadeEngine`` dans
    ``cascade.core`` ; ce prédicteur ne gère que les étages pré-LLM.

    Attributes:
        config: Dictionnaire de configuration issu de ``config.json``.
        class_names: Noms des 4 classes dans l'ordre du modèle
            (against, not_related, supporting, undetermined).
        weights: Poids de l'ensemble Duo (``minilm_v7`` et ``deberta_large_v8``).
        thresholds: Seuils de décision (``global_tau``, ``against_binary_threshold``, …).
        base_dir: Dossier racine contenant les sous-dossiers de modèles.
        models: Dictionnaire ``{nom: CrossEncoder}`` pour les 3 modèles chargés.
        device_fast: Périphérique d'inférence rapide (``cuda`` | ``mps`` | ``cpu``).
    """

    def __init__(self, config_path: str | Path | None = None):
        """Charge la configuration et les trois modèles (MiniLM, DeBERTa, Reranker).

        Args:
            config_path: Chemin explicite vers ``config.json``. Si ``None``,
                utilise ``default_v8_root() / "config.json"``.

        Raises:
            FileNotFoundError: Si ``config.json`` ou l'un des poids est absent.
            ImportError: Si ``sentence-transformers`` n'est pas installé.
        """
        root = default_v8_root()
        cfg_path = Path(config_path) if config_path else root / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Config cascade V8 introuvable: {cfg_path}. "
                "Vérifiez AI_annotation/cascade_annotation_v8_complete/..."
            )
        with open(cfg_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.class_names: list[str] = list(self.config["class_names"])
        self.weights = self.config["weights"]
        self.thresholds = self.config["thresholds"]
        self.base_dir = cfg_path.parent.resolve()

        device_fast = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        # DeBERTa Large : CPU obligatoire (NaNs observés sur GPU/MPS chez le collègue)
        deberta_device = self.config.get("deberta_device", "cpu")

        print("[CascadePredictor V8] Chargement MiniLM / DeBERTa / Reranker…")
        path_minilm = self.base_dir / self.config["paths"]["minilm_v7"]
        path_deberta = self.base_dir / self.config["paths"]["deberta_large_v8"]
        path_reranker = self.base_dir / self.config["paths"]["reranker"]
        for p, name in (
            (path_minilm, "MiniLM"),
            (path_deberta, "DeBERTa"),
            (path_reranker, "Reranker"),
        ):
            if not p.exists():
                raise FileNotFoundError(f"Modèle {name} manquant: {p}")

        self.models: dict[str, Any] = {
            "minilm_v7": CrossEncoder(str(path_minilm), device=device_fast),
            "deberta_large_v8": CrossEncoder(str(path_deberta), device=deberta_device),
            "reranker": CrossEncoder(str(path_reranker), device=device_fast),
        }
        self.device_fast = device_fast
        print(
            f"[CascadePredictor V8] OK (fast={device_fast}, deberta={deberta_device})"
        )

    def _get_model_probs(self, model, sentence_pairs: list) -> np.ndarray:
        """Calcule les probabilités softmax pour un modèle cross-encoder.

        Gère trois cas de sortie :
        - Logit scalaire (binaire) → converti en 4 colonnes via sigmoïde.
        - Logits 3 classes (NLI) → remappés vers 4 classes VLDBench.
        - Logits 4 classes → softmax direct.

        Args:
            model: Instance ``CrossEncoder`` à appeler.
            sentence_pairs: Liste de paires ``[ancre, cible]``.

        Returns:
            Tableau NumPy de forme ``(N, 4)`` avec les probabilités par classe.
        """
        logits = model.predict(
            sentence_pairs,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            scores = logits.ravel()
            p_rel = expit(scores)
            probs = np.zeros((len(sentence_pairs), 4))
            probs[:, 1] = 1.0 - p_rel
            probs[:, 2] = p_rel
            return probs

        probs = softmax(logits, axis=1)
        # Mapping 3 classes -> 4 classes (contre / not_related / entail / neutral)
        if probs.shape[1] == 3:
            p_c, p_n, p_e = probs[:, 0], probs[:, 1], probs[:, 2]
            mapped = np.zeros((len(sentence_pairs), 4))
            mapped[:, 0] = p_c * 0.45
            mapped[:, 1] = p_c * 0.55
            mapped[:, 2] = p_e
            mapped[:, 3] = p_n
            return mapped
        return probs

    def predict_duo_probabilities(self, sentence_pairs: list):
        """Calcule les probabilités ensemble MiniLM + DeBERTa (Duo).

        L'ensemble est une combinaison linéaire pondérée :
        ``P_ensemble = weights[minilm] * P_mini + weights[deberta] * P_large``.

        Args:
            sentence_pairs: Liste de paires ``[ancre, cible]``.

        Returns:
            Tuple ``(p_ensemble, p_minilm, p_deberta)`` — trois tableaux
            NumPy de forme ``(N, 4)``.
        """
        p_mini = self._get_model_probs(self.models["minilm_v7"], sentence_pairs)
        p_large = self._get_model_probs(self.models["deberta_large_v8"], sentence_pairs)
        p_ensemble = (
            self.weights["minilm_v7"] * p_mini
            + self.weights["deberta_large_v8"] * p_large
        )
        return p_ensemble, p_mini, p_large

    def annotate_pairs(self, sentence_pairs: list[list[str] | tuple[str, str]]) -> list[dict]:
        """Applique le Duo Zero Faute puis le Reranker sur une liste de paires.

        Chaque paire produit un dict avec les champs :
            ``index``, ``status`` (AUTO_ANNOTATED | HUMAN_REVIEW), ``label``,
            ``confidence``, ``probabilities``, ``minilm_label``, ``deberta_label``,
            ``rule_triggered``, ``stage``, ``reranker_undetermined_prob`` (si rejet duo).

        Les paires avec ``status == HUMAN_REVIEW`` après le Reranker sont
        transmises à Qwen par ``CascadeEngine`` (hors de cette méthode).

        Args:
            sentence_pairs: Liste de paires ``[ancre, cible]`` ou ``(ancre, cible)``.

        Returns:
            Liste de dicts de résultats, un par paire, dans le même ordre.
        """
        if not sentence_pairs:
            return []

        p_ensemble, p_minilm, p_deberta = self.predict_duo_probabilities(sentence_pairs)
        results: list[dict] = []
        rejected_indices: list[int] = []

        for i, p in enumerate(p_ensemble):
            prob_dict = {cls: float(p[idx]) for idx, cls in enumerate(self.class_names)}
            max_idx = int(np.argmax(p))
            max_prob = float(p[max_idx])
            max_cls = self.class_names[max_idx]
            mini_cls = self.class_names[int(np.argmax(p_minilm[i]))]
            large_cls = self.class_names[int(np.argmax(p_deberta[i]))]

            status = "HUMAN_REVIEW"
            rule = ""

            if mini_cls != large_cls:
                rule = "Rule 5 (Models Disagree)"
            elif prob_dict["against"] > self.thresholds["against_binary_threshold"]:
                status = "AUTO_ANNOTATED"
                max_cls = "against"
                rule = "Rule 1 (Against Specialist)"
            elif prob_dict["undetermined"] > self.thresholds["undetermined_rejection_threshold"]:
                rule = "Rule 2 (Undetermined Trap)"
            elif max_prob >= self.thresholds["global_tau"]:
                status = "AUTO_ANNOTATED"
                rule = "Rule 3 (Global Confidence)"
            else:
                rule = "Rule 4 (Low Confidence)"

            results.append(
                {
                    "index": i,
                    "status": status,
                    "label": max_cls if status == "AUTO_ANNOTATED" else None,
                    "confidence": max_prob,
                    "probabilities": prob_dict,
                    "minilm_label": mini_cls,
                    "deberta_label": large_cls,
                    "rule_triggered": rule,
                    "stage": "Duo_Zero_Faute",
                }
            )
            if status == "HUMAN_REVIEW":
                rejected_indices.append(i)

        if rejected_indices:
            rejected_pairs = [sentence_pairs[i] for i in rejected_indices]
            logits_reranker = self.models["reranker"].predict(
                rejected_pairs, batch_size=32, show_progress_bar=False
            )
            probs_reranker = expit(np.asarray(logits_reranker)).flatten()
            reranker_thresh = float(
                self.thresholds.get("reranker_undetermined_threshold", 0.70)
            )
            for list_idx, original_idx in enumerate(rejected_indices):
                prob_undet = float(probs_reranker[list_idx])
                results[original_idx]["reranker_undetermined_prob"] = prob_undet
                if prob_undet >= reranker_thresh:
                    results[original_idx]["status"] = "AUTO_ANNOTATED"
                    results[original_idx]["label"] = "undetermined"
                    results[original_idx]["confidence"] = prob_undet
                    results[original_idx]["rule_triggered"] = (
                        "Reranker_Undetermined_Detected"
                    )
                    results[original_idx]["stage"] = "Reranker"

        return results

    def annotate_one(self, anchor: str, target: str) -> dict:
        """Raccourci pour annoter une seule paire (ancre, cible).

        Args:
            anchor: Texte de l'article de référence.
            target: Texte de l'article candidat.

        Returns:
            Dict de résultat identique à un élément de ``annotate_pairs``.
        """
        return self.annotate_pairs([[anchor, target]])[0]
