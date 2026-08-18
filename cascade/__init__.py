"""Package cascade — pipeline d'annotation automatique VLDBench.

Architecture (Cascade V8) :
    Duo Zero Faute (MiniLM v7 + DeBERTa Large v8.1)
        → Reranker undetermined
        → Qwen 2.5-7B P3 (LLM local via Ollama)

Points d'entrée publics :
    annotate_pairs : annote une liste de paires (ancre, cible).
    load_config    : charge ``cascade/config.json`` avec surcharges ``.env``.
"""
from cascade.core import annotate_pairs, load_config

__all__ = ["annotate_pairs", "load_config"]
