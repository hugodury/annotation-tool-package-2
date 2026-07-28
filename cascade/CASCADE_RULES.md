# Cascade V8 — règles (Duo Zero Faute + Reranker + Qwen)

Package collègue : MiniLM v7 + DeBERTa Large v8.1 + Reranker undetermined (BGE).  
Dans l’app : mode **`v8_qwen`** = ces 3 modèles puis **Qwen P3 + post-LLM** si besoin.

## Classes (ordre logits)

`against` · `not_related` · `supporting` · `undetermined`

## Pondération Duo

- MiniLM : **0.40**
- DeBERTa Large : **0.60**  
→ `p_ensemble = 0.4·p_mini + 0.6·p_deberta`

## Seuils (`config.json`)

| Clé | Défaut | Rôle |
| :--- | ---: | :--- |
| `global_tau` | 0.98 | Confiance globale pour auto-annoter |
| `against_binary_threshold` | 0.35 | Spécialiste against |
| `undetermined_rejection_threshold` | 0.10 | Piège undetermined (rejette le duo) |
| `reranker_undetermined_threshold` | 0.70 | Filet undetermined |

DeBERTa Large tourne en **CPU** (NaNs observés sur GPU/MPS).

## Étape 1 — Duo Zero Faute

Pour chaque paire `(T_ref, T_n)` :

1. **Rule 5 — Models Disagree**  
   `argmax(MiniLM) ≠ argmax(DeBERTa)` → **reject** (pas d’auto).

2. **Rule 1 — Against Specialist**  
   `P_ensemble(against) > 0.35` → **AUTO `against`**.

3. **Rule 2 — Undetermined Trap**  
   `P_ensemble(undetermined) > 0.10` → **reject** (trop de masse undetermined).

4. **Rule 3 — Global Confidence**  
   `max(P_ensemble) ≥ 0.98` et accord MiniLM/DeBERTa → **AUTO** (classe argmax).

5. **Rule 4 — Low Confidence**  
   sinon → **reject**.

## Étape 2 — Reranker (filet undetermined)

Uniquement sur les paires **reject** du duo :

- Score binaire sigmoid = P(undetermined)
- Si `P ≥ 0.70` → **AUTO `undetermined`**
- Sinon → encore **HUMAN_REVIEW** côté modèles

## Étape 3 — Qwen (app `v8_qwen`)

Si toujours pas auto après duo+reranker :

1. Prompt **P3** (protocole + few-shot + HARD)
2. **Post-LLM** déterministe (`postprocess_prediction`)
3. Route app : `v8_qwen` (label écrit) ou `human` si erreur Ollama

```text
Pair
  │
  ├─ Duo MiniLM+DeBERTa ── AUTO? ──► label (v8_duo)
  │         │ reject
  ▼
  Reranker undetermined ── P≥0.70? ──► undetermined (v8_reranker)
  │         │ non
  ▼
  Qwen P3 + post-LLM ──► label (v8_qwen) ou revue humaine
```

## Fichiers

| Fichier | Rôle |
| :--- | :--- |
| `predictor.py` | Référence collègue (duo + reranker) |
| `config.json` | Seuils + chemins modèles |
| `models/minilm_full_v7` | Cross-encoder 4 classes |
| `models/deberta_large_v8.1` | Cross-encoder 4 classes (CPU) |
| `models/reranker_undetermined_v8` | BGE reranker binaire undetermined |
| `annotation-tool-package-2/cascade/v8_predictor.py` | Intégration app |
| `annotation-tool-package-2/cascade/core.py` | Mode `v8_qwen` + Qwen |
