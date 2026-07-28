# Revue experte — 505 erreurs Qwen vs dataset_valid

## Verdict global
- **Qwen clairement faux** (surtout against→NR) : majorité des 268 against→not_related. Gold cohérent (sim_g souvent 0.7–0.9, ¬P clair).
- **Gold discutable** : une partie des `supporting` sur UBI « delusion/trap » ↔ « bold solution », Musk jobs take↔increase workforce — ce sont des **against** au sens P/¬P. On suit le protocole, pas le gold fautif.
- **Ambigus** : supporting↔undetermined sur digests (Bulletin, highlights) ; against soft sans ¬P lexical.

## Correctifs déployés
1. Protocole + HARD : never NR si même P avec outcomes opposés ; exemples against typiques.
2. Few-shot against (G7 fastest↔worst, student loan block↔resume, CA wage kill↔didn't).
3. Post-LLM `_VALID_POLAR_PAIRS` + ne plus écraser against via sim basse.
4. Templates NR assouplis (Wordle/crossword seulement ; price-today → undetermined).
5. Faux ponts lexicaux (William, video of X≠Y, chaos, mascots…) → NR forcé.

## Métriques offline — 2ᵉ passe (n=1624, titres via `pair_index`)
| | Brut | Post-LLM |
|---|---:|---:|
| Accuracy | 68,9 % | **83,5 %** |
| Against recall | 0,13 | **0,69** |
| Against F1 | 0,23 | **0,80** |
| against→NR restent NR | 268 | **0** (152→against, 116→undet) |

Holdout relations (n=352) : **86,4 %**. Unanimité 5/6 : **98,0 %**.

**Piège évaluateur** : ne pas matcher via `anchor_id` (souvent `null`) ni `ref.news` — utiliser `pair_index → dataset_valid[i]`.
