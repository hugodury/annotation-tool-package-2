# VLDBench annotation protocol

Labels (exact): `supporting`, `against`, `undetermined`, `not_related`, `dismissed`.
JSON fields to set: only `related` and `similarity_annotation`.

## Similarity scale
- **1.0** identical / paraphrase · **0.7–0.9** same event, minor extra detail · **0.4–0.6** same story, different angle/sequel · **0.1–0.3** weak topical link · **0.0** none.

## Score ↔ label
- sim ≤ 0.2 → MUST `not_related` (only if no shared brand/topic).
- 0.2 < sim ≤ 0.3 → `not_related` if weak; shared entity/theme → `undetermined`.
- sim > 0.3 → NEVER `not_related`.

## Method
1. Read $T_{ref}$ (event, entities, claim X).
2. Each $T_n$ alone: is X **true**, **false**, or **neither**?
3. Score overlap **before** the label — no fake 0.1 to unlock `not_related`.

## Labels
- **supporting** — same claim/POV clearly (paraphrase, same fact, same event + commentary).
- **against** — exact opposite claim. Not mere negative wording.
- **undetermined** — related theme/brand/policy but no clear stance (angles, polls, SKUs, city variants of same list).
- **not_related** — zero topical bridge. Different city/person alone is NOT enough if theme/brand shared.
- **dismissed** — UI non-news only; `similarity_annotation: null`.

## Same basis + extra detail
Aligning extra detail → `supporting`. Nullifying detail → `against`.

## Guardrails (mandatory)
1. Polarity ≠ against (Demiral suspension paraphrase → supporting).
2. Theme ≠ supporting (Biden 33% vs 66% → undetermined).
3. No sim crush to force not_related when mid overlap exists.
4. Fact + commentary on same event → supporting.
5. Shared brand / person / event / policy (Starbucks, tips tax, Serbia lithium, Russia space) → NEVER `not_related`.
6. Soft undetermined trap: truly separate events, no shared claim (Mets vs Nats; DWTS A vs B) → `not_related`.
7. Same product/list family (backpacks; Best Lawyers City A vs City B) → `undetermined`.
8. Use supporting when the claim match is clear.

## Decision order
1. Guardrails + honest overlap score.
2. Shared brand/topic → never `not_related`.
3. No real bridge → `not_related`.
4. Exact contradiction → `against`.
5. Same claim clear → `supporting`.
6. Related theme, unclear stance → `undetermined`.
