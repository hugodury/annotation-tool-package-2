# ANNOTATION PROTOCOL & RULES (VLDBench)

*Updated team protocol — homogeneous annotations. JSON label for neutral stance: `undetermined`.*

---

## 1. RULES REFERENCE

### Semantic Similarity Scale
- **1.0 (Identical):** Exactly the same information, minor paraphrase, or syndicated copy.
- **0.7 – 0.9 (High Overlap):** Same specific event and entities; only minor secondary details differ.
- **0.4 – 0.6 (Partial Overlap):** Same general story but from a different angle, a sequel, or a consequence. **Valid and expected range.**
- **0.1 – 0.3 (Low Overlap):** Broad common theme but different specific events or figures.
- **0.0 (No Relation):** No factual or topical connection.

### Position Labels (summary)
- **`supporting`:** Same argument / claim / point of view — **only if 100% sure**.
- **`against`:** **Exact** contradiction or opposite stance (including contradictory overlap).
- **`undetermined`:** Same topic or event, but **no clear** support or opposition (neutral, ambiguous, or stance impossible to tell).
- **`not_related`:** No logical connection (see threshold below).
- **`dismissed`:** Pair/section intentionally skipped (administrative state in the app, not a semantic stance).

---

## 2. CORE METHODOLOGY

### The Anchor Method
1. Read $T_{ref}$ first (event, entities, central claim).
2. Evaluate each target ($T_1$ to $T_6$) **independently**.

### Logical constraints (scores)
- **Threshold:** If `similarity_annotation` $\le$ 0.2 → `related` **MUST** be `not_related`.
- **0.3 band:** If score is around **0.3** (strictly above 0.2 and below 0.4), `not_related` is **still allowed** when the link is weak/topical only.
- **From 0.4:** If `similarity_annotation` $\ge$ 0.4 → `not_related` is **forbidden**. Use `undetermined`, `supporting`, or `against` only.
- **0 vs 0.1:** Use `0.1` only if JSON `topic` matches but there is no factual link; otherwise `0`.

### Same basis, extra information (critical rule)
When $T_{ref}$ and $T_n$ share the **same factual basis** (or very close) and one adds a detail:
- Added detail **aligns with / reinforces** $T_{ref}$ → **`supporting`** (even with extra context).
- Added detail **contradicts or nullifies** $T_{ref}$ → **`against`**.

**Mini-example (stance on an added fact):**
- $T_{ref}$: *"Guillaume just won the French Open."*
- $T_n$: *"Guillaume just won the French Open; Hugo is happy about it."* → **`supporting`**
- $T_{ref}$: *"Guillaume just won the French Open; Hugo is **not** happy about it."*
- $T_n$: *"Guillaume just won the French Open; Hugo is happy about it."* → **`against`**

---

## 3. POSITION LABELS — DEFINITIONS & EXAMPLES

### `supporting`
**Definition:** $T_n$ must **clearly support, reinforce, or validate** the **same argument, claim, or point of view** as $T_{ref}$.

**100% rule:** Use `supporting` **only when you are completely sure**. If there is any doubt → `undetermined` (not `supporting`).

**Special case — additional info:** Overlap on the supporting core, and $T_n$ adds context or extra details that **do not contradict** $T_{ref}$ → still **`supporting`**.

| Type | S1 ($T_{ref}$) | S2 ($T_n$) |
|------|----------------|------------|
| Direct support | Remote work significantly boosts productivity by eliminating long commutes. | Workers save hours by not commuting, leading to higher daily output. |
| Additional info | A carbon tax reduces industrial emissions. | A carbon tax cuts factory pollution, **and** revenue can fund green energy. (Extra detail; core still supports S1.) |

### `against`
**Definition:** $T_n$ must **clearly contradict, refute, or take the opposite stance** to $T_{ref}$ — they **oppose each other exactly** on the comparable claim.

**Special case — contradictory overlap:** Same overlap as $T_{ref}$, but added data/claims **directly oppose or nullify** it → **`against`**.

| Type | S1 ($T_{ref}$) | S2 ($T_n$) |
|------|----------------|------------|
| Direct contradiction | Nuclear energy is safe and reliable vs fossil fuels. | Nuclear power poses catastrophic risks and cannot be trusted. |
| Contradictory overlap | Trial proved the drug highly effective with **zero** side effects. | Trial showed the drug works, but **many patients had severe adverse effects.** |

### `undetermined`
**Definition:** Relationship is **unclear, neutral, or uncertain** — even when titles are semantically close (often scores **0.3–0.6**). Same topic/event but **no explicit** agreement or disagreement, or stance cannot be inferred.

| Type | S1 ($T_{ref}$) | S2 ($T_n$) |
|------|----------------|------------|
| Neutral / no stance | School board met to debate the new dress code. | Parents awaited the results of the dress code meeting. (No pro/con on the policy.) |
| Ambiguous | Stock plummeted after CEO's controversial interview. | CEO discussed future outlook in a major interview. (Context only; does not clearly support or oppose the drop claim.) |

**Do not use `undetermined` as a lazy default when a clear `supporting` or `against` fits the rules above.**

### `not_related`
No factual or topical bridge strong enough to compare claims (see score $\le$ 0.2 rule).

### `dismissed` (app state)
Use `dismissed` only when a pair or a full section is explicitly dismissed in the UI.
Required JSON output for dismissed entries:
- `"similarity_annotation": null`
- `"related": "dismissed"`

**When to dismiss — document/file-like titles:**
If a title looks more like a **document/file/section/profile page title** than an actual news headline — e.g. *"Data on Women Power"*, *"Michael Cohen's big 2024 plans"*, *"Rhinoceroses - Page 2"*, *"Artificial Intelligence - TIME"* (topic page), *"FishOutofWater - Daily Kos"* (user profile), *"Resources for Responsible Online Gaming"* (resource page) — it has no real news claim to compare and must not be force-fitted into `supporting`/`against`/`undetermined`/`not_related`.

**Scope of the dismiss — only the offending item, not its siblings:**
- If only **one** $T_n$ target has such a non-news title → dismiss **only that specific target**. The 5 other $T_n$ in the same section remain annotated normally per §3–4; **do not** dismiss them.
- If $T_{ref}$ itself is a non-news title → there is no valid comparable claim, so dismiss **all** pairs of that section (because every $T_n$ would compare to an invalid $T_{ref}$).
- If several $T_n$ inside the same section are non-news → dismiss each one individually; leave the valid $T_n$ alone.

Use the **Dismiss** button in the UI: per-pair (single target) or *Dismiss section* (only when $T_{ref}$ is the problem). The app writes `similarity_annotation: null` and `related: "dismissed"` for every dismissed pair.

---

## 4. QUICK DECISION FLOW (per pair)

1. **Score $\le$ 0.2?** → `not_related`.
2. **Score $\ge$ 0.4?** → never `not_related` (choose among `against` / `supporting` / `undetermined`).
3. **Exact contradiction or contradictory overlap?** → `against`.
4. **Same claim/POV reinforced, 100% sure?** (incl. same basis + non-contradictory extra info) → `supporting`.
5. **Otherwise** (overlap but neutral, ambiguous, or any doubt) → `undetermined`.

**Critical — do not confuse `supporting` and `undetermined`:**
- Same topic / same event / high similarity does **NOT** mean `supporting`.
- `supporting` = explicit agreement on the **same claim/POV**, only if completely sure.
- If unsure, partial overlap, or no clear stance → **`undetermined`** (default for ambiguous related pairs).
- Never default to `supporting`.

---

## 5. CALIBRATION (Gap vs `similarity_avg`)

Compare your semantic score to `similarity_avg`:
1. **If Gap $\ge$ 0.4:** Final score = `similarity_avg` shifted by **0.2** toward your semantic score.
2. **If Gap $<$ 0.4:** Keep your semantic score (max nudge **±0.1** toward average if needed).

**Stance is not decided by `similarity_avg` alone** — labels follow §3–4 first.

---

## 6. OUTPUT & AUDIT

- **Modified fields:** ONLY `similarity_annotation` and `related`.
- **Scores:** `0` and `1` as integers; else one decimal (`0.1` … `0.9`).
- **Labels (exact strings):** `supporting`, `against`, `not_related`, `undetermined`, `dismissed`.

