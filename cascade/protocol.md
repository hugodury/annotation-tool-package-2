# VLDBench annotation protocol (short)

Labels: `supporting` | `against` | `undetermined` | `not_related` | `dismissed`
Set only `related` and `similarity_annotation`.

## One test before every label
1. Extract claim/event **P** from $T_{ref}$ in one short sentence.
2. Does $T_n$ report the **same P** (paraphrase, sequel, cause→effect, commentary on P)? → `supporting`
3. Does $T_n$ assert **¬P** (exact opposite outcome / incompatible fact on the **same** P)? → `against`
4. Same person/brand/theme but **not** same P and **not** ¬P? → `undetermined`
5. No real topical bridge? → `not_related`

## Similarity
1.0 paraphrase · 0.7–0.9 same event · 0.4–0.6 same theme different angle · ≤0.3 weak / none  
sim ≤ 0.2 + no bridge → `not_related` · **sim > 0.3 → never `not_related`**  
If lexical overlap is weak but sim is already mid/high, prefer `undetermined` (same theme, different claim) over `not_related`, unless Wordle/crossword/homonym/filler.

## against — critical
`against` when outcomes **clash** on the same policy/event/entity — even across outlets/dates.

True against examples:
- UK *second-fastest* G7 growth ↔ UK *worst-performing* G7
- inflation *rise* ↔ inflation *moderates* / *best news*
- student-loan forgiveness *blocked* ↔ *resumes/allowed*
- CA $20 wage *killed jobs* ↔ *didn't kill jobs*
- ceasefire *passes* ↔ *vetoes/fails*
- TikTok ban *bad idea* ↔ ban *at its Best*

**Never** `not_related` if that shared policy/event has opposite outcomes.

**Not against**: negative tone / backlash / “fury/heat/debate” on the **same** policy without a clear ¬P → `undetermined` (or `supporting` if same unfolding story).

If unsure supporting vs against without clear ¬P → `supporting`.  
If unsure against vs undetermined (no clear ¬P) → `undetermined`.

## supporting
Use `supporting` only when **same claim P is explicit**:
- near-duplicate / paraphrase
- clear sequel or cause→effect on the **same** event
- same concrete dossier update (same shortlist / interview / recruitment claim)
- same survey series framing (e.g. Pew “Views of …” / trust-in-science wave)

Do **not** use supporting for “same person / same election / same show” alone.

## undetermined
Default when there is a **real topical bridge** but **not** the same P and **not** ¬P:
- same person/brand/war/policy/show, different angles or sub-questions
- overlapping names without a shared claim phrase (e.g. same figure, different dispute)
- backlash vs positioning; poll issues vs issue comparison
- shared weak lexical cue only (chaos, video, elite, donors…) with mid similarity → still undetermined, not NR
- sim in (0.3, 0.5] with little/no token overlap but thematic proximity → undetermined, **not** NR

## not_related
**Zero bridge only.** Wordle #N vs #M / crossword templates. Homonymy (Trump ticket ≠ Trump movie). Sports fillers Mets vs Nationals.  
If you already assign sim > 0.3, do **not** choose `not_related`.

## Decision order
bridge? → ¬P? → same P? → undetermined → not_related

## Post-LLM (automatic)
Code may fix false NR when sim>0.3, recover clear same-P supporting, and block false against without ¬P. Still apply the P/¬P test yourself.
