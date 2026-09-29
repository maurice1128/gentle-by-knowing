# Protocol index (English guide to `PREREGISTRATION.md`)

`PREREGISTRATION.md` is the dated protocol document, copied here **unchanged** (SHA-256 in `../CHECKSUMS.sha256`).
It is written mostly in Traditional Chinese. This index tells an English reader where each round's
hypotheses were written.

**Status of the document (please read).** The protocol was kept locally by the author and was **not**
registered with an external registry (OSF, AsPredicted, etc.). The project folder was not under version
control. Each row is dated by the author, and each row says which data had and had not been seen when it was
written. Nothing external proves when a row was written. The SHA-256 hash only fixes the file as it was
when this release was assembled (2026-09-29).

## How the document is organised

- **Sections 0–9** (lines 1–576) are the original protocol for an **earlier phase** of the project:
  predictive vs. constraint-based gentleness, hypotheses H1–H3, and the fidelity crossover. They are **not**
  the hypotheses tested in this paper. They are kept because §0 sets the freeze and amendment rules
  (§0.1: after the freeze, every change is appended to §10 as "date / what changed / why / what data had
  been seen"; the main text is never rewritten).
- **Section 10 "Amendments"** (starts at line 577) is a table with one row per protocol entry. Its columns
  are *date (round, when written)*, *what was changed / specified*, *why*, and *data already seen*. Rows are
  **not** in strict chronological order: new rows were inserted near the top of the table.
- Rows for rounds 7–16 (lines 584–593 and 606–622) belong to earlier task definitions (tissue load, lift
  planner). They are not used in this paper.

## Rounds used in the paper (rounds 17–26 + audit)

Line numbers refer to `PREREGISTRATION.md` in this folder. Each round has a single table row, and the
hypotheses are inside that row.

| Round | Date written | Line | Written before | Hypotheses in that row | Confirmation set |
|---|---|---|---|---|---|
| 17 | 2026-09-20 | 605 | any round-17 data | H17-1 (primary: M_8 − F mean harm < 0), H17-2 (vs reactive R), H17-3 (record-error tolerance ε = 0/4/8/16°), H17-4 (restricted persons only), H17-5 (gap to oracle). Also fixes the population (shoulder p = 0.4, U(4,16)°; elbow p = 0.3, U(0,10)°), the 48-strategy space, harm definition, train/test seed split, and arms F / R / M_ε / O | seeds 300–419, first 40 valid |
| 17 (bug amendment) | 2026-09-21 | 604 | seeing any corrected M-arm result | none new. Documents the body-copy joint-range bug; only `test_copy` was re-run. F, R and O results had already been seen (listed in the row) | same 40 |
| 18 | 2026-09-21 | 603 | any round-18 test-person data | H18-1 (primary: M_id(4°) − F mean harm < 0, completions not fewer than F − 2), H18-2 (vs R), H18-3 (identification vs population model), H18-4 (ε = 0/4/8°), H18-5 (probe harm, gap to O). **Also records the change of the primary record error from 8° to 4° after round 17 had been seen** | seeds 420–559, first 40 valid |
| 18R | 2026-09-21 | 602 | any 18R data | H18R-1..H18R-4: frozen direct replication of H18-1..H18-4; interpretation rule fixed in advance (H18R-1 fails, so round 18 may not be claimed) | seeds 560–999, first 120 valid |
| 19 | 2026-09-21 | 594 | any round-19 data | H19-1 (primary: harmful-outcome rate M_id − F < 0 **and** safe completions ≥ F − 3), H19-2 (vs R), H19-3 (mean harm), H19-4 (value of probing: M_id vs M_pop), H19-5 (by restriction type, declines, time). **Introduces the outcome classes (safe completion / harmful outcome ≥ 0.5 N·m) and the non-inferiority condition, replacing paired mean harm as primary outcome.** Declares rounds 18 + 18R (160 persons) as development data only | seeds 680+, first 120 valid |
| 20 | – | – | – | **No protocol row.** Development only (hybrid model + force guard + retries). It was not confirmed because development data showed no improvement. See `out/round20/`. | – |
| 21 | 2026-09-23 | 601 | any round-21 data | H21-1 (primary, equivalence ±0.05: model M vs record threshold S_C 3.5°), H21-2 (routing without declining no better than F; null), H21-3 (no force guard reduces harm by significantly > 5 points), H21-4 (within-person Spearman(force, load) < 0.5). Thresholds 2 / 10.5 / 3.5° locked on 280 seen persons | seeds 802+, first 120 valid |
| 22 | 2026-09-24 | 600 | any round-22 data | H22-1 (H21-3 under soft end-feel 6° and 12°), H22-2 (M − F under soft end-feel), H22-3 (Spearman under soft end-feel), H22-4 (primary, equivalence M vs S_C, n = 600). Defines the soft-limit implementation | valid list positions 1–600 (seeds 923–1529); part A = first 120 |
| 23 | 2026-09-24 | 599 | any round-23 data | H23-1 (force guard at 20°/25° end-feel, both thresholds), H23-2 (M − F at 20°/25°), H23-3 (M − F with 10°/20° record error), H23-4 (equivalence M vs S_C), H23-5 (categorical flags, reported). **Changes the primary harm measure to robot-caused (net, baseline-subtracted) end-range load, because round 22 counted resting passive tension as harm. Adds the 2.5 N·m threshold.** | valid list positions 601–1200 (seeds 1530–2133); W = first 120 |
| 24 | 2026-09-24 | 598 | any round-24 data | H24-1 (primary: L(20) − M(20) harmful rate < 0 and safe-completion non-inferiority −0.02), H24-2 (same at ε = 10°), H24-3 (L(20) vs F, margin −0.025), H24-4 (information ladder, descriptive). Freezes L (kNN, K = 25, features, τ = 0.65 locked in `out/round24/policy.json` on 400 development persons) | valid list positions 1201–1500 (seeds 2134–2435) |
| 25 | 2026-09-24 | 597 | any round-25 data | H25-1 (replication L vs M and F), H25-2 (half of restrictions unrecorded), H25-3 (posture noise SD 5° / no posture input, L_norest τ = 0.60 locked in `out/round25/policy.json`), H25-4 (shifted population), H25-5 (20° end-feel) | A: positions 1501–1800 (seeds 2436–2738); B: positions 1801–2050 (seeds 2739–2990, shifted population) |
| Audit | 2026-09-25 | 596 | computing any audit number | A1 (is the fixed rule a straw man), A2 (load shift to glenohumeral shear), A3 (stale records +5°/+10°). Each has a decision rule that fixes the paper's wording | reuses rounds 24 + 25 A (600 persons) |
| 26 | 2026-09-25 | 595 | any round-26 data | H26-1 (L − F harmful rate < 0 in both misspecified worlds), H26-2 (L − M < 0). World S = reflex gain ×3 (`REFLEX_GAIN_SCALE` 2.0 → 6.0); world V = body variation σ 0.25 → 0.40 | S: seeds 3000+, first 300 valid; V: seeds 4000+, first 300 valid |

## Amendments and deviations (where they are written)

| Deviation (paper, Supplementary Table S2) | Where recorded |
|---|---|
| **Round-17 implementation bug.** The robot's internal body copy did not restore joint ranges between episodes, so the body-model arm was never really executed. Only `test_copy` was re-run, after F, R and O had been seen. | Line 604 (row "round 17 修正"). The affected outputs are in `out/audit_only/round17/`. |
| **Primary record error 8° → 4°** in round 18, decided after round 17 had been seen | Line 603 (round 18 row, stated explicitly) |
| **Primary outcome change**: paired mean harm (rounds 17–18) → harmful-outcome rate + safe-completion non-inferiority (round 19 onward) | Line 594 (round 19 row, "結果分類") |
| **Harm definition change**: gross → net (robot-caused) end-range load from round 23. Round 22 soft-12° results used the gross measure; a round-23 bridge re-run was done at 12°. | Line 599 (round 23 row, "傷害量修正(事前)") |
| **Reuse of persons as development/training data** (160 persons for the round-19 rule; 280 for the S thresholds; 400 for L) | Lines 594, 601, 598 |
| **Round 20 not confirmed** (development only) | No protocol row. Lab log only (not released). The data are in `out/round20/`. |
| **Round-25 analysis-code bug** (operator precedence in `set | {F} - {None}`). Fixed during the smoke test, before any confirmation data. | **Not in the protocol.** Recorded in the author's lab log (`notes/PHASE0_LOG.md` §10.76, not part of this release). |
| **Confidence intervals use t = 2.1 for all n > 20** (conservative, about 6% wider than exact) | Not a protocol row. It is in the analysis code (`code/round14_recompute.py`, imported as `ci`). |
| **Post hoc analyses** (threshold sweep, lifted-only comparisons, flags-only policy, type lookup, re-tuned M and S, trade-off curves) | Not preregistered. `code/sensitivity_threshold.py` and `code/posthoc_baselines.py` say so in their docstrings. |
