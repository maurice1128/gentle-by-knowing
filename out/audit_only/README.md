# Audit-only files (not used by any reported result)

These files were produced by the **first** round-17 evaluation. That run had an implementation bug: the
robot's internal body copy (`body_variation = 0`) did not restore joint ranges between episodes in
`reset()`. Person-specific limits therefore accumulated, completion inside the copy was only 2-5%, and the
body-model arm (M) fell back to the fixed rule for 39 of 40 persons. The M-arm results of this run are
invalid. The F, R and O arms do not use the copy and are unaffected.

The bug was fixed in `shoulder_task.run` (joint ranges are restored at the start of every episode). Only
`test_copy` was re-run, on the same 40 persons, after the F, R and O results of this first evaluation had
been seen. The amendment is `../../protocol/PREREGISTRATION.md` line 604. The deviation is listed in
Supplementary Table S2 of the paper.

| File | What it is |
|---|---|
| `round17/eval_BUGrange.txt` | First (invalid for M) round-17 evaluation |
| `round17/test_copy_BUGrange.json` | Body-copy grid produced with the bug |

The corrected files used in the paper are `../round17/eval.txt` and `../round17/test_copy.json`. We release
the affected files for transparency. No script in `../../code/` reads them.
