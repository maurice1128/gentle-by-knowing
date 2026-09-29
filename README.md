# Gentle by knowing: code, data and protocol

Code, simulation outputs and the dated protocol for the manuscript

> **Gentle by knowing: how much must a care robot know about a person to lift their arm safely?**
> Mu-Hua (Maurice) Wang, National Yang Ming Chiao Tung University (NYCU), Taiwan.
> Brief Research Report, manuscript in preparation (2026).

**What the paper is.** In a MuJoCo/MyoSuite musculoskeletal simulation, a robot supports a simulated
person's forearm from below and lifts it. We ask how much the robot must know about the person (nothing, a
limitation flag, recorded joint ranges, or everything) to avoid pressing the shoulder or elbow into the end
of its range, and whether a contact-force threshold can replace that knowledge. **This is a simulation
study only.** There are no human participants and no hardware. "Harm" is a **mechanical exposure proxy**:
the peak 50-ms mean joint-limit constraint torque that the robot causes at the shoulder-elevation and
elbow-flexion limits. It is not a measure of pain or injury.

The study ran in confirmation rounds 17–26 plus a preregistered audit. Before each round's data were
generated, its hypotheses, arms, thresholds and sample sizes were written into a dated protocol
(`protocol/`). That protocol was kept locally and **not externally registered** (see
`protocol/PROTOCOL_INDEX.md`). Every confirmation set uses new simulated persons (random seeds).

---

## Directory layout

```
gentle-by-knowing/
├── README.md                  this file
├── LICENSE                    MIT (code/)
├── requirements.txt           simulation environment (numpy, mujoco, myosuite)
├── requirements-figures.txt   plotting / offline-analysis environment (numpy, matplotlib, pillow, mujoco)
├── CHECKSUMS.sha256           SHA-256 of protocol/PREREGISTRATION.md and every file in out/
├── code/          42 .py      simulation, round scripts, analyses, figure scripts (see "Code")
├── out/          ~54 MB      confirmation-set outputs, per-episode grids, seed lists, evaluation logs
│   ├── LICENSE-DATA.md        CC BY 4.0
│   ├── round17 … round26/     one folder per round (layout identical to the author's out/ folder)
│   ├── audit_claims.txt  sensitivity_threshold.txt  posthoc_baselines.txt  forest_numbers.txt
│   ├── figs/fig1_frames.npz   cached rendered frames for Fig. 1
│   ├── videos/                selection + per-episode metadata of the web videos (frames not included)
│   └── audit_only/round17/    outputs affected by the round-17 bug (transparency; see its README)
├── figures/                   the six paper figures, PDF + PNG
└── protocol/
    ├── PREREGISTRATION.md     dated protocol, copied unchanged (mostly Traditional Chinese)
    └── PROTOCOL_INDEX.md      English index: round → date → line of its hypotheses; amendments
```

The scripts read and write `<release root>/out/<round>/…` (paths are built relative to each script: `TOP` is the parent of `code/`), so they run directly from the release root. Scripts that regenerate outputs overwrite files under `out/`; work on a copy if you want to keep the released files pristine, and check them against `CHECKSUMS.sha256` (`sha256sum -c CHECKSUMS.sha256`).

---

## Environment

Two Python environments were used, both **CPython 3.11.15** on Windows 11 (created with `uv`):

| Environment | Used for | Pinned packages (imported by the code) |
|---|---|---|
| `requirements.txt` (author's `.venv_myo`) | all simulation, round `eval` stages, `dev_*.py`, frame capture | numpy 2.4.6, **mujoco 3.6.0**, **myosuite 2.12.2** (MyoSuite brought gymnasium 1.2.3, h5py, imageio, … as its own dependencies) |
| `requirements-figures.txt` (author's `.venv_mm`) | figures and the offline post hoc / audit analyses | numpy 1.26.4, matplotlib 3.11.0, pillow 12.3.0, mujoco 3.10.0 (only for the module-level import in `make_figure_scene.py`) |

```bash
python -m venv .venv_myo && .venv_myo/bin/pip install -r requirements.txt
python -m venv .venv_mm  && .venv_mm/bin/pip install -r requirements-figures.txt
```

The two environments cannot be merged as pinned, because they need different numpy and MuJoCo versions.
The offline analyses and all `eval` stages import neither MuJoCo nor MyoSuite. The self-test verified the
`eval` stages in the simulation environment and the analyses and figures in the figures environment.

Other requirements:
- **Rendering** (`make_figure_scene.py`, `make_web_videos_gentle.py capture`) needs an OpenGL context.
  The scripts set `MUJOCO_GL=glfw` on Windows/macOS and `egl` on Linux.
- **Video composition** (`make_web_videos_gentle.py compose`) needs `ffmpeg` on `PATH` (author: 9.0.1).
- Simulation is parallelised with `multiprocessing`. Set the number of workers with `WM_PROCS`
  (default 8; the author used 6–8). Set `OPENBLAS_NUM_THREADS=1`; most scripts set it themselves.
- The arm model is MyoSuite's `myoArm` (`myosuite/simhive/myo_sim/arm/myoarm.xml`). It is loaded from the
  installed package and is not redistributed here.

---

## Reproducing the paper

Figure numbers follow the Brief Research Report (`manuscript_brief.tex`) and its supplement.
All commands are run from the release root after creating `out` (see above). "myo" = simulation
environment, "mm" = figures environment.

### A. From the released data (no simulation, minutes)

| Paper item | Command | Env | Output | Runtime* |
|---|---|---|---|---|
| Fig. 1 (task; proximal vs distal lift) | `python code/make_figure_scene.py compose` (uses the cached `out/figs/fig1_frames.npz`) | mm | `out/figs/fig1_task.{pdf,png}` | 5 s |
| Fig. 2 (force guard) | `python code/make_figure_forceguard.py` (parses `round21/eval.txt`, `round23/eval.txt`) | mm | `out/figs/figG_force_guard.{pdf,png}` | 15 s |
| Fig. 3 (information ladder) and Table S4 | `python code/make_figure_ladder.py` (parses `round24/eval.txt`) | mm | `out/figs/figL_information_ladder.{pdf,png}` | 30 s |
| Fig. 4 (robustness forest plot) and Table S5 | `python code/make_figure_forest.py` (parses round 24–26 eval logs) | mm | `out/figs/figR_robustness.{pdf,png}`, `out/forest_numbers.txt` | 30 s |
| Fig. S1 (harm-threshold sweep, **post hoc**); threshold-free mean load (6.73 / 3.19 / 0.79 N·m) | `python code/sensitivity_threshold.py` | mm | `out/sensitivity_threshold.txt`, `out/figs/figS_threshold.{pdf,png}` | 30 s |
| Fig. S2 (trade-off curves, **post hoc**); flags-only policy, type lookup, re-tuned M/S; lifted-only comparison (423 persons, 53 vs 95) | `python code/posthoc_baselines.py` | mm | `out/posthoc_baselines.txt`, `out/figs/figP_pareto.{pdf,png}` | 4 min |
| Audit A1–A3 (fixed rule not a straw man; shear; stale records) | `python code/audit_claims.py > out/audit_claims.txt` | mm (verified; needs only numpy) | `out/audit_claims.txt` | 30 s |
| Table S1 rows H17–H26: every preregistered test | `python code/roundNN.py eval > out/roundNN/eval.txt` for NN = 17, 18, 19, 21, 22, 23, 24, 25 | myo | `out/roundNN/eval.txt` | 3–30 s each |
| Round 18R (frozen replication) | `WM_R18_REP=1 python code/round18.py eval > out/round18/eval_rep.txt` | myo | `out/round18/eval_rep.txt` | 30 s |
| Round 19 rule on the development set (not evidence) | `WM_R19_DEV=1 python code/round19.py eval > out/round19/eval_DEVSET.txt` | myo | `out/round19/eval_DEVSET.txt` | 15 s |
| Round 26 (misspecified test physics) | `WM_WORLD=S python code/round26.py eval > out/round26/eval_S.txt`; same with `WM_WORLD=V` → `eval_V.txt` | myo | `out/round26/eval_{S,V}.txt` | 5 s |
| Development analyses (not evidence): round-19 rule, round-20 hybrid, L cross-validation | `python code/dev_r19_rule.py`; `python code/dev_r20_rule.py > out/round20/dev_rule.txt`; `python code/analysis_abstain.py > out/round20/analysis_abstain.txt`; `python code/dev_r24_learned.py` | myo | stdout / files shown | 3 s; 12 s; 7 s; 9.5 min |

\*Measured during the self-test on the author's machine, single process.

The main-text statistics come from these files: guard first-attempt fractions and Spearman correlations
from `round21/22/23/eval.txt`, M vs F from `round22/eval.txt`, ladder counts from `round24/eval.txt`,
replication and degradations from `round25/eval.txt` and `round26/eval_*.txt`, audit numbers from
`audit_claims.txt`, and the post hoc numbers from `sensitivity_threshold.txt`, `posthoc_baselines.txt` and the files in the table below.

Post hoc analyses added after the internal review (each reads stored results only, seconds to minutes, no simulation):

| Paper item | Command | Output |
|---|---|---|
| Round-24 gap to the oracle (63 missed; 47 unnecessary refusals; 18 of 25 harmful with no safe strategy) | `python code/gap_round24.py` | `out/gap_round24.txt` |
| Force guard did not trip in harmful first attempts (33–79% hard limit, 42–95% graded resistance) | `python code/posthoc_guard_trips.py` | `out/posthoc_guard_trips.txt` |
| Contact force of harmful lifts vs ISO/TS 15066 lower-arm limit (160 N) | `python code/posthoc_iso_force.py` | `out/posthoc_iso_force.txt` |
| Leave-one-out re-selection of the decline threshold tau (0.60: 378 safe / 73 harmful) | `python code/posthoc_loo_tau.py` | `out/posthoc_loo_tau.txt` |

### B. Full re-simulation (hours)

Each round has stages selected by `argv[1]`. Stages that already have an output file skip it or refuse to
overwrite it, so delete the round's JSONs first if you want a clean re-run. Episode counts and wall-clock
times are taken from the round logs (`out/round*/*.log`). They were measured with 6–8 worker processes
on the author's shared Windows machine, and later rounds are slower per episode because they use soft
limits and longer runs.

| Round | Stages (in order) | Episodes | Wall clock (logged) |
|---|---|---|---|
| 17 | `screen`, `train`, `select`, `test`, `eval` | test: 7 680 (body copy) + truth + guard grids | test_copy ≈ 13 min |
| 18 | `library` (199 bodies, used by every later body-model arm), `loo`, `test`, `eval`; then `WM_R18_REP=1` `test`, `eval` | library 9 950; test 2 120; 18R 6 360 | ≈ 30 min; 6 min; 17 min |
| 19 | `test`, `eval` | 6 360 | ≈ 13 min |
| 20 (dev only) | `guards`, `devgrid` | 7 680 | ≈ 55 min |
| 21 | `select` (locks thresholds on seen persons), `test`, `eval` | 7 200 | ≈ 43 min |
| 22 | `test`, `eval` | 9 336 | ≈ 59 min |
| 23 | `test`, `eval` | 5 676 | ≈ 68 min |
| 24 | `select` (locks τ on 400 development persons), `test`, `eval` | 14 400 | ≈ 67 min |
| 25 | `select`, `test1`, `test2`, `eval` | 14 650 + 448 | ≈ 53 min + 2 min |
| 26 | per world, `WM_WORLD=S` or `V`: `stage1`, `stage2`, `eval` | per world ≈ 700 screen + 300 F + ~130 picks | a few minutes per world (partially logged) |
| Fig. 1 frames | `python code/make_figure_scene.py` (renders person 104, then `compose`) | 2 | < 1 min |
| Web videos (not in paper) | `make_web_videos_gentle.py capture NAME…` (myo), then `compose` (mm) | 6 | minutes |

Smoke tests use `WM_SMOKE=1` and write `*_smoke.json`. Those files are not released.

---

## Confirmation sets (seed ranges)

A "person" is a simulated body drawn from a random seed. "Valid" means it passed the screening in
`round17._screen_job` (via `round18._valid`): no probe contact at the start position and no solver
divergence during reset. Screening lists are in `test_pool_valid*.json` /
`*_valid.json`.

| Round | Role | Seeds actually used (n) |
|---|---|---|
| 17 | train (fixed rule F, reactive R thresholds) | 100–162 (first 60 valid of 100–199) |
| 17 | confirmation | 300–341 (first 40 valid of 300–419) |
| 18 | body library (world model, unrestricted bodies) | 100–299 (199 valid) |
| 18 | confirmation | 420–460 (40) |
| 18R | frozen replication | 560–679 (120) |
| 19 | confirmation | 680–801 (120) |
| 21 | confirmation | 802–922 (120) |
| 22 | confirmation, B (hard limit) / A (end-feel, first 120 of B) | 923–1529 (600) / 923–1043 (120) |
| 23 | confirmation, R (record quality) / W (wide end-feel, first 120 of R) | 1530–2133 (600) / 1530–1650 (120) |
| 24 | confirmation (information ladder) | 2134–2435 (300) |
| 25 | A replication / B shifted population | 2436–2738 (300) / 2739–2990 (250) |
| Audit | reuses 24 + 25 A | (600) |
| 26 | world S (reflex ×3) / world V (σ = 0.40) | 3000–3302 (300) / 4000–4308 (300) |

Rounds 22–25 draw consecutive blocks from one screening list, `round22/test_pool_valid.json`
(positions 1–600, 601–1200, 1201–1500, 1501–1800, 1801–2050).
**Development / training reuse:** rounds 18 + 18R (160 persons) were used to develop the round-19 rule.
Rounds 18, 18R and 19 (280) were used to lock the record thresholds S. Rounds 18, 18R, 19 and 21 (400) are
the training set of the learned policy L. No person was reused as confirmation data.

---

## Known deviations (summary of Supplementary Table S2)

| Item | Description |
|---|---|
| Bug, round 17 | The robot's internal body copy did not restore joint ranges between episodes. This was found at the first round-17 evaluation. The body-model arm was re-run on the same 40 persons after F and R had been seen. The affected outputs are kept in `out/audit_only/`. |
| Harm definition, round 22 | Resting passive tension was counted as harm under wide end-feel. This was found after round 22, and net (robot-caused) harm is used from round 23 on. The round-22 6° and 12° results use gross harm. The 12° condition was re-run under net harm in round 23 with the same conclusions; 4/120 persons were affected at 12°. |
| Primary outcome | Paired mean harm (rounds 17–18) was replaced by the harmful-outcome rate plus safe-completion non-inferiority (round 19 onward). |
| Record error, round 18 | The primary record error was changed from 8° to 4° after round 17 had been seen. |
| Reuse of persons | The round-19 decline rule was developed on rounds 18 + 18R (160). The S threshold was locked on rounds 18, 18R and 19 (280). L was trained on rounds 18, 18R, 19 and 21 (400). The round-21 hypotheses were motivated by post hoc analyses of round 19. All later confirmation sets are new. |
| Development only | The round-20 hybrid (body model + force guard + retries) did not improve on development data and was not confirmed. |
| Analysis code | A round-25 set-precedence bug was fixed during the smoke test, before any confirmation data. Confidence intervals use t = 2.1 for all n > 20 (`round14_recompute.ci`), which is conservative (about 6% wider than exact). |
| Post hoc | The threshold sweep, paired lifted-only comparisons, flags-only policy, type lookup, re-tuned M and S, and trade-off curves were not preregistered. |

`protocol/PROTOCOL_INDEX.md` gives the protocol line where each of these is recorded.

---

## Code

`code/` contains the 42 modules needed to reproduce the paper. The set was found by tracing local imports
recursively from the round scripts, analyses and figure scripts. Main entry points:

- **Simulation / task**: `care_env6.py` (6-DoF supported-arm scene: capsule end effector, free humerus),
  `care_env.py` (body variation), `reflex.py` (stretch reflex), `tissue.py`, `shoulder_task.py`
  (the lift task, harm measure, soft end-feel), `diag_lift_oracle.py`, `diag_shoulder1.py`,
  `arm_scene.py` / `supported_scene.py` (MyoSuite arm model loading).
- **Rounds**: `round17.py` … `round26.py`. `round14_recompute.py` provides the shared `ci` / `fmt` helpers.
- **Development (not evidence)**: `dev_r19_rule.py`, `dev_r20_rule.py`, `dev_r24_learned.py`
  (defines the learned policy `Learned` and the net harm function), `analysis_abstain.py`,
  `check_r18_discrepancy.py`.
- **Audit / post hoc**: `audit_claims.py`, `sensitivity_threshold.py`, `posthoc_baselines.py`.
- **Figures / media**: `make_figure_scene.py`, `make_figure_forceguard.py`, `make_figure_ladder.py`,
  `make_figure_forest.py`, `make_web_videos_gentle.py`.
- **Imported only as dependencies** (from earlier project phases): `crossover6.py`, `crossover8.py`,
  `degrade.py`, `gate_b_prime.py`, `knn_confirm.py`, `lift_train.py`, `pareto_lift.py`, `predictive.py`,
  `predictive6.py`. Their own `main()` functions need earlier-phase data that is not released. Note in
  particular that `python code/round14_recompute.py` as a script needs `out/lift_train`, `out/crossover8`
  and `out/crossover13`. It is released only for its helpers.

The code was copied unchanged from the author's working tree. Comments and docstrings refer to
"PREREGISTRATION section 10" (that is, `protocol/PREREGISTRATION.md`) and sometimes to lab-log files that
are not released (`notes/…`, `paper/…`).

### Hard-coded paths and environment assumptions

- **No absolute paths are used for data.** All data paths are built relative to the script
  (`TOP/out/...`). 
- `make_web_videos_gentle.py` (compose step, lines 181–182) tries the Windows fonts
  `C:/Windows/Fonts/segoeui*.ttf` / `arial*.ttf`. It falls back to PIL's default font on other systems,
  so video captions look different there. This does not affect any number.
- Docstrings mention the author's environment names (`../.venv_mm`, `../.venv_myo`). These are comments
  only.
- Environment variables that change behaviour: `WM_PROCS`, `WM_SMOKE`, `WM_R18_REP`, `WM_R19_DEV`,
  `WM_WORLD` (round 26, which sets `WM_REFLEX_SCALE` / `WM_BODY_VAR` itself), plus several `WM_*` knobs
  from earlier phases in `crossover6.py` / `crossover8.py` / `tissue.py` whose defaults are what the paper
  uses. Do not set them.
- **Data files contain the author's local path in one error message.** One screening episode (seed 108)
  failed with a transient MyoSuite PNG-decoding error. The error string, which includes
  `C:\Users/maurice/Desktop/robotic_research/.venv_myo/...`, is stored in `out/round17/screen.json`
  (1 occurrence) and `out/round17/run.log` (32 lines of the same traceback). The files are released
  byte-identical to the originals. Seed 108 was marked invalid by the screening, which was fixed before any
  confirmation data.

---

## Self-test (2026-09-29)

This was run on a scratch copy of this release (`code/` + `out/`). No simulation was run.
- **Imports**: 36 modules import cleanly under the simulation environment. The two top-level plotting
  scripts `make_figure_forceguard.py` and `make_figure_ladder.py` need matplotlib, so they were run in the
  figures environment instead. Four modules run their analysis at import time (`check_r18_discrepancy`,
  `analysis_abstain`, `dev_r19_rule`, `dev_r20_rule`); they were executed as scripts, and all four
  completed. The offline analyses import without MuJoCo/MyoSuite in the figures environment.
- **Byte-identical regeneration** (CRLF and the stderr MyoSuite banner ignored):
  `sensitivity_threshold.txt`, `posthoc_baselines.txt`, `forest_numbers.txt`, `audit_claims.txt`,
  the eval logs of rounds 17, 18, 18R, 19, 19-DEVSET, 21, 22, 23, 24, 25, 26 S and 26 V, and
  `round20/dev_rule.txt` and `round20/analysis_abstain.txt`. `dev_r19_rule.py` and `dev_r24_learned.py`
  reproduce the development numbers quoted in the protocol rows (94/94/97 safe, 42/61/61 harmful;
  L 240 safe / 30 harmful of 400).
- Figures 1 (compose step from the cached frames), 2–4, S1 and S2 were regenerated from the data without
  errors. All six regenerated PNGs are pixel-identical to the released copies in `figures/`.

---

## Licenses

- Code (`code/`): **MIT** (`LICENSE`).
- Data, figures and protocol (`out/`, `figures/`, `protocol/`): **CC BY 4.0** (`out/LICENSE-DATA.md`).
- MyoSuite and MuJoCo are third-party dependencies (Apache-2.0) and are not redistributed here.

*Note to the author: these licenses are placeholders chosen for the release draft and may be changed
before publishing.*

---

## Citation

```bibtex
@misc{wang2026gentle,
  author       = {Wang, Mu-Hua (Maurice)},
  title        = {Gentle by knowing: how much must a care robot know about a person to lift their arm safely?},
  year         = {2026},
  note         = {Manuscript in preparation. Code and data: https://github.com/<user>/gentle-by-knowing},
  institution  = {National Yang Ming Chiao Tung University}
}
```

Replace the URL (and add a Zenodo DOI) once the repository is published.
