"""Round 9: train a lift strategy on simulated bodies, test on unseen ones.

PREREGISTRATION section 10, round 9.  Stages (argv[1]):

  screen   run the individual screen on seeds 0-21 (check: excludes exactly
           7, 11, 15?) and on seeds 100-259; writes out/lift_train/screen.json
  generate open-loop lifts over the full site grid (8 offsets x 5 angles) at
           0.10 and 0.25 m/s for the first 60 valid training seeds;
           writes out/lift_train/train.json
  test025  the same grid at 0.25 m/s for the 19 test individuals (needed for
           the test oracle at the planner speed); writes out/lift_train/test025.json

Learning and evaluation are in lift_train_eval.py (run after these).
All runs use the literature-calibrated tissue spread (set WM_SIGMA_TAPER=0.10).
"""
import os
import sys
import json
import math
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "lift_train")
sys.path.insert(0, ROOT)

import diag_lift_oracle as dlo

TEST_SEEDS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21]
N_TRAIN = 60
TRAIN_POOL = list(range(100, 260))
SPEEDS = (0.10, 0.25)
ELBOW_MIN = 0.3          # rad, settled elbow flexion below this = forearm hanging


def _screen_job(seed):
    import mujoco
    e = dlo.env()
    e.reset(seed=seed)
    elbow = float(e.d.qpos[e.elbow_adr])
    reasons = []
    if e.start_contact:
        reasons.append("start_contact")
    if e.reset_diverged:
        reasons.append("reset_diverged")
    if elbow < ELBOW_MIN:
        reasons.append(f"elbow {elbow:.2f} rad")
    return dict(seed=seed, valid=not reasons, reasons=reasons, elbow=elbow,
                t_mid=float(e.tissue["t_mid"]), taper_u=float(e.tissue["taper_u"]))


def _pool(func, jobs, label):
    from multiprocessing import Pool
    n_proc = int(os.environ.get("WM_PROCS", 8))
    rows = []
    t0 = time.time()
    print(f"=== {label}: {len(jobs)} jobs, {n_proc} workers ===", flush=True)
    with Pool(processes=n_proc) as pool:
        for k, r in enumerate(pool.imap_unordered(func, jobs)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def screen():
    os.makedirs(OUT, exist_ok=True)
    rows = _pool(_screen_job, list(range(0, 22)) + TRAIN_POOL, "screen")
    rows.sort(key=lambda r: r["seed"])
    ev = [r for r in rows if r["seed"] < 22]
    excl = [r["seed"] for r in ev if not r["valid"]]
    print("seeds 0-21 excluded by the rule:", excl, " (evaluation list excludes 7, 11, 15)",
          "-> MATCH" if excl == [7, 11, 15] else "-> DIFFERENT (evaluation keeps its original list)")
    for r in ev:
        if not r["valid"] or r["seed"] in (7, 11, 15):
            print(f"   seed {r['seed']}: valid={r['valid']} {r['reasons']} elbow {r['elbow']:.2f}")
    tr = [r for r in rows if r["seed"] >= 100]
    valid = [r["seed"] for r in tr if r["valid"]]
    print(f"training pool: {len(valid)}/{len(tr)} valid; first {N_TRAIN}: {valid[:N_TRAIN]}")
    json.dump(dict(rows=rows, train_seeds=valid[:N_TRAIN], rule=f"no start contact, no reset divergence, elbow >= {ELBOW_MIN} rad"),
              open(os.path.join(OUT, "screen.json"), "w"), indent=1)
    return 0


def generate(seeds, speeds, name):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, o, a, sp) for s in seeds for o in dlo.OFFSETS for a in dlo.ANGLES for sp in speeds]
    rows = _pool(dlo._job, jobs, name)
    json.dump(dict(rows=rows, seeds=seeds, speeds=speeds, offsets=dlo.OFFSETS, angles=dlo.ANGLES,
                   sigma_taper=os.environ.get("WM_SIGMA_TAPER")), open(os.path.join(OUT, f"{name}.json"), "w"))
    print("wrote", os.path.join(OUT, f"{name}.json"))
    return 0


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if os.environ.get("WM_SIGMA_TAPER") != "0.10":
        print("WARNING: WM_SIGMA_TAPER is not 0.10 (literature calibration)")
    if stage == "screen":
        return screen()
    if stage == "generate":
        sc = json.load(open(os.path.join(OUT, "screen.json")))
        return generate(sc["train_seeds"], SPEEDS, "train")
    if stage == "test025":
        return generate(TEST_SEEDS, (0.25,), "test025")
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
