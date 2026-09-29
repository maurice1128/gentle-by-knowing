"""Round 11: gentleness-time frontier of the lift task.  PREREGISTRATION §10.

Stages (argv[1]):
  generate   open-loop grid 40 sites x speeds {0.10,0.15,0.25,0.40,0.60} for the
             60 training seeds (out/lift_train/train_speeds.json) and the 19
             test seeds (out/lift_train/test_speeds.json), one Pool each, in
             sequence.  Set WM_SIGMA_TAPER=0.10.
  eval       learn pi_med per speed on the training grid, then on the test
             individuals: per-speed paired P (H11-1), equal-load time saving
             (H11-2), equal-time load saving (H11-3), kNN frontier (H11-4).
"""
import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "lift_train")
sys.path.insert(0, ROOT)

SPEEDS = (0.10, 0.15, 0.25, 0.40, 0.60)
TEST = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21]
TASK_OK = 0.95
F_BUDGET = 100.0
DEFAULT = (0.0, 0.0)


def generate():
    import diag_lift_oracle as dlo
    import lift_train as lt
    sc = json.load(open(os.path.join(OUT, "screen.json")))
    for seeds, name in ((sc["train_seeds"], "train_speeds"), (TEST, "test_speeds")):
        lt.generate(seeds, SPEEDS, name)
    return 0


def _grid(rows):
    G = {}
    for r in rows:
        if not r["site_ok"] or r.get("diverged", False):
            continue
        G.setdefault(round(r["speed"], 3), {}).setdefault(r["seed"], {})[(r["offset"], r["angle"])] = r
    return G


def _learn(Gs, seeds):
    """pi_med for one speed: feasible sites (>=95% complete, mean force <= budget),
    lowest median ratio to each training individual's best site."""
    import numpy as np
    sites = sorted({k for s in seeds for k in Gs.get(s, {})})
    best = None
    for st in sites:
        rows = [Gs[s][st] for s in seeds if st in Gs[s]]
        if sum(r["reached"] >= TASK_OK for r in rows) / len(seeds) < TASK_OK:
            continue
        if np.mean([r["peak_force"] for r in rows]) > F_BUDGET:
            continue
        rr = []
        for s in seeds:
            ok = [r["peak_p"] for r in Gs[s].values() if r["reached"] >= TASK_OK]
            r = Gs[s].get(st)
            if ok:
                rr.append(r["peak_p"] / min(ok) if (r and r["reached"] >= TASK_OK) else np.inf)
        m = float(np.median(rr))
        if best is None or m < best[0]:
            best = (m, st)
    return best[1] if best else None


def _ci(d):
    import numpy as np
    d = np.asarray(d, float); n = len(d)
    T = {19: 2.101, 18: 2.110, 17: 2.120, 16: 2.131, 15: 2.145, 14: 2.160, 13: 2.179, 12: 2.201}
    se = d.std(ddof=1) / math.sqrt(n); t = T.get(n, 2.2)
    return d.mean(), d.mean() - t * se, d.mean() + t * se, n


def _fmt(c):
    return f"{c[0]:+7.2f} [{c[1]:+7.2f}, {c[2]:+7.2f}] n={c[3]}"


def evaluate():
    import numpy as np
    Gtr = _grid(json.load(open(os.path.join(OUT, "train_speeds.json")))["rows"])
    Gte = _grid(json.load(open(os.path.join(OUT, "test_speeds.json")))["rows"])
    screen = json.load(open(os.path.join(OUT, "screen.json")))
    train_seeds = screen["train_seeds"]
    elbow = {r["seed"]: r["elbow"] for r in screen["rows"]}
    pol = {sp: _learn(Gtr[sp], train_seeds) for sp in SPEEDS}
    print("offline policy per speed (trained on seeds 100-160):", {sp: pol[sp] for sp in SPEEDS})

    def point(sp, s, site):
        r = Gte[sp].get(s, {}).get(site)
        if r is None or r["reached"] < TASK_OK or r["t_done"] is None:
            return None
        return r["t_done"], r["peak_p"]

    # H11-1: per-speed paired P
    print(f"\n{'speed':>5} | {'pi_med site':>12} | {'P pi_med':>8} {'P default':>9} | {'t pi_med':>8} {'t default':>9} | {'dP paired [CI]':>32} done")
    h1 = True
    for sp in SPEEDS:
        d = []; tp = []; td = []; pp = []; pd = []
        for s in TEST:
            a = point(sp, s, pol[sp]); b = point(sp, s, DEFAULT)
            if a and b:
                d.append(a[1] - b[1]); tp.append(a[0]); td.append(b[0]); pp.append(a[1]); pd.append(b[1])
        c = _ci(d) if len(d) >= 3 else None
        if c:
            h1 = h1 and c[2] < 0
        print(f"{sp:5.2f} | {str(pol[sp]):>12} | {np.mean(pp):8.1f} {np.mean(pd):9.1f} | {np.mean(tp):8.2f} {np.mean(td):9.2f} | {(_fmt(c) if c else 'n/a'):>32} {len(d)}/19")
    print(f"H11-1 (pi_med gentler at every speed): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")

    # frontier families
    def fam(site_fn):
        return {s: [(sp,) + point(sp, s, site_fn(sp, s)) for sp in SPEEDS if point(sp, s, site_fn(sp, s))] for s in TEST}
    F_pol = fam(lambda sp, s: pol[sp])
    F_def = fam(lambda sp, s: DEFAULT)

    # H11-2: equal load -> time saved.  target = default @0.10 P
    dt, ratio = [], []
    for s in TEST:
        tgt = point(0.10, s, DEFAULT)
        if not tgt:
            continue
        cands = [(t, p, sp) for sp, t, p in F_pol[s] if p <= tgt[1]]
        if not cands:
            continue
        t_best = min(cands)[0]
        dt.append(t_best - tgt[0]); ratio.append(t_best / tgt[0])
    c2 = _ci(dt)
    print(f"\nH11-2 equal load (target = default@0.10): time pi_med - default {_fmt(c2)} s; median time ratio {np.median(ratio):.2f} "
          f"(need CI < 0 and median saving >= 30%) -> {'SUPPORTED' if c2[2] < 0 and np.median(ratio) <= 0.70 else 'NOT SUPPORTED'}  ({len(dt)}/19 individuals had a feasible faster point)")

    # H11-3: equal time -> load saved.  target = default @0.25 t_done
    dp = []
    for s in TEST:
        tgt = point(0.25, s, DEFAULT)
        if not tgt:
            continue
        cands = [p for sp, t, p in F_pol[s] if t <= tgt[0]]
        if not cands:
            continue
        dp.append(min(cands) - tgt[1])
    c3 = _ci(dp)
    print(f"H11-3 equal time (target = default@0.25): P pi_med - default {_fmt(c3)} (need CI < 0) -> {'SUPPORTED' if c3[2] < 0 else 'NOT SUPPORTED'}  ({len(dp)}/19)")

    # H11-4 exploratory: kNN on settled elbow, k=10, per speed
    def knn_site(sp, s, k=10):
        nn = sorted(train_seeds, key=lambda t: abs(elbow[t] - elbow[s]))[:k]
        return _learn(Gtr[sp], nn)
    F_knn = fam(knn_site)
    print("\nH11-4 (exploratory) kNN(settled elbow, k=10) vs pi_med, per speed, paired dP:")
    for sp in SPEEDS:
        d = [dict((x[0], x) for x in F_knn[s])[sp][2] - dict((x[0], x) for x in F_pol[s])[sp][2] for s in TEST
             if sp in dict((x[0], x) for x in F_knn[s]) and sp in dict((x[0], x) for x in F_pol[s])]
        if len(d) >= 3:
            print(f"  {sp:5.2f}: kNN - pi_med {_fmt(_ci(d))}")
    # frontier table for the paper (median over individuals)
    print("\nmedian frontier (t_done s, P) over test individuals:")
    for name, F in (("default site", F_def), ("offline policy", F_pol), ("kNN policy", F_knn)):
        pts = []
        for sp in SPEEDS:
            ts = [t for s in TEST for sp_, t, p in F[s] if sp_ == sp]; ps = [p for s in TEST for sp_, t, p in F[s] if sp_ == sp]
            if ts:
                pts.append(f"{sp:.2f}: ({np.median(ts):.2f} s, {np.median(ps):.0f}, n={len(ts)})")
        print(f"  {name:15s} " + "  ".join(pts))
    json.dump(dict(policy={str(k): v for k, v in pol.items()}, speeds=SPEEDS), open(os.path.join(OUT, "pareto_policy.json"), "w"), indent=1)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if os.environ.get("WM_SIGMA_TAPER") != "0.10":
        print("WARNING: WM_SIGMA_TAPER is not 0.10 (literature calibration)")
    if stage == "generate":
        return generate()
    if stage == "eval":
        return evaluate()
    print(__doc__); return 1


if __name__ == "__main__":
    raise SystemExit(main())
