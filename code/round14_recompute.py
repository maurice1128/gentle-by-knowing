"""Round 14 (2026-09-20): offline recomputation of the lift results under
evidence-based gentleness measures.  PREREGISTRATION section 10, round 14.

  M1  pain ratio  PR = sum_contacts F_n / F_thr(site)   (Mainz FP-0317 Table 3
      median pain-onset force; site map fixed before looking)
  M3  shear       sum_contacts F_t (same step)
  M4  impact      peak wrist force of the episode
  P   tissue load index as stored (secondary, unvalidated)
No simulation is run.  Output printed; tee to out/round14/eval.txt.
"""
import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
LT = os.path.join(TOP, "out", "lift_train")
sys.path.insert(0, ROOT)

from pareto_lift import SPEEDS, DEFAULT, TASK_OK, F_BUDGET, _grid, _learn
from knn_confirm import fresh_seeds, K

THR = dict(muscle=30.0, bone=40.0, nerve=43.0, hand=50.5, upper=41.0)
HAND = float(os.environ.get("WM_HAND_THR", THR["hand"]))
USE_MAX = os.environ.get("WM_PR_MAX", "0") == "1"
T = {20: 2.093, 19: 2.101, 18: 2.110, 17: 2.120, 16: 2.131, 15: 2.145, 14: 2.160, 13: 2.179, 12: 2.201,
     11: 2.228, 10: 2.262, 9: 2.306, 8: 2.365, 7: 2.447, 6: 2.571, 5: 2.776, 4: 3.182, 3: 4.303}


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def site_thr(u, phi, cls):
    if cls == "hand":
        return HAND
    if cls != "forearm" or phi is None:
        return THR["upper"]
    if abs(_wrap(phi)) <= 45.0 or abs(_wrap(phi - 180.0)) <= 45.0:
        return THR["bone"]
    if abs(_wrap(phi + 90.0)) <= 45.0 and u is not None and u < -0.66:
        return THR["nerve"]
    return THR["muscle"]


def measures(r):
    pr = [fn / site_thr(u, phi, cls) for u, phi, fn, ft, cls in r["contacts"]]
    return dict(PR=(max(pr) if USE_MAX else sum(pr)) if pr else 0.0,
                shear=sum(c[3] for c in r["contacts"]), impact=r["peak_force"], P=r["peak_p"],
                Fn=sum(c[2] for c in r["contacts"]))


MS = ("PR", "shear", "impact", "P")


def ci(d):
    n = len(d)
    if n < 3:
        return None
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n); t = T.get(n, 2.1)
    return m, m - t * se, m + t * se, n


def fmt(c):
    if c is None:
        return f"{'n/a':>30}"
    flag = "-" if c[2] < 0 else ("+" if c[1] > 0 else "0")
    return f"{c[0]:+8.2f} [{c[1]:+8.2f},{c[2]:+8.2f}] {flag}"


def paired(A, B, seeds):
    """A, B: seed -> row (or None).  Task-matched paired differences A - B for every measure."""
    out = {}
    ss = [s for s in seeds if A.get(s) and B.get(s) and A[s]["reached"] >= TASK_OK and B[s]["reached"] >= TASK_OK]
    for k in MS:
        out[k] = ci([measures(A[s])[k] - measures(B[s])[k] for s in ss])
    return out


def verdict(res, want_neg=True):
    """Overall rule: no measure of M1/M3/M4 significantly worse, at least one significantly better."""
    sign = (lambda c: (c[2] < 0) - (c[1] > 0)) if want_neg else (lambda c: (c[1] > 0) - (c[2] < 0))
    s = [sign(res[k]) for k in ("PR", "shear", "impact") if res[k] is not None]
    if not s:
        return "n/a"
    if min(s) < 0:
        return "WORSE on at least one evidence-based measure"
    return "GENTLER (rule met)" if max(s) > 0 else "no difference on evidence-based measures"


def show(title, res, want_neg=True):
    print(f"  {title}")
    for k in MS:
        print(f"      {k:7s} {fmt(res[k])}" + (f"  n={res[k][3]}" if res[k] else ""))
    print(f"      -> {verdict(res, want_neg)}")


def closed_loop(path, seeds, label):
    if not os.path.exists(path):
        print(f"\n[{label}] {os.path.basename(path)} not available yet"); return
    rows = json.load(open(path))["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["cond"], {})[r["seed"]] = r
    print(f"\n=== closed loop, {label} ===")
    print(f"  {'condition':14s} {'n ok':>4} " + " ".join(f"{k:>8s}" for k in MS) + f" {'sumFn':>8s}")
    for c in sorted(by):
        ok = [r for r in by[c].values() if r["reached"] >= TASK_OK]
        if ok:
            mm = [measures(r) for r in ok]
            print(f"  {c:14s} {len(ok):4d} " + " ".join(f"{sum(m[k] for m in mm) / len(mm):8.2f}" for k in MS + ("Fn",)))
    pairs = [("nomap", "popmap", False, "H14-1 nomap - popmap (want > 0)")]
    for a, b, tt in (("nomap_small", "popmap_small", "nomap_small - popmap_small (want > 0)"),):
        if a in by:
            pairs.append((a, b, False, tt))
    for a in ("trained@0.25", "knn@0.25"):
        for b in ("press@0.25", "popmap", "popmap_small"):
            if a in by and b in by:
                pairs.append((a, b, True, f"{a} - {b} (want < 0)"))
    for a, b, neg, tt in pairs:
        if a in by and b in by:
            show(tt, paired(by[a], by[b], seeds), want_neg=neg)


def open_loop(test_file, test, label):
    Gtr = _grid(json.load(open(os.path.join(LT, "train_speeds.json")))["rows"])
    Gte = _grid(json.load(open(os.path.join(LT, test_file)))["rows"])
    sc = json.load(open(os.path.join(LT, "screen.json")))
    train = sc["train_seeds"]; elbow = {r["seed"]: r["elbow"] for r in sc["rows"]}
    print(f"\n=== open loop, {label} ({len(test)} individuals) ===")
    for sp in SPEEDS:
        pol = _learn(Gtr[sp], train)
        D = {s: Gte[sp].get(s, {}).get(DEFAULT) for s in test}
        Pm = {s: Gte[sp].get(s, {}).get(pol) for s in test}
        Kn = {s: Gte[sp].get(s, {}).get(_learn(Gtr[sp], sorted(train, key=lambda t: abs(elbow[t] - elbow[s]))[:K])) for s in test}
        print(f" speed {sp:.2f}  pi_med = {pol}")
        show("H14-2 pi_med - default (want < 0)", paired(Pm, D, test))
        show("H14-3 kNN - default (want < 0)", paired(Kn, D, test))
    return Gtr, Gte, train


def learn_pr(Gs, seeds):
    """Same rule as pareto_lift._learn with PR as the objective."""
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
            ok = [measures(r)["PR"] for r in Gs[s].values() if r["reached"] >= TASK_OK]
            r = Gs[s].get(st)
            if ok and r and r["reached"] >= TASK_OK and min(ok) > 0:
                rr.append(measures(r)["PR"] / min(ok))
        if rr and (best is None or np.median(rr) < best[0]):
            best = (float(np.median(rr)), st)
    return best[1] if best else None


def exploratory(Gtr, Gte, train, test):
    print("\n=== H14-4 exploratory: strategy re-selected with PR as the objective (fresh bodies) ===")
    for sp in SPEEDS:
        pol_p = _learn(Gtr[sp], train); pol_pr = learn_pr(Gtr[sp], train)
        D = {s: Gte[sp].get(s, {}).get(DEFAULT) for s in test}
        A = {s: Gte[sp].get(s, {}).get(pol_pr) for s in test}
        B = {s: Gte[sp].get(s, {}).get(pol_p) for s in test}
        print(f" speed {sp:.2f}  pi_PR = {pol_pr}   (pi_P = {pol_p})")
        show("pi_PR - default (want < 0)", paired(A, D, test))
        show("pi_PR - pi_P (want < 0)", paired(A, B, test))


def main():
    from crossover8 import SEEDS as OLD
    print(f"thresholds {THR}, hand {HAND}, PR aggregation {'max' if USE_MAX else 'sum'}")
    closed_loop(os.path.join(TOP, "out", "crossover8", "crossover8_calib_c10.json"), OLD, "original 19 (round 8/10C data)")
    closed_loop(os.path.join(TOP, "out", "crossover13", "crossover13.json"), fresh_seeds(), "fresh 20 (round 13)")
    Gtr, Gte, train = open_loop("fresh_speeds.json", fresh_seeds(), "fresh 20 (round 12 data)")
    open_loop("test_speeds.json", OLD, "original 19 (round 11 data)")
    exploratory(Gtr, Gte, train, fresh_seeds())


if __name__ == "__main__":
    main()
