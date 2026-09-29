"""Round 12: confirmatory test of the kNN-conditioned lift strategy on FRESH
simulated bodies (never used before).  PREREGISTRATION §10, round 12.

Stages (argv[1]):
  generate   40 sites x 5 speeds for the first 20 valid seeds > 160 in
             screen.json (out/lift_train/fresh_speeds.json)
  eval       H12-1..4 on the fresh set; round-11 hypotheses reported too.
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "lift_train")
sys.path.insert(0, ROOT)

from pareto_lift import SPEEDS, DEFAULT, TASK_OK, _grid, _learn, _ci, _fmt

K = 10


def fresh_seeds():
    sc = json.load(open(os.path.join(OUT, "screen.json")))
    return [r["seed"] for r in sc["rows"] if r["seed"] > 160 and r["valid"]][:20]


def generate():
    import lift_train as lt
    seeds = fresh_seeds()
    print("fresh test seeds:", seeds)
    return lt.generate(seeds, SPEEDS, "fresh_speeds")


def evaluate():
    import numpy as np
    Gtr = _grid(json.load(open(os.path.join(OUT, "train_speeds.json")))["rows"])
    Gte = _grid(json.load(open(os.path.join(OUT, "fresh_speeds.json")))["rows"])
    sc = json.load(open(os.path.join(OUT, "screen.json")))
    train_seeds = sc["train_seeds"]
    elbow = {r["seed"]: r["elbow"] for r in sc["rows"]}
    test = fresh_seeds()
    assert not set(test) & set(train_seeds)
    pol = {sp: _learn(Gtr[sp], train_seeds) for sp in SPEEDS}

    def knn_site(sp, s):
        nn = sorted(train_seeds, key=lambda t: abs(elbow[t] - elbow[s]))[:K]
        return _learn(Gtr[sp], nn)

    def point(sp, s, site):
        r = Gte[sp].get(s, {}).get(site) if site else None
        if r is None or r["reached"] < TASK_OK or r["t_done"] is None:
            return None
        return r["t_done"], r["peak_p"]

    fams = {"default": lambda sp, s: DEFAULT, "pi_med": lambda sp, s: pol[sp], "kNN": knn_site}
    F = {name: {s: {sp: point(sp, s, f(sp, s)) for sp in SPEEDS} for s in test} for name, f in fams.items()}

    print(f"fresh test individuals: {len(test)}  (train {len(train_seeds)}, disjoint)")
    print(f"\n{'speed':>5} | {'P default':>9} {'P pi_med':>8} {'P kNN':>6} | {'kNN - pi_med':>28} | {'kNN - default':>28} | {'pi_med - default':>28}")
    h1 = h4 = True; h11_1 = True
    for sp in SPEEDS:
        rows = [(F["default"][s][sp], F["pi_med"][s][sp], F["kNN"][s][sp]) for s in test]
        rows = [r for r in rows if all(r)]
        d_kp = _ci([k[1] - p[1] for d, p, k in rows]); d_kd = _ci([k[1] - d[1] for d, p, k in rows]); d_pd = _ci([p[1] - d[1] for d, p, k in rows])
        if abs(sp - 0.25) < 1e-9:
            h1 = d_kp[2] < 0
        h4 = h4 and d_kd[2] < 0
        h11_1 = h11_1 and d_pd[2] < 0
        print(f"{sp:5.2f} | {np.mean([d[1] for d, p, k in rows]):9.1f} {np.mean([p[1] for d, p, k in rows]):8.1f} {np.mean([k[1] for d, p, k in rows]):6.1f} | "
              f"{_fmt(d_kp):>28} | {_fmt(d_kd):>28} | {_fmt(d_pd):>28}")
    print(f"\nH12-1 kNN < pi_med @0.25: {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    print(f"H12-4 kNN < default at every speed: {'SUPPORTED' if h4 else 'NOT SUPPORTED'}")
    print(f"(round 11 H11-1 on fresh set, pi_med < default at every speed: {'SUPPORTED' if h11_1 else 'NOT SUPPORTED'})")

    def equal_load(fam, label):
        dt, ratio = [], []
        for s in test:
            tgt = F["default"][s][0.10]
            if not tgt:
                continue
            c = [(pt[0], sp) for sp, pt in F[fam][s].items() if pt and pt[1] <= tgt[1]]
            if c:
                t = min(c)[0]; dt.append(t - tgt[0]); ratio.append(t / tgt[0])
        c = _ci(dt)
        ok = c[2] < 0 and np.median(ratio) <= 0.70
        print(f"{label} equal load (target default@0.10): dt {_fmt(c)} s, median time ratio {np.median(ratio):.2f} ({len(dt)}/{len(test)}) -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")

    def equal_time(fam, label):
        dp = []
        for s in test:
            tgt = F["default"][s][0.25]
            if not tgt:
                continue
            c = [pt[1] for sp, pt in F[fam][s].items() if pt and pt[0] <= tgt[0]]
            if c:
                dp.append(min(c) - tgt[1])
        c = _ci(dp)
        print(f"{label} equal time (target default@0.25): dP {_fmt(c)} ({len(dp)}/{len(test)}) -> {'SUPPORTED' if c[2] < 0 else 'NOT SUPPORTED'}")

    equal_load("kNN", "H12-2"); equal_time("kNN", "H12-3")
    equal_load("pi_med", "(H11-2 fresh)"); equal_time("pi_med", "(H11-3 fresh)")
    print("\nmedian frontier (t_done s, P):")
    for name in ("default", "pi_med", "kNN"):
        pts = []
        for sp in SPEEDS:
            v = [F[name][s][sp] for s in test if F[name][s][sp]]
            if v:
                pts.append(f"{sp:.2f}: ({np.median([x[0] for x in v]):.2f} s, {np.median([x[1] for x in v]):.0f}, n={len(v)})")
        print(f"  {name:8s} " + "  ".join(pts))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if os.environ.get("WM_SIGMA_TAPER") != "0.10":
        print("WARNING: WM_SIGMA_TAPER is not 0.10")
    if stage == "generate":
        return generate()
    if stage == "eval":
        return evaluate()
    print(__doc__); return 1


if __name__ == "__main__":
    raise SystemExit(main())
