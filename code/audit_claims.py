"""Preregistered audit (PREREGISTRATION section 10, 'audit' row) on the 600 confirmation persons with
full 48-strategy grids (round 24 + round 25 A, hard limits).
A1 is the fixed rule a strawman?  A2 does L shift load to glenohumeral shear?  A3 stale records
(systematic over-estimation of free range) for L and M."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person
from round21 import _pol, choose
from dev_r24_learned import load_train, harm
from round25 import KNN, record

THR = 0.5


def grids():
    T = {}
    for path in (("round24", "test.json"), ("round25", "test1.json")):
        for x in json.load(open(os.path.join(TOP, "out", *path)))["rows"]:
            if "error" in x or x.get("cond", "hard") != "hard":
                continue
            T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    return {s: g for s, g in T.items() if len(g) == len(STRATS)}


def main():
    F, _, _ = _pol()
    T = grids(); people = sorted(T)
    tr = load_train(); Ls = KNN(tr, sorted(tr), True)
    tau = json.load(open(os.path.join(TOP, "out", "round25", "policy.json")))["tau_std"]
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    print(f"audit set: {len(people)} confirmation persons with full grids (round 24 + round 25 A)")

    def rest(s):
        x = T[s][F]
        return (x["rest_sh_deg"], x["rest_el_deg"])

    def outcome(picks):
        safe = bad = dec = 0; shear = []
        for s in people:
            k = picks[s]
            if k is None:
                dec += 1; continue
            x = T[s][k]; h = harm(x)
            safe += x["done"] and h < THR; bad += h >= THR; shear.append(x["gh_shear"])
        return safe, bad, dec, shear

    L_picks = {s: Ls.act(record(s, person(s)), rest(s), tau, F) for s in people}
    M_picks = {s: (F if min(record(s, person(s))) >= UNRESTRICTED else choose(P, record(s, person(s)), F)) for s in people}
    sL, bL, dL, shL = outcome(L_picks)
    sF, bF, dF, shF = outcome({s: F for s in people})
    sM, bM, dM, shM = outcome(M_picks)
    print(f"reference: F safe {sF} harmful {bF} | M safe {sM} harmful {bM} declined {dM} | L safe {sL} harmful {bL} declined {dL}")

    # A1
    print("\n--- A1: every fixed strategy on these persons (hindsight) ---")
    rows = []
    for k in STRATS:
        s_, b_, _, _ = outcome({s: k for s in people})
        rows.append((b_, -s_, k))
    rows.sort()
    for b_, ms, k in rows[:6]:
        print(f"   {str(k):22s} harmful {b_:4d}  safe {-ms:4d}")
    ok_safe = [r for r in rows if -r[1] >= sF - 0.025 * len(people)]
    best_any, best_ok = rows[0], (ok_safe[0] if ok_safe else None)
    print(f"   preregistered F {F}: harmful {bF}, safe {sF}; rank by harmful {[r[2] for r in rows].index(F) + 1} of {len(rows)}")
    print(f"   hindsight-best fixed (fewest harmful): {best_any[2]} harmful {best_any[0]} safe {-best_any[1]}")
    if best_ok:
        print(f"   hindsight-best fixed with safe >= F - 2.5%: {best_ok[2]} harmful {best_ok[0]} safe {-best_ok[1]}")
    ref = best_ok[0] if best_ok else best_any[0]
    verdict = ("RESTRICT CLAIM: a hindsight fixed strategy is as good as L" if ref <= bL else
               "CLAIM STANDS: best fixed strategy has > 1.5x L's harmful outcomes" if ref > 1.5 * bL else
               "PARTIAL: best fixed strategy within 1.5x of L")
    print(f"   A1 verdict: {verdict}  (L harmful {bL})")

    # A2
    print("\n--- A2: glenohumeral shear of the executed lifts (N, 50 ms peak) ---")
    for name, sh in (("F", shF), ("M", shM), ("L", shL)):
        print(f"   {name}: mean {np.mean(sh):6.1f}  median {np.median(sh):6.1f}  p90 {np.percentile(sh, 90):6.1f}  (n={len(sh)} executed)")
    sub = [s for s in people if L_picks[s] is not None]
    d = [T[s][L_picks[s]]["gh_shear"] - T[s][F]["gh_shear"] for s in sub]
    print(f"   paired on persons L lifts (n={len(sub)}): L - F mean {np.mean(d):+.1f} N, median {np.median(d):+.1f} N")
    worse = np.mean(shL) > 1.2 * np.mean(shF) or np.median(shL) > 1.2 * np.median(shF)
    print(f"   A2 verdict: {'SIDE EFFECT: L raises GH shear by > 20%' if worse else 'NO LOAD SHIFT to GH shear (within 20% or lower)'}")

    # A3
    print("\n--- A3: stale records (free range over-estimated by b degrees on restricted joints, plus N(0,20)) ---")
    base = None
    for b in (0.0, 5.0, 10.0):
        def rec_b(s):
            tr_ = person(s); r = record(s, tr_)
            return tuple(ri + b if ti < UNRESTRICTED else ri for ri, ti in zip(r, tr_))
        Lb = {s: Ls.act(rec_b(s), rest(s), tau, F) for s in people}
        Mb = {s: (F if min(rec_b(s)) >= UNRESTRICTED else choose(P, rec_b(s), F)) for s in people}
        sl, bl, dl, _ = outcome(Lb); sm, bm, dm, _ = outcome(Mb)
        if base is None:
            base = bl
        print(f"   bias +{b:4.1f} deg: L safe {sl} harmful {bl} declined {dl} | M safe {sm} harmful {bm} declined {dm}")
    print(f"   A3 verdict: {'RESTRICT: L harmful rises > 50% at +10 deg' if bl > 1.5 * base else 'L robust to +10 deg over-estimation (harmful rise <= 50%)'}  ({base} -> {bl})")


if __name__ == "__main__":
    main()
