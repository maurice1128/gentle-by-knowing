"""Round 23: wide end-feel (W) and realistic care-record quality (R).
PREREGISTRATION section 10, round 23.

W: n=120, conditions soft12 / soft20 / soft25; arms F, reactive x4 guards, model M (4 deg record), screen S_C.
R: n=600, hard limits; record error 4 / 10 / 20 deg and categorical flags; arms F, M(e), S_C(e), S_B(e), CAT, CAT_half.
Harm (primary) = robot-caused end-range load: per joint max(0, peak - pre-contact baseline); harmful at 0.5 AND 2.5 N m.
Stages: test | eval
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round23")
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import round17 as r17
import round18 as r18
from round17 import UNRESTRICTED, person
from round21 import _pol, choose, MULTS, EQUIV

CONDS_W = {"soft12": 12.0, "soft20": 20.0, "soft25": 25.0}
ERRS = (4.0, 10.0, 20.0)
THRS = (0.5, 2.5)
N_W, N_R = 120, 600


def _seeds():
    v = json.load(open(os.path.join(TOP, "out", "round22", "test_pool_valid.json")))
    return v[600:600 + N_R]


def _job(spec):
    import shoulder_task as st
    cond, kind, seed, fs, fe, o, a, sp, guard = spec
    try:
        r = st.run(seed, fs, fe, o, a, sp, stop_force=guard, soft_deg=(None if cond == "hard" else CONDS_W[cond]))
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r.update(cond=cond, kind=kind, tag=guard)
    return r


def _lib():
    lib = r18._load_library()
    return r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)


def pick(seed, F, P, err):
    rec = r18.recorded(seed, err)
    if min(rec) >= UNRESTRICTED:
        return F
    return choose(P, rec, F)


def test():
    F, order, guard = _pol()
    smoke = os.environ.get("WM_SMOKE") == "1"
    R = [100, 104, 105, 109] if smoke else _seeds()
    W = [104, 105, 109] if smoke else R[:N_W]
    tag = "_smoke" if smoke else ""
    json.dump(dict(W=W, R=R), open(os.path.join(OUT, f"seeds{tag}.json"), "w"))
    path = os.path.join(OUT, f"test{tag}.json")
    if os.path.exists(path) and not smoke:
        print("exists", path); return
    P = _lib()
    jobs = set()
    for s in R:
        for k in {F} | {pick(s, F, P, e) for e in ERRS} - {None}:
            jobs.add(("hard", "truth", s) + person(s) + k + (None,))
    for s in W:
        for c in CONDS_W:
            for k in {F, pick(s, F, P, 4.0)} - {None}:
                jobs.add((c, "truth", s) + person(s) + k + (None,))
            for k in order:
                for m in MULTS:
                    if guard.get(k):
                        jobs.add((c, "guard", s) + person(s) + k + (round(guard[k] * m, 3),))
    jobs = sorted(jobs, key=lambda j: (j[2], j[0], j[1]))
    rows = r17._pool(_job, jobs, f"round 23 (W {len(W)} x 3 conditions, R {len(R)} hard)")
    json.dump(dict(rows=rows), open(path, "w"))


def evaluate():
    import numpy as np
    from round14_recompute import ci, fmt
    F, order, guard = _pol()
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    S = json.load(open(os.path.join(OUT, f"seeds{tag}.json")))
    rows = json.load(open(os.path.join(OUT, f"test{tag}.json")))["rows"]
    P = _lib()
    T, Gd = {}, {}
    err = sum("error" in x for x in rows)
    for x in rows:
        if "error" in x:
            continue
        k = (x["offset"], x["angle"], x["speed"])
        if x["kind"] == "truth":
            T.setdefault(x["cond"], {}).setdefault(x["seed"], {})[k] = x
        else:
            Gd.setdefault(x["cond"], {}).setdefault(x["seed"], {}).setdefault(k, {})[round(x["tag"], 3)] = x
    print(f"round 23 {'SMOKE' if tag else ''}: {len(rows)} episodes, {err} errors; W {len(S['W'])} persons, R {len(S['R'])} persons; F = {F}")

    def net(x):
        return max(0.0, x["lim_sh"] - x.get("lim_base_sh", 0.0)) + max(0.0, x["lim_el"] - x.get("lim_base_el", 0.0))

    def gross(x):
        return x["lim_sh"] + x["lim_el"]

    AB = dict(done=False, net=0.0, gross=0.0, abstain=True)

    def out(c, s, k):
        if k is None:
            return dict(AB)
        x = T[c][s][k]
        return dict(done=bool(x["done"]), net=net(x), gross=gross(x), abstain=False)

    def reactive(c, s, m):
        o = dict(done=False, net=0.0, gross=0.0, abstain=False, first=None)
        for i, k in enumerate(order):
            g = Gd.get(c, {}).get(s, {}).get(k, {}).get(round(guard[k] * m, 3))
            if g is None:
                continue
            o["net"] = max(o["net"], net(g)); o["gross"] = max(o["gross"], gross(g))
            if i == 0:
                o["first"] = net(g)
            if g["done"]:
                o["done"] = True
                return o
        return o

    bad = lambda o, t, key="net": o[key] >= t
    safe = lambda o, t, key="net": o["done"] and o[key] < t

    def show(people, res, t, key="net"):
        print(f"  harm = {key}, harmful if >= {t} N m")
        print(f"    {'arm':16s} {'safe':>5} {'harmful':>8} {'abstain':>8} {'mean':>7} {'p90':>7}")
        for a, R in res.items():
            hv = [R[s][key] for s in people]
            print(f"    {a:16s} {sum(safe(R[s], t, key) for s in people):5d} {sum(bad(R[s], t, key) for s in people):8d} "
                  f"{sum(R[s]['abstain'] for s in people):8d} {np.mean(hv):7.2f} {np.percentile(hv, 90):7.2f}")

    # ---------------- W
    W = S["W"]
    for c in CONDS_W:
        print(f"\n=== W, condition {c} (n={len(W)}) ===")
        res = {"F": {s: out(c, s, F) for s in W}, "M": {s: out(c, s, pick(s, F, P, 4.0)) for s in W},
               "S_C (3.5)": {s: (dict(AB) if min(r18.recorded(s, 4.0)) < 3.5 else out(c, s, F)) for s in W}}
        for m in MULTS:
            res[f"R x{m * 1.2:.1f}p95"] = {s: reactive(c, s, m) for s in W}
        for t in THRS:
            show(W, res, t)
            any_help = False
            for m in MULTS:
                a = f"R x{m * 1.2:.1f}p95"
                d = ci([float(bad(res[a][s], t)) - float(bad(res["F"][s], t)) for s in W])
                Rb = [s for s in W if bad(res[a][s], t)]
                fh = sum(1 for s in Rb if res[a][s]["first"] is not None and res[a][s]["first"] >= t)
                if d is None:
                    continue
                helps = d[2] < -EQUIV; any_help |= helps
                print(f"    R - F {a:12s} harmful rate {fmt(d)}  safe {sum(safe(res[a][s], t) for s in W)}  harmed-in-first-attempt {fh}/{len(Rb)}{'  <- HELPS' if helps else ''}")
            d = ci([float(bad(res["M"][s], t)) - float(bad(res["F"][s], t)) for s in W])
            nM, nF = sum(safe(res["M"][s], t) for s in W), sum(safe(res["F"][s], t) for s in W)
            if c != "soft12" and d is not None:
                print(f"    H23-1 [{c}, {t}] force guard never helps by > 5 points: {'SUPPORTED' if not any_help else 'NOT SUPPORTED'}")
                print(f"    H23-2 [{c}, {t}] M - F harmful rate {fmt(d)}, safe {nM} vs {nF} -> {'SUPPORTED' if (d[2] < 0 and nM >= nF - 3) else 'NOT SUPPORTED'}")
            elif d is not None:
                print(f"    [soft12 bridge, {t}] M - F harmful rate {fmt(d)}, safe {nM} vs {nF}")
        show(W, res, 0.5, key="gross")
        pre = [T[c][s][F].get("lim_base_sh", 0) + T[c][s][F].get("lim_base_el", 0) for s in W]
        print(f"  persons with passive end-range tension >= 0.5 N m before contact: {sum(p >= 0.5 for p in pre)}/{len(W)}")

    # ---------------- R
    R = S["R"]
    print(f"\n=== R, hard limits, record quality (n={len(R)}) ===")
    rng = np.random.default_rng(2323)
    documented = {s: bool(rng.random() < 0.5) for s in R}
    limited = {s: min(person(s)) < UNRESTRICTED for s in R}
    res = {"F": {s: out("hard", s, F) for s in R}}
    for e in ERRS:
        res[f"M({e:g})"] = {s: out("hard", s, pick(s, F, P, e)) for s in R}
        res[f"S_C({e:g})"] = {s: (dict(AB) if min(r18.recorded(s, e)) < 3.5 else out("hard", s, F)) for s in R}
        res[f"S_B({e:g})"] = {s: (dict(AB) if min(r18.recorded(s, e)) < 10.5 else out("hard", s, F)) for s in R}
    res["CAT"] = {s: (dict(AB) if limited[s] else out("hard", s, F)) for s in R}
    res["CAT_half"] = {s: (dict(AB) if (limited[s] and documented[s]) else out("hard", s, F)) for s in R}
    for t in THRS:
        show(R, res, t)
        for e in (10.0, 20.0):
            d = ci([float(bad(res[f"M({e:g})"][s], t)) - float(bad(res["F"][s], t)) for s in R])
            nM, nF = sum(safe(res[f"M({e:g})"][s], t) for s in R), sum(safe(res["F"][s], t) for s in R)
            print(f"    H23-3 [{t}] M({e:g}) - F harmful rate {fmt(d)}, safe {nM} vs {nF} -> {'SUPPORTED' if (d[2] < 0 and nM >= nF - 15) else 'NOT SUPPORTED'}")
        for e in ERRS:
            d = ci([float(bad(res[f"M({e:g})"][s], t)) - float(bad(res[f"S_C({e:g})"][s], t)) for s in R])
            v = ("EQUIVALENT" if (d[1] > -EQUIV and d[2] < EQUIV) else "MODEL BETTER" if d[2] < 0 else "SCREEN BETTER" if d[1] > 0 else "UNDECIDED")
            print(f"    {'H23-4 ' if e != 4.0 else ''}[{t}] M({e:g}) - S_C({e:g}) harmful rate {fmt(d)} | safe {sum(safe(res[f'M({e:g})'][s], t) for s in R)} vs {sum(safe(res[f'S_C({e:g})'][s], t) for s in R)} -> {v}")
        for a in ("CAT", "CAT_half"):
            d1 = ci([float(bad(res[a][s], t)) - float(bad(res["F"][s], t)) for s in R])
            d2 = ci([float(bad(res[a][s], t)) - float(bad(res["M(20)"][s], t)) for s in R])
            print(f"    H23-5 [{t}] {a:8s} - F {fmt(d1)} | - M(20) {fmt(d2)} | safe {sum(safe(res[a][s], t) for s in R)} (F {sum(safe(res['F'][s], t) for s in R)}, M(20) {sum(safe(res['M(20)'][s], t) for s in R)})")


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(test=test, eval=evaluate).get(st)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True); f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
