"""Round 24: the information ladder, with a decision policy learned offline in simulation.
PREREGISTRATION section 10, round 24.
Stages: select (lock tau on the 400 development persons) | test (full grid on 300 fresh persons) | eval
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round24")
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import round17 as r17
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person
from round21 import _pol, choose
from dev_r24_learned import load_train, Learned, harm, evaluate, TAUS

ERRS = (4.0, 10.0, 20.0)
THRS = (0.5, 2.5)
N_TEST = 300


def _seeds():
    return json.load(open(os.path.join(TOP, "out", "round22", "test_pool_valid.json")))[1200:1200 + N_TEST]


def select():
    os.makedirs(OUT, exist_ok=True)
    F, _, _ = _pol()
    T = load_train(); seeds = sorted(T)
    nF = evaluate(T, seeds, lambda s: F)[0]
    pol = dict(train_persons=len(seeds), K=25, fixed_safe=nF, tau={})
    for e in ERRS:
        L = Learned(T, seeds, e)
        best = None
        for tau in TAUS:
            sc, bd, dc = evaluate(T, seeds, lambda s: L.act(s, T[s], e, tau, F))
            if sc >= nF - 0.025 * len(seeds) and (best is None or bd < best[1]):
                best = (tau, bd, sc, dc)
        pol["tau"][f"{e:g}"] = best[0]
        print(f"eps {e:g}: tau {best[0]}  (in-sample: safe {best[2]}, harmful {best[1]}, declined {best[3]})")
    json.dump(pol, open(os.path.join(OUT, "policy.json"), "w"), indent=1)


def _job(spec):
    import shoulder_task as st
    seed, fs, fe, o, a, sp = spec
    try:
        r = st.run(seed, fs, fe, o, a, sp)
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r["kind"] = "truth"
    return r


def test():
    smoke = os.environ.get("WM_SMOKE") == "1"
    te = [104, 105] if smoke else _seeds()
    tag = "_smoke" if smoke else ""
    json.dump(te, open(os.path.join(OUT, f"seeds{tag}.json"), "w"))
    path = os.path.join(OUT, f"test{tag}.json")
    if os.path.exists(path) and not smoke:
        print("exists", path); return
    jobs = [(s,) + person(s) + k for s in te for k in STRATS]
    json.dump(dict(rows=r17._pool(_job, jobs, f"round 24 full grid ({len(te)} persons)")), open(path, "w"))


def evaluate_stage():
    import numpy as np
    from round14_recompute import ci, fmt
    F, _, _ = _pol()
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    te = json.load(open(os.path.join(OUT, f"seeds{tag}.json")))
    rows = json.load(open(os.path.join(OUT, f"test{tag}.json")))["rows"]
    pol = json.load(open(os.path.join(OUT, "policy.json")))
    T = {}
    err = sum("error" in x for x in rows)
    for x in rows:
        if "error" not in x:
            T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    Tr = load_train(); tr = sorted(Tr)
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    rng = np.random.default_rng(2424)
    documented = {s: bool(rng.random() < 0.5) for s in te}
    limited = {s: min(person(s)) < UNRESTRICTED for s in te}
    print(f"round 24 {'SMOKE' if tag else ''}: {len(rows)} episodes, {err} errors, {len(te)} persons; F = {F}; tau {pol['tau']}")

    def oracle(s, t):
        c = [(harm(x), -x["done"], k) for k, x in T[s].items() if x["done"] and harm(x) < t]
        return min(c)[2] if c else None

    arms = {"F": lambda s, t: F,
            "CAT_half": lambda s, t: None if (limited[s] and documented[s]) else F,
            "CAT": lambda s, t: None if limited[s] else F}
    for e in (20.0, 10.0, 4.0):
        L = Learned(Tr, tr, e); tau = pol["tau"][f"{e:g}"]
        arms[f"S_C({e:g})"] = (lambda e: (lambda s, t: None if min(r18.recorded(s, e)) < 3.5 else F))(e)
        arms[f"M({e:g})"] = (lambda e: (lambda s, t: F if min(r18.recorded(s, e)) >= UNRESTRICTED else choose(P, r18.recorded(s, e), F)))(e)
        arms[f"L({e:g})"] = (lambda e, L, tau: (lambda s, t: L.act(s, T[s], e, tau, F)))(e, L, tau)
    arms["O (perfect)"] = oracle

    for t in THRS:
        res = {}
        for a, f in arms.items():
            res[a] = {}
            for s in te:
                k = f(s, t)
                if k is None:
                    res[a][s] = dict(safe=0.0, bad=0.0, dec=1.0, load=0.0)
                else:
                    x = T[s][k]; h = harm(x)
                    res[a][s] = dict(safe=float(x["done"] and h < t), bad=float(h >= t), dec=0.0, load=h)
        print(f"\n=== harmful if robot-caused end-range load >= {t} N m ===")
        print(f"  {'rung':12s} {'safe':>5} {'harmful':>8} {'declined':>9} {'mean load':>10}")
        for a, R in res.items():
            print(f"  {a:12s} {int(sum(R[s]['safe'] for s in te)):5d} {int(sum(R[s]['bad'] for s in te)):8d} {int(sum(R[s]['dec'] for s in te)):9d} {np.mean([R[s]['load'] for s in te]):10.2f}")

        def pair(a, b, key):
            return ci([res[a][s][key] - res[b][s][key] for s in te])
        for name, a, b, tol in (("H24-1", "L(20)", "M(20)", -0.02), ("H24-2", "L(10)", "M(10)", -0.02), ("H24-3", "L(20)", "F", -0.025)):
            db, ds = pair(a, b, "bad"), pair(a, b, "safe")
            ok = db[2] < 0 and ds[1] >= tol
            print(f"  {name} [{t}] {a} - {b}: harmful rate {fmt(db)} | safe rate {fmt(ds)} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}{'' if t == 0.5 else ' (secondary threshold)'}")
        for e in (20.0, 10.0, 4.0):
            sM, sL, sO = (sum(res[a][s]["safe"] for s in te) for a in (f"M({e:g})", f"L({e:g})", "O (perfect)"))
            bM, bL, bO = (sum(res[a][s]["bad"] for s in te) for a in (f"M({e:g})", f"L({e:g})", "O (perfect)"))
            gs = (sL - sM) / (sO - sM) if sO != sM else float("nan"); gb = (bM - bL) / (bM - bO) if bM != bO else float("nan")
            print(f"  H24-4 [{t}] eps {e:g}: L closes {gs:.0%} of the M->oracle gap in safe completions and {gb:.0%} in harmful outcomes")


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(select=select, test=test, eval=evaluate_stage).get(st)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True); f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
