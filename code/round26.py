"""Round 26: model misspecification.  L, M and F unchanged (trained / tuned in the standard world);
the test world differs: S = stretch-reflex gains x3 (spastic), V = body variation sigma 0.40.
PREREGISTRATION section 10, round 26.

Run one world per process (the world is set through environment variables before any import):
  WM_WORLD=S  -> WM_REFLEX_SCALE=6.0, WM_BODY_VAR=0.25
  WM_WORLD=V  -> WM_REFLEX_SCALE=2.0, WM_BODY_VAR=0.40
Stages: stage1 (screen + F episodes) | stage2 (L and M picks) | eval
"""
import os
import sys
import json

WORLD = os.environ.get("WM_WORLD", "S")
os.environ["WM_BODY_VAR"] = "0.40" if WORLD == "V" else "0.25"
os.environ["WM_REFLEX_SCALE"] = "2.0" if WORLD == "V" else "6.0"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round26")
sys.path.insert(0, ROOT)

import numpy as np

N = 300
POOL = {"S": list(range(3000, 3700)), "V": list(range(4000, 4700))}


def _world():
    import care_env6
    care_env6.REFLEX_GAIN_SCALE = float(os.environ["WM_REFLEX_SCALE"])


def _screen(seed):
    _world()
    import shoulder_task as st
    try:
        e = st.env(); e.reset(seed=seed)
        return dict(seed=seed, valid=bool(not e.start_contact and not e.reset_diverged))
    except Exception as ex:
        return dict(seed=seed, valid=False, error=repr(ex)[:120])


def _job(spec):
    _world()
    import shoulder_task as st
    seed, fs, fe, o, a, sp = spec
    try:
        r = st.run(seed, fs, fe, o, a, sp)
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r.update(world=WORLD, reflex_scale=float(os.environ["WM_REFLEX_SCALE"]), body_var=float(os.environ["WM_BODY_VAR"]))
    return r


def _pool(func, jobs, label):
    import round17 as r17
    return r17._pool(func, jobs, label)


def _paths(stage):
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    return os.path.join(OUT, f"{WORLD}_{stage}{tag}.json")


def stage1():
    from round17 import person
    from round21 import _pol
    os.makedirs(OUT, exist_ok=True)
    F, _, _ = _pol()
    if os.environ.get("WM_SMOKE") == "1":
        seeds = [104, 105]
    else:
        sp = os.path.join(OUT, f"{WORLD}_valid.json")
        if not os.path.exists(sp):
            rows = _pool(_screen, POOL[WORLD], f"screen world {WORLD}")
            json.dump(sorted(r["seed"] for r in rows if r["valid"]), open(sp, "w"))
        seeds = json.load(open(sp))[:N]
    path = _paths("stage1")
    if os.path.exists(path) and os.environ.get("WM_SMOKE") != "1":
        print("exists", path); return
    json.dump(dict(seeds=seeds, rows=_pool(_job, [(s,) + person(s) + F for s in seeds], f"world {WORLD} F episodes")), open(path, "w"))


def _decide():
    import round18 as r18
    from round17 import person, UNRESTRICTED
    from round21 import _pol, choose
    from dev_r24_learned import load_train
    from round25 import KNN, record
    F, _, _ = _pol()
    D = json.load(open(_paths("stage1")))
    Fr = {x["seed"]: x for x in D["rows"] if "error" not in x}
    tr = load_train(); L = KNN(tr, sorted(tr), True)
    tau = json.load(open(os.path.join(TOP, "out", "round25", "policy.json")))["tau_std"]
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    picks = {}
    for s in D["seeds"]:
        if s not in Fr:
            continue
        rec = record(s, person(s)); rest = (Fr[s]["rest_sh_deg"], Fr[s]["rest_el_deg"])
        picks[s] = dict(L=L.act(rec, rest, tau, F), M=(F if min(rec) >= UNRESTRICTED else choose(P, rec, F)))
    return F, Fr, picks


def stage2():
    from round17 import person
    F, Fr, picks = _decide()
    jobs = sorted({(s,) + person(s) + k for s, p in picks.items() for k in p.values() if k is not None and k != F})
    path = _paths("stage2")
    if os.path.exists(path) and os.environ.get("WM_SMOKE") != "1":
        print("exists", path); return
    json.dump(dict(rows=_pool(_job, jobs, f"world {WORLD} L/M picks")), open(path, "w"))


def evaluate():
    from round14_recompute import ci, fmt
    from dev_r24_learned import harm
    F, Fr, picks = _decide()
    rows2 = json.load(open(_paths("stage2")))["rows"] if os.path.exists(_paths("stage2")) else []
    T = {}
    for x in list(Fr.values()) + [x for x in rows2 if "error" not in x]:
        T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    err = sum("error" in x for x in rows2)
    people = sorted(picks)
    print(f"round 26 world {WORLD} (reflex scale {os.environ['WM_REFLEX_SCALE']}, body variation {os.environ['WM_BODY_VAR']}): "
          f"{len(people)} persons, stage-2 errors {err}")
    res = {}
    for t in (0.5, 2.5):
        res[t] = {}
        for a in ("F", "M", "L"):
            res[t][a] = {}
            for s in people:
                k = F if a == "F" else picks[s][a]
                if k is None:
                    res[t][a][s] = dict(safe=0.0, bad=0.0, dec=1.0)
                else:
                    x = T[s][k]; h = harm(x)
                    res[t][a][s] = dict(safe=float(x["done"] and h < t), bad=float(h >= t), dec=0.0)
        print(f"\n  harmful if >= {t} N m")
        for a in ("F", "M", "L"):
            R = res[t][a]
            print(f"    {a}: safe {int(sum(R[s]['safe'] for s in people)):4d}  harmful {int(sum(R[s]['bad'] for s in people)):4d}  declined {int(sum(R[s]['dec'] for s in people)):4d}")
        for name, a, b in (("H26-1", "L", "F"), ("H26-2", "L", "M")):
            db = ci([res[t][a][s]["bad"] - res[t][b][s]["bad"] for s in people])
            ds = ci([res[t][a][s]["safe"] - res[t][b][s]["safe"] for s in people])
            print(f"    {name} [{t}] {a} - {b}: harmful {fmt(db)} | safe {fmt(ds)} -> {'SUPPORTED' if db[2] < 0 else 'NOT SUPPORTED'}{'' if t == 0.5 else ' (secondary)'}")


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(stage1=stage1, stage2=stage2, eval=evaluate).get(st)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True); f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
