"""Round 18 (shoulder route): identify THIS body by gentle probing against a
library of simulated bodies, then choose how to lift.

The robot's world model is a library of simulated bodies (the generative model
of the population).  Before the lift it makes two short, gentle probe lifts
(1 cm, 0.10 m/s: one proximal site, one distal) and observes what a robot can
observe: the resting shoulder / elbow angles, how much each joint moved, the
wrist force.  The K library bodies whose simulated probe responses are closest
predict, for every strategy, how far the shoulder and the elbow will move on
this person.  With the care record of the person's free range and a fixed
margin, the robot picks a strategy predicted to stay inside both ranges.

Stages (argv[1]):
  library   unrestricted 48-strategy grid + the two probes on the library bodies
  loo       leave-one-out gate on the library (no test person touched)
  test      new persons: restricted truth grid, probes, guarded reactive episodes
  eval      H18-1..5
"""
import os
import sys
import json
import math
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round18")
R17 = os.path.join(TOP, "out", "round17")
sys.path.insert(0, ROOT)

import round17 as r17
from round17 import STRATS, UNRESTRICTED, person, harm, PAINFREE, RETRACT_S

LIB_POOL = list(range(100, 300))
REP = os.environ.get("WM_R18_REP") == "1"              # round 18R: frozen design, new and larger test set
TEST_POOL = list(range(560, 1000)) if REP else list(range(420, 560))
N_TEST = 120 if REP else 40
SUF = "_rep" if REP else ""
DONE_TOL = 6 if REP else 2
PROBES = ((-0.06, -40.0, 0.10), (0.12, -40.0, 0.10))     # proximal, distal; -40 deg completes in 100% (step 4)
PROBE_RISE, PROBE_HOLD = 0.01, 0.2
K_NN = 5
MARGIN_DEG = 3.0
REC_ERRS = (0.0, 4.0, 8.0)
FEATS = ("rest_sh_deg", "rest_el_deg", "p0_sh", "p0_el", "p0_f", "p1_sh", "p1_el", "p1_f")


def recorded(seed, err):
    import numpy as np
    fs, fe = person(seed)
    rng = np.random.default_rng(seed * 104729 + int(err * 10) + 5)
    f = lambda x: x if x >= UNRESTRICTED else max(0.0, x + float(rng.normal(0, err)))
    return f(fs), f(fe)


def _job(spec):
    import shoulder_task as st
    kind, seed, fs, fe, o, a, sp, extra = spec
    try:
        if kind == "probe":
            r = st.run(seed, fs, fe, o, a, sp, rise_m=PROBE_RISE, hold_s=PROBE_HOLD)
        else:
            r = st.run(seed, fs, fe, o, a, sp, stop_force=extra if kind == "guard" else None)
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r["kind"] = kind
    return r


def _valid(pool, path):
    if os.path.exists(path):
        return json.load(open(path))
    rows = r17._pool(r17._screen_job, pool, "screen")
    v = sorted(r["seed"] for r in rows if r["valid"])
    json.dump(v, open(path, "w"))
    return v


def library():
    os.makedirs(OUT, exist_ok=True)
    lib = _valid(LIB_POOL, os.path.join(OUT, "lib_seeds.json"))
    path = os.path.join(OUT, "library.json")
    if os.path.exists(path):
        print("exists", path); return
    jobs = [("truth", s, UNRESTRICTED, UNRESTRICTED) + k + (None,) for s in lib for k in STRATS]
    jobs += [("probe", s, UNRESTRICTED, UNRESTRICTED) + k + (None,) for s in lib for k in PROBES]
    json.dump(dict(rows=r17._pool(_job, jobs, f"library ({len(lib)} bodies)")), open(path, "w"))


def _features(probe_rows):
    """seed -> feature vector from the two probe episodes."""
    F = {}
    for r in probe_rows:
        if "error" in r:
            continue
        i = PROBES.index((r["offset"], r["angle"], r["speed"]))
        d = F.setdefault(r["seed"], {})
        d[f"p{i}_sh"] = r["d_sh_deg"]; d[f"p{i}_el"] = r["d_el_deg"]; d[f"p{i}_f"] = r["peak_force"]
        d["rest_sh_deg"] = r["rest_sh_deg"]; d["rest_el_deg"] = r["rest_el_deg"]
        d["probe_harm"] = max(d.get("probe_harm", 0.0), harm(r)); d["probe_shear"] = max(d.get("probe_shear", 0.0), r["gh_shear"])
        d["probe_t"] = d.get("probe_t", 0.0) + r["t_end"] + RETRACT_S
    return {s: d for s, d in F.items() if all(k in d for k in FEATS)}


def _load_library():
    import numpy as np
    rows = json.load(open(os.path.join(OUT, "library.json")))["rows"]
    feats = _features([r for r in rows if r.get("kind") == "probe"])
    G = {}
    for r in rows:
        if r.get("kind") == "truth" and "error" not in r:
            G.setdefault(r["seed"], {})[(r["offset"], r["angle"], r["speed"])] = r
    seeds = sorted(s for s in feats if s in G)
    X = np.array([[feats[s][k] for k in FEATS] for s in seeds])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    return seeds, feats, G, mu, sd


def _predict(fvec, seeds, feats, G, mu, sd, exclude=None, k_nn=K_NN):
    """strategy -> (pred shoulder deg, pred elbow deg, pred completion, pred shear, pred time) from the K nearest bodies."""
    import numpy as np
    cand = [s for s in seeds if s != exclude]
    Z = np.array([[(feats[s][k] - mu[i]) / sd[i] for i, k in enumerate(FEATS)] for s in cand])
    z = np.array([(fvec[k] - mu[i]) / sd[i] for i, k in enumerate(FEATS)])
    nn = [cand[i] for i in np.argsort(np.linalg.norm(Z - z, axis=1))[:k_nn]] if k_nn else cand
    out = {}
    for k in STRATS:
        R = [G[s][k] for s in nn if k in G[s]]
        D = [r for r in R if r["done"]]
        if not D:
            continue
        out[k] = (float(np.mean([r["d_sh_deg"] for r in D])), float(np.mean([r["d_el_deg"] for r in D])),
                  len(D) / max(1, len(R)), float(np.mean([r["gh_shear"] for r in D])), float(np.mean([r["t_end"] for r in D])))
    return out


def loo():
    """Gate, library only: does identification predict a held-out body's joint excursions better than the population mean?"""
    import numpy as np
    seeds, feats, G, mu, sd = _load_library()
    err = {"id": {"sh": [], "el": []}, "pop": {"sh": [], "el": []}}
    agree = {"id": [], "pop": []}
    for s in seeds:
        for name, kk in (("id", K_NN), ("pop", 0)):
            P = _predict(feats[s], seeds, feats, G, mu, sd, exclude=s, k_nn=kk)
            for k, p in P.items():
                t = G[s].get(k)
                if t is None:
                    continue
                agree[name].append((p[2] >= 0.8) == t["done"])
                if t["done"]:
                    err[name]["sh"].append(p[0] - t["d_sh_deg"]); err[name]["el"].append(p[1] - t["d_el_deg"])
    print(f"library bodies {len(seeds)}; leave-one-out, K = {K_NN}")
    for j in ("sh", "el"):
        a, b = np.std(err["id"][j]), np.std(err["pop"][j])
        print(f"  {'shoulder' if j == 'sh' else 'elbow':8s} excursion error SD: identified {a:.2f} deg, population mean {b:.2f} deg -> reduction {1 - a / b:.0%}")
    print(f"  completion prediction agrees: identified {np.mean(agree['id']):.0%}, population {np.mean(agree['pop']):.0%}")
    red = 1 - np.std(err["id"]["sh"]) / np.std(err["pop"]["sh"])
    print(f"GATE (shoulder error SD reduced by >= 30%): {'PASS' if red >= 0.30 else 'FAIL'}")


def test():
    te = _valid(TEST_POOL, os.path.join(OUT, f"test_pool_valid{SUF}.json"))[:N_TEST]
    if os.environ.get("WM_SMOKE") == "1":
        te = [100, 101]                                  # library bodies: the smoke test must not touch the test persons
    json.dump(te, open(os.path.join(OUT, "test_seeds_smoke.json" if os.environ.get("WM_SMOKE") == "1" else f"test_seeds{SUF}.json"), "w"))
    pol = json.load(open(os.path.join(R17, "policy.json")))
    path = os.path.join(OUT, "test_smoke.json" if os.environ.get("WM_SMOKE") == "1" else f"test{SUF}.json")
    if os.path.exists(path) and os.environ.get("WM_SMOKE") != "1":
        print("exists", path); return
    jobs = [("truth", s) + person(s) + k + (None,) for s in te for k in STRATS]
    jobs += [("probe", s) + person(s) + k + (None,) for s in te for k in PROBES]
    jobs += [("guard", s) + person(s) + tuple(k) + (pol["guard"][str(tuple(k))],) for s in te for k in pol["order"] if pol["guard"][str(tuple(k))]]
    json.dump(dict(rows=r17._pool(_job, jobs, f"test ({len(te)} persons)")), open(path, "w"))


def evaluate():
    import numpy as np
    from round14_recompute import ci, fmt
    smoke = os.environ.get("WM_SMOKE") == "1"
    te = json.load(open(os.path.join(OUT, "test_seeds_smoke.json" if smoke else f"test_seeds{SUF}.json")))
    rows = json.load(open(os.path.join(OUT, "test_smoke.json" if smoke else f"test{SUF}.json")))["rows"]
    pol = json.load(open(os.path.join(R17, "policy.json")))
    F = tuple(pol["F"]); order = [tuple(k) for k in pol["order"]]
    seeds, feats, G, mu, sd = _load_library()
    T, Gd = {}, {}
    for r in rows:
        if "error" in r:
            continue
        k = (r["offset"], r["angle"], r["speed"])
        if r["kind"] == "truth":
            T.setdefault(r["seed"], {})[k] = r
        elif r["kind"] == "guard":
            Gd.setdefault(r["seed"], {})[k] = r
    pf = _features([r for r in rows if r.get("kind") == "probe"])
    FAIL_T = 3.5

    def out_truth(s, k, probe=None):
        r = T.get(s, {}).get(k)
        if r is None:
            o = dict(done=False, harm=float("nan"), shear=float("nan"), t=FAIL_T)
        else:
            o = dict(done=r["done"], harm=harm(r), shear=r["gh_shear"], t=(r["t_end"] if r["done"] else FAIL_T))
        if probe is not None and not math.isnan(o["harm"]):
            o = dict(done=o["done"], harm=max(o["harm"], probe["probe_harm"]), shear=max(o["shear"], probe["probe_shear"]), t=o["t"] + probe["probe_t"])
        return o

    def arm_R(s):
        h = sh = t = 0.0
        for k in order:
            r = Gd.get(s, {}).get(k)
            if r is None:
                continue
            h = max(h, harm(r)); sh = max(sh, r["gh_shear"]); t += r["t_end"]
            if r["done"]:
                return dict(done=True, harm=h, shear=sh, t=t)
            t += RETRACT_S
        return dict(done=False, harm=h, shear=sh, t=t)

    def choose(P, rec):
        fs, fe = rec
        cand = [(k, p) for k, p in P.items() if p[2] >= 0.8]
        over = lambda p: max(p[0] - (fs - MARGIN_DEG) if fs < UNRESTRICTED else -1e9, p[1] - (fe - MARGIN_DEG) if fe < UNRESTRICTED else -1e9)
        ok = [(p[3], p[4], k) for k, p in cand if over(p) <= 0]
        if ok:
            return min(ok)[2]
        return min((over(p), p[3], k) for k, p in cand)[2] if cand else F

    def arm_M(er, identified):
        def f(s):
            if identified and s not in pf:
                return out_truth(s, F)
            P = _predict(pf[s] if identified else feats[seeds[0]], seeds, feats, G, mu, sd, exclude=(s if s in seeds else None), k_nn=(K_NN if identified else 0))
            return out_truth(s, choose(P, recorded(s, er)), probe=(pf[s] if identified else None))
        return f

    def arm_O(s):
        c = [(round(harm(r), 1), r["gh_shear"], r["t_end"], k) for k, r in T.get(s, {}).items() if r["done"]]
        return out_truth(s, min(c)[3]) if c else out_truth(s, F)

    arms = {"F fixed rule": lambda s: out_truth(s, F), "R reactive": arm_R}
    for er in REC_ERRS:
        arms[f"M_pop rec+-{er:g}"] = arm_M(er, False)
    for er in REC_ERRS:
        arms[f"M_id rec+-{er:g}"] = arm_M(er, True)
    arms["O oracle"] = arm_O
    res = {a: {s: f(s) for s in te} for a, f in arms.items()}
    lim = [s for s in te if min(person(s)) < UNRESTRICTED]
    print(f"test persons {len(te)} ({len(lim)} with a restricted joint); probes available for {sum(s in pf for s in te)}; F = {F}; K = {K_NN}, margin {MARGIN_DEG} deg")
    print(f"\n{'arm':18s} {'done':>6} {'harm median':>11} {'mean':>7} {'p90':>7} {'no end-range load':>17} {'GH shear':>8} {'time':>6}")
    for a, R in res.items():
        D = [R[s] for s in te if R[s]["done"]]
        hv = [R[s]["harm"] for s in te if not math.isnan(R[s]["harm"])]
        print(f"{a:18s} {len(D):3d}/{len(te):<2d} {np.median(hv):11.2f} {np.mean(hv):7.2f} {np.percentile(hv, 90):7.2f} {np.mean([h < PAINFREE for h in hv]):17.0%} "
              f"{np.median([r['shear'] for r in D]) if D else float('nan'):8.1f} {np.median([R[s]['t'] for s in te]):6.2f}")

    def pair(a, b, key, people=te):
        return ci([res[a][s][key] - res[b][s][key] for s in people if not math.isnan(res[a][s][key]) and not math.isnan(res[b][s][key])])

    MI = "M_id rec+-4"
    print("\npaired differences (negative = first arm gentler / faster)")
    for name, b, need_done in (("H18-1 (primary)", "F fixed rule", True), ("H18-2 (key)", "R reactive", True), ("H18-3 (identification)", "M_pop rec+-4", False)):
        ch, cs, ct = pair(MI, b, "harm"), pair(MI, b, "shear"), pair(MI, b, "t")
        na = sum(res[MI][s]["done"] for s in te); nb = sum(res[b][s]["done"] for s in te)
        ok = ch is not None and ch[2] < 0 and (not need_done or na >= nb - DONE_TOL)
        print(f"{name} {MI} - {b}: harm {fmt(ch)} | GH shear {fmt(cs)} | time {fmt(ct)} | done {na} vs {nb} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print("H18-4 record-error tolerance (harm, M_id - F):")
    for er in REC_ERRS:
        print(f"   +-{er:g} deg: {fmt(pair(f'M_id rec+-{er:g}', 'F fixed rule', 'harm'))}    (M_pop - F: {fmt(pair(f'M_pop rec+-{er:g}', 'F fixed rule', 'harm'))})")
    ph = [pf[s]["probe_harm"] for s in te if s in pf]
    print(f"H18-5 probe harm: > {PAINFREE} N m in {sum(h > PAINFREE for h in ph)}/{len(ph)} persons, median {np.median(ph):.2f}, max {max(ph):.2f} N m; "
          f"M_id - O harm {fmt(pair(MI, 'O oracle', 'harm'))}")
    print(f"restricted persons only (n={len(lim)}): M_id - F {fmt(pair(MI, 'F fixed rule', 'harm', lim))} | M_id - R {fmt(pair(MI, 'R reactive', 'harm', lim))} | M_id - M_pop {fmt(pair(MI, 'M_pop rec+-4', 'harm', lim))}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(library=library, loo=loo, test=test, eval=evaluate).get(stage)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True)
    f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
