"""Round 19 (shoulder route): a body model that knows when to keep the usual
lift, when to change it, and when not to lift at all.
PREREGISTRATION section 10, round 19.  Rule developed on the 160 persons of
rounds 18 / 18R (scripts/dev_r19_rule.py), frozen here, confirmed on new persons.

Stages (argv[1]):  test | eval        (WM_R19_DEV=1 eval: re-evaluate the development set with this code)
"""
import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round19")
R17 = os.path.join(TOP, "out", "round17")
R18 = os.path.join(TOP, "out", "round18")
sys.path.insert(0, ROOT)

import round17 as r17
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person, harm, PAINFREE, RETRACT_S

TEST_POOL = list(range(680, 1300))
N_TEST = 120
ABSTAIN_OVER = 3.0
SAFE_TOL = 3
REC_ERRS = (0.0, 4.0, 8.0)


def choose(P, rec, F):
    """Returns a strategy, or None = do not lift."""
    fs, fe = rec
    M = r18.MARGIN_DEG
    over = lambda p: max(p[0] - (fs - M) if fs < UNRESTRICTED else -1e9, p[1] - (fe - M) if fe < UNRESTRICTED else -1e9)
    if F in P and over(P[F]) <= 0:
        return F
    cand = [(k, p) for k, p in P.items() if p[2] >= 0.8]
    ok = [(-p[2], p[3], p[4], k) for k, p in cand if over(p) <= 0]
    if ok:
        return min(ok)[3]
    if not cand:
        return None
    o, _, _, k = min((over(p), -p[2], p[3], k) for k, p in cand)
    return None if o > ABSTAIN_OVER else k


def test():
    os.makedirs(OUT, exist_ok=True)
    te = r18._valid(TEST_POOL, os.path.join(OUT, "test_pool_valid.json"))[:N_TEST]
    json.dump(te, open(os.path.join(OUT, "test_seeds.json"), "w"))
    path = os.path.join(OUT, "test.json")
    if os.path.exists(path):
        print("exists", path); return
    pol = json.load(open(os.path.join(R17, "policy.json")))
    jobs = [("truth", s) + person(s) + k + (None,) for s in te for k in STRATS]
    jobs += [("probe", s) + person(s) + k + (None,) for s in te for k in r18.PROBES]
    jobs += [("guard", s) + person(s) + tuple(k) + (pol["guard"][str(tuple(k))],) for s in te for k in pol["order"] if pol["guard"][str(tuple(k))]]
    json.dump(dict(rows=r17._pool(r18._job, jobs, f"round 19 test ({len(te)} persons)")), open(path, "w"))


def _load(dev):
    if dev:
        te, rows = [], []
        for suf in ("", "_rep"):
            te += json.load(open(os.path.join(R18, f"test_seeds{suf}.json")))
            rows += json.load(open(os.path.join(R18, f"test{suf}.json")))["rows"]
        return te, rows
    return json.load(open(os.path.join(OUT, "test_seeds.json"))), json.load(open(os.path.join(OUT, "test.json")))["rows"]


def evaluate():
    import numpy as np
    from round14_recompute import ci, fmt
    dev = os.environ.get("WM_R19_DEV") == "1"
    te, rows = _load(dev)
    pol = json.load(open(os.path.join(R17, "policy.json")))
    F = tuple(pol["F"]); order = [tuple(k) for k in pol["order"]]
    seeds, feats, G, mu, sd = r18._load_library()
    T, Gd = {}, {}
    for x in rows:
        if "error" in x:
            continue
        k = (x["offset"], x["angle"], x["speed"])
        if x["kind"] == "truth":
            T.setdefault(x["seed"], {})[k] = x
        elif x["kind"] == "guard":
            Gd.setdefault(x["seed"], {})[k] = x
    pf = r18._features([x for x in rows if x.get("kind") == "probe"])
    P_pop = r18._predict(feats[seeds[0]], seeds, feats, G, mu, sd, k_nn=0)

    def typ(s):
        fs, fe = person(s)
        return ("S" if fs < UNRESTRICTED else "") + ("E" if fe < UNRESTRICTED else "") or "N"

    def truth(s, k, probe_h=0.0, probe_t=0.0, probe_s=0.0):
        if k is None:
            return dict(done=False, harm=probe_h, shear=probe_s, t=probe_t, abstain=True)
        x = T[s][k]
        return dict(done=bool(x["done"]), harm=max(harm(x), probe_h), shear=max(x["gh_shear"], probe_s), t=x["t_end"] + probe_t, abstain=False)

    def arm_R(s):
        h = sh = t = 0.0
        for k in order:
            g = Gd.get(s, {}).get(k)
            if g is None:
                continue
            h = max(h, harm(g)); sh = max(sh, g["gh_shear"]); t += g["t_end"]
            if g["done"]:
                return dict(done=True, harm=h, shear=sh, t=t, abstain=False)
            t += RETRACT_S
        return dict(done=False, harm=h, shear=sh, t=t, abstain=False)

    def arm_M(er, identified):
        def f(s):
            rec = r18.recorded(s, er)
            if min(rec) >= UNRESTRICTED:                       # nothing restricted on the record: usual lift, no probing
                return truth(s, F)
            if identified:
                P = r18._predict(pf[s], seeds, feats, G, mu, sd, exclude=(s if s in seeds else None), k_nn=r18.K_NN)
                return truth(s, choose(P, rec, F), pf[s]["probe_harm"], pf[s]["probe_t"], pf[s]["probe_shear"])
            return truth(s, choose(P_pop, rec, F))
        return f

    arms = {"F fixed rule": lambda s: truth(s, F), "R reactive": arm_R}
    for er in REC_ERRS:
        arms[f"M_pop +-{er:g}"] = arm_M(er, False)
    for er in REC_ERRS:
        arms[f"M_id +-{er:g}"] = arm_M(er, True)
    res = {a: {s: f(s) for s in te} for a, f in arms.items()}
    safe = lambda o: o["done"] and o["harm"] < PAINFREE
    bad = lambda o: o["harm"] >= PAINFREE
    from collections import Counter
    print(f"{'DEVELOPMENT SET (not evidence)' if dev else 'CONFIRMATION SET'}: {len(te)} persons, types {dict(Counter(typ(s) for s in te))}; F = {F}")
    print(f"\n{'arm':14s} {'safe completions':>16} {'harmful outcomes':>16} {'abstained':>9} {'mean harm':>9} {'p90':>7} {'done':>5} {'median time':>11}")
    for a, R in res.items():
        hv = [R[s]["harm"] for s in te]
        print(f"{a:14s} {sum(safe(R[s]) for s in te):16d} {sum(bad(R[s]) for s in te):16d} {sum(R[s]['abstain'] for s in te):9d} "
              f"{np.mean(hv):9.2f} {np.percentile(hv, 90):7.2f} {sum(R[s]['done'] for s in te):5d} {np.median([R[s]['t'] for s in te]):11.2f}")

    def pair(a, b, f, people=te):
        return ci([f(res[a][s]) - f(res[b][s]) for s in people])

    MI = "M_id +-4"
    print()
    for name, b in (("H19-1 (primary)", "F fixed rule"), ("H19-2 (key)", "R reactive")):
        cb = pair(MI, b, lambda o: float(bad(o))); ns_a = sum(safe(res[MI][s]) for s in te); ns_b = sum(safe(res[b][s]) for s in te)
        ok = cb is not None and cb[2] < 0 and ns_a >= ns_b - SAFE_TOL
        print(f"{name} {MI} - {b}: harmful-outcome rate {cb[0]:+.3f} [{cb[1]:+.3f}, {cb[2]:+.3f}] | safe completions {ns_a} vs {ns_b} (tolerance {SAFE_TOL}) -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    for b in ("F fixed rule", "R reactive"):
        c = pair(MI, b, lambda o: o["harm"])
        print(f"H19-3 mean harm {MI} - {b}: {fmt(c)} -> {'SUPPORTED' if c[2] < 0 else 'NOT SUPPORTED'}")
    cb = pair(MI, "M_pop +-4", lambda o: float(bad(o))); ch = pair(MI, "M_pop +-4", lambda o: o["harm"])
    print(f"H19-4 identification, {MI} - M_pop +-4: harmful-outcome rate {cb[0]:+.3f} [{cb[1]:+.3f}, {cb[2]:+.3f}] | harm {fmt(ch)} | safe completions {sum(safe(res[MI][s]) for s in te)} vs {sum(safe(res['M_pop +-4'][s]) for s in te)}")
    print("record-error sweep (M_id - F): " + " ; ".join(
        f"+-{er:g}: harmful {pair(f'M_id +-{er:g}', 'F fixed rule', lambda o: float(bad(o)))[0]:+.3f}, harm {pair(f'M_id +-{er:g}', 'F fixed rule', lambda o: o['harm'])[0]:+.2f}, safe {sum(safe(res[f'M_id +-{er:g}'][s]) for s in te)}" for er in REC_ERRS))
    print("\nH19-5 by type (M_id +-4 / F / R):")
    for t in ("S", "E", "SE", "N"):
        pp = [s for s in te if typ(s) == t]
        if not pp:
            continue
        row = lambda a: f"safe {sum(safe(res[a][s]) for s in pp):3d} harmful {sum(bad(res[a][s]) for s in pp):3d} harm {np.mean([res[a][s]['harm'] for s in pp]):6.2f}"
        print(f"  {t:2s} n={len(pp):3d} | M_id: {row(MI)} abstain {sum(res[MI][s]['abstain'] for s in pp):2d} | F: {row('F fixed rule')} | R: {row('R reactive')}")
    ab = [s for s in te if res[MI][s]["abstain"]]
    if ab:
        print(f"abstentions: {len(ab)}; on those persons the fixed rule's harm: mean {np.mean([res['F fixed rule'][s]['harm'] for s in ab]):.2f}, "
              f"harmful in {sum(bad(res['F fixed rule'][s]) for s in ab)}/{len(ab)}; an oracle could have completed safely in {sum(any(x['done'] and harm(x) < PAINFREE for x in T[s].values()) for s in ab)}/{len(ab)}")
    pr = [s for s in te if min(r18.recorded(s, 4.0)) < UNRESTRICTED]
    print(f"probing cost: {len(pr)}/{len(te)} persons probed; median extra time {np.median([pf[s]['probe_t'] for s in pr]):.2f} s; probe harm >= {PAINFREE} N m in {sum(pf[s]['probe_harm'] >= PAINFREE for s in pr)}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(test=test, eval=evaluate).get(stage)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True)
    f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
