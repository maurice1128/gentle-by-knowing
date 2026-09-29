"""Round 22: soft (progressive) joint end-feel robustness + body model vs one-line record screen.
PREREGISTRATION section 10, round 22.

Part A (n=120, conditions hard / soft6 / soft12): F, reactive at 4 guards, model M, screen S_C,
         12 strategies for the within-person Spearman.
Part B (n=600, hard): F and the model's pick only -> H22-4 equivalence test M vs S_C.
Stages: test | eval
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round22")
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import round17 as r17
import round18 as r18
from round17 import UNRESTRICTED, person, harm, PAINFREE
from round21 import _pol, choose, AB, safe, bad, MULTS, REC_ERR, EQUIV

TEST_POOL = list(range(923, 3000))
N_A, N_B = 120, 600
CONDS = {"hard": None, "soft6": 6.0, "soft12": 12.0}
RHO_STRATS = [(o, -40.0, 0.10) for o in (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15)] + \
             [(o, -20.0, 0.10) for o in (-0.06, 0.03, 0.09, 0.15)]


def _job(spec):
    import shoulder_task as st
    cond, kind, seed, fs, fe, o, a, sp, guard = spec
    try:
        r = st.run(seed, fs, fe, o, a, sp, stop_force=guard, soft_deg=CONDS[cond])
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r.update(cond=cond, kind=kind, tag=guard)
    return r


def _model_pick(seed, F, P_POP):
    rec = r18.recorded(seed, REC_ERR)
    if min(rec) >= UNRESTRICTED:
        return F
    return choose(P_POP, rec, F)


def test():
    F, order, guard = _pol()
    smoke = os.environ.get("WM_SMOKE") == "1"
    seeds = [100, 104, 105, 109] if smoke else r18._valid(TEST_POOL, os.path.join(OUT, "test_pool_valid.json"))[:N_B]
    tag = "_smoke" if smoke else ""
    A = [104, 105, 109] if smoke else seeds[:N_A]          # smoke: library bodies with a restricted joint
    json.dump(dict(A=A, B=seeds), open(os.path.join(OUT, f"seeds{tag}.json"), "w"))
    path = os.path.join(OUT, f"test{tag}.json")
    if os.path.exists(path) and not smoke:
        print("exists", path); return
    lib = r18._load_library()
    P_POP = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    jobs = set()
    for s in seeds:                                   # part B: hard, F + model pick
        pk = _model_pick(s, F, P_POP)
        for k in {F, pk} - {None}:
            jobs.add(("hard", "truth", s) + person(s) + k + (None,))
    for s in A:                                       # part A: all conditions
        pk = _model_pick(s, F, P_POP)
        for c in CONDS:
            for k in set(RHO_STRATS) | {F, pk} - {None}:
                jobs.add((c, "truth", s) + person(s) + k + (None,))
            for k in order:
                for m in MULTS:
                    if guard.get(k):
                        jobs.add((c, "guard", s) + person(s) + k + (round(guard[k] * m, 3),))
    jobs = sorted(jobs, key=lambda j: (j[2], j[0], j[1]))
    rows = r17._pool(_job, jobs, f"round 22 ({len(A)} persons x 3 conditions + {len(seeds)} persons hard)")
    json.dump(dict(rows=rows), open(path, "w"))


def evaluate():
    import numpy as np
    from round14_recompute import ci, fmt
    F, order, guard = _pol()
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    S = json.load(open(os.path.join(OUT, f"seeds{tag}.json")))
    rows = json.load(open(os.path.join(OUT, f"test{tag}.json")))["rows"]
    sp = json.load(open(os.path.join(TOP, "out", "round21", "screen_policy.json")))
    lib = r18._load_library()
    P_POP = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
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
    print(f"round 22 {'SMOKE' if tag else ''}: {len(rows)} episodes, {err} errors; part A {len(S['A'])} persons, part B {len(S['B'])} persons; F = {F}")

    def out(c, s, k):
        if k is None:
            return dict(AB)
        x = T[c][s][k]
        return dict(done=bool(x["done"]), harm=harm(x), abstain=False)

    def reactive(c, s, m):
        h = 0.0; first = None
        for i, k in enumerate(order):
            g = Gd.get(c, {}).get(s, {}).get(k, {}).get(round(guard[k] * m, 3))
            if g is None:
                continue
            h = max(h, harm(g))
            if i == 0:
                first = harm(g)
            if g["done"]:
                return dict(done=True, harm=h, abstain=False, first=first)
        return dict(done=False, harm=h, abstain=False, first=first)

    screen = lambda c, s, t: dict(AB) if min(r18.recorded(s, REC_ERR)) < t else out(c, s, F)
    rate = lambda o: float(bad(o))

    def table(c, people, arms):
        res = {a: {s: f(s) for s in people} for a, f in arms.items()}
        print(f"\n  {'arm':14s} {'safe':>5} {'harmful':>8} {'abstain':>8} {'mean':>7} {'p90':>7}")
        for a, R in res.items():
            hv = [R[s]["harm"] for s in people]
            print(f"  {a:14s} {sum(safe(R[s]) for s in people):5d} {sum(bad(R[s]) for s in people):8d} {sum(R[s]['abstain'] for s in people):8d} {np.mean(hv):7.2f} {np.percentile(hv, 90):7.2f}")
        return res

    A = S["A"]
    for c in CONDS:
        print(f"\n=== part A, condition {c} (n={len(A)}) ===")
        arms = {"F": lambda s, c=c: out(c, s, F), "M": lambda s, c=c: out(c, s, _model_pick(s, F, P_POP)),
                f"S_C ({sp['t_C']:g})": lambda s, c=c: screen(c, s, sp["t_C"])}
        for m in MULTS:
            arms[f"R x{m * 1.2:.1f}p95"] = (lambda m, c: (lambda s: reactive(c, s, m)))(m, c)
        res = table(c, A, arms)
        any_help = False
        for m in MULTS:
            a = f"R x{m * 1.2:.1f}p95"; d = ci([rate(res[a][s]) - rate(res["F"][s]) for s in A])
            Rb = [s for s in A if bad(res[a][s])]; fh = sum(1 for s in Rb if res[a][s]["first"] is not None and res[a][s]["first"] >= PAINFREE)
            if d is None:
                continue
            helps = d[2] < -EQUIV; any_help |= helps
            print(f"  R - F {a:12s} harmful rate {fmt(d)}  harmed-in-first-attempt {fh}/{len(Rb)}{'  <- HELPS' if helps else ''}")
        d = ci([rate(res["M"][s]) - rate(res["F"][s]) for s in A])
        nM, nF = sum(safe(res["M"][s]) for s in A), sum(safe(res["F"][s]) for s in A)
        rho = []
        for s in A:
            R = [T[c][s][k] for k in RHO_STRATS if k in T[c].get(s, {}) and T[c][s][k]["done"]]
            if len(R) >= 6:
                f = np.array([x["peak_force"] for x in R]); h = np.array([harm(x) for x in R])
                if h.std() > 0:
                    rho.append(float(np.corrcoef(np.argsort(np.argsort(f)), np.argsort(np.argsort(h)))[0, 1]))
        med = float(np.median(rho)) if rho else float("nan")
        if c != "hard":
            print(f"  H22-1 [{c}] force guard useless at every threshold: {'SUPPORTED' if not any_help else 'NOT SUPPORTED'}")
            print(f"  H22-2 [{c}] M - F harmful rate {fmt(d)}, safe {nM} vs {nF} -> {'SUPPORTED' if (d[2] < 0 and nM >= nF - 3) else 'NOT SUPPORTED'}")
            print(f"  H22-3 [{c}] Spearman(force, load) median {med:.2f} (n={len(rho)}) -> {'SUPPORTED' if med < 0.5 else 'NOT SUPPORTED'}")
        else:
            print(f"  [hard, same persons] M - F {fmt(d)}, safe {nM} vs {nF}; Spearman median {med:.2f} (n={len(rho)})")

    B = S["B"]
    print(f"\n=== part B, hard (n={len(B)}) ===")
    arms = {"F": lambda s: out("hard", s, F), "M": lambda s: out("hard", s, _model_pick(s, F, P_POP))}
    for lab in ("A", "B", "C"):
        arms[f"S_{lab} ({sp['t_' + lab]:g})"] = (lambda t: (lambda s: screen("hard", s, t)))(sp["t_" + lab])
    res = table("hard", B, arms)
    for lab in ("C", "A", "B"):
        a = f"S_{lab} ({sp['t_' + lab]:g})"
        d = ci([rate(res["M"][s]) - rate(res[a][s]) for s in B])
        v = ("EQUIVALENT (CI inside +-%.2f)" % EQUIV if (d[1] > -EQUIV and d[2] < EQUIV) else
             "MODEL BETTER" if d[2] < 0 else "SCREEN BETTER" if d[1] > 0 else "STILL UNDECIDED")
        print(f"  {'H22-4 (primary) ' if lab == 'C' else ''}M - {a}: harmful rate {fmt(d)} | load {fmt(ci([res['M'][s]['harm'] - res[a][s]['harm'] for s in B]))} | safe {sum(safe(res['M'][s]) for s in B)} vs {sum(safe(res[a][s]) for s in B)} -> {v}")
    d = ci([rate(res["M"][s]) - rate(res["F"][s]) for s in B])
    print(f"  M - F (fourth replication): harmful rate {fmt(d)} | safe {sum(safe(res['M'][s]) for s in B)} vs {sum(safe(res['F'][s]) for s in B)}")


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(test=test, eval=evaluate).get(st)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True); f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
