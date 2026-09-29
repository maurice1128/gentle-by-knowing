"""Round 21: make the post-hoc claims of PHASE0_LOG 10.67 clean, or refute them.
PREREGISTRATION section 10, round 21.  Fresh persons (seed 802+, first 120 valid).

Arms (all on the same truth grid / guarded episodes of each person):
  F        fixed rule (round-17 policy)
  R(m)     reactive: force guard = m x (95th pct of harmless force), switch order of round 17
  M        preregistered round-19 model rule (population body + record, margin 3, abstain if over > 3)
  M_noabs  the same rule but never abstains (takes the least-over strategy)
  S_A, S_B record-only screen: abstain if min(recorded shoulder, elbow free range) < t, else F;
           t chosen on the 280 persons already seen (rounds 18, 18R, 19) by criterion A / B, then locked
Stages: select | test | eval
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round21")
R17 = os.path.join(TOP, "out", "round17")
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import round17 as r17
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person, harm, PAINFREE

TEST_POOL = list(range(802, 1500))
N_TEST = 120
MULTS = (0.5, 0.667, 0.833, 1.0)          # x the round-17 guard (1.2 x p95) -> 0.6, 0.8, 1.0, 1.2 x p95
REC_ERR = 4.0
EQUIV = 0.05                              # equivalence band on the harmful-outcome rate


def _pol():
    p = json.load(open(os.path.join(R17, "policy.json")))
    return tuple(p["F"]), [tuple(k) for k in p["order"]], {tuple(eval(k)): v for k, v in p["guard"].items()}


def _tables(rows):
    T, Gd = {}, {}
    for x in rows:
        if "error" in x:
            continue
        k = (x["offset"], x["angle"], x["speed"])
        if x["kind"] == "truth":
            T.setdefault(x["seed"], {})[k] = x
        elif x["kind"] == "guard":
            Gd.setdefault(x["seed"], {}).setdefault(k, {})[round(x["tag"], 3)] = x
    return T, Gd


AB = dict(done=False, harm=0.0, abstain=True)
safe = lambda o: o["done"] and o["harm"] < PAINFREE
bad = lambda o: o["harm"] >= PAINFREE


def outcome(T, s, k):
    if k is None:
        return dict(AB)
    x = T[s][k]
    return dict(done=bool(x["done"]), harm=harm(x), abstain=False)


def choose(P, rec, F, margin=3.0, abstain_over=3.0):
    fs, fe = rec
    over = lambda p: max(p[0] - (fs - margin) if fs < UNRESTRICTED else -1e9, p[1] - (fe - margin) if fe < UNRESTRICTED else -1e9)
    if F in P and over(P[F]) <= 0:
        return F
    cand = [(k, p) for k, p in P.items() if p[2] >= 0.8]
    ok = [(-p[2], p[3], p[4], k) for k, p in cand if over(p) <= 0]
    if ok:
        return min(ok)[3]
    if not cand:
        return None
    o, _, _, k = min((over(p), -p[2], p[3], k) for k, p in cand)
    return None if o > abstain_over else k


def select():
    """Lock the screen thresholds on the 280 persons already seen."""
    os.makedirs(OUT, exist_ok=True)
    F, _, _ = _pol()
    te, rows = [], []
    for d, sufs in (("round18", ("", "_rep")), ("round19", ("",))):
        for suf in sufs:
            te += json.load(open(os.path.join(TOP, "out", d, f"test_seeds{suf}.json")))
            rows += json.load(open(os.path.join(TOP, "out", d, f"test{suf}.json")))["rows"]
    T = {}
    for x in rows:
        if x["kind"] == "truth" and "error" not in x:
            T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    Fo = {s: outcome(T, s, F) for s in te}
    nF = sum(safe(Fo[s]) for s in te)
    tab = []
    for t in [x / 2 for x in range(2, 41)]:
        O = {s: (dict(AB) if min(r18.recorded(s, REC_ERR)) < t else Fo[s]) for s in te}
        tab.append((t, sum(bad(O[s]) for s in te), sum(safe(O[s]) for s in te)))
    A = min([r for r in tab if r[2] >= nF - 3], key=lambda r: (r[1], -r[2]))
    B = min(tab, key=lambda r: (r[1] + (nF - r[2]), r[1]))
    # C: matched cost - the threshold whose safe completions on these 280 are closest to the model rule's
    seeds, feats, G, mu, sd = r18._load_library()
    P_POP = r18._predict(feats[seeds[0]], seeds, feats, G, mu, sd, k_nn=0)
    Mo = {s: (Fo[s] if min(r18.recorded(s, REC_ERR)) >= UNRESTRICTED else outcome(T, s, choose(P_POP, r18.recorded(s, REC_ERR), F))) for s in te}
    nM, hM = sum(safe(Mo[s]) for s in te), sum(bad(Mo[s]) for s in te)
    C = min(tab, key=lambda r: (abs(r[2] - nM), r[1]))
    pol = dict(persons=len(te), fixed_safe=nF, fixed_harmful=sum(bad(Fo[s]) for s in te), model_safe=nM, model_harmful=hM,
               t_A=A[0], A=dict(harmful=A[1], safe=A[2]), t_B=B[0], B=dict(harmful=B[1], safe=B[2]), t_C=C[0], C=dict(harmful=C[1], safe=C[2]),
               criterion_A="lowest harmful with safe >= fixed - 3", criterion_B="lowest (harmful + safe lost)",
               criterion_C="safe completions closest to the model rule's (matched cost)", sweep=tab)
    json.dump(pol, open(os.path.join(OUT, "screen_policy.json"), "w"), indent=1)
    print({k: v for k, v in pol.items() if k != "sweep"})


def _gjob(spec):
    r = r18._job(spec)
    r["tag"] = spec[7]
    return r


def test():
    F, order, guard = _pol()
    smoke = os.environ.get("WM_SMOKE") == "1"
    te = [100, 101] if smoke else r18._valid(TEST_POOL, os.path.join(OUT, "test_pool_valid.json"))[:N_TEST]
    tag = "_smoke" if smoke else ""
    json.dump(te, open(os.path.join(OUT, f"test_seeds{tag}.json"), "w"))
    path = os.path.join(OUT, f"test{tag}.json")
    if os.path.exists(path) and not smoke:
        print("exists", path); return
    jobs = [("truth", s) + person(s) + k + (None,) for s in te for k in STRATS]
    jobs += [("guard", s) + person(s) + k + (guard[k] * m,) for s in te for k in order for m in MULTS if guard.get(k)]
    rows = r17._pool(_gjob, jobs, f"round 21 test ({len(te)} persons)")
    json.dump(dict(rows=rows), open(path, "w"))


def evaluate():
    import numpy as np
    from round14_recompute import ci, fmt
    F, order, guard = _pol()
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    te = json.load(open(os.path.join(OUT, f"test_seeds{tag}.json")))
    T, Gd = _tables(json.load(open(os.path.join(OUT, f"test{tag}.json")))["rows"])
    sp = json.load(open(os.path.join(OUT, "screen_policy.json")))
    seeds, feats, G, mu, sd = r18._load_library()
    P_POP = r18._predict(feats[seeds[0]], seeds, feats, G, mu, sd, k_nn=0)

    def reactive(s, m):
        h = 0.0; first = None
        for i, k in enumerate(order):
            g = Gd.get(s, {}).get(k, {}).get(round(guard[k] * m, 3))
            if g is None:
                continue
            h = max(h, harm(g))
            if i == 0:
                first = harm(g)
            if g["done"]:
                return dict(done=True, harm=h, abstain=False, first=first)
        return dict(done=False, harm=h, abstain=False, first=first)

    def model(abstain):
        def f(s):
            rec = r18.recorded(s, REC_ERR)
            if min(rec) >= UNRESTRICTED:
                return outcome(T, s, F)
            return outcome(T, s, choose(P_POP, rec, F, 3.0, 3.0 if abstain else 1e9))
        return f

    def screen(t):
        return lambda s: (dict(AB) if min(r18.recorded(s, REC_ERR)) < t else outcome(T, s, F))

    SA = f"S_A (t={sp['t_A']:g})"; SB = f"S_B (t={sp['t_B']:g})"; SC = f"S_C (t={sp['t_C']:g})"
    arms = {"F": lambda s: outcome(T, s, F)}
    for m in MULTS:
        arms[f"R x{m * 1.2:.1f}p95"] = (lambda m: (lambda s: reactive(s, m)))(m)
    arms["M"] = model(True); arms["M_noabs"] = model(False)
    arms[SA] = screen(sp["t_A"]); arms[SB] = screen(sp["t_B"]); arms[SC] = screen(sp["t_C"])
    res = {a: {s: f(s) for s in te} for a, f in arms.items()}
    print(f"round 21 {'SMOKE' if tag else 'CONFIRMATION'} set: {len(te)} persons; F = {F}; screen thresholds A {sp['t_A']} B {sp['t_B']} (locked on {sp['persons']} seen persons)")
    print(f"\n{'arm':16s} {'safe':>5} {'harmful':>8} {'abstain':>8} {'mean harm':>10} {'p90':>7}")
    for a, R in res.items():
        hv = [R[s]["harm"] for s in te]
        print(f"{a:16s} {sum(safe(R[s]) for s in te):5d} {sum(bad(R[s]) for s in te):8d} {sum(R[s]['abstain'] for s in te):8d} {np.mean(hv):10.2f} {np.percentile(hv, 90):7.2f}")

    def pair(a, b, f):
        return ci([f(res[a][s]) - f(res[b][s]) for s in te])
    rate = lambda o: float(bad(o)); load = lambda o: o["harm"]
    print()
    for nm, S in (("H21-1 (primary, matched cost) model vs screen C", SC), ("H21-1b model vs screen A", SA), ("H21-1c model vs screen B", SB)):
        c = pair("M", S, rate)
        verdict = ("EQUIVALENT (CI inside +-%.2f)" % EQUIV if (c[1] > -EQUIV and c[2] < EQUIV) else
                   "MODEL BETTER" if c[2] < 0 else "SCREEN BETTER" if c[1] > 0 else "INCONCLUSIVE")
        print(f"{nm}: harmful rate M - S {fmt(c)} | load {fmt(pair('M', S, load))} | safe {sum(safe(res['M'][s]) for s in te)} vs {sum(safe(res[S][s]) for s in te)} -> {verdict}")
    c = pair("M_noabs", "F", rate)
    print(f"H21-2 routing alone (model without abstention) vs F: harmful rate {fmt(c)} | load {fmt(pair('M_noabs', 'F', load))} -> {'ROUTING HELPS' if c[2] < 0 else ('ROUTING ADDS NOTHING (CI includes 0)' if c[1] <= 0 else 'ROUTING HURTS')}")
    print("H21-3 reactive at each guard vs F (harmful rate; 'helps' = CI upper < -0.05):")
    any_help = False
    for m in MULTS:
        a = f"R x{m * 1.2:.1f}p95"; c = pair(a, "F", rate)
        Rb = [s for s in te if bad(res[a][s])]; fh = sum(1 for s in Rb if res[a][s]["first"] is not None and res[a][s]["first"] >= PAINFREE)
        helps = c[2] < -EQUIV; any_help |= helps
        print(f"   {a:12s} {fmt(c)}  safe {sum(safe(res[a][s]) for s in te)}  harmed-in-first-attempt {fh}/{len(Rb)} -> {'HELPS' if helps else 'no'}")
    print(f"   -> H21-3 {'NOT SUPPORTED: a force guard reduces harm at some threshold' if any_help else 'SUPPORTED: no guard threshold reduces harm by more than 5 points'}")
    rho = []
    for s in te:
        R = [x for x in T[s].values() if x["done"]]
        if len(R) >= 6:
            f = np.array([x["peak_force"] for x in R]); h = np.array([harm(x) for x in R])
            if h.std() > 0:
                rf = np.argsort(np.argsort(f)); rh = np.argsort(np.argsort(h)); rho.append(float(np.corrcoef(rf, rh)[0, 1]))
    print(f"H21-4 within-person Spearman(peak wrist force, end-range load) across completed strategies: median {np.median(rho):.2f} (n={len(rho)} persons with any load) -> {'SUPPORTED (< 0.5)' if np.median(rho) < 0.5 else 'NOT SUPPORTED'}")
    print("replication of round 19 on fresh persons:")
    for a in ("M", SA, SB, SC):
        c = pair(a, "F", rate)
        print(f"   {a:16s} - F: harmful rate {fmt(c)}  load {fmt(pair(a, 'F', load))}  safe {sum(safe(res[a][s]) for s in te)} vs {sum(safe(res['F'][s]) for s in te)}")


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(select=select, test=test, eval=evaluate).get(st)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True); f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
