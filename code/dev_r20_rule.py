"""Round 20 rule development on the 160 development persons (rounds 18 + 18R): hybrids of the body model
(population body + care record, no probing) and a wrist-force guard.  Not evidence."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); TOP = os.path.dirname(ROOT); sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import round18 as r18
from round17 import UNRESTRICTED, person, harm, PAINFREE, RETRACT_S
from analysis_abstain import load, F, ORDER, P_POP, typ, choose, outcome, reactive, safe, bad

GD = {}
for x in json.load(open(os.path.join(TOP, "out", "round20", "dev_guard.json")))["rows"]:
    if "error" not in x:
        GD.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x


def ranked(P, rec, margin):
    fs, fe = rec
    over = lambda p: max(p[0] - (fs - margin) if fs < UNRESTRICTED else -1e9, p[1] - (fe - margin) if fe < UNRESTRICTED else -1e9)
    cand = [(k, p) for k, p in P.items() if p[2] >= 0.8]
    fit = sorted([(-p[2], p[3], p[4], k) for k, p in cand if over(p) <= 0])
    rest = sorted([(over(p), -p[2], p[3], k) for k, p in cand if over(p) > 0])
    first = [F] if (F in P and over(P[F]) <= 0) else []
    order = first + [x[3] for x in fit if x[3] not in first]
    return order, [(x[0], x[3]) for x in rest]


def hybrid(s, margin, n_try, abstain_over, guard_fit=True):
    rec = r18.recorded(s, 4.0)
    if min(rec) >= UNRESTRICTED:
        return None                                   # same as the fixed rule
    fit, rest = ranked(P_POP, rec, margin)
    plan = fit + [k for o, k in rest if o <= abstain_over]
    plan = plan[:n_try]
    if not plan:
        return dict(done=False, harm=0.0, abstain=True, tries=0)
    h = 0.0
    for i, k in enumerate(plan):
        g = GD.get(s, {}).get(k)
        if g is None:
            continue
        h = max(h, harm(g))
        if g["done"]:
            return dict(done=True, harm=h, abstain=False, tries=i + 1)
    return dict(done=False, harm=h, abstain=False, tries=len(plan))


te, T, Gd = load("dev")
Fo = {s: outcome(T, s, F) for s in te}; Ro = {s: reactive(Gd, s) for s in te}
M19 = {s: (outcome(T, s, F) if min(r18.recorded(s, 4.0)) >= UNRESTRICTED else outcome(T, s, choose(P_POP, r18.recorded(s, 4.0), 3.0, 3.0))) for s in te}
def row(name, O):
    print(f"{name:44s} safe {sum(safe(O[s]) for s in te):4d}  harmful {sum(bad(O[s]) for s in te):4d}  abstain {sum(O[s].get('abstain', False) for s in te):3d}  mean harm {np.mean([O[s]['harm'] for s in te]):6.2f}  p90 {np.percentile([O[s]['harm'] for s in te], 90):6.2f}")
print(f"development set {len(te)} persons (NOT evidence)")
row("F fixed rule", Fo); row("R reactive (round 17)", Ro); row("M round-19 rule (no probe)", M19)
for margin in (1.5, 3.0, 4.5):
    for n_try in (1, 2, 3):
        for ab in (-99.0, 3.0, 99.0):
            O = {}
            for s in te:
                h = hybrid(s, margin, n_try, ab)
                O[s] = Fo[s] if h is None else h
            lab = "never try unfitted" if ab < -50 else ("try all" if ab > 50 else f"try unfitted if over <= {ab:g}")
            row(f"H margin {margin} tries {n_try} {lab}", O)
