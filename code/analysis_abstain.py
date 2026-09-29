"""Route A analyses on existing data (no new simulation).  EXPLORATORY: the
round-19 confirmation set has already been evaluated with the frozen rule;
everything here is a post hoc description, labelled as such.

(1) Abstention trade-off: sweep the safety margin and the abstention threshold
    of the round-19 rule; for each setting count harmful outcomes, safe
    completions, abstentions, and how many abstentions were "right" (the fixed
    rule would have harmed the person) or "wasted" (a safe completion existed).
(2) Judge before vs feel during: per person, which arm harmed whom and how badly;
    for the reactive arm, whether the harm happened in the first attempt (felt
    only after the joint was already loaded).
Sets: development (rounds 18 + 18R, 160 persons) and confirmation (round 19, 120).
Output: printed; tee to out/round20/analysis_abstain.txt
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import round18 as r18
from round17 import UNRESTRICTED, person, harm, PAINFREE, RETRACT_S

R17 = os.path.join(TOP, "out", "round17"); R18 = os.path.join(TOP, "out", "round18"); R19 = os.path.join(TOP, "out", "round19")
POL = json.load(open(os.path.join(R17, "policy.json")))
F = tuple(POL["F"]); ORDER = [tuple(k) for k in POL["order"]]
SEEDS, FEATS, G, MU, SD = r18._load_library()
P_POP = r18._predict(FEATS[SEEDS[0]], SEEDS, FEATS, G, MU, SD, k_nn=0)


def load(which):
    if which == "dev":
        te, rows = [], []
        for suf in ("", "_rep"):
            te += json.load(open(os.path.join(R18, f"test_seeds{suf}.json")))
            rows += json.load(open(os.path.join(R18, f"test{suf}.json")))["rows"]
    else:
        te = json.load(open(os.path.join(R19, "test_seeds.json"))); rows = json.load(open(os.path.join(R19, "test.json")))["rows"]
    T, Gd = {}, {}
    for x in rows:
        if "error" in x:
            continue
        k = (x["offset"], x["angle"], x["speed"])
        if x["kind"] == "truth":
            T.setdefault(x["seed"], {})[k] = x
        elif x["kind"] == "guard":
            Gd.setdefault(x["seed"], {})[k] = x
    return te, T, Gd


def typ(s):
    fs, fe = person(s)
    return ("S" if fs < UNRESTRICTED else "") + ("E" if fe < UNRESTRICTED else "") or "N"


def choose(P, rec, margin, abstain_over):
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


def outcome(T, s, k):
    if k is None:
        return dict(done=False, harm=0.0, abstain=True)
    x = T[s][k]
    return dict(done=bool(x["done"]), harm=harm(x), abstain=False)


def reactive(Gd, s):
    h = 0.0; first = None
    for i, k in enumerate(ORDER):
        g = Gd.get(s, {}).get(k)
        if g is None:
            continue
        h = max(h, harm(g))
        if i == 0:
            first = harm(g)
        if g["done"]:
            return dict(done=True, harm=h, abstain=False, first=first)
    return dict(done=False, harm=h, abstain=False, first=first)


safe = lambda o: o["done"] and o["harm"] < PAINFREE
bad = lambda o: o["harm"] >= PAINFREE


def sweep(which):
    te, T, Gd = load(which)
    oracle_safe = {s: any(x["done"] and harm(x) < PAINFREE for x in T[s].values()) for s in te}
    Fo = {s: outcome(T, s, F) for s in te}
    Ro = {s: reactive(Gd, s) for s in te}
    print(f"\n==== ({1}) abstention trade-off, {which} set, {len(te)} persons; model = population body + record (4 deg error), no probing ====")
    print(f"reference: fixed rule harmful {sum(bad(Fo[s]) for s in te)}, safe {sum(safe(Fo[s]) for s in te)} | reactive harmful {sum(bad(Ro[s]) for s in te)}, safe {sum(safe(Ro[s]) for s in te)} | "
          f"a safe completion exists for {sum(oracle_safe.values())}")
    print(f"{'margin':>6} {'abstain if over':>15} | {'harmful':>7} {'safe':>5} {'abstain':>7} | {'right (F would harm)':>20} {'wasted (safe existed)':>21}")
    for margin in (0.0, 1.5, 3.0, 4.5, 6.0):
        for ab in (-99.0, 0.0, 1.5, 3.0, 6.0, 99.0):
            O = {s: (outcome(T, s, F) if min(r18.recorded(s, 4.0)) >= UNRESTRICTED else outcome(T, s, choose(P_POP, r18.recorded(s, 4.0), margin, ab))) for s in te}
            A = [s for s in te if O[s]["abstain"]]
            tag = "  <- round-19 frozen rule" if (margin == 3.0 and ab == 3.0) else ""
            print(f"{margin:6.1f} {('never' if ab >= 99 else ('always' if ab <= -99 else f'{ab:.1f} deg')):>15} | {sum(bad(O[s]) for s in te):7d} {sum(safe(O[s]) for s in te):5d} {len(A):7d} | "
                  f"{sum(bad(Fo[s]) for s in A):20d} {sum(oracle_safe[s] for s in A):21d}{tag}")
    return te, T, Gd, Fo, Ro


def judge_vs_feel(which, te, T, Gd, Fo, Ro):
    M = {s: (outcome(T, s, F) if min(r18.recorded(s, 4.0)) >= UNRESTRICTED else outcome(T, s, choose(P_POP, r18.recorded(s, 4.0), 3.0, 3.0))) for s in te}
    print(f"\n==== (2) judge before vs feel during, {which} set ====")
    both = [s for s in te if bad(M[s]) and bad(Ro[s])]; onlyR = [s for s in te if bad(Ro[s]) and not bad(M[s])]; onlyM = [s for s in te if bad(M[s]) and not bad(Ro[s])]
    print(f"harmed by both {len(both)}, only by reactive {len(onlyR)}, only by model {len(onlyM)}")
    for name, S in (("only reactive", onlyR), ("only model", onlyM), ("both", both)):
        if S:
            print(f"  {name:13s}: types {dict((t, sum(typ(s) == t for s in S)) for t in ('S', 'E', 'SE', 'N') if any(typ(s) == t for s in S))}; "
                  f"harm reactive mean {np.mean([Ro[s]['harm'] for s in S]):.1f} / model {np.mean([M[s]['harm'] for s in S]):.1f}")
    Rb = [s for s in te if bad(Ro[s])]
    firsthit = [s for s in Rb if Ro[s]["first"] is not None and Ro[s]["first"] >= PAINFREE]
    print(f"reactive harmful outcomes {len(Rb)}: in {len(firsthit)} the joint was already loaded (>= {PAINFREE} N m) during the FIRST attempt, before any switch")
    sev = lambda S, D: np.mean([D[s]["harm"] for s in S]) if S else float("nan")
    top = sorted(te, key=lambda s: -Fo[s]["harm"])[:max(1, len(te) // 10)]
    print(f"worst 10% under the fixed rule (n={len(top)}): mean harm fixed {sev(top, Fo):.1f} | reactive {sev(top, Ro):.1f} | model {sev(top, M):.1f}; model abstained on {sum(M[s]['abstain'] for s in top)}")
    ab = [s for s in te if M[s]["abstain"]]
    print(f"model abstentions {len(ab)}: reactive outcome on those persons -> harmful {sum(bad(Ro[s]) for s in ab)}, safe {sum(safe(Ro[s]) for s in ab)}, mean harm {sev(ab, Ro):.1f}")


def main():
    print(__doc__.split("\n")[0])
    for which in ("dev", "confirmation"):
        args = sweep(which)
        judge_vs_feel(which, *args)


if __name__ == "__main__":
    main()
