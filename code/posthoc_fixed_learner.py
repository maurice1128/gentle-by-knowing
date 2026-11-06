"""POST HOC, requested by Reviewer 1 (comment 4): hold the decision rule fixed and vary only the knowledge.
No new simulation.  Uses the stored 48-strategy outcome grids of the 400 training persons and of the 600
confirmation persons of rounds 24 and 25A (hard limits).

One learner on every rung: kNN (K 25) on standardised features, estimates P(safe) and P(harmful) per strategy,
takes the strategy with the highest P(safe) (ties: lower P(harmful)), declines if P(safe) < tau.
Rungs differ only in the features (the settled posture is in every rung):
  posture      posture only (no record)
  half flag    merged flag, recorded for half of the restricted persons
  merged flag  one complete flag (shoulder and elbow merged)
  joint flags  two complete per-joint flags
  ranges 20    recorded ranges, error SD 20 deg (the features of L)
  ranges 4     recorded ranges, error SD 4 deg
Two rule variants:
  uniform   the learner decides for every person
  shortcut  a record that says unrestricted goes to the fixed rule, as L does (needs a record, so not for posture)
tau: (a) fixed at L's 0.65, (b) tuned per rung on the 400 training persons with L's criterion (fewest harmful with
safe >= F - 2.5 pp), (c) the same tuning with each training person left out of its own neighbours.
Check: ranges 20 + shortcut + tau 0.65 must give L's published counts.
Writes out/posthoc_fixed_learner.txt and out/posthoc_fixed_learner.json.  Run with the .venv_mm python."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person
from round21 import _pol
from round25 import documented, feat
from dev_r24_learned import load_train, harm
from posthoc_baselines import grids, ci, fmt

K = 25
THR = 0.5
TAUS = [x / 20 for x in range(0, 21)]
TAU_L = 0.65
KEYS = list(STRATS)


def rest(G, s):
    x = next(iter(G[s].values()))
    return x["rest_sh_deg"], x["rest_el_deg"]


def limited(s):
    a, b = person(s)
    return a < UNRESTRICTED, b < UNRESTRICTED


def f_posture(G, s):
    return np.array(rest(G, s))


def f_half(G, s):
    return np.array([float(any(limited(s)) and documented(s)), *rest(G, s)])


def f_merged(G, s):
    return np.array([float(any(limited(s))), *rest(G, s)])


def f_joint(G, s):
    a, b = limited(s)
    return np.array([float(a), float(b), *rest(G, s)])


def f_ranges(eps):
    return lambda G, s: feat(r18.recorded(s, eps), rest(G, s), True)


RUNGS = [("posture", f_posture, None),
         ("half flag", f_half, lambda s: not (any(limited(s)) and documented(s))),
         ("merged flag", f_merged, lambda s: not any(limited(s))),
         ("joint flags", f_joint, lambda s: not any(limited(s))),
         ("ranges 20", f_ranges(20.0), lambda s: not any(limited(s))),
         ("ranges 4", f_ranges(4.0), lambda s: not any(limited(s)))]


def label_arrays(G, seeds):
    S = np.full((len(seeds), len(KEYS)), np.nan)
    B = np.full((len(seeds), len(KEYS)), np.nan)
    for i, s in enumerate(seeds):
        for j, k in enumerate(KEYS):
            x = G[s].get(k)
            if x is not None:
                S[i, j] = float(x["done"] and harm(x) < THR)
                B[i, j] = float(harm(x) >= THR)
    return S, B


def decide(Z, S, B, z, skip=None):
    """Best strategy index and its P(safe) for standardised feature z."""
    d = np.linalg.norm(Z - z, axis=1)
    if skip is not None:
        d[skip] = np.inf
    nn = np.argsort(d, kind="stable")[:K]
    best, bp, bb = None, -1.0, 0.0
    for j in range(len(KEYS)):
        v = S[nn, j]; v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        p = float(v.mean()); b = float(np.nanmean(B[nn, j]))
        if (p, -b) > (bp, -bb):
            best, bp, bb = j, p, b
    return best, bp


def score(G, persons, choice, tau, shortcut, F):
    """choice: person -> (strategy index, P(safe)).  Returns counts and per-person rows."""
    safe = bad = dec = 0
    per = {}
    for s in persons:
        if shortcut is not None and shortcut(s):
            k = F
        else:
            j, p = choice[s]
            k = KEYS[j] if p >= tau else None
        if k is None:
            dec += 1; per[s] = (0.0, 0.0, 1.0, 0.0); continue
        x = G[s][k]; h = harm(x); sf = float(x["done"] and h < THR)
        safe += sf; bad += h >= THR; per[s] = (sf, float(h >= THR), 0.0, h)
    return int(safe), int(bad), dec, per


def tune(Tr, trp, choice, shortcut, F):
    sF = score(Tr, trp, {}, 0.0, lambda s: True, F)[0]
    best = None
    for t in TAUS:
        s_, b_, d_, _ = score(Tr, trp, choice, t, shortcut, F)
        if s_ >= sF - 0.025 * len(trp) and (best is None or (b_, -s_) < (best[1], -best[2])):
            best = (t, b_, s_)
    return best[0] if best else None


def main():
    F = _pol()[0]
    T, SET = grids(); people = sorted(T)
    Tr = load_train(); trp = sorted(Tr)
    r24 = [s for s in people if SET[s] == "r24"]
    S, B = label_arrays(Tr, trp)
    out, store = [], {}
    say = lambda t="": (print(t, flush=True), out.append(t))
    say(f"fixed-learner ladder (post hoc): {len(trp)} training persons; confirmation {len(people)} (round 24: {len(r24)})")
    base = {}
    for tag, ps in (("round 24", r24), ("all 600", people)):
        a = score(T, ps, {}, 0.0, lambda s: True, F)
        base[tag] = a[3]
        say(f"  F on {tag}: safe {a[0]} harmful {a[1]} declined {a[2]} mean load {np.mean([a[3][s][3] for s in ps]):.2f}")

    per_all = {}
    for name, fn, short in RUNGS:
        X = np.array([fn(Tr, s) for s in trp]); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
        test = {s: decide(Z, S, B, (fn(T, s) - mu) / sd) for s in people}
        tr_in = {s: decide(Z, S, B, Z[i]) for i, s in enumerate(trp)}
        tr_loo = {s: decide(Z, S, B, Z[i], skip=i) for i, s in enumerate(trp)}
        for variant, sc in (("uniform", None), ("shortcut", short)):
            if variant == "shortcut" and short is None:
                continue
            t_in = tune(Tr, trp, tr_in, sc, F); t_loo = tune(Tr, trp, tr_loo, sc, F)
            for tl, tau in (("tau 0.65", TAU_L), (f"tuned {t_in}", t_in), (f"LOO-tuned {t_loo}", t_loo)):
                if tau is None:
                    say(f"  {name:12s} {variant:8s} {tl:18s} no tau meets the completion criterion")
                    continue
                row = f"  {name:12s} {variant:8s} {tl:18s}"
                for tag, ps in (("round 24", r24), ("all 600", people)):
                    a = score(T, ps, test, tau, sc, F)
                    row += f" | {tag}: safe {a[0]:3d} harmful {a[1]:3d} declined {a[2]:3d} load {np.mean([a[3][s][3] for s in ps]):.2f}"
                    store[f"{name}|{variant}|{tl.split()[0]}|{tag}"] = {"tau": tau, "safe": a[0], "harmful": a[1], "declined": a[2],
                                                                        "failed": len(ps) - a[0] - a[1] - a[2]}
                    if tag == "all 600":
                        per_all[(name, variant, tl.split()[0])] = a[3]
                say(row)
            curve = [score(T, people, test, t, sc, F)[:2] for t in TAUS]
            store[f"{name}|{variant}|curve600"] = curve
            say(f"    tau sweep (600; safe, harmful): " + " ".join(f"({a},{b})" for a, b in curve))

    say("\npaired differences on all 600, uniform rule, tau 0.65 (later rung minus earlier rung): harmful | safe")
    names = [r[0] for r in RUNGS]
    for a, b in zip(names, names[1:]):
        pa, pb = per_all[(a, "uniform", "tau")], per_all[(b, "uniform", "tau")]
        say(f"  {b} - {a}: harmful {fmt(ci([pb[s][1] - pa[s][1] for s in people]))} | safe {fmt(ci([pb[s][0] - pa[s][0] for s in people]))}")
    say("\nsame, vs the fixed rule F:")
    for b in names:
        pb = per_all[(b, "uniform", "tau")]
        say(f"  {b} - F: harmful {fmt(ci([pb[s][1] - base['all 600'][s][1] for s in people]))} | safe {fmt(ci([pb[s][0] - base['all 600'][s][0] for s in people]))}")

    open(os.path.join(TOP, "out", "posthoc_fixed_learner.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump(store, open(os.path.join(TOP, "out", "posthoc_fixed_learner.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
