"""POST HOC (not preregistered; no new simulation): leave-one-out re-selection of the decline threshold tau.

The preregistered learned policy L (round25.KNN, use_rest=True, K=25, trained on the 400 persons of
dev_r24_learned.load_train(), record error 20 deg) had tau = 0.65 selected IN-SAMPLE on those same 400 persons
(round24.select): each training person is its own nearest neighbour.  Here tau is re-selected with leave-one-out
(the person itself is removed from its own neighbour set), with the same criterion:
    fewest harmful outcomes among tau in {0, 0.05, ..., 1} with safe completions >= F_safe - 2.5 % of n
    (ties: more safe completions, then the smaller tau; round24.select used strictly-fewer-harmful, i.e. the
    smallest tau among ties in harmful, which is also reported).
F = round21._pol()[0]; record = round18.recorded(seed, 20); rest angles from the F row.
Then L at tau = 0.65 (preregistered) and at the LOO tau is evaluated on the 600 confirmation persons with full
48-strategy grids (round 24 test.json + round 25 test1.json, hard limits; same set as posthoc_baselines.py), with
paired differences vs F and 95% t intervals (posthoc_baselines.ci).  The same is done for the flags + posture kNN
of posthoc_baselines.py (FLAGKNN).
Writes out/posthoc_loo_tau.txt.  Run with the .venv_mm python."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round18 as r18
from round17 import STRATS, UNRESTRICTED
from round21 import _pol
from dev_r24_learned import load_train, harm
from round25 import KNN, feat, K
from posthoc_baselines import grids, ci, fmt

EPS = 20.0
THR = 0.5
TAUS = [x / 20 for x in range(0, 21)]


class Picker:
    """kNN strategy scorer over the training persons; X = raw feature rows of the training persons."""

    def __init__(self, X, lab):
        self.X = np.asarray(X, float); self.mu, self.sd = self.X.mean(0), self.X.std(0) + 1e-9
        self.Z = (self.X - self.mu) / self.sd; self.lab = lab

    def best(self, f, exclude=None):
        d = np.linalg.norm(self.Z - (np.asarray(f, float) - self.mu) / self.sd, axis=1)
        if exclude is not None:
            d[exclude] = np.inf
        nn = np.argsort(d)[:K]
        best = None
        for k in STRATS:                       # identical scoring / tie-breaking to round25.KNN.act
            v = [self.lab[i][0][k] for i in nn if k in self.lab[i][0]]
            b = [self.lab[i][1][k] for i in nn if k in self.lab[i][1]]
            if v:
                c = (float(np.mean(v)), -float(np.mean(b)), k)
                if best is None or c[:2] > best[:2]:
                    best = c
        return best                            # (p_safe, -p_harm, strategy)


def main():
    F, _, _ = _pol()
    Tr = load_train(); trp = sorted(Tr); n = len(trp)
    T, SET = grids(); people = sorted(T)
    rec = lambda s: r18.recorded(s, EPS)
    rest = lambda G, s: (G[s][F]["rest_sh_deg"], G[s][F]["rest_el_deg"])
    flags = lambda r: (r[0] < UNRESTRICTED, r[1] < UNRESTRICTED)
    lab = [({k: float(x["done"] and harm(x) < THR) for k, x in Tr[s].items()},
            {k: float(harm(x) >= THR) for k, x in Tr[s].items()}) for s in trp]
    out = []
    say = lambda t="": (print(t), out.append(t))

    # ---- L: features exactly as round25.KNN(use_rest=True); check equality with the preregistered object
    Lref = KNN(Tr, trp, True)
    XL = [feat(rec(s), rest(Tr, s), True) for s in trp]
    L = Picker(XL, lab)
    rest_diff = max(abs(np.asarray(XL) - Lref.X).max(), 0.0)
    # ---- FLAGKNN: flags + settled angles (posthoc_baselines.fk_feats)
    fk = lambda G, s: np.array([float(flags(rec(s))[0]), float(flags(rec(s))[1]), *rest(G, s)])
    FK = Picker([fk(Tr, s) for s in trp], lab)

    def L_pick(G, tau, loo_index=None):
        def f(s):
            r = rec(s)
            if min(r) >= UNRESTRICTED:
                return F
            b = L.best(feat(r, rest(G, s), True), exclude=loo_index(s) if loo_index else None)
            return b[2] if b[0] >= tau else None
        return f

    def FK_pick(G, tau, loo_index=None):
        def f(s):
            if not any(flags(rec(s))):
                return F
            b = FK.best(fk(G, s), exclude=loo_index(s) if loo_index else None)
            return b[2] if b[0] >= tau else None
        return f

    def outcome(G, persons, pick):
        per = {}
        for s in persons:
            k = pick(s)
            if k is None:
                per[s] = (0.0, 0.0, 1.0); continue
            x = G[s][k]; h = harm(x)
            per[s] = (float(x["done"] and h < THR), float(h >= THR), 0.0)
        return tuple(int(sum(v[j] for v in per.values())) for j in range(3)), per

    idx = {s: i for i, s in enumerate(trp)}
    loo = lambda s: idx[s]
    (sF, bF, _), _ = outcome(Tr, trp, lambda s: F)
    floor = sF - 0.025 * n
    say(f"POST HOC leave-one-out tau selection.  Training persons n={n}; F = {F}: safe {sF}, harmful {bF}; "
        f"criterion: fewest harmful with safe >= {sF} - 2.5% x {n} = {floor:g}")
    say(f"  (check: L features identical to round25.KNN(use_rest=True): max |diff| = {rest_diff:.2e}; K = {K})")

    sel = {}
    for name, P in (("L (round25.KNN, use_rest=True)", L_pick), ("FLAGKNN (flags + posture kNN, posthoc_baselines)", FK_pick)):
        say(f"\n=== {name}: training table (400 persons) ===")
        say(f"  {'tau':>5} | {'in-sample safe/harm/decl':>25} | {'LOO safe/harm/decl':>20}")
        rows = {}
        for tau in TAUS:
            a, _ = outcome(Tr, trp, P(Tr, tau))
            b, _ = outcome(Tr, trp, P(Tr, tau, loo))
            rows[tau] = (a, b)
            say(f"  {tau:5.2f} | {a[0]:7d} {a[1]:7d} {a[2]:7d}{'  ok' if a[0] >= floor else '    '}   | {b[0]:5d} {b[1]:5d} {b[2]:5d}{'  ok' if b[0] >= floor else ''}")
        for j, lbl in ((0, "in-sample"), (1, "LOO")):
            ok = [(rows[t][j][1], -rows[t][j][0], t) for t in TAUS if rows[t][j][0] >= floor]
            t_sel = min(ok)[2]
            t_r24 = min(ok, key=lambda r: (r[0], r[2]))[2]      # round24.select tie rule (smallest tau among fewest harmful)
            sel[(name[:1], lbl)] = t_sel
            say(f"  selected tau ({lbl}): {t_sel:g}  (train safe {rows[t_sel][j][0]}, harmful {rows[t_sel][j][1]}, declined {rows[t_sel][j][2]})"
                + ("" if t_r24 == t_sel else f"; with round24.select's tie rule: {t_r24:g}"))

    # ---- confirmation persons
    say(f"\n=== confirmation: {len(people)} persons with full grids (r24 {sum(SET[s] == 'r24' for s in people)}, "
        f"r25A {sum(SET[s] == 'r25A' for s in people)}), hard limits, record error {EPS:g} deg ===")
    arms = {"F": lambda s: F}
    tL, tFi, tFl = sel[("L", "LOO")], sel[("F", "in-sample")], sel[("F", "LOO")]
    arms["L(tau 0.65, prereg)"] = L_pick(T, 0.65)
    arms[f"L(tau {tL:g}, LOO)"] = L_pick(T, tL)
    arms[f"FLAGKNN(tau {tFi:g}, in-sample)"] = FK_pick(T, tFi)
    if tFl != tFi:
        arms[f"FLAGKNN(tau {tFl:g}, LOO)"] = FK_pick(T, tFl)
    res = {}
    say(f"  {'arm':30s} {'safe':>5} {'harmful':>8} {'declined':>9}")
    for a, f in arms.items():
        c, res[a] = outcome(T, people, f)
        say(f"  {a:30s} {c[0]:5d} {c[1]:8d} {c[2]:9d}")
    say("  paired differences (per-person indicator; mean and 95% t CI):")
    for a in arms:
        if a == "F":
            continue
        db = ci([res[a][s][1] - res["F"][s][1] for s in people]); ds = ci([res[a][s][0] - res["F"][s][0] for s in people])
        say(f"    {a:30s} - F: harmful {fmt(db)} | safe {fmt(ds)}")
    La, Lb = "L(tau 0.65, prereg)", f"L(tau {tL:g}, LOO)"
    if tL != 0.65:
        db = ci([res[Lb][s][1] - res[La][s][1] for s in people]); ds = ci([res[Lb][s][0] - res[La][s][0] for s in people])
        say(f"    {Lb:30s} - {La}: harmful {fmt(db)} | safe {fmt(ds)}")
    say("\n  confirmation sweep of L over tau (safe, harmful, declined) for reference:")
    say("   " + " ".join(f"{t:g}:{outcome(T, people, L_pick(T, t))[0]}" for t in TAUS if 0.4 <= t <= 0.85))
    open(os.path.join(TOP, "out", "posthoc_loo_tau.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
