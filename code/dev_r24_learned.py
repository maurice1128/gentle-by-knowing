"""Round 24 development (seen persons only, NOT evidence): a decision policy learned offline in
simulation.  Training data = persons with full 48-strategy truth grids and rest angles
(rounds 18, 18R, 19, 21: 400 persons, hard limits).  5-fold cross-validation by person.

Policy L(eps): features = recorded free range (shoulder, elbow; degrees with error eps, 90 = unrestricted)
+ observable settled shoulder / elbow angles.  For each strategy, kNN (k persons in standardised
feature space) estimates P(safe completion) and P(harmful).  Choose the strategy with the highest
P(safe); decline if that P(safe) < tau.  Unrestricted record -> fixed rule F (as M does).
tau is chosen on the training folds: lowest harmful count with safe >= F - 2.5 %.
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person
from round21 import _pol, choose

THR = 0.5
K = 25
TAUS = [x / 20 for x in range(0, 20)]


def harm(x):
    return max(0.0, x["lim_sh"] - x.get("lim_base_sh", 0.0)) + max(0.0, x["lim_el"] - x.get("lim_base_el", 0.0))


def load_train():
    T = {}
    for d, fs in (("round18", ("test.json", "test_rep.json")), ("round19", ("test.json",)), ("round21", ("test.json",))):
        for f in fs:
            for x in json.load(open(os.path.join(TOP, "out", d, f)))["rows"]:
                if x.get("kind") == "truth" and "error" not in x and "rest_sh_deg" in x:
                    T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    return {s: g for s, g in T.items() if len(g) >= 40}


def feats(s, g, eps):
    fs, fe = r18.recorded(s, eps)
    x = next(iter(g.values()))
    return np.array([min(fs, 30.0), min(fe, 30.0), float(fs >= UNRESTRICTED), float(fe >= UNRESTRICTED), x["rest_sh_deg"], x["rest_el_deg"]])


def labels(g):
    safe = {k: float(x["done"] and harm(x) < THR) for k, x in g.items()}
    bad = {k: float(harm(x) >= THR) for k, x in g.items()}
    return safe, bad


class Learned:
    def __init__(self, T, seeds, eps, k=K):
        self.X = np.array([feats(s, T[s], eps) for s in seeds])
        self.mu, self.sd = self.X.mean(0), self.X.std(0) + 1e-9
        self.Z = (self.X - self.mu) / self.sd
        self.lab = [labels(T[s]) for s in seeds]
        self.k = k

    def scores(self, f):
        z = (f - self.mu) / self.sd
        nn = np.argsort(np.linalg.norm(self.Z - z, axis=1))[:self.k]
        out = {}
        for key in STRATS:
            v = [self.lab[i][0][key] for i in nn if key in self.lab[i][0]]
            b = [self.lab[i][1][key] for i in nn if key in self.lab[i][1]]
            if v:
                out[key] = (float(np.mean(v)), float(np.mean(b)))
        return out

    def act(self, s, g, eps, tau, F):
        if min(r18.recorded(s, eps)) >= UNRESTRICTED:
            return F
        sc = self.scores(feats(s, g, eps))
        best = max(sc.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
        return best[0] if best[1][0] >= tau else None


def evaluate(T, people, rule):
    safe = bad = dec = 0
    for s in people:
        k = rule(s)
        if k is None:
            dec += 1; continue
        x = T[s][k]
        h = harm(x)
        safe += x["done"] and h < THR
        bad += h >= THR
    return safe, bad, dec


def main():
    F, _, _ = _pol()
    T = load_train()
    seeds = sorted(T)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(seeds), 5)
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    print(f"development persons with full grids and rest angles: {len(seeds)}")
    for eps in (4.0, 10.0, 20.0):
        tot = dict(F=[0, 0, 0], M=[0, 0, 0], L=[0, 0, 0], O=[0, 0, 0])
        taus = []
        for i in range(5):
            test = list(folds[i]); train = [s for j in range(5) if j != i for s in folds[j]]
            L = Learned(T, train, eps)
            nF = evaluate(T, train, lambda s: F)[0]
            best = None
            for tau in TAUS:
                sc, bd, _ = evaluate(T, train, lambda s: L.act(s, T[s], eps, tau, F))
                if sc >= nF - 0.025 * len(train) and (best is None or bd < best[1]):
                    best = (tau, bd)
            tau = best[0] if best else 0.0; taus.append(tau)
            for name, rule in (("F", lambda s: F),
                               ("M", lambda s: F if min(r18.recorded(s, eps)) >= UNRESTRICTED else choose(P, r18.recorded(s, eps), F)),
                               ("L", lambda s: L.act(s, T[s], eps, tau, F)),
                               ("O", lambda s: min(((harm(x), -x["done"], k) for k, x in T[s].items() if x["done"] and harm(x) < THR), default=(0, 0, None))[2])):
                r = evaluate(T, test, rule)
                for j in range(3):
                    tot[name][j] += r[j]
        print(f"\nrecord error {eps:g} deg (5-fold CV, tau per fold {taus})")
        for name, (sc, bd, dc) in tot.items():
            print(f"   {name}: safe {sc:4d}  harmful {bd:4d}  declined {dc:4d}   (of {len(seeds)})")


if __name__ == "__main__":
    main()
