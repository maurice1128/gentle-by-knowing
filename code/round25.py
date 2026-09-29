"""Round 25: replication and stress tests of the offline-learned decision policy L.
PREREGISTRATION section 10, round 25.

Stages:
  select  lock tau for L_norest on the 400 development persons
  test1   A (300, standard population, hard): full 48-strategy grid;  B (250, shifted population): F only
  test2   A first 150 under soft20: F + every rule's pick;  B: every rule's pick
  eval
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round25")
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round17 as r17
import round18 as r18
from round17 import STRATS, UNRESTRICTED
from round21 import _pol, choose
from dev_r24_learned import load_train, labels, harm, TAUS

EPS = 20.0
K = 25
N_SOFT = 150


def person_std(seed):
    return r17.person(seed)


def person_shift(seed):
    rng = np.random.default_rng(seed * 7919 + 17)
    fs = float(rng.uniform(2.0, 12.0)) if rng.random() < 0.6 else UNRESTRICTED
    fe = float(rng.uniform(0.0, 8.0)) if rng.random() < 0.5 else UNRESTRICTED
    return fs, fe


def record(seed, true, err=EPS, documented=True):
    if not documented:
        return (UNRESTRICTED, UNRESTRICTED)
    rng = np.random.default_rng(seed * 104729 + int(err * 10) + 5)
    return tuple(x if x >= UNRESTRICTED else max(0.0, x + float(rng.normal(0, err))) for x in true)


def documented(seed):
    return bool(np.random.default_rng(seed * 31 + 2525).random() < 0.5)


def rest_noise(seed):
    return np.random.default_rng(seed * 97 + 25).normal(0.0, 5.0, 2)


def feat(rec, rest, use_rest=True):
    fs, fe = rec
    f = [min(fs, 30.0), min(fe, 30.0), float(fs >= UNRESTRICTED), float(fe >= UNRESTRICTED)]
    return np.array(f + ([rest[0], rest[1]] if use_rest else []))


class KNN:
    def __init__(self, T, seeds, use_rest):
        self.use_rest = use_rest
        X = []
        for s in seeds:
            x = next(iter(T[s].values()))
            X.append(feat(r18.recorded(s, EPS), (x["rest_sh_deg"], x["rest_el_deg"]), use_rest))
        self.X = np.array(X); self.mu, self.sd = self.X.mean(0), self.X.std(0) + 1e-9
        self.Z = (self.X - self.mu) / self.sd
        self.lab = [labels(T[s]) for s in seeds]

    def act(self, rec, rest, tau, F):
        if min(rec) >= UNRESTRICTED:
            return F
        z = (feat(rec, rest, self.use_rest) - self.mu) / self.sd
        nn = np.argsort(np.linalg.norm(self.Z - z, axis=1))[:K]
        best = None
        for k in STRATS:
            v = [self.lab[i][0][k] for i in nn if k in self.lab[i][0]]
            b = [self.lab[i][1][k] for i in nn if k in self.lab[i][1]]
            if v:
                c = (float(np.mean(v)), -float(np.mean(b)), k)
                if best is None or c[:2] > best[:2]:
                    best = c
        return best[2] if best and best[0] >= tau else None


def _sets():
    v = json.load(open(os.path.join(TOP, "out", "round22", "test_pool_valid.json")))
    return v[1500:1800], v[1800:2050]


def _policies():
    F, _, _ = _pol()
    T = load_train(); seeds = sorted(T)
    pol = json.load(open(os.path.join(OUT, "policy.json")))
    return F, KNN(T, seeds, True), KNN(T, seeds, False), pol


def select():
    os.makedirs(OUT, exist_ok=True)
    F, _, _ = _pol()
    T = load_train(); seeds = sorted(T)
    L0 = KNN(T, seeds, False)
    nF = sum(T[s][F]["done"] and harm(T[s][F]) < 0.5 for s in seeds)
    best = None
    for tau in TAUS:
        sc = bd = 0
        for s in seeds:
            x0 = next(iter(T[s].values()))
            k = L0.act(r18.recorded(s, EPS), (x0["rest_sh_deg"], x0["rest_el_deg"]), tau, F)
            if k is None:
                continue
            x = T[s][k]; sc += x["done"] and harm(x) < 0.5; bd += harm(x) >= 0.5
        if sc >= nF - 0.025 * len(seeds) and (best is None or bd < best[1]):
            best = (tau, bd, sc)
    tau_std = json.load(open(os.path.join(TOP, "out", "round24", "policy.json")))["tau"]["20"]
    json.dump(dict(tau_std=tau_std, tau_norest=best[0], norest_insample=dict(harmful=best[1], safe=best[2])),
              open(os.path.join(OUT, "policy.json"), "w"), indent=1)
    print("tau_std", tau_std, "tau_norest", best)


def _job(spec):
    import shoulder_task as st
    cond, seed, fs, fe, o, a, sp = spec
    try:
        r = st.run(seed, fs, fe, o, a, sp, soft_deg=(20.0 if cond == "soft20" else None))
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r.update(cond=cond, kind="truth")
    return r


def _run(jobs, name, label):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        print("exists", path); return
    json.dump(dict(rows=r17._pool(_job, sorted(set(jobs)), label)), open(path, "w"))


def test1():
    F, _, _ = _pol()
    A, B = _sets()
    if os.environ.get("WM_SMOKE") == "1":
        A, B = [104, 105], [106, 108]
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    json.dump(dict(A=A, B=B), open(os.path.join(OUT, f"seeds{tag}.json"), "w"))
    jobs = [("hard", s) + person_std(s) + k for s in A for k in STRATS]
    jobs += [("shift", s) + person_shift(s) + F for s in B]
    _run(jobs, f"test1{tag}.json", "round 25 stage 1")


def _rules(F, Ls, Ln, pol, P, true, seed, rest):
    """every rule's pick for this person: dict name -> strategy or None"""
    rn = np.asarray(rest) + rest_noise(seed)
    rec, rec_half = record(seed, true), record(seed, true, documented=documented(seed))
    limited = min(true) < UNRESTRICTED
    out = {"F": F,
           "M": F if min(rec) >= UNRESTRICTED else choose(P, rec, F),
           "S_C": None if min(rec) < 3.5 else F,
           "L_std": Ls.act(rec, rest, pol["tau_std"], F),
           "L_norest": Ln.act(rec, rest, pol["tau_norest"], F),
           "L_restnoise": Ls.act(rec, rn, pol["tau_std"], F),
           "M_half": F if min(rec_half) >= UNRESTRICTED else choose(P, rec_half, F),
           "L_std_half": Ls.act(rec_half, rest, pol["tau_std"], F),
           "L_norest_half": Ln.act(rec_half, rest, pol["tau_norest"], F),
           "L_restnoise_half": Ls.act(rec_half, rn, pol["tau_std"], F),
           "CAT_half": None if (limited and documented(seed)) else F}
    return out


def _load(names):
    rows = []
    for n in names:
        p = os.path.join(OUT, n)
        if os.path.exists(p):
            rows += json.load(open(p))["rows"]
    T = {}
    for x in rows:
        if "error" not in x:
            T.setdefault(x["cond"], {}).setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    return T, sum("error" in x for x in rows), len(rows)


def test2():
    F, Ls, Ln, pol = _policies()
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    S = json.load(open(os.path.join(OUT, f"seeds{tag}.json")))
    T, _, _ = _load([f"test1{tag}.json"])
    jobs = []
    for s in S["A"][:N_SOFT]:
        x = T["hard"][s][F]
        picks = _rules(F, Ls, Ln, pol, P, person_std(s), s, (x["rest_sh_deg"], x["rest_el_deg"]))
        for k in (set(picks.values()) | {F}) - {None}:
            jobs.append(("soft20", s) + person_std(s) + k)
    for s in S["B"]:
        x = T["shift"][s][F]
        picks = _rules(F, Ls, Ln, pol, P, person_shift(s), s, (x["rest_sh_deg"], x["rest_el_deg"]))
        for k in set(picks.values()) - {None, F}:
            jobs.append(("shift", s) + person_shift(s) + k)
    _run(jobs, f"test2{tag}.json", "round 25 stage 2")


def evaluate():
    from round14_recompute import ci, fmt
    F, Ls, Ln, pol = _policies()
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    tag = "_smoke" if os.environ.get("WM_SMOKE") == "1" else ""
    S = json.load(open(os.path.join(OUT, f"seeds{tag}.json")))
    T, err, n = _load([f"test1{tag}.json", f"test2{tag}.json"])
    print(f"round 25 {'SMOKE' if tag else ''}: {n} episodes, {err} errors; A {len(S['A'])}, B {len(S['B'])}; tau {pol}")

    def table(cond, people, pfun, extra_oracle=False):
        res = {}
        for s in people:
            x0 = T[cond][s][F]
            picks = _rules(F, Ls, Ln, pol, P, pfun(s), s, (x0["rest_sh_deg"], x0["rest_el_deg"]))
            if extra_oracle:
                picks["O"] = "ORACLE"
            for a, k in picks.items():
                res.setdefault(a, {})
                if k == "ORACLE":
                    c = [(harm(x), -x["done"], kk) for kk, x in T[cond][s].items() if x["done"] and harm(x) < 0.5]
                    k = min(c)[2] if c else None
                if k is None:
                    res[a][s] = dict(safe=0.0, bad=0.0, bad25=0.0, safe25=0.0, dec=1.0)
                else:
                    x = T[cond][s][k]; h = harm(x)
                    res[a][s] = dict(safe=float(x["done"] and h < 0.5), bad=float(h >= 0.5),
                                     safe25=float(x["done"] and h < 2.5), bad25=float(h >= 2.5), dec=0.0)
        print(f"  {'rule':17s} {'safe':>5} {'harmful':>8} {'declined':>9} | {'safe@2.5':>8} {'harm@2.5':>8}")
        for a, R in res.items():
            print(f"  {a:17s} {int(sum(R[s]['safe'] for s in people)):5d} {int(sum(R[s]['bad'] for s in people)):8d} {int(sum(R[s]['dec'] for s in people)):9d} | "
                  f"{int(sum(R[s]['safe25'] for s in people)):8d} {int(sum(R[s]['bad25'] for s in people)):8d}")
        return res

    def pair(res, a, b, key, people):
        return ci([res[a][s][key] - res[b][s][key] for s in people])

    def test(name, res, people, a, b, need_safe=None):
        db, ds = pair(res, a, b, "bad", people), pair(res, a, b, "safe", people)
        db25 = pair(res, a, b, "bad25", people)
        ok = db is not None and db[2] < 0 and (need_safe is None or ds[1] >= need_safe)
        print(f"  {name} {a} - {b}: harmful {fmt(db)} | safe {fmt(ds)} | harmful@2.5 {fmt(db25)} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")

    A = S["A"]
    print(f"\n=== A: standard population, hard limits (n={len(A)}) ===")
    R = table("hard", A, person_std, extra_oracle=True)
    test("H25-1", R, A, "L_std", "M", need_safe=-0.02)
    test("H25-1", R, A, "L_std", "F")
    test("H25-2", R, A, "L_std_half", "F")
    for a in ("M_half", "CAT_half"):
        test("  (report)", R, A, "L_std_half", a)
    und = [s for s in A if min(person_std(s)) < UNRESTRICTED and not documented(s)]
    print(f"  undocumented restricted persons: {len(und)}; on them L_std_half harmful {int(sum(R['L_std_half'][s]['bad'] for s in und))} vs F {int(sum(R['F'][s]['bad'] for s in und))}")
    test("H25-3", R, A, "L_restnoise", "M")
    test("H25-3", R, A, "L_norest", "M")
    Asoft = A[:N_SOFT]
    print(f"\n=== A first {len(Asoft)}: soft20 end-feel, net harm ===")
    Rs = table("soft20", Asoft, person_std)
    test("H25-5", Rs, Asoft, "L_std", "F")
    test("H25-5", Rs, Asoft, "L_std", "M")
    B = S["B"]
    print(f"\n=== B: shifted population (more and tighter contractures), hard limits (n={len(B)}) ===")
    print(f"  restricted in B: shoulder {sum(person_shift(s)[0] < UNRESTRICTED for s in B)}, elbow {sum(person_shift(s)[1] < UNRESTRICTED for s in B)}")
    Rb = table("shift", B, person_shift)
    test("H25-4", Rb, B, "L_std", "F")
    test("H25-4", Rb, B, "L_std", "M")


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(select=select, test1=test1, test2=test2, eval=evaluate).get(st)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True); f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
