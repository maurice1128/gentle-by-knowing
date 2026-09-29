"""POST HOC analyses requested by the manuscript audit (paper/frontiers/REVIEW_AUDIT.md, C2/C7/C8/C9/C11).
Not preregistered; no new simulation.  Uses the existing full 48-strategy outcome grids of the round-24 and
round-25 A confirmation persons (600, hard limits, record error 20 deg) and the 400 training persons of L.

Every tuned baseline is tuned ONLY on the 400 training persons with the same criterion used for L's tau:
fewest harmful outcomes among settings whose safe completions are >= F's - 2.5 percentage points.

  L        the preregistered learned policy (kNN on recorded degrees + flags + posture, tau 0.65)
  TYPE     lookup table: restriction type from the record flags (none / shoulder / elbow / both); per type the
           strategy with the highest training safe rate; decline if that rate < tau_T
  FLAGKNN  kNN (K 25) on the two flags + two settled angles only (no recorded degrees); tau tuned
  S*       record threshold re-tuned for 20 deg error
  M*       body-model rule with margin / abstain threshold re-tuned for 20 deg error
  CAT      complete flag -> decline
Writes out/posthoc_baselines.txt and out/figs/figP_pareto.{pdf,png}.  Run with the .venv_mm python."""
import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person
from round21 import _pol, choose
from dev_r24_learned import load_train, harm
from round25 import KNN

EPS = 20.0
THR = 0.5
TAUS = [x / 20 for x in range(0, 21)]


def grids():
    T, sets = {}, {}
    for tag, path in (("r24", ("round24", "test.json")), ("r25A", ("round25", "test1.json"))):
        for x in json.load(open(os.path.join(TOP, "out", *path)))["rows"]:
            if "error" in x or x.get("cond", "hard") != "hard":
                continue
            T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
            sets[x["seed"]] = tag
    T = {s: g for s, g in T.items() if len(g) == len(STRATS)}
    return T, {s: sets[s] for s in T}


def tcrit(n):
    # two-sided 95% t quantile (Cornish-Fisher approximation, accurate to ~1e-3 for n >= 10)
    z = 1.959964
    d = n - 1
    return z + (z ** 3 + z) / (4 * d) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * d ** 2)


def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    return m, m - tcrit(n) * se, m + tcrit(n) * se


def fmt(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def main():
    F, _, _ = _pol()
    T, SET = grids(); people = sorted(T)
    Tr = load_train(); trp = sorted(Tr)
    tau_L = json.load(open(os.path.join(TOP, "out", "round25", "policy.json")))["tau_std"]
    Lk = KNN(Tr, trp, True)
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    rec = lambda s: r18.recorded(s, EPS)
    rest = lambda G, s: (G[s][F]["rest_sh_deg"], G[s][F]["rest_el_deg"])
    flags = lambda r: (r[0] < UNRESTRICTED, r[1] < UNRESTRICTED)
    out = []
    say = lambda t="": (print(t), out.append(t))

    def outcome(G, persons, pick):
        safe = bad = dec = 0
        per = {}
        for s in persons:
            k = pick(s)
            if k is None:
                dec += 1; per[s] = (0.0, 0.0, 1.0, 0.0); continue
            x = G[s][k]; h = harm(x); sf = float(x["done"] and h < THR)
            safe += sf; bad += h >= THR; per[s] = (sf, float(h >= THR), 0.0, h)
        return int(safe), int(bad), dec, per

    def tune(name, family):
        """family: list of (param, pick_fn_on_training).  Criterion as for L's tau."""
        sF, _, _, _ = outcome(Tr, trp, lambda s: F)
        best = None
        for par, fn in family:
            s_, b_, d_, _ = outcome(Tr, trp, fn)
            if s_ >= sF - 0.025 * len(trp):
                if best is None or (b_, -s_) < (best[1], -best[2]):
                    best = (par, b_, s_, d_)
        say(f"  tuned {name} on 400 training persons: param {best[0]}  (train harmful {best[1]}, safe {best[2]}, declined {best[3]}; F safe {sF})")
        return best[0]

    # ---- TYPE lookup (trained on the 400) ----
    def type_of(r):
        a, b = flags(r)
        return {(False, False): "N", (True, False): "S", (False, True): "E", (True, True): "SE"}[(a, b)]
    stat = {}
    for s in trp:
        ty = type_of(rec(s))
        for k, x in Tr[s].items():
            h = harm(x); st = stat.setdefault(ty, {}).setdefault(k, [0, 0, 0])
            st[0] += x["done"] and h < THR; st[1] += h >= THR; st[2] += 1
    table = {}
    for ty, d in stat.items():
        k, (sf, bd, n) = max(d.items(), key=lambda kv: (kv[1][0] / kv[1][2], -kv[1][1] / kv[1][2]))
        table[ty] = (k, sf / n)
    say("TYPE lookup table (training): " + "; ".join(f"{ty}: {table[ty][0]} p_safe {table[ty][1]:.2f}" for ty in sorted(table)))

    def type_pick(tau):
        def f(s):
            ty = type_of(rec(s))
            if ty == "N":
                return F
            k, p = table[ty]
            return k if p >= tau else None
        return f

    # ---- flags + posture kNN ----
    def fk_feats(G, s):
        a, b = flags(rec(s)); r = rest(G, s)
        return np.array([float(a), float(b), r[0], r[1]])
    X = np.array([fk_feats(Tr, s) for s in trp]); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    lab = [{k: (float(x["done"] and harm(x) < THR), float(harm(x) >= THR)) for k, x in Tr[s].items()} for s in trp]

    def fk_pick(G, tau, K=25):
        def f(s):
            if not any(flags(rec(s))):
                return F
            z = (fk_feats(G, s) - mu) / sd
            nn = np.argsort(np.linalg.norm(Z - z, axis=1))[:K]
            sc = {k: (np.mean([lab[i][k][0] for i in nn if k in lab[i]]), np.mean([lab[i][k][1] for i in nn if k in lab[i]])) for k in STRATS}
            k, (p, b) = max(sc.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
            return k if p >= tau else None
        return f

    L_pick = lambda G: (lambda s: Lk.act(rec(s), rest(G, s), tau_L, F))
    L_tau = lambda G, tau: (lambda s: Lk.act(rec(s), rest(G, s), tau, F))
    S_pick = lambda t: (lambda s: None if min(rec(s)) < t else F)
    M_pick = lambda mg, ab: (lambda s: F if min(rec(s)) >= UNRESTRICTED else choose(P, rec(s), F, margin=mg, abstain_over=ab))
    CAT = lambda s: None if any(flags(rec(s))) else F

    say(f"post hoc baselines: {len(people)} confirmation persons with full grids (r24 {sum(SET[s]=='r24' for s in people)}, r25A {sum(SET[s]=='r25A' for s in people)}), record error {EPS:g} deg, harmful >= {THR} N m")
    say("tuning (criterion: fewest harmful with safe >= F - 2.5 pp, on training persons only):")
    tau_T = tune("TYPE tau", [(t, type_pick(t)) for t in TAUS])
    tau_FK = tune("FLAGKNN tau", [(t, fk_pick(Tr, t)) for t in TAUS])
    t_S = tune("S threshold (deg)", [(t, S_pick(t)) for t in np.arange(0, 30.5, 0.5)])
    mm = tune("M (margin, abstain_over)", [((mg, ab), M_pick(mg, ab)) for mg in (0, 1.5, 3, 4.5, 6, 9, 12, 15) for ab in (0, 1.5, 3, 4.5, 6, 9, 1e9)])

    arms = {"F": lambda s: F, "CAT": CAT, "S(3.5, prereg)": S_pick(3.5), f"S*({t_S:g})": S_pick(t_S),
            "M(prereg)": M_pick(3, 3), f"M*{mm}": M_pick(*mm), "L(prereg)": L_pick(T),
            f"TYPE(tau {tau_T:g})": type_pick(tau_T), f"FLAGKNN(tau {tau_FK:g})": fk_pick(T, tau_FK)}
    res = {}
    for tag, ps in (("all 600", people), ("r24", [s for s in people if SET[s] == "r24"]), ("r25A", [s for s in people if SET[s] == "r25A"])):
        say(f"\n=== confirmation persons: {tag} (n={len(ps)}) ===")
        say(f"  {'arm':28s} {'safe':>5} {'harmful':>8} {'declined':>9}")
        for a, f in arms.items():
            s_, b_, d_, per = outcome(T, ps, f)
            res[(tag, a)] = per
            say(f"  {a:28s} {s_:5d} {b_:8d} {d_:9d}")
        for a in arms:
            if a.startswith("L(") or a == "F":
                continue
            db = ci([res[(tag, "L(prereg)")][s][1] - res[(tag, a)][s][1] for s in ps])
            ds = ci([res[(tag, "L(prereg)")][s][0] - res[(tag, a)][s][0] for s in ps])
            say(f"  L - {a:24s}: harmful {fmt(db)} | safe {fmt(ds)}")

    # ---- paired lifted-only and type breakdown (all 600) ----
    say("\n=== paired, on the persons each rule lifts (all 600) ===")
    for a in ("L(prereg)", "M(prereg)"):
        lifted = [s for s in people if res[("all 600", a)][s][2] == 0]
        ha = sum(res[("all 600", a)][s][1] for s in lifted); hf = sum(res[("all 600", "F")][s][1] for s in lifted)
        la = np.mean([res[("all 600", a)][s][3] for s in lifted]); lf = np.mean([res[("all 600", "F")][s][3] for s in lifted])
        say(f"  {a}: lifted {len(lifted)}; harmful {int(ha)} vs F {int(hf)}  ({fmt(ci([res[('all 600', a)][s][1] - res[('all 600', 'F')][s][1] for s in lifted]))}); mean load {la:.2f} vs F {lf:.2f}")
    say("\n=== by true restriction type (all 600): safe / harmful / declined ===")
    tt = lambda s: {(False, False): "none", (True, False): "shoulder", (False, True): "elbow", (True, True): "both"}[tuple(x < UNRESTRICTED for x in person(s))]
    for ty in ("none", "shoulder", "elbow", "both"):
        ps = [s for s in people if tt(s) == ty]
        row = []
        for a in ("F", "M(prereg)", "L(prereg)"):
            R = res[("all 600", a)]
            row.append(f"{a}: {int(sum(R[s][0] for s in ps))}/{int(sum(R[s][1] for s in ps))}/{int(sum(R[s][2] for s in ps))}")
        orc = sum(any(x["done"] and harm(x) < THR for x in T[s].values()) for s in ps)
        say(f"  {ty:8s} n={len(ps):3d} | " + " | ".join(row) + f" | oracle safe {orc}")

    # ---- oracle-gap decomposition (all 600) ----
    R = res[("all 600", "L(prereg)")]
    has_safe = {s: any(x["done"] and harm(x) < THR for x in T[s].values()) for s in people}
    unnec = sum(R[s][2] == 1 and has_safe[s] for s in people)
    harm_nosafe = sum(R[s][1] == 1 and not has_safe[s] for s in people)
    harm_safe = sum(R[s][1] == 1 and has_safe[s] for s in people)
    incomplete = sum(R[s][2] == 0 and R[s][0] == 0 and R[s][1] == 0 and has_safe[s] for s in people)
    say(f"\n=== L vs oracle (all 600): oracle safe {sum(has_safe.values())}, L safe {int(sum(R[s][0] for s in people))}; "
        f"missing safe completions = unnecessary refusals {unnec} + harmful where safe existed {harm_safe} + incomplete where safe existed {incomplete}; "
        f"L harmful where no safe strategy existed {harm_nosafe}")

    # ---- Pareto sweeps on the confirmation persons (all 600) ----
    curves = {
        "learned policy L (tau)": [outcome(T, people, L_tau(T, t))[:2] for t in TAUS],
        "type lookup (tau)": [outcome(T, people, type_pick(t))[:2] for t in TAUS],
        "flags + posture kNN (tau)": [outcome(T, people, fk_pick(T, t))[:2] for t in TAUS],
        "record threshold S (deg)": [outcome(T, people, S_pick(t))[:2] for t in np.arange(0, 30.5, 1.0)],
        "body model M (margin)": [outcome(T, people, M_pick(mg, 3))[:2] for mg in (0, 1.5, 3, 4.5, 6, 9, 12, 15, 20)],
    }
    pts = {"fixed rule F": outcome(T, people, lambda s: F)[:2], "flag -> decline": outcome(T, people, CAT)[:2],
           "L (preregistered)": outcome(T, people, L_pick(T))[:2],
           "oracle": (sum(has_safe.values()), 0)}
    say("\n=== Pareto sweeps (all 600; safe, harmful) ===")
    for k, v in curves.items():
        say(f"  {k}: " + " ".join(f"({a},{b})" for a, b in v))
    for k, v in pts.items():
        say(f"  {k}: {v}")
    open(os.path.join(TOP, "out", "posthoc_baselines.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Frontiers: single-column figure, canvas = printed size (85 mm wide), all text >= 8 pt, legend below the axes.
    MM = 1 / 25.4
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(85 * MM, 118 * MM), layout="constrained")
    cols = ["#2ca05a", "#8c564b", "#9467bd", "#d98c1f", "#2060b0"]
    shown = {"learned policy L (tau)": "learned policy L (τ sweep)", "type lookup (tau)": "type lookup (τ sweep)",
             "flags + posture kNN (tau)": "flags + posture kNN (τ sweep)",
             "record threshold S (deg)": "record threshold S (degree sweep)", "body model M (margin)": "body model M (margin sweep)",
             "fixed rule F": "fixed rule F", "flag -> decline": "flag → decline (CAT)",
             "L (preregistered)": "L (preregistered)", "oracle": "perfect knowledge O (ceiling)"}
    for (k, v), c in zip(curves.items(), cols):
        ax.plot([a for a, _ in v], [b for _, b in v], "-o", color=c, ms=2.5, lw=1.1, label=shown[k])
    mk = {"fixed rule F": ("#666666", "o"), "flag -> decline": ("#000000", "s"), "L (preregistered)": ("#2ca05a", "*"), "oracle": ("#000000", "P")}
    for k, v in pts.items():
        ax.plot(*v, marker=mk[k][1], color=mk[k][0], ms=8 if k.startswith("L") else 6, ls="none", label=shown[k])
    ax.set_xlabel("safe completions (of 600)", fontsize=8); ax.set_ylabel("harmful outcomes (of 600)", fontsize=8)
    ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="outside lower center", ncol=1, fontsize=8, frameon=False, handletextpad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(TOP, "out", "figs", f"figP_pareto.{ext}"), dpi=600)

if __name__ == "__main__":
    main()
