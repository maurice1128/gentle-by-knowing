"""POST HOC, requested by Reviewer 1 (comments 1, 2 and minor 5).  No new simulation; stored guard episodes only.
For every guard cell: paired differences R - F in harmful-outcome rate, safe-completion rate and mean load,
the non-superiority test (lower bound of the harmful-rate interval above -5 points), and the load by attempt.
Intervals use t = 2.1, as the round evaluations do.  Counts are checked against each round's eval.txt.
Writes out/posthoc_guard_load.txt and out/posthoc_guard_load.json.  Run with the .venv_mm python."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from posthoc_guard_prevented import tables, policy, eval_counts, gross, net, MULTS

MARGIN = -0.05


def ci(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    return float(m), float(m - 2.1 * se), float(m + 2.1 * se)


def episode(G, s, m, order, guard, hf):
    """As reactive() in the round evaluations, keeping the load of every attempt."""
    loads = []; done = False
    for k in order:
        g = G.get(s, {}).get(k, {}).get(round(guard[k] * m, 3))
        if g is None:
            continue
        loads.append(hf(g))
        if g["done"]:
            done = True
            break
    return done, loads


def main():
    F, order, guard = policy()
    S21 = json.load(open(os.path.join(TOP, "out", "round21", "test_seeds.json")))
    S22 = json.load(open(os.path.join(TOP, "out", "round22", "seeds.json")))["A"]
    S23 = json.load(open(os.path.join(TOP, "out", "round23", "seeds.json")))["W"]
    cells = [("round21", "hard", S21, gross, 0.5, "H21-3"), ("round22", "soft6", S22, gross, 0.5, "H22-1"),
             ("round22", "soft12", S22, gross, 0.5, "H22-1"), ("round23", "soft12", S23, net, 0.5, "Fig. 2"),
             ("round23", "soft20", S23, net, 0.5, "H23-1"), ("round23", "soft25", S23, net, 0.5, "H23-1"),
             ("round23", "soft20", S23, net, 2.5, "H23-1"), ("round23", "soft25", S23, net, 2.5, "H23-1")]
    out, store, cache = [], [], {}
    say = lambda t="": (print(t, flush=True), out.append(t))
    say("POST HOC guard analysis.  Non-superiority: lower bound of R - F harmful rate above -0.05.")
    for rnd, cond, ppl, hf, thr, hyp in cells:
        if rnd not in cache:
            cache[rnd] = tables(rnd)
        T, G = cache[rnd][0].get(cond, {}), cache[rnd][1].get(cond, {})
        Fh = {s: hf(T[s][F]) for s in ppl}; Fd = {s: bool(T[s][F]["done"]) for s in ppl}
        say(f"\n=== {rnd} {cond}, threshold {thr}, n={len(ppl)} ({hyp}); F harmful {sum(Fh[s] >= thr for s in ppl)}, "
            f"safe {sum(Fd[s] and Fh[s] < thr for s in ppl)}, mean load {np.mean(list(Fh.values())):.2f}")
        ev = eval_counts(rnd, cond) if thr == 0.5 else None
        for m in MULTS:
            a = f"R x{m * 1.2:.1f}p95"
            E = {s: episode(G, s, m, order, guard, hf) for s in ppl}
            H = {s: max(E[s][1]) if E[s][1] else 0.0 for s in ppl}
            bad = sum(H[s] >= thr for s in ppl); safe = sum(E[s][0] and H[s] < thr for s in ppl)
            check = "" if ev is None else (" count check ok" if ev[a] == (safe, bad) else " COUNT MISMATCH")
            dh = ci([float(H[s] >= thr) - float(Fh[s] >= thr) for s in ppl])
            ds = ci([float(E[s][0] and H[s] < thr) - float(Fd[s] and Fh[s] < thr) for s in ppl])
            dl = ci([H[s] - Fh[s] for s in ppl])
            verdict = "non-superiority shown" if dh[1] > MARGIN else "inconclusive"
            att = []
            for n in range(3):
                v = [E[s][1][n] for s in ppl if len(E[s][1]) > n]
                att.append((len(v), float(np.mean(v)) if v else 0.0))
            later = sum(len(E[s][1]) > 1 and max(E[s][1][1:]) > E[s][1][0] for s in ppl)
            say(f"  {a}: harmful {bad} safe {safe} load {np.mean(list(H.values())):.2f} | dH {dh[0]:+.3f} [{dh[1]:+.3f}, {dh[2]:+.3f}] {verdict}"
                f" | dS {ds[0]:+.3f} [{ds[1]:+.3f}, {ds[2]:+.3f}] | dLoad {dl[0]:+.2f} [{dl[1]:+.2f}, {dl[2]:+.2f}] ({100 * dl[0] / np.mean(list(Fh.values())):+.0f}%)"
                f" | by attempt n/mean " + " ".join(f"{n}/{x:.2f}" for n, x in att) + f" | peak in a later attempt: {later}" + check)
            store.append(dict(round=rnd, cond=cond, thr=thr, hyp=hyp, m=round(m * 1.2, 1), harmful=bad, safe=safe,
                              load=float(np.mean(list(H.values()))), load_F=float(np.mean(list(Fh.values()))),
                              dH=dh, dS=ds, dL=dl, verdict=verdict, attempts=att, later=later))
    open(os.path.join(TOP, "out", "posthoc_guard_load.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump(store, open(os.path.join(TOP, "out", "posthoc_guard_load.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
