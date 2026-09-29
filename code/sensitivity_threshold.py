"""Post-hoc sensitivity analysis (NOT preregistered; reported as such): does the ranking of controllers in
round 24 depend on the harm threshold?  Controller decisions are fixed exactly as in round 24 (they do not
depend on the evaluation threshold, except the oracle); only the evaluation threshold is swept.
Also reports the threshold-free mean robot-caused end-range load with paired 95% CIs.
Writes out/sensitivity_threshold.txt and out/figs/figS_threshold.{pdf,png}.  Run with the .venv_mm python."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import round18 as r18
from round17 import UNRESTRICTED
from round21 import _pol, choose
from dev_r24_learned import load_train, harm, Learned
from round14_recompute import ci, fmt

OUT = os.path.join(TOP, "out", "round24")
THRS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def main():
    F, _, _ = _pol()
    te = json.load(open(os.path.join(OUT, "seeds.json")))
    T = {}
    for x in json.load(open(os.path.join(OUT, "test.json")))["rows"]:
        if "error" not in x:
            T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    pol = json.load(open(os.path.join(OUT, "policy.json")))
    Tr = load_train(); tr = sorted(Tr)
    lib = r18._load_library(); P = r18._predict(lib[1][lib[0][0]], *lib, k_nn=0)
    e = 20.0
    L = Learned(Tr, tr, e); tau = pol["tau"]["20"]
    picks = {"F": {s: F for s in te},
             "S_C(20)": {s: (None if min(r18.recorded(s, e)) < 3.5 else F) for s in te},
             "M(20)": {s: (F if min(r18.recorded(s, e)) >= UNRESTRICTED else choose(P, r18.recorded(s, e), F)) for s in te},
             "L(20)": {s: L.act(s, T[s], e, tau, F) for s in te}}
    lines = [f"post-hoc threshold sensitivity, round 24 confirmation set ({len(te)} persons, record error 20 deg); "
             "decisions fixed, evaluation threshold swept"]

    def load(a, s):
        k = picks[a][s]
        return 0.0 if k is None else harm(T[s][k])

    lines.append("\nthreshold-free: mean robot-caused end-range load (N m), declined = 0")
    for a in picks:
        v = [load(a, s) for s in te]
        lines.append(f"  {a:8s} mean {np.mean(v):5.2f}  median {np.median(v):5.2f}  p90 {np.percentile(v, 90):5.2f}")
    for a, b in (("L(20)", "F"), ("L(20)", "M(20)")):
        lines.append(f"  {a} - {b}: mean load {fmt(ci([load(a, s) - load(b, s) for s in te]))}")
    lines.append("  executed lifts only (declines excluded):")
    for a in picks:
        v = [load(a, s) for s in te if picks[a][s] is not None]
        lines.append(f"  {a:8s} n={len(v):3d} mean {np.mean(v):5.2f}  median {np.median(v):5.2f}")

    lines.append("\nharmful outcomes / safe completions by threshold")
    lines.append("  thr   " + "  ".join(f"{a:>13s}" for a in list(picks) + ["O"]))
    curves = {a: [] for a in list(picks) + ["O"]}
    for t in THRS:
        row = []
        for a in picks:
            bad = sum(load(a, s) >= t for s in te)
            safe = sum(picks[a][s] is not None and T[s][picks[a][s]]["done"] and load(a, s) < t for s in te)
            curves[a].append((bad, safe)); row.append(f"{bad:4d} / {safe:4d}")
        ob = 0; os_ = sum(any(x["done"] and harm(x) < t for x in T[s].values()) for s in te)
        curves["O"].append((ob, os_)); row.append(f"{ob:4d} / {os_:4d}")
        lines.append(f"  {t:4.2f}  " + "  ".join(f"{r:>13s}" for r in row))
    rank_ok = all(curves["L(20)"][i][0] < min(curves["M(20)"][i][0], curves["S_C(20)"][i][0], curves["F"][i][0]) for i in range(len(THRS)))
    lines.append(f"\nL has the fewest harmful outcomes of F/S/M/L at every threshold: {rank_ok}")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(TOP, "out", "sensitivity_threshold.txt"), "w").write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = {"F": ("#666666", "o", "fixed rule F"), "S_C(20)": ("#d98c1f", "^", "record threshold S"),
           "M(20)": ("#2060b0", "D", "body model M"), "L(20)": ("#2ca05a", "*", "learned policy L"),
           "O": ("#000000", "P", "oracle O")}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for a, (c, m, lab) in sty.items():
        ax[0].plot(THRS, [b for b, _ in curves[a]], color=c, marker=m, ms=4, lw=1.2, label=lab)
        ax[1].plot(THRS, [s for _, s in curves[a]], color=c, marker=m, ms=4, lw=1.2, label=lab)
    for a_ in ax:
        a_.set_xscale("log"); a_.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2.5, 5]))
        a_.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.25", "0.5", "1", "2.5", "5"])); a_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a_.set_xlabel("harm threshold (N·m)", fontsize=8); a_.tick_params(labelsize=7)
        a_.axvline(0.5, color="#bbbbbb", lw=0.8, ls=":"); a_.axvline(2.5, color="#bbbbbb", lw=0.8, ls=":")
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("harmful outcomes (of 300)", fontsize=8); ax[1].set_ylabel("safe completions (of 300)", fontsize=8)
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(TOP, "out", "figs", f"figS_threshold.{ext}"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
