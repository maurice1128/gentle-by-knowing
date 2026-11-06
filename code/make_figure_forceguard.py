"""Force-guard figure (revised for Reviewer 1, comments 1 and 2): paired differences of each guard from the fixed
rule, with the intervals the non-superiority test is decided on.  Reads out/posthoc_guard_load.json
(scripts/posthoc_guard_load.py).  Hollow marker: the harmful-rate interval reaches past -5 points (inconclusive).
Run with ../.venv_mm."""
import json
import os

TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open(os.path.join(TOP, "out", "posthoc_guard_load.json")))
series = [("hard limit", "round21", "hard", "#555555"), ("graded 12°", "round23", "soft12", "#e08a1e"),
          ("graded 20°", "round23", "soft20", "#1f5fa8"), ("graded 25°", "round23", "soft25", "#27ae60")]
xs = [0.6, 0.8, 1.0, 1.2]
dodge = [-0.03, -0.01, 0.01, 0.03]

MM = 1 / 25.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 3, figsize=(180 * MM, 80 * MM), layout="constrained")
panels = [("dH", 100, "harmful outcomes, guard − F (points)", "harmful outcomes move by 0–7 points"),
          ("dS", 100, "safe completions, guard − F (points)", "tight guards lose safe completions"),
          ("dL", 1, "mean load, guard − F (N·m)", "tight guards lower the load")]
n_inc = 0
for (name, rnd, cond, col), dx in zip(series, dodge):
    rows = sorted((r for r in D if r["round"] == rnd and r["cond"] == cond and r["thr"] == 0.5), key=lambda r: r["m"])
    assert [r["m"] for r in rows] == xs
    for a, (key, sc, _, _) in zip(ax, panels):
        for r, x in zip(rows, xs):
            est, lo, hi = (v * sc for v in r[key])
            hollow = r["verdict"] == "inconclusive"
            a.plot([x + dx] * 2, [lo, hi], color=col, lw=0.9)
            a.plot(x + dx, est, "o", ms=3.6, mec=col, mfc="white" if hollow else col, mew=0.9, ls="none",
                   label=name if (r is rows[0] and key == "dH") else None)
    n_inc += sum(r["verdict"] == "inconclusive" for r in rows)
assert n_inc == 6, n_inc
for a, (key, _, yl, title) in zip(ax, panels):
    a.axhline(0, color="#999999", lw=0.8)
    a.set_ylabel(yl); a.set_xticks(xs); a.set_xlabel("force guard (× 95th percentile)")
    a.set_title(title, loc="left")
ax[0].axhline(-5, color="#b03030", lw=0.8, ls="--")
ax[0].text(1.27, -5.3, "−5 points", color="#b03030", ha="right", va="top", fontsize=8)
for a, lab in zip(ax, "ABC"):
    a.text(-0.16, 1.04, lab, transform=a.transAxes, ha="left", va="bottom", fontsize=10,
           fontweight="bold").set_in_layout(False)
handles = [plt.Line2D([], [], marker="o", ms=3.6, color=c, ls="none", label=n) for n, _, _, c in series]
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, handletextpad=0.3)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(TOP, "out", "figs", f"figG_force_guard.{ext}"), dpi=600)
print("wrote figG_force_guard; inconclusive cells:", n_inc)
