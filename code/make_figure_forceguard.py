"""Force-guard figure: harmful outcomes and safe completions of the reactive controller at four
guard thresholds, relative to the fixed rule, across joint end-feel conditions.  Parsed from the
preregistered eval outputs (round 21 hard; round 23 soft12 / soft20 / soft25 with robot-caused harm).
Run with ../.venv_mm."""
import os
import re

TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW = re.compile(r"^\s*(F|R x[\d.]+p95)\s+(\d+)\s+(\d+)\s+(\d+)")


def parse(block):
    out = {}
    for line in block.splitlines():
        m = ROW.match(line)
        if m and m.group(1) not in out:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


r21 = open(os.path.join(TOP, "out", "round21", "eval.txt"), encoding="utf-8").read()
r23 = open(os.path.join(TOP, "out", "round23", "eval.txt"), encoding="utf-8").read()
series = {"hard limit (n=120)": parse(r21.split("H21-1")[0])}
for c, w in (("soft12", "12°"), ("soft20", "20°"), ("soft25", "25°")):
    blk = r23.split(f"=== W, condition {c} (n=120) ===")[1].split("harm = net, harmful if >= 2.5")[0]
    series[f"graded resistance {w}"] = parse(blk)

# Frontiers: full-width figure, canvas = printed size (180 mm), all text >= 8 pt.
MM = 1 / 25.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM), layout="constrained")
mults = ["R x0.6p95", "R x0.8p95", "R x1.0p95", "R x1.2p95"]
xs = [0.6, 0.8, 1.0, 1.2]
cols = ["#555555", "#e08a1e", "#1f5fa8", "#27ae60"]
for (name, d), col in zip(series.items(), cols):
    fs, fh = d["F"]
    ax[0].plot(xs, [d[m][1] for m in mults], "o-", color=col, ms=3.5, lw=1.1, label=name)
    ax[0].axhline(fh, color=col, lw=0.8, ls=":")
    ax[1].plot(xs, [d[m][0] for m in mults], "o-", color=col, ms=3.5, lw=1.1)
    ax[1].axhline(fs, color=col, lw=0.8, ls=":")
ax[0].set_ylabel("harmful outcomes (of 120)"); ax[1].set_ylabel("safe completions (of 120)")
for a in ax:
    a.set_ylim(0, 75)
    a.set_xlabel("force guard (× 95th pct of harmless force)"); a.set_xticks(xs)
ax[0].set_title("tightening the guard barely reduces harm", loc="left")
ax[1].set_title("…but removes safe completions", loc="left")
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False,
           handlelength=1.8, columnspacing=1.4)
fig.get_layout_engine().set(h_pad=2 * MM)
fig.text(0.5, 0.005, "Graded resistance 12°/20°/25°: n=120, same persons.  "
         "Dotted lines: fixed rule without a guard (same colour).", ha="center", va="bottom", fontsize=8)
fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(TOP, "out", "figs", f"figG_force_guard.{ext}"), dpi=600)
print({k: v for k, v in series.items()})
