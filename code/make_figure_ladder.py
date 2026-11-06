"""Information-ladder figure (revised for Reviewer 1, comments 3 and 4), round-24 confirmation set, 300 new persons.
A: harmful outcomes against safe completions; the line joins one learner (post hoc, rule held fixed, tau 0.65) across
six levels of knowledge; separate markers are the prespecified controllers.
B: what happens to all 300 persons, declines included.
Controllers are parsed from out/round24/eval.txt, the learner rungs from out/posthoc_fixed_learner.json.
Run with ../.venv_mm."""
import json
import os
import re

TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

txt = open(os.path.join(TOP, "out", "round24", "eval.txt"), encoding="utf-8").read()
block = txt.split("=== harmful if robot-caused end-range load >= 0.5 N m ===")[1].split("H24-1")[0]
rows = {}
for line in block.splitlines():
    m = re.match(r"\s+(\S.*?)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$", line)
    if m:
        rows[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
assert rows["L(20)"] == (174, 25, 91)
FL = json.load(open(os.path.join(TOP, "out", "posthoc_fixed_learner.json")))
RUNGS = [("posture", "posture only"), ("half flag", "half flag"), ("merged flag", "merged flag"),
         ("joint flags", "joint flags"), ("ranges 20", "ranges ±20°"), ("ranges 4", "ranges ±4°")]
learn = []
for key, lab in RUNGS:
    r = FL[f"{key}|uniform|tau|round 24"]
    learn.append((lab, r["safe"], r["harmful"], r["declined"]))

MM = 1 / 25.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(180 * MM, 105 * MM), layout="constrained", width_ratios=[1, 1.15])
a = ax[0]
groups = [("F", "F", "#999999", "o", (-6, 0), "right"), ("CAT_half", "CAT½", "#c7a76c", "s", (-6, 0), "right"),
          ("CAT", "CAT", "#8c6d31", "s", (-6, 0), "right"), ("S_C(20)", "S", "#e08a1e", "^", (-6, 0), "right"),
          ("M(20)", "M", "#1f5fa8", "D", (6, 3), "left"), ("L(20)", "L", "#27ae60", "*", (-7, 2), "right"),
          ("O (perfect)", "O", "#000000", "P", (0, 7), "center")]
for key, let, col, mk, off, ha in groups:
    s, h, d = rows[key]
    a.scatter(s, h, s=(110 if mk == "*" else 40), color=col, marker=mk, zorder=3)
    a.annotate(let, (s, h), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=8, zorder=4)
a.plot([x[1] for x in learn], [x[2] for x in learn], "-o", color="#7a3fa0", ms=3.5, lw=0.9, zorder=2,
       label="one learner, knowledge varied (post hoc)")
offs = {"posture only": (5, 3, "left"), "half flag": (5, 3, "left"), "merged flag": (10, -6, "left"), "joint flags": (6, 5, "left"),
        "ranges ±20°": (-34, 17, "center"), "ranges ±4°": (7, -6, "left")}
for lab, s, h, d in learn:
    ox, oy, ha = offs[lab]
    kw = dict(arrowprops=dict(arrowstyle="-", color="#7a3fa0", lw=0.5)) if lab == "ranges ±20°" else {}
    a.annotate(lab, (s, h), xytext=(ox, oy), textcoords="offset points", fontsize=8, color="#7a3fa0", ha=ha, **kw)
a.set_xlabel("safe completions (of 300)"); a.set_ylabel("harmful outcomes (of 300)")
a.set_xlim(70, 262); a.set_ylim(-8, 118)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)

b = ax[1]
names = [("F", rows["F"]), ("CAT½", rows["CAT_half"]), ("CAT", rows["CAT"]), ("S", rows["S_C(20)"]), ("M", rows["M(20)"]),
         ("L", rows["L(20)"])] + [(lab, (s, h, d)) for lab, s, h, d in learn] + [("O", rows["O (perfect)"])]
parts = [("safe completion", "#4caf72"), ("harmful", "#c0392b"), ("failed lift", "#bbbbbb"), ("declined", "#9cc3e6")]
y = list(range(len(names)))[::-1]
for yy, (n, (s, h, d)) in zip(y, names):
    left = 0
    for val, (lab, col) in zip((s, h, 300 - s - h - d, d), parts):
        b.barh(yy, val, left=left, color=col, height=0.74, label=lab if yy == y[0] else None)
        if val >= 25:
            b.text(left + val / 2, yy, str(val), ha="center", va="center", fontsize=8)
        left += val
b.set_yticks(y); b.set_yticklabels([n for n, _ in names]); b.set_xlim(0, 300); b.set_xlabel("persons (of 300)")
b.spines["left"].set_visible(False); b.tick_params(axis="y", length=0)
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, frameon=False, handlelength=1, columnspacing=1)
for x, lab in zip(ax, "AB"):
    x.text(-0.13, 1.02, lab, transform=x.transAxes, ha="left", va="bottom", fontsize=10,
           fontweight="bold").set_in_layout(False)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(TOP, "out", "figs", f"figL_information_ladder.{ext}"), dpi=600)
print("wrote figL_information_ladder", learn)
