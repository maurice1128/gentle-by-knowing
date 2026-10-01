"""Information-ladder figure from out/round24/eval.txt (confirmation set, 300 new persons).
Run with ../.venv_mm.  Numbers are parsed from the eval output, not typed in.
Added 2026-09-30: one POST HOC marker (hollow) - Lflag = complete flags + settled posture, kNN K=25, tau 0.65
(scripts/posthoc_baselines.py), on the same round-24 300 persons, parsed from the "r24 (n=300)" block of
out/posthoc_baselines.txt (FLAGKNN line: safe 183, harmful 25, declined 81)."""
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

pb = open(os.path.join(TOP, "out", "posthoc_baselines.txt"), encoding="utf-8").read()
pb = pb.split("=== confirmation persons: r24 (n=300) ===")[1].split("===")[0]
m = re.search(r"^\s+FLAGKNN\(tau 0\.65\)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", pb, re.M)
LFLAG = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
assert rows["L(20)"] == (174, 25, 91) and LFLAG == (183, 25, 81), (rows["L(20)"], LFLAG)

# Frontiers: single-column figure, canvas = printed size (85 mm wide), all text >= 8 pt.
MM = 1 / 25.4
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(85 * MM, 145 * MM), layout="constrained")
ax = fig.add_subplot()
# key, letter used in the paper, legend text, colour, marker, letter offset (points), alignment
groups = [
    ("F", "F", "fixed rule, no knowledge", "#999999", "o", (-6, 0), "right"),
    ("CAT_half", "CAT½", "flag, half documented → decline", "#c7a76c", "s", (-6, 0), "right"),
    ("CAT", "CAT", "flag, complete → decline", "#8c6d31", "s", (6, 0), "left"),
    ("S_C(20)", "S", "ranges, SD 20° error + record threshold", "#e08a1e", "^", (-6, 0), "right"),
    ("M(20)", "M", "ranges, SD 20° error + body model", "#1f5fa8", "D", (-6, 3), "right"),
    ("M(4)", "M 4°", "ranges, SD 4° error + body model", "#6b9bd1", "D", (6, 3), "left"),
    ("L(20)", "L", "ranges, SD 20° error + learned policy", "#27ae60", "*", (-7, 0), "right"),
    ("L(4)", "L 4°", "ranges, SD 4° error + learned policy", "#7fd49a", "*", (3, -7), "left"),
    ("O (perfect)", "O", "perfect knowledge (ceiling)", "#000000", "P", (0, 7), "center"),
]
for key, let, lab, col, mk, off, ha in groups:
    s, h, d = rows[key]
    ax.scatter(s, h, s=(110 if mk == "*" else 45), color=col, marker=mk, zorder=3, label=f"{let}: {lab}")
    ax.annotate(let, (s, h), xytext=off, textcoords="offset points", ha=ha,
                va=("bottom" if off[1] > 4 else "top" if off[1] < 0 else "center"), fontsize=8, zorder=4)
    if key == "L(4)":        # post hoc marker, hollow, legend entry right after the L rungs
        ax.scatter(LFLAG[0], LFLAG[1], s=45, marker="*", facecolors="none", edgecolors="#1e8449", linewidths=0.9,
                   zorder=5, label="Lflag: complete flag + posture\n+ learned policy (post hoc)")
        ax.annotate("Lflag", (LFLAG[0], LFLAG[1]), xytext=(5, 5), textcoords="offset points", ha="left",
                    va="bottom", fontsize=8, zorder=4)
ax.set_xlabel("safe completions (of 300)")
ax.set_ylabel("harmful outcomes (of 300)")
fig.suptitle("how much must the robot know about the person?", fontsize=8)
ax.set_xlim(100, 248); ax.set_ylim(-8, 118)
fig.legend(loc="outside lower center", ncol=1, frameon=False, handletextpad=0.4, borderaxespad=0.2)
ax.annotate("", xy=(rows["O (perfect)"][0], 2), xytext=(rows["L(20)"][0], rows["L(20)"][1]),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8, ls="--"))
os.makedirs(os.path.join(TOP, "out", "figs"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(TOP, "out", "figs", f"figL_information_ladder.{ext}"), dpi=600)
print("wrote figL_information_ladder", rows)
