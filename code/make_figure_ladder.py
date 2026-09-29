"""Information-ladder figure from out/round24/eval.txt (confirmation set, 300 new persons).
Run with ../.venv_mm.  Numbers are parsed from the eval output, not typed in."""
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

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(4.6, 3.4))
groups = [
    ("F", "no knowledge (fixed rule)", "#999999", "o"),
    ("CAT_half", "limitation flag, half documented", "#c7a76c", "s"),
    ("CAT", "perfect flag, decline if flagged", "#8c6d31", "s"),
    ("S_C(20)", "degrees ±20° + one-line threshold", "#e08a1e", "^"),
    ("M(20)", "degrees ±20° + population body model", "#1f5fa8", "D"),
    ("M(4)", "degrees ±4° + population body model", "#6b9bd1", "D"),
    ("L(20)", "degrees ±20° + policy learned in simulation", "#27ae60", "*"),
    ("L(4)", "degrees ±4° + policy learned in simulation", "#7fd49a", "*"),
    ("O (perfect)", "perfect knowledge (ceiling)", "#000000", "P"),
]
for key, lab, col, mk in groups:
    s, h, d = rows[key]
    ax.scatter(s, h, s=(110 if mk == "*" else 45), color=col, marker=mk, zorder=3, label=lab)
ax.set_xlabel("safe completions (of 300)")
ax.set_ylabel("harmful outcomes (of 300)")
ax.set_title("how much must the robot know about the person?", fontsize=8, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax.annotate("", xy=(rows["O (perfect)"][0], 2), xytext=(rows["L(20)"][0], rows["L(20)"][1]),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8, ls="--"))
fig.tight_layout()
os.makedirs(os.path.join(TOP, "out", "figs"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(TOP, "out", "figs", f"figL_information_ladder.{ext}"), bbox_inches="tight")
print("wrote figL_information_ladder", rows)
