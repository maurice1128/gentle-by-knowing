"""Fig: robustness summary.  Paired differences (95% t-CI) of the learned policy L against the fixed rule F
and the body-model rule M, across every preregistered confirmation condition (rounds 24-26, harm >= 0.5 N m).
Numbers are parsed from the evaluation logs, not retyped.  Run with the .venv_mm python."""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
O = os.path.join(TOP, "out")

NUM = r"([+-]\d+\.\d+)\s*\[\s*([+-]\d+\.\d+),\s*([+-]\d+\.\d+)\]"


def grab(path, section, pattern):
    """First line in `section` (text after the section header) matching pattern -> (harm, safe) triples."""
    txt = open(os.path.join(O, path)).read()
    if section:
        txt = txt[txt.index(section):]
    for line in txt.splitlines():
        if re.search(pattern, line):
            v = re.findall(NUM, line)
            return tuple(map(float, v[0])), tuple(map(float, v[1]))
    raise KeyError((path, section, pattern))


ROWS = [  # label, (file, section, pattern vs F), (file, section, pattern vs M)
    ("Ladder set, record error 20° (n=300)", ("round24/eval.txt", "", r"H24-3 \[0.5\]"), ("round24/eval.txt", "", r"H24-1 \[0.5\]")),
    ("Replication (n=300)", ("round25/eval.txt", "", r"H25-1 L_std - F"), ("round25/eval.txt", "", r"H25-1 L_std - M")),
    ("Half of restrictions unrecorded", ("round25/eval.txt", "", r"H25-2 L_std_half - F"), ("round25/eval.txt", "", r"L_std_half - M_half")),
    ("Posture noise, SD 5°", None, ("round25/eval.txt", "", r"H25-3 L_restnoise - M")),
    ("No posture input", None, ("round25/eval.txt", "", r"H25-3 L_norest - M")),
    ("Graded end-feel 20° (n=150)", ("round25/eval.txt", "soft20", r"H25-5 L_std - F"), ("round25/eval.txt", "soft20", r"H25-5 L_std - M")),
    ("More, tighter contractures (n=250)", ("round25/eval.txt", "=== B", r"H25-4 L_std - F"), ("round25/eval.txt", "=== B", r"H25-4 L_std - M")),
    ("Test physics: reflex gain ×3", ("round26/eval_S.txt", "", r"H26-1 \[0.5\]"), ("round26/eval_S.txt", "", r"H26-2 \[0.5\]")),
    ("Test physics: body variation σ=0.40", ("round26/eval_V.txt", "", r"H26-1 \[0.5\]"), ("round26/eval_V.txt", "", r"H26-2 \[0.5\]")),
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [(lab, grab(*f) if f else None, grab(*m) if m else None) for lab, f, m in ROWS]
    with open(os.path.join(O, "forest_numbers.txt"), "w", encoding="utf-8") as fh:
        for lab, f, m in data:
            fh.write(f"{lab}: vs F {f} | vs M {m}\n")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    n = len(data)
    for j, (title, idx) in enumerate((("harmful-outcome rate, L minus comparator", 0), ("safe-completion rate, L minus comparator", 1))):
        a = ax[j]
        for i, (lab, f, m) in enumerate(data):
            y = n - 1 - i
            for d, off, c, mk, name in ((f, 0.15, "#666666", "o", "vs fixed rule F"), (m, -0.15, "#2060b0", "D", "vs body model M")):
                if d is None:
                    continue
                v, lo, hi = d[idx]
                a.plot([lo, hi], [y + off] * 2, color=c, lw=1.4)
                a.plot(v, y + off, marker=mk, color=c, ms=4, label=name if i == 0 else None)
        a.axvline(0, color="#999999", lw=0.8)
        a.set_title(title, fontsize=8); a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_yticks(range(n)); ax[0].set_yticklabels([d[0] for d in data][::-1], fontsize=7)
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    ax[0].set_xlim(-0.65, 0.05); ax[1].set_xlim(-0.15, 0.15)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(O, "figs", f"figR_robustness.{ext}"), bbox_inches="tight", dpi=200)
    print(open(os.path.join(O, "forest_numbers.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
