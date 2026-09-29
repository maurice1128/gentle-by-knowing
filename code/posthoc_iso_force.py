"""POST HOC (not preregistered; no new simulation): contact force in harmful lifts vs the ISO/TS 15066 limits.

Episodes: the 600 confirmation persons with full 48-strategy grids (round 24 test.json + round 25 test1.json, hard
limits; posthoc_baselines.grids), harm = dev_r24_learned.harm (robot-caused end-range load), harmful >= 0.5 N m.
Arms: the fixed rule F (every person), and the preregistered learned policy L (round25.KNN use_rest=True, tau 0.65,
record error 20 deg) on the persons it lifts (its executed picks).  For context also all lifts of F and every
strategy in the grids.

Stored force signal: shoulder_task.run returns peak_force = max over the whole episode of care_env6 step_peak
"force", i.e. the per-physics-substep maximum of the TOTAL normal contact force on the capsule (probe_tip geom),
summed over all contacts and all partner bodies (care_env6.probe_force; noise-free ground truth, NOT the noisy
observation, NOT a 50-ms average - the 50-ms average is used only for the joint loads lim_sh / lim_el).  Partner
bodies are not stored per episode, so the force cannot be split into forearm vs hand/other contact.

ISO/TS 15066:2016 Annex A, Table A.2 (values as recorded in notes/lit_pain_sites_2026-09-20.md sec. 1 and
notes/lit_index_validation_2026-09-20.md sec. 2.1, both read from the standard's page images; also
scripts/harm_oracle.py): body region "lower arms and wrist joints": maximum permissible quasi-static force 160 N,
transient multiplier 2 -> 320 N; quasi-static pressure forearm muscle (area 15) 180 N/cm^2, radial bone (14)
190 N/cm^2, arm nerve (16) 180 N/cm^2 (transient x2).  Force column = AIS-1 minor-injury literature review
(footnote b), pressure column = 75th percentile pain onset (footnote a).
Pressure: no contact area / pressure is stored for these episodes.  The only area estimate in the code is the
nominal pressure of care_env6 (force / (pi r_probe^2), r_probe = 2.5 cm -> 19.6 cm^2), which is a disc of the
capsule radius, not a measured contact patch; it is reported only as an indicative number, together with a
deliberately pessimistic bound (all force on the 1.4 x 1.4 cm face of the ISO / Mainz pressure test probe).
Writes out/posthoc_iso_force.txt.  Run with the .venv_mm python."""
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
from round21 import _pol
from dev_r24_learned import load_train, harm
from round25 import KNN
from posthoc_baselines import grids

EPS = 20.0
THR = 0.5
ISO_QS, ISO_TR = 160.0, 320.0                 # N, lower arms and wrist joints (quasi-static, transient x2)
ISO_P = {"forearm muscle 180": 180.0, "radial bone 190": 190.0, "arm nerve 180": 180.0}   # N/cm^2 quasi-static
AREA = math.pi * 2.5 ** 2                     # cm^2, care_env6 nominal (probe_radius 0.025 m)
AREA_ISO = 1.4 * 1.4                          # cm^2, flat face of the Mainz / ISO Table A.2 pressure test probe


def main():
    F, _, _ = _pol()
    T, SET = grids(); people = sorted(T)
    Tr = load_train(); trp = sorted(Tr)
    tau = json.load(open(os.path.join(TOP, "out", "round25", "policy.json")))["tau_std"]
    Lk = KNN(Tr, trp, True)
    rec = lambda s: r18.recorded(s, EPS)
    rest = lambda s: (T[s][F]["rest_sh_deg"], T[s][F]["rest_el_deg"])
    out = []
    say = lambda t="": (print(t), out.append(t))

    def summ(label, xs):
        f = np.array([x["peak_force"] for x in xs], float)
        if not len(f):
            say(f"  {label}: n=0"); return
        say(f"  {label}: n={len(f)}  peak contact force median {np.median(f):.1f} N, p90 {np.percentile(f, 90):.1f} N, "
            f"max {f.max():.1f} N (min {f.min():.1f})")
        say(f"      below {ISO_QS:g} N (quasi-static): {np.mean(f < ISO_QS):.1%} ({int((f < ISO_QS).sum())}/{len(f)});  "
            f"below {ISO_TR:g} N (transient): {np.mean(f < ISO_TR):.1%} ({int((f < ISO_TR).sum())}/{len(f)})")
        p = f / AREA
        say(f"      indicative nominal pressure (force / {AREA:.1f} cm^2): median {np.median(p):.2f}, max {p.max():.2f} N/cm^2 "
            f"-> below 180 N/cm^2: {np.mean(p < 180):.0%}; force needed to reach 180 N/cm^2 on this area: {180 * AREA:.0f} N")
        q = f / AREA_ISO
        say(f"      worst-case pressure if all force sat on the ISO test-probe face ({AREA_ISO:.2f} cm^2): median {np.median(q):.1f}, "
            f"max {q.max():.1f} N/cm^2 -> below 180 N/cm^2: {np.mean(q < 180):.1%}")

    Fl = [T[s][F] for s in people]
    Fh = [x for x in Fl if harm(x) >= THR]
    Lp = {s: Lk.act(rec(s), rest(s), tau, F) for s in people}
    Ll = [T[s][k] for s, k in Lp.items() if k is not None]
    Lh = [x for x in Ll if harm(x) >= THR]
    allx = [x for s in people for x in T[s].values()]
    allh = [x for x in allx if harm(x) >= THR]
    say(f"POST HOC contact force vs ISO/TS 15066 Annex A (lower arms and wrist joints: {ISO_QS:g} N quasi-static, "
        f"{ISO_TR:g} N transient).")
    say(f"Episodes: {len(people)} confirmation persons with full grids (r24 {sum(SET[s] == 'r24' for s in people)}, "
        f"r25A {sum(SET[s] == 'r25A' for s in people)}), hard limits; harmful = robot-caused end-range load >= {THR} N m.")
    say("Force signal: peak_force = raw per-substep peak of the total normal contact force on the capsule (all contacts, "
        "noise-free; not 50-ms averaged).")
    say("")
    say(f"Fixed rule F {F}:")
    summ("harmful lifts", Fh)
    summ("all lifts (context)", Fl)
    say(f"Learned policy L (tau {tau:g}), executed picks:")
    summ("harmful lifts", Lh)
    summ("all lifted persons (context)", Ll)
    say("All 48 strategies x all persons (context):")
    summ("harmful episodes", allh)
    summ("all episodes", allx)
    hf = np.array([harm(x) for x in Fh]); ff = np.array([x["peak_force"] for x in Fh])
    if len(hf) > 2:
        rs = np.corrcoef(np.argsort(np.argsort(ff)), np.argsort(np.argsort(hf)))[0, 1]
        say(f"\nWithin F's harmful lifts: end-range load median {np.median(hf):.2f} N m (max {hf.max():.2f}); "
            f"Spearman(peak force, load) = {rs:.2f}")
    say("\nPressure: not computable from stored data (no contact area / patch is logged; the capsule-forearm contact is a")
    say("MuJoCo point/line contact).  The nominal-pressure lines above use care_env6's own disc area (pi x 2.5^2 cm^2) and are")
    say("indicative only; ISO pressure limits (180-190 N/cm^2) refer to 1-mm^2 peak pressure under a 1.4 x 1.4 cm probe.")
    open(os.path.join(TOP, "out", "posthoc_iso_force.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
