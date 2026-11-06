"""POST HOC, requested by Reviewer 1 (minor 8): the contact force equivalent to 0.5 N m over the 600 confirmation
persons and the eight support sites.  Same arithmetic as lever_arms() in posthoc_guard_prevented.py, one mj_forward per
person (settled posture, body variation of the seed), no rollout.  Lift direction: approach angle of F.
Writes out/posthoc_lever_range.txt.  Run with the .venv_myo python."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import mujoco
import diag_shoulder1 as s1
import diag_lift_oracle as dlo
from care_env6 import AIM_DISTAL, _unit
from supported_scene import _relaxed_qpos
from round17 import STRATS
from posthoc_guard_prevented import policy


def rests():
    R = {}
    for path in (("round24", "test.json"), ("round25", "test1.json")):
        for x in json.load(open(os.path.join(TOP, "out", *path)))["rows"]:
            if "error" not in x and x.get("cond", "hard") == "hard" and "rest_sh_deg" in x:
                R.setdefault(x["seed"], (x["rest_sh_deg"], x["rest_el_deg"]))
    return R


def main():
    F = policy()[0]
    sites = sorted({k[0] for k in STRATS})
    R = rests()
    e = s1.env()
    m = e.m; d = mujoco.MjData(m)
    u0 = None
    rows = {"shoulder": [], "elbow": []}
    for i, (seed, (rsh, rel)) in enumerate(sorted(R.items())):
        e._apply_body_variation(seed)
        mujoco.mj_resetData(m, d)
        _relaxed_qpos(m, d)
        d.qpos[m.jnt_qposadr[e._sj]] = math.radians(rsh)
        d.qpos[m.jnt_qposadr[e._ej]] = math.radians(rel)
        d.qpos[e.qadr] = np.concatenate([e.park - e.base, e.park_rot])
        mujoco.mj_forward(m, d)
        rid = e.rid
        Rm = np.array(d.geom_xmat[rid]).reshape(3, 3)
        ax = _unit(Rm[:, 2])
        if np.dot(ax, e.axis) < 0:
            ax = -ax
        u = dlo.lift_dir(e, F[1])
        rad = float(m.geom_size[rid][0])
        for off in sites:
            surf = np.array(d.geom_xpos[rid]) + (AIM_DISTAL + off) * ax - rad * u
            for nm, j in (("shoulder", e._sj), ("elbow", e._ej)):
                c = np.array(d.xanchor[j]); a = _unit(np.array(d.xaxis[j]))
                r = surf - c
                rows[nm].append((seed, off, float(np.linalg.norm(r)), abs(float(np.dot(np.cross(r, u), a)))))
        if i % 100 == 0:
            print(i, flush=True)
    out = [f"contact force equivalent to 0.5 N m; {len(R)} persons x {len(sites)} sites {sites}; lift direction of F (angle {F[1]})"]
    for nm, v in rows.items():
        dist = np.array([x[2] for x in v]); eff = np.array([x[3] for x in v])
        fd = 0.5 / dist; fe = 0.5 / np.maximum(eff, 1e-6)
        q = lambda a: f"min {a.min():.2f}  p5 {np.percentile(a, 5):.2f}  median {np.median(a):.2f}  p95 {np.percentile(a, 95):.2f}  max {a.max():.2f}"
        out.append(f"{nm}: full lever arm, N: {q(fd)}")
        out.append(f"{nm}: along the lift direction, N: {q(fe)}")
        for off in sites:
            sel = [x[3] for x in v if x[1] == off]
            out.append(f"   site {off:+.2f}: along-lift force median {np.median(0.5 / np.maximum(sel, 1e-6)):.2f} N, full-arm median {np.median([0.5 / x[2] for x in v if x[1] == off]):.2f} N")
    print("\n".join(out))
    open(os.path.join(TOP, "out", "posthoc_lever_range.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
