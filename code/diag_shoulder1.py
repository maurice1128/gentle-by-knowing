"""Shoulder route, step 1 (exploratory diagnostic, 2026-09-20).

The humerus is NOT welded: the arm hangs from the shoulder complex of myoArm.
The probe lifts the forearm from below until the forearm centre has risen
5 cm, then holds 0.5 s.  Logged every physics substep: the net intersegmental
force at the glenohumeral joint (cfrc_int of the humerus), split into
  distraction D  = tension along the humerus (positive pulls the head out of
                   the socket - the subluxation direction), and
  shear S        = the component perpendicular to the humerus,
both as change from the contact-free rest value, plus the wrist force.

Questions (gates written before looking at any result):
  Q1  stability: no divergence, site reachable, task completed in >= 80% of
      episodes.
  Q2  is the shoulder load visible at the wrist?  Within each individual,
      across strategies, Spearman(peak wrist force, peak GH shear) and
      (peak wrist force, peak GH distraction change).  Gate: median |rho| < 0.7
      for at least one of the two -> not readable from the wrist alone.
  Q3  does the least-wrist-force strategy protect the shoulder?  Ratio of its
      GH load to the individual's best GH load.  Gate: median >= 1.2.
  Q4  does the best-for-the-shoulder strategy differ between individuals?
Output: out/shoulder/step1.json
"""
import os
import sys
import json
import math
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "shoulder")
sys.path.insert(0, ROOT)

SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]   # training pool only
OFFSETS = (-0.06, 0.0, 0.06, 0.12)
ANGLES = (-40.0, 0.0, 40.0)
SPEEDS = (0.10, 0.25)
RISE_M = 0.05
HOLD_S = 0.5
T_MAX = 3.0
CLEAR = 0.03
_ENV = None


def env():
    global _ENV
    if _ENV is None:
        import mujoco
        from care_env6 import CareContactEnv6
        import crossover6 as c6
        e = CareContactEnv6(body_variation=c6.BODY_VARIATION, weld_humerus=False)
        e.site_mode = "lift"
        e._hb = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_BODY, "humerus")
        e._sj = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_elv")
        e._ej = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        e._fg = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
        e._gh_log = None
        orig = e._instantaneous_harm

        def hook():
            h = orig()
            if e._gh_log is not None:
                e._gh_log.append(gh_load(e))
            return h
        e._instantaneous_harm = hook
        _ENV = e
    return _ENV


def gh_load(e):
    import mujoco
    import numpy as np
    mujoco.mj_rnePostConstraint(e.m, e.d)
    f = np.array(e.d.cfrc_int[e._hb][3:])                 # force of the parent on the humerus subtree, world axes
    a = np.array(e.d.xanchor[e._ej]) - np.array(e.d.xanchor[e._sj])
    a = a / (np.linalg.norm(a) + 1e-12)                   # shoulder -> elbow
    along = float(np.dot(f, a))
    return -along, float(np.linalg.norm(f - along * a)), float(np.linalg.norm(f))


def _job(spec):
    import mujoco
    import numpy as np
    import diag_lift_oracle as dlo
    from care_env6 import AIM_DISTAL
    seed, off, ang, speed = spec
    e = env()
    e._gh_log = None
    try:
        obs = e.reset(seed=seed)
    except Exception as ex:
        return dict(seed=seed, offset=off, angle=ang, speed=speed, error=repr(ex)[:200])
    e.tissue_load()
    rest = gh_load(e)
    dvec = dlo.lift_dir(e, ang)
    try:
        pos, rot = e.approach_start(dvec, e.axis, clearance=CLEAR, aim_shift=AIM_DISTAL + off)
    except Exception as ex:
        return dict(seed=seed, offset=off, angle=ang, speed=speed, error=repr(ex)[:200])
    e._write_probe(pos, rot)
    e.cmd = np.concatenate([pos - e.base, rot])
    for _ in range(int(0.2 / e.dt)):
        act = e.reflex.step(e.d); e.d.ctrl[e.mids] = act; mujoco.mj_step(e.m, e.d)
    _, tot = e._force_by_partner()
    site_ok = tot <= 0.5
    z0 = float(e.d.geom_xpos[e._fg][2])
    q0 = {k: float(e.d.qpos[e.m.jnt_qposadr[j]]) for k, j in (("elv", e._sj), ("elbow", e._ej))}
    rest2 = gh_load(e)
    cmd = np.array(pos)
    e._gh_log = []
    peak_f = peak_r = 0.0
    shear_c = 0.0
    t_done = None; n_hold = 0
    for _ in range(int(T_MAX * 50)):
        if t_done is None:
            cmd = cmd + dvec * speed / 50
        obs, info = e.step(cmd, rot)
        peak_f = max(peak_f, info["step_peak"]["force"]); peak_r = max(peak_r, info["step_peak"]["reflex"])
        shear_c = max(shear_c, sum(c[3] for c in dlo.contacts_now(e)))
        if t_done is None and float(e.d.geom_xpos[e._fg][2]) - z0 >= RISE_M:
            t_done = e.t
        if t_done is not None:
            n_hold += 1
            if n_hold >= int(HOLD_S * 50):
                break
    g = np.array(e._gh_log)[50:]; e._gh_log = None        # drop the first 0.1 s (pre-contact start-up transient)
    ker = np.ones(25) / 25.0                               # 50 ms moving average: sustained load, not the 2 ms spike
    gs = np.column_stack([np.convolve(g[:, k], ker, mode="valid") for k in range(3)])
    rise = float(e.d.geom_xpos[e._fg][2]) - z0
    return dict(seed=seed, offset=off, angle=ang, speed=speed, site_ok=bool(site_ok), diverged=bool(e.diverged),
                reset_diverged=int(e.reset_diverged), rest_D=rest2[0], rest_S=rest2[1], rest_F=rest2[2],
                peak_D=float(gs[:, 0].max()), min_D=float(gs[:, 0].min()), peak_S=float(gs[:, 1].max()),
                spike_S=float(g[:, 1].max()), spike_D=float(g[:, 0].max()),
                end_D=float(g[-25:, 0].mean()), end_S=float(g[-25:, 1].mean()),
                peak_force=peak_f, peak_reflex=peak_r, contact_shear=shear_c, t_done=t_done,
                reached=float(1.0 if (t_done is not None and rise >= 0.8 * RISE_M) else min(0.94, max(0.0, rise / RISE_M))),
                end_rise=rise,
                d_elv_deg=math.degrees(float(e.d.qpos[e.m.jnt_qposadr[e._sj]]) - q0["elv"]),
                d_elbow_deg=math.degrees(float(e.d.qpos[e.m.jnt_qposadr[e._ej]]) - q0["elbow"]))


def spearman(x, y):
    import numpy as np
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and np.std(rx) > 0 and np.std(ry) > 0 else float("nan")


def analyse(rows):
    import numpy as np
    err = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    ok = [r for r in rows if r["site_ok"] and not r["diverged"] and r["reached"] >= 0.95]
    print(f"episodes {len(rows) + len(err)}: errors {len(err)}, diverged {sum(r['diverged'] for r in rows)}, site not ok {sum(not r['site_ok'] for r in rows)}, "
          f"completed {len(ok)} ({len(ok) / max(1, len(rows) + len(err)):.0%})  -> Q1 {'PASS' if len(ok) / max(1, len(rows) + len(err)) >= 0.8 else 'FAIL'}")
    for e_ in err[:3]:
        print("   error:", e_["error"])
    if not ok:
        return
    print(f"rest: GH distraction {np.median([r['rest_D'] for r in ok]):.1f} N, shear {np.median([r['rest_S'] for r in ok]):.1f} N; "
          f"lift changes shoulder elevation by {np.median([r['d_elv_deg'] for r in ok]):+.1f} deg, elbow by {np.median([r['d_elbow_deg'] for r in ok]):+.1f} deg (medians)")
    print(f"\n{'seed':>4} {'n':>3} | {'rho(F,S)':>8} {'rho(F,dD)':>9} | {'least-force: S / best S':>24} {'|dD| / best':>12} | best-for-shoulder strategy (by peak S)")
    rs_, rd_, q3s, q3d, best = [], [], [], [], []
    for s in sorted({r["seed"] for r in ok}):
        R = [r for r in ok if r["seed"] == s]
        if len(R) < 6:
            continue
        F = [r["peak_force"] for r in R]; S = [r["peak_S"] for r in R]
        dD = [max(abs(r["peak_D"] - r["rest_D"]), abs(r["min_D"] - r["rest_D"])) for r in R]
        rs, rd = spearman(F, S), spearman(F, dD)
        lf = int(np.argmin(F)); bs = int(np.argmin(S))
        rs_.append(rs); rd_.append(rd); q3s.append(S[lf] / max(min(S), 1e-9)); q3d.append(dD[lf] / max(min(dD), 1e-9))
        best.append((R[bs]["offset"], R[bs]["angle"], R[bs]["speed"]))
        print(f"{s:4d} {len(R):3d} | {rs:8.2f} {rd:9.2f} | {q3s[-1]:24.2f} {q3d[-1]:12.2f} | {best[-1]}  S {S[bs]:.1f} N (range {min(S):.1f}-{max(S):.1f}), F {F[bs]:.1f} N (range {min(F):.1f}-{max(F):.1f})")
    print(f"\nQ2 median rho(F, S) {np.nanmedian(rs_):.2f}, median rho(F, dD) {np.nanmedian(rd_):.2f} -> "
          f"{'PASS (not readable from the wrist)' if min(abs(np.nanmedian(rs_)), abs(np.nanmedian(rd_))) < 0.7 else 'FAIL (wrist force tracks the shoulder load)'}")
    print(f"Q3 least-wrist-force strategy vs best: shear ratio median {np.median(q3s):.2f}, distraction-change ratio median {np.median(q3d):.2f} -> "
          f"{'PASS' if max(np.median(q3s), np.median(q3d)) >= 1.2 else 'FAIL'}")
    from collections import Counter
    print("Q4 best-for-shoulder strategies:", Counter(best).most_common())


def main():
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "step1.json")
    if len(sys.argv) > 1 and sys.argv[1] == "--analyse":
        return analyse(json.load(open(path))["rows"])
    seeds = SEEDS[:1] if os.environ.get("WM_SMOKE") == "1" else SEEDS
    jobs = [(s, o, a, sp) for s in seeds for o in OFFSETS for a in ANGLES for sp in SPEEDS]
    n_proc = int(os.environ.get("WM_PROCS", 8))
    print(f"=== shoulder step 1: {len(jobs)} episodes, {n_proc} workers ===", flush=True)
    rows = []; t0 = time.time()
    with Pool(processes=n_proc) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if (k + 1) % 24 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(rows=rows), open(path if len(seeds) > 1 else path.replace(".json", "_smoke.json"), "w"))
    analyse(rows)


if __name__ == "__main__":
    main()
