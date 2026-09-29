"""Lift task, feasibility gate (2026-09-16).

Task: support the forearm from below and lift it, elbow flexion +15 deg, then
hold for 0.5 s.  Unlike the press task, the least-force contact site (distal,
long lever arm) is where the tissue is thin (distal taper, styloids, hand),
and the thick site (mid-forearm) needs more force.  If that trade-off is real
on these 19 individuals, a controller that only minimises force walks onto
thin tissue and a body model is needed to avoid it.

For each individual, an open-loop lift at 0.10 m/s from every candidate site
(8 axial offsets x 5 angles around the underside).  Recorded: true tissue
load P (peak over lift + hold), wrist force (peak and mean during the hold),
task, and the contacts at the instant of peak P so P can be re-evaluated under
the population map and under no map.

Gate (PREREGISTRATION.md section 10, written before this run):
  G1 least-force site != least-P site in >= 12 of 19 individuals
  G2 choosing the least-force site gives median P >= 1.3 x the oracle site
  G3a choosing by the population map beats choosing by force: median ratio
      lower by >= 0.2
  G3b population-map choice beats the best single fixed site (reported;
      the fixed site is picked in-sample, so this is optimistic for the rule)
Output: out/diag_lift/lift_oracle.json
"""
import os
import sys
import json
import math
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "diag_lift")
sys.path.insert(0, ROOT)

import crossover6 as c6
import tissue

SEEDS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21]
OFFSETS = (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
ANGLES = (-40.0, -20.0, 0.0, 20.0, 40.0)
SPEED = float(os.environ.get("WM_SPEED", 0.10))
TASK_DEG = 15.0
HOLD_S = 0.5
T_MAX = 3.0
CLEAR = 0.03

_ENV = None


def env():
    global _ENV
    if _ENV is None:
        from care_env6 import CareContactEnv6
        _ENV = CareContactEnv6(body_variation=c6.BODY_VARIATION)
    return _ENV


def lift_dir(e, ang_deg):
    """Unit direction perpendicular to the forearm, pointing up, rotated by
    ang_deg about the bone axis."""
    import numpy as np
    from care_env6 import _unit
    ax = _unit(e.axis)
    up = np.array([0.0, 0.0, 1.0])
    n = _unit(up - np.dot(up, ax) * ax)
    a = math.radians(ang_deg)
    return _unit(n * math.cos(a) + np.cross(ax, n) * math.sin(a))


def contacts_now(e):
    import mujoco
    import numpy as np
    out = []
    buf = np.zeros(6)
    d, m = e.d, e.m
    classes = e._tissue_classes
    for i in range(d.ncon):
        c = d.contact[i]
        if e.pgeom not in (c.geom1, c.geom2):
            continue
        o = c.geom2 if c.geom1 == e.pgeom else c.geom1
        cls = classes.get(int(m.geom_bodyid[o]))
        if cls is None:
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        fn = abs(float(buf[0])); ft = math.hypot(float(buf[1]), float(buf[2]))
        if cls == "forearm":
            u, phi = e.tissue_map.locate(d, c.pos)
        else:
            u, phi = None, None
        out.append([u, phi, fn, ft, cls])
    return out


def p_under(contacts, params):
    p = 0.0
    for u, phi, fn, ft, cls in contacts:
        if cls == "forearm":
            t = tissue.thickness(params, u, phi)
        elif cls == "hand":
            t = params.get("t_hand", tissue.T_HAND)
        else:
            t = params["t_mid"]
        p += (fn + ft) * tissue.T_REF / t
    return p


def _job(spec):
    import mujoco
    import numpy as np
    from care_env6 import AIM_DISTAL
    seed, off, ang = spec[:3]
    speed = float(spec[3]) if len(spec) > 3 else SPEED
    e = env()
    obs = e.reset(seed=seed)
    e.tissue_load()                                   # builds _tissue_classes
    dvec = lift_dir(e, ang)
    pos, rot = e.approach_start(dvec, e.axis, clearance=CLEAR, aim_shift=AIM_DISTAL + off)
    shift_used = float(e.aim_shift)
    e._write_probe(pos, rot)
    e.cmd = np.concatenate([pos - e.base, rot])
    for _ in range(int(0.2 / e.dt)):
        act = e.reflex.step(e.d); e.d.ctrl[e.mids] = act; mujoco.mj_step(e.m, e.d)
    _, tot = e._force_by_partner()
    site_ok = tot <= 0.5 and abs(shift_used - (AIM_DISTAL + off)) < 1e-6
    obs = e.observe()
    e0_obs = float(obs["other_kin"][0])
    e0 = float(e.d.qpos[e.elbow_adr])
    task = math.radians(TASK_DEG)
    cmd = np.array(pos)
    peak = dict(p=0.0, pb=0.0, sb=0.0, force=0.0, reflex=0.0)
    t_min = 1e9
    t_done = None
    hold_f = []
    peak_contacts = []
    best_end_p = 0.0
    n_hold = 0
    for _ in range(int(T_MAX * 50)):
        prog_obs = float(obs["other_kin"][0]) - e0_obs
        if t_done is None and prog_obs < task:
            cmd = cmd + dvec * speed / 50
        obs, info = e.step(cmd, rot)
        sp = info["step_peak"]
        for k in peak:
            peak[k] = max(peak[k], sp[k])
        if sp["t_min"] > 0:
            t_min = min(t_min, sp["t_min"])
        # contacts at the step end with the highest load.  The first version
        # compared the step-end load with the WITHIN-step peak, which is
        # usually higher, so contacts were often never captured (empty) or
        # captured at an early low-load instant (bug found 2026-09-16)
        L = e.tissue_load()
        if L["p"] > best_end_p:
            best_end_p = L["p"]
            peak_contacts = contacts_now(e)
        if t_done is None and float(e.d.qpos[e.elbow_adr]) - e0 >= task:
            t_done = e.t
        if t_done is not None:
            hold_f.append(float(sp["force"]))
            n_hold += 1
            if n_hold >= int(HOLD_S * 50):
                break
    lifted = float(e.d.qpos[e.elbow_adr]) - e0
    return dict(seed=seed, offset=off, angle=ang, speed=speed, site_ok=bool(site_ok), aim_shift=shift_used,
                peak_p=peak["p"], peak_pb=peak["pb"], peak_sb=peak["sb"], peak_force=peak["force"],
                peak_reflex=peak["reflex"], hold_force=(float(np.mean(hold_f)) if hold_f else None),
                t_min=(t_min if t_min < 1e8 else 0.0), t_done=t_done,
                reached=float(min(1.0, max(0.0, lifted / task))), lifted_deg=math.degrees(lifted),
                contacts=peak_contacts, step_end_p=best_end_p, diverged=bool(e.diverged),
                t_mid=float(e.tissue["t_mid"]), taper_u=float(e.tissue["taper_u"]))


def analyse(rows):
    import numpy as np
    nom = tissue.nominal_params()
    uni = tissue.uniform(nom)
    seeds = sorted({r["seed"] for r in rows})
    valid = lambda r: r["site_ok"] and r["reached"] >= 0.95 and not r["diverged"]
    print(f"\n{'seed':>4} {'t_mid':>5} {'taper':>5} | {'oracle (off,ang)':>17} {'P':>6} {'F':>5} | {'least-force':>17} {'P':>6} {'F':>5} | {'pop-map pick':>17} {'P':>6} | {'no-map pick':>17} {'P':>6}")
    ratio = dict(force=[], popmap=[], nomap=[], default=[])
    g1 = 0; n_ind = 0
    for s in seeds:
        rs = [r for r in rows if r["seed"] == s and valid(r)]
        if len(rs) < 3:
            print(f"{s:4d}: only {len(rs)} valid sites"); continue
        n_ind += 1
        orc = min(rs, key=lambda r: r["peak_p"])
        lf = min(rs, key=lambda r: r["peak_force"])
        pm = min(rs, key=lambda r: p_under(r["contacts"], nom))
        nm = min(rs, key=lambda r: p_under(r["contacts"], uni))
        dflt = [r for r in rs if r["offset"] == 0.0 and r["angle"] == 0.0]
        g1 += int((lf["offset"], lf["angle"]) != (orc["offset"], orc["angle"]))
        ratio["force"].append(lf["peak_p"] / orc["peak_p"])
        ratio["popmap"].append(pm["peak_p"] / orc["peak_p"])
        ratio["nomap"].append(nm["peak_p"] / orc["peak_p"])
        if dflt:
            ratio["default"].append(dflt[0]["peak_p"] / orc["peak_p"])
        f = lambda r: f"({r['offset']:+.2f},{r['angle']:+3.0f})"
        print(f"{s:4d} {rs[0]['t_mid']:5.1f} {rs[0]['taper_u']:+5.2f} | {f(orc):>17} {orc['peak_p']:6.1f} {orc['peak_force']:5.0f} | "
              f"{f(lf):>17} {lf['peak_p']:6.1f} {lf['peak_force']:5.0f} | {f(pm):>17} {pm['peak_p']:6.1f} | {f(nm):>17} {nm['peak_p']:6.1f}")
    med = {k: (float(np.median(v)) if v else float("nan")) for k, v in ratio.items()}
    print("\nP relative to this individual's best site (median [IQR]):")
    for k, v in ratio.items():
        if v:
            print(f"  {k:8s} {np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]  n={len(v)}")
    # fixed rules: one site for everybody (chosen in-sample)
    sites = sorted({(r["offset"], r["angle"]) for r in rows})
    fixed = []
    for st in sites:
        rr = []
        for s in seeds:
            rs = [r for r in rows if r["seed"] == s and valid(r)]
            if len(rs) < 3:
                continue
            orc = min(r["peak_p"] for r in rs)
            hit = [r for r in rs if (r["offset"], r["angle"]) == st]
            rr.append(hit[0]["peak_p"] / orc if hit else float("inf"))   # rule fails the task = inf
        fixed.append((float(np.median(rr)), sum(np.isinf(rr)), st))
    fixed.sort()
    print("\nbest fixed sites (same site for everyone; inf = task not done there):")
    for mdn, nf, st in fixed[:5]:
        print(f"  ({st[0]:+.2f},{st[1]:+3.0f}): median {mdn:.2f}x oracle, fails task for {nf} individuals")
    best_rule = fixed[0][0]
    print("\n=== gate ===")
    print(f"  G1 least-force site != least-P site: {g1}/{n_ind}  -> {'PASS' if g1 >= 12 else 'FAIL'} (need >= 12 of 19)")
    print(f"  G2 least-force choice median ratio {med['force']:.2f}  -> {'PASS' if med['force'] >= 1.3 else 'FAIL'} (need >= 1.3)")
    print(f"  G3a population map {med['popmap']:.2f} vs force {med['force']:.2f}: lower by {med['force'] - med['popmap']:.2f}  -> {'PASS' if med['force'] - med['popmap'] >= 0.2 else 'FAIL'} (need >= 0.2)")
    print(f"  G3b population map {med['popmap']:.2f} vs best fixed site {best_rule:.2f}  -> {'map better' if med['popmap'] < best_rule else 'rule as good or better'} (reported, in-sample rule)")
    # the trade-off itself: force and P across axial offset (angle 0), median over individuals
    print("\naxial profile at angle 0 (median over individuals): offset -> force, P, t_min")
    for off in OFFSETS:
        rs = [r for r in rows if r["offset"] == off and r["angle"] == 0.0 and valid(r)]
        if rs:
            print(f"  {off:+.2f}: F {np.median([r['peak_force'] for r in rs]):5.1f}  P {np.median([r['peak_p'] for r in rs]):6.1f}  t_min {np.median([r['t_min'] for r in rs]):5.1f}  valid {len(rs)}/19")


def main():
    from multiprocessing import Pool
    if len(sys.argv) > 1 and sys.argv[1] == "--analyse":
        analyse(json.load(open(sys.argv[2]))["rows"]); return 0
    os.makedirs(OUT, exist_ok=True)
    seeds = SEEDS[:2] if os.environ.get("WM_SMOKE") == "1" else SEEDS
    jobs = [(s, o, a) for s in seeds for o in OFFSETS for a in ANGLES]
    n_proc = int(os.environ.get("WM_PROCS", 8))
    print(f"=== lift oracle: {len(jobs)} episodes, {n_proc} workers ===", flush=True)
    t0 = time.time()
    rows = []
    with Pool(processes=n_proc) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if (k + 1) % 40 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    tag = os.environ.get("WM_TAG", "")
    name = f"lift_oracle_smoke{tag}.json" if os.environ.get("WM_SMOKE") == "1" else f"lift_oracle{tag}.json"
    path = os.path.join(OUT, name)
    json.dump(dict(rows=rows, offsets=OFFSETS, angles=ANGLES, speed=SPEED, task_deg=TASK_DEG, hold_s=HOLD_S), open(path, "w"))
    print("wrote", path)
    analyse(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
