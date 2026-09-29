"""Gate B-prime: does co-contraction become manner-sensitive on a SUPPORTED limb?

Why this test exists
--------------------
Gate B failed on a free-hanging limb, and the mechanism was traced rather than
guessed: the probe's push is absorbed by gross limb displacement instead of
muscle stretch.  At 139 N the shoulder swings 0.385 rad (22 deg) while the elbow
muscle-tendon length changes only ~1.7 mm -- LESS than the 4.8 mm seen at 20 N.
Force and muscle stretch are decoupled when the limb is free to swing away, so a
stretch reflex structurally cannot encode how the robot touched.  Sweeping the
reflex gain x5/x20/x60 never produced both a live reflex and manner sensitivity.

That failure is therefore conditional on the posture, not a property of the
body.  A real care recipient's arm rests on a bed or armrest; it cannot swing
away.  Blocking gross displacement should force the load into muscle stretch --
which is exactly the regime SpasticMyoElbow (Yu et al., arXiv:2412.04700)
operates in when a robot stretches a supported elbow at constant velocity.

Implementation: instead of a friction pad the arm slides off (the pad version
moved the forearm 94 mm and it landed on the pad edge), the upper arm is welded
to the world with a MuJoCo equality constraint.  That is the kinematic idealisation
of "the upper arm is resting on the bed": the humerus cannot translate, the elbow
is still free, so pushing the forearm has to flex the elbow and stretch the
elbow muscles.

Everything else is held fixed: same criteria, same seeds, same frozen reflex
gain (x2, torque increment 1.83 N*m, inside the healthy band).  The ONLY change
from the failing run is the boundary condition on the limb.
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), "out", "gate_b_prime")
sys.path.insert(0, ROOT)

GAIN_SCALE = 2.0
SPEEDS = (0.02, 0.05, 0.10, 0.20)
DEPTHS = (0.03, 0.05, 0.07, 0.09)
SEEDS = tuple(range(8))
JITTER = 0.01
B3_TARGET = 0.01
SEP_MIN = 3.0
RHO_MIN = 0.5


def build_supported(probe_radius=0.05, probe_stiffness=6000.0, settle_s=2.0,
                    weld_humerus=True, strict=True):
    """myoArm with the humerus welded to the world, probe above the forearm."""
    import mujoco
    import numpy as np
    import arm_scene
    from supported_scene import _relaxed_qpos

    spec = mujoco.MjSpec.from_file(arm_scene.myoarm_path())

    if weld_humerus:
        eq = spec.add_equality()
        eq.name = "humerus_weld"
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.name1 = "humerus"
        eq.name2 = "world"
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.active = True

    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetData(m0, d0)
    _relaxed_qpos(m0, d0)
    for _ in range(int(settle_s / m0.opt.timestep)):
        mujoco.mj_step(m0, d0)
    mujoco.mj_forward(m0, d0)

    gid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
    cen = np.array(d0.geom_xpos[gid])
    cap_r = float(m0.geom_size[gid][0])

    pb = spec.worldbody.add_body(
        name="probe", pos=[float(cen[0]), float(cen[1]),
                           float(cen[2] + cap_r + probe_radius + 0.04)])
    j = pb.add_joint()
    j.name = "probe_slide"
    j.type = mujoco.mjtJoint.mjJNT_SLIDE
    j.axis = [0.0, 0.0, -1.0]
    j.range = [-0.02, 0.20]
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = probe_stiffness / 60.0
    g = pb.add_geom()
    g.name = "probe_tip"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [probe_radius, 0, 0]
    g.rgba = [0.9, 0.2, 0.2, 1.0]
    g.mass = 0.5
    g.contype = 1
    g.conaffinity = 1
    site = pb.add_site()
    site.name = "probe_site"
    site.pos = [0, 0, 0]

    a = spec.add_actuator()
    a.name = "probe_act"
    a.target = "probe_slide"
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    gp = np.zeros(10); gp[0] = probe_stiffness
    bp = np.zeros(10); bp[1] = -probe_stiffness; bp[2] = -probe_stiffness / 60.0
    a.gainprm = gp
    a.biasprm = bp
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.ctrlrange = [-0.02, 0.20]
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    m = spec.compile()
    meta = dict(welded=bool(weld_humerus), forearm=cen.tolist(),
                probe_radius=probe_radius, probe_stiffness=probe_stiffness,
                settle_s=settle_s)
    if strict:
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        _relaxed_qpos(m, d)
        d.ctrl[:] = 0.0
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        bad = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                     c.geom2 if c.geom1 == pid else c.geom1))
               for c in (d.contact[i] for i in range(d.ncon))
               if pid in (c.geom1, c.geom2)]
        if bad:
            raise RuntimeError(f"probe touching at rest: {bad}")
        meta["retracted_ncon"] = 0
    return m, meta


def event(m, speed, depth, seed, contact=True, hold_s=0.8, settle_s=2.0):
    import mujoco
    import numpy as np
    from supported_scene import _relaxed_qpos, force_by_partner, torso_share
    from reflex import StretchReflex, cci
    from arm_scene import muscle_ids, ELBOW_FLEXORS, ELBOW_EXTENSORS

    fids = muscle_ids(m, ELBOW_FLEXORS)
    eids = muscle_ids(m, ELBOW_EXTENSORS)
    all_ids = fids + eids
    nf = len(fids)

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    _relaxed_qpos(m, d)
    rng = np.random.default_rng(seed)
    for jj in range(m.njnt):
        adr = m.jnt_qposadr[jj]
        if m.jnt_limited[jj]:
            lo, hi = m.jnt_range[jj]
            d.qpos[adr] = float(np.clip(d.qpos[adr] + rng.normal(0, JITTER),
                                        lo + 1e-4, hi - 1e-4))
    d.ctrl[:] = 0.0
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)

    rfx = StretchReflex(m, all_ids, gain_v=0.6 * GAIN_SCALE,
                        gain_l=4.0 * GAIN_SCALE)
    try:
        rfx.reset(d, keep_act=True)
    except TypeError:
        rfx.reset(d)

    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "probe_act")
    for _ in range(int(0.3 / m.opt.timestep)):
        act = rfx.step(d)
        d.ctrl[all_ids] = act
        d.ctrl[aid] = 0.0
        mujoco.mj_step(m, d)
    base = cci(rfx.act[:nf], rfx.act[nf:])["magnitude"]
    base_len = np.array(d.actuator_length[all_ids])

    n_ramp = max(1, int((depth / speed) / m.opt.timestep))
    n_hold = int(hold_s / m.opt.timestep)
    peak = base
    peak_f = 0.0
    peak_torso = 0.0
    peak_stretch = 0.0
    for i in range(n_ramp + n_hold):
        cmd = (depth * min(1.0, (i + 1) / n_ramp)) if contact else 0.0
        act = rfx.step(d)
        d.ctrl[all_ids] = act
        d.ctrl[aid] = float(cmd)
        mujoco.mj_step(m, d)
        peak = max(peak, cci(rfx.act[:nf], rfx.act[nf:])["magnitude"])
        parts, tot = force_by_partner(m, d)
        peak_f = max(peak_f, tot)
        peak_torso = max(peak_torso, torso_share(parts, tot))
        stretch = float(np.max(np.abs(
            np.array(d.actuator_length[all_ids]) - base_len)))
        peak_stretch = max(peak_stretch, stretch)
    return dict(rise=peak - base, peak_force=peak_f, torso_share=peak_torso,
                peak_stretch_m=peak_stretch)


def spearman(x, y):
    import numpy as np
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def evaluate(m, tag):
    import numpy as np
    from collections import defaultdict
    rows = []
    for sp in SPEEDS:
        for dp in DEPTHS:
            for s in SEEDS:
                r = event(m, sp, dp, s)
                r.update(speed=sp, depth=dp, seed=s)
                rows.append(r)
            sub = rows[-len(SEEDS):]
            print(f"   [{tag}] v={sp:.2f} d={dp:.2f}  "
                  f"rise={np.mean([x['rise'] for x in sub]):.5f}  "
                  f"F={np.mean([x['peak_force'] for x in sub]):6.1f} N  "
                  f"stretch={np.mean([x['peak_stretch_m'] for x in sub]) * 1000:5.2f} mm")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["speed"], r["depth"])].append(r["rise"])
    means = np.array([np.mean(v) for v in groups.values()])
    within = float(np.mean([np.std(v, ddof=1) for v in groups.values()]))
    sep = float(means.std(ddof=1) / within) if within > 1e-12 else float("inf")
    det = dict(tag=tag, separation=sep,
               rho_speed=spearman([r["speed"] for r in rows],
                                  [r["rise"] for r in rows]),
               rho_depth=spearman([r["depth"] for r in rows],
                                  [r["rise"] for r in rows]),
               max_rise=float(max(r["rise"] for r in rows)),
               mean_force=float(np.mean([r["peak_force"] for r in rows])),
               mean_stretch_mm=float(np.mean([r["peak_stretch_m"]
                                              for r in rows]) * 1000),
               max_torso=float(max(r["torso_share"] for r in rows)))
    det["B1"] = det["separation"] >= SEP_MIN
    det["B2"] = (abs(det["rho_speed"]) >= RHO_MIN
                 and abs(det["rho_depth"]) >= RHO_MIN)
    det["B3"] = det["max_rise"] >= B3_TARGET
    det["passed"] = bool(det["B1"] and det["B2"] and det["B3"])
    return rows, det


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Gate B-prime: supported (welded) vs free limb ===")
    print(f"reflex gain frozen at x{GAIN_SCALE}; only the boundary condition "
          f"differs\n")

    out = {}
    for welded in (True, False):
        tag = "welded" if welded else "free"
        try:
            m, meta = build_supported(weld_humerus=welded)
        except RuntimeError as e:
            print(f"[{tag}] build rejected: {e}")
            continue
        print(f"[{tag}] built, retracted_ncon={meta.get('retracted_ncon')}")
        rows, det = evaluate(m, tag)
        out[tag] = dict(rows=rows, detail=det)
        print(f"   -> separation={det['separation']:.2f} "
              f"rho_v={det['rho_speed']:+.3f} rho_d={det['rho_depth']:+.3f} "
              f"max_rise={det['max_rise']:.5f} "
              f"meanF={det['mean_force']:.1f} N "
              f"stretch={det['mean_stretch_mm']:.2f} mm "
              f"-> {'PASS' if det['passed'] else 'FAIL'}\n")

    print("--- comparison ---")
    for tag, v in out.items():
        d = v["detail"]
        print(f"  {tag:7s} B1={d['B1']!s:5s} B2={d['B2']!s:5s} B3={d['B3']!s:5s} "
              f"sep={d['separation']:5.2f} rise={d['max_rise']:.5f} "
              f"stretch={d['mean_stretch_mm']:.2f} mm")

    w = out.get("welded", {}).get("detail")
    f = out.get("free", {}).get("detail")
    if w and f:
        print("\nmechanism check: does welding move the load into muscle stretch?")
        print(f"   mean peak MTU stretch  free={f['mean_stretch_mm']:.2f} mm  "
              f"welded={w['mean_stretch_mm']:.2f} mm  "
              f"({w['mean_stretch_mm'] / max(f['mean_stretch_mm'], 1e-9):.1f}x)")
        if w["passed"] and not f["passed"]:
            print("\n=> GATE B-prime PASSES on the supported limb while failing "
                  "free. The correct claim is the narrower one: co-contraction "
                  "carries manner information only when the limb cannot swing "
                  "away, which is the care posture of interest.")
        elif not w["passed"]:
            print("\n=> Still FAILS with the limb supported. The displacement "
                  "escape route is not the explanation; fall back to L0/L1 per "
                  "the preregistration.")
    with open(os.path.join(OUT, "gate_b_prime.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
