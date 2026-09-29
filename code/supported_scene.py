"""Care-posture scene: forearm resting on a support surface, probe above it.

Why this replaces the free-hanging setup
----------------------------------------
The first scene put a probe next to a myoArm whose forearm hangs free.  Three
independent checks killed it on 2026-09-04:

* diagnose_contact.py - the probe sat wedged in the torso at rest (48 N with the
  command at zero) and travelled 2.4 mm across the whole depth sweep;
* calibrate_probe.py  - contact force tracked probe stiffness, not commanded
  depth, the signature of pressing on something that will not move;
* render_check.py     - at deep commands 75% of the force landed on the torso
  collision proxy (an invisible 95x140x140 mm ellipsoid), force was
  non-monotone (65 -> 49 -> 83 -> 126 N), and the probe was pressing on the
  elbow joint line rather than the forearm.

The physics underneath all three is the same and it is not a bug: a relaxed limb
hanging in free space swings away from a push, so it can only ever register
~10-20 N before the probe slides off it and into the trunk.  Reaching the
force band that matters for care contact requires the limb to be SUPPORTED,
which is also what the target scenario actually looks like: a recipient lying
down with the forearm resting on a bed or armrest.

So this scene adds:
  1. a static support pad under the settled forearm (the "bed"),
  2. relaxed finger flexion so the hand is not rigidly splayed and the ROM
     metric does not saturate on the fingers from step 0,
  3. a probe aimed at the forearm MID-SHAFT, above the support, pressing down,
  4. a partner-resolved force check, because a scalar |F| hides exactly the
     torso-contamination that scalar checks missed twice already.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SETTLE_S = 2.0
ARM_BODIES = {"humerus", "ulna", "radius", "lunate", "scaphoid", "pisiform",
              "triquetrum", "capitate", "trapezium", "trapezoid", "hamate",
              "firstmc", "secondmc", "thirdmc", "fourthmc", "fifthmc"}
FINGER_JOINT_PREFIXES = ("mcp", "pm", "md", "ip_", "cmc")
RELAXED_FINGER_FLEX = 0.45  # rad, a loosely curled hand


def _relaxed_qpos(m, d):
    """Set a relaxed hand instead of the rigidly splayed default."""
    import mujoco
    for j in range(m.njnt):
        name = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        if not name.startswith(FINGER_JOINT_PREFIXES):
            continue
        if "abduction" in name:
            continue
        adr = m.jnt_qposadr[j]
        lo, hi = m.jnt_range[j]
        d.qpos[adr] = min(max(RELAXED_FINGER_FLEX, lo + 1e-3), hi - 1e-3)


def build(support=True, probe_radius=0.035, probe_stiffness=400.0,
          probe_damping=10.0, support_margin=0.005, settle_s=SETTLE_S,
          strict=True, relaxed_hand=True):
    """Return (MjModel, meta).

    The support pad is placed under the forearm mid-shaft at its settled height
    minus `support_margin`, so the limb comes to rest ON the pad rather than
    hanging.  The probe is placed above the same point, pressing down (-z).
    """
    import mujoco
    import numpy as np

    spec = mujoco.MjSpec.from_file(
        __import__("arm_scene").myoarm_path())

    # --- pass 1: where does the forearm settle, with a relaxed hand? ---
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetData(m0, d0)
    if relaxed_hand:
        _relaxed_qpos(m0, d0)
    for _ in range(int(settle_s / m0.opt.timestep)):
        mujoco.mj_step(m0, d0)
    mujoco.mj_forward(m0, d0)

    # Use the actual COLLISION capsule, not body origins.  Body frames of `ulna`
    # and `lunate` do not sit on the bone shaft, and using them put the support
    # pad's top face (z=1.052) ABOVE the forearm centre (z=1.034), i.e. the pad
    # swallowed the limb and the probe hit the pad instead of the arm.
    gid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
    if gid < 0:
        raise RuntimeError("radius_coll geom not found")
    cen = np.array(d0.geom_xpos[gid])
    R = np.array(d0.geom_xmat[gid]).reshape(3, 3)
    cap_r = float(m0.geom_size[gid][0])
    cap_half = float(m0.geom_size[gid][1])
    elbow = cen + cap_half * R[:, 2]      # proximal end
    wrist = cen - cap_half * R[:, 2]      # distal end
    if elbow[2] < wrist[2]:
        elbow, wrist = wrist, elbow
    mid = cen                              # forearm mid-shaft, on the shaft
    axis_len = 2.0 * cap_half
    forearm_radius = cap_r
    forearm_bottom_z = float(cen[2] - cap_r)
    forearm_top_z = float(cen[2] + cap_r)

    world = spec.worldbody
    pad_half_h = 0.02
    # top face of the pad sits just under the underside of the forearm capsule
    pad_z = forearm_bottom_z - support_margin - pad_half_h
    if support:
        sb = world.add_body(name="support", pos=[float(mid[0]), float(mid[1]),
                                                 float(pad_z)])
        sg = sb.add_geom()
        sg.name = "support_pad"
        sg.type = mujoco.mjtGeom.mjGEOM_BOX
        # narrow across the limb so a probe aimed at the shaft cannot land on
        # the pad instead of the arm
        sg.size = [0.09, 0.035, pad_half_h]
        sg.rgba = [0.55, 0.55, 0.62, 1.0]
        sg.contype = 1
        sg.conaffinity = 1          # myoArm geoms are conaffinity=0
        sg.condim = 3

    # --- pass 2: with the pad in place, where does the forearm ACTUALLY rest? ---
    # Adding support changes the limb's equilibrium, so aiming the probe with the
    # unsupported pose puts it beside the arm and it lands on the pad instead.
    # (Observed 2026-09-04: 100% of contact force went to support_pad.)
    if support:
        m1 = spec.compile()
        d1 = mujoco.MjData(m1)
        mujoco.mj_resetData(m1, d1)
        if relaxed_hand:
            _relaxed_qpos(m1, d1)
        for _ in range(int(settle_s / m1.opt.timestep)):
            mujoco.mj_step(m1, d1)
        mujoco.mj_forward(m1, d1)
        gid1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
        cen1 = np.array(d1.geom_xpos[gid1])
        aim = cen1
        forearm_top_z = float(cen1[2] + cap_r)
    else:
        aim = mid

    # --- probe: above the forearm mid-shaft, pressing straight down ---
    # start just above the top of the limb so commanded depth ~= indentation
    start_h = 0.05
    probe_z0 = forearm_top_z + probe_radius + start_h
    pb = world.add_body(name="probe",
                        pos=[float(aim[0]), float(aim[1]), float(probe_z0)])
    j = pb.add_joint()
    j.name = "probe_slide"
    j.type = mujoco.mjtJoint.mjJNT_SLIDE
    j.axis = [0.0, 0.0, -1.0]        # +ctrl drives downward into the limb
    j.range = [-0.02, 0.20]
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = probe_damping

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
    bp = np.zeros(10); bp[1] = -probe_stiffness; bp[2] = -probe_damping
    a.gainprm = gp
    a.biasprm = bp
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.ctrlrange = [-0.02, 0.20]
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    m = spec.compile()
    meta = dict(forearm_elbow=elbow.tolist(), forearm_wrist=wrist.tolist(),
                forearm_mid=mid.tolist(), forearm_aim=list(map(float, aim)),
                forearm_len=axis_len,
                forearm_radius=forearm_radius,
                forearm_bottom_z=forearm_bottom_z, forearm_top_z=forearm_top_z,
                pad_z=float(pad_z), pad_top_z=float(pad_z + pad_half_h),
                probe_z0=float(probe_z0),
                probe_start_height=start_h, support=bool(support),
                probe_radius=probe_radius, probe_stiffness=probe_stiffness,
                settle_s=float(settle_s), relaxed_hand=bool(relaxed_hand))

    if strict:
        d = mujoco.MjData(m)
        reset(m, d, relaxed_hand=relaxed_hand, settle_s=settle_s)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        bad = [(str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                      c.geom2 if c.geom1 == pid else c.geom1)),
                round(float(c.dist), 6))
               for c in (d.contact[i] for i in range(d.ncon))
               if pid in (c.geom1, c.geom2)]
        if bad:
            raise RuntimeError(f"probe touching at rest: {bad}")
        meta["retracted_ncon"] = 0
        meta["arm_on_support"] = arm_support_contacts(m, d)
    return m, meta


def reset(m, d, relaxed_hand=True, settle_s=SETTLE_S, seed=None, jitter=0.0):
    """Reset, optionally jitter the posture, then settle. Returns d."""
    import mujoco
    import numpy as np
    mujoco.mj_resetData(m, d)
    if relaxed_hand:
        _relaxed_qpos(m, d)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        for j in range(m.njnt):
            adr = m.jnt_qposadr[j]
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                d.qpos[adr] = float(np.clip(d.qpos[adr] + rng.normal(0, jitter),
                                            lo + 1e-4, hi - 1e-4))
            else:
                d.qpos[adr] += rng.normal(0, jitter)
    d.ctrl[:] = 0.0
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def arm_support_contacts(m, d):
    """How many arm<->support contacts exist (is the limb actually resting?)."""
    import mujoco
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "support_pad")
    if sid < 0:
        return 0
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if sid in (c.geom1, c.geom2):
            n += 1
    return n


def force_by_partner(m, d):
    """Contact force on the probe, split by partner body.

    Scalar |F| hid torso contamination twice in this project, so every gate from
    here on uses this instead.  Returns (dict partner->newtons, total).
    """
    import mujoco
    import numpy as np
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
    out = {}
    buf = np.zeros(6)
    total = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if pid not in (c.geom1, c.geom2):
            continue
        o = c.geom2 if c.geom1 == pid else c.geom1
        bn = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[o]))
        gn = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o))
        tag = bn if bn not in ("None", "world") else gn
        mujoco.mj_contactForce(m, d, i, buf)
        f = float(abs(buf[0]))   # normal component in contact frame
        out[tag] = out.get(tag, 0.0) + f
        total += f
    return out, total


def torso_share(parts, total):
    if total <= 1e-9:
        return 0.0
    bad = sum(v for k, v in parts.items() if k not in ARM_BODIES)
    return bad / total


if __name__ == "__main__":
    import mujoco
    import numpy as np

    print("=== supported care-posture scene ===")
    for support in (True, False):
        try:
            m, meta = build(support=support)
        except RuntimeError as e:
            print(f"support={support}: BUILD REJECTED: {e}")
            continue
        d = mujoco.MjData(m)
        reset(m, d)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "probe_act")
        print(f"\nsupport={support}  forearm_mid={np.round(meta['forearm_mid'],3).tolist()}  "
              f"arm-on-pad contacts at rest={meta.get('arm_on_support')}")
        print(f"{'ctrl':>6} {'F_total':>9} {'torso%':>7}  partners")
        for cmd in np.arange(0.0, 0.145, 0.02):
            reset(m, d)
            for _ in range(int(1.2 / m.opt.timestep)):
                d.ctrl[aid] = float(cmd)
                mujoco.mj_step(m, d)
            parts, tot = force_by_partner(m, d)
            ts = torso_share(parts, tot)
            desc = ", ".join(f"{k}={v:.1f}" for k, v in sorted(parts.items()))
            print(f"{cmd:6.3f} {tot:9.2f} {ts * 100:6.1f}%  {desc}")
