"""Build the Phase-0 scene: myoArm (care recipient) + a rigid probe (robot end-effector).

Design notes
------------
* myoArm's collision geoms are contype=1, conaffinity=0, i.e. they collide with
  nothing by default.  The probe therefore MUST carry conaffinity=1 or no contact
  is ever generated.  `verify_contact()` asserts this rather than assuming it.
* The probe is a position-controlled slide joint along one world axis, so contact
  depth and approach speed are directly commandable and reproducible.
* Finger joints are not driven; they are left free but the probe targets the
  forearm (radius/ulna) or upper arm (humerus), whose collision geoms exist.

Everything writes only inside this folder (ROOT).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), "out")

# antagonist muscle groups spanning elbow_flexion, from probe_model.py
ELBOW_FLEXORS = ["BIClong", "BICshort", "BRA", "BRD"]
ELBOW_EXTENSORS = ["TRIlong", "TRIlat", "TRImed", "ANC"]
SHOULDER_ABD = ["DELT1", "DELT2", "DELT3", "SUPSP"]
SHOULDER_ADD = ["PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "TMAJ"]


def myoarm_path():
    import myosuite
    return os.path.join(os.path.dirname(myosuite.__file__),
                        "simhive", "myo_sim", "arm", "myoarm.xml")


SETTLE_S = 2.0  # arm drops noticeably under gravity; see settle_check.py


def build(probe_radius=0.03, probe_axis=(1.0, 0.0, 0.0), target_body="radius",
          probe_offset=(0.12, 0.0, 0.0), probe_stiffness=2000.0,
          probe_damping=50.0, settle_s=SETTLE_S, strict=True):
    """Return (MjModel, meta dict).

    The probe is a sphere on a slide joint whose zero position sits
    `probe_offset` away from `target_body`'s frame origin, so commanding the
    joint toward negative values drives it into the limb.
    """
    import mujoco
    import numpy as np

    spec = mujoco.MjSpec.from_file(myoarm_path())

    # Locate the target body AFTER gravity settling, not at qpos0.
    # Found 2026-09-04: myoArm's qpos0 is not an equilibrium.  The elbow swings
    # from 0 to ~0.70 rad over ~2 s, so a probe placed relative to the INITIAL
    # forearm position ends up elsewhere once the limb drops -- in the first
    # version of this file it ended up embedded in the torso, in permanent 48 N
    # contact with thorax_coll1, while a short-transient gate test still
    # reported clean forearm contact.
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetData(m0, d0)
    for _ in range(int(settle_s / m0.opt.timestep)):
        mujoco.mj_step(m0, d0)
    mujoco.mj_forward(m0, d0)
    bid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, target_body)
    if bid < 0:
        raise ValueError(f"no body named {target_body!r}")
    settled_pos = np.array(d0.xpos[bid])
    base = settled_pos + np.array(probe_offset)

    world = spec.worldbody
    pb = world.add_body(name="probe", pos=base.tolist())
    j = pb.add_joint()
    j.name = "probe_slide"
    j.type = mujoco.mjtJoint.mjJNT_SLIDE
    j.axis = list(probe_axis)
    j.range = [-0.30, 0.30]
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = probe_damping

    g = pb.add_geom()
    g.name = "probe_tip"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [probe_radius, 0, 0]
    g.rgba = [0.9, 0.2, 0.2, 1.0]
    g.mass = 0.5
    # THE critical line: myoArm geoms are conaffinity=0, so the probe must
    # supply the affinity bit for any contact to exist at all.
    g.contype = 1
    g.conaffinity = 1

    a = spec.add_actuator()
    a.name = "probe_act"
    a.target = "probe_slide"
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    gp = np.zeros(10); gp[0] = probe_stiffness
    bp = np.zeros(10); bp[1] = -probe_stiffness; bp[2] = -probe_damping
    a.gainprm = gp
    a.biasprm = bp
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.ctrlrange = [-0.30, 0.30]
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    s = spec.add_sensor()
    s.name = "probe_force"
    s.type = mujoco.mjtSensor.mjSENS_FORCE
    s.objtype = mujoco.mjtObj.mjOBJ_SITE
    # force sensor needs a site on the probe body
    site = pb.add_site()
    site.name = "probe_site"
    site.pos = [0, 0, 0]
    s.objname = "probe_site"

    m = spec.compile()
    meta = dict(target_body=target_body, probe_base=base.tolist(),
                settled_target_pos=settled_pos.tolist(),
                probe_axis=list(probe_axis), probe_radius=probe_radius,
                probe_stiffness=float(probe_stiffness), settle_s=float(settle_s))

    if strict:
        # Steady-state retracted state must be contact-free for the probe.
        # This is the check that the transient gate-A test failed to make.
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0.0
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        bad = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == pid or c.geom2 == pid:
                o = c.geom2 if c.geom1 == pid else c.geom1
                bad.append((str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o)),
                            round(float(c.dist), 6)))
        if bad:
            raise RuntimeError(
                f"probe already in contact at rest with {bad}; offset "
                f"{tuple(probe_offset)} puts it inside the body. Settled "
                f"{target_body} at {np.round(settled_pos, 4).tolist()}.")
        meta["retracted_ncon"] = 0
    return m, meta


def muscle_ids(m, names):
    import mujoco
    out = []
    for n in names:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if i < 0:
            raise ValueError(f"no actuator {n!r}")
        out.append(i)
    return out


def probe_contact_force(m, d, probe_geoms=None):
    """Total contact force magnitude on the probe geom, in newtons."""
    import mujoco
    import numpy as np
    if probe_geoms is None:
        probe_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")}
    tot = np.zeros(3)
    n = 0
    buf = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in probe_geoms or c.geom2 in probe_geoms:
            mujoco.mj_contactForce(m, d, i, buf)
            # contact frame -> world
            frame = np.array(c.frame).reshape(3, 3)
            tot += frame.T @ buf[:3]
            n += 1
    return float(np.linalg.norm(tot)), n


def contacting_geom_names(m, d):
    import mujoco
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
    names = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 == pid or c.geom2 == pid:
            other = c.geom2 if c.geom1 == pid else c.geom1
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other])
            names.add(f"{gn}@{bn}")
    return names


if __name__ == "__main__":
    import mujoco
    import numpy as np

    m, meta = build()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print("built scene:", meta)
    print(f"nq={m.nq} nu={m.nu}  (myoArm 63 muscles + 1 probe actuator)")
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
    print(f"probe geom id={pid} contype={m.geom_contype[pid]} "
          f"conaffinity={m.geom_conaffinity[pid]}")
    print("elbow flexors:", muscle_ids(m, ELBOW_FLEXORS))
    print("elbow extensors:", muscle_ids(m, ELBOW_EXTENSORS))
