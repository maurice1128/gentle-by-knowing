"""Six-DoF contact environment: the action space where a wrong model can be
wrong in a HARMFUL direction.

Why this exists
---------------
The 1-DoF slide experiment produced half the centrepiece.  Predictive gentleness
beat the constraint baseline at every fidelity, and degrading the model never
made the predictive arm more harmful - it made it withdraw.  That asymmetry is
an artefact of the action space: on a single axis a controller can only go
forward or back, so a mistaken model can only be too cautious or too eager along
one line, and the harm budget binds before the error can express itself.  The
Tian et al. (2022) branch of the crossover, where overconfidence causes injury,
has no room to appear.

It also fixes a baseline problem the preregistration flagged: on one axis a
control barrier function degenerates toward a threshold brake, which is the
strawman this project forbids.  With six degrees of freedom the barrier has to
choose among directions, which is what a CBF actually does.

Design
------
The probe is an idealised end-effector: a chain of three slide joints (world x,
y, z) and three hinge joints (x, y, z), each under a stiff position servo.  It
is idealised deliberately: a Menagerie manipulator would add inverse kinematics,
joint limits and self-collision, none of which bear on the question, and all of
which would confound "the model was wrong" with "the arm could not get there".
Swapping in a real manipulator is the next step and is stated as a limitation,
not hidden.

The translational servo stiffness is the SAME 6000 N/m the 1-DoF probe used,
so a commanded displacement means the same thing in both experiments and the
constraint baseline's equilibrium bound carries over unchanged.

Why not a mocap weld (the first version): MjSpec's default equality `data` is
the polynomial-coefficient default [0, 1, 0, ...], which a weld reads as an
anchor one metre along the body's y-axis.  The "servo" was a soft pendulum
swinging around a point a metre away, the probe fell 0.56 m at reset and never
touched the limb.  Found 2026-09-09, the first time this file was executed.

The probe tip is a CAPSULE rather than a sphere, so orientation matters: the
same penetration can be delivered along the bone or across it, and those are not
equally gentle.  The capsule is symmetric about its own axis, so the third
hinge (spin about the capsule axis) cannot change the outcome; it is kept so the
commanded quantity is a full end-effector pose, but only five degrees of
freedom are outcome-relevant and the paper must say so.

Everything else is unchanged from care_env: same reflex, same L2 (activation
rise of the lengthened muscles), same observation boundary with the same audit,
same divergence guard.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

REFLEX_GAIN_SCALE = 2.0
LENGTHEN_EPS = 1e-4
SETTLE_S = 2.0
BASELINE_S = 0.3
CLEARANCE = 0.05          # m, free gap used by the direction self-test
HOME_CLEARANCE = 0.04     # m, free gap at home (= the 1-DoF probe's 0.04)
AIM_DISTAL = 0.03         # m, contact point distal of the forearm mid-shaft

ARM_BODIES = {"humerus", "ulna", "radius", "lunate", "scaphoid", "pisiform",
              "triquetrum", "capitate", "trapezium", "trapezoid", "hamate",
              "firstmc", "secondmc", "thirdmc", "fourthmc", "fifthmc"}

SLIDES = (("probe_x", (1.0, 0.0, 0.0)), ("probe_y", (0.0, 1.0, 0.0)),
          ("probe_z", (0.0, 0.0, 1.0)))
HINGES = (("probe_rx", (1.0, 0.0, 0.0)), ("probe_ry", (0.0, 1.0, 0.0)),
          ("probe_rz", (0.0, 0.0, 1.0)))


# ---------------------------------------------------------------- geometry
def rot_to_axis(rot):
    """World direction of the capsule axis for hinge angles (rx, ry, rz).

    Chain is Rx(a) Ry(b) Rz(c) applied to the local z-axis; Rz drops out.
    """
    import numpy as np
    a, b = float(rot[0]), float(rot[1])
    return np.array([np.sin(b), -np.sin(a) * np.cos(b), np.cos(a) * np.cos(b)])


def axis_to_rot(u):
    """Hinge angles (rx, ry, 0) that point the capsule axis along unit `u`."""
    import numpy as np
    u = np.asarray(u, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)
    b = float(np.arcsin(np.clip(u[0], -1.0, 1.0)))
    a = float(np.arctan2(-u[1], u[2]))
    return np.array([a, b, 0.0])


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def probe_force(m, d, pgeom, want_vec=False):
    """Contact force on the probe, split by partner body.

    Returns (dict partner -> N, total N) and, with want_vec, the world-frame
    sum of the normal forces acting ON the probe (what a wrist F/T sensor
    reports).  Shared by the real environment and the internal copies so a
    prediction is computed exactly like the measurement it predicts.
    """
    import mujoco
    import numpy as np
    out = {}
    buf = np.zeros(6)
    tot = 0.0
    fvec = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if pgeom not in (c.geom1, c.geom2):
            continue
        o = c.geom2 if c.geom1 == pgeom else c.geom1
        bn = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[o]))
        gn = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o))
        tag = bn if bn not in ("None", "world") else gn
        mujoco.mj_contactForce(m, d, i, buf)
        f = float(abs(buf[0]))
        out[tag] = out.get(tag, 0.0) + f
        tot += f
        if want_vec:
            frame = np.array(c.frame).reshape(3, 3)
            n = frame[0]                      # contact normal, geom1 -> geom2
            sign = 1.0 if c.geom1 == pgeom else -1.0
            fvec += -sign * n * buf[0]        # force ON the probe
    if want_vec:
        return out, tot, fvec
    return out, tot


# ---------------------------------------------------------------- scene
_BUILD_CACHE = {}


def build6(probe_radius=0.025, probe_half_len=0.05, kp=6000.0, kr=20.0,
           settle_s=SETTLE_S, weld_humerus=True, strict=True):
    """myoArm with a 6-DoF capsule probe hovering above the forearm.

    Returns (MjModel, meta).  `meta` carries the forearm geometry the
    controllers are allowed to know (it is the robot's own workspace model):
    centre, long axis (elbow -> wrist), radius, elbow hinge axis.

    Compiled once per process and cached; callers get a deep copy they may
    mutate.  Each MjSpec compile of myoArm (46 meshes) leaves ~0.5 GB of
    committed heap behind, and a worker used to compile twice per internal
    copy per episode, reaching 2.2 GB commit and running the machine out of
    memory when other jobs share it (2026-09-10, pitfall 22).
    """
    import copy
    key = (float(probe_radius), float(probe_half_len), float(kp), float(kr),
           float(settle_s), bool(weld_humerus), bool(strict))
    if key not in _BUILD_CACHE:
        _BUILD_CACHE[key] = _build6_uncached(*key)
    m, meta = _BUILD_CACHE[key]
    return copy.deepcopy(m), dict(meta)


def _build6_uncached(probe_radius, probe_half_len, kp, kr, settle_s,
                     weld_humerus, strict):
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
    R = np.array(d0.geom_xmat[gid]).reshape(3, 3)
    cap_r = float(m0.geom_size[gid][0])
    cap_half = float(m0.geom_size[gid][1])
    axis = _unit(R[:, 2])
    ej = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
    elbow_anchor = np.array(d0.xanchor[ej])
    hinge = _unit(d0.xaxis[ej])
    # orient the long axis from elbow toward wrist
    if np.dot(cen - elbow_anchor, axis) < 0:
        axis = -axis
    # in-plane normal: perpendicular to the bone inside the flexion plane
    flex_normal = _unit(np.cross(hinge, axis))

    start = cen + np.array([0.0, 0.0,
                            cap_r + probe_half_len + probe_radius + CLEARANCE])

    # --- probe: slide x, y, z then hinge x, y, z, capsule on the last body
    parent = spec.worldbody.add_body(name="probe_base", pos=start.tolist())
    prev = parent
    for i, (nm, ax) in enumerate(SLIDES + HINGES):
        b = prev.add_body(name=nm + "_body", pos=[0, 0, 0])
        j = b.add_joint()
        j.name = nm
        j.type = (mujoco.mjtJoint.mjJNT_SLIDE if i < 3
                  else mujoco.mjtJoint.mjJNT_HINGE)
        j.axis = list(ax)
        if i < 3:
            j.range = [-0.30, 0.30]
            j.damping = kp / 60.0
        else:
            j.range = [-3.1, 3.1]
            # the capsule's inertia about its own axis is tiny (1.6e-4), so
            # the spin hinge needs more damping than the slides for the
            # explicit servo spring to stay stable at dt = 2 ms
            j.damping = kr / 10.0
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        # intermediate links: massless is illegal, so a tiny non-colliding bead
        if i < 5:
            g = b.add_geom()
            g.name = nm + "_link"
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [0.005, 0, 0]
            g.mass = 0.01
            g.contype = 0
            g.conaffinity = 0
            g.group = 3
        prev = b
    tip = prev
    g = tip.add_geom()
    g.name = "probe_tip"
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size = [probe_radius, probe_half_len, 0]
    g.rgba = [0.9, 0.2, 0.2, 1.0]
    g.mass = 0.5
    g.contype = 1
    g.conaffinity = 1
    site = tip.add_site()
    site.name = "probe_site"
    site.pos = [0, 0, 0]

    for i, (nm, _) in enumerate(SLIDES + HINGES):
        a = spec.add_actuator()
        a.name = nm + "_act"
        a.target = nm
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        k = kp if i < 3 else kr
        gp = np.zeros(10); gp[0] = k
        bp = np.zeros(10); bp[1] = -k; bp[2] = -(k / 60.0 if i < 3 else k / 10.0)
        a.gainprm = gp
        a.biasprm = bp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.ctrlrange = [-0.30, 0.30] if i < 3 else [-3.1, 3.1]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    m = spec.compile()
    meta = dict(forearm_centre=cen.tolist(), forearm_axis=axis.tolist(),
                forearm_radius=cap_r, forearm_half_len=cap_half,
                elbow_anchor=elbow_anchor.tolist(), elbow_hinge=hinge.tolist(),
                flex_normal=flex_normal.tolist(),
                probe_start=start.tolist(), probe_radius=probe_radius,
                probe_half_len=probe_half_len, kp=float(kp), kr=float(kr),
                welded=bool(weld_humerus), settle_s=float(settle_s))

    if strict:
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        _relaxed_qpos(m, d)
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        bad = [str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                     c.geom2 if c.geom1 == pid else c.geom1))
               for c in (d.contact[i] for i in range(d.ncon))
               if pid in (c.geom1, c.geom2)]
        if bad:
            raise RuntimeError(f"probe touching at rest: {bad}")
        pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe_rz_body")
        sag = float(np.linalg.norm(np.array(d.xpos[pb]) - start))
        if sag > 0.005:
            raise RuntimeError(f"probe servo sags {sag * 1000:.1f} mm at rest")
        meta["retracted_ncon"] = 0
        meta["rest_sag_mm"] = sag * 1000
    return m, meta


class CareContactEnv6:
    """6-DoF version of CareContactEnv. Same boundary, same audit, same L2.

    Commands are an absolute end-effector pose: capsule-centre position in
    world metres and hinge angles (rx, ry, rz) in radians.  `axis_to_rot`
    converts a desired capsule direction into hinge angles.
    """

    def __init__(self, probe_radius=0.025, probe_half_len=0.05,
                 kin_noise=0.01, force_noise=0.5, settle_s=SETTLE_S,
                 control_hz=50.0, body_variation=0.0, weld_humerus=True,
                 kp=6000.0, kr=20.0):
        import mujoco
        import numpy as np
        from arm_scene import muscle_ids, ELBOW_FLEXORS, ELBOW_EXTENSORS

        self.m, self.meta = build6(probe_radius=probe_radius,
                                   probe_half_len=probe_half_len,
                                   kp=kp, kr=kr, settle_s=settle_s,
                                   weld_humerus=weld_humerus, strict=True)
        self.d = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)
        self.substeps = max(1, int(round((1.0 / control_hz) / self.dt)))
        self.kin_noise = float(kin_noise)
        self.force_noise = float(force_noise)
        self.settle_s = float(settle_s)
        self.body_variation = float(body_variation)
        self._pristine = None

        self.fids = muscle_ids(self.m, ELBOW_FLEXORS)
        self.eids = muscle_ids(self.m, ELBOW_EXTENSORS)
        self.mids = self.fids + self.eids
        self.nf = len(ELBOW_FLEXORS)
        self.pbody = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                       "probe_rz_body")
        self.pgeom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                       "probe_tip")
        names = [nm for nm, _ in SLIDES + HINGES]
        self.jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in names]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.jids])
        self.aids = np.array([mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act") for n in names])
        self.obs_joints = []
        for nm in ("elbow_flexion", "pro_sup", "shoulder_elv"):
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if j >= 0:
                self.obs_joints.append((nm, self.m.jnt_qposadr[j],
                                        self.m.jnt_dofadr[j]))
        self._harm_reads = 0
        self._rng = np.random.default_rng(0)
        self.base = np.array(self.meta["probe_start"])   # joint-chain origin
        self.forearm = np.array(self.meta["forearm_centre"])
        self.axis = np.array(self.meta["forearm_axis"])
        self.hinge = np.array(self.meta["elbow_hinge"])
        self.flex_normal = np.array(self.meta["flex_normal"])
        self.cap_r = float(self.meta["forearm_radius"])
        self.probe_r = float(probe_radius)
        self.probe_half = float(probe_half_len)
        self.down = np.array([0.0, 0.0, -1.0])
        self.press_dir = np.array(self.down)      # set_site() may rotate it
        ej = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        self.elbow_adr = int(self.m.jnt_qposadr[ej])
        # soft-tissue map (tissue.py): the recipient-side quantity a wrist
        # sensor cannot see.  Built per individual at reset.
        self.rid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
        self.hid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "humerus_coll")
        self.ulna_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ulna")
        self.tissue = None
        self.tissue_map = None
        self.site = dict(offset=0.0, phi=0.0)
        # scratch data for geometric queries (never stepped)
        self._scratch = mujoco.MjData(self.m)
        mujoco.mj_resetData(self.m, self._scratch)
        # HOME: capsule along the bone, vertically above the mid-shaft, with
        # the same 0.04 m free travel the 1-DoF probe had, so the nominal
        # command depth means the same indentation in both experiments.
        self.home_rot = axis_to_rot(self.axis)
        self.park_rot = axis_to_rot(self.axis)
        self.park = self.base + np.array([0.0, 0.0, 0.10])   # well clear
        from supported_scene import _relaxed_qpos as _rq
        mujoco.mj_resetData(self.m, self.d)
        _rq(self.m, self.d)
        self._write_probe(self.park, self.park_rot)
        for _ in range(int(self.settle_s / self.dt)):
            mujoco.mj_step(self.m, self.d)
        self.home, _ = self.approach_start(self.down, self.axis,
                                           clearance=HOME_CLEARANCE)
        self._elbow0 = 0.0

    # ---- population (identical perturbation to the 1-DoF env) ----
    def _apply_body_variation(self, seed):
        from care_env import CareContactEnv
        CareContactEnv._apply_body_variation(self, seed)

    # ---- pose helpers ----
    def approach_start(self, direction, capsule_axis=None, clearance=CLEARANCE,
                       aim_shift=None):
        """Pose from which the capsule, moving along `direction`, first touches
        the forearm mid-shaft after `clearance` metres of free travel.

        The gap is measured with the actual collision geometry (capsule to
        capsule), not from centre-to-centre distances, so an oblique approach
        or a tilted capsule gets the same free travel as a vertical one.
        `capsule_axis` defaults to the bone axis (capsule lies along the limb).
        Returns (pos, rot).
        """
        import mujoco
        import numpy as np
        dvec = _unit(direction)
        ca = self.axis if capsule_axis is None else _unit(capsule_axis)
        rot = axis_to_rot(ca)
        sd = self._scratch
        sd.qpos[:] = self.d.qpos          # the limb as it is NOW (per seed)
        mujoco.mj_kinematics(self.m, sd)
        rid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
        hid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "humerus_coll")
        # Aim distal of the mid-shaft: a capsule lying along the bone reaches
        # 0.075 m toward the elbow from its centre, and on bodies that settle
        # more flexed its proximal end touched the welded humerus (seeds 0,
        # 2, 7 of 8, 2026-09-09).  The clearance is bisected against the
        # forearm only (the humerus lies beside the descent path, so folding
        # it into the bisection sends home 26 cm up); the humerus is then
        # checked at the result and the aim moved further distal if needed.
        R = np.array(sd.geom_xmat[rid]).reshape(3, 3)
        ax = _unit(R[:, 2])
        if np.dot(ax, self.axis) < 0:
            ax = -ax
        fromto = np.zeros(6)

        def dist(pos, g):
            q = np.concatenate([pos - self.base, rot])
            sd.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, sd)
            return mujoco.mj_geomDistance(self.m, sd, self.pgeom, g, 1.0, fromto)

        shift = AIM_DISTAL if aim_shift is None else float(aim_shift)
        for _ in range(4):
            cen = np.array(sd.geom_xpos[rid]) + shift * ax
            lo, hi = 0.0, 0.6
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if dist(cen - dvec * mid, rid) < clearance:
                    lo = mid
                else:
                    hi = mid
            pos = cen - dvec * hi
            if hid < 0 or dist(pos, hid) >= 0.01:
                break
            shift += 0.02
        self.aim_shift = shift
        return pos, rot

    def site_direction(self, phi_deg):
        """Approach direction for a contact site `phi_deg` around the forearm
        axis from straight-above (positive toward the dorsal side)."""
        import numpy as np
        ax = _unit(self.axis)
        if getattr(self, "site_mode", "press") == "lift":
            # lift task (round 8): perpendicular to the bone, pointing up, so
            # the probe approaches from BELOW; same convention as
            # diag_lift_oracle.lift_dir
            up = np.array([0.0, 0.0, 1.0])
            d = _unit(up - np.dot(up, ax) * ax)
        else:
            d = np.array(self.down, dtype=float)
        a = np.radians(float(phi_deg))
        return _unit(d * np.cos(a) + np.cross(ax, d) * np.sin(a)
                     + ax * np.dot(ax, d) * (1 - np.cos(a)))

    def set_site(self, offset_m=0.0, phi_deg=0.0):
        """Move the parked probe (in the air, before any contact) to a
        different contact site: `offset_m` along the bone from the default
        aim (positive distal) and `phi_deg` around it.  This is the "where
        to touch" decision; it needs no contact, so the reflex baseline
        stands.  Returns False (and leaves the probe at home) if the new
        pose already touches the recipient."""
        import mujoco
        import numpy as np
        d = self.site_direction(phi_deg)
        pos, rot = self.approach_start(d, self.axis, clearance=HOME_CLEARANCE,
                                       aim_shift=AIM_DISTAL + float(offset_m))
        old = (np.array(self.home), np.array(self.cmd), np.array(self.press_dir),
               dict(self.site), np.array(self.d.qpos), np.array(self.d.qvel))
        self._write_probe(pos, rot)
        self.cmd = np.concatenate([pos - self.base, rot])
        for _ in range(int(0.1 / self.dt)):
            act = self.reflex.step(self.d)
            self.d.ctrl[self.mids] = act
            mujoco.mj_step(self.m, self.d)
        parts, tot = self._force_by_partner()
        if tot > 0.5:
            self.home, self.cmd, self.press_dir, self.site = old[:4]
            self.d.qpos[:], self.d.qvel[:] = old[4], old[5]
            mujoco.mj_forward(self.m, self.d)
            self._write_probe(self.home, self.home_rot)
            return False
        self.home = np.array(pos)
        self.press_dir = d
        self.site = dict(offset=float(offset_m), phi=float(phi_deg))
        return True

    def tissue_load(self):
        """Evaluation-side bone-proximity load of the current contacts."""
        import tissue
        if not hasattr(self, "_tissue_classes"):
            self._tissue_classes = tissue.body_classes(self.m, arm_bodies=ARM_BODIES)
        return tissue.load_index(self.m, self.d, self.pgeom, self.tissue_map,
                                 self.tissue, self._tissue_classes)

    def _write_probe(self, pos, rot):
        """Set probe joint positions AND servo targets to a pose."""
        import numpy as np
        q = np.concatenate([np.asarray(pos, dtype=float) - self.base,
                            np.asarray(rot, dtype=float)])
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.dadr] = 0.0
        self.d.ctrl[self.aids] = q

    # ---- lifecycle ----
    def reset(self, seed=0, jitter=0.01, start_pos=None, start_rot=None):
        """Reset the recipient and place the probe at `start_pos`/`start_rot`
        (default: home, vertical above the forearm).  The reflex baseline is
        taken with the probe parked, so a start pose that already touches the
        limb is flagged in `self.start_contact` and must not be used."""
        import mujoco
        import numpy as np
        from supported_scene import _relaxed_qpos
        from reflex import StretchReflex

        if self.body_variation > 0:
            self._apply_body_variation(seed)
        mujoco.mj_resetData(self.m, self.d)
        _relaxed_qpos(self.m, self.d)
        rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(seed + 10_000)
        probe_j = set(self.jids)
        for j in range(self.m.njnt):
            if j in probe_j:
                continue
            adr = self.m.jnt_qposadr[j]
            if self.m.jnt_limited[j]:
                lo, hi = self.m.jnt_range[j]
                self.d.qpos[adr] = float(np.clip(
                    self.d.qpos[adr] + rng.normal(0, jitter),
                    lo + 1e-4, hi - 1e-4))
        # Settle the limb with the probe parked well clear.  A fixed home
        # computed on the nominal body touched the limb of seed 0 at rest
        # (51 N under no-op, 2026-09-09): individuals settle differently, so
        # the parking pose has to be chosen AFTER this body has settled, the
        # way a robot would place itself above the limb it can see.
        w_reset0 = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        self._write_probe(self.park, self.park_rot)
        for _ in range(int(self.settle_s / self.dt)):
            mujoco.mj_step(self.m, self.d)
        if start_pos is None:
            self.home, _ = self.approach_start(self.down, self.axis,
                                               clearance=HOME_CLEARANCE)
        self.press_dir = np.array(self.down)
        self.site = dict(offset=0.0, phi=0.0)
        # this individual's tissue map and the contact->(u, phi) mapper,
        # anchored to the settled limb
        import tissue
        self.tissue = tissue.individual_params(seed, **tissue.spread_from_env())
        self.tissue_map = tissue.ContactMapper(self.m, self.d, self.rid, self.axis,
                                               self.flex_normal, self.ulna_body)
        pos = self.home if start_pos is None else np.asarray(start_pos)
        rot = self.home_rot if start_rot is None else np.asarray(start_rot)
        self._write_probe(pos, rot)
        self.cmd = np.concatenate([pos - self.base, rot])
        for _ in range(int(0.3 / self.dt)):
            mujoco.mj_step(self.m, self.d)

        self.reflex = StretchReflex(self.m, self.mids,
                                    gain_v=0.6 * REFLEX_GAIN_SCALE,
                                    gain_l=4.0 * REFLEX_GAIN_SCALE)
        try:
            self.reflex.reset(self.d, keep_act=True)
        except TypeError:
            self.reflex.reset(self.d)
        for _ in range(int(BASELINE_S / self.dt)):
            act = self.reflex.step(self.d)
            self.d.ctrl[self.mids] = act
            mujoco.mj_step(self.m, self.d)
        self._base_act = np.array(self.reflex.act, dtype=float)
        self._base_len = np.array(self.d.actuator_length[self.mids], dtype=float)
        self._elbow0 = float(self.d.qpos[self.elbow_adr])
        parts, tot = self._force_by_partner()
        self.start_contact = sorted(parts) if tot > 0.5 else []
        # A solver reset during settling means the baseline was taken on a
        # limb MuJoCo silently re-initialised; the episode must be flagged.
        self.reset_diverged = int(
            self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) - w_reset0
        self._harm_reads = 0
        self.t = 0.0
        self.diverged = False
        self.divergence_count = 0
        return self.observe()

    def step(self, target_pos, target_rot=None):
        """Command an absolute end-effector pose. Returns (obs, info)."""
        import mujoco
        import numpy as np
        w0 = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        pos = np.asarray(target_pos, dtype=float) - self.base
        rot = (self.cmd[3:] if target_rot is None
               else np.asarray(target_rot, dtype=float))
        self.cmd = np.concatenate([pos, rot])
        self.d.ctrl[self.aids] = self.cmd
        peak = dict(force=0.0, reflex=0.0, nonarm=0.0, pressure=0.0,
                    p=0.0, pb=0.0, sb=0.0)
        t_min = 1e9
        for _ in range(self.substeps):
            act = self.reflex.step(self.d)
            self.d.ctrl[self.mids] = act
            mujoco.mj_step(self.m, self.d)
            h = self._instantaneous_harm()
            for k in peak:
                peak[k] = max(peak[k], h[k])
            if h["t_min"] > 0:
                t_min = min(t_min, h["t_min"])
        peak["t_min"] = t_min if t_min < 1e8 else 0.0
        self.t += self.substeps * self.dt
        w1 = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        if w1 > w0:
            self.diverged = True
            self.divergence_count += w1 - w0
        return self.observe(), dict(step_peak=peak,
                                    diverged=bool(self.diverged))

    # ---- boundary ----
    def observe(self):
        """EVERYTHING a controller is allowed to see. Nothing else.

        Compared with the 1-DoF boundary this adds the contact-force VECTOR,
        which is what a wrist force/torque sensor reports; both arms get it.
        """
        import numpy as np
        parts, tot, fvec = self._force_by_partner(want_vec=True)
        f_obs = float(tot + self._rng.normal(0, self.force_noise))
        fvec_obs = fvec + self._rng.normal(0, self.force_noise, 3)
        pos = np.array(self.d.xpos[self.pbody])
        rot = np.array(self.d.qpos[self.qadr[3:]])
        vel = np.array(self.d.qvel[self.dadr[:3]])
        other = []
        for _, adr, dof in self.obs_joints:
            other.append(float(self.d.qpos[adr]
                               + self._rng.normal(0, self.kin_noise)))
            other.append(float(self.d.qvel[dof]
                               + self._rng.normal(0, self.kin_noise * 10)))
        return dict(probe_pos=pos, probe_rot=rot, probe_vel=vel,
                    force=max(0.0, f_obs), force_vec=fvec_obs,
                    other_kin=np.array(other, dtype=float), t=self.t)

    def _force_by_partner(self, want_vec=False):
        return probe_force(self.m, self.d, self.pgeom, want_vec=want_vec)

    def _instantaneous_harm(self):
        import numpy as np
        parts, tot = self._force_by_partner()
        nonarm = sum(v for k, v in parts.items() if k not in ARM_BODIES)
        dl = np.array(self.d.actuator_length[self.mids],
                      dtype=float) - self._base_len
        ln = dl > LENGTHEN_EPS
        rise = (float(np.mean(self.reflex.act[ln] - self._base_act[ln]))
                if ln.any() else 0.0)
        area = max(1e-6, np.pi * self.probe_r ** 2)
        L = self.tissue_load() if self.tissue_map is not None else dict(p=0.0, pb=0.0, sb=0.0, t_min=0.0)
        return dict(force=float(tot), reflex=max(0.0, rise),
                    nonarm=float(nonarm), pressure=float(tot / area / 1e4),
                    lengthen_m=float(dl.max()) if dl.size else 0.0,
                    p=float(L["p"]), pb=float(L["pb"]), sb=float(L["sb"]),
                    t_min=float(L["t_min"]))

    def harm(self):
        self._harm_reads += 1
        return self._instantaneous_harm()

    @property
    def harm_read_count(self):
        return self._harm_reads

    def elbow_change(self):
        """Evaluation-side: signed elbow_flexion change since reset (rad)."""
        return float(self.d.qpos[self.elbow_adr]) - self._elbow0

    def contact_partners(self):
        """Evaluation-side: who the probe is touching right now."""
        parts, tot = self._force_by_partner()
        return {k: round(v, 1) for k, v in parts.items()}


def audit_no_harm_access(controller_fn, env, n_steps=20):
    before = env.harm_read_count
    obs = env.observe()
    for _ in range(n_steps):
        pos, rot = controller_fn(obs)
        obs, _ = env.step(pos, rot)
    leaked = env.harm_read_count - before
    if leaked:
        raise AssertionError(
            f"controller read ground-truth harm {leaked} times; the "
            f"predictive-vs-constraint comparison would be invalid")
    return True


def approach_set(env, n_phi=8):
    """Directions a controller can choose among, all PERPENDICULAR to the bone.

    phi sweeps the plane normal to the forearm axis: phi=0 pushes so the elbow
    flexes (stretching the extensors), phi=180 the opposite, phi=+-90 pushes
    along the elbow hinge axis (out of the flexion plane, which the welded
    humerus cannot yield to).  Pushing along the bone is not a mid-shaft
    contact (the capsule would start inside the limb) and is excluded.
    Returns an ordered dict label -> unit direction.
    """
    import numpy as np
    n, h = env.flex_normal, env.hinge
    out = {}
    for k in range(n_phi):
        phi = 2 * np.pi * k / n_phi
        out[f"phi{int(round(np.degrees(phi))):03d}"] = _unit(
            np.cos(phi) * n + np.sin(phi) * h)
    return out


def capsule_axis_for(env, direction, mode="along"):
    """Capsule orientation for a push direction: lying along the bone, or
    across it (perpendicular to both bone and push)."""
    import numpy as np
    d = _unit(direction)
    if mode == "along":
        return env.axis
    return _unit(np.cross(d, env.axis))


def _selftest():
    import numpy as np
    print("=== care_env6 self-test ===")
    env = CareContactEnv6(body_variation=0.0)
    obs = env.reset(seed=0)
    print(f"probe home   {np.round(env.home, 3).tolist()}  "
          f"rot {np.round(env.home_rot, 3).tolist()}")
    print(f"forearm at   {np.round(env.forearm, 3).tolist()}  "
          f"axis {np.round(env.axis, 3).tolist()}  r={env.cap_r:.3f}")
    print(f"elbow hinge  {np.round(env.hinge, 3).tolist()}  "
          f"flex normal {np.round(env.flex_normal, 3).tolist()}")
    print(f"rest sag     {env.meta['rest_sag_mm']:.2f} mm")
    print(f"obs keys: {sorted(obs)}")

    # harm audit must still catch a cheater
    def honest(o):
        return env.home + np.array([0, 0, -0.01 * o["t"]]), env.home_rot

    def cheater(o):
        env.harm()
        return env.home, env.home_rot
    env.reset(seed=0)
    audit_no_harm_access(honest, env, n_steps=5)
    env.reset(seed=0)
    try:
        audit_no_harm_access(cheater, env, n_steps=5)
        print("CHEATER NOT CAUGHT -- fix the audit")
        return 1
    except AssertionError:
        print("harm audit: honest passes, cheater caught")

    # servo tracking in free space: command 5 cm UP (down would touch the
    # limb, home is only 0.04 m above it), measure the lag
    env.reset(seed=0)
    tgt = env.home + np.array([0, 0, 0.05])
    for _ in range(25):
        obs, _ = env.step(tgt, env.home_rot)
    err = np.linalg.norm(obs["probe_pos"] - tgt)
    print(f"free-space tracking error after 0.5 s: {err * 1000:.2f} mm")

    # the question the 6-DoF scene must answer: does DIRECTION change harm?
    import mujoco
    ej = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
    eadr = env.m.jnt_qposadr[ej]
    advance = 0.08
    print(f"\n{'approach':16s} {'force':>7} {'reflex':>8} {'stretch':>8} "
          f"{'d_elbow':>8} {'nonarm%':>8}  partners")
    rows = {}
    for label, dvec in approach_set(env).items():
        for cap_lab in ("along", "across"):
            cap_ax = capsule_axis_for(env, dvec, cap_lab)
            pos, rot = env.approach_start(dvec, capsule_axis=cap_ax)
            env.reset(seed=0, start_pos=pos, start_rot=rot)
            if env.start_contact:
                print(f"{label + '/' + cap_lab:16s} start touches "
                      f"{env.start_contact} -- skipped")
                continue
            q0 = float(env.d.qpos[eadr])
            pf = pr = na = st = de = 0.0
            partners = {}
            for i in range(30):
                frac = min(1.0, (i + 1) / 15.0)
                _, info = env.step(pos + dvec * advance * frac, rot)
                p = info["step_peak"]
                pf = max(pf, p["force"]); pr = max(pr, p["reflex"])
                na = max(na, p["nonarm"])
                st = max(st, env._instantaneous_harm()["lengthen_m"])
                de = max(de, abs(float(env.d.qpos[eadr]) - q0))
                for k, v in env.contact_partners().items():
                    partners[k] = max(partners.get(k, 0.0), v)
            share = 100 * na / pf if pf > 0 else 0.0
            rows[(label, cap_lab)] = (pf, pr)
            print(f"{label + '/' + cap_lab:16s} {pf:7.1f} {pr:8.5f} "
                  f"{st * 1000:6.2f}mm {np.degrees(de):7.2f}d {share:7.1f}%  "
                  f"{partners}")
    # the 1-DoF task, replayed here: straight down from home to depth 0.09
    env.reset(seed=0)
    pf = pr = 0.0
    for i in range(30):
        tgt = env.home + env.down * min(0.09, 0.02 * (i + 1))
        _, info = env.step(tgt, env.home_rot)
        pf = max(pf, info["step_peak"]["force"])
        pr = max(pr, info["step_peak"]["reflex"])
    print(f"\nnominal task (down 0.09 from home, greedy): force {pf:.1f} N  "
          f"reflex {pr:.5f}  elbow change {np.degrees(env.elbow_change()):+.2f} deg  "
          f"partners {env.contact_partners()}")

    forces = [v[0] for v in rows.values() if v[0] > 0]
    reflexes = [v[1] for v in rows.values() if v[0] > 0]
    if len(forces) >= 2:
        print(f"\nforce  range across directions : "
              f"{min(forces):.1f} .. {max(forces):.1f} N "
              f"({max(forces) / max(min(forces), 1e-9):.1f}x)")
        print(f"reflex range across directions : "
              f"{min(reflexes):.5f} .. {max(reflexes):.5f} "
              f"({max(reflexes) / max(min(reflexes), 1e-9):.1f}x)")
        print("=> a direction choice exists for a model to get wrong"
              if max(reflexes) > 3 * max(min(reflexes), 1e-9) else
              "=> directions are nearly equivalent; 6-DoF buys little here")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
