"""Shoulder route: the arm-lift task on the free-shoulder arm, shared by the
diagnostics and the main experiment.

Body: myoArm with the humerus NOT welded (the arm hangs from the shoulder
complex), person-specific passive range of motion: the shoulder can elevate
`free_sh` degrees and the elbow can flex `free_el` degrees above the person's
own settled posture before the joint limit engages.

Task: raise the forearm centre by RISE_M and keep it there for HOLD_S.  The
lift is closed on the OBSERVED forearm height (kinematics the robot can see):
the probe advances at `speed` while the height is below target and stops
otherwise, so a limb that swings back is followed instead of dropped.

Harm (never shown to a controller):
  lim_sh, lim_el   peak torque pressed into the shoulder / elbow limit, N m,
                   50 ms moving average  (end-range loading)
  gh_shear         peak glenohumeral shear, N, 50 ms moving average
Observable: wrist force magnitude (peak), forearm height, joint angles.
"""
import math

RISE_M = 0.05
HOLD_S = 0.5
T_MAX = 3.0
CLEAR = 0.03
_ENV = None


def env():
    import diag_shoulder1 as s1
    return s1.env()


def limit_torque(e):
    import mujoco
    d = e.d
    sh = el = 0.0
    for i in range(d.nefc):
        if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
            j = int(d.efc_id[i])
            if j == e._sj:
                sh += abs(float(d.efc_force[i]))
            elif j == e._ej:
                el += abs(float(d.efc_force[i]))
    return sh, el


def run(seed, free_sh, free_el, offset, angle, speed, stop_force=None, model_err=None, rise_m=None, hold_s=None, soft_deg=None):
    """One open-loop strategy (site, angle, speed) on one person.
    stop_force: optional reactive guard - abort (retract) when the wrist force
    exceeds this value; returns aborted=True and the harm up to that point."""
    import mujoco
    import numpy as np
    import diag_lift_oracle as dlo
    import diag_shoulder1 as s1
    from care_env6 import AIM_DISTAL
    RISE = RISE_M if rise_m is None else float(rise_m)      # a short, gentle probe uses a smaller rise
    HOLD = HOLD_S if hold_s is None else float(hold_s)
    e = env()
    e._gh_log = None
    # The person-specific limits below are written into the model.  reset() restores the joint ranges
    # only when body_variation > 0, so in the robot's copy (body_variation = 0) the limits of earlier
    # episodes accumulated (bug found 2026-09-21; PREREGISTRATION round 17 fix).  Restore them here.
    if getattr(e, "_jr0", None) is None:
        e._jr0 = np.array(e.m.jnt_range)
        e._jm0 = np.array(e.m.jnt_margin)
        e._jsi0 = np.array(e.m.jnt_solimp)
    e.m.jnt_range[:] = e._jr0
    e.m.jnt_margin[:] = e._jm0            # soft-limit settings must not leak between episodes either
    e.m.jnt_solimp[:] = e._jsi0
    e.reset(seed=seed)
    e.tissue_load()
    m, d = e.m, e.d
    rest = {}
    for name, j, free in (("sh", e._sj, free_sh), ("el", e._ej, free_el)):
        q = float(d.qpos[m.jnt_qposadr[j]]); rest[name] = q
        if soft_deg and free < 90.0:
            # progressive end-feel (round 22): resistance starts soft_deg/2 BEFORE the recorded limit and
            # stiffens over soft_deg, centred on it -- MuJoCo limit margin + impedance ramp (solimp width)
            w = math.radians(soft_deg)
            m.jnt_range[j, 1] = min(float(m.jnt_range[j, 1]), q + math.radians(free) + 0.5 * w)
            m.jnt_margin[j] = w
            m.jnt_solimp[j] = [0.01, 0.95, w, 0.5, 2.0]
        else:
            m.jnt_range[j, 1] = min(float(m.jnt_range[j, 1]), q + math.radians(free))
    dvec = dlo.lift_dir(e, angle)
    pos, rot = e.approach_start(dvec, e.axis, clearance=CLEAR, aim_shift=AIM_DISTAL + offset)
    e._write_probe(pos, rot)
    e.cmd = np.concatenate([pos - e.base, rot])
    for _ in range(int(0.2 / e.dt)):
        act = e.reflex.step(d); d.ctrl[e.mids] = act; mujoco.mj_step(m, d)
    _, tot = e._force_by_partner()
    site_ok = tot <= 0.5
    z0 = float(d.geom_xpos[e._fg][2])
    base_sh, base_el = limit_torque(e)     # round 23: passive end-range tension already present before contact
    lim = []
    orig = e._instantaneous_harm

    def hook():
        h = orig()
        lim.append(limit_torque(e))
        return h
    e._instantaneous_harm = hook
    e._gh_log = []
    cmd = np.array(pos); t_done = None; n_hold = 0; peak_f = 0.0; aborted = False
    try:
        for _ in range(int(T_MAX * 50)):
            rise = float(d.geom_xpos[e._fg][2]) - z0
            if rise < RISE:
                cmd = cmd + dvec * speed / 50
            obs, info = e.step(cmd, rot)
            f = info["step_peak"]["force"]; peak_f = max(peak_f, f)
            if stop_force is not None and f > stop_force:
                aborted = True
                break
            rise = float(d.geom_xpos[e._fg][2]) - z0
            if t_done is None and rise >= RISE:
                t_done = e.t
            if rise >= 0.9 * RISE and t_done is not None:
                n_hold += 1
                if n_hold >= int(HOLD * 50):
                    break
    finally:
        e._instantaneous_harm = orig
    g = np.array(e._gh_log); e._gh_log = None
    L = np.array(lim)
    n0 = min(50, max(0, len(L) - 30))
    ker = np.ones(25) / 25.0
    sm = lambda x: np.convolve(x, ker, mode="valid") if len(x) >= 25 else np.array([float(np.mean(x))] if len(x) else [0.0])
    rise = float(d.geom_xpos[e._fg][2]) - z0
    done = bool(site_ok and not e.diverged and not aborted and t_done is not None and n_hold >= int(HOLD * 50))
    return dict(seed=seed, free_sh=free_sh, free_el=free_el, offset=offset, angle=angle, speed=speed,
                site_ok=bool(site_ok), diverged=bool(e.diverged), aborted=aborted, done=done,
                t_done=t_done, t_end=float(e.t), peak_force=peak_f,
                lim_sh=float(sm(L[n0:, 0]).max()), lim_el=float(sm(L[n0:, 1]).max()),
                gh_shear=float(sm(g[n0:, 1]).max()),
                d_sh_deg=math.degrees(float(d.qpos[m.jnt_qposadr[e._sj]]) - rest["sh"]),
                d_el_deg=math.degrees(float(d.qpos[m.jnt_qposadr[e._ej]]) - rest["el"]), end_rise=rise,
                rest_sh_deg=math.degrees(rest["sh"]), rest_el_deg=math.degrees(rest["el"]),
                lim_base_sh=float(base_sh), lim_base_el=float(base_el))


def _job(spec):
    try:
        return run(*spec)
    except Exception as ex:                       # keep the pool alive; the caller counts errors
        return dict(seed=spec[0], free_sh=spec[1], free_el=spec[2], offset=spec[3], angle=spec[4], speed=spec[5],
                    error=repr(ex)[:200], done=False)
