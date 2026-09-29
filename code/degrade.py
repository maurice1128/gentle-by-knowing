"""Fidelity-degradation axes for the care-recipient body model.

This is the independent variable of the paper's centrepiece experiment: how
wrong may the robot's model of the other body be before predictive gentleness
stops beating constraint-based gentleness?

Design requirements taken from the literature audit (notes/GAP_CLAIM.md §3):

* **Bidirectional.**  Zhao et al. (ACC 2020) found no safety crossover because
  their bad predictors made the robot conservative, which a separation-distance
  metric rewards.  Tian et al. (ICRA 2022) found one because their bad model made
  the robot overconfident.  So every axis must be swept in BOTH directions:
  under-estimating stiffness -> overconfident -> harm;
  over-estimating stiffness -> over-conservative -> task failure.
  This mirrors ISO 10218-2 Annex M's simplified body model, which under-estimates
  by >100 N for the hand and over-estimates for shoulder/back.
* **Physically meaningful units.**  Each knob is a real body parameter with a
  unit, not an abstract noise level, so the resulting crossover reads as a
  specification ("the model must be within X% on muscle strength").
* **Reflex knobs included.**  Added after the 2026-09-04 literature round: older
  adults co-contract MORE and LATER (Sanders et al. 2019), complete SCI amplifies
  and spatially expands the reflex (Andersen et al. 2004), acute spinal shock
  suppresses it.  Reflex gain and delay are therefore first-class fidelity axes,
  not nuisance parameters.

A degradation is a multiplicative factor `f` applied to a parameter group.
`f = 1.0` is the true body.  `f < 1` and `f > 1` are the two failure directions.
The TRUE body is never modified; only the copy handed to the robot is.
"""
import os
import copy

ROOT = os.path.dirname(os.path.abspath(__file__))
REFLEX_GAIN_SCALE = 2.0   # frozen, must equal care_env.REFLEX_GAIN_SCALE

# knob name -> (human-readable, unit, direction meaning at f<1)
AXES = {
    "muscle_force": ("peak isometric muscle force (all 63 MTUs)", "N",
                     "weaker body than truth -> robot expects less resistance"),
    # NOTE: myoArm's jnt_stiffness is identically zero (verified 2026-09-04),
    # so passive resistance comes from the Hill-type MTU passive-force element,
    # not from joint springs.  The knob therefore scales muscle fpmax
    # (gainprm[7]), which is where passive stiffness actually lives.
    "passive_stiffness": ("MTU passive-force scale (muscle fpmax)", "-",
                          "floppier body than truth"),
    "joint_damping": ("joint damping", "N*m*s/rad", "less damped than truth"),
    "rom": ("joint range of motion width", "rad", "narrower ROM than truth"),
    "segment_mass": ("limb segment mass", "kg", "lighter limb than truth"),
    "reflex_gain": ("stretch-reflex gain (Gv and Gl)", "-",
                    "weaker defensive response than truth"),
    "reflex_delay": ("afferent delay", "s", "faster reflex than truth"),
}

# default sweep: log-spaced either side of truth
DEFAULT_FACTORS = (0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0)


def degraded_model(base_model, axis, factor):
    """Return a COPY of `base_model` with `axis` scaled by `factor`.

    The input model is never mutated.  Reflex axes are not model parameters, so
    for those this returns an unchanged copy and the factor must be applied to
    the StretchReflex constructor instead (see `reflex_kwargs`).
    """
    import mujoco
    import numpy as np

    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; known: {sorted(AXES)}")
    m = copy.deepcopy(base_model)
    f = float(factor)

    if axis == "muscle_force":
        # MuJoCo muscle: gainprm[2] is peak force (scale), see mju_muscleGain.
        # Only touch muscle actuators, never the probe actuator.
        for i in range(m.nu):
            if m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE:
                if m.actuator_gainprm[i, 2] > 0:
                    m.actuator_gainprm[i, 2] *= f
                if m.actuator_biasprm[i, 2] > 0:
                    m.actuator_biasprm[i, 2] *= f
    elif axis == "passive_stiffness":
        # gainprm layout for mjGAIN_MUSCLE:
        #   0 range0, 1 range1, 2 force, 3 scale, 4 lmin, 5 lmax,
        #   6 vmax, 7 fpmax, 8 fvmax
        # fpmax is the passive-force peak, i.e. the passive stiffness of the MTU.
        # Not fully orthogonal to `muscle_force`, which scales gainprm[2]; both
        # are reported so the paper can state the coupling.
        for i in range(m.nu):
            if m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE:
                m.actuator_gainprm[i, 7] *= f
                m.actuator_biasprm[i, 7] *= f
    elif axis == "joint_damping":
        # recipient joints only: the probe's own servo damping is not a body
        # parameter (same mistake as care_env's body variation, 2026-09-10)
        for j in range(m.njnt):
            if _is_probe_joint(m, j):
                continue
            a = m.jnt_dofadr[j]
            n = 6 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (
                3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1)
            m.dof_damping[a:a + n] *= f
    elif axis == "rom":
        # scale the half-width about the midpoint, keep the midpoint fixed.
        # Recipient joints only: until 2026-09-10 this also shrank the 6-DoF
        # probe's slide (+-0.3 m) and hinge (+-3.1 rad) ranges inside the
        # robot's internal copy, so at x0.5 the copy's own end-effector could
        # not reach the commanded pose - the "rom paralysis" of the first
        # 6-DoF round was at least partly that artefact.
        for j in range(m.njnt):
            if _is_probe_joint(m, j):
                continue
            lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * f
            m.jnt_range[j, 0] = mid - half
            m.jnt_range[j, 1] = mid + half
    elif axis == "segment_mass":
        # arm chain only; leave the world/thorax alone so the trunk stays put
        arm = {"humerus", "ulna", "radius"}
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if name in arm:
                m.body_mass[b] *= f
                m.body_inertia[b] *= f
    elif axis in ("reflex_gain", "reflex_delay"):
        pass  # handled by reflex_kwargs
    return m


def _is_probe_joint(m, j):
    import mujoco
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    return nm is not None and str(nm).startswith("probe")


def reflex_kwargs(axis, factor, base=None):
    """Reflex-controller kwargs for a given degradation.

    Returns a dict to merge into StretchReflex(...).  For non-reflex axes this
    is empty, i.e. the reflex is untouched.
    """
    # The TRUE recipient runs the frozen calibrated reflex (care_env.py,
    # gate_b_prime.py): gain x2 on the SpasticMyoElbow defaults.  Until
    # 2026-09-09 this default was the UNscaled (0.6, 4.0), so every "truth"
    # internal model in predictive.py silently carried half the real reflex
    # gain and the whole reflex_gain axis was shifted by one octave: the row
    # labelled truth was really x0.5, the row labelled x2 was the truth, and
    # the "paralysis at x4" was paralysis at x2.  Round-2 crossover numbers
    # (out/crossover/crossover.json) were produced under that bug.
    base = (dict(gain_v=0.6 * REFLEX_GAIN_SCALE, gain_l=4.0 * REFLEX_GAIN_SCALE,
                 delay_s=0.025) if base is None else dict(base))
    f = float(factor)
    if axis == "reflex_gain":
        base["gain_v"] *= f
        base["gain_l"] *= f
        return base
    if axis == "reflex_delay":
        base["delay_s"] *= f
        return base
    return base


def sweep_spec(axes=None, factors=DEFAULT_FACTORS):
    """All (axis, factor) pairs for a full sweep, truth included exactly once."""
    axes = list(AXES) if axes is None else list(axes)
    out = []
    seen_truth = False
    for a in axes:
        for f in factors:
            if abs(f - 1.0) < 1e-12:
                if seen_truth:
                    continue
                seen_truth = True
                out.append(("truth", 1.0))
            else:
                out.append((a, float(f)))
    return out


def _verify():
    """Prove every axis actually changes the model, and f=1 changes nothing.

    A degradation knob that silently does nothing would make the crossover curve
    flat for a reason that has nothing to do with the science.  This is the
    cheapest possible guard against that.
    """
    import sys
    import numpy as np
    import mujoco
    sys.path.insert(0, ROOT)
    from care_env6 import build6

    # the 6-DoF scene: it carries probe joints, so this also checks that the
    # joint_damping and rom knobs leave the robot's own joints alone
    m0, _ = build6(strict=False)
    probe_dofs = [m0.jnt_dofadr[j] for j in range(m0.njnt)
                  if _is_probe_joint(m0, j)]
    probe_jnts = [j for j in range(m0.njnt) if _is_probe_joint(m0, j)]
    print(f"{'axis':16s} {'f':>6s}  changed?  witness")
    ok = True
    for axis in AXES:
        for f in (0.5, 1.0, 2.0):
            m = degraded_model(m0, axis, f)
            rk = reflex_kwargs(axis, f)
            if axis == "muscle_force":
                w0 = float(np.sum(m0.actuator_gainprm[:, 2]))
                w1 = float(np.sum(m.actuator_gainprm[:, 2]))
            elif axis == "passive_stiffness":
                w0 = float(m0.actuator_gainprm[:, 7].sum())
                w1 = float(m.actuator_gainprm[:, 7].sum())
            elif axis == "joint_damping":
                w0, w1 = float(m0.dof_damping.sum()), float(m.dof_damping.sum())
            elif axis == "rom":
                w0 = float((m0.jnt_range[:, 1] - m0.jnt_range[:, 0]).sum())
                w1 = float((m.jnt_range[:, 1] - m.jnt_range[:, 0]).sum())
            elif axis == "segment_mass":
                w0, w1 = float(m0.body_mass.sum()), float(m.body_mass.sum())
            elif axis == "reflex_gain":
                w0, w1 = 0.6 * REFLEX_GAIN_SCALE, rk["gain_v"]
            else:
                w0, w1 = 0.025, rk["delay_s"]
            changed = abs(w1 - w0) > 1e-12
            expect = (abs(f - 1.0) > 1e-12)
            good = (changed == expect)
            ok = ok and good
            print(f"{axis:16s} {f:6.2f}  {str(changed):8s}  {w0:.6g} -> {w1:.6g}"
                  f"{'' if good else '   <-- WRONG'}")
    print("\nknob verification:", "PASS" if ok else "FAIL")

    md = degraded_model(m0, "joint_damping", 0.5)
    mr = degraded_model(m0, "rom", 0.5)
    leak = (not np.allclose(md.dof_damping[probe_dofs], m0.dof_damping[probe_dofs])
            or not np.allclose(mr.jnt_range[probe_jnts], m0.jnt_range[probe_jnts]))
    print("probe joints untouched by joint_damping / rom:",
          "FAIL" if leak else "PASS")
    ok = ok and not leak

    # base model must be untouched by any degradation
    m_check, _ = build6(strict=False)
    same = (np.allclose(m_check.actuator_gainprm, m0.actuator_gainprm)
            and np.allclose(m_check.jnt_range, m0.jnt_range)
            and np.allclose(m_check.body_mass, m0.body_mass))
    print("base model unmutated:", "PASS" if same else "FAIL")
    return 0 if (ok and same) else 1


if __name__ == "__main__":
    raise SystemExit(_verify())
