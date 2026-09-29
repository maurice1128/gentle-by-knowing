"""One environment every controller shares, with the observation boundary
enforced in code rather than by convention.

The paper's whole argument rests on a single separation:

    controller-visible : own proprioception + contact force + the other's
                         NOISY kinematics.  Both the predictive controller and
                         the constraint (CBF) baseline see exactly this.
    other-model output : the recipient's internal state, including muscle
                         activation and the harm channel.  Only the predictive
                         controller has this, and only by PREDICTING it.
    evaluation         : the simulator's TRUE harm.  Neither controller sees it.

If a controller could read the true internal state, the comparison would be
rigged and the result meaningless.  So `observe()` is the only way to get data
out of this class for a controller, and `harm()` deliberately lives behind a
separate call that records every access, so an audit can prove no controller
touched it during a rollout.

Configuration is the one Gate B validated (notes/PHASE0_LOG.md §6):
  * humerus welded to the world - the care posture, arm resting on a bed.
    Gate B reported a manner/noise separation of 943 here against 3.15 for a
    free limb, but read that carefully: the constraint fixes the contact
    geometry, so POSTURE jitter barely moves the outcome (a 0.036 rad spread
    changes peak harm by 0.03%).  The high separation says manner beats posture
    variation, NOT that it beats individual variation.  Individuals differ in
    strength, limb mass and range of motion, which is what `body_variation`
    perturbs; with it at 0.25 the same seeds spread harm by 11.7%.
  * L2 = activation rise of the LENGTHENED muscles, NOT a co-contraction index
    (CCI is 2*min(flexors, extensors) and cannot move under a one-sided reflex)
  * reflex gain frozen at x2, whose torque increment 1.83 N*m sits inside the
    healthy passive-resistance band
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

REFLEX_GAIN_SCALE = 2.0
LENGTHEN_EPS = 1e-4
SETTLE_S = 2.0
BASELINE_S = 0.3

# quantities a controller may never read directly; asserted below
FORBIDDEN_FIELDS = ("act", "actuator_length", "actuator_velocity",
                    "actuator_force", "ten_length", "ten_velocity",
                    "qfrc_actuator", "qfrc_constraint")


class CareContactEnv:
    """Robot probe contacting a supported, reflexive forearm.

    Parameters
    ----------
    welded          : humerus fixed to world (care posture) vs free limb
    probe_radius    : m
    probe_stiffness : N/m, position-servo gain of the probe
    kin_noise       : rad, gaussian noise on the other's kinematics as seen by
                      the controller.  This is the ONLY view of the other body
                      any controller gets.
    force_noise     : N, gaussian noise on the force reading
    """

    def __init__(self, welded=True, probe_radius=0.05, probe_stiffness=6000.0,
                 kin_noise=0.01, force_noise=0.5, settle_s=SETTLE_S,
                 control_hz=50.0, body_variation=0.0, tracked_joints=None):
        import mujoco
        import numpy as np
        from gate_b_prime import build_supported
        from arm_scene import muscle_ids, ELBOW_FLEXORS, ELBOW_EXTENSORS

        self.m, self.meta = build_supported(
            weld_humerus=welded, probe_radius=probe_radius,
            probe_stiffness=probe_stiffness, settle_s=settle_s, strict=True)
        self.d = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)
        self.substeps = max(1, int(round((1.0 / control_hz) / self.dt)))
        self.kin_noise = float(kin_noise)
        self.force_noise = float(force_noise)
        self.settle_s = float(settle_s)

        self.fids = muscle_ids(self.m, ELBOW_FLEXORS)
        self.eids = muscle_ids(self.m, ELBOW_EXTENSORS)
        self.mids = self.fids + self.eids
        self.aid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     "probe_act")
        self.jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                     "probe_slide")
        self.j_adr = self.m.jnt_qposadr[self.jid]
        self.j_dof = self.m.jnt_dofadr[self.jid]
        # joints the controller is allowed to see (noisy), the other's posture
        # tracked_joints overrides the default three (the learned other-model
        # of other_model_welded.py was trained on five); whichever list is
        # used, EVERY controller in that run sees the same one
        self.obs_joints = []
        for nm in (tracked_joints or ("elbow_flexion", "pro_sup", "shoulder_elv")):
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if j >= 0:
                self.obs_joints.append((nm, self.m.jnt_qposadr[j],
                                        self.m.jnt_dofadr[j]))
        self._harm_reads = 0
        self._rng = np.random.default_rng(0)
        self.reflex = None
        self._base_act = None
        self._base_len = None
        # How much recipients differ from one another.  Posture jitter alone is
        # NOT individual variation: measured 2026-09-04, on the supported limb a
        # 0.036 rad posture spread moves peak harm by only 0.03%, because the
        # constraint fixes the contact geometry.  Real recipients differ in
        # strength, limb mass and range of motion, so a "population" has to vary
        # those.  0.0 reproduces the earlier posture-only behaviour.
        self.body_variation = float(body_variation)
        self._pristine = None

    # ---------------- lifecycle ----------------
    def reset(self, seed=0, jitter=0.01):
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
        for j in range(self.m.njnt):
            adr = self.m.jnt_qposadr[j]
            if self.m.jnt_limited[j]:
                lo, hi = self.m.jnt_range[j]
                self.d.qpos[adr] = float(np.clip(
                    self.d.qpos[adr] + rng.normal(0, jitter),
                    lo + 1e-4, hi - 1e-4))
        self.d.ctrl[:] = 0.0
        for _ in range(int(self.settle_s / self.dt)):
            mujoco.mj_step(self.m, self.d)

        self.reflex = StretchReflex(
            self.m, self.mids, gain_v=0.6 * REFLEX_GAIN_SCALE,
            gain_l=4.0 * REFLEX_GAIN_SCALE)
        try:
            self.reflex.reset(self.d, keep_act=True)
        except TypeError:
            self.reflex.reset(self.d)

        for _ in range(int(BASELINE_S / self.dt)):
            act = self.reflex.step(self.d)
            self.d.ctrl[self.mids] = act
            self.d.ctrl[self.aid] = 0.0
            mujoco.mj_step(self.m, self.d)

        self._base_act = np.array(self.reflex.act, dtype=float)
        self._base_len = np.array(self.d.actuator_length[self.mids], dtype=float)
        self._harm_reads = 0
        self.t = 0.0
        self.diverged = False
        self.divergence_count = 0
        return self.observe()

    def step(self, probe_cmd):
        """Advance one control interval. Returns (obs, info).

        Also watches the solver.  Deep contact against a stiff position servo can
        trip mjWARN_BADQACC; MuJoCo then resets the offending state underneath
        and the episode continues, producing data that looks ordinary and is not
        physical.  One episode in 400 did this during other-model data
        collection, so the warning counters are read every control step and the
        episode is flagged rather than silently kept.
        """
        import mujoco
        peak = dict(force=0.0, reflex=0.0, torso=0.0)
        w0 = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        for _ in range(self.substeps):
            act = self.reflex.step(self.d)
            self.d.ctrl[self.mids] = act
            self.d.ctrl[self.aid] = float(probe_cmd)
            mujoco.mj_step(self.m, self.d)
            h = self._instantaneous_harm()
            peak["force"] = max(peak["force"], h["force"])
            peak["reflex"] = max(peak["reflex"], h["reflex"])
            peak["torso"] = max(peak["torso"], h["torso_share"])
        self.t += self.substeps * self.dt
        w1 = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        if w1 > w0:
            self.diverged = True
            self.divergence_count = getattr(self, "divergence_count", 0) + (w1 - w0)
        return self.observe(), dict(step_peak=peak,
                                    diverged=bool(getattr(self, "diverged", False)))

    def _apply_body_variation(self, seed):
        """Give this seed a different BODY, not just a different posture.

        Perturbs the same physical quantities the fidelity axes use, so an
        individual and a modelling error are commensurable: the paper can say
        how the required model accuracy compares with the spread of the
        population it must serve.
        """
        import mujoco
        import numpy as np
        if self._pristine is None:
            self._pristine = dict(
                gainprm=np.array(self.m.actuator_gainprm),
                biasprm=np.array(self.m.actuator_biasprm),
                dof_damping=np.array(self.m.dof_damping),
                body_mass=np.array(self.m.body_mass),
                body_inertia=np.array(self.m.body_inertia),
                jnt_range=np.array(self.m.jnt_range))
        p = self._pristine
        self.m.actuator_gainprm[:] = p["gainprm"]
        self.m.actuator_biasprm[:] = p["biasprm"]
        self.m.dof_damping[:] = p["dof_damping"]
        self.m.body_mass[:] = p["body_mass"]
        self.m.body_inertia[:] = p["body_inertia"]
        self.m.jnt_range[:] = p["jnt_range"]

        rng = np.random.default_rng(seed + 777_000)
        sd = self.body_variation
        for i in range(self.m.nu):
            if self.m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE:
                f = float(np.exp(rng.normal(0, sd)))
                if self.m.actuator_gainprm[i, 2] > 0:
                    self.m.actuator_gainprm[i, 2] *= f
                if self.m.actuator_biasprm[i, 2] > 0:
                    self.m.actuator_biasprm[i, 2] *= f
                g = float(np.exp(rng.normal(0, sd)))
                self.m.actuator_gainprm[i, 7] *= g
                self.m.actuator_biasprm[i, 7] *= g
        # Only the recipient's DOFs: until 2026-09-09 this line also scaled
        # the PROBE joints' damping (the robot's own servo is not part of the
        # population), which on the 6-DoF chain drove a hinge unstable.
        f_damp = float(np.exp(rng.normal(0, sd)))
        recipient = np.ones(self.m.nv, dtype=bool)
        for j in range(self.m.njnt):
            nm = str(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j))
            if nm.startswith("probe"):
                a = self.m.jnt_dofadr[j]
                recipient[a:a + (6 if self.m.jnt_type[j] == 0 else 1)] = False
        self.m.dof_damping[:] = p["dof_damping"]
        self.m.dof_damping[recipient] = p["dof_damping"][recipient] * f_damp
        for b in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm in ("humerus", "ulna", "radius"):
                f = float(np.exp(rng.normal(0, sd * 0.7)))
                self.m.body_mass[b] *= f
                self.m.body_inertia[b] *= f

    # ---------------- the boundary ----------------
    def observe(self):
        """EVERYTHING a controller is allowed to see. Nothing else."""
        import numpy as np
        from supported_scene import force_by_partner
        _, tot = force_by_partner(self.m, self.d)
        f_obs = float(tot + self._rng.normal(0, self.force_noise))
        probe_q = float(self.d.qpos[self.j_adr])
        probe_v = float(self.d.qvel[self.j_dof])
        other = []
        for _, adr, dof in self.obs_joints:
            other.append(float(self.d.qpos[adr]
                               + self._rng.normal(0, self.kin_noise)))
            other.append(float(self.d.qvel[dof]
                               + self._rng.normal(0, self.kin_noise * 10)))
        return dict(probe_q=probe_q, probe_v=probe_v, force=max(0.0, f_obs),
                    other_kin=np.array(other, dtype=float), t=self.t)

    def _instantaneous_harm(self):
        import numpy as np
        from supported_scene import force_by_partner, torso_share
        parts, tot = force_by_partner(self.m, self.d)
        dl = np.array(self.d.actuator_length[self.mids],
                      dtype=float) - self._base_len
        lengthened = dl > LENGTHEN_EPS
        if lengthened.any():
            reflex_rise = float(np.mean(self.reflex.act[lengthened]
                                        - self._base_act[lengthened]))
        else:
            reflex_rise = 0.0
        return dict(force=float(tot), reflex=max(0.0, reflex_rise),
                    torso_share=torso_share(parts, tot),
                    lengthen_m=float(dl.max()) if dl.size else 0.0)

    def harm(self):
        """Ground-truth harm. EVALUATION ONLY - never inside a controller.

        Every call is counted so a rollout can be audited afterwards.
        """
        self._harm_reads += 1
        return self._instantaneous_harm()

    @property
    def harm_read_count(self):
        return self._harm_reads


def audit_no_harm_access(controller_fn, env, n_steps=20):
    """Run a controller briefly and assert it never called env.harm().

    This is the mechanical guarantee that the comparison is not rigged.  It is
    cheap, so every experiment should call it before collecting results.
    """
    before = env.harm_read_count
    obs = env.observe()
    for _ in range(n_steps):
        cmd = controller_fn(obs)
        obs, _ = env.step(cmd)
    leaked = env.harm_read_count - before
    if leaked:
        raise AssertionError(
            f"controller read ground-truth harm {leaked} times; the "
            f"predictive-vs-constraint comparison would be invalid")
    return True


def _selftest():
    import numpy as np
    print("=== care_env self-test ===")
    env = CareContactEnv(welded=True)
    obs = env.reset(seed=0)
    print(f"observation keys: {sorted(obs)}")
    print(f"other_kin dim   : {obs['other_kin'].shape}")
    print(f"substeps/control: {env.substeps} (dt={env.dt})")

    # a controller that only uses observations must pass the audit
    def honest(o):
        return min(0.09, 0.002 + 0.6 * o["t"])
    env.reset(seed=0)
    ok = audit_no_harm_access(honest, env)
    print(f"honest controller passes harm audit: {ok}")

    # a controller that peeks must be caught
    def cheater(o):
        env.harm()
        return 0.05
    env.reset(seed=0)
    try:
        audit_no_harm_access(cheater, env)
        print("CHEATER NOT CAUGHT -- the audit is decorative, fix it")
        return 1
    except AssertionError as e:
        print(f"cheating controller correctly caught: {str(e)[:60]}...")

    # forbidden fields must not appear in observe()'s source
    import inspect
    src = inspect.getsource(CareContactEnv.observe)
    bad = [f for f in FORBIDDEN_FIELDS if f"d.{f}" in src]
    print(f"forbidden fields referenced in observe(): {bad if bad else 'none'}")

    # a real contact event, showing harm responds
    env.reset(seed=0)
    peaks = []
    for i in range(40):
        cmd = min(0.09, 0.0025 * i)
        _, info = env.step(cmd)
        peaks.append(info["step_peak"])
    print(f"peak force  = {max(p['force'] for p in peaks):.1f} N")
    print(f"peak reflex = {max(p['reflex'] for p in peaks):.5f}")
    print(f"max torso   = {max(p['torso'] for p in peaks) * 100:.1f}%")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
