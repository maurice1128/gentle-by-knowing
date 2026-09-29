"""Predictive gentleness: plan against a model of the OTHER body.

The robot carries its own copy of the recipient's body and rolls it forward to
ask "if I move like this, what happens inside them?"  It then picks the action
whose PREDICTED harm is acceptable.  Crucially the copy can be wrong, and how
wrong it is allowed to be is the paper's independent variable.

Why the internal model is a degraded simulator rather than a learned network
---------------------------------------------------------------------------
Both are legitimate instantiations of "the robot holds a model of the other
body", but they answer different questions.  A learned model conflates two
sources of error: what the architecture failed to learn, and how wrong the body
parameters are.  Only the second is the quantity this paper is about, and only
the second has physical units that turn a crossover into a specification
("the model must be within X% on muscle strength").  So the internal model is a
MuJoCo model of the recipient whose parameters are perturbed along the axes in
degrade.py.  A learned-model arm is a complement, not a substitute.

Fidelity levels
    truth      : internal parameters equal the real body (upper bound anchor)
    degraded   : one axis scaled by a factor, both directions
    randprior  : parameters randomised - the phase-1 control that showed a
                 knowledge-free model is worse than no model at all

The controller sees only what care_env.observe() returns.  It never reads the
true internal state; care_env.audit_no_harm_access enforces that mechanically.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

REFLEX_GAIN_SCALE = 2.0
LENGTHEN_EPS = 1e-4


class OtherModel:
    """The robot's private, possibly-wrong copy of the recipient's body."""

    def __init__(self, axis="truth", factor=1.0, welded=True, probe_radius=0.05,
                 probe_stiffness=6000.0, seed=0):
        import mujoco
        import numpy as np
        from gate_b_prime import build_supported
        from degrade import degraded_model, reflex_kwargs
        from arm_scene import muscle_ids, ELBOW_FLEXORS, ELBOW_EXTENSORS

        base, _ = build_supported(weld_humerus=welded,
                                  probe_radius=probe_radius,
                                  probe_stiffness=probe_stiffness, strict=False)
        if axis == "randprior":
            m = self._randomise(base, seed)
            self.reflex_kw = dict(gain_v=0.6 * REFLEX_GAIN_SCALE,
                                  gain_l=4.0 * REFLEX_GAIN_SCALE)
            rng = np.random.default_rng(seed)
            self.reflex_kw["gain_v"] *= float(np.exp(rng.normal(0, 1.0)))
            self.reflex_kw["gain_l"] *= float(np.exp(rng.normal(0, 1.0)))
        elif axis == "truth":
            m = base
            self.reflex_kw = reflex_kwargs("truth", 1.0)
        else:
            m = degraded_model(base, axis, factor)
            self.reflex_kw = reflex_kwargs(axis, factor)
        # A "true" model must carry the recipient's ACTUAL reflex.  Before
        # 2026-09-09 reflex_kwargs defaulted to the unscaled gains and the
        # truth model silently ran at half the real reflex gain.
        tk = reflex_kwargs("truth", 1.0)
        assert abs(tk["gain_v"] - 0.6 * REFLEX_GAIN_SCALE) < 1e-12 and             abs(tk["gain_l"] - 4.0 * REFLEX_GAIN_SCALE) < 1e-12,             "internal-model reflex gain does not match care_env's frozen gain"
        self.m = m
        self.d = mujoco.MjData(m)
        self.axis = axis
        self.factor = float(factor)
        self.mids = muscle_ids(m, ELBOW_FLEXORS) + muscle_ids(m, ELBOW_EXTENSORS)
        self.aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "probe_act")
        self.jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "probe_slide")
        self.j_adr = m.jnt_qposadr[self.jid]

    @staticmethod
    def _randomise(base, seed):
        """Knowledge-free body: every degradable parameter randomised.

        The phase-1 randprior control, ported.  It is NOT 'no model' - it is a
        model with the same machinery and no information, which is the only way
        to show the benefit comes from the information rather than the pipeline.
        """
        import copy
        import numpy as np
        import mujoco
        m = copy.deepcopy(base)
        rng = np.random.default_rng(seed)
        for i in range(m.nu):
            if m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE:
                s = float(np.exp(rng.normal(0, 1.0)))
                if m.actuator_gainprm[i, 2] > 0:
                    m.actuator_gainprm[i, 2] *= s
                if m.actuator_biasprm[i, 2] > 0:
                    m.actuator_biasprm[i, 2] *= s
                p = float(np.exp(rng.normal(0, 1.0)))
                m.actuator_gainprm[i, 7] *= p
                m.actuator_biasprm[i, 7] *= p
        m.dof_damping[:] = m.dof_damping * float(np.exp(rng.normal(0, 1.0)))
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm in ("humerus", "ulna", "radius"):
                s = float(np.exp(rng.normal(0, 0.7)))
                m.body_mass[b] *= s
                m.body_inertia[b] *= s
        return m

    # ---- prediction ----
    def start(self, settle_s=2.0, baseline_s=0.3):
        """Initialise the internal copy once, mirroring the real reset."""
        import mujoco
        import numpy as np
        from supported_scene import _relaxed_qpos
        from reflex import StretchReflex
        mujoco.mj_resetData(self.m, self.d)
        _relaxed_qpos(self.m, self.d)
        for _ in range(int(settle_s / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        self.rfx = StretchReflex(self.m, self.mids, **self.reflex_kw)
        try:
            self.rfx.reset(self.d, keep_act=True)
        except TypeError:
            self.rfx.reset(self.d)
        for _ in range(int(baseline_s / self.m.opt.timestep)):
            act = self.rfx.step(self.d)
            self.d.ctrl[self.mids] = act
            self.d.ctrl[self.aid] = 0.0
            mujoco.mj_step(self.m, self.d)
        self.base_act = np.array(self.rfx.act, dtype=float)
        self.base_len = np.array(self.d.actuator_length[self.mids], dtype=float)
        self._started = True

    def sync(self, obs, true_env=None):
        """Correct the running internal copy with what the robot can observe.

        The first version reset and re-settled the copy on EVERY control step,
        so each rollout started from an untouched limb and predicted only the
        incremental harm of the next fragment (0.002-0.048) rather than where
        the trajectory was heading (0.09-0.20).  A budget placed in the real
        harm range then never bound, and truth / wrong / random models produced
        identical behaviour.  The copy must run alongside the real system and be
        corrected by observation, not restarted.

        Only the joints the robot can actually see are written in, and they
        arrive with noise.  Everything else stays whatever the copy believes,
        which is the situation the paper is about.
        """
        import mujoco
        import numpy as np
        if not getattr(self, "_started", False):
            self.start()
        if true_env is not None:
            for k, (nm, adr, dof) in enumerate(true_env.obs_joints):
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, nm)
                if j < 0:
                    continue
                a = self.m.jnt_qposadr[j]
                self.d.qpos[a] = float(obs["other_kin"][2 * k])
        self.d.qpos[self.j_adr] = float(obs["probe_q"])
        mujoco.mj_forward(self.m, self.d)
        self._snapshot = (np.array(self.d.qpos), np.array(self.d.qvel),
                          np.array(self.rfx.act))

    def advance(self, cmd, substeps=10):
        """Step the internal copy with the command actually executed."""
        import mujoco
        for _ in range(substeps):
            act = self.rfx.step(self.d)
            self.d.ctrl[self.mids] = act
            self.d.ctrl[self.aid] = float(cmd)
            mujoco.mj_step(self.m, self.d)

    def rollout_harm(self, cmd_seq, horizon_substeps=10):
        """Predict peak harm for a command sequence. Returns (peak_reflex, peak_F)."""
        import mujoco
        import numpy as np
        from reflex import StretchReflex
        from supported_scene import force_by_partner

        qpos, qvel, act0 = self._snapshot
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)

        # Reuse the RUNNING reflex state, and score against the SAME baseline
        # the real environment uses (set once at reset), so a prediction is
        # comparable to the measurement it is predicting.
        rfx = StretchReflex(self.m, self.mids, **self.reflex_kw)
        rfx.rest_len = np.array(self.base_len)
        rfx.buf_l[:] = self.base_len
        rfx.buf_v[:] = 0.0
        rfx.act[:] = act0
        base_act = np.array(self.base_act, dtype=float)
        base_len = np.array(self.base_len, dtype=float)

        peak_r = 0.0
        peak_f = 0.0
        for cmd in cmd_seq:
            for _ in range(horizon_substeps):
                act = rfx.step(self.d)
                self.d.ctrl[self.mids] = act
                self.d.ctrl[self.aid] = float(cmd)
                mujoco.mj_step(self.m, self.d)
                dl = np.array(self.d.actuator_length[self.mids],
                              dtype=float) - base_len
                ln = dl > LENGTHEN_EPS
                if ln.any():
                    peak_r = max(peak_r, float(np.mean(rfx.act[ln]
                                                       - base_act[ln])))
                _, tot = force_by_partner(self.m, self.d)
                peak_f = max(peak_f, float(tot))
        # leave the copy where sync() put it; each candidate must be scored
        # from the same starting state
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)
        return peak_r, peak_f


class PredictiveController:
    """Sampling MPC over probe commands, scored by PREDICTED recipient harm.

    Objective per candidate: reach the task depth while keeping predicted
    reflex response below `harm_budget`.  Nothing here reads ground truth.
    """

    # Budget and horizon are NOT free parameters: diagnose_predictive.py
    # measured the attainable range first.  True peak reflex spans 0.0005
    # (cmd 0.01) to 0.183 (cmd 0.09); predicted harm spans 0.0016 to 0.203.
    # The first version used budget=0.03, which sat below almost every
    # prediction, so the constraint never bound and truth / 4x-wrong /
    # randomised models produced bit-identical behaviour.  0.10 sits mid-range
    # and binds at roughly cmd 0.05 (about 130 N, near the ISO forearm limit).
    # Horizon 10 (200 ms) already separates the models; 25 adds nothing.
    def __init__(self, model, target_depth=0.09, harm_budget=0.10,
                 n_candidates=7, horizon=10, max_rate=0.02):
        self.model = model
        self.target = float(target_depth)
        self.budget = float(harm_budget)
        self.n = int(n_candidates)
        self.horizon = int(horizon)
        self.max_rate = float(max_rate)
        self.env = None
        self.last_cmd = 0.0

    def bind(self, env):
        self.env = env
        return self

    def __call__(self, obs):
        import numpy as np
        q = float(obs["probe_q"])
        self.model.sync(obs, true_env=self.env)
        lo = q - self.max_rate
        hi = min(self.target, q + self.max_rate)
        cands = np.linspace(max(-0.01, lo), max(hi, lo + 1e-4), self.n)
        best, best_score = float(cands[0]), None
        for c in cands:
            # roll out the REMAINING trajectory under this command, not just a
            # fragment: hold c, then continue toward the target at max_rate.
            seq = [c] * 2
            nxt = c
            while len(seq) < self.horizon and nxt < self.target - 1e-6:
                nxt = min(self.target, nxt + self.max_rate)
                seq.append(nxt)
            while len(seq) < self.horizon:
                seq.append(nxt)
            pr, pf = self.model.rollout_harm(seq)
            over = max(0.0, pr - self.budget)
            progress = -abs(self.target - c)
            score = progress - 50.0 * over
            if best_score is None or score > best_score:
                best_score, best = score, float(c)
        self.last_cmd = best
        self.model.advance(best)
        return best


def _selftest():
    from care_env import CareContactEnv, audit_no_harm_access
    print("=== predictive controller self-test ===")
    env = CareContactEnv(welded=True)
    for axis, factor in (("truth", 1.0), ("muscle_force", 0.25),
                         ("randprior", 1.0)):
        mdl = OtherModel(axis=axis, factor=factor)
        ctrl = PredictiveController(mdl).bind(env)
        env.reset(seed=0)
        audit_no_harm_access(ctrl, env, n_steps=5)
        env.reset(seed=0)
        obs = env.observe()
        peak_r = peak_f = 0.0
        for _ in range(30):
            cmd = ctrl(obs)
            obs, info = env.step(cmd)
            peak_r = max(peak_r, info["step_peak"]["reflex"])
            peak_f = max(peak_f, info["step_peak"]["force"])
        print(f"  {axis:14s} f={factor:4.2f}  final_cmd={ctrl.last_cmd:.4f}  "
              f"true peak reflex={peak_r:.5f}  force={peak_f:6.1f} N  "
              f"(harm audit passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
