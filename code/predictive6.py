"""Predictive gentleness in a 6-DoF action space: plan the DIRECTION of
contact against a model of the other body.

Everything conceptual is inherited from predictive.py (the 1-DoF version):
the robot carries its own, possibly wrong, MuJoCo copy of the recipient,
runs it alongside the real system corrected only by what it can observe, and
scores candidate commands by the harm the copy PREDICTS.  What is new is the
candidate set.  On one axis the only choices were "how far" and "how fast";
here the controller also chooses WHERE to push - straight down, obliquely so
the elbow yields, or sideways where the welded humerus cannot yield - and the
care_env6 self-test (2026-09-09) showed those differ by 8.4x in reflex and
3.1x in force, with the two harm channels disagreeing on which direction is
gentle.  That is the freedom a wrong model can misuse.

Two predicted budgets, not one.  The predictive arm must keep the predicted
reflex response under `harm_budget` (the 1-DoF criterion) AND the predicted
contact force under `force_budget` (the ISO forearm limit that the constraint
baseline enforces reactively).  Both arms therefore answer to the same force
limit; the difference is that one predicts it and the other reacts to it.
Without the second budget a reflex-only planner would, with a CORRECT model,
choose the lateral push (139 N at reflex 0.009): a real finding about the
metric, but it would make the true-model arm exceed the standard and hide the
model-fidelity question the experiment is about.

Task.  The nominal task is the 1-DoF one - press the forearm from above to a
commanded depth of 0.09 m - but success is measured on the RECIPIENT: the
elbow must extend by `task_deg` degrees, which is what that press does to the
limb.  A controller that slides along the surface or loads the joint
laterally makes no progress, so the task cannot be gamed by moving the probe
without moving the limb.  Both arms observe the (noisy) elbow angle through
the same boundary.

Nothing here reads ground-truth harm; care_env6.audit_no_harm_access enforces
that mechanically.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

REFLEX_GAIN_SCALE = 2.0
LENGTHEN_EPS = 1e-4


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


class OtherModel6:
    """The robot's private, possibly-wrong copy of the recipient (6-DoF scene)."""

    def __init__(self, axis="truth", factor=1.0, seed=0, probe_radius=0.025,
                 probe_half_len=0.05, tissue="nominal"):
        """tissue: the copy's soft-tissue map (tissue.py).  "nominal" (the
        population map), "individual" (this person's map), ("scale", f)
        (every thickness x f), ("shift", mm) (landmarks displaced), or
        "uniform" (no map: t_mid everywhere).  The mechanical axes below are
        unchanged; the map is a second, independent thing the copy can have
        wrong."""
        import mujoco
        import numpy as np
        from care_env6 import build6, SLIDES, HINGES, axis_to_rot
        from degrade import degraded_model, reflex_kwargs
        from arm_scene import muscle_ids, ELBOW_FLEXORS, ELBOW_EXTENSORS
        from predictive import OtherModel
        import tissue as _tissue

        self.tissue_spec = tissue
        nom = _tissue.nominal_params()
        if tissue == "nominal":
            self.tissue = nom
        elif tissue == "individual":
            self.tissue = _tissue.individual_params(seed, **_tissue.spread_from_env())
        elif tissue == "uniform":
            self.tissue = _tissue.uniform(nom)
        elif isinstance(tissue, (tuple, list)) and tissue[0] == "scale":
            self.tissue = _tissue.scaled(nom, tissue[1])
        elif isinstance(tissue, (tuple, list)) and tissue[0] == "shift":
            self.tissue = _tissue.shifted(nom, tissue[1])
        else:
            raise ValueError(f"unknown tissue spec {tissue!r}")
        self.mapper = None

        base, meta = build6(probe_radius=probe_radius,
                            probe_half_len=probe_half_len, strict=False)
        if axis == "randprior":
            m = OtherModel._randomise(base, seed)
            self.reflex_kw = dict(gain_v=0.6 * REFLEX_GAIN_SCALE,
                                  gain_l=4.0 * REFLEX_GAIN_SCALE)
            rng = np.random.default_rng(seed)
            self.reflex_kw["gain_v"] *= float(np.exp(rng.normal(0, 1.0)))
            self.reflex_kw["gain_l"] *= float(np.exp(rng.normal(0, 1.0)))
        elif axis == "truth":
            m = base
            self.reflex_kw = reflex_kwargs("truth", 1.0)
        elif axis == "robotonly":
            # No model of the OTHER body: a passive arm with no active muscle
            # force and no reflex (passive tissue stiffness kept).  Its
            # predicted reflex rise is identically zero, so only the force
            # constraint can ever bind.  This is the "predicts force but knows
            # nothing about the person" control the council review asked for
            # (2026-09-13 §2.2): it separates force prediction from the body
            # model, which the knowledge-free copy (a WRONG model) cannot.
            m = degraded_model(base, "muscle_force", 1e-6)
            self.reflex_kw = reflex_kwargs("reflex_gain", 1e-6)
        elif axis == "individual":
            # The genuinely true model: the copy carries THIS seed's body
            # variation, not the nominal body.  "truth" elsewhere means the
            # nominal body, which differs from the individual by sigma=0.25;
            # this arm separates that error from the fidelity axes.
            import copy as _copy
            from care_env import CareContactEnv
            m = _copy.deepcopy(base)
            shim = type("S", (), {})()
            shim.m, shim._pristine = m, None
            shim.body_variation = float(os.environ.get("WM_BODY_VAR", 0.25))
            CareContactEnv._apply_body_variation(shim, seed)
            self.reflex_kw = reflex_kwargs("truth", 1.0)
        else:
            m = degraded_model(base, axis, factor)
            self.reflex_kw = reflex_kwargs(axis, factor)
        tk = reflex_kwargs("truth", 1.0)
        assert abs(tk["gain_v"] - 0.6 * REFLEX_GAIN_SCALE) < 1e-12, \
            "internal-model reflex gain does not match care_env's frozen gain"
        self.m = m
        self.d = mujoco.MjData(m)
        self.meta = meta
        self.axis = axis
        self.factor = float(factor)
        self.mids = muscle_ids(m, ELBOW_FLEXORS) + muscle_ids(m, ELBOW_EXTENSORS)
        names = [nm for nm, _ in SLIDES + HINGES]
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jids])
        self.aids = np.array([mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act") for n in names])
        self.pgeom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip")
        self.base = np.array(meta["probe_start"])
        ej = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        self.elbow_adr = int(m.jnt_qposadr[ej])
        self._started = False
        # The copy can diverge too (a randomised body can be unstable under
        # the same servo).  MuJoCo resets state silently on mjWARN_BADQACC,
        # so count it: a prediction from a reset copy is not a prediction.
        self.diverged_count = 0

    def _check_div(self):
        import mujoco
        w = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        if w > self._warn0:
            self.diverged_count += w - self._warn0
            self._warn0 = w

    # ---- prediction ----
    def start(self, home_pos, home_rot, settle_s=2.0, baseline_s=0.3):
        """Initialise the copy once, mirroring the real reset at the same
        probe pose (the robot knows where it parked its own end-effector)."""
        import mujoco
        import numpy as np
        from supported_scene import _relaxed_qpos
        from reflex import StretchReflex
        mujoco.mj_resetData(self.m, self.d)
        _relaxed_qpos(self.m, self.d)
        # count solver resets from the copy's own settling too (a randomised
        # body can be unstable before the robot has done anything)
        self._warn0 = int(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        q = np.concatenate([np.asarray(home_pos) - self.base,
                            np.asarray(home_rot)])
        self.d.qpos[self.qadr] = q
        self.d.ctrl[self.aids] = q
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
            mujoco.mj_step(self.m, self.d)
        self.base_act = np.array(self.rfx.act, dtype=float)
        self.base_len = np.array(self.d.actuator_length[self.mids], dtype=float)
        self.elbow0 = float(self.d.qpos[self.elbow_adr])
        if getattr(self, "elbow_ref_obs", None) is not None:
            # round 8e: progress is measured from the OBSERVED start angle,
            # not from where this copy happens to settle (a limp or
            # re-parameterised copy settles elsewhere; a copy started with the
            # probe under the limb could be reset by the solver)
            self.elbow0 = float(self.elbow_ref_obs)
        # the copy's own contact->(u, phi) mapper and body classes, built on
        # the copy's settled limb (the frame vectors ride with the bone)
        import tissue as _tissue
        from care_env6 import ARM_BODIES
        rid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
        ulna = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ulna")
        self.mapper = _tissue.ContactMapper(self.m, self.d, rid, np.array(self.meta["forearm_axis"]),
                                            np.array(self.meta["flex_normal"]), ulna)
        self.classes = _tissue.body_classes(self.m, arm_bodies=ARM_BODIES)
        self._check_div()
        self._started = True

    def tissue_load(self):
        """Predicted bone-proximity load of the copy's current contacts."""
        import tissue as _tissue
        return _tissue.load_index(self.m, self.d, self.pgeom, self.mapper,
                                  self.tissue, self.classes)["p"]

    def sync(self, obs, true_env):
        """Correct the running copy with what the robot can observe: its own
        end-effector pose and the recipient's noisy joint angles."""
        import mujoco
        import numpy as np
        if not self._started:
            self.start(true_env.home, true_env.home_rot)
        for k, (nm, adr, dof) in enumerate(true_env.obs_joints):
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if j < 0:
                continue
            self.d.qpos[self.m.jnt_qposadr[j]] = float(obs["other_kin"][2 * k])
        self.d.qpos[self.qadr] = np.concatenate(
            [np.asarray(obs["probe_pos"]) - self.base, obs["probe_rot"]])
        mujoco.mj_forward(self.m, self.d)
        if getattr(self, "rebaseline_first_sync", False) and not getattr(self, "_rebased", False):
            # round 8e: reflex baseline lengths at the observed start posture
            self.base_len = np.array(self.d.actuator_length[self.mids], dtype=float)
            self._rebased = True
        self._snapshot = (np.array(self.d.qpos), np.array(self.d.qvel),
                          np.array(self.rfx.act))

    def advance(self, cmd6, substeps=10):
        """Step the copy with the command actually executed."""
        import mujoco
        for _ in range(substeps):
            act = self.rfx.step(self.d)
            self.d.ctrl[self.mids] = act
            self.d.ctrl[self.aids] = cmd6
            mujoco.mj_step(self.m, self.d)
        self._check_div()

    def rollout(self, cmd_pos, rot, step, task, sign, max_rate, horizon=10,
                hold=2, horizon_substeps=10, probe_pose=None, want_p=True):
        """Predict (peak_reflex, peak_force, elbow_change) for: apply `step`
        every control step until the copy's elbow change reaches `task`
        (at least `hold` steps), then hold.

        This is the policy the controller will actually follow, so the peak is
        the peak of the remaining trajectory, not of an unbounded plunge.  The
        first version continued at max rate for the whole horizon (0.16 m) and
        predicted every candidate over budget; the true-model arm then held
        still at reflex 0.02 against a budget of 0.06.  Copy is restored
        afterwards."""
        import mujoco
        import numpy as np
        from reflex import StretchReflex
        from care_env6 import probe_force

        qpos, qvel, act0 = self._snapshot
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        if probe_pose is not None:
            # site selection: the probe teleports (in the air) to a candidate
            # parking pose before the plan starts
            self.d.qpos[self.qadr] = np.concatenate(
                [np.asarray(probe_pose[0], float) - self.base, probe_pose[1]])
            self.d.qvel[self.dadr] = 0.0
        mujoco.mj_forward(self.m, self.d)
        rfx = StretchReflex(self.m, self.mids, **self.reflex_kw)
        rfx.rest_len = np.array(self.base_len)
        rfx.buf_l[:] = self.base_len
        rfx.buf_v[:] = 0.0
        rfx.act[:] = act0
        peak_r = peak_f = peak_p = 0.0
        p = np.array(cmd_pos, dtype=float)
        moving = np.linalg.norm(step) > 1e-9
        for k in range(horizon):
            prog = sign * (float(self.d.qpos[self.elbow_adr]) - self.elbow0)
            # the plan is "this step, at this rate, until the task is met":
            # continuing at MAX rate (second version) made every plan's peak
            # the peak of its max-rate tail, so rate was never a real choice
            # and the true-model arm held at reflex 0.02 against budget 0.06
            if k < hold or (moving and prog < task):
                p = p + step
            self.d.ctrl[self.aids] = np.concatenate([p - self.base, rot])
            for _ in range(horizon_substeps):
                act = rfx.step(self.d)
                self.d.ctrl[self.mids] = act
                mujoco.mj_step(self.m, self.d)
                dl = np.array(self.d.actuator_length[self.mids],
                              dtype=float) - self.base_len
                ln = dl > LENGTHEN_EPS
                if ln.any():
                    peak_r = max(peak_r, float(np.mean(rfx.act[ln]
                                                       - self.base_act[ln])))
                _, tot = probe_force(self.m, self.d, self.pgeom)
                peak_f = max(peak_f, float(tot))
                if want_p and self.mapper is not None:
                    peak_p = max(peak_p, self.tissue_load())
        d_elbow = float(self.d.qpos[self.elbow_adr]) - self.elbow0
        self.last_peak_p = peak_p
        self._check_div()
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)
        return peak_r, peak_f, d_elbow


def candidate_directions(env, tilts=(0.6, 1.5)):
    """Unit push directions: straight down plus down tilted toward +-flexion
    normal and +-hinge axis.  Down is the nominal (1-DoF) direction; the
    tilts are how a planner can trade limb yielding against joint loading."""
    import numpy as np
    down = getattr(env, "press_dir", env.down)
    dirs = [("down", down)]
    for nm, v in (("flex", env.flex_normal), ("ext", -env.flex_normal),
                  ("lat+", env.hinge), ("lat-", -env.hinge)):
        for w in tilts:
            dirs.append((f"{nm}{w:g}", _unit(down + w * v)))
    return dirs


class PredictiveController6:
    """Sampling MPC over 3-D probe displacement, scored by PREDICTED harm.

    Each control step: for every candidate (direction, rate) roll the copy
    forward holding that motion, then continuing it at max rate for the rest
    of the horizon; keep the candidate that makes the most predicted task
    progress while respecting both predicted budgets.  Orientation is held at
    home (capsule along the bone) in this round.
    """

    def __init__(self, model, task_deg=15.0, task_sign=-1.0, harm_budget=0.06,
                 force_budget=160.0, horizon=10, max_rate=0.02,
                 rates=(0.25, 0.5, 1.0), penalty=50.0, orient=False,
                 plan_fractions=(1.0,), time_cost=0.0, p_budget=float("inf"),
                 p_cost=0.0, p_in_step=True, elbow_ref="copy", site_select=False,
                 site_offsets=(-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12),
                 site_phis=(-25.0, 0.0, 25.0)):
        """orient=True lets the planner also choose the capsule orientation
        (along the bone vs across it) on its first call, against the copy;
        the choice is then held for the episode.  Re-orienting mid-contact
        would be a violent motion, so it is a choice of approach, not a
        per-step action."""
        import numpy as np
        self.orient = bool(orient)
        self.chosen_orient = "along"
        # plan_fractions: each candidate plan pushes until the copy's progress
        # reaches fraction*task, then holds.  (1.0,) is the round-1 planner;
        # (1/3, 2/3, 1) gives the robot the "stop part way" option that the
        # 1-DoF DepthPlanController turned out to need (PHASE0_LOG §10.12).
        # time_cost (rad per control step) breaks ties toward faster plans.
        self.fractions = tuple(plan_fractions)
        self.time_cost = float(time_cost)
        self.model = model
        self.task = float(np.radians(task_deg))
        self.sign = float(task_sign)
        self.budget = float(harm_budget)
        self.f_budget = float(force_budget)
        # bone-proximity load budget (tissue.py); inf = the v0.3 planner
        self.p_budget = float(p_budget)
        # tissue load as a COST (rad of task progress per unit P): the planner
        # then minimises predicted load while completing the task, instead of
        # refusing thin individuals outright (a hard budget did that)
        self.p_cost = float(p_cost)
        # p_in_step=False: the tissue term only chooses WHERE to touch (site
        # selection); per-step planning keeps progress + reflex + force.  With
        # the term in every step, copies that believe the tissue is thin
        # preferred holding to lifting (round 8 closed loop, 2026-09-16)
        self.p_in_step = bool(p_in_step)
        self.elbow_ref = str(elbow_ref)
        self.site_select = bool(site_select)
        self.site_offsets = tuple(site_offsets)
        self.site_phis = tuple(site_phis)
        self.chosen_site = dict(offset=0.0, phi=0.0)
        self.horizon = int(horizon)
        self.max_rate = float(max_rate)
        self.rates = tuple(rates)
        self.penalty = float(penalty)
        self.env = None
        self.cmd_pos = None
        self.last_choice = None
        self.n_bind = 0
        self.n_calls = 0

    def bind(self, env):
        import numpy as np
        self.env = env
        self.cmd_pos = np.array(env.home)
        self.rot = np.array(env.home_rot)
        self.dirs = candidate_directions(env)
        return self

    def _cmd6(self, pos):
        import numpy as np
        return np.concatenate([pos - self.model.base, self.rot])

    def __call__(self, obs):
        import numpy as np
        self.n_calls += 1
        self.model.sync(obs, true_env=self.env)
        # observed task progress (noisy elbow angle, same boundary as CBF)
        prog_obs = self.sign * (float(obs["other_kin"][0]) - self._elbow_obs0)
        if prog_obs >= self.task:
            self.last_choice = "hold(done)"
            self.model.advance(self._cmd6(self.cmd_pos))
            return self.cmd_pos, self.rot

        cands = [("hold", np.zeros(3))]
        cands.append(("retreat", -getattr(self.env, "press_dir", self.env.down) * self.max_rate))
        for nm, dvec in self.dirs:
            for r in self.rates:
                cands.append((f"{nm}@{r:g}", dvec * self.max_rate * r))
        rots = [("along", self.rot)]
        if self.orient and self.n_calls == 1:
            from care_env6 import axis_to_rot, capsule_axis_for
            rots.append(("across", axis_to_rot(
                capsule_axis_for(self.env, self.env.down, "across"))))
        best, best_score, best_lab, best_rot = None, None, None, None
        best_pred = (0.0, 0.0, 0.0)
        # for the binding diagnosis: the plan with most predicted progress,
        # ignoring the constraints, and whether it violated any
        free_best, free_prog, free_over = None, None, (0.0, 0.0, 0.0)
        for olab, rot in rots:
            for lab, step in cands:
                for frac in self.fractions:
                    pr, pf, de = self.model.rollout(
                        self.cmd_pos, rot, step, self.task * frac, self.sign,
                        self.max_rate, self.horizon, want_p=self.p_in_step)
                    pp = float(getattr(self.model, "last_peak_p", 0.0)) if self.p_in_step else 0.0
                    progress = -abs(self.task - min(self.task, self.sign * de))
                    over_r = max(0.0, pr - self.budget)
                    over_f = max(0.0, (pf - self.f_budget) / self.f_budget)
                    over_p = (max(0.0, (pp - self.p_budget) / self.p_budget)
                              if np.isfinite(self.p_budget) else 0.0)
                    n_move = (0 if np.linalg.norm(step) < 1e-9 else
                              self.horizon)
                    score = (progress - self.penalty * (over_r + over_f + over_p)
                             - self.p_cost * pp
                             - self.time_cost * n_move * (1.0 - frac))
                    if best_score is None or score > best_score:
                        best, best_score = step, score
                        best_lab = lab if frac >= 1.0 else f"{lab}~{frac:.2f}"
                        best_rot = (olab, rot)
                        best_pred = (pr, pf, pp)
                    fp = progress - self.time_cost * n_move * (1.0 - frac)
                    if free_prog is None or fp > free_prog:
                        free_best, free_prog, free_over = lab, fp, (over_r, over_f, over_p)
        if len(rots) > 1:
            self.chosen_orient, self.rot = best_rot[0], np.array(best_rot[1])
        if best_lab != "down@1":
            self.n_bind += 1
        # diagnostics for the paper: what the executed plan predicted, and
        # which constraint (if any) kept the planner off its unconstrained
        # first choice this step
        self.log_pred_reflex.append(float(best_pred[0]))
        self.log_pred_force.append(float(best_pred[1]))
        self.log_pred_p.append(float(best_pred[2]))
        if free_best != best_lab and any(o > 0 for o in free_over):
            n_over = sum(o > 0 for o in free_over)
            if n_over > 1:
                self.n_bind_both += 1
            elif free_over[0] > 0:
                self.n_bind_reflex += 1
            elif free_over[1] > 0:
                self.n_bind_force += 1
            else:
                self.n_bind_tissue += 1
        self.last_choice = best_lab
        self.cmd_pos = self.cmd_pos + best
        self.model.advance(self._cmd6(self.cmd_pos))
        return self.cmd_pos, self.rot

    def start_episode(self, obs):
        """Call after env.reset(): home is per-episode (placed above the
        limb as this body settled)."""
        import numpy as np
        self._elbow_obs0 = float(obs["other_kin"][0])
        if self.elbow_ref == "observed":
            self.model.elbow_ref_obs = self._elbow_obs0
            self.model.rebaseline_first_sync = True
        self.cmd_pos = np.array(self.env.home)
        self.rot = np.array(self.env.home_rot)
        self.log_pred_reflex, self.log_pred_force, self.log_pred_p = [], [], []
        self.n_bind_reflex = self.n_bind_force = self.n_bind_both = 0
        self.n_bind_tissue = 0
        self.chosen_site = dict(offset=0.0, phi=0.0)
        self.site_scores = None
        if self.site_select:
            self.choose_site(obs)

    def choose_site(self, obs):
        """Where to touch: before any contact, roll the nominal press out in
        the copy from each candidate parking pose (axial offset along the
        bone x approach angle around it) and park the real probe at the pose
        with the best predicted (progress, constraints) score.  The candidate
        poses are computed on the limb as the robot sees it (its geometry, not
        its state); the copy supplies the predicted response, force and
        tissue load.  Reactive controllers press at the default site."""
        import numpy as np
        from care_env6 import HOME_CLEARANCE, AIM_DISTAL
        self.model.sync(obs, true_env=self.env)
        best = None
        scores = []
        for off in self.site_offsets:
            for phi in self.site_phis:
                d = self.env.site_direction(phi)
                pos, rot = self.env.approach_start(d, self.env.axis, clearance=HOME_CLEARANCE,
                                                   aim_shift=AIM_DISTAL + off)
                # evaluate each site with the gentle approach the planner will
                # actually use (slowest rate); at 0.5 the predicted reflex sat
                # on its budget everywhere and the site ranking became
                # "least reflex" (= most distal) instead of least tissue load
                pr, pf, de = self.model.rollout(pos, rot, d * self.max_rate * self.rates[0], self.task,
                                                self.sign, self.max_rate, horizon=24,
                                                probe_pose=(pos, rot))
                pp = float(getattr(self.model, "last_peak_p", 0.0))
                progress = -abs(self.task - min(self.task, self.sign * de))
                over = (max(0.0, pr - self.budget) + max(0.0, (pf - self.f_budget) / self.f_budget)
                        + (max(0.0, (pp - self.p_budget) / self.p_budget) if np.isfinite(self.p_budget) else 0.0))
                score = progress - self.penalty * over - self.p_cost * pp
                scores.append(dict(offset=off, phi=phi, score=score, pred_p=pp, pred_r=pr, pred_f=pf,
                                   progress=float(self.sign * de)))
                if best is None or score > best[0]:
                    best = (score, off, phi)
        self.site_scores = scores
        # commit to the best-scored site the probe can actually park at: a
        # parking pose that already touches the limb is felt by the wrist
        # sensor, so the robot falls back to the next candidate (wide approach
        # angles cannot always park on every individual)
        for s in sorted(scores, key=lambda s: -s["score"]):
            if self.env.set_site(s["offset"], s["phi"]):
                self.chosen_site = dict(offset=float(s["offset"]), phi=float(s["phi"]))
                break
        self.cmd_pos = np.array(self.env.home)
        self.rot = np.array(self.env.home_rot)
        self.dirs = candidate_directions(self.env)
        self.model.sync(self.env.observe(), true_env=self.env)


def _selftest():
    import numpy as np
    from care_env6 import CareContactEnv6, audit_no_harm_access
    print("=== predictive6 self-test ===")
    env = CareContactEnv6()
    for axis, factor in (("truth", 1.0), ("passive_stiffness", 0.25),
                         ("randprior", 1.0)):
        mdl = OtherModel6(axis=axis, factor=factor)
        ctrl = PredictiveController6(mdl).bind(env)
        obs = env.reset(seed=0)
        ctrl.start_episode(obs)
        audit_no_harm_access(ctrl, env, n_steps=3)
        obs = env.reset(seed=0)
        ctrl = PredictiveController6(mdl).bind(env)
        ctrl.start_episode(obs)
        peak_r = peak_f = 0.0
        choices = []
        for _ in range(20):
            pos, rot = ctrl(obs)
            obs, info = env.step(pos, rot)
            peak_r = max(peak_r, info["step_peak"]["reflex"])
            peak_f = max(peak_f, info["step_peak"]["force"])
            choices.append(ctrl.last_choice)
        print(f"  {axis:18s} f={factor:4.2f}  reflex={peak_r:.5f}  "
              f"force={peak_f:6.1f} N  elbow={np.degrees(env.elbow_change()):+.1f}d  "
              f"choices={choices[:8]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
