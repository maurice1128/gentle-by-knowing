"""The centrepiece in the action space where it can actually be decided:
where does predictive gentleness stop beating constraint-based gentleness as
the model of the other body gets worse, when the robot can choose WHERE to
push?

Design is crossover.py's, ported to care_env6 (see that file's docstring for
the rationale of every arm).  What changes:

* action space : 3-D probe displacement per control step (orientation held
                 along the bone this round).  Both arms get it.
* task         : the nominal 1-DoF press (straight down, 0.09 m) measured on
                 the recipient: extend the elbow by TASK_DEG.  Sliding along
                 the surface or loading the joint sideways makes no progress.
* predictive   : plans direction AND rate against its internal copy, subject
                 to a predicted-reflex budget and the ISO force limit.
* constraint   : CBF on the measured force VECTOR with the QP closed form -
                 when the normal bound binds, the excess normal push is
                 removed and the tangential part of the nominal command is
                 kept.  It slides rather than brakes, which is what
                 distinguishes a barrier function from a threshold and is
                 why this experiment needs more than one axis.
* fidelity     : the three round-2 axes plus `rom`, the axis preregistered
                 for the "pushed past the joint limit" failure and never yet
                 swept.

Both arms observe exactly the same boundary (care_env6.observe) and neither
reads ground truth; audit_no_harm_access checks the predictive one.
"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), "out", "crossover6")
sys.path.insert(0, ROOT)

TASK_DEG = float(os.environ.get("WM_TASK_DEG", 15.0))
TASK_SIGN = -1.0          # elbow_flexion DEcreases when the forearm is pressed
HARM_BUDGET = float(os.environ.get("WM_HARM_BUDGET", 0.06))
# Common force limit for BOTH arms.  Not the ISO 160 N: under greedy the
# capsule's impact peak is 153 N regardless of task depth, so at 160 N the
# barrier never activates (bind rate 2%) and the "constraint arm" is the
# greedy arm.  100 N is where it is measurably active (bind 8%, peak 126 N,
# i.e. the same protects-steady-misses-peak behaviour the 1-DoF round found
# at 160/215).  ISO exceedance is still reported as an evaluation metric.
FORCE_BUDGET = float(os.environ.get("WM_FORCE_BUDGET", 100.0))
# Ablation: the predictive arm's force budget alone.  The no-crossover result
# in [1/16, 16] was traced to this budget (the under-estimation branch
# saturates at 96-99 N); setting it huge tests that explanation directly.
PRED_FORCE_BUDGET = float(os.environ.get("WM_PRED_FORCE_BUDGET", FORCE_BUDGET))
TAG = os.environ.get("WM_TAG", "")
ORIENT = os.environ.get("WM_ORIENT", "0") == "1"
PLAN = os.environ.get("WM_PLAN", "0") == "1"     # partial-progress plans
N_STEPS = 30
SEEDS = tuple(range(int(os.environ.get("WM_NSEEDS", 8))))
BODY_VARIATION = float(os.environ.get("WM_BODY_VAR", 0.25))
ISO_FOREARM_N = 160.0
SMOKE = os.environ.get("WM_SMOKE", "0") == "1"
# PREREGISTRATION.md §7.3: if no crossover inside [0.25, 4] (H2-F1), ONE
# symmetric extension to [1/16, 16] on every swept axis at once.  Triggered
# 2026-09-10 by round 1 (all harm CIs below zero).  Reference arms and the
# original conditions are reused from crossover6.json (same seeds), only the
# new factors are run, and the merged result goes to crossover6_ext.json.
EXTEND = os.environ.get("WM_EXTEND", "0") == "1"
EXT_FACTORS = (0.0625, 0.125, 8.0, 16.0)

AXES = {
    "muscle_force": (0.25, 0.5, 2.0, 4.0),
    "passive_stiffness": (0.25, 0.5, 2.0, 4.0),
    "reflex_gain": (0.25, 0.5, 2.0, 4.0),
    "rom": (0.5, 2.0, 4.0),
}
# WM_AXES="joint_damping,segment_mass,reflex_delay" restricts the sweep to
# those axes; WM_FACTORS="0.0625,0.125,0.25,0.5,2,4,8,16" overrides factors.
if os.environ.get("WM_AXES"):
    _f = tuple(float(x) for x in os.environ.get(
        "WM_FACTORS", "0.25,0.5,2,4").split(","))
    AXES = {a: _f for a in os.environ["WM_AXES"].split(",")}


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def _score(env, peak_r, peak_f, extra=None, t_done=None):
    """t_done: simulated time (s) at which the TRUE elbow first met the task,
    None if never - the time axis the council review asked for (a slow
    press can be gentler; the question is what it costs in time)."""
    import numpy as np
    task = float(np.radians(TASK_DEG))
    prog = TASK_SIGN * env.elbow_change()
    out = dict(peak_reflex=peak_r, peak_force=peak_f,
               elbow_deg=float(np.degrees(env.elbow_change())),
               reached=float(min(1.0, max(0.0, prog / task))),
               t_done=(None if t_done is None else float(t_done)),
               diverged=bool(env.diverged))
    if extra:
        out.update(extra)
    return out


def run_fixed6(env, seed, mode):
    import numpy as np
    obs = env.reset(seed=seed)
    e0 = float(obs["other_kin"][0])
    task = float(np.radians(TASK_DEG))
    cmd = np.array(env.home)
    peak_r = peak_f = 0.0
    t_done = None
    for _ in range(N_STEPS):
        prog = TASK_SIGN * (float(obs["other_kin"][0]) - e0)
        if mode == "greedy" and prog < task:
            cmd = cmd + env.down * 0.02
        obs, info = env.step(cmd, env.home_rot)
        peak_r = max(peak_r, info["step_peak"]["reflex"])
        peak_f = max(peak_f, info["step_peak"]["force"])
        if t_done is None and TASK_SIGN * env.elbow_change() >= task:
            t_done = env.t
    return _score(env, peak_r, peak_f, t_done=t_done)


def run_cbf6(env, seed, f_max=FORCE_BUDGET, alpha=2.0, k_prior=2400.0,
             lam=0.98, max_step=0.02):
    """Constraint arm: force barrier on the measured force vector, no model of
    the other body.  Same two bounds as crossover.run_cbf (rate and
    equilibrium), applied along the contact normal; the QP closed form keeps
    the tangential component of the nominal command."""
    import numpy as np
    obs = env.reset(seed=seed)
    e0 = float(obs["other_kin"][0])
    task = float(np.radians(TASK_DEG))
    cmd = np.array(env.home)
    prev_f = 0.0
    prev_push = 0.0
    g_hat = k_prior
    p_cov = 1.0
    peak_r = peak_f = 0.0
    n_bind = n_slide = 0
    t_done = None
    for _ in range(N_STEPS):
        f = float(obs["force"])
        fv = np.asarray(obs["force_vec"], dtype=float)
        prog = TASK_SIGN * (float(obs["other_kin"][0]) - e0)
        u = np.zeros(3)
        if prog < task:
            u_nom = env.down * max_step
            if f > 1.0 and np.linalg.norm(fv) > 1e-6:
                push_dir = -_unit(fv)              # into the limb
                pen = max(0.0, float(np.dot(cmd - obs["probe_pos"], push_dir)))
                if pen > 1e-4:
                    g_meas = f / pen
                    k_gain = p_cov / (lam + p_cov)
                    g_hat = g_hat + k_gain * (g_meas - g_hat)
                    p_cov = (p_cov - k_gain * p_cov) / lam
                slope = ((f - prev_f) / prev_push) if prev_push > 1e-6 else 0.0
                h = f_max - f
                push_nom = float(np.dot(u_nom, push_dir))
                bound = push_nom
                if slope > 1e-6:
                    bound = min(bound, max(0.0, alpha * h / slope))
                if g_hat > 1e-6:
                    bound = min(bound, max(0.0, h / g_hat))
                if bound < push_nom - 1e-9:
                    n_bind += 1
                u = u_nom + (bound - push_nom) * push_dir
                if np.linalg.norm(u - push_dir * np.dot(u, push_dir)) > 1e-6 \
                        and bound < push_nom - 1e-9:
                    n_slide += 1
                prev_push = float(np.dot(u, push_dir))
            else:
                u = u_nom
                prev_push = 0.0
        prev_f = f
        cmd = cmd + u
        obs, info = env.step(cmd, env.home_rot)
        peak_r = max(peak_r, info["step_peak"]["reflex"])
        peak_f = max(peak_f, info["step_peak"]["force"])
        if t_done is None and TASK_SIGN * env.elbow_change() >= task:
            t_done = env.t
    return _score(env, peak_r, peak_f,
                  dict(bind_rate=n_bind / N_STEPS, slide_rate=n_slide / N_STEPS,
                       g_hat=float(g_hat)), t_done=t_done)


def run_predictive6(env, axis, factor, seed):
    import numpy as np
    from predictive6 import OtherModel6, PredictiveController6
    from collections import Counter
    mdl = OtherModel6(axis=axis, factor=factor, seed=seed)
    ctrl = PredictiveController6(mdl, task_deg=TASK_DEG, task_sign=TASK_SIGN,
                                 harm_budget=HARM_BUDGET,
                                 force_budget=PRED_FORCE_BUDGET,
                                 orient=ORIENT,
                                 plan_fractions=((1 / 3, 2 / 3, 1.0) if PLAN
                                                 else (1.0,)),
                                 time_cost=(0.002 if PLAN else 0.0)).bind(env)
    obs = env.reset(seed=seed)
    ctrl.start_episode(obs)
    peak_r = peak_f = 0.0
    choices = Counter()
    t_done = None
    task = float(np.radians(TASK_DEG))
    for _ in range(N_STEPS):
        pos, rot = ctrl(obs)
        obs, info = env.step(pos, rot)
        peak_r = max(peak_r, info["step_peak"]["reflex"])
        peak_f = max(peak_f, info["step_peak"]["force"])
        choices[ctrl.last_choice] += 1
        if t_done is None and TASK_SIGN * env.elbow_change() >= task:
            t_done = env.t
    return _score(env, peak_r, peak_f,
                  dict(choices=dict(choices), bind_rate=ctrl.n_bind / N_STEPS,
                       model_diverged=int(mdl.diverged_count),
                       orient=ctrl.chosen_orient,
                       # what the executed plans predicted vs what happened
                       pred_peak_reflex=float(max(ctrl.log_pred_reflex or [0.0])),
                       pred_peak_force=float(max(ctrl.log_pred_force or [0.0])),
                       bind_reflex=ctrl.n_bind_reflex / N_STEPS,
                       bind_force=ctrl.n_bind_force / N_STEPS,
                       bind_both=ctrl.n_bind_both / N_STEPS), t_done=t_done)


def _summ(res):
    import numpy as np
    return (f"{np.mean([r['reached'] for r in res]):8.3f} "
            f"{np.mean([r['peak_reflex'] for r in res]):9.5f} "
            f"{np.mean([r['peak_force'] for r in res]):9.1f} "
            f"{np.mean([r['peak_force'] > ISO_FOREARM_N for r in res]):6.2f} "
            f"{np.mean([r['elbow_deg'] for r in res]):7.1f}")



# ---------------------------------------------------------------- parallel
_ENV = None


def _worker_env():
    """One environment per worker process, built lazily."""
    global _ENV
    if _ENV is None:
        from care_env6 import CareContactEnv6
        _ENV = CareContactEnv6(body_variation=BODY_VARIATION)
    return _ENV


def _job(spec):
    arm, axis, factor, seed = spec
    env = _worker_env()
    if arm in ("noop", "greedy"):
        r = run_fixed6(env, seed, arm)
    elif arm == "cbf":
        r = run_cbf6(env, seed)
    else:
        r = run_predictive6(env, axis, factor, seed)
    r.update(arm=arm, axis=axis, factor=factor, seed=seed,
             start_contact=list(env.start_contact),
             reset_diverged=int(env.reset_diverged))
    return r


def main():
    import numpy as np
    from multiprocessing import Pool
    from care_env6 import CareContactEnv6, audit_no_harm_access
    from predictive6 import OtherModel6, PredictiveController6

    os.makedirs(OUT, exist_ok=True)
    env = CareContactEnv6(body_variation=BODY_VARIATION)

    conditions = [("truth", 1.0), ("randprior", 1.0)]
    for ax, facs in AXES.items():
        conditions += [(ax, f) for f in facs]
    if SMOKE:
        conditions = [("truth", 1.0), ("passive_stiffness", 0.25),
                      ("reflex_gain", 0.25)]
    prior_rows = []
    if EXTEND:
        prior = json.load(open(os.path.join(OUT, "crossover6" + TAG + ".json")))
        prior_rows = prior["rows"]
        prior_conditions = conditions
        conditions = [(ax, f) for ax in AXES for f in EXT_FACTORS]
    n_proc = int(os.environ.get("WM_PROCS", 8))

    print("=== crossover experiment, 6-DoF ===")
    print(f"task: extend elbow {TASK_DEG} deg (nominal press 0.09 m from above); "
          f"reflex budget {HARM_BUDGET}, force budget {FORCE_BUDGET} N "
          f"(predictive arm: {PRED_FORCE_BUDGET} N), "
          f"{len(SEEDS)} paired seeds, body_variation {BODY_VARIATION}, "
          f"{n_proc} worker processes")
    print(f"ISO forearm quasi-static limit {ISO_FOREARM_N} N "
          f"(AIS-1 injury onset, not pain onset)\n")

    ctrl = PredictiveController6(OtherModel6("truth", 1.0)).bind(env)
    obs = env.reset(seed=0)
    ctrl.start_episode(obs)
    audit_no_harm_access(ctrl, env, n_steps=3)
    print("harm-access audit passed for the predictive controller\n")

    # Seed validity.  Excluded: a body whose settled posture leaves no
    # contact-free parking pose (seed 7: elbow 0.04 rad, forearm hanging
    # straight below the humerus, the same seed the 1-DoF round excluded),
    # a solver reset during settling, or no contact under greedy.
    valid = []
    for s in (tuple(prior["seeds"]) if EXTEND else SEEDS):
        if EXTEND:
            valid.append(s)
            continue
        env.reset(seed=s)
        if env.start_contact:
            print(f"  seed {s}: probe touches {env.start_contact} at home "
                  f"- EXCLUDED")
            continue
        if env.reset_diverged:
            print(f"  seed {s}: solver reset during settling - EXCLUDED")
            continue
        chk = run_fixed6(env, s, "greedy")
        if chk["peak_force"] <= 5.0:
            print(f"  seed {s} makes no contact even under greedy "
                  f"({chk['peak_force']:.2f} N) - EXCLUDED")
            continue
        valid.append(s)
    if len(valid) < 4 and not SMOKE:
        print(f"only {len(valid)} valid seeds; aborting")
        return 1
    seeds = tuple(valid)
    print(f"using {len(seeds)} valid seeds: {seeds}\n")

    jobs = []
    if not EXTEND:
        jobs = [(m, m, 1.0, s) for m in ("noop", "greedy") for s in seeds]
        jobs += [("cbf", "cbf", 1.0, s) for s in seeds]
    jobs += [("predictive", ax, f, s) for ax, f in conditions for s in seeds]
    rows = list(prior_rows)
    # Resume: a partial file from an interrupted run of the SAME tag holds
    # rows saved every 10 episodes; skip the jobs it already covers.
    part = os.path.join(OUT, "crossover6_partial" + TAG + ".json")
    if os.environ.get("WM_RESUME", "0") == "1" and os.path.exists(part):
        done_rows = json.load(open(part))["rows"]
        have = {(r["arm"], r["axis"], r["factor"], r["seed"]) for r in done_rows}
        keep = [r for r in done_rows
                if (r["arm"], r["axis"], r["factor"], r["seed"]) not in
                {(x["arm"], x["axis"], x["factor"], x["seed"]) for x in rows}]
        rows += keep
        jobs = [j for j in jobs if j not in have]
        print(f"resuming: {len(keep)} rows from {os.path.basename(part)}, "
              f"{len(jobs)} jobs left\n")
    t0 = time.time()
    with Pool(processes=n_proc) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if r["arm"] == "predictive":
                print(f"  [{k + 1}/{len(jobs)}] {r['axis']}@{r['factor']} seed {r['seed']}: "
                      f"reach={r['reached']:.3f} harm={r['peak_reflex']:.4f} "
                      f"F={r['peak_force']:.1f}  ({time.time() - t0:.0f}s)")
            if (k + 1) % 10 == 0:
                with open(os.path.join(OUT, "crossover6_partial" + TAG + ".json"), "w") as fh:
                    json.dump(dict(rows=rows), fh, indent=1)
    print(f"\nall {len(jobs)} episodes done in {time.time() - t0:.0f}s\n")
    if EXTEND:
        # report the merged sweep, original domain first, ordered by factor
        conditions = prior_conditions[:2] + sorted(
            [c for c in prior_conditions[2:]] + conditions,
            key=lambda c: (list(AXES).index(c[0]), c[1]))

    def group(arm, axis=None, factor=None):
        return [r for r in rows if r["arm"] == arm
                and (axis is None or r["axis"] == axis)
                and (factor is None or r["factor"] == factor)]

    hdr = f"{'arm':30s} {'reached':>8} {'harm':>9} {'force':>9} {'>ISO':>6} {'elbow':>7}"
    print(hdr)
    for mode in ("noop", "greedy"):
        print(f"{mode:30s} {_summ(group(mode))}")
    res = group("cbf")
    print(f"{'cbf (constraint)':30s} {_summ(res)}   "
          f"bind={np.mean([r['bind_rate'] for r in res]):.2f} "
          f"slide={np.mean([r['slide_rate'] for r in res]):.2f}")
    print()
    for axis, factor in conditions:
        res = group("predictive", axis, factor)
        top = {}
        for r in res:
            for k, v in r["choices"].items():
                top[k] = top.get(k, 0) + v
        top = sorted(top.items(), key=lambda kv: -kv[1])[:3]
        label = f"predictive {axis}@{factor}"
        print(f"{label:30s} {_summ(res)}   {top} "
              f"mdiv={sum(r['model_diverged'] for r in res)} "
              f"div={sum(r['diverged'] for r in res)}")

    print("\n--- predictive vs constraint, paired by seed ---")
    print(f"{'condition':30s} {'d(harm)':>10} {'95% CI':>22} {'d(force)':>9} "
          f"{'d(reach)':>9}")
    cbf_by_seed = {r["seed"]: r for r in rows if r["arm"] == "cbf"}
    analysis = {}
    for axis, factor in conditions:
        sub = sorted(group("predictive", axis, factor), key=lambda r: r["seed"])
        dh = np.array([r["peak_reflex"] - cbf_by_seed[r["seed"]]["peak_reflex"]
                       for r in sub])
        df = np.array([r["peak_force"] - cbf_by_seed[r["seed"]]["peak_force"]
                       for r in sub])
        dr = np.array([r["reached"] - cbf_by_seed[r["seed"]]["reached"]
                       for r in sub])
        n = len(dh)
        se = float(dh.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        # t(0.975, n-1).  The first version's table stopped at n = 8 and fell
        # back to 2.365 for larger n, which made the 19-seed intervals about
        # 12% too wide (found by audit_draft.py, 2026-09-11).
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
                 8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 15: 2.145,
                 19: 2.101, 20: 2.093, 22: 2.080, 30: 2.045}.get(
                     n, 1.96 + 2.4 / max(n - 1, 1))
        ci = (float(dh.mean() - tcrit * se), float(dh.mean() + tcrit * se))
        analysis[f"{axis}@{factor}"] = dict(d_harm=float(dh.mean()), ci=ci,
                                            d_force=float(df.mean()),
                                            d_reach=float(dr.mean()))
        flag = "  <- predictive WORSE" if ci[0] > 0 else (
            "  <- predictive better" if ci[1] < 0 else "")
        print(f"predictive {axis}@{factor:<8} {dh.mean():10.5f} "
              f"[{ci[0]:+9.5f},{ci[1]:+9.5f}] {df.mean():9.1f} {dr.mean():9.3f}"
              f"{flag}")

    name = ("crossover6_ext" if EXTEND else "crossover6") + TAG + ".json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(dict(rows=rows, analysis=analysis, task_deg=TASK_DEG,
                       budget=HARM_BUDGET, force_budget=FORCE_BUDGET,
                       seeds=list(seeds), extended=EXTEND), fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
