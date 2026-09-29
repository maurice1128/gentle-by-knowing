"""Main experiment, round 8 (lift task, closed loop), 2026-09-16.

Task: support the forearm from below and lift it, elbow flexion +15 deg, then
hold for 0.5 s.  Gentleness = tissue load index P (tissue.py).  The wrist
force budget (100 N) and the reflex constraint (0.06) stay as constraints.

Preregistration section 10, round 8c:
  H8-1 (primary)  population-map copy P < no-map copy P, task-matched, paired
                  95% CI excluding 0.  In the press task the force budget kept
                  a no-map planner off thin tissue (thin sites needed MORE
                  force); in the lift task thin sites need LESS force, so the
                  prediction is that the budget does not rescue the no-map copy.
  H8-2            population map vs individual copy: reported (expected small).
  H8-3            reactive lifts at the default site (0.10, 0.25 m/s) and the
                  force barrier: P higher than the population-map copy.

Controllers (same 19 individuals):
  press@0.10, press@0.25  open-loop lift at the default site (below, mid-shaft)
  cbf@0.25                the same with the 50 Hz force barrier (100 N)
  rule@0.10               open-loop lift at the fixed site (+6 cm, -20 deg)
                          that the population oracle picked (leave-one-out,
                          19/19) - a compiled population prior, labelled so
  predictive copies (site selection over 8 offsets x 5 angles, then planning):
    popmap      population body + population map
    nomap       population body + no map (18 mm everywhere, hand included)
    individual  this person's body + this person's map
    indmap      population body + this person's map
    nosite      population body + population map, no site selection
    robotonly   passive arm + population map
    scale@0.5, scale@2, shift@20   population map wrong
Output: out/crossover8/crossover8<TAG>.json
"""
import os
import sys
import json
import math
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "crossover8")
sys.path.insert(0, ROOT)

import crossover6 as c6

SEEDS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21]
TASK_DEG = 15.0
TASK_SIGN = 1.0
HOLD_S = 0.5
T_MAX = 3.0                 # open-loop lifts
N_PLAN_MAX = 75             # closed loop: 1.5 s including the hold
P_COST = float(os.environ.get("WM_P_COST", 0.001))
P_IN_STEP = os.environ.get("WM_P_IN_STEP", "0") == "1"   # round 8d: site choice only
ELBOW_REF = os.environ.get("WM_ELBOW_REF", "observed")     # round 8e
TAG = os.environ.get("WM_TAG", "")
SMOKE = os.environ.get("WM_SMOKE", "0") == "1"
SITE_OFFSETS = (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
SITE_ANGLES = (-40.0, -20.0, 0.0, 20.0, 40.0)
RULE_SITE = (0.06, -20.0)
TRAINED_SITE = (0.03, -20.0)   # round 9 pi_med@0.25 (trained on seeds 100-160)
WITH_TRAINED = os.environ.get("WM_TRAINED_ARM", "0") == "1"

CONDITIONS = [  # (label, axis, factor, tissue, site_select)
    ("popmap", "truth", 1.0, "nominal", True),
    ("nomap", "truth", 1.0, "uniform", True),
    ("individual", "individual", 1.0, "individual", True),
    ("indmap", "truth", 1.0, "individual", True),
    ("nosite", "truth", 1.0, "nominal", False),
    ("robotonly", "robotonly", 1.0, "nominal", True),
    ("scale@0.5", "truth", 1.0, ("scale", 0.5), True),
    ("scale@2", "truth", 1.0, ("scale", 2.0), True),
    ("shift@20", "truth", 1.0, ("shift", 20.0), True),
]
if os.environ.get("WM_CONDS"):
    keep = set(os.environ["WM_CONDS"].split(","))
    CONDITIONS = [c for c in CONDITIONS if c[0] in keep]

_ENV = None


def env():
    global _ENV
    if _ENV is None:
        from care_env6 import CareContactEnv6
        _ENV = CareContactEnv6(body_variation=c6.BODY_VARIATION)
        _ENV.site_mode = "lift"
    return _ENV


def _contacts(e, state):
    """Keep the contacts of the step end with the highest tissue load, the
    same rule the open-loop oracle uses (round 10C)."""
    from diag_lift_oracle import contacts_now
    L = e.tissue_load()
    if L["p"] > state["best_end_p"]:
        state["best_end_p"] = L["p"]
        state["contacts"] = contacts_now(e)


def _acc(peak, info, tmin):
    for k in ("p", "pb", "sb", "reflex", "force"):
        peak[k] = max(peak[k], info["step_peak"][k])
    t = info["step_peak"]["t_min"]
    return min(tmin, t) if t > 0 else tmin


def _finish(e, peak, tmin, t_done, extra):
    task = math.radians(TASK_DEG)
    prog = TASK_SIGN * e.elbow_change()
    out = dict(peak_p=peak["p"], peak_pb=peak["pb"], peak_sb=peak["sb"],
               contacts=extra.pop("_contacts", []), step_end_p=extra.pop("_step_end_p", 0.0),
               t_min=(tmin if tmin < 1e8 else 0.0), peak_reflex=peak["reflex"],
               peak_force=peak["force"], lifted_deg=math.degrees(e.elbow_change()),
               reached=float(min(1.0, max(0.0, prog / task))),
               t_done=(None if t_done is None else float(t_done)),
               diverged=bool(e.diverged), site=dict(e.site))
    out.update(extra)
    return out


def run_press(e, seed, speed, barrier=False, site=(0.0, 0.0), hz=50, f_max=None):
    import numpy as np
    f_max = c6.FORCE_BUDGET if f_max is None else f_max
    e.reset(seed=seed)
    site_ok = e.set_site(*site)
    obs = e.observe()
    e0_obs = float(obs["other_kin"][0])
    task = math.radians(TASK_DEG)
    step = speed / hz
    cmd = np.array(e.home)
    peak = dict(p=0.0, pb=0.0, sb=0.0, reflex=0.0, force=0.0)
    tmin = 1e9
    t_done = None
    n_hold = 0
    cstate = dict(best_end_p=0.0, contacts=[])
    prev_f = prev_push = 0.0
    g_hat, p_cov = 2400.0, 1.0
    n_bind = 0
    for _ in range(int(T_MAX * hz)):
        prog = TASK_SIGN * (float(obs["other_kin"][0]) - e0_obs)
        u = np.zeros(3)
        if t_done is None and prog < task:
            u_nom = e.press_dir * step
            f = float(obs["force"]); fv = np.asarray(obs["force_vec"], float)
            if barrier and f > 1.0 and np.linalg.norm(fv) > 1e-6:
                push_dir = -c6._unit(fv)
                pen = max(0.0, float(np.dot(cmd - obs["probe_pos"], push_dir)))
                if pen > 1e-4:
                    g_meas = f / pen
                    k_gain = p_cov / (0.98 + p_cov)
                    g_hat = g_hat + k_gain * (g_meas - g_hat)
                    p_cov = (p_cov - k_gain * p_cov) / 0.98
                slope = ((f - prev_f) / prev_push) if prev_push > 1e-6 else 0.0
                h = f_max - f
                push_nom = float(np.dot(u_nom, push_dir))
                bound = push_nom
                if slope > 1e-6:
                    bound = min(bound, max(0.0, 2.0 * h / slope))
                if g_hat > 1e-6:
                    bound = min(bound, max(0.0, h / g_hat))
                if bound < push_nom - 1e-9:
                    n_bind += 1
                u = u_nom + (bound - push_nom) * push_dir
                prev_push = float(np.dot(u, push_dir))
            else:
                u = u_nom; prev_push = 0.0
            prev_f = f
        cmd = cmd + u
        obs, info = e.step(cmd, e.home_rot)
        tmin = _acc(peak, info, tmin)
        _contacts(e, cstate)
        if t_done is None and TASK_SIGN * e.elbow_change() >= task:
            t_done = e.t
        if t_done is not None:
            n_hold += 1
            if n_hold >= int(HOLD_S * hz):
                break
    return _finish(e, peak, tmin, t_done, dict(speed=speed, barrier=barrier, site_ok=bool(site_ok),
                                               bind_rate=n_bind / int(T_MAX * hz),
                                               _contacts=cstate["contacts"], _step_end_p=cstate["best_end_p"]))


def run_predictive(e, seed, axis, factor, tissue, site_select, rates=None, n_max=None):
    import numpy as np
    from predictive6 import OtherModel6, PredictiveController6
    from collections import Counter
    mdl = OtherModel6(axis=axis, factor=factor, seed=seed, tissue=tissue)
    e.reset(seed=seed)
    site_ok = True
    kw = {} if rates is None else dict(rates=tuple(rates))   # round 13: step cap arm
    ctrl = PredictiveController6(mdl, task_deg=TASK_DEG, task_sign=TASK_SIGN, **kw,
                                 harm_budget=c6.HARM_BUDGET, force_budget=c6.FORCE_BUDGET,
                                 p_cost=P_COST, p_in_step=P_IN_STEP, elbow_ref=ELBOW_REF,
                                 site_select=site_select,
                                 site_offsets=SITE_OFFSETS, site_phis=SITE_ANGLES).bind(e)
    # with site selection, start_episode starts the copy with the probe
    # parked above and then moves to the chosen site
    ctrl.start_episode(e.observe())
    if not site_select:
        # round 8e: same order without site selection - start the copy with
        # the probe parked above, then move to the default site below (the
        # first version moved first, and the copy started under the limb)
        mdl.sync(e.observe(), true_env=e)
        site_ok = e.set_site(0.0, 0.0)
        ctrl.bind(e)
    obs = e.observe()
    peak = dict(p=0.0, pb=0.0, sb=0.0, reflex=0.0, force=0.0)
    tmin = 1e9
    t_done = None
    n_hold = 0
    task = math.radians(TASK_DEG)
    choices = Counter()
    cstate = dict(best_end_p=0.0, contacts=[])
    n = 0
    for n in range(1, (n_max or N_PLAN_MAX) + 1):
        pos, rot = ctrl(obs)
        obs, info = e.step(pos, rot)
        tmin = _acc(peak, info, tmin)
        _contacts(e, cstate)
        choices[ctrl.last_choice] += 1
        if t_done is None and TASK_SIGN * e.elbow_change() >= task:
            t_done = e.t
        if t_done is not None:
            n_hold += 1
            if n_hold >= int(HOLD_S * 50):
                break
    return _finish(e, peak, tmin, t_done, dict(
        site_ok=bool(site_ok and (not site_select or ctrl.chosen_site is not None)),
        choices=dict(choices), chosen_site=dict(ctrl.chosen_site), site_scores=ctrl.site_scores,
        _contacts=cstate["contacts"], _step_end_p=cstate["best_end_p"],
        model_diverged=int(mdl.diverged_count), n_steps=n, model_elbow0=float(getattr(mdl, "elbow0", float("nan"))),
        pred_peak_reflex=float(max(ctrl.log_pred_reflex or [0.0])),
        pred_peak_force=float(max(ctrl.log_pred_force or [0.0])),
        pred_peak_p=float(max(ctrl.log_pred_p or [0.0])),
        bind_reflex=ctrl.n_bind_reflex / max(n, 1), bind_force=ctrl.n_bind_force / max(n, 1),
        bind_tissue=ctrl.n_bind_tissue / max(n, 1), bind_both=ctrl.n_bind_both / max(n, 1)))


def _job(spec):
    e = env()
    kind, seed, a, b, c_, d_ = spec
    if kind == "press":
        r = run_press(e, seed, a)
        r.update(arm="press", cond=f"press@{a:.2f}")
    elif kind == "cbf":
        r = run_press(e, seed, a, barrier=True)
        r.update(arm="cbf", cond=f"cbf@{a:.2f}")
    elif kind == "rule":
        r = run_press(e, seed, a, site=RULE_SITE)
        r.update(arm="rule", cond=f"rule@{a:.2f}")
    elif kind == "trained":
        # round 10C: pi_med@0.25 learned on training seeds 100-160 (round 9),
        # run in this harness alongside the planners
        r = run_press(e, seed, a, site=TRAINED_SITE)
        r.update(arm="trained", cond=f"trained@{a:.2f}")
    else:
        label, axis, factor, tissue, site = a, b, c_, d_[0], d_[1]
        r = run_predictive(e, seed, axis, factor, tissue, site)
        r.update(arm="predictive", cond=label, axis=axis, factor=factor,
                 tissue=str(tissue), site_select=bool(site))
    r.update(seed=seed, start_contact=list(e.start_contact), reset_diverged=int(e.reset_diverged),
             tissue_t_mid=float(e.tissue["t_mid"]), tissue_taper_u=float(e.tissue["taper_u"]))
    return r


T_TAB = {19: 2.101, 18: 2.110, 17: 2.120, 16: 2.131, 15: 2.145, 14: 2.160, 13: 2.179,
         12: 2.201, 11: 2.228, 10: 2.262, 9: 2.306, 8: 2.365, 7: 2.447, 6: 2.571, 5: 2.776, 4: 3.182, 3: 4.303}


def _paired(A, B, matched):
    a = {r["seed"]: r for r in A}; b = {r["seed"]: r for r in B}
    ss = [s for s in a if s in b and (not matched or (a[s]["reached"] >= 0.95 and b[s]["reached"] >= 0.95))]
    if len(ss) < 3:
        return None
    d = [a[s]["peak_p"] - b[s]["peak_p"] for s in ss]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); se = sd / math.sqrt(n); t = T_TAB.get(n, 2.0)
    return m, m - t * se, m + t * se, n, sum(x > 0 for x in d)


def summarize(rows):
    import numpy as np
    from collections import Counter
    by = {}
    for r in rows:
        by.setdefault(r["cond"], []).append(r)
    print(f"\n{'condition':14s} {'n':>2} {'task':>5} {'>=.95':>5} {'t_done':>6} {'P':>6} {'Pn':>5} {'Ps':>5} {'t_min':>5} {'R':>6} {'F':>5}  sites (top 3)")
    order = ["press@0.10", "press@0.25", "cbf@0.25", "rule@0.10", "nosite", "nomap", "popmap", "indmap", "individual",
             "robotonly", "scale@0.5", "scale@2", "shift@20"]
    for c in order + sorted(set(by) - set(order)):
        rs = by.get(c)
        if not rs:
            continue
        td = [r["t_done"] for r in rs if r["t_done"] is not None]
        line = (f"{c:14s} {len(rs):2d} {np.mean([r['reached'] for r in rs]):5.2f} {sum(r['reached'] >= 0.95 for r in rs):5d} "
                f"{(np.mean(td) if td else float('nan')):6.2f} {np.mean([r['peak_p'] for r in rs]):6.1f} "
                f"{np.mean([r['peak_pb'] for r in rs]):5.1f} {np.mean([r['peak_sb'] for r in rs]):5.1f} "
                f"{np.mean([r['t_min'] for r in rs]):5.1f} {np.mean([r['peak_reflex'] for r in rs]):6.3f} "
                f"{np.mean([r['peak_force'] for r in rs]):5.0f}  ")
        if rs[0]["arm"] == "predictive":
            cs = Counter((round(r["chosen_site"]["offset"], 2), int(round(r["chosen_site"]["phi"]))) for r in rs)
            line += ", ".join(f"{k}:{v}" for k, v in cs.most_common(3))
        print(line)
    if "popmap" not in by:
        return
    print("\npaired dP = first - popmap  (positive = more tissue load than the population-map copy)")
    print(f"{'first':14s} | {'all 19: dP [95% CI]  first worse':>40} | {'task-matched: dP [95% CI]  n  first worse':>46}")
    for c in ("nomap", "individual", "indmap", "nosite", "robotonly", "press@0.10", "press@0.25", "cbf@0.25", "rule@0.10",
              "scale@0.5", "scale@2", "shift@20"):
        if c not in by:
            continue
        a = _paired(by[c], by["popmap"], False); m = _paired(by[c], by["popmap"], True)
        fa = f"{a[0]:+7.1f} [{a[1]:+7.1f}, {a[2]:+7.1f}] {a[4]:2d}/{a[3]:<2d}" if a else "n/a"
        fm = f"{m[0]:+7.1f} [{m[1]:+7.1f}, {m[2]:+7.1f}] n={m[3]:2d} {m[4]:2d}/{m[3]:<2d}" if m else "fewer than 3 complete both"
        print(f"{c:14s} | {fa:>40} | {fm:>46}")
    if "nomap" in by:
        m = _paired(by["nomap"], by["popmap"], True)
        if m:
            print(f"\nH8-1 (primary, task-matched): nomap - popmap = {m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}], n={m[3]} -> "
                  f"{'SUPPORTED (CI excludes 0, map lower)' if m[1] > 0 else 'NOT SUPPORTED'}")


def main():
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--analyse":
        summarize(json.load(open(sys.argv[2]))["rows"]); return 0
    seeds = SEEDS[:2] if SMOKE else SEEDS
    jobs = []
    for s in seeds:
        jobs += [("press", s, sp, None, None, None) for sp in (0.10, 0.25)]
        jobs += [("cbf", s, 0.25, None, None, None), ("rule", s, 0.10, None, None, None)]
        if WITH_TRAINED:
            jobs += [("trained", s, 0.25, None, None, None), ("rule", s, 0.25, None, None, None)]
        jobs += [("pred", s, lab, ax, f, (ti, site)) for lab, ax, f, ti, site in CONDITIONS]
    n_proc = int(os.environ.get("WM_PROCS", 8))
    print(f"=== crossover8 (lift, closed loop): {len(jobs)} episodes, {n_proc} workers, P cost {P_COST}, P in step {P_IN_STEP}, elbow ref {ELBOW_REF} ===", flush=True)
    t0 = time.time()
    rows = []
    part = os.path.join(OUT, f"crossover8_partial{TAG}.json")
    with Pool(processes=n_proc) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            print(f"  [{k + 1}/{len(jobs)}] {r['cond']:12s} seed {r['seed']:2d}: task {r['reached']:.2f} P {r['peak_p']:6.1f} "
                  f"t_min {r['t_min']:4.1f} F {r['peak_force']:5.1f} R {r['peak_reflex']:.3f} "
                  f"site {r.get('chosen_site', r['site'])} ok {r.get('site_ok')} ({time.time() - t0:.0f}s)", flush=True)
            if (k + 1) % 10 == 0:
                json.dump(dict(rows=rows), open(part, "w"))
    name = f"crossover8{'_smoke' if SMOKE else ''}{TAG}.json"
    path = os.path.join(OUT, name)
    json.dump(dict(rows=rows, seeds=seeds, p_cost=P_COST, p_in_step=P_IN_STEP, elbow_ref=ELBOW_REF, force_budget=c6.FORCE_BUDGET, harm_budget=c6.HARM_BUDGET,
                   task_deg=TASK_DEG, hold_s=HOLD_S, site_offsets=SITE_OFFSETS, site_angles=SITE_ANGLES,
                   rule_site=RULE_SITE, conditions=[c[0] for c in CONDITIONS]), open(path, "w"), indent=1)
    print("wrote", path)
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
