"""Round 17 (shoulder route, main experiment): does a body model that knows
THIS person's joint range make the lift gentler than a fixed rule and than a
reactive, feel-your-way controller?  PREREGISTRATION section 10, round 17.

Population: myoArm, free shoulder, body variation 0.25, plus a person-specific
passive range (shoulder elevation / elbow flexion free range above the settled
posture), drawn by `person()`.

Stages (argv[1]):
  screen    which seeds give a usable body in the free-shoulder scene
  train     48-strategy truth grid on the training persons
  select    fixed rule F, reactive fallback order and guard thresholds  (locks policy.json)
  test      test persons: truth grid, guarded episodes for the reactive arm,
            grids inside the robot's copy (population body + recorded range
            with error) for the model arms
  eval      H17-1..5
"""
import os
import sys
import json
import math
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round17")
sys.path.insert(0, ROOT)

OFFSETS = (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
ANGLES = (-40.0, -20.0, 0.0)
SPEEDS = (0.10, 0.25)
STRATS = [(o, a, sp) for o in OFFSETS for a in ANGLES for sp in SPEEDS]
TRAIN_POOL = list(range(100, 200))
TEST_POOL = list(range(300, 420))
N_TRAIN, N_TEST = 60, 40
ROM_ERRS = (0.0, 4.0, 8.0, 16.0)          # deg, SD of the error of the recorded range
UNRESTRICTED = 90.0
# population parameters: fixed in the preregistration BEFORE any round-17 data
P_SH_LIMITED, SH_RANGE = 0.4, (4.0, 16.0)
P_EL_LIMITED, EL_RANGE = 0.3, (0.0, 10.0)
PAINFREE = 0.5                              # N m: below this an episode counts as end-range free
N_ATTEMPTS = 3
RETRACT_S = 0.5


def person(seed):
    """(free_sh, free_el) in degrees for this person."""
    import numpy as np
    rng = np.random.default_rng(seed * 7919 + 17)
    fs = float(rng.uniform(*SH_RANGE)) if rng.random() < P_SH_LIMITED else UNRESTRICTED
    fe = float(rng.uniform(*EL_RANGE)) if rng.random() < P_EL_LIMITED else UNRESTRICTED
    return fs, fe


def recorded(seed, err_deg):
    """What the care record says about this person's free range (goniometry error)."""
    import numpy as np
    fs, fe = person(seed)
    rng = np.random.default_rng(seed * 104729 + int(err_deg * 10) + 5)
    f = lambda x: x if x >= UNRESTRICTED else max(0.0, x + float(rng.normal(0, err_deg)))
    return f(fs), f(fe)


# ---------------------------------------------------------------- workers
_COPY = None


def _copy_env():
    """The robot's internal copy: the POPULATION body (no individual parameters)."""
    global _COPY
    if _COPY is None:
        import mujoco
        from care_env6 import CareContactEnv6
        e = CareContactEnv6(body_variation=0.0, weld_humerus=False)
        e.site_mode = "lift"
        e._hb = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_BODY, "humerus")
        e._sj = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_elv")
        e._ej = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        e._fg = mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_GEOM, "radius_coll")
        e._gh_log = None
        import diag_shoulder1 as s1
        orig = e._instantaneous_harm

        def hook():
            h = orig()
            if e._gh_log is not None:
                e._gh_log.append(s1.gh_load(e))
            return h
        e._instantaneous_harm = hook
        _COPY = e
    return _COPY


def _job(spec):
    import shoulder_task as st
    kind, seed, fs, fe, o, a, sp, extra = spec
    try:
        if kind == "copy":
            real = st.env
            st.env = _copy_env
            try:
                r = st.run(0, fs, fe, o, a, sp)       # population body; seed is irrelevant without variation
            finally:
                st.env = real
            r["seed"] = seed
        else:
            r = st.run(seed, fs, fe, o, a, sp, stop_force=extra if kind == "guard" else None)
    except Exception as ex:
        r = dict(seed=seed, offset=o, angle=a, speed=sp, error=repr(ex)[:200], done=False)
    r["kind"] = kind; r["tag"] = extra if kind == "copy" else None
    return r


def _screen_job(seed):
    import shoulder_task as st
    try:
        e = st.env(); e.reset(seed=seed)
        return dict(seed=seed, valid=bool(not e.start_contact and not e.reset_diverged))
    except Exception as ex:
        return dict(seed=seed, valid=False, error=repr(ex)[:120])


def _pool(func, jobs, label):
    from multiprocessing import Pool
    n = int(os.environ.get("WM_PROCS", 8)); rows = []; t0 = time.time()
    print(f"=== {label}: {len(jobs)} jobs, {n} workers ===", flush=True)
    with Pool(n) as p:
        for k, r in enumerate(p.imap_unordered(func, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 400 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------- stages
def screen():
    os.makedirs(OUT, exist_ok=True)
    rows = _pool(_screen_job, TRAIN_POOL + TEST_POOL, "screen")
    v = sorted(r["seed"] for r in rows if r["valid"])
    tr = [s for s in v if s in TRAIN_POOL][:N_TRAIN]; te = [s for s in v if s in TEST_POOL][:N_TEST]
    json.dump(dict(train=tr, test=te, rows=rows), open(os.path.join(OUT, "screen.json"), "w"))
    print(f"train {len(tr)} test {len(te)}")


def _seeds():
    return json.load(open(os.path.join(OUT, "screen.json")))


def train():
    tr = _seeds()["train"]
    jobs = [("truth", s) + person(s) + st_ + (None,) for s in tr for st_ in STRATS]
    json.dump(dict(rows=_pool(_job, jobs, "train grid")), open(os.path.join(OUT, "train.json"), "w"))


def harm(r):
    return r["lim_sh"] + r["lim_el"]


def _table(rows):
    T = {}
    for r in rows:
        if "error" not in r:
            T.setdefault(r["seed"], {})[(r["offset"], r["angle"], r["speed"])] = r
    return T


def select():
    import numpy as np
    T = _table(json.load(open(os.path.join(OUT, "train.json")))["rows"])
    people = sorted(T)
    stats = []
    for k in STRATS:
        R = [T[s][k] for s in people if k in T[s]]
        comp = sum(r["done"] for r in R) / len(people)
        D = [r for r in R if r["done"]]
        if comp >= 0.90 and D:
            stats.append((float(np.median([harm(r) for r in D])), float(np.mean([harm(r) for r in D])),
                          float(np.median([r["gh_shear"] for r in D])), k, comp))
    stats.sort(key=lambda x: (round(x[0], 1), x[1], x[2]))
    # reactive fallback: first the fixed rule, then the best strategy that loads the OTHER joint
    # (site at least 9 cm away along the forearm), then the best of the rest
    order = [stats[0][3]]
    far = [s[3] for s in stats if abs(s[3][0] - order[0][0]) >= 0.09]
    if far:
        order.append(far[0])
    order += [s[3] for s in stats if s[3] not in order][:N_ATTEMPTS - len(order)]
    guard = {}
    for k in order:
        free = [T[s][k]["peak_force"] for s in people if k in T[s] and T[s][k]["done"] and harm(T[s][k]) < PAINFREE]
        guard[str(k)] = 1.2 * float(np.percentile(free, 95)) if free else None
    pol = dict(F=order[0], order=order, guard=guard, table=[dict(strategy=s[3], median_harm=s[0], mean_harm=s[1], gh_shear=s[2], completion=s[4]) for s in stats[:10]])
    json.dump(pol, open(os.path.join(OUT, "policy.json"), "w"), indent=1)
    print("fixed rule F =", order[0], "\nreactive order =", order, "\nguards (N) =", guard)
    for s in stats[:6]:
        print(f"   {s[3]}  median harm {s[0]:.2f}  mean {s[1]:.2f}  GH shear {s[2]:.1f}  completion {s[4]:.0%}")


def test():
    te = _seeds()["test"]
    pol = json.load(open(os.path.join(OUT, "policy.json")))
    for name, jobs in (
        ("test_truth", [("truth", s) + person(s) + k + (None,) for s in te for k in STRATS]),
        ("test_guard", [("guard", s) + person(s) + tuple(k) + (pol["guard"][str(tuple(k))],) for s in te for k in pol["order"] if pol["guard"][str(tuple(k))]]),
        ("test_copy", [("copy", s) + recorded(s, er) + k + (er,) for s in te for er in ROM_ERRS for k in STRATS]),
    ):
        path = os.path.join(OUT, name + ".json")
        if os.path.exists(path):
            print("exists, skipping", path); continue
        json.dump(dict(rows=_pool(_job, jobs, name)), open(path, "w"))


def evaluate():
    import numpy as np
    from round14_recompute import ci, fmt
    te = _seeds()["test"]
    pol = json.load(open(os.path.join(OUT, "policy.json")))
    F = tuple(pol["F"]); order = [tuple(k) for k in pol["order"]]
    T = _table(json.load(open(os.path.join(OUT, "test_truth.json")))["rows"])
    G = {}
    for r in json.load(open(os.path.join(OUT, "test_guard.json")))["rows"]:
        if "error" not in r:
            G.setdefault(r["seed"], {})[(r["offset"], r["angle"], r["speed"])] = r
    C = {}
    for r in json.load(open(os.path.join(OUT, "test_copy.json")))["rows"]:
        if "error" not in r:
            C.setdefault(r["tag"], {}).setdefault(r["seed"], {})[(r["offset"], r["angle"], r["speed"])] = r
    FAIL_T = 3.5

    def out_truth(s, k):
        r = T.get(s, {}).get(k)
        if r is None:
            return dict(done=False, harm=float("nan"), shear=float("nan"), t=FAIL_T)
        return dict(done=r["done"], harm=harm(r), shear=r["gh_shear"], t=(r["t_end"] if r["done"] else FAIL_T))

    def arm_F(s):
        return out_truth(s, F)

    def arm_R(s):
        h = sh = 0.0; t = 0.0
        for k in order:
            r = G.get(s, {}).get(k)
            if r is None:
                continue
            h = max(h, harm(r)); sh = max(sh, r["gh_shear"]); t += r["t_end"]
            if r["done"]:
                return dict(done=True, harm=h, shear=sh, t=t)
            t += RETRACT_S
        return dict(done=False, harm=h, shear=sh, t=t)

    def arm_M(er):
        def f(s):
            P = C.get(er, {}).get(s, {})
            cand = [(round(harm(r), 1), r["gh_shear"], r["t_end"], k) for k, r in P.items() if r["done"]]
            if not cand:
                return out_truth(s, F)
            return out_truth(s, min(cand)[3])
        return f

    def arm_O(s):
        cand = [(round(harm(r), 1), r["gh_shear"], r["t_end"], k) for k, r in T.get(s, {}).items() if r["done"]]
        return out_truth(s, min(cand)[3]) if cand else out_truth(s, F)

    arms = {"F fixed rule": arm_F, "R reactive": arm_R}
    for er in ROM_ERRS:
        arms[f"M record +-{er:g} deg"] = arm_M(er)
    arms["O oracle"] = arm_O
    res = {a: {s: f(s) for s in te} for a, f in arms.items()}
    lim = [s for s in te if min(person(s)) < UNRESTRICTED]
    print(f"test persons {len(te)} ({len(lim)} with a restricted joint); F = {F}; reactive order {order}")
    print(f"\n{'arm':22s} {'done':>6} {'harm median':>11} {'mean':>7} {'p90':>7} {'pain-free':>9} {'GH shear':>8} {'time':>6}")
    for a, R in res.items():
        D = [R[s] for s in te if R[s]["done"]]
        hv = [R[s]["harm"] for s in te if not math.isnan(R[s]["harm"])]
        print(f"{a:22s} {len(D):3d}/{len(te):<2d} {np.median(hv):11.2f} {np.mean(hv):7.2f} {np.percentile(hv, 90):7.2f} "
              f"{np.mean([h < PAINFREE for h in hv]):9.0%} {np.median([r['shear'] for r in D]) if D else float('nan'):8.1f} {np.median([R[s]['t'] for s in te]):6.2f}")

    def pair(a, b, key, people=te):
        d = [res[a][s][key] - res[b][s][key] for s in people if not math.isnan(res[a][s][key]) and not math.isnan(res[b][s][key])]
        return ci(d)

    M5 = "M record +-8 deg"
    print("\npaired differences (negative = first arm gentler / faster)")
    for name, a, b in (("H17-1", M5, "F fixed rule"), ("H17-2", M5, "R reactive")):
        ch, cs, ct = pair(a, b, "harm"), pair(a, b, "shear"), pair(a, b, "t")
        nd_a = sum(res[a][s]["done"] for s in te); nd_b = sum(res[b][s]["done"] for s in te)
        ok = ch is not None and ch[2] < 0 and not (cs is not None and cs[1] > 0) and nd_a >= nd_b
        print(f"{name} {a} - {b}: harm {fmt(ch)} | GH shear {fmt(cs)} | time {fmt(ct)} | done {nd_a} vs {nd_b} -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    print("H17-3 tolerance curve (harm, M - F):")
    for er in ROM_ERRS:
        print(f"   record error {er:4.0f} deg: {fmt(pair(f'M record +-{er:g} deg', 'F fixed rule', 'harm'))}")
    print(f"H17-4 restricted persons only (n={len(lim)}): M5 - F harm {fmt(pair(M5, 'F fixed rule', 'harm', lim))} | M5 - R harm {fmt(pair(M5, 'R reactive', 'harm', lim))}")
    print(f"H17-5 gap to the oracle: M5 - O harm {fmt(pair(M5, 'O oracle', 'harm'))}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(screen=screen, train=train, select=select, test=test, eval=evaluate).get(stage)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True)
    f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
