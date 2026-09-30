"""POST HOC (not preregistered; no new simulation): how many of the fixed rule's harmful outcomes does the force guard prevent?

Answers a reviewer point on out/posthoc_guard_trips.txt: that statistic ("share of HARMFUL first attempts in which the
guard never tripped") is conditioned on the outcome - attempts where the guard tripped in time and prevented harm are
excluded.  Here the conditioning is on the UNGUARDED fixed rule F instead:

  among persons whose unguarded fixed-rule lift F is harmful (harm >= 0.5 N m),
  the share whose whole guarded episode R(m) (all attempts of the round-17 switch order, exactly as each round's
  evaluate() computes it: harm = max over the attempts made, done = some attempt completed) is NOT harmful
  = "share of F's harmful outcomes prevented by the guard",
  split into prevented -> safe completion (R done, harm < 0.5) vs prevented -> no completion (every attempt aborted
  by the guard or otherwise incomplete; harm < 0.5).
Complement: persons NOT harmed by F but harmed under R(m) (harm introduced by switching strategy after a trip).
Identity checked per cell: harmful(R) = harmful(F) - prevented + introduced, and harmful(F), harmful(R), safe(R)
are cross-checked against each round's stored eval.txt (must match).

Data (stored rows only):
  round 21  hard limits                 n=120, F = truth row, R = guard rows, gross harm (round17.harm)
  round 22  part A hard / soft6 / soft12 n=120, gross harm (round22.evaluate)
  round 23  W soft12 / soft20 / soft25   n=120, NET harm = robot-caused load (round23.evaluate, primary)
Guard m in {0.6, 0.8, 1.0, 1.2} x p95 of harmless force  (= round21.MULTS 0.5/0.667/0.833/1.0 x the round-17 guard).

Lever-arm arithmetic (secondary; needs mujoco -> run the whole script with the .venv_myo python):
  model = the shoulder-task body (diag_shoulder1.env(): CareContactEnv6, humerus not welded, body variation 0.25),
  body variation of seed 104 applied (it scales the humerus/ulna/radius masses), shoulder_elv and elbow_flexion set to
  seed 104's settled rest angles as stored in every seed-104 episode row (out/round18/library.json: rest_sh_deg,
  rest_el_deg), all other joints at the model's relaxed posture, then ONE mj_forward (kinematics only, no rollout).
  Note: constructing the env runs its built-in model build/settle (as every episode does); no task episode is run.
  Contact point of F (offset 0.03 m, angle -40 deg): aim point on the forearm axis = centre of geom radius_coll +
  (AIM_DISTAL + 0.03) x bone axis (care_env6.approach_start), and the surface point one forearm radius below it
  along the lift direction (diag_lift_oracle.lift_dir, the probe comes from below along it).
  Moment arm reported three ways: |r| (joint centre -> contact point, as requested), the distance perpendicular to
  the joint axis (largest torque per newton any force direction can produce), and |(r x u) . axis| for u = F's lift
  direction (torque per newton of force along the push).  Wrist-force change for 0.5 N m = 0.5 / arm.
  Forearm + hand weight = sum of body_mass over the ulna subtree (ulna, radius, carpals, hand, fingers) x g, with the
  seed's body variation applied (only ulna/radius/humerus masses vary); the nominal (pre-variation) value is also given.
  Caveat: the stored harm is the constraint torque of the joint limit, to which gravity and muscle forces also
  contribute, so this is order-of-magnitude arithmetic, not a decomposition of the stored loads.
Writes out/posthoc_guard_prevented.txt.
"""
import os
import re
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

THR = 0.5
MULTS = (0.5, 0.667, 0.833, 1.0)          # round21.MULTS: x the round-17 guard (1.2 x p95)


def gross(x):                              # = round17.harm (rounds 21, 22)
    return x["lim_sh"] + x["lim_el"]


def net(x):                                # = round23.evaluate net (primary there)
    return max(0.0, x["lim_sh"] - x.get("lim_base_sh", 0.0)) + max(0.0, x["lim_el"] - x.get("lim_base_el", 0.0))


def policy():
    p = json.load(open(os.path.join(TOP, "out", "round17", "policy.json")))
    return tuple(p["F"]), [tuple(k) for k in p["order"]], {tuple(eval(k)): v for k, v in p["guard"].items()}


def tables(rnd):
    T, G = {}, {}
    for x in json.load(open(os.path.join(TOP, "out", rnd, "test.json")))["rows"]:
        if "error" in x:
            continue
        c = x.get("cond", "hard")
        k = (x["offset"], x["angle"], x["speed"])
        if x["kind"] == "truth":
            T.setdefault(c, {}).setdefault(x["seed"], {})[k] = x
        elif x["kind"] == "guard":
            G.setdefault(c, {}).setdefault(x["seed"], {}).setdefault(k, {})[round(x["tag"], 3)] = x
    return T, G


def reactive(G, s, m, order, guard, hf):
    """Identical to round21/22/23.evaluate reactive(): try the switch order, stop at the first completed attempt."""
    h = 0.0; att = 0; tripped = []
    for k in order:
        g = G.get(s, {}).get(k, {}).get(round(guard[k] * m, 3))
        if g is None:
            continue
        att += 1
        h = max(h, hf(g)); tripped.append(bool(g["aborted"]))
        if g["done"]:
            return dict(done=True, harm=h, n_att=att, tripped=tripped)
    return dict(done=False, harm=h, n_att=att, tripped=tripped)


def eval_counts(rnd, cond):
    """(safe, harmful) per arm from the round's stored eval.txt (primary harm, 0.5 N m table)."""
    txt = open(os.path.join(TOP, "out", rnd, "eval.txt"), encoding="utf-8").read().splitlines()
    if rnd == "round21":
        i0 = 0
    elif rnd == "round22":
        i0 = next(i for i, l in enumerate(txt) if l.startswith(f"=== part A, condition {cond} "))
    else:
        i0 = next(i for i, l in enumerate(txt) if l.startswith(f"=== W, condition {cond} "))
        i0 = next(i for i in range(i0, len(txt)) if "harm = net, harmful if >= 0.5" in txt[i])
    i = next(i for i in range(i0, len(txt)) if re.match(r"\s*arm\s+safe\s+harmful", txt[i]))
    out = {}
    for l in txt[i + 1:]:
        mm = re.match(r"\s*(F|R x\d\.\dp95)\s+(\d+)\s+(\d+)\s", l)
        if mm:
            out[mm.group(1)] = (int(mm.group(2)), int(mm.group(3)))
        elif not l.strip() or l.strip().startswith(("R - F", "H2", "[")):
            if out:
                break
    return out


def lever_arms():
    import math
    import numpy as np
    import mujoco
    import diag_shoulder1 as s1
    import diag_lift_oracle as dlo
    from care_env6 import AIM_DISTAL, _unit
    F, _, _ = policy()
    seed = 104
    lib = json.load(open(os.path.join(TOP, "out", "round18", "library.json")))
    rows = lib["rows"] if isinstance(lib, dict) and "rows" in lib else lib
    rest = {(round(x["rest_sh_deg"], 6), round(x["rest_el_deg"], 6)) for x in rows
            if isinstance(x, dict) and x.get("seed") == seed and "rest_sh_deg" in x}
    assert len(rest) == 1, rest
    rsh, rel = rest.pop()
    e = s1.env()
    e._apply_body_variation(seed)
    m = e.m; d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    from supported_scene import _relaxed_qpos
    _relaxed_qpos(m, d)
    d.qpos[m.jnt_qposadr[e._sj]] = math.radians(rsh)
    d.qpos[m.jnt_qposadr[e._ej]] = math.radians(rel)
    d.qpos[e.qadr] = np.concatenate([e.park - e.base, e.park_rot])     # probe parked, clear of the limb
    mujoco.mj_forward(m, d)
    rid = e.rid
    R = np.array(d.geom_xmat[rid]).reshape(3, 3)
    ax = _unit(R[:, 2])
    if np.dot(ax, e.axis) < 0:
        ax = -ax
    aim = np.array(d.geom_xpos[rid]) + (AIM_DISTAL + F[0]) * ax
    u = dlo.lift_dir(e, F[1])
    rad = float(m.geom_size[rid][0])
    surf = aim - rad * u
    res = dict(seed=seed, rest_sh_deg=rsh, rest_el_deg=rel, forearm_radius=rad, lift_dir=u.tolist())
    for nm, j in (("shoulder (shoulder_elv)", e._sj), ("elbow (elbow_flexion)", e._ej)):
        c = np.array(d.xanchor[j]); a = _unit(np.array(d.xaxis[j]))
        for lab, p in (("aim", aim), ("surface", surf)):
            r = p - c
            perp = float(np.linalg.norm(r - np.dot(r, a) * a))
            eff = abs(float(np.dot(np.cross(r, u), a)))
            res[(nm, lab)] = dict(dist=float(np.linalg.norm(r)), perp=perp, eff=eff)
    # forearm + hand = ulna subtree
    ul = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ulna")
    sub = [b for b in range(m.nbody) if _is_desc(m, b, ul)]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in sub]
    mass = float(sum(m.body_mass[b] for b in sub))
    hum = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humerus")
    g = float(np.linalg.norm(m.opt.gravity))
    nom = float(sum(e._pristine["body_mass"][b] for b in sub))          # before the seed's body variation
    res.update(forearm_hand_mass=mass, g=g, humerus_mass=float(m.body_mass[hum]), n_bodies=len(sub),
               bodies=names, forearm_hand_mass_nominal=nom)
    return res


def _is_desc(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def main():
    F, order, guard = policy()
    assert order[0] == F
    out = []
    say = lambda t="": (print(t), out.append(t))
    S21 = json.load(open(os.path.join(TOP, "out", "round21", "test_seeds.json")))
    S22 = json.load(open(os.path.join(TOP, "out", "round22", "seeds.json")))["A"]
    S23 = json.load(open(os.path.join(TOP, "out", "round23", "seeds.json")))["W"]
    cells = [("round21", "hard", S21, gross, "gross", "hard"),
             ("round22", "hard", S22, gross, "gross", "hard"),
             ("round22", "soft6", S22, gross, "gross", "graded"),
             ("round22", "soft12", S22, gross, "gross", "graded"),
             ("round23", "soft12", S23, net, "net", "graded"),
             ("round23", "soft20", S23, net, "net", "graded"),
             ("round23", "soft25", S23, net, "net", "graded")]
    say("POST HOC: share of the fixed rule's harmful outcomes that the force guard PREVENTS (conditioned on the unguarded")
    say(f"fixed rule F = {F}, not on the guarded outcome).  harmful = harm >= {THR} N m.  R(m) = whole guarded episode,")
    say("all attempts of the round-17 switch order (as in each round's evaluate()).  No new simulation.")
    say("  F harm   = harmful under unguarded F;  prev = of those, NOT harmful under R(m)  [share = prev / F harm]")
    say("  -> safe  = prevented AND R completed (safe completion);  -> none = prevented, no attempt completed (aborted/incomplete)")
    say("  intro    = NOT harmful under F but harmful under R(m)  (harm introduced by switching);  net = intro - prev")
    say("  check    = harmful(F), harmful(R), safe(R) equal the round's eval.txt, and harmful(R) = F harm - prev + intro")
    say("")
    summ = {"hard": [], "graded": []}
    allok = True
    rows_tab = []
    cache = {}
    for rnd, cond, people, hf, hname, kind in cells:
        if rnd not in cache:
            cache[rnd] = tables(rnd)
        T, G = cache[rnd]
        T, G = T.get(cond, {}), G.get(cond, {})
        ev = eval_counts(rnd, cond)
        say(f"=== {rnd} {cond} ({'hard limit' if kind == 'hard' else 'graded end-feel'}), n={len(people)}, harm = {hname} ===")
        Fo = {s: T[s][F] for s in people}
        Fh = [s for s in people if hf(Fo[s]) >= THR]
        Fsafe = sum(bool(Fo[s]["done"]) and hf(Fo[s]) < THR for s in people)
        okF = ev["F"] == (Fsafe, len(Fh))
        say(f"  F: harmful {len(Fh)}, safe {Fsafe}   (eval.txt F safe {ev['F'][0]}, harmful {ev['F'][1]}: {'match' if okF else 'MISMATCH'})")
        say(f"  {'guard':>16s} {'F harm':>6s} {'prev':>5s} {'share':>7s} {'->safe':>7s} {'->none':>7s} {'intro':>6s} {'net':>5s} "
            f"{'R harm':>6s} {'R safe':>6s} {'check':>6s}  prevented: 1st attempt tripped / completed by attempt #2,#3")
        allok &= okF
        for m in MULTS:
            Ro = {s: reactive(G, s, m, order, guard, hf) for s in people}
            bad = lambda s: Ro[s]["harm"] >= THR
            prev = [s for s in Fh if not bad(s)]
            p_safe = [s for s in prev if Ro[s]["done"]]
            p_none = [s for s in prev if not Ro[s]["done"]]
            intro = [s for s in people if hf(Fo[s]) < THR and bad(s)]
            Rh = sum(bad(s) for s in people); Rs = sum(Ro[s]["done"] and not bad(s) for s in people)
            a = f"R x{m * 1.2:.1f}p95"
            ok = ev[a] == (Rs, Rh) and Rh == len(Fh) - len(prev) + len(intro)
            allok &= ok
            trip1 = sum(Ro[s]["tripped"][0] for s in prev)
            byatt = {n: sum(Ro[s]["done"] and Ro[s]["n_att"] == n for s in prev) for n in (1, 2, 3)}
            sh = 100 * len(prev) / len(Fh) if Fh else float("nan")
            say(f"  {'x' + format(m * 1.2, '.1f') + ' p95':>9s} ({round(guard[F] * m, 3):5.1f} N) {len(Fh):6d} {len(prev):5d} {sh:6.1f}% {len(p_safe):7d} {len(p_none):7d} "
                f"{len(intro):6d} {len(intro) - len(prev):+5d} {Rh:6d} {Rs:6d} {'ok' if ok else 'FAIL':>6s}  "
                f"{trip1}/{len(prev)} / #2 {byatt[2]}, #3 {byatt[3]}" + (f", #1 {byatt[1]}" if byatt[1] else ""))
            summ[kind].append((sh, f"{rnd} {cond} x{m * 1.2:.1f}", len(prev), len(Fh), len(p_safe), len(p_none), len(intro)))
        say("")
    say("=== overall ranges: share of F's harmful outcomes prevented by the guard ===")
    for kind in ("hard", "graded"):
        v = summ[kind]
        lo, hi = min(v), max(v)
        P = sum(x[2] for x in v); N = sum(x[3] for x in v); Ps = sum(x[4] for x in v); Pn = sum(x[5] for x in v); I = sum(x[6] for x in v)
        say(f"  {'hard limit' if kind == 'hard' else 'graded end-feel'} ({len(v)} cells): {lo[0]:.1f}% ({lo[1]}, {lo[2]}/{lo[3]}) to "
            f"{hi[0]:.1f}% ({hi[1]}, {hi[2]}/{hi[3]})")
        say(f"    pooled over cells (persons repeat across guard levels/conditions; descriptive only): prevented {P}/{N} "
            f"({100 * P / N:.1f}%), of which safe completion {Ps}, no completion {Pn}; introduced {I}")
        for m in MULTS:
            lab = f"x{m * 1.2:.1f}"
            vm = [x for x in v if x[1].endswith(lab)]
            say(f"    guard {lab} p95: prevented {', '.join(f'{x[2]}/{x[3]}' for x in vm)}  -> safe {sum(x[4] for x in vm)}, "
                f"none {sum(x[5] for x in vm)}; introduced {', '.join(str(x[6]) for x in vm)}")
    say("")
    say(f"cross-check against eval.txt (F and R harmful / safe counts) and harmful(R) identity: {'ALL MATCH' if allok else 'MISMATCH - see FAIL'}")
    say("")

    # ---------------- lever-arm arithmetic
    say("=== lever-arm arithmetic: wrist-force change equivalent to 0.5 N m of joint-limit torque, F support site ===")
    try:
        L = lever_arms()
    except ImportError as ex:
        say(f"  (skipped: {ex}; run this script with the .venv_myo python)")
        L = None
    if L is not None:
        g = L["g"]
        say(f"  model: shoulder-task body (humerus not welded), body variation of seed {L['seed']}, settled rest angles from the stored")
        say(f"  seed-{L['seed']} rows: shoulder_elv {L['rest_sh_deg']:.2f} deg, elbow_flexion {L['rest_el_deg']:.2f} deg; one mj_forward, no rollout.")
        say(f"  contact: F offset {F[0]:+.2f} m -> aim {0.03 + F[0]:.2f} m distal of the radius_coll centre on the bone axis; surface point")
        say(f"  {L['forearm_radius'] * 100:.1f} cm below it along the lift direction (angle {F[1]:g} deg).")
        say(f"  {'joint':26s} {'point':8s} {'|r| m':>7s} {'F@0.5Nm':>8s} | {'perp m':>7s} {'F@0.5Nm':>8s} | {'along push m':>12s} {'F@0.5Nm':>8s}")
        for nm in ("shoulder (shoulder_elv)", "elbow (elbow_flexion)"):
            for lab in ("aim", "surface"):
                x = L[(nm, lab)]
                f = lambda a: f"{THR / a:6.2f} N" if a > 1e-4 else "     inf"
                say(f"  {nm:26s} {lab:8s} {x['dist']:7.3f} {f(x['dist']):>8s} | {x['perp']:7.3f} {f(x['perp']):>8s} | {x['eff']:12.3f} {f(x['eff']):>8s}")
        W = L["forearm_hand_mass"] * g
        say(f"  forearm + hand = ulna subtree, {L['n_bodies']} bodies: mass {L['forearm_hand_mass']:.3f} kg -> weight {W:.2f} N "
            f"(g = {g:.2f} m/s^2; nominal body before seed variation {L['forearm_hand_mass_nominal']:.3f} kg = {L['forearm_hand_mass_nominal'] * g:.2f} N)")
        say(f"  (humerus alone {L['humerus_mass']:.3f} kg = {L['humerus_mass'] * g:.2f} N)")
        sd = L[("shoulder (shoulder_elv)", "aim")]["dist"]; ed = L[("elbow (elbow_flexion)", "aim")]["dist"]
        say(f"  => with |r| as the arm: 0.5 N m at the shoulder ~ {THR / sd:.1f} N at the wrist sensor, at the elbow ~ {THR / ed:.1f} N;")
        say(f"     i.e. {100 * THR / sd / W:.0f}% / {100 * THR / ed / W:.0f}% of the forearm+hand weight ({W:.1f} N); for comparison the guard on")
        say(f"     the first attempt is {' / '.join(format(round(guard[F] * mm, 3), '.1f') for mm in MULTS)} N at x0.6 / 0.8 / 1.0 / 1.2 p95.")
        say(f"  bodies: {', '.join(L['bodies'])}")
    open(os.path.join(TOP, "out", "posthoc_guard_prevented.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
