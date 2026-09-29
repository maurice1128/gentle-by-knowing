"""POST HOC (not preregistered; no new simulation): did the reactive force guard fire in harmful first attempts?

Uses the stored guarded episodes (kind == "guard") of the reactive arm R(m) in
  round 21  hard limits                       (n=120, gross harm lim_sh + lim_el, as round21.evaluate)
  round 22  part A: hard / soft6 / soft12     (n=120, gross harm, as round22.evaluate)
  round 23  W: soft12 / soft20 / soft25       (n=120, NET harm = dev_r24_learned.harm, as round23.evaluate)
Guard = m x the round-17 guard of the first strategy (m in round21.MULTS -> 0.6 / 0.8 / 1.0 / 1.2 x p95 of harmless force).

First attempt = order[0] of the round-17 switch order (= the fixed rule F's strategy) with its guard.
"Harmful first attempt": harm of that guarded episode >= 0.5 N m.
  (a) guard never tripped:  aborted == False
  (b) guard tripped:        aborted == True.
Timing: the episode log stores no time series (no time of the trip, no time the load crossed 0.5 N m).  But
shoulder_task.run stops simulating at the control step in which the guard trips (it breaks out of the loop; no
retraction is simulated) and the stored load is the peak over the simulated samples only.  Hence in every tripped
episode that is harmful, the >= 0.5 N m load was reached before or within the 20-ms control step in which the guard
tripped, i.e. the guard was too late.  The exact lead time is not stored.
Also reported (secondary): the other harm definition (net vs gross); rounds 21/22 have no pre-contact baseline
stored, so net == gross there.
Writes out/posthoc_guard_trips.txt.  Run with the .venv_mm python (json + numpy only)."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

THR = 0.5
MULTS = (0.5, 0.667, 0.833, 1.0)          # round21.MULTS: x the round-17 guard (1.2 x p95)


def gross(x):
    return x["lim_sh"] + x["lim_el"]


def net(x):                                # = dev_r24_learned.harm
    return max(0.0, x["lim_sh"] - x.get("lim_base_sh", 0.0)) + max(0.0, x["lim_el"] - x.get("lim_base_el", 0.0))


def policy():
    p = json.load(open(os.path.join(TOP, "out", "round17", "policy.json")))
    return [tuple(k) for k in p["order"]], {tuple(eval(k)): v for k, v in p["guard"].items()}


def guard_table(rnd):
    rows = json.load(open(os.path.join(TOP, "out", rnd, "test.json")))["rows"]
    G = {}
    for x in rows:
        if "error" in x or x.get("kind") != "guard":
            continue
        G.setdefault(x.get("cond", "hard"), {}).setdefault(x["seed"], {}).setdefault(
            (x["offset"], x["angle"], x["speed"]), {})[round(x["tag"], 3)] = x
    return G


def main():
    order, guard = policy()
    k0 = order[0]
    out = []
    say = lambda t="": (print(t), out.append(t))
    cells = [("round21", "hard", json.load(open(os.path.join(TOP, "out", "round21", "test_seeds.json"))), gross, "gross", "hard"),
             ("round22", "hard", json.load(open(os.path.join(TOP, "out", "round22", "seeds.json")))["A"], gross, "gross", "hard"),
             ("round22", "soft6", json.load(open(os.path.join(TOP, "out", "round22", "seeds.json")))["A"], gross, "gross", "graded"),
             ("round22", "soft12", json.load(open(os.path.join(TOP, "out", "round22", "seeds.json")))["A"], gross, "gross", "graded"),
             ("round23", "soft12", json.load(open(os.path.join(TOP, "out", "round23", "seeds.json")))["W"], net, "net", "graded"),
             ("round23", "soft20", json.load(open(os.path.join(TOP, "out", "round23", "seeds.json")))["W"], net, "net", "graded"),
             ("round23", "soft25", json.load(open(os.path.join(TOP, "out", "round23", "seeds.json")))["W"], net, "net", "graded")]
    alt = {"gross": ("net", net), "net": ("gross", gross)}
    say("POST HOC: reactive force guard in HARMFUL FIRST ATTEMPTS (first attempt = fixed-rule strategy "
        f"{k0}, round-17 guard {guard[k0]:.2f} N x m)")
    say(f"harmful = harm of the guarded first-attempt episode >= {THR} N m; (a) guard never tripped; (b) guard tripped "
        "(load >= 0.5 already reached at or before the trip step - see note)")
    say("")
    tables = {}
    summ = {"hard": [], "graded": []}
    for rnd, cond, people, hf, hname, kind in cells:
        G = guard_table(rnd).get(cond, {})
        tables[(rnd, cond)] = G
        say(f"=== {rnd} {cond} ({'hard limit' if kind == 'hard' else 'graded end-feel'}), n={len(people)}, harm = {hname} ===")
        say(f"  {'guard':>16s} {'N':>6s} {'missing':>7s} {'harmful 1st':>11s} {'(a) no trip':>14s} {'(b) tripped':>14s} | "
            f"{'alt harm (' + alt[hname][0] + ')':>22s}")
        for m in MULTS:
            g_n = round(guard[k0] * m, 3)
            eps = [G.get(s, {}).get(k0, {}).get(g_n) for s in people]
            miss = sum(e is None for e in eps)
            eps = [e for e in eps if e is not None]
            H = [e for e in eps if hf(e) >= THR]
            a = sum(not e["aborted"] for e in H); b = sum(bool(e["aborted"]) for e in H)
            Ha = [e for e in eps if alt[hname][1](e) >= THR]
            aa = sum(not e["aborted"] for e in Ha)
            pa = 100 * a / len(H) if H else float("nan"); pb = 100 * b / len(H) if H else float("nan")
            say(f"  {'x' + format(m * 1.2, '.1f') + ' p95':>9s} ({g_n:5.1f} N) {len(eps):6d} {miss:7d} {len(H):11d} {a:6d} ({pa:5.1f}%) {b:6d} ({pb:5.1f}%) | "
                f"no trip {aa}/{len(Ha)} ({100 * aa / len(Ha) if Ha else float('nan'):.1f}%)")
            if H:
                summ[kind].append((pa, f"{rnd} {cond} x{m * 1.2:.1f}", a, len(H)))
        say("")
    say("=== overall ranges of '(a) guard never tripped' among harmful first attempts (primary harm definition) ===")
    for kind in ("hard", "graded"):
        v = summ[kind]
        lo, hi = min(v), max(v)
        say(f"  {'hard limit' if kind == 'hard' else 'graded end-feel'}: {lo[0]:.0f}% ({lo[1]}, {lo[2]}/{lo[3]}) to {hi[0]:.0f}% ({hi[1]}, {hi[2]}/{hi[3]}) over {len(v)} cells")
        if kind == "hard":
            v21 = [x for x in v if x[1].startswith("round21")]
            say(f"    round 21 only: {min(v21)[0]:.0f}% to {max(v21)[0]:.0f}%;  round 22 hard only: "
                f"{min(x for x in v if x[1].startswith('round22'))[0]:.0f}% to {max(x for x in v if x[1].startswith('round22'))[0]:.0f}%")
        else:
            for r in ("round22", "round23"):
                vr = [x for x in v if x[1].startswith(r)]
                say(f"    {r} only: {min(vr)[0]:.0f}% to {max(vr)[0]:.0f}%")
    say("")
    say("Timing note: no time series is stored (no trip time, no time the load crossed 0.5 N m).  shoulder_task.run stops the")
    say("simulation in the control step in which the wrist force first exceeds the guard (break; no retraction simulated),")
    say("and lim_sh / lim_el are peaks of a 50-ms moving average over the simulated samples only.  So every harmful TRIPPED")
    say("episode had reached >= 0.5 N m before or within the 20-ms control step of the trip: category (b) = all tripped harmful")
    say("first attempts.  How long before the trip the load crossed 0.5 N m is not stored.  A real retraction after the trip")
    say("is not simulated, so (b) is if anything an underestimate of the load in those episodes.")
    open(os.path.join(TOP, "out", "posthoc_guard_trips.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
