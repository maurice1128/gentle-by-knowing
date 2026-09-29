"""Round 20 (shoulder route): hybrid of the body model and reactive feel.
The model (population body + care record, no probing - round 19 showed probing
does not help) decides which lift to start with and ranks the alternatives;
every attempt runs under a wrist-force guard and is abandoned when the force
exceeds what that lift normally needs; the next model-ranked lift is tried.

Stages (argv[1]):
  guards    per-strategy force guard from the library of unrestricted bodies
  devgrid   guarded 48-strategy grid on the 160 development persons (rounds 18 + 18R)
  test      new persons: truth grid + guarded grid + the round-17 reactive episodes
  eval      (WM_R20_DEV=1: development set)
"""
import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "round20")
R17 = os.path.join(TOP, "out", "round17")
R18 = os.path.join(TOP, "out", "round18")
sys.path.insert(0, ROOT)

import round17 as r17
import round18 as r18
from round17 import STRATS, UNRESTRICTED, person, harm, PAINFREE, RETRACT_S

TEST_POOL = list(range(802, 1500))
N_TEST = 120
GUARD_MULT = 1.2           # same formula as the reactive baseline of round 17: 1.2 x 95th percentile of harmless force


def guards():
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(os.path.join(R18, "library.json")))["rows"]
    g = {}
    for k in STRATS:
        f = [r["peak_force"] for r in rows if r.get("kind") == "truth" and "error" not in r and r["done"]
             and (r["offset"], r["angle"], r["speed"]) == k]
        g[str(k)] = GUARD_MULT * float(np.percentile(f, 95)) if len(f) >= 20 else None
    json.dump(g, open(os.path.join(OUT, "guards.json"), "w"), indent=1)
    print("guards for", sum(v is not None for v in g.values()), "of", len(g), "strategies; median", np.median([v for v in g.values() if v]))


def _guard_jobs(seeds):
    g = json.load(open(os.path.join(OUT, "guards.json")))
    return [("guard", s) + person(s) + k + (g[str(k)],) for s in seeds for k in STRATS if g[str(k)]]


def devgrid():
    path = os.path.join(OUT, "dev_guard.json")
    if os.path.exists(path):
        print("exists", path); return
    dev = json.load(open(os.path.join(R18, "test_seeds.json"))) + json.load(open(os.path.join(R18, "test_seeds_rep.json")))
    json.dump(dict(rows=r17._pool(r18._job, _guard_jobs(dev), f"dev guarded grid ({len(dev)} persons)")), open(path, "w"))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    f = dict(guards=guards, devgrid=devgrid).get(stage)
    if f is None:
        print(__doc__); return 1
    os.makedirs(OUT, exist_ok=True)
    f(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
