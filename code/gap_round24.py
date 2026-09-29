"""Decomposition of the gap between the learned policy L and the oracle O in round 24 (record error 20 deg,
harmful >= 0.5 N m): missed safe completions = unnecessary refusals + harmful lifts where a safe strategy
existed + incomplete lifts where a safe strategy existed; and L's harmful outcomes in persons with no safe
strategy.  Reads out/round24 only; no simulation.  (Post hoc; reported in the paper's results.)"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from round21 import _pol
from dev_r24_learned import load_train, harm, Learned

THR = 0.5


def main():
    F, _, _ = _pol()
    out = os.path.join(TOP, "out", "round24")
    te = json.load(open(os.path.join(out, "seeds.json")))
    T = {}
    for x in json.load(open(os.path.join(out, "test.json")))["rows"]:
        if "error" not in x:
            T.setdefault(x["seed"], {})[(x["offset"], x["angle"], x["speed"])] = x
    tau = json.load(open(os.path.join(out, "policy.json")))["tau"]["20"]
    Tr = load_train(); L = Learned(Tr, sorted(Tr), 20.0)
    has_safe = {s: any(x["done"] and harm(x) < THR for x in T[s].values()) for s in te}
    unnec = harm_safe = incompl = harm_nosafe = safe = 0
    for s in te:
        k = L.act(s, T[s], 20.0, tau, F)
        if k is None:
            unnec += has_safe[s]; continue
        x = T[s][k]; h = harm(x)
        if x["done"] and h < THR:
            safe += 1
        elif h >= THR:
            harm_safe += has_safe[s]; harm_nosafe += not has_safe[s]
        else:
            incompl += has_safe[s]
    print(f"round 24, eps 20: oracle safe {sum(has_safe.values())}, L safe {safe}, missed {sum(has_safe.values()) - safe}")
    print(f"  unnecessary refusals {unnec}; harmful where safe existed {harm_safe}; incomplete where safe existed {incompl}")
    print(f"  L harmful where no safe strategy existed {harm_nosafe} (of {harm_safe + harm_nosafe} harmful)")


if __name__ == "__main__":
    main()
