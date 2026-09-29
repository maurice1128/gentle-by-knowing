"""Why did round 18 (n=40) support H18-1 and the frozen replication (n=120) not?  Diagnostics only."""
import os, sys, json, math
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); TOP = os.path.dirname(ROOT); sys.path.insert(0, ROOT)
import round18 as r
from round17 import person, harm, UNRESTRICTED
OUT = os.path.join(TOP, "out", "round18"); R17 = os.path.join(TOP, "out", "round17")
pol = json.load(open(os.path.join(R17, "policy.json"))); F = tuple(pol["F"]); order = [tuple(k) for k in pol["order"]]
seeds, feats, G, mu, sd = r._load_library()

def load(suf):
    te = json.load(open(os.path.join(OUT, f"test_seeds{suf}.json")))
    rows = json.load(open(os.path.join(OUT, f"test{suf}.json")))["rows"]
    T, Gd = {}, {}
    for x in rows:
        k = (x["offset"], x["angle"], x["speed"])
        (T if x["kind"] == "truth" else Gd if x["kind"] == "guard" else {}).setdefault(x["seed"], {})[k] = x
    pf = r._features([x for x in rows if x["kind"] == "probe"])
    return te, T, Gd, pf

def choose(P, rec):
    fs, fe = rec
    cand = [(k, p) for k, p in P.items() if p[2] >= 0.8]
    over = lambda p: max(p[0] - (fs - r.MARGIN_DEG) if fs < UNRESTRICTED else -1e9, p[1] - (fe - r.MARGIN_DEG) if fe < UNRESTRICTED else -1e9)
    ok = [(p[3], p[4], k) for k, p in cand if over(p) <= 0]
    return min(ok)[2] if ok else min((over(p), p[3], k) for k, p in cand)[2]

def per_person(suf):
    te, T, Gd, pf = load(suf)
    out = []
    for s in te:
        P = r._predict(pf[s], seeds, feats, G, mu, sd, k_nn=r.K_NN)
        k = choose(P, r.recorded(s, 4.0)); t = T[s][k]; f = T[s][F]
        hr = 0.0
        for kk in order:
            g = Gd[s].get(kk)
            if g is None: continue
            hr = max(hr, harm(g))
            if g["done"]: break
        fs, fe = person(s)
        typ = ("S" if fs < 90 else "") + ("E" if fe < 90 else "") or "N"
        out.append(dict(seed=s, typ=typ, fs=fs, fe=fe, hM=max(harm(t), pf[s]["probe_harm"]), hF=harm(f), hR=hr, doneM=t["done"], doneF=f["done"], pick=k))
    return out

A, B = per_person(""), per_person("_rep")
for name, X in (("round 18 (n=40)", A), ("round 18R (n=120)", B)):
    d = np.array([x["hM"] - x["hF"] for x in X])
    from collections import Counter
    print(f"\n{name}: types {dict(Counter(x['typ'] for x in X))}")
    print(f"  mean harm F {np.mean([x['hF'] for x in X]):.2f}  M_id {np.mean([x['hM'] for x in X]):.2f}  R {np.mean([x['hR'] for x in X]):.2f};  diff M-F mean {d.mean():+.2f}, median {np.median(d):+.2f}, SD {d.std(ddof=1):.2f}")
    print(f"  persons with |diff| > 5 N m: {int((np.abs(d) > 5).sum())}  -> M better in {int((d < -5).sum())}, F better in {int((d > 5).sum())};  diffs exactly 0: {int((d == 0).sum())}")
    big = sorted(X, key=lambda x: x["hM"] - x["hF"])
    print("  largest M wins :", [(x["seed"], x["typ"], round(x["fs"]), round(x["fe"]), round(x["hF"], 1), round(x["hM"], 1)) for x in big[:5]])
    print("  largest M losses:", [(x["seed"], x["typ"], round(x["fs"]), round(x["fe"]), round(x["hF"], 1), round(x["hM"], 1)) for x in big[-5:]])
    for t in ("S", "E", "SE", "N"):
        dd = [x["hM"] - x["hF"] for x in X if x["typ"] == t]
        if dd: print(f"    type {t:2s} n={len(dd):3d}  mean diff {np.mean(dd):+.2f}  F harm {np.mean([x['hF'] for x in X if x['typ'] == t]):.2f}")

# how often does a random subset of 40 of the 120 look like round 18?
rng = np.random.default_rng(0)
dB = np.array([x["hM"] - x["hF"] for x in B]); n = 0; sig = 0; m = []
for _ in range(20000):
    s = rng.choice(dB, 40, replace=False); mm = s.mean(); se = s.std(ddof=1) / math.sqrt(40)
    m.append(mm); sig += (mm + 2.1 * se < 0); n += (mm <= -3.80)
print(f"\nrandom 40 of the 120: P(CI excludes 0 in favour of M) = {sig / 20000:.1%};  P(mean diff <= -3.80) = {n / 20000:.2%};  subset mean: 5th-95th pct {np.percentile(m, 5):+.2f} .. {np.percentile(m, 95):+.2f}")
dA = np.array([x["hM"] - x["hF"] for x in A])
pooled = np.concatenate([dA, dB]); se = pooled.std(ddof=1) / math.sqrt(len(pooled))
print(f"pooled n=160 (secondary, labelled): mean diff {pooled.mean():+.2f} [{pooled.mean() - 1.98 * se:+.2f}, {pooled.mean() + 1.98 * se:+.2f}]")
# difference between the two sets, Welch
se2 = math.sqrt(dA.var(ddof=1) / len(dA) + dB.var(ddof=1) / len(dB))
print(f"difference between the two sets' effects: {dA.mean() - dB.mean():+.2f}, z = {(dA.mean() - dB.mean()) / se2:.2f}")

print("\n--- subgroup CIs (post hoc) and the shoulder free range of the S persons ---")
for name, X in (("n=40", A), ("n=120", B)):
    for t in ("S", "E", "SE"):
        dd = np.array([x["hM"] - x["hF"] for x in X if x["typ"] == t]); dr = np.array([x["hM"] - x["hR"] for x in X if x["typ"] == t])
        if len(dd) > 2:
            se = dd.std(ddof=1) / math.sqrt(len(dd)); ser = dr.std(ddof=1) / math.sqrt(len(dr)); tt = 2.2 if len(dd) < 15 else 2.03
            print(f"  {name} type {t:2s} n={len(dd):2d}: M-F {dd.mean():+.2f} [{dd.mean() - tt * se:+.2f}, {dd.mean() + tt * se:+.2f}]   M-R {dr.mean():+.2f} [{dr.mean() - tt * ser:+.2f}, {dr.mean() + tt * ser:+.2f}]   done M {sum(x['doneM'] for x in X if x['typ'] == t)} F {sum(x['doneF'] for x in X if x['typ'] == t)}")
    fs = sorted(round(x["fs"], 1) for x in X if x["typ"] == "S")
    print(f"  {name} S persons' shoulder free range: median {np.median(fs):.1f}, share < 9 deg {np.mean(np.array(fs) < 9):.0%}")
print("\nF harm against shoulder free range, S persons, both sets:")
for lo, hi in ((4, 7), (7, 10), (10, 13), (13, 16)):
    v = [(x["hF"], x["hM"]) for x in A + B if x["typ"] == "S" and lo <= x["fs"] < hi]
    print(f"  free {lo:2d}-{hi:2d} deg  n={len(v):2d}  F {np.mean([a for a, b in v]):6.2f}   M_id {np.mean([b for a, b in v]):6.2f}")
