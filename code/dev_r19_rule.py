"""Round 19 rule development on the 160 persons already seen (rounds 18 + 18R = development data now).
Variants: predicted-completion threshold, and abstaining when no strategy is predicted to fit."""
import os, sys, json, math
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); TOP = os.path.dirname(ROOT); sys.path.insert(0, ROOT)
import round18 as r
from round17 import person, harm, UNRESTRICTED
from check_r18_discrepancy import load, F, order, seeds, feats, G, mu, sd

def choose(P, rec, comp_thr, abstain_over):
    fs, fe = rec
    cand = [(k, p) for k, p in P.items() if p[2] >= comp_thr]
    over = lambda p: max(p[0] - (fs - r.MARGIN_DEG) if fs < UNRESTRICTED else -1e9, p[1] - (fe - r.MARGIN_DEG) if fe < UNRESTRICTED else -1e9)
    ok = [(p[3], p[4], k) for k, p in cand if over(p) <= 0]
    if ok: return min(ok)[2]
    if not cand: return None
    o, _, k = min((over(p), p[3], k) for k, p in cand)
    return None if (abstain_over is not None and o > abstain_over) else k

data = []
for suf in ("", "_rep"):
    te, T, Gd, pf = load(suf)
    for s in te:
        data.append((s, T[s], Gd[s], pf[s], r._predict(pf[s], seeds, feats, G, mu, sd, k_nn=r.K_NN)))

def typ(s):
    fs, fe = person(s); return ("S" if fs < 90 else "") + ("E" if fe < 90 else "") or "N"

print(f"{'variant':34s} {'group':5s} {'n':>3} | {'M harm':>7} {'F harm':>7} {'R harm':>7} | {'M-F':>22} | {'done M / F / R':>16} | abstain")
for comp_thr in (0.8, 1.0):
    for ab in (None, 3.0, 0.0):
        for grp in ("S+E", "SE", "all"):
            rowsM, rowsF, rowsR = [], [], []
            for s, T, Gd, pfs, P in data:
                t = typ(s)
                if (grp == "S+E" and t not in ("S", "E")) or (grp == "SE" and t != "SE"): continue
                k = choose(P, r.recorded(s, 4.0), comp_thr, ab)
                if k is None: hm, dm, abst = pfs["probe_harm"], False, True
                else: hm, dm, abst = max(harm(T[k]), pfs["probe_harm"]), T[k]["done"], False
                hr, dr = 0.0, False
                for kk in order:
                    g = Gd.get(kk)
                    if g is None: continue
                    hr = max(hr, harm(g))
                    if g["done"]: dr = True; break
                rowsM.append((hm, dm, abst)); rowsF.append((harm(T[F]), T[F]["done"])); rowsR.append((hr, dr))
            d = np.array([m[0] - f[0] for m, f in zip(rowsM, rowsF)]); se = d.std(ddof=1) / math.sqrt(len(d))
            print(f"comp>={comp_thr:.1f} abstain_if_over>{str(ab):4s}        {grp:5s} {len(d):3d} | {np.mean([m[0] for m in rowsM]):7.2f} {np.mean([f[0] for f in rowsF]):7.2f} {np.mean([x[0] for x in rowsR]):7.2f} | "
                  f"{d.mean():+6.2f} [{d.mean() - 2 * se:+6.2f},{d.mean() + 2 * se:+6.2f}] | {sum(m[1] for m in rowsM):4d} /{sum(f[1] for f in rowsF):4d} /{sum(x[1] for x in rowsR):4d} | {sum(m[2] for m in rowsM)}")

print("\n=== outcome classes: safe completion = done & end-range load < 0.5 N m; harmful = load >= 0.5 N m (done or not) ===")
print(f"{'variant':30s} {'group':5s} {'n':>3} | safe completions M / F / R | harmful outcomes M / F / R | mean harm M / F / R")
for comp_thr, ab in ((0.8, None), (0.8, 3.0), (1.0, 3.0), (1.0, 0.0)):
    for grp in ("S+E", "SE", "N", "all"):
        sc = dict(M=0, F=0, R=0); hz = dict(M=0, F=0, R=0); hh = dict(M=[], F=[], R=[]); n = 0
        for s, T, Gd, pfs, P in data:
            t = typ(s)
            if (grp == "S+E" and t not in ("S", "E")) or (grp in ("SE", "N") and t != grp): continue
            n += 1
            k = choose(P, r.recorded(s, 4.0), comp_thr, ab)
            hm, dm = (pfs["probe_harm"], False) if k is None else (max(harm(T[k]), pfs["probe_harm"]), T[k]["done"])
            hr, dr = 0.0, False
            for kk in order:
                g = Gd.get(kk)
                if g is None: continue
                hr = max(hr, harm(g))
                if g["done"]: dr = True; break
            for a, h, dn in (("M", hm, dm), ("F", harm(T[F]), T[F]["done"]), ("R", hr, dr)):
                sc[a] += (dn and h < 0.5); hz[a] += (h >= 0.5); hh[a].append(h)
        print(f"comp>={comp_thr:.1f} abstain>{str(ab):4s}          {grp:5s} {n:3d} | {sc['M']:10d} /{sc['F']:4d} /{sc['R']:4d}   | {hz['M']:10d} /{hz['F']:4d} /{hz['R']:4d}    | {np.mean(hh['M']):6.2f} /{np.mean(hh['F']):6.2f} /{np.mean(hh['R']):6.2f}")

def choose2(P, rec, ab):
    """F-first: keep the fixed rule whenever the identified body predicts it fits; otherwise the fitting strategy
    most likely to complete (then lowest shear); abstain when nothing is predicted to fit within `ab` degrees."""
    fs, fe = rec
    over = lambda p: max(p[0] - (fs - r.MARGIN_DEG) if fs < UNRESTRICTED else -1e9, p[1] - (fe - r.MARGIN_DEG) if fe < UNRESTRICTED else -1e9)
    if F in P and over(P[F]) <= 0: return F
    if fs >= UNRESTRICTED and fe >= UNRESTRICTED: return F
    cand = [(k, p) for k, p in P.items() if p[2] >= 0.8]
    ok = [(-p[2], p[3], p[4], k) for k, p in cand if over(p) <= 0]
    if ok: return min(ok)[3]
    if not cand: return None
    o, _, _, k = min((over(p), -p[2], p[3], k) for k, p in cand)
    return None if (ab is not None and o > ab) else k

print("\n=== F-first rule ===")
print(f"{'variant':22s} {'group':5s} {'n':>3} | safe completions M / F / R | harmful outcomes M / F / R | mean harm M / F / R | M-F harm [approx 95% CI] | abstain")
for ab in (None, 3.0):
    for grp in ("S+E", "SE", "N", "all"):
        sc = dict(M=0, F=0, R=0); hz = dict(M=0, F=0, R=0); hh = dict(M=[], F=[], R=[]); n = 0; na = 0
        for s, T, Gd, pfs, P in data:
            t = typ(s)
            if (grp == "S+E" and t not in ("S", "E")) or (grp in ("SE", "N") and t != grp): continue
            n += 1
            k = choose2(P, r.recorded(s, 4.0), ab); na += k is None
            probe = 0.0 if (k == F and min(person(s)) >= UNRESTRICTED) else pfs["probe_harm"]      # unrestricted record -> no probing needed
            hm, dm = (probe, False) if k is None else (max(harm(T[k]), probe), T[k]["done"])
            hr, dr = 0.0, False
            for kk in order:
                g = Gd.get(kk)
                if g is None: continue
                hr = max(hr, harm(g))
                if g["done"]: dr = True; break
            for a, h, dn in (("M", hm, dm), ("F", harm(T[F]), T[F]["done"]), ("R", hr, dr)):
                sc[a] += (dn and h < 0.5); hz[a] += (h >= 0.5); hh[a].append(h)
        d = np.array(hh["M"]) - np.array(hh["F"]); se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0
        print(f"F-first abstain>{str(ab):4s}   {grp:5s} {n:3d} | {sc['M']:10d} /{sc['F']:4d} /{sc['R']:4d}   | {hz['M']:10d} /{hz['F']:4d} /{hz['R']:4d}    | {np.mean(hh['M']):6.2f} /{np.mean(hh['F']):6.2f} /{np.mean(hh['R']):6.2f} | {d.mean():+6.2f} [{d.mean() - 2 * se:+6.2f},{d.mean() + 2 * se:+6.2f}] | {na}")
