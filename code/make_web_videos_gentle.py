"""Web videos for the gentle-care project page (2026-09-29).

Two steps, as in make_figure_scene.py:
  1. capture (run with ../.venv_myo: MuJoCo + MyoSuite)
       python make_web_videos_gentle.py capture NAME [NAME ...]
       python make_web_videos_gentle.py probe SEED [SEED ...]      # guard timing, no frames
     Each episode is the UNMODIFIED shoulder_task.run(); frames and signals are recorded through
     read-only hooks (rendering and limit_torque do not change the dynamics).  The returned run()
     dict is stored next to the frames, and the live harm curve is recomputed from the recorded
     per-substep joint-limit torques exactly as run() does (skip n0 samples, 25-sample = 50 ms
     moving average, subtract the pre-contact baseline, clip at 0, sum shoulder + elbow peaks);
     the script asserts that this equals run()'s harm.
  2. compose (run with ../.venv_mm: matplotlib + PIL; ffmpeg on PATH)
       python make_web_videos_gentle.py compose
Output: out/videos/*.mp4, *_poster.png, raw/*.npz, raw/*.json
"""
import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
OUT = os.path.join(TOP, "out", "videos")
RAW = os.path.join(OUT, "raw")
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")

import numpy as np

PW, PH = 632, 474          # panel (render) size
EVERY = 4                  # render every 4th physics substep (dt 2 ms -> 8 ms of simulated time)
FPS = 30
SLOW = 0.25                # playback factor (0.25x real time)
THR = 0.5                  # N m, harmful threshold
F = (0.03, -40.0, 0.1)
GUARD_P95_MULT = 0.5       # round 21: guard[k] * 0.5 = 0.6 x p95 of harmless wrist force

# episodes: name -> (seed, free_sh, free_el, strategy)   free ranges from round17.person(seed)
def episodes():
    from round17 import person
    E = {"prox104": (104, 90.0, 90.0, (-0.06, -40.0, 0.1)),
         "dist104": (104, 90.0, 90.0, (0.12, -40.0, 0.1))}
    sel = json.load(open(os.path.join(RAW, "selection.json"))) if os.path.exists(os.path.join(RAW, "selection.json")) else {}
    if "switch" in sel:
        s = sel["switch"]["seed"]
        E["swF"] = (s,) + person(s) + (F,)
        E["swL"] = (s,) + person(s) + (tuple(sel["switch"]["L"]),)
    if "decline" in sel:
        s = sel["decline"]["seed"]
        E["decF"] = (s,) + person(s) + (F,)
    if "guard" in sel:
        s = sel["guard"]["seed"]
        E["guardF"] = (s,) + person(s) + (F,)
    return E


def harm_of(x):
    return max(0.0, x["lim_sh"] - x.get("lim_base_sh", 0.0)) + max(0.0, x["lim_el"] - x.get("lim_base_el", 0.0))


def guard_threshold():
    from round21 import _pol
    Fp, _, guard = _pol()
    return float(guard[Fp] * GUARD_P95_MULT)


# ---------------------------------------------------------------- capture (myo venv)
def capture(seed, fs, fe, strat, render=True):
    import mujoco
    import shoulder_task as st
    e = st.env()
    inner_h, inner_step = e._instantaneous_harm, e.step
    S, frames, ftimes = [], [], []
    R = {}
    state = dict(active=False)

    def snap():
        if "r" not in R:
            R["r"] = mujoco.Renderer(e.m, PH, PW)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = np.array(e.d.geom_xpos[e._fg])
            cam.azimuth, cam.elevation, cam.distance = 180.0, -15.0, 0.75
            R["cam"] = cam
        R["r"].update_scene(e.d, R["cam"])
        return R["r"].render().copy()

    def hook():
        h = inner_h()
        if state["active"]:
            d, m = e.d, e.m
            sh, el = st.limit_torque(e)
            S.append((float(h["force"]), sh, el,
                      float(d.qpos[m.jnt_qposadr[e._sj]]), float(d.qpos[m.jnt_qposadr[e._ej]]),
                      float(d.geom_xpos[e._fg][2])))
            if render and len(S) % EVERY == 0:
                frames.append(snap()); ftimes.append(len(S) * e.dt)
        return h

    def step(*a, **k):
        if not state["active"]:
            if render:
                frames.append(snap()); ftimes.append(0.0)
            state["active"] = True
        return inner_step(*a, **k)

    e._instantaneous_harm = hook
    e.step = step
    try:
        r = st.run(seed, fs, fe, *strat)
    finally:
        e._instantaneous_harm = inner_h
        del e.step                       # back to the class method
        if "r" in R:
            R["r"].close()
    S = np.array(S)
    sig = live_signals(S, r, e.dt, e.substeps)
    r["harm"] = harm_of(r)
    assert abs(sig["harm_recomputed"] - r["harm"]) < 1e-9, (sig["harm_recomputed"], r["harm"])
    return r, S, np.array(frames) if render else None, np.array(ftimes), float(e.dt), int(e.substeps)


def live_signals(S, r, dt, substeps):
    """Per-substep robot-caused joint-limit load, computed exactly as shoulder_task.run() does."""
    n = len(S)
    n0 = min(50, max(0, n - 30))
    ker = np.ones(25) / 25.0
    out = {}
    tot_peak = 0.0
    inst = np.zeros(n); peak = np.zeros(n)
    for c, (col, base) in enumerate(((1, r["lim_base_sh"]), (2, r["lim_base_el"]))):
        x = S[n0:, col]
        sm = np.convolve(x, ker, mode="valid") if len(x) >= 25 else np.array([float(np.mean(x))])
        v = np.zeros(n)
        v[n0 + 24: n0 + 24 + len(sm)] = sm            # value available at the last sample of each window
        rc = np.clip(v - base, 0.0, None)
        rc[: n0 + 24] = 0.0                           # not counted by the metric
        inst += rc
        pk = np.maximum.accumulate(rc); peak += pk
        tot_peak += max(0.0, float(sm.max()) - base)
        out["sm_" + ("sh" if c == 0 else "el")] = rc
    out.update(t=(np.arange(n) + 1) * dt, inst=inst, peak=peak, harm_recomputed=tot_peak, n0=n0,
               excluded_raw_max=float(np.abs(S[:n0 + 24, 1:3]).max()) if n else 0.0)
    # guard semantics: run() checks the per-control-step peak wrist force after each step
    stepf = S[: (n // substeps) * substeps, 0].reshape(-1, substeps).max(1) if n >= substeps else np.array([])
    out["step_force"] = stepf
    return out


def save_episode(name, spec):
    seed, fs, fe, strat = spec
    r, S, fr, ft, dt, sub = capture(seed, fs, fe, strat)
    os.makedirs(RAW, exist_ok=True)
    np.savez_compressed(os.path.join(RAW, name + ".npz"), frames=fr, ftimes=ft, S=S, dt=dt, substeps=sub)
    json.dump(r, open(os.path.join(RAW, name + ".json"), "w"), indent=1)
    print(name, "seed", seed, "strat", strat, "harm", r["harm"], "done", r["done"], "frames", len(fr),
          "d_sh", r["d_sh_deg"], "d_el", r["d_el_deg"], flush=True)


def probe(seed):
    from round17 import person
    fs, fe = person(seed)
    r, S, _, _, dt, sub = capture(seed, fs, fe, F, render=False)
    sig = live_signals(S, r, dt, sub)
    g = guard_threshold()
    k_load = np.nonzero(sig["inst"] >= THR)[0]
    t_load = float(sig["t"][k_load[0]]) if len(k_load) else None
    k_trip = np.nonzero(sig["step_force"] > g)[0]
    t_trip = float((k_trip[0] + 1) * sub * dt) if len(k_trip) else None
    load_at_trip = float(sig["peak"][min(len(S), (k_trip[0] + 1) * sub) - 1]) if len(k_trip) else None
    print(json.dumps(dict(seed=seed, free=(fs, fe), harm=r["harm"], done=r["done"], t_end=r["t_end"],
                          guard=g, t_load_ge_0_5=t_load, t_guard_trip=t_trip, peak_load_at_trip=load_at_trip,
                          excluded_raw_max=sig["excluded_raw_max"])), flush=True)


# ---------------------------------------------------------------- compose (mm venv)
def font(sz, bold=False):
    from PIL import ImageFont
    for c in (("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
              "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(c, sz)
        except Exception:
            pass
    return ImageFont.load_default()


BG = (250, 250, 248); INK = (25, 25, 25); MUTED = (95, 95, 95); RED = (200, 45, 35); BLUE = (35, 95, 170)
GREEN = (30, 130, 70)


def load(name):
    z = np.load(os.path.join(RAW, name + ".npz"))
    r = json.load(open(os.path.join(RAW, name + ".json")))
    sig = live_signals(z["S"], r, float(z["dt"]), int(z["substeps"]))
    return dict(frames=z["frames"], ftimes=z["ftimes"], S=z["S"], dt=float(z["dt"]), sub=int(z["substeps"]), r=r, sig=sig)


def timeline(eps, intro=1.0, outro=1.5):
    """Video frame -> simulated time.  intro: still pre-lift frame; outro: hold the last frame."""
    T = max(ep["ftimes"][-1] for ep in eps)
    n_intro, n_play, n_out = int(intro * FPS), int(math.ceil(T / SLOW * FPS)) + 1, int(outro * FPS)
    ts = [None] * n_intro + [min(T, k / FPS * SLOW) for k in range(n_play)] + [T] * n_out
    return ts


def at(ep, t):
    """index of the frame / substep shown at simulated time t (None = pre-lift)."""
    if t is None:
        return 0, -1
    fi = int(np.searchsorted(ep["ftimes"], t + 1e-9, side="right") - 1)
    fi = max(0, min(fi, len(ep["frames"]) - 1))
    si = int(round(ep["ftimes"][fi] / ep["dt"])) - 1
    if t >= ep["ftimes"][-1] - 1e-9:
        si = len(ep["S"]) - 1          # end of episode: final state = the values run() reports
    return fi, min(si, len(ep["S"]) - 1)


def draw_bar(dr, x, y, w, val, peak, vmax, fs):
    col = RED if peak >= THR else BLUE
    dr.rectangle([x, y, x + w, y + 14], outline=(150, 150, 150), fill=(235, 235, 232))
    dr.rectangle([x, y, x + int(w * min(val, vmax) / vmax), y + 14], fill=col)
    tx = x + int(w * THR / vmax)
    dr.line([tx, y - 4, tx, y + 18], fill=RED, width=2)
    dr.text((tx + 3, y + 16), "0.5 harm threshold", fill=RED, font=fs)


def side_by_side(name, eps, titles, subtitles, header, footer, readout, right_card=None):
    from PIL import Image, ImageDraw
    fT, fS, fR, fH = font(22, True), font(15), font(18, True), font(15)
    pad, top, bot = 8, 30 + 58, 120
    Wv = 2 * PW + 3 * pad; Hv = top + PH + bot
    Wv += Wv % 2; Hv += Hv % 2
    ts = timeline([ep for ep in eps if ep is not None])
    out = []
    for t in ts:
        im = Image.new("RGB", (Wv, Hv), BG); dr = ImageDraw.Draw(im)
        dr.text((pad, 5), header + ("" if t is not None else "   |   before contact"), fill=MUTED, font=fH)
        for i in range(2):
            x = pad + i * (PW + pad)
            dr.text((x + 2, 28), titles[i], fill=INK, font=fT)
            dr.text((x + 2, 58), subtitles[i], fill=MUTED, font=fS)
            ep = eps[i]
            if ep is None:
                im.paste(right_card, (x, top))
                continue
            fi, si = at(ep, t)
            im.paste(Image.fromarray(ep["frames"][fi]), (x, top))
            tt = 0.0 if t is None else min(t, ep["ftimes"][-1])
            dr.text((x + 8, top + 6), f"t = {tt:.2f} s (sim)", fill=(255, 255, 255), font=fS, stroke_width=2, stroke_fill=(0, 0, 0))
            if t is not None and t >= ep["ftimes"][-1] - 1e-9:
                msg = "lift complete: raised 5 cm, held 0.5 s" if ep["r"]["done"] else "episode ended (not completed)"
                dr.text((x + 8, top + PH - 28), msg, fill=(255, 255, 255), font=fS, stroke_width=2, stroke_fill=(0, 0, 0))
            readout(dr, x, top + PH + 8, ep, si)
        dr.text((pad, Hv - 22), footer, fill=MUTED, font=font(13))
        out.append(im)
    encode(out, name)


def encode(frames, name, crf=23):
    import subprocess
    w, h = frames[0].size
    mp4 = os.path.join(OUT, name + ".mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-preset", "slow", "-movflags", "+faststart", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    # poster: the last frame (end state with final numbers)
    frames[-1].save(os.path.join(OUT, name + "_poster.png"), optimize=True)
    print("wrote", mp4, len(frames), "frames", f"{len(frames) / FPS:.1f} s", f"{os.path.getsize(mp4) / 1e6:.2f} MB", flush=True)


def fmt_strat(k):
    o, a, s = k
    where = f"{abs(o + 0.03) * 100:.0f} cm {'distal' if o + 0.03 > 0 else 'proximal'} of forearm mid-shaft" if abs(o + 0.03) > 1e-9 else "at forearm mid-shaft"
    return f"support {where}, lift angle {a:+.0f}\u00b0, {s:.2f} m/s"


def compose():
    from PIL import ImageDraw
    sel = json.load(open(os.path.join(RAW, "selection.json")))
    meta = {}
    slow = f"{SLOW:g}\u00d7 speed"
    sim = "MuJoCo + MyoSuite myoArm simulation"

    # ---- 1: proximal vs distal
    P, D = load("prox104"), load("dist104")

    def angles(dr, x, y, ep, si):
        fS, fR = font(15), font(20, True)
        r = ep["r"]
        if si < 0:
            dsh = del_ = 0.0
        else:
            dsh = math.degrees(ep["S"][si, 3]) - r["rest_sh_deg"]; del_ = math.degrees(ep["S"][si, 4]) - r["rest_el_deg"]
        dr.text((x + 4, y), f"shoulder elevation \u0394 {dsh:+5.1f}\u00b0", fill=INK, font=fR)
        dr.text((x + 4 + 320, y), f"elbow flexion \u0394 {del_:+5.1f}\u00b0", fill=INK, font=fR)
        rise = 0.0 if si < 0 else (ep["S"][si, 5] - ep["S"][0, 5])
        dr.text((x + 4, y + 30), "joint-angle change from the resting posture, measured in the simulation", fill=MUTED, font=fS)
    ps, pe, ds, de = P["r"]["d_sh_deg"], P["r"]["d_el_deg"], D["r"]["d_sh_deg"], D["r"]["d_el_deg"]
    t1 = "Support near the elbow \u2192 shoulder moves" if abs(ps) > abs(pe) else "Support near the elbow"
    t2 = "Support near the wrist \u2192 elbow also bends" if abs(de) > abs(pe) else "Support near the wrist"
    side_by_side("lift_proximal_vs_distal", [P, D], [t1, t2],
                 [fmt_strat((P["r"]["offset"], P["r"]["angle"], P["r"]["speed"])),
                  fmt_strat((D["r"]["offset"], D["r"]["angle"], D["r"]["speed"]))],
                 f"{sim}  |  same simulated person (seed 104, no joint restriction)  |  {slow}",
                 "Task: raise the forearm 5 cm and hold 0.5 s. Only the support site differs.", angles)
    meta["lift_proximal_vs_distal"] = dict(seed=104, prox=P["r"], dist=D["r"])

    # ---- harm readout (videos 2, 3)
    def harm_readout(vmax):
        def f(dr, x, y, ep, si):
            fS, fR = font(14), font(20, True)
            v = 0.0 if si < 0 else float(ep["sig"]["inst"][si]); pk = 0.0 if si < 0 else float(ep["sig"]["peak"][si])
            col = RED if pk >= THR else INK
            dr.text((x + 4, y - 2), f"load pressed into joint limit: {v:.2f} N\u00b7m", fill=RED if v >= THR else INK, font=fR)
            dr.text((x + 400, y + 1), f"peak so far {pk:.2f} N\u00b7m", fill=col, font=font(17, True))
            draw_bar(dr, x + 6, y + 32, PW - 20, v, pk, vmax, fS)
            dr.text((x + 4, y + 68), "robot-caused (baseline-subtracted), 50 ms average, shoulder + elbow", fill=MUTED, font=fS)
        return f

    sw = sel["switch"]; A, B = load("swF"), load("swL")
    fs, fe = sw["free"]
    who = ("shoulder" if fs < 90 else "") + (" and " if fs < 90 and fe < 90 else "") + ("elbow" if fe < 90 else "")
    rng = ", ".join(([f"shoulder {fs:.1f}\u00b0"] if fs < 90 else []) + ([f"elbow {fe:.1f}\u00b0"] if fe < 90 else []))
    vmax = max(1.0, math.ceil(max(A["r"]["harm"], B["r"]["harm"]) * 1.15))
    side_by_side("fixed_vs_learned", [A, B], ["Fixed lift (no knowledge)", "Learned policy (knows which joint is restricted)"],
                 [fmt_strat(F), fmt_strat(tuple(sw["L"]))],
                 f"{sim}  |  simulated person seed {sw['seed']}: restricted {who} (free range {rng})  |  {slow}",
                 "Harm = torque the robot presses into the person's joint limit (not visible to the robot). Harmful if \u2265 0.5 N\u00b7m.",
                 harm_readout(vmax))
    meta["fixed_vs_learned"] = dict(seed=sw["seed"], free=sw["free"], F=A["r"], L=B["r"], stored=sw)

    # ---- 2b: declines
    if "decline" in sel:
        from PIL import Image
        dc = sel["decline"]; A = load("decF")
        card = Image.new("RGB", (PW, PH), (238, 242, 238)); cd = ImageDraw.Draw(card)
        cd.text((40, 150), "Learned policy: declines,", fill=GREEN, font=font(34, True))
        cd.text((40, 195), "asks a carer", fill=GREEN, font=font(34, True))
        cd.text((40, 265), "The care record shows both the shoulder and", fill=INK, font=font(18))
        cd.text((40, 290), "the elbow restricted. No lift was predicted safe", fill=INK, font=font(18))
        cd.text((40, 315), f"for \u2265 {dc['tau'] * 100:.0f}% of similar persons (best: {dc['best_score'] * 100:.0f}%),", fill=INK, font=font(18))
        cd.text((40, 340), "so the robot does not lift.", fill=INK, font=font(18))
        cd.text((40, 380), "load pressed into joint limit: 0.00 N\u00b7m (no contact)", fill=INK, font=font(17, True))
        fs, fe = dc["free"]
        vmax = max(1.0, math.ceil(A["r"]["harm"] * 1.15))
        side_by_side("learned_declines", [A, None], ["Fixed lift (no knowledge)", "Learned policy"],
                     [fmt_strat(F), "no lift attempted"],
                     f"{sim}  |  simulated person seed {dc['seed']}: shoulder {fs:.1f}\u00b0 and elbow {fe:.1f}\u00b0 free range  |  {slow}",
                     "Harm = torque the robot presses into the person's joint limit (not visible to the robot). Harmful if \u2265 0.5 N\u00b7m.",
                     harm_readout(vmax), right_card=card)
        meta["learned_declines"] = dict(seed=dc["seed"], free=dc["free"], F=A["r"], stored=dc)

    # ---- 3: force guard
    meta["force_guard_too_late"] = guard_video(sel)
    json.dump(meta, open(os.path.join(RAW, "meta.json"), "w"), indent=1, default=float)


def guard_video(sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    g = sel["guard"]; ep = load("guardF"); thr = g["guard"]
    sig, S, dt, sub = ep["sig"], ep["S"], ep["dt"], ep["sub"]
    t = sig["t"]; load_ = sig["inst"]; force = S[:, 0]
    k_load = int(np.nonzero(load_ >= THR)[0][0])
    k_trip_step = int(np.nonzero(sig["step_force"] > thr)[0][0])
    t_load = float(t[k_load]); t_trip = (k_trip_step + 1) * sub * dt
    k_trip = min(len(S), (k_trip_step + 1) * sub) - 1
    load_at_trip = float(sig["peak"][k_trip])
    T = ep["ftimes"][-1]
    fT, fS, fH = font(22, True), font(15), font(15)
    pad, top, bot = 8, 88, 40
    Wv = 2 * PW + 3 * pad; Hv = top + PH + bot
    Wv += Wv % 2; Hv += Hv % 2
    dpi = 100
    fig, ax1 = plt.subplots(figsize=(PW / dpi, PH / dpi), dpi=dpi)
    fig.patch.set_facecolor(tuple(c / 255 for c in BG))
    ax2 = ax1.twinx()
    ymax1 = max(1.0, float(load_.max()) * 1.45); ymax2 = max(thr * 1.5, float(force.max()) * 1.4)
    ax1.set_xlim(0, T); ax1.set_ylim(0, ymax1); ax2.set_ylim(0, ymax2)
    ax1.set_xlabel("simulated time since lift start (s)")
    ax1.set_ylabel("joint-limit load (N\u00b7m)", color="#c42d23"); ax2.set_ylabel("wrist force (N)", color="#235faa")
    ax1.axhline(THR, color="#c42d23", ls=":", lw=1.2)
    ax1.text(0.99 * T, THR + 0.015 * ymax1, "harm threshold 0.5 N\u00b7m", color="#c42d23", fontsize=8, ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax2.axhline(thr, color="#235faa", ls="--", lw=1.2)
    ax2.text(0.99 * T, thr + 0.012 * ymax2, f"force guard {thr:.1f} N\n(0.6\u00d7 p95 of harmless force)", color="#235faa", fontsize=8, ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    l1, = ax1.plot([], [], color="#c42d23", lw=2, label="joint-limit load (robot-caused)")
    l2, = ax2.plot([], [], color="#235faa", lw=1.6, label="wrist force")
    for a in (ax1, ax2):
        a.spines["top"].set_visible(False)
    ax2.legend(handles=[l1, l2], loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=8, frameon=False)
    fig.tight_layout()
    marks = []
    ts = timeline([ep])
    out = []
    for tv in ts:
        fi, si = at(ep, tv)
        n = si + 1
        l1.set_data(t[:n], load_[:n]); l2.set_data(t[:n], force[:n])
        if n > k_load and "load" not in [m[0] for m in marks]:
            marks.append(("load", ax1.axvline(t_load, color="#c42d23", lw=1)))
            ax2.annotate(f"load \u2265 0.5 N\u00b7m\nat {t_load:.2f} s", (t_load, 0.93), xycoords=("data", "axes fraction"),
                         color="#c42d23", fontsize=8, xytext=(-4, 0), textcoords="offset points", ha="right", va="center")
        if n > k_trip and "trip" not in [m[0] for m in marks]:
            marks.append(("trip", ax1.axvline(t_trip, color="#235faa", lw=1)))
            ax2.annotate(f"guard trips at {t_trip:.2f} s\n(peak load already {load_at_trip:.2f} N\u00b7m)", (t_trip, 0.93),
                         xycoords=("data", "axes fraction"), color="#235faa", fontsize=8, xytext=(4, 0),
                         textcoords="offset points", ha="left", va="center")
        fig.canvas.draw()
        plot = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()).resize((PW, PH))
        im = Image.new("RGB", (Wv, Hv), BG); dr = ImageDraw.Draw(im)
        fs_, fe_ = g["free"]
        rng = ", ".join(([f"shoulder {fs_:.1f}\u00b0"] if fs_ < 90 else []) + ([f"elbow {fe_:.1f}\u00b0"] if fe_ < 90 else []))
        dr.text((pad, 5), f"MuJoCo + MyoSuite myoArm simulation  |  seed {g['seed']}, restricted {rng}  |  {SLOW:g}\u00d7 speed  |  typical case",
                fill=MUTED, font=fH)
        dr.text((pad + 2, 28), "Fixed lift, unguarded", fill=INK, font=fT)
        dr.text((pad + 2, 58), fmt_strat(F), fill=MUTED, font=fS)
        dr.text((2 * pad + PW + 2, 28), "Would a wrist-force guard stop it in time?", fill=INK, font=fT)
        dr.text((2 * pad + PW + 2, 58), f"load first \u2265 0.5 N\u00b7m at {t_load:.2f} s; guard would trip at {t_trip:.2f} s", fill=MUTED, font=fS)
        im.paste(Image.fromarray(ep["frames"][fi]), (pad, top))
        tt = 0.0 if tv is None else min(tv, T)
        dr.text((pad + 8, top + 6), f"t = {tt:.2f} s (sim)", fill=(255, 255, 255), font=fS, stroke_width=2, stroke_fill=(0, 0, 0))
        v = 0.0 if si < 0 else float(load_[si])
        dr.text((pad + 8, top + PH - 30), f"joint-limit load {v:.2f} N\u00b7m   wrist force {0.0 if si < 0 else force[si]:.1f} N",
                fill=(255, 120, 110) if v >= THR else (255, 255, 255), font=font(17, True), stroke_width=2, stroke_fill=(0, 0, 0))
        im.paste(plot, (2 * pad + PW, top))
        dr.text((pad, Hv - 24), "Curves are the simulated signals of this episode. Guard check as in round 21: stop when the per-step peak wrist force exceeds the threshold.",
                fill=MUTED, font=font(13))
        out.append(im)
    plt.close(fig)
    encode(out, "force_guard_too_late")
    return dict(seed=g["seed"], free=g["free"], guard_N=thr, t_load=t_load, t_trip=t_trip, load_at_trip=load_at_trip,
                harm=ep["r"]["harm"], r=ep["r"])


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 1
    if a[0] == "capture":
        E = episodes()
        for n in a[1:]:
            save_episode(n, E[n])
    elif a[0] == "probe":
        for s in a[1:]:
            probe(int(s))
    elif a[0] == "compose":
        compose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
