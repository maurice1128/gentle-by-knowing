"""Fig. 1: the task.  Renders one simulated person (free shoulder) before and after a proximal and a
distal support lift, to show that the support site decides which joint moves.  Run with ../.venv_myo.

  python make_figure_scene.py            render the frames (myo venv) -> out/figs/fig1_frames.npz
  python make_figure_scene.py compose    two-panel Fig. 1 (88 mm, original 2026-09-29 layout)
  python make_figure_scene.py compose3   three-panel Fig. 1 (180 mm, 2026-10-01): A, B frames + C key insight
                                         from stored traces out/videos/raw/guardF.* (no simulation);
                                         also writes out/figs/fig1_panelC_numbers.txt
Both compose options run with ../.venv_mm and write out/figs/fig1_task.{pdf,png}."""
import os
import sys
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")

import numpy as np
import mujoco
import shoulder_task as st

W, H = 520, 420
SEED = 104


def lift_and_capture(offset, angle, speed=0.10):
    """Run the lift, capturing the frame at contact and at the end of the hold."""
    import diag_lift_oracle as dlo
    from care_env6 import AIM_DISTAL
    e = st.env()
    frames = {}
    r = st.run(SEED, 90.0, 90.0, offset, angle, speed)
    # re-run to capture: set up identically, then step manually
    e.m.jnt_range[:] = e._jr0
    e.reset(seed=SEED)
    dvec = dlo.lift_dir(e, angle)
    pos, rot = e.approach_start(dvec, e.axis, clearance=st.CLEAR, aim_shift=AIM_DISTAL + offset)
    e._write_probe(pos, rot)
    e.cmd = np.concatenate([pos - e.base, rot])
    for _ in range(int(0.2 / e.dt)):
        act = e.reflex.step(e.d); e.d.ctrl[e.mids] = act; mujoco.mj_step(e.m, e.d)
    frames["before"] = snap(e)
    z0 = float(e.d.geom_xpos[e._fg][2]); cmd = np.array(pos); n_hold = 0
    for _ in range(int(st.T_MAX * 50)):
        if float(e.d.geom_xpos[e._fg][2]) - z0 < st.RISE_M:
            cmd = cmd + dvec * speed / 50
        e.step(cmd, rot)
        if float(e.d.geom_xpos[e._fg][2]) - z0 >= 0.9 * st.RISE_M:
            n_hold += 1
            if n_hold >= 10:
                break
    frames["after"] = snap(e)
    return frames, r


_R = {}


def snap(e):
    if "r" not in _R:
        _R["r"] = mujoco.Renderer(e.m, H, W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.array(e.d.geom_xpos[e._fg])
        cam.azimuth, cam.elevation, cam.distance = 180.0, -15.0, 0.75
        _R["cam"] = cam
    _R["r"].update_scene(e.d, _R["cam"])
    return _R["r"].render().copy()


def compose3(frames_npz):
    """Three-panel Fig. 1 (2026-10-01): (A, B) the two frames as in `compose`, (C) the key insight from
    stored data only: seed 2222, fixed rule F, unguarded (out/videos/raw/guardF.{npz,json}, recorded by
    make_web_videos_gentle.py capture).  The load is recomputed with make_web_videos_gentle.live_signals
    (the same 50 ms moving average / baseline as the harm metric; asserted equal to the stored harm).
    Full Frontiers text width (180 mm), all text >= 8 pt at printed size.  No simulation is run."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_web_videos_gentle as vid

    z = np.load(frames_npz)
    # ---- panel C data (stored traces; no simulation)
    sel = json.load(open(os.path.join(vid.RAW, "selection.json")))["guard"]
    ep = vid.load("guardF")
    r, sig, S = ep["r"], ep["sig"], ep["S"]
    assert r["seed"] == 2222 and sel["seed"] == 2222
    assert abs(sig["harm_recomputed"] - r["harm"]) < 1e-9 and abs(r["harm"] - sel["stored_F"]["harm"]) < 1e-9
    thr_g = float(sel["guard"])                      # 0.6 x p95 of harmless wrist force (round 21)
    thr_h = vid.THR                                  # 0.5 N m harm threshold
    t, load, force = sig["t"], sig["inst"], S[:, 0]  # load: robot-caused, 50 ms avg, shoulder + elbow
    k_load = int(np.nonzero(load >= thr_h)[0][0])
    k_force = int(np.nonzero(force > thr_g)[0][0])
    k_step = int(np.nonzero(sig["step_force"] > thr_g)[0][0])
    t_load, t_force = float(t[k_load]), float(t[k_force])
    t_trip = (k_step + 1) * ep["sub"] * ep["dt"]     # end of the first 20 ms control step over the guard
    k_trip = min(len(S), (k_step + 1) * ep["sub"]) - 1
    k_pk = int(np.argmax(load)); t_pk, pk = float(t[k_pk]), float(load[k_pk])
    assert abs(pk - r["harm"]) < 1e-9
    if not (t_load < t_force <= t_trip):
        raise SystemExit(f"load does NOT cross {thr_h} N m before the force crosses the guard "
                         f"(t_load {t_load}, t_force {t_force}, t_trip {t_trip}); title claim not supported")
    nums = [
        "Fig. 1C numbers (seed 2222, fixed rule F = (offset 0.03 m, angle -40 deg, 0.10 m/s), unguarded;",
        f"  restricted elbow, free range {r['free_el']:.1f} deg (exact {r['free_el']!r}); shoulder unrestricted)",
        "  source: out/videos/raw/guardF.npz (S: per-2-ms substep signals) + guardF.json (run() dict) + selection.json",
        f"  sampling dt = {ep['dt']} s, control step = {ep['sub']} substeps; episode length {t[-1]:.3f} s, done = {r['done']}",
        f"harm threshold                         : {thr_h} N m (robot-caused joint-limit load, 50 ms moving average)",
        f"tightest force guard (0.6x p95)        : {thr_g:.2f} N (exact {thr_g!r})",
        f"load first >= 0.5 N m                  : t = {t_load:.3f} s (load {load[k_load]:.4f} N m; previous sample {load[k_load - 1]:.4f})",
        f"wrist/contact force first > guard      : t = {t_force:.3f} s (force {force[k_force]:.3f} N; previous sample {force[k_force - 1]:.3f})",
        f"guard would trip (end of control step) : t = {t_trip:.3f} s (as in run(stop_force=...))",
        f"lead of load over force crossing       : {1000 * (t_force - t_load):.0f} ms; over guard trip: {1000 * (t_trip - t_load):.0f} ms",
        f"load at force crossing                 : {load[k_force]:.4f} N m (peak so far {sig['peak'][k_force]:.4f})",
        f"load at guard trip                     : {load[k_trip]:.4f} N m (peak so far {sig['peak'][k_trip]:.4f})",
        f"peak load (= stored harm)              : {pk:.10f} N m at t = {t_pk:.3f} s (elbow {r['lim_el']:.10f}, shoulder {r['lim_sh']:.1f})",
        f"peak wrist/contact force               : {force.max():.3f} N (run() peak_force {r['peak_force']:.3f})",
    ]

    # ---- layout (mm; canvas = printed size)
    MM = 1 / 25.4
    fw = 180.0
    iw = 46.0
    ih = iw * z["prox"].shape[0] / z["prox"].shape[1]
    top, bot = 9.5, 9.0                              # two 8-pt title lines; x tick labels + x label of C
    fh = bot + ih + top
    gap = 2.0
    cx0, cx1 = 2 * iw + gap + 12.0, fw - 9.3          # C axes: room for left / right y labels
    RED, BLUE, INK, MUTED = "#B2182B", "#2166AC", "#222222", "#555555"   # RdBu poles; CVD dE00k >= 21
    sg = lambda v: f"{v:+.0f}".replace("-", "−")
    titles = (f"proximal support:\nshoulder {sg(z['ps'])}°, elbow {sg(z['pe'])}°",
              f"distal support:\nshoulder {sg(z['ds'])}°, elbow {sg(z['de'])}°",
              "harmful lift: joint loaded\nbefore the force guard trips")
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})
    ttl = dict(fontsize=8, pad=2, linespacing=1.15)
    sizes = {}
    for ext, interp, dpi in (("pdf", "none", 300), ("png", "antialiased", 600)):
        fig = plt.figure(figsize=(fw * MM, fh * MM))
        for k, key in enumerate(("prox", "dist")):
            a = fig.add_axes([k * (iw + gap) / fw, bot / fh, iw / fw, ih / fh])
            a.imshow(z[key], interpolation=interp)  # 'none' embeds the raw frame in the PDF
            a.set_title(titles[k], **ttl)
            a.axis("off")
        ax1 = fig.add_axes([cx0 / fw, bot / fh, (cx1 - cx0) / fw, ih / fh])
        ax2 = ax1.twinx()
        ax1.set_title(titles[2], **ttl)
        T = float(t[-1])
        ax1.set_xlim(0, T)
        ym1, ym2 = 4.0, 64.0   # headroom: labels + legend sit above the data
        ax1.set_ylim(0, ym1); ax2.set_ylim(0, ym2)
        ax1.set_xlabel("simulated time (s)", labelpad=1.5)
        ax1.set_ylabel("robot-caused joint-limit\nload (N·m)", color=RED, labelpad=2, linespacing=1.1)
        ax2.set_ylabel("wrist force (N)", color=BLUE, labelpad=2)
        ax1.tick_params(axis="y", colors=RED); ax2.tick_params(axis="y", colors=BLUE)
        ax1.spines["left"].set_color(RED); ax2.spines["right"].set_color(BLUE)
        ax2.spines["left"].set_visible(False); ax1.spines["right"].set_visible(False)
        for a in (ax1, ax2):
            a.spines["top"].set_visible(False)
        ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)   # load drawn on top
        ax2.plot(t, force, color=BLUE, lw=0.6, label="wrist force")
        ax1.plot(t, load, color=RED, lw=1.5, label="joint-limit load")
        hh = ax1.axhline(thr_h, color=RED, ls=(0, (4, 2)), lw=0.8, label="harm threshold 0.5 N·m")
        hg = ax2.axhline(thr_g, color=BLUE, ls=(0, (4, 2)), lw=0.8, label="tightest force guard\n(0.6× 95th percentile)")
        ax1.axvline(t_load, color=RED, lw=0.8, ls=":")
        ax1.axvline(t_force, color=BLUE, lw=0.8, ls=":")
        # crossing labels stacked in the empty pre-contact region, left of the two (adjacent) markers
        ax1.annotate(f"load ≥ 0.5\n{t_load:.3f} s", (t_load, 0.99), xycoords=("data", "axes fraction"), color=RED,
                     fontsize=8, xytext=(-2, 0), textcoords="offset points", ha="right", va="top", linespacing=1.1)
        ax1.annotate(f"force > guard\n{t_force:.3f} s", (t_load, 0.99), xycoords=("data", "axes fraction"), color=BLUE,
                     fontsize=8, xytext=(-2, -21), textcoords="offset points", ha="right", va="top", linespacing=1.1)
        hp, = ax1.plot([t_pk], [pk], "o", ms=3, color=RED, zorder=5, label=f"peak load {pk:.2f} N·m")
        lg = ax1.legend(handles=[hh, hg, hp], loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=True,
                        handlelength=1.4, handletextpad=0.4, borderpad=0.3, labelspacing=0.25, borderaxespad=0.2)
        lg.get_frame().set_linewidth(0); lg.get_frame().set_alpha(0.9)
        ax1.set_yticks([0, 1, 2]); ax1.spines["left"].set_bounds(0, 2.5)          # axes stop below the label band
        ax2.set_yticks([0, 10, 20, 30, 40]); ax2.spines["right"].set_bounds(0, 40)
        fig.text(0.0, 1.0, "A", fontsize=10, fontweight="bold", ha="left", va="top")
        fig.text((iw + gap) / fw, 1.0, "B", fontsize=10, fontweight="bold", ha="left", va="top")
        fig.text((2 * iw + gap + 3.0) / fw, 1.0, "C", fontsize=10, fontweight="bold", ha="left", va="top")
        fig.canvas.draw()
        sizes[ext] = min(tx.get_fontsize() for tx in fig.findobj(matplotlib.text.Text) if tx.get_text().strip())
        fig.savefig(os.path.join(TOP, "out", "figs", f"fig1_task.{ext}"), dpi=dpi)
        plt.close(fig)
    nums.append(f"figure: {fw:.0f} x {fh:.1f} mm; min font size {min(sizes.values()):g} pt; frames {iw:.0f} mm wide "
                f"({z['prox'].shape[1]} px -> {z['prox'].shape[1] / (iw * MM):.0f} dpi)")
    open(os.path.join(TOP, "out", "figs", "fig1_panelC_numbers.txt"), "w", encoding="utf-8").write("\n".join(nums) + "\n")
    print("\n".join(nums))


def main():
    out = os.path.join(TOP, "out", "figs", "fig1_frames.npz")
    if sys.argv[1:] == ["compose3"]:
        return compose3(out)
    if sys.argv[1:] == ["compose"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.load(out)
        # Frontiers: single-column figure, canvas = printed size (88 mm wide), text >= 8 pt.
        # Each 520x420 frame is printed 43 mm wide -> ~307 dpi at native resolution.
        MM = 1 / 25.4
        fw, gap, iw = 88.0, 2.0, 43.0
        ih = iw * z["prox"].shape[0] / z["prox"].shape[1]
        top = 9.0  # two 8-pt title lines
        fh = ih + top + 0.5
        sg = lambda v: f"{v:+.0f}".replace("-", "−")
        titles = (f"proximal support:\nshoulder {sg(z['ps'])}°, elbow {sg(z['pe'])}°",
                  f"distal support:\nshoulder {sg(z['ds'])}°, elbow {sg(z['de'])}°")
        plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
        x0 = (fw - 2 * iw - gap) / 2
        for ext, interp, dpi in (("pdf", "none", 300), ("png", "antialiased", 600)):
            fig = plt.figure(figsize=(fw * MM, fh * MM))
            for k, (key, tt) in enumerate(zip(("prox", "dist"), titles)):
                a = fig.add_axes([(x0 + k * (iw + gap)) / fw, 0.5 / fh, iw / fw, ih / fh])
                a.imshow(z[key], interpolation=interp)  # 'none' embeds the raw frame in the PDF
                a.set_title(tt, fontsize=8, pad=2, linespacing=1.15)
                a.axis("off")
            fig.savefig(os.path.join(TOP, "out", "figs", f"fig1_task.{ext}"), dpi=dpi)
            plt.close(fig)
        print("wrote fig1_task", f"{fw:.0f} x {fh:.1f} mm")
        return
    prox, rp = lift_and_capture(-0.06, -40.0)
    dist, rd = lift_and_capture(0.12, -40.0)
    np.savez_compressed(out, before=dist["before"], prox=prox["after"], dist=dist["after"],
                        ps=rp["d_sh_deg"], pe=rp["d_el_deg"], ds=rd["d_sh_deg"], de=rd["d_el_deg"])
    print("frames saved", rp["d_sh_deg"], rp["d_el_deg"], rd["d_sh_deg"], rd["d_el_deg"])


if __name__ == "__main__":
    main()
