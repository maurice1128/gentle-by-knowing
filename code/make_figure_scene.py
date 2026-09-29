"""Fig. 1: the task.  Renders one simulated person (free shoulder) before and after a proximal and a
distal support lift, to show that the support site decides which joint moves.  Run with ../.venv_myo."""
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


def main():
    out = os.path.join(TOP, "out", "figs", "fig1_frames.npz")
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
