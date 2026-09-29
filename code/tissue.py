"""Soft-tissue thickness map of the forearm and the bone-proximity load index.

Why (2026-09-14, after the council review and the load diagnostic): in a
rigid-body arm every mechanical quantity on the recipient's side (joint load,
contact shear, contact pressure) is the wrist wrench times geometry, so a
wrist sensor sees it and a reactive barrier can limit it; the stretch reflex
is the only recipient-side quantity not fixed by the wrist wrench, and it is a
readout of approach speed.  Neither leaves a body model anything to do.

What a body model IS needed for in care contact: where the tissue is thin over
bone.  Pressure and shear injury in frail skin concentrate over bony
prominences (ulnar border, styloids, olecranon), where the same wrist force
produces several times the internal tissue stress.  The wrist cannot see
thickness, and slowing down does not change it; only knowing WHERE to touch
this person does.  This module gives each simulated individual a thickness
map t(u, phi) over the forearm capsule (u: axial, -1 proximal .. +1 distal;
phi: circumferential angle from the ulnar border, positive toward dorsal),
and defines the bone-proximity load index of a contact as

    P = (F_n + F_t) * t_ref / t(u, phi)          [N-equivalent, t_ref = 18 mm]

i.e. the wrist load re-weighted by how thin the tissue under it is: 50 N on
mid-forearm muscle (18 mm) scores 50; the same 50 N on a 3 mm styloid scores
300.  It is an INDEX with the ordering of the tissue-stress literature, not a
calibrated stress.  Normal and shear parts are also reported separately.

Numbers: forearm soft-tissue thickness (skin + fat + muscle to bone) from
ultrasound is about 15-30 mm over the volar mid-forearm muscle bellies,
8-15 mm dorsally, 5-10 mm at the distal quarter, 3-6 mm over the subcutaneous
ulnar border and 2-4 mm over the styloids and olecranon; the paper cites the
sources.  Individual variation: log-normal scale on every thickness (sigma
0.3), the distal thinning starting at a different axial position, and each
landmark displaced by a few millimetres and degrees.
"""
import math

T_REF = 18.0           # mm, reference thickness (mid-forearm muscle)
HALF_LEN_MM = 119.0    # radius_coll capsule half-length, mm (u = z / half)

# landmark: (name, u, phi_deg, sigma_u, sigma_phi_deg)
LANDMARKS = (
    ("ulnar_styloid", 0.95, 10.0, 0.08, 20.0),
    ("radial_styloid", 0.97, 180.0, 0.08, 20.0),
    ("olecranon", -0.95, 70.0, 0.10, 25.0),
)


def nominal_params():
    # t_hand: wrist/hand contacts.  Part of the map: a copy with no map must
    # not know the hand is bony (round 7 "uniform" leaked this, 2026-09-16)
    return dict(t_mid=18.0, t_dorsal=0.7, t_ulnar=5.0, t_bone=3.0, t_hand=4.0,
                taper_u=0.3, taper_to=0.25, ulnar_phi=0.0, ulnar_sig=22.0,
                lm=[dict(name=n, u=u, phi=p, su=su, sp=sp) for n, u, p, su, sp in LANDMARKS])


def individual_params(seed, sigma=0.3, sigma_dorsal=0.15, sigma_taper=0.2,
                      sigma_taper_to=0.2, sigma_ulnar_phi=12.0, sigma_lm_u=0.04,
                      sigma_lm_phi=10.0, taper_clip=(-0.4, 0.8), scale_all=1.0):
    """This individual's map: thickness scales, taper onset and landmark
    positions drawn from the seed (independent of the mechanical draw).

    The spreads are arguments so they can be calibrated from the literature
    (2026-09-16); the defaults are the round 7/8 values.  The random draws
    are taken in the same order whatever the spreads, so changing a sigma
    rescales each individual's deviation without reshuffling individuals."""
    import numpy as np
    rng = np.random.default_rng(seed + 777_000)
    p = nominal_params()
    p["t_mid"] *= float(np.exp(rng.normal(0, 1.0) * sigma))
    p["t_ulnar"] *= float(np.exp(rng.normal(0, 1.0) * sigma))
    p["t_bone"] *= float(np.exp(rng.normal(0, 1.0) * sigma))
    p["t_dorsal"] = float(np.clip(p["t_dorsal"] * np.exp(rng.normal(0, 1.0) * sigma_dorsal), 0.4, 1.0))
    p["taper_u"] = float(np.clip(p["taper_u"] + rng.normal(0, 1.0) * sigma_taper, *taper_clip))
    p["taper_to"] = float(np.clip(p["taper_to"] * np.exp(rng.normal(0, 1.0) * sigma_taper_to), 0.25, 0.8))
    p["ulnar_phi"] = float(rng.normal(0, 1.0) * sigma_ulnar_phi)
    for lm in p["lm"]:
        lm["u"] = float(lm["u"] + rng.normal(0, 1.0) * sigma_lm_u)
        lm["phi"] = float(lm["phi"] + rng.normal(0, 1.0) * sigma_lm_phi)
    if scale_all != 1.0:
        # round 10B: a systematically thinner population (frail / cachectic),
        # applied after the individual draws so the individuals are the same
        # people, just thinner
        for k in ("t_mid", "t_ulnar", "t_bone", "t_hand"):
            p[k] *= float(scale_all)
    return p


def spread_from_env():
    """Spreads for individual_params from the environment (literature
    calibration runs, round 8b): WM_SIGMA_THICK, WM_SIGMA_TAPER, WM_SIGMA_LM_U.
    Unset variables keep the round 7/8 defaults."""
    import os
    out = {}
    for key, var in (("sigma", "WM_SIGMA_THICK"), ("sigma_taper", "WM_SIGMA_TAPER"),
                     ("sigma_lm_u", "WM_SIGMA_LM_U"), ("scale_all", "WM_TISSUE_SCALE")):
        if os.environ.get(var):
            out[key] = float(os.environ[var])
    return out


def scaled(params, factor):
    """Fidelity axis: every thickness multiplied by `factor` (the copy thinks
    the tissue is thicker (>1, over-confident) or thinner (<1, conservative))."""
    import copy
    p = copy.deepcopy(params)
    for k in ("t_mid", "t_ulnar", "t_bone", "t_hand"):
        p[k] *= float(factor)
    return p


def shifted(params, mm):
    """Fidelity axis: landmark positions and the ulnar border displaced by
    `mm` millimetres axially (distal) and the same amount as degrees of
    circumference (the copy has the person's bony landmarks in the wrong
    place)."""
    import copy
    p = copy.deepcopy(params)
    du = float(mm) / HALF_LEN_MM
    p["taper_u"] += du
    p["ulnar_phi"] += float(mm)
    for lm in p["lm"]:
        lm["u"] += du
        lm["phi"] += float(mm)
    return p


def uniform(params):
    """No map: the copy believes the tissue is t_mid everywhere."""
    import copy
    p = copy.deepcopy(params)
    p["t_ulnar"] = p["t_bone"] = p["t_hand"] = p["t_mid"]
    p["t_dorsal"] = 1.0
    p["taper_to"] = 1.0
    return p


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def thickness(params, u, phi_deg):
    """Soft-tissue thickness (mm) at axial u in [-1, 1] and circumferential
    angle phi (deg, 0 = ulnar border, +90 = dorsal, -90 = volar, 180 = radial)."""
    p = params
    # axial: constant proximally, tapering distally from taper_u to taper_to * t_mid
    s = min(1.0, max(0.0, (u - p["taper_u"]) / max(1e-6, 1.0 - p["taper_u"])))
    t = p["t_mid"] * (1.0 - (1.0 - p["taper_to"]) * s)
    # circumferential: volar thick, dorsal thinner
    a = 0.5 * (1.0 + p["t_dorsal"]) - 0.5 * (1.0 - p["t_dorsal"]) * math.sin(math.radians(phi_deg))
    t *= a
    # subcutaneous ulnar border: a thin band along the whole length
    dphi = _wrap(phi_deg - p["ulnar_phi"])
    w = math.exp(-0.5 * (dphi / p["ulnar_sig"]) ** 2)
    t = t - (t - min(t, p["t_ulnar"])) * w
    # bony prominences
    for lm in p["lm"]:
        du = (u - lm["u"]) / lm["su"]
        dp = _wrap(phi_deg - lm["phi"]) / lm["sp"]
        w = math.exp(-0.5 * (du * du + dp * dp))
        t = t - (t - min(t, p["t_bone"])) * w
    return max(0.5, float(t))


class ContactMapper:
    """Maps a world-frame contact point on the forearm capsule to (u, phi)
    using the radius_coll geom frame and two reference directions fixed in
    that frame at reset: ulnar (toward the ulna) and dorsal (-flexion normal)."""

    def __init__(self, m, d, rid, forearm_axis_world, flex_normal_world, ulna_body):
        import numpy as np
        self.m, self.rid = m, rid
        R = np.array(d.geom_xmat[rid]).reshape(3, 3)
        c = np.array(d.geom_xpos[rid])
        ax = R[:, 2]
        self.flip = -1.0 if float(np.dot(ax, forearm_axis_world)) < 0 else 1.0
        axw = ax * self.flip
        ul = np.array(d.xpos[ulna_body]) - c
        ul -= np.dot(ul, axw) * axw
        if np.linalg.norm(ul) < 1e-6:
            ul = np.cross(axw, flex_normal_world)
        e_u = ul / np.linalg.norm(ul)
        dz = -np.asarray(flex_normal_world, float)
        dz -= np.dot(dz, axw) * axw
        dz -= np.dot(dz, e_u) * e_u
        if np.linalg.norm(dz) < 1e-6:
            dz = np.cross(axw, e_u)
        e_d = dz / np.linalg.norm(dz)
        # store in the geom's local frame so they ride with the bone
        self.e_u_loc = R.T @ e_u
        self.e_d_loc = R.T @ e_d
        self.half = float(m.geom_size[rid][1])

    def locate(self, d, pos_world):
        import numpy as np
        R = np.array(d.geom_xmat[self.rid]).reshape(3, 3)
        c = np.array(d.geom_xpos[self.rid])
        r = R.T @ (np.asarray(pos_world, float) - c)     # local
        z = r[2] * self.flip
        u = float(np.clip(z / self.half, -1.2, 1.2))
        rho = r.copy(); rho[2] = 0.0
        phi = math.degrees(math.atan2(float(np.dot(rho, self.e_d_loc)),
                                      float(np.dot(rho, self.e_u_loc))))
        return u, phi


T_HAND = 4.0           # mm, wrist/hand: bony, thin everywhere


def body_classes(m, forearm=("radius", "ulna"), upper=("humerus",), arm_bodies=()):
    """body id -> 'forearm' | 'upper' | 'hand' for the recipient's bodies."""
    import mujoco
    out = {}
    for nm in arm_bodies:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if b < 0:
            continue
        out[b] = "forearm" if nm in forearm else ("upper" if nm in upper else "hand")
    return out


def load_index(m, d, pgeom, mapper, params, classes):
    """Bone-proximity load of all probe contacts this instant.

    Every contact with a forearm body (whichever of its collision capsules)
    is located on the forearm map; hand contacts get the thin bony hand
    thickness; upper-arm contacts the mid thickness.  Contacts with anything
    else (bed, torso) are not the recipient and are ignored here.
    Returns dict(pb=normal part, sb=shear part, p=sum, t_min=thinnest tissue
    touched (mm), u, phi of the highest-load contact)."""
    import mujoco
    import numpy as np
    buf = np.zeros(6)
    pb = sb = 0.0
    t_min = 1e9
    worst = (0.0, None, None)
    for i in range(d.ncon):
        c = d.contact[i]
        if pgeom not in (c.geom1, c.geom2):
            continue
        o = c.geom2 if c.geom1 == pgeom else c.geom1
        cls = classes.get(int(m.geom_bodyid[o]))
        if cls is None:
            continue                                     # not the recipient
        mujoco.mj_contactForce(m, d, i, buf)
        fn = abs(float(buf[0]))
        ft = math.hypot(float(buf[1]), float(buf[2]))
        if cls == "forearm":
            u, phi = mapper.locate(d, c.pos)
            t = thickness(params, u, phi)
        elif cls == "hand":
            u, phi, t = 1.0, None, params.get("t_hand", T_HAND)
        else:
            u, phi, t = None, None, params["t_mid"]      # upper arm: not mapped
        w = T_REF / t
        pb += fn * w
        sb += ft * w
        t_min = min(t_min, t)
        if (fn + ft) * w > worst[0]:
            worst = ((fn + ft) * w, u, phi)
    return dict(pb=pb, sb=sb, p=pb + sb, t_min=(t_min if t_min < 1e8 else 0.0),
                u=worst[1], phi=worst[2])


def _selftest():
    p = nominal_params()
    print("nominal thickness (mm):")
    for u in (-0.9, -0.5, 0.0, 0.25, 0.5, 0.8, 0.95):
        row = "  u=%+.2f " % u + " ".join(
            f"{ph:4.0f}:{thickness(p, u, ph):5.1f}" for ph in (-90, -45, 0, 45, 90, 135, 180))
        print(row)
    import numpy as np
    for s in (0, 1, 2, 3):
        q = individual_params(s)
        print(f"seed {s}: t_mid {q['t_mid']:.1f} t_ulnar {q['t_ulnar']:.1f} t_bone {q['t_bone']:.1f} taper_u {q['taper_u']:+.2f} "
              f"ulnar_phi {q['ulnar_phi']:+.0f}  default site (u=0.25, phi=-90): {thickness(q, 0.25, -90):.1f} mm")


if __name__ == "__main__":
    _selftest()
