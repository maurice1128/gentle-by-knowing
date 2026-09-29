"""Stretch-reflex controller for the myoArm care recipient.

Structure follows SpasticMyoElbow (Yu et al., arXiv:2412.04700), whose result is
that a reflex model with feedback on BOTH muscle-fibre velocity and length
reproduces passive-stretch joint resistance better than force-only feedback.

We are honest about the approximation: MuJoCo exposes MTU (muscle-tendon unit)
length and velocity, not fibre length directly, so `actuator_length` /
`actuator_velocity` stand in for the spindle afferent.  This is the usual
Hill-type + spindle-feedback simplification and must be stated in the paper.

    a_i(t) = clip( Gv * [v_i(t-tau)]_+^p + Gl * [l_i(t-tau) - l0_i - dl]_+ , 0, 1)

with a first-order activation lag (tau_act) so the command cannot step.
`Gv`, `Gl`, `tau` are exactly the fidelity-degradation knobs the paper needs
(reflex gain and reflex delay), so they are parameters, not constants.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


class StretchReflex:
    """Velocity+length stretch reflex over a chosen muscle set.

    Parameters
    ----------
    gain_v, gain_l : float   reflex gains (fidelity knobs)
    delay_s        : float   afferent delay, e.g. 0.025 (short) or 0.050 (long)
    vel_exp        : float   velocity exponent (1.0 linear; Hill spindle ~0.6)
    len_thresh     : float   stretch beyond rest length before length term fires
    tau_act        : float   activation first-order lag, seconds
    baseline       : float   tonic activation floor (muscle tone)
    """

    def __init__(self, m, muscle_ids, gain_v=0.6, gain_l=4.0, delay_s=0.025,
                 vel_exp=1.0, len_thresh=0.002, tau_act=0.030, baseline=0.01):
        import numpy as np
        self.m = m
        self.ids = np.asarray(muscle_ids, dtype=int)
        self.gain_v = float(gain_v)
        self.gain_l = float(gain_l)
        self.vel_exp = float(vel_exp)
        self.len_thresh = float(len_thresh)
        self.tau_act = float(tau_act)
        self.baseline = float(baseline)
        self.dt = float(m.opt.timestep)
        self.delay_steps = max(1, int(round(delay_s / self.dt)))
        n = len(self.ids)
        self.buf_l = np.zeros((self.delay_steps, n))
        self.buf_v = np.zeros((self.delay_steps, n))
        self.k = 0
        self.rest_len = None
        self.act = np.full(n, self.baseline, dtype=float)

    def reset(self, d, keep_act=False):
        """(Re)baseline the spindle set-point to the CURRENT posture.

        keep_act=True re-baselines rest_len and the afferent buffers but leaves
        the current activation untouched.  That is what you want after a
        gravity-settling phase: the limb has already recruited tone to hold
        itself up, and zeroing that tone back to `baseline` would make the arm
        drop and inject a spurious stretch transient into the trial you are
        about to measure.
        """
        import numpy as np
        self.rest_len = np.array(d.actuator_length[self.ids], dtype=float)
        self.buf_l[:] = self.rest_len
        self.buf_v[:] = 0.0
        self.k = 0
        if not keep_act:
            self.act[:] = self.baseline

    def step(self, d):
        """Advance one control step; returns activation vector for self.ids."""
        import numpy as np
        if self.rest_len is None:
            self.reset(d)
        cur_l = np.array(d.actuator_length[self.ids], dtype=float)
        cur_v = np.array(d.actuator_velocity[self.ids], dtype=float)

        # read delayed afferent, then overwrite slot with current sample
        dl = self.buf_l[self.k].copy()
        dv = self.buf_v[self.k].copy()
        self.buf_l[self.k] = cur_l
        self.buf_v[self.k] = cur_v
        self.k = (self.k + 1) % self.delay_steps

        stretch_v = np.clip(dv, 0.0, None) ** self.vel_exp
        stretch_l = np.clip(dl - self.rest_len - self.len_thresh, 0.0, None)
        target = self.baseline + self.gain_v * stretch_v + self.gain_l * stretch_l
        target = np.clip(target, 0.0, 1.0)

        alpha = self.dt / max(self.tau_act, self.dt)
        self.act += alpha * (target - self.act)
        np.clip(self.act, 0.0, 1.0, out=self.act)
        return self.act.copy()


def cci(act_a, act_b):
    """Co-contraction index for antagonist groups a and b.

    Rudolph/Falconer-Winter form: CCI = 2 * min(A,B) / (A+B) * (A+B) = 2*min,
    but we report the common normalised variant
        CCI = 2 * sum(min(a_i, b_j)) / (sum a + sum b) * mean level
    Here we use the simple, widely used Falconer-Winter index on group means:
        CCI = 2 * min(A, B) / (A + B),  scaled by (A + B) to keep magnitude info
    which reduces to 2 * min(A, B).  We return both the ratio and the magnitude.
    """
    import numpy as np
    A = float(np.mean(act_a))
    B = float(np.mean(act_b))
    s = A + B
    ratio = (2.0 * min(A, B) / s) if s > 1e-12 else 0.0
    return dict(A=A, B=B, ratio=ratio, magnitude=2.0 * min(A, B), sum=s)
