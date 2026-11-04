# The fake term is a coboundary L_F log nu. For a T-invariant volume (symmetric frame window) and T-reversible
# conservative dynamics it is exactly odd under T:(th,om)->(th,-om); physical friction (force odd in velocity) has
# T-even divergence. How much of the fake term survives T-symmetrisation (which is label-free: reverse the window)?
import numpy as np
from common import *
thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")
rms = lambda x: np.sqrt((x ** 2).mean())
for k in (0.3, 0.6):
    loga = np.log(a_analytic(thetas, k, ss=4))
    la = interp_circ(thetas, loga)
    dla = interp_circ(thetas, dlog_lowpass(loga))
    for gamma in (0.0, 0.05):
        F = (OM, -np.sin(TH) - 2 * gamma * OM)
        # 2-frame window nu2 = dt a(th) a(th - om dt): analytic gradient
        g_th = dla(TH) + dla(TH - OM * DT); g_om = -DT * dla(TH - OM * DT)
        fd2 = F[0] * g_th + F[1] * g_om
        fd2_T = fd2[:, ::-1]                                 # value at (th, -om): omegas grid is symmetric
        even2 = 0.5 * (fd2 + fd2_T)
        # label-free partner for 2 frames: swap the frames -> state (th - om dt, -om)
        def fd2_at(th, om):
            Fth, Fom = om, -np.sin(th) - 2 * gamma * om
            return Fth * (dla(th) + dla(th - om * DT)) + Fom * (-DT * dla(th - om * DT))
        swap2 = 0.5 * (fd2 + fd2_at(TH - OM * DT, -OM))
        # symmetric 3-frame window nu3 = dt sqrt(4 a+^2 a-^2 + a0^2 (a+^2 + a-^2)), a_pm = a(th +- om dt)
        def lognu3(th, om):
            ap, a0, am = np.exp(la(th + om * DT)), np.exp(la(th)), np.exp(la(th - om * DT))
            return np.log(DT) + 0.5 * np.log(4 * ap**2 * am**2 + a0**2 * (ap**2 + am**2))
        e = 1e-4
        fd3 = F[0] * (lognu3(TH + e, OM) - lognu3(TH - e, OM)) / (2 * e) + F[1] * (lognu3(TH, OM + e) - lognu3(TH, OM - e)) / (2 * e)
        even3 = 0.5 * (fd3 + fd3[:, ::-1])
        print(f"k={k} gamma={gamma}: raw fake RMS 2-frame {rms(fd2):.3f}, 3-frame {rms(fd3):.3f} | after T-symmetrisation: "
              f"2-frame (exact T partner) {rms(even2):.4f}, 2-frame (frame swap) {rms(swap2):.4f}, 3-frame (window reversal) {rms(even3):.5f}"
              f"   [signal 2*gamma = {2*gamma:.2f}]")
