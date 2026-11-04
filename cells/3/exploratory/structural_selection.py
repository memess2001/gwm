"""Non-interventional mechanism test from densities alone, using the cell's own premise (same growth field U=G(K*A)):
regress observed rhodot on  r_L = Lenia reaction rate (clip(A+dt*U)-A)/dt  and  r_F = -div(A grad U)  (Flow-Lenia-style transport).
Lenia frames: Orbium glider.  Transport frames: 20 Euler steps of A <- A + dt*(-div(A grad U)) from the same Orbium state."""
import numpy as np, sys
sys.argv = ['x']
exec(open('lenia_glider_check.py').read().split('if __name__')[0])
g0 = lambda f: (np.roll(f,-1,0)-np.roll(f,1,0))/2; g1 = lambda f: (np.roll(f,-1,1)-np.roll(f,1,1))/2
def rates(A):
    U = growth(potential(A))
    rL = (np.clip(A + DT*U, 0, 1) - A)/DT
    rF = -(g0(A*g0(U)) + g1(A*g1(U)))
    return rL, rF
def fit(rd, rL, rF):
    X = np.c_[rL.ravel(), rF.ravel()]; coef, *_ = np.linalg.lstsq(X, rd.ravel(), rcond=None)
    res = np.linalg.norm(rd.ravel() - X @ coef)/np.linalg.norm(rd)
    return coef, res
A = np.zeros((N, N)); A[N//2-10:N//2+10, N//2-10:N//2+10] = ORBIUM
fr = [A]
for t in range(1300):
    A = step(A); fr.append(A)
print("Lenia glider frames (forward one-step rate):  coef(reaction, transport), rel. residual")
for t0 in [1000, 1100, 1200, 1299]:
    rd = (fr[t0+1] - fr[t0])/DT if t0 < 1300 else None
    rL, rF = rates(fr[t0]); c, r = fit(rd, rL, rF); print(f"  t={t0}: {np.round(c, 4)}, {r:.2e}")
B = fr[1200].copy(); fl = [B]
dtF = 0.02
for t in range(20):
    U = growth(potential(B)); B = B + dtF*(-(g0(B*g0(U)) + g1(B*g1(U)))); fl.append(B)
print(f"transport frames: mass drift {abs(fl[-1].sum()-fl[0].sum())/fl[0].sum():.1e}, min value {min(f.min() for f in fl):.1e}")
for t0 in [0, 10, 19]:
    rd = (fl[t0+1] - fl[t0])/dtF; rL, rF = rates(fl[t0]); c, r = fit(rd, rL, rF); print(f"  step {t0}: {np.round(c, 4)}, {r:.2e}")
