import numpy as np, sys
from scipy import ndimage
sys.argv = ['x']
exec(open('lenia_glider_check.py').read().split('if __name__')[0])
A = np.zeros((N, N)); A[N//2-10:N//2+10, N//2-10:N//2+10] = ORBIUM
fr = [A]
for t in range(1500):
    A = step(A); fr.append(A)
fr = np.array(fr)
th = 2*np.pi*np.arange(N)/N
def com(a):
    m = a.sum(); return np.array([np.angle((a.sum(1)*np.exp(1j*th)).sum()/m), np.angle((a.sum(0)*np.exp(1j*th)).sum()/m)])*N/(2*np.pi)
P = np.unwrap(np.array([com(f) for f in fr])*2*np.pi/N, axis=0)*N/(2*np.pi)
tt = np.arange(750, 1501); c = np.array([np.polyfit(tt*DT, P[750:, k], 1)[0] for k in range(2)])
g0 = lambda f: (np.roll(f,-1,0)-np.roll(f,1,0))/2; g1 = lambda f: (np.roll(f,-1,1)-np.roll(f,1,1))/2
print(f"|c| = {np.linalg.norm(c):.3f} cells/time")
print(" t0  ncomp  small-comp mass  share(d=w/4)  share(d=w)  share(d=4w)  share(d=8w) | same, per-comp mass modes removed (d=w, 8w) | rigid-bilinear (w/4, w, 8w)")
rows = []
for t0 in range(1000, 1201, 20):
    At = fr[t0]
    ci, cj = np.unravel_index(np.argmax(potential(At)), At.shape); roll = (N//2 - ci, N//2 - cj)
    cen = lambda a: np.roll(a, roll, axis=(0, 1))[N//2-32:N//2+32, N//2-32:N//2+32]
    rho = cen(At); rd = cen((fr[t0+1] - fr[t0-1])/(2*DT))
    lab, nc = ndimage.label(rho > 0)
    masses = ndimage.sum(rho, lab, range(1, nc+1)) if nc else []
    small = float(np.sum([m for m in masses if m < 0.05*rho.sum()]))
    # remove per-component mass modes (uniform g on each component; also lone growth in empty cells left as is)
    rd0 = rd.copy()
    for k in range(1, nc+1):
        msk = lab == k
        rd0[msk] -= rd[msk].sum() * rho[msk] / rho[msk].sum()
    # rigid reference with compact (bilinear) shifts
    sp_ = lambda a, s: ndimage.shift(a, s, order=1, mode='wrap')
    rr = cen((sp_(At, c*DT) - sp_(At, -c*DT))/(2*DT))
    w = 4.0
    s = [split_2d(rho, rd, q*w) for q in [0.25, 1, 4, 8]]
    s0 = [split_2d(rho, rd0, q*w) for q in [1, 8]]
    sr = [split_2d(rho, rr, q*w) for q in [0.25, 1, 8]]
    rows.append(s[1])
    print(f"{t0:5d}  {nc:3d}   {small:8.4f}       {s[0]:.3f}        {s[1]:.3f}      {s[2]:.3f}       {s[3]:.3f}   |   {s0[0]:.3f}  {s0[1]:.3f}   |   {sr[0]:.3f} {sr[1]:.3f} {sr[2]:.3f}", flush=True)
rows = np.array(rows)
print(f"share(delta=w) across frames: mean {rows.mean():.3f}, CV {rows.std()/rows.mean():.3f}")
