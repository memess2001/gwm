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
w = 4.0
print("frames where the instantaneous split is well posed (no growth >=2 cells from support); per-component mass modes removed")
print(" t0   |  Lenia obs (w/4, w, 4w)  |  rigid -c.grad A (w/4, w, 4w)  |  FlowLenia-style -div(A grad U) (w/4, w, 4w)")
for t0 in range(1060, 1301, 10):
    At = fr[t0]
    ci, cj = np.unravel_index(np.argmax(potential(At)), At.shape); roll = (N//2 - ci, N//2 - cj)
    cen = lambda a: np.roll(a, roll, axis=(0, 1))[N//2-32:N//2+32, N//2-32:N//2+32]
    rho = cen(At); rd = cen((fr[t0+1] - fr[t0-1])/(2*DT))
    reach = ndimage.binary_dilation(rho > 0, structure=ndimage.generate_binary_structure(2, 1))
    if np.any((np.abs(rd) > 0) & ~reach):
        continue
    lab, nc = ndimage.label(rho > 0)
    for k in range(1, nc+1):
        msk = lab == k; rd[msk] -= rd[msk].sum() * rho[msk] / rho[msk].sum()
    rr = cen(-(c[0]*g0(At) + c[1]*g1(At)))
    U = growth(potential(At)); rf = cen(-(g0(At*g0(U)) + g1(At*g1(U))))
    qs = [0.25, 1, 4]
    a = [split_2d(rho, rd, q*w) for q in qs]; b = [split_2d(rho, rr, q*w) for q in qs]; f = [split_2d(rho, rf, q*w) for q in qs]
    print(f"{t0:5d} |  {a[0]:.3f} {a[1]:.3f} {a[2]:.3f}       |  {b[0]:.3f} {b[1]:.3f} {b[2]:.3f}              |  {f[0]:.3f} {f[1]:.3f} {f[2]:.3f}", flush=True)
