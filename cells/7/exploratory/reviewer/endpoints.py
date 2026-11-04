import numpy as np
from analyze_profiles import readings, i_e, i_m
np.set_printoptions(linewidth=200, precision=3, suppress=True)
for T in (0.5, 1.0, 2.0, 4.0):
    Z = np.load(f"prof_T{T}_s1.npz")
    out = readings(Z)
    print(f"T={T}: median reading at w=0 / w=1 (should be -1 / +1); fraction within 0.05 of both")
    for nm, A in out.items():
        ok = np.mean((np.abs(A[:, 0] + 1) < 0.05) & (np.abs(A[:, -1] - 1) < 0.05))
        print(f"   {nm:26s} {np.median(A[:,0]):+.2f} / {np.median(A[:,-1]):+.2f}   ok={ok:.2f}")
Z = np.load("prof_T2.0_s1.npz")
print("\ndraw 5, T=2: rho(w), Fisher residual of best e-line J_e, best m-line J_m, and their ratio")
print("rho ", Z["RHO"][5]); print("J_e ", Z["JF"][5, :, i_e]); print("J_m ", Z["JF"][5, :, i_m]); print("Jm/Je", Z["JF"][5, :, i_m] / Z["JF"][5, :, i_e])
