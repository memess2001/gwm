import numpy as np
Z = np.load("run_T2_s1.npz"); R, RHO, WS = Z["R"], Z["RHO"], Z["WS"]
C = np.load("contrast_T2.0_s1.npz")["R"]
np.set_printoptions(linewidth=200, precision=2, suppress=True)
for d in [0, 1, 2, 3, 4, 5]:
    print(f"draw {d}: rho   {RHO[d]}")
    print(f"         fisherA {R[d,:,1]}")
    print(f"         osc_t0  {R[d,:,4]}")
    print(f"         Cfisher {C[d,:,0]}")
