import numpy as np
from scipy.stats import spearmanr
Z = np.load("run_T2_s1.npz"); R, RHO, WS, NAMES = Z["R"], Z["RHO"], Z["WS"], list(Z["NAMES"])
mixed = (WS > 0) & (WS < 1)
svd = R[:, mixed, 0]; fis = R[:, mixed, 1]
print("mixed segments with svd alpha_hat == +1.000 exactly:", np.mean(np.isclose(svd, 1.0)), " fisher:", np.mean(np.isclose(fis, 1.0)))
print("mixed segments at grid edge |alpha|=4: svd", np.mean(np.abs(svd) >= 3.999), " fisher", np.mean(np.abs(fis) >= 3.999))
# dip magnitudes at w=0.1 for the argmin readings
for k, nm in enumerate(NAMES):
    a = R[:, 1, k]
    print(f"{nm:18s} w=0.1: frac<-1.02 {np.mean(a<-1.02):.2f}, median of dips {np.median(a[a<-1.02]) if np.any(a<-1.02) else np.nan:+.2f}; "
          f"w=0.9: frac>1.02 {np.mean(R[:,-2,k]>1.02):.2f}")
C = np.load("contrast_T2.0_s1.npz"); RC, CN = C["R"], list(C["NAMES"])
print()
for k, nm in enumerate(CN):
    A = RC[:, :, k]
    drops = np.min(np.diff(A, axis=1), axis=1)       # most negative step per draw
    nviol = np.sum(np.diff(A, axis=1) < -1e-9, axis=1)
    print(f"{nm:20s} draws with any drop: {np.mean(drops < -1e-9):.2f}; median worst drop {np.median(drops[drops<-1e-9]):+.2f}; "
          f"frac draws worst drop < -0.2: {np.mean(drops < -0.2):.2f}; mean #violating steps {nviol.mean():.1f}/10")
