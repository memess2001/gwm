import numpy as np, zoo
from common import *
thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
d = dlog_lowpass(np.log(a_analytic(thetas, 0.3, ss=4)))
w = zoo.make("pendulum")
for s0, name in (([0.0, 1.2], "libration E=-0.28"), ([0.0, 2.3], "rotation  E=+1.65"), ([2.9, 0.0], "near-separatrix libration")):
    tr = w.rollout(np.array([s0]), 20000)[0]          # 2000 time units
    fd = fake_div_2frame(thetas, d, tr[:, 0], tr[:, 1])
    print(f"{name:<28s}: pointwise fake RMS along orbit {np.sqrt((fd**2).mean()):.3f}; time-average {fd.mean():+.5f}")
