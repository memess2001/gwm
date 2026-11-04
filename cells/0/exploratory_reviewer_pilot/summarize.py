import json, sys
import numpy as np
fn = sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl"
rows = [json.loads(l) for l in open(fn)]
rows.sort(key=lambda d: (d["obs"], d["var"], d["seed"]))
for d in rows:
    b = d["bars"]
    top = b[0] if b else [np.nan, np.nan]; sec = b[1] if len(b) > 1 else [np.nan, np.nan]
    print(f"{d['obs']:9s} {d['var']:11s} s{d['seed']} st{d['steps']:5d} gate(nmse0)={d.get('nmse0', float('nan')):.4f} "
          f"circ={d.get('rec_circ_err', float('nan')):.4f} effrank={d['eff_rank']:.2f} minstd={d['min_std']:.2f} "
          f"R2(cos,sin,E)=({d['r2_cos']:.3f},{d['r2_sin']:.3f},{d['r2_E']:.3f}) seam={d['seam_ratio']:.1f} diam={d['latent_diam']:.1f} "
          f"top=({top[0]:.2f},{top[1]:.2f}) x{top[1]/top[0]:.2f} 2nd=({sec[0]:.2f},{sec[1]:.2f}) wind={d['wind']}")
