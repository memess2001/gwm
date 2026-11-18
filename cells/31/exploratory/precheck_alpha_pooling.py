"""Cell 31, check before preregistration (ground truth only; nothing is learned).

alpha-integration of K predictive densities (Amari 2007): m_alpha ~ (sum_i w_i p_i^((1-alpha)/2))^(2/(1-alpha)),
alpha = -1 the mixture (linear pool), alpha -> 1 the normalised geometric mean (log pool). The cell predicts that the
log-score-optimal alpha is a property of the WORLD (multimodal -> m, consensus -> e). Here the world is fixed and only
the ensemble changes; the optimal alpha moves across the range, so it is a property of how the members err.
Runs in a few seconds.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

rng = np.random.default_rng(0)
Y = np.linspace(-10, 10, 8001)
DY = Y[1] - Y[0]
ALPHAS = np.round(np.concatenate([np.linspace(-1, 0.95, 40), [1.0], np.linspace(1.05, 5, 40)]), 3)


def log_pool(logp, alpha):
    """Normalised log density of the alpha-integration of the rows of logp (equal weights)."""
    K = len(logp)
    if alpha == 1.0:
        lm = logp.mean(0)
    else:
        r = (1 - alpha) / 2
        lm = (logsumexp(r * logp - np.log(K), axis=0)) / r
    return lm - (logsumexp(lm) + np.log(DY))


def best_alpha(logp, logq_true):
    q = np.exp(logq_true)
    scores = [np.sum(q * log_pool(logp, a)) * DY for a in ALPHAS]
    return ALPHAS[int(np.argmax(scores))], max(scores)


def mix(means, sds, ws=None):
    ws = np.ones(len(means)) / len(means) if ws is None else np.asarray(ws)
    return logsumexp(np.log(ws)[:, None] + norm.logpdf(Y[None, :], np.asarray(means)[:, None], np.asarray(sds)[:, None]), axis=0)


def main():
    worlds = {"bimodal": mix([-2, 2], [0.5, 0.5]), "unimodal": norm.logpdf(Y, 0, 1)}
    ensembles = {
        "bimodal": {
            "specialists (one mode each)": lambda: np.stack([norm.logpdf(Y, -2, 0.5), norm.logpdf(Y, 2, 0.5)]),
            "noisy generalists (both modes, jittered)": lambda: np.stack(
                [mix([-2 + rng.normal(0, 0.3), 2 + rng.normal(0, 0.3)], [0.5 * np.exp(rng.normal(0, 0.3))] * 2,
                     [w := rng.uniform(0.3, 0.7), 1 - w]) for _ in range(5)]),
            "over-dispersed generalists": lambda: np.stack(
                [mix([-2 + rng.normal(0, 0.1), 2 + rng.normal(0, 0.1)], [1.2, 1.2]) for _ in range(5)]),
        },
        "unimodal": {
            "over-confident, biased members": lambda: np.stack([norm.logpdf(Y, m, 0.6) for m in rng.normal(0, 0.8, 5)]),
            "under-confident members": lambda: np.stack([norm.logpdf(Y, m, 1.8) for m in rng.normal(0, 0.2, 5)]),
        },
    }
    for wname, lq in worlds.items():
        print(f"world: {wname}")
        for ename, make in ensembles[wname].items():
            a = [best_alpha(make(), lq)[0] for _ in range(20)]
            print(f"   {ename:42} optimal alpha over 20 draws: median {np.median(a):+.2f}, range [{min(a):+.2f}, {max(a):+.2f}], sd {np.std(a):.2f}")


if __name__ == "__main__":
    main()
