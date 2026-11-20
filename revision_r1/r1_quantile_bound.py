"""Finite-n bound on the realised pass fraction of the empirical alpha-quantile rule.

Rule: thr = numpy.quantile(P, alpha) (linear interpolation), pass if P < thr (strict).
Claim checked: rho <= ceil((n - 1) alpha) / n <= alpha + (1 - alpha) / n, attained for distinct scores,
and FPR = (rho - pi TPR) / (1 - pi) <= min(1, rho / (1 - pi)).
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / 'results' / 'r1_quantile_bound.json'


def main():
    rng = np.random.default_rng(20261121)
    worst, viol, n_cases = 0.0, 0, 0
    for _ in range(200000):
        n = int(rng.integers(1, 40))
        a = float(rng.choice([rng.uniform(0, 1), 0.001, 0.01, 0.05, 0.5]))
        p = rng.integers(0, 4, n).astype(float) if rng.uniform() < 0.3 else rng.uniform(size=n)
        rho = float((p < np.quantile(p, a)).mean())
        bound = math.ceil((n - 1) * a - 1e-12) / n
        n_cases += 1
        viol += rho > bound + 1e-12
        worst = max(worst, (rho - a) * n)
    ex = np.array([0.1, 0.7])
    example = {'P': ex.tolist(), 'alpha': 0.4, 'threshold': float(np.quantile(ex, 0.4)),
               'rho': float((ex < np.quantile(ex, 0.4)).mean()), 'bound': math.ceil(1 * 0.4) / 2}
    res = {'n_cases': n_cases, 'violations_of_ceil_bound': int(viol),
           'max_excess_in_events_(rho-alpha)*n': worst, 'example_rho_exceeds_alpha': example}
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print(res)


if __name__ == '__main__':
    main()
