#!/usr/bin/env python3
"""Semantic audit of the event-window Poisson tail score.

1. Hand-calculable event sequences check what the implemented count N_prior contains
   (current event, centre pixel, window endpoint, timestamp ties, sensor border).
2. Monte-Carlo calibration of the score under a known-rate homogeneous Poisson null:
   a valid p-value must be super-uniform, P(P <= a) <= a, for every a.
   The implemented exact upper tail (=1 at N_prior = 0) is compared with the legacy score
   sf(max(N_prior - 1, 0)) and with the off-by-one alternative that counts the current event;
   super-uniformity is checked at every attainable value a, not only on a grid.
Writes results/r1_semantic_audit.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import event_driven as ed  # noqa: E402

OUT = HERE.parent / 'results' / 'r1_semantic_audit.json'
SEED = 20261119
ALPHAS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
DT = np.dtype([('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')])


def make(rows):
    ev = np.zeros(len(rows), dtype=DT)
    for i, (x, y, t) in enumerate(rows):
        ev[i] = (x, y, t, 1)
    return ev


def counts(ev, shape, radius, window):
    return ed._window_counts(ev['x'], ev['y'], ev['t'], shape[0], shape[1], radius, float(window))


def score_of(ev, shape, radius, window, rate_hz):
    rate = np.full(shape[0] * shape[1], rate_hz * 1e-6)
    return np.asarray(ed.sliding_window_lr(ev, shape, radius=radius, window_us=window, rate=rate), float)


def hand_cases():
    shape = (5, 5)
    # (x, y, t): event 2 is a timestamp tie with event 1; event 3 sits exactly W after event 1.
    seq = [(2, 2, 0), (2, 3, 5), (2, 2, 5), (3, 3, 15), (2, 2, 16), (0, 0, 16)]
    expected = [0, 1, 2, 2, 1, 0]
    got = counts(make(seq), shape, 1, 10).astype(int).tolist()
    seq2 = [(1, 1, 0), (1, 1, 1), (1, 1, 2)]
    got2 = counts(make(seq2), shape, 0, 10).astype(int).tolist()
    tie_a = counts(make([(2, 2, 7), (2, 3, 7)]), shape, 1, 10).astype(int).tolist()
    tie_b = counts(make([(2, 3, 7), (2, 2, 7)]), shape, 1, 10).astype(int).tolist()
    dup = counts(make([(2, 2, 3), (2, 2, 3), (2, 2, 3)]), shape, 1, 10).astype(int).tolist()
    edge = counts(make([(2, 2, 0), (2, 2, 10), (2, 2, 20), (2, 2, 31)]), shape, 1, 10).astype(int).tolist()
    border = counts(make([(0, 0, 0), (1, 1, 1), (4, 4, 2), (0, 0, 3)]), shape, 1, 10).astype(int).tolist()
    # edge-padded neighbourhood rate: the box sum over (2r+1)^2 pixels with replicated edges
    rate = np.arange(25, dtype=float).reshape(5, 5).ravel() * 1e-6
    box = ed._neighbourhood_rate(rate, shape, 1)
    padded = np.pad(rate.reshape(shape), 1, mode='edge')
    box_ref = np.array([[padded[y:y + 3, x:x + 3].sum() for x in range(5)] for y in range(5)])
    lam = 2.0
    n = np.arange(5)
    exact_ref = [1.0] + [float(1.0 - poisson.cdf(v - 1, lam)) for v in range(1, 5)]
    legacy = poisson.sf(np.maximum(n - 1.0, 0.0), lam)
    # score of an isolated event (N_prior = 0) through the full implementation
    iso = score_of(make([(2, 2, 0), (0, 0, 50)]), shape, 1, 10, 1.0)
    # counterexample to super-uniformity of the legacy score at small Lambda
    lam_small = 0.01
    a_star = 1.0 - np.exp(-lam_small)
    # float32 output: Pr(X >= n) rounds to exactly 1.0 when 1 - P < 6e-8
    sat = np.float32(ed.poisson_upper_tail(np.array([1.0]), np.array([20.0]))[0])
    low = ed.pixel_noise_rate(make([(0, 0, 0), (1, 1, 5), (2, 2, 9)]), shape, min_events=4)
    return {
        'sequence_xyt': seq, 'radius': 1, 'window': 10,
        'expected_N_prior': expected, 'implemented_N_prior': got, 'match': got == expected,
        'centre_pixel_repeat_counts_r0': got2, 'centre_pixel_ok': got2 == [0, 1, 2],
        'tie_order_a': tie_a, 'tie_order_b': tie_b, 'tie_ok': tie_a == [0, 1] and tie_b == [0, 1],
        'duplicates_same_xyt': dup, 'duplicates_ok': dup == [0, 1, 2],
        'window_edge_t_0_10_20_31': edge, 'window_edge_ok': edge == [0, 1, 1, 0],
        'border_counts': border, 'border_ok': border == [0, 1, 0, 2],
        'edge_padding_max_abs_err': float(np.abs(box.reshape(shape) - box_ref).max()),
        'interpretation': (
            'N_prior counts earlier events in the (2r+1)^2 box clipped at the sensor border, centre pixel '
            'included, with t_i - t_j <= W (closed window). The current event is not counted. Events sharing '
            't_i are counted only if they precede e_i in the stable input order; duplicates are separate events; '
            'no later-indexed event is used. At the start of a recording the window is truncated (fewer prior '
            'events, conservative).'),
        'score_formula': 'P_i = Pr(X >= N_prior) = 1 if N_prior = 0 else sf(N_prior - 1; Lambda_i), X ~ Poisson(Lambda_i)',
        'implemented_by_N_lam2': [float(v) for v in ed.poisson_upper_tail(n, np.full(5, lam))],
        'reference_Pr_X_ge_N_lam2': exact_ref,
        'implementation_matches_reference': bool(np.allclose(ed.poisson_upper_tail(n, np.full(5, lam)), exact_ref)),
        'submitted_legacy_by_N_lam2': [float(v) for v in legacy],
        'isolated_event_score': float(iso[0]), 'isolated_event_ok': bool(iso[0] == 1.0),
        'legacy_counterexample': {
            'Lambda': lam_small, 'a': float(a_star),
            'legacy_Pr_score_le_a': 1.0,
            'reason': 'every legacy score is sf(max(N-1,0)) <= sf(0) = 1 - exp(-Lambda) = a, so Pr(score <= a) = 1 > a',
            'exact_Pr_score_le_a': float(poisson.sf(0, lam_small)),
        },
        'float32_saturation_example': {'Lambda': 20.0, 'N_prior': 1, 'float32_score': float(sat)},
        'low_activity_rate_fallback_hz': float(low.max() * 1e6),
    }


def simulate(rng, shape, rate_hz, duration_s, tick_us):
    H, W = shape
    n = rng.poisson(rate_hz * duration_s * H * W)
    t = np.sort(rng.uniform(0, duration_s * 1e6, n))
    t = (np.floor(t / tick_us) * tick_us).astype(np.int64)
    ev = np.zeros(n, dtype=DT)
    ev['t'] = t
    ev['x'] = rng.integers(0, W, n)
    ev['y'] = rng.integers(0, H, n)
    return ev


def superuniformity(p, m):
    """max over attainable a of Pr(P <= a) - a, with the binomial s.e. at that a."""
    v = np.sort(p[m])
    if len(v) == 0:
        return {'max_excess': None}
    a, idx = np.unique(v, return_index=True)
    F = (np.r_[idx[1:], len(v)]) / len(v)
    ex = F - a
    j = int(np.argmax(ex))
    return {'max_excess': float(ex[j]), 'at_a': float(a[j]), 'F_at_a': float(F[j]),
            'se': float(np.sqrt(F[j] * (1 - F[j]) / len(v))), 'n': int(len(v))}


def calibration(rng, shape=(48, 48), rate_hz=5.0, duration_s=40.0, radius=3, k=16.0, tick_us=1):
    ev = simulate(rng, shape, rate_hz, duration_s, tick_us)
    rate = np.full(shape[0] * shape[1], rate_hz * 1e-6)
    w = k / (rate_hz * 1e-6 * (2 * radius + 1) ** 2)
    n_prior = counts(ev, shape, radius, w)
    H, W = shape
    interior = ((ev['x'] >= radius) & (ev['x'] < W - radius) & (ev['y'] >= radius)
                & (ev['y'] < H - radius) & (ev['t'] >= ev['t'].min() + w))
    box = ed._neighbourhood_rate(rate, shape, radius)
    lam = box[ev['y'], ev['x']] * w
    implemented = np.asarray(ed.sliding_window_lr(ev, shape, radius=radius, window_us=w, rate=rate), float)
    variants = {
        'implemented_exact_Pr_X_ge_Nprior': implemented,
        'submitted_legacy_sf_max_Nprior_minus_1': poisson.sf(np.maximum(n_prior - 1.0, 0.0), lam),
        'off_by_one_Pr_X_ge_Nprior_plus_1': poisson.sf(n_prior, lam),
    }
    res = {'shape': list(shape), 'rate_hz': rate_hz, 'duration_s': duration_s, 'radius': radius,
           'k': k, 'window_us': w, 'tick_us': tick_us, 'n_events': int(len(ev)),
           'n_interior': int(interior.sum()), 'frac_N0_interior': float((n_prior[interior] == 0).mean()),
           'implemented_equals_reference': bool(np.allclose(
               implemented, np.where(n_prior <= 0, 1.0, poisson.sf(n_prior - 1.0, lam)), atol=1e-6))}
    for name, p in variants.items():
        res[name] = {sub: {f'{a:g}': float(np.mean(p[mm] <= a)) for a in ALPHAS}
                     for sub, mm in (('all', np.ones(len(p), bool)), ('interior_after_first_window', interior))}
        res[name]['superuniformity_interior'] = superuniformity(p, interior)
        res[name]['superuniformity_all'] = superuniformity(p, np.ones(len(p), bool))
    return res


def main():
    rng = np.random.default_rng(SEED)
    out = {'seed': SEED, 'alphas': ALPHAS, 'hand_cases': hand_cases(), 'calibration': []}
    for kw in ({}, {'k': 4.0}, {'k': 64.0, 'duration_s': 80.0}, {'tick_us': 1000},
               {'tick_us': 10000}, {'k': 0.05}, {'k': 0.5}):
        out['calibration'].append(calibration(rng, **kw))
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['hand_cases'], indent=1))
    for c in out['calibration']:
        print(c['k'], c['tick_us'], {v: c[v]['interior_after_first_window'] for v in
              ('implemented_exact_Pr_X_ge_Nprior', 'submitted_legacy_sf_max_Nprior_minus_1')},
              {v: c[v]['superuniformity_interior']['max_excess'] for v in
               ('implemented_exact_Pr_X_ge_Nprior', 'submitted_legacy_sf_max_Nprior_minus_1',
                'off_by_one_Pr_X_ge_Nprior_plus_1')})


if __name__ == '__main__':
    main()
