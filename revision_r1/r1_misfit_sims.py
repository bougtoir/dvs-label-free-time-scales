#!/usr/bin/env python3
"""Controlled departures from the homogeneous-Poisson background.

Each scenario superposes a transient target (a dot crossing the array) on a simulated
background; one scenario adds a persistent source that dwells on the same pixels for the
whole recording.  The deployment detector (r = 3, k = 16, q = 0.99, rate estimated from the
stream) and an oracle that is given the true background rate are scored.
Reported: AUC of transient-target events against background, Pr(P_i <= a) on background
events (null calibration), and rho / FPR / TPR of the strict alpha-quantile rule.
All generating parameters and seeds are written to results/r1_misfit_sims.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import event_driven as ed  # noqa: E402
import sp_evaluation as sp  # noqa: E402

OUT = HERE.parent / 'results' / 'r1_misfit_sims.json'
SEED = 20261120
SHAPE = (64, 64)
DURATION_S = 20.0
BG_HZ = 1.0
ALPHAS = (1e-3, 1e-2, 5e-2)
N_REP = 5
TARGET = {'speed_px_s': 16.0, 'size_px': 2, 'rate_hz_per_px': 100.0, 'n_crossings': 4}
DWELL = {'size_px': 3, 'rate_hz_per_px': 20.0, 'centre': (40, 20)}
SCENARIOS = {
    'poisson': {'kind': 'poisson'},
    'heterogeneous_hot_pixels': {'kind': 'lognormal', 'sigma': 1.0},
    'refractory_renewal': {'kind': 'refractory', 'dead_time_s': 0.3},
    'burst_overdispersed': {'kind': 'burst', 'extra_mean': 1.0, 'spread_s': 0.002},
    'nonstationary_drift': {'kind': 'drift', 'amplitude': 0.8, 'period_s': 10.0},
    'persistent_source': {'kind': 'poisson', 'dwell': True},
}


def bg_times(rng, kind, spec, rate_hz):
    T = DURATION_S
    if kind in ('poisson', 'lognormal'):
        return np.sort(rng.uniform(0, T, rng.poisson(rate_hz * T)))
    if kind == 'refractory':
        d = spec['dead_time_s']
        mean_exp = 1.0 / rate_hz - d
        n = int(rate_hz * T * 2 + 20)
        t = np.cumsum(d + rng.exponential(mean_exp, n)) - rng.uniform(0, 1.0 / rate_hz)
        return t[(t >= 0) & (t < T)]
    if kind == 'burst':
        m = spec['extra_mean']
        parents = np.sort(rng.uniform(0, T, rng.poisson(rate_hz / (1 + m) * T)))
        extra = rng.geometric(1.0 / (1 + m), len(parents)) - 1
        kids = np.repeat(parents, extra) + rng.uniform(0, spec['spread_s'], int(extra.sum()))
        t = np.concatenate([parents, kids])
        return np.sort(t[t < T])
    if kind == 'drift':
        lam_max = rate_hz * (1 + spec['amplitude'])
        t = np.sort(rng.uniform(0, T, rng.poisson(lam_max * T)))
        keep = rng.uniform(0, 1, len(t)) < (1 + spec['amplitude'] * np.sin(2 * np.pi * t / spec['period_s'])) / (1 + spec['amplitude'])
        return t[keep]
    raise ValueError(kind)


def simulate(rng, spec):
    H, W = SHAPE
    rates = np.full(H * W, BG_HZ)
    if spec['kind'] == 'lognormal':
        s = spec['sigma']
        rates = BG_HZ * rng.lognormal(-0.5 * s * s, s, H * W)
    xs, ys, ts, lab = [], [], [], []
    for p in range(H * W):
        t = bg_times(rng, spec['kind'], spec, rates[p])
        xs.append(np.full(len(t), p % W)); ys.append(np.full(len(t), p // W)); ts.append(t); lab.append(np.zeros(len(t), np.int8))
    # transient target: horizontal crossings at random rows and start times
    for c in range(TARGET['n_crossings']):
        row = rng.integers(8, H - 8)
        t_start = (c + rng.uniform(0.1, 0.5)) * DURATION_S / TARGET['n_crossings']
        dwell = 1.0 / TARGET['speed_px_s']
        for x in range(W):
            for dy in range(TARGET['size_px']):
                t0 = t_start + x * dwell
                n = rng.poisson(TARGET['rate_hz_per_px'] * dwell)
                t = t0 + rng.uniform(0, dwell, n)
                t = t[t < DURATION_S]
                xs.append(np.full(len(t), x)); ys.append(np.full(len(t), row + dy)); ts.append(t); lab.append(np.ones(len(t), np.int8))
    if spec.get('dwell'):
        cx, cy = DWELL['centre']
        for dx in range(DWELL['size_px']):
            for dy in range(DWELL['size_px']):
                t = np.sort(rng.uniform(0, DURATION_S, rng.poisson(DWELL['rate_hz_per_px'] * DURATION_S)))
                xs.append(np.full(len(t), cx + dx)); ys.append(np.full(len(t), cy + dy)); ts.append(t); lab.append(np.full(len(t), 2, np.int8))
    t = np.concatenate(ts)
    order = np.argsort(t, kind='stable')
    ev = np.zeros(len(t), dtype=[('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')])
    ev['x'] = np.concatenate(xs)[order]; ev['y'] = np.concatenate(ys)[order]
    ev['t'] = np.round(t[order] * 1e6).astype(np.int64); ev['p'] = 1
    return ev, np.concatenate(lab)[order], rates


def evaluate(p, lab):
    bg, tr, dw = lab == 0, lab == 1, lab == 2
    out = {'auc_transient_vs_bg': float(roc_auc_score(tr[bg | tr], 1 - p[bg | tr])),
           'null_calibration_bg': {f'{a:g}': float((p[bg] <= a).mean()) for a in ALPHAS},
           'signal_prevalence': float((lab > 0).mean()), 'quantile_rule': {}}
    if dw.any():
        out['auc_dwell_vs_bg'] = float(roc_auc_score(dw[bg | dw], 1 - p[bg | dw]))
    for a in ALPHAS:
        f = p < np.quantile(p, a)
        q = {'rho': float(f.mean()), 'fpr': float(f[bg].mean()), 'tpr_transient': float(f[tr].mean())}
        if dw.any():
            q['tpr_dwell'] = float(f[dw].mean())
        out['quantile_rule'][f'{a:g}'] = q
    return out


def main():
    zc = sp.ZERO_KNOWLEDGE_CONFIG
    rng = np.random.default_rng(SEED)
    res = {'seed': SEED, 'shape': SHAPE, 'duration_s': DURATION_S, 'bg_rate_hz': BG_HZ, 'n_rep': N_REP,
           'target': TARGET, 'dwell': DWELL, 'scenarios': SCENARIOS, 'detector': zc, 'results': {}}
    for name, spec in SCENARIOS.items():
        reps = []
        for rep in range(N_REP):
            ev, lab, rates = simulate(rng, spec)
            rate, w = ed.adaptive_window_us(ev, SHAPE, zc['radius'], zc['expected_count'], zc['interval_quantile'])
            p_est = np.asarray(ed.sliding_window_lr(ev, SHAPE, radius=zc['radius'], window_us=w, rate=rate), float)
            p_orc = np.asarray(ed.sliding_window_lr(ev, SHAPE, radius=zc['radius'], window_us=w, rate=rates * 1e-6), float)
            bg_rate_est = float(np.median(rate) * 1e6)
            reps.append({'n_events': int(len(ev)), 'window_ms': w / 1e3, 'median_estimated_rate_hz': bg_rate_est,
                         'median_true_rate_hz': float(np.median(rates)),
                         'estimated_rate': evaluate(p_est, lab), 'oracle_rate': evaluate(p_orc, lab)})
            if spec.get('dwell'):
                cx, cy = DWELL['centre']
                reps[-1]['dwell_pixel_estimated_rate_hz'] = float(rate[cy * SHAPE[1] + cx] * 1e6)
        res['results'][name] = reps
        e = [r['estimated_rate'] for r in reps]
        print(f"{name:26s} AUC={np.mean([x['auc_transient_vs_bg'] for x in e]):.3f} "
              f"Pr(P<=.01|bg)={np.mean([x['null_calibration_bg']['0.01'] for x in e]):.4f} "
              f"FPR@.01={np.mean([x['quantile_rule']['0.01']['fpr'] for x in e]):.4f} "
              + (f"dwellAUC={np.mean([x['auc_dwell_vs_bg'] for x in e]):.3f}" if spec.get('dwell') else ''), flush=True)
    OUT.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
