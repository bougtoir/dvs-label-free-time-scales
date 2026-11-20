#!/usr/bin/env python3
"""Focused analyses on the labelled EBSSA recordings (43 recordings, sp.load_recordings).

Settings were fixed before execution:
* base = deployment configuration r = 3, k = 16, q = 0.99, fallback = median rate of pixels
  with >= 4 events (sp.ZERO_KNOWLEDGE_CONFIG);
* one-factor-at-a-time variants: q in {0.95, 0.975, 0.99, 0.995}; k in {2, 4, 8, 16, 32, 64};
  r in {2, 3, 4}; fallback rule in {median_active (base), global_mean, p10_active, p90_active}
  and min_events in {2, 4, 8};
* metrics per recording: AUC, window W, saturated fraction (P_i == 1), pass fraction rho, FPR and
  TPR of the strict alpha-quantile rule for alpha in {1e-3, 1e-2, 5e-2}; for the base also the
  fixed-threshold null calibration Pr(P_i <= a | background), fallback-pixel and hot-pixel
  diagnostics, Poisson diagnostics of background events, and an end-to-end timing breakdown.
No variant is promoted to the main configuration; all variants are reported.
Writes results/r1_ebssa_analyses.json (per-recording values plus summaries).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import event_driven as ed  # noqa: E402
import sp_evaluation as sp  # noqa: E402

OUT = HERE.parent / 'results' / 'r1_ebssa_analyses.json'
ALPHAS = (1e-3, 1e-2, 5e-2)
BASE = {'radius': 3, 'k': 16.0, 'q': 0.99, 'fallback': 'median_active', 'min_events': 4}
VARIANTS = ([('q', v) for v in (0.95, 0.975, 0.995)] + [('k', v) for v in (2.0, 4.0, 8.0, 32.0, 64.0)]
            + [('radius', v) for v in (2, 4)]
            + [('fallback', v) for v in ('global_mean', 'p10_active', 'p90_active')]
            + [('min_events', v) for v in (2, 8)])
FANO_BINS_US = (1e4, 1e5, 1e6)
SHORT_IV_US = (1e3, 1e4)
N_BLOCKS = 10
MIN_BG_EVENTS_PIXEL = 20
N_TIMING = 3
YNOISE_DEFAULT = {'dt_us': 30000.0, 'radius': 2}
CASCADE_F = (0.1, 0.2)


def pixel_rate(events, shape, q, min_events, fallback):
    W = shape[1]
    pix = events['y'].astype(np.int64) * W + events['x'].astype(np.int64)
    order = np.argsort(pix, kind='stable')
    iv = ed._quantile_interval_per_pixel(pix[order], events['t'][order].astype(np.float64),
                                         shape[0] * shape[1], int(min_events), float(q))
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = -np.log(1.0 - q) / iv
    finite = np.isfinite(rate)
    duration = max(float(events['t'].max() - events['t'].min()), 1.0)
    if fallback == 'median_active':
        fb = float(np.median(rate[finite]))
    elif fallback == 'global_mean':
        fb = len(events) / duration / (shape[0] * shape[1])
    elif fallback == 'p10_active':
        fb = float(np.quantile(rate[finite], 0.10))
    elif fallback == 'p90_active':
        fb = float(np.quantile(rate[finite], 0.90))
    else:
        raise ValueError(fallback)
    out = rate.copy()
    out[~finite] = fb
    return np.maximum(out, 1e-12), ~finite, fb


def score(events, shape, cfg):
    rate, fb_mask, fb = pixel_rate(events, shape, cfg['q'], cfg['min_events'], cfg['fallback'])
    _, w = ed.adaptive_window_us(events, shape, cfg['radius'], cfg['k'], cfg['q'], rate=rate)
    p = ed.sliding_window_lr(events, shape, radius=cfg['radius'], window_us=w, rate=rate)
    return np.asarray(p, dtype=np.float64), w, rate, fb_mask, fb


def quantile_rule(p, labels):
    out = {}
    for a in ALPHAS:
        thr = np.quantile(p, a)
        f = p < thr
        out[f'{a:g}'] = {'rho': float(f.mean()), 'fpr': float(f[~labels].mean()),
                         'tpr': float(f[labels].mean()), 'n_tied_at_threshold': int((p == thr).sum())}
    return out


def metrics(p, labels, w):
    return {'auc': float(roc_auc_score(labels, 1.0 - p)), 'window_ms': w / 1e3,
            'frac_saturated': float((p >= 1.0).mean()), 'quantile_rule': quantile_rule(p, labels)}


def poisson_diagnostics(events, labels, shape):
    bg = events[~labels]
    W = shape[1]
    t0, t1 = float(events['t'].min()), float(events['t'].max())
    dur = t1 - t0
    pix = bg['y'].astype(np.int64) * W + bg['x'].astype(np.int64)
    order = np.lexsort((bg['t'], pix))
    pix, t = pix[order], bg['t'][order].astype(np.float64)
    uniq, start, cnt = np.unique(pix, return_index=True, return_counts=True)
    sel = cnt >= MIN_BG_EVENTS_PIXEL
    cv, short_obs, short_exp, fano = [], {s: [] for s in SHORT_IV_US}, {s: [] for s in SHORT_IV_US}, {b: [] for b in FANO_BINS_US}
    for s0, c in zip(start[sel], cnt[sel]):
        tt = t[s0:s0 + c]
        iv = np.diff(tt)
        cv.append(float(iv.std() / iv.mean()) if iv.mean() > 0 else np.nan)
        lam = c / dur
        for s in SHORT_IV_US:
            short_obs[s].append(float((iv < s).mean()))
            short_exp[s].append(float(1.0 - np.exp(-lam * s)))
        for b in FANO_BINS_US:
            nb = int(dur // b)
            if nb >= 5:
                full = tt[tt < t0 + nb * b]  # complete bins only; the trailing partial bin is dropped
                h = np.bincount(((full - t0) // b).astype(np.int64), minlength=nb)[:nb]
                if h.mean() > 0:
                    fano[b].append(float(h.var(ddof=1) / h.mean()))
    blocks = np.bincount(np.minimum(((bg['t'] - t0) / dur * N_BLOCKS).astype(np.int64), N_BLOCKS - 1),
                         minlength=N_BLOCKS)
    return {
        'duration_s': dur / 1e6, 'n_background': int(len(bg)), 'n_pixels_any_bg': int(len(uniq)),
        'n_pixels_tested': int(sel.sum()),
        'frac_bg_events_on_tested_pixels': float(cnt[sel].sum() / max(len(bg), 1)),
        'interval_cv_median': float(np.nanmedian(cv)) if cv else None,
        'short_interval': {f'{s / 1e3:g}ms': {'observed_median': float(np.median(short_obs[s])) if cv else None,
                                              'expected_poisson_median': float(np.median(short_exp[s])) if cv else None}
                           for s in SHORT_IV_US},
        'fano_median': {f'{b / 1e3:g}ms': (float(np.median(fano[b])) if fano[b] else None) for b in FANO_BINS_US},
        'fano_n_pixels': {f'{b / 1e3:g}ms': len(fano[b]) for b in FANO_BINS_US},
        'block_rate_cv': float(blocks.std() / blocks.mean()),
        'block_rate_max_over_min': float(blocks.max() / max(blocks.min(), 1)),
        'block_rate_poisson_cv': float(1.0 / np.sqrt(blocks.mean())),
    }


def best_time(fn):
    best, out = np.inf, None
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def timing(events, shape, labels):
    zc = sp.ZERO_KNOWLEDGE_CONFIG
    r, k, q = zc['radius'], zc['expected_count'], zc['interval_quantile']
    rate, t_rate = best_time(lambda: ed.pixel_noise_rate(events, shape, interval_quantile=q))
    (_, w), t_w = best_time(lambda: ed.adaptive_window_us(events, shape, r, k, q, rate=rate))
    p, t_score = best_time(lambda: np.asarray(ed.sliding_window_lr(events, shape, radius=r, window_us=w, rate=rate)))
    _, t_q = best_time(lambda: p < np.quantile(p, 0.01))
    _, t_y = best_time(lambda: sp.ynoise_filter(events, shape, **YNOISE_DEFAULT))
    out = {'n_events': int(len(events)), 'rate_s': t_rate, 'window_s': t_w, 'score_s': t_score,
           'quantile_s': t_q, 'stage1_total_s': t_rate + t_w + t_score + t_q, 'ynoise_default_s': t_y}
    for f in CASCADE_F:
        cand = p < np.quantile(p, f)
        if cand.sum() < 0.5 * f * len(p):
            cand = p <= np.quantile(p, f)
        sub = events[cand]
        _, t2 = best_time(lambda: sp.ynoise_filter(sub, shape, dt_us=w, radius=r))
        _, ts = best_time(lambda: events[p < np.quantile(p, f)])
        out[f'cascade_{f:g}'] = {'stage2_s': t2, 'select_s': ts,
                                 'total_s': out['stage1_total_s'] + ts + t2}
    return out


def rerun_poisson():
    """Recompute only the Poisson diagnostics in an existing result file."""
    res = json.loads(OUT.read_text())
    by_idx = {r['idx']: r for r in res['recordings']}
    for idx, events, labels, shape, rec_id in sp.load_recordings():
        by_idx[int(idx)]['poisson'] = poisson_diagnostics(events, labels, shape)
        print('poisson', idx, by_idx[int(idx)]['poisson']['fano_median'], flush=True)
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('wrote', OUT)


def rerun_timing():
    """Recompute only the end-to-end timings in an existing result file (run on an otherwise idle machine)."""
    res = json.loads(OUT.read_text())
    by_idx = {r['idx']: r for r in res['recordings']}
    warm = True
    for idx, events, labels, shape, rec_id in sp.load_recordings():
        if warm:
            timing(events[:5000], shape, labels[:5000])
            warm = False
        by_idx[int(idx)]['timing'] = timing(events, shape, labels)
        print('timing', idx, round(by_idx[int(idx)]['timing']['stage1_total_s'], 4), flush=True)
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('wrote', OUT)


def main():
    if '--poisson-only' in sys.argv:
        return rerun_poisson()
    if '--timing-only' in sys.argv:
        return rerun_timing()
    no_timing = '--no-timing' in sys.argv
    base = dict(BASE)
    recs = []
    warm = True
    for idx, events, labels, shape, rec_id in sp.load_recordings():
        t0 = time.time()
        if warm and not no_timing:  # JIT compilation excluded from timings
            timing(events[:5000], shape, labels[:5000])
            warm = False
        p, w, rate, fb_mask, fb = score(events, shape, base)
        dur_s = float(events['t'].max() - events['t'].min()) / 1e6
        r = {'idx': int(idx), 'recording_id': str(rec_id), 'shape': list(map(int, shape)),
             'n_events': int(len(events)), 'event_rate_hz': len(events) / max(dur_s, 1e-6),
             'signal_fraction': float(labels.mean()), 'base': metrics(p, labels, w)}
        pix = events['y'].astype(np.int64) * shape[1] + events['x'].astype(np.int64)
        on_fb = fb_mask[pix]
        hot = rate >= np.quantile(rate[~fb_mask], 0.99) if (~fb_mask).any() else np.zeros_like(fb_mask)
        on_hot = hot[pix] & ~on_fb
        thr = np.quantile(p, 0.01)
        passed = p < thr
        r['base']['null_calibration_background'] = {f'{a:g}': float((p[~labels] <= a).mean()) for a in ALPHAS}
        r['base']['fallback'] = {
            'frac_pixels_fallback': float(fb_mask.mean()),
            'frac_active_pixels_fallback': float(fb_mask[np.unique(pix)].mean()),
            'frac_events_on_fallback': float(on_fb.mean()),
            'fallback_rate_hz': fb * 1e6, 'median_active_rate_hz': float(np.median(rate[~fb_mask])) * 1e6,
            'signal_frac_on_fallback': float(labels[on_fb].mean()) if on_fb.any() else None,
            'pass_rate_on_fallback_a0.01': float(passed[on_fb].mean()) if on_fb.any() else None,
            'frac_events_on_hot': float(on_hot.mean()),
            'signal_frac_on_hot': float(labels[on_hot].mean()) if on_hot.any() else None,
            'pass_rate_on_hot_a0.01': float(passed[on_hot].mean()) if on_hot.any() else None,
        }
        r['variants'] = {}
        for key, val in VARIANTS:
            cfg = dict(base, **{key: val})
            pv, wv, *_ = score(events, shape, cfg)
            r['variants'][f'{key}={val}'] = metrics(pv, labels, wv)
        r['poisson'] = poisson_diagnostics(events, labels, shape)
        r['timing'] = None if no_timing else timing(events, shape, labels)
        recs.append(r)
        print(f"rec {idx} auc={r['base']['auc']:.3f} W={w / 1e3:.0f}ms fpr@.01={r['base']['quantile_rule']['0.01']['fpr']:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
    OUT.write_text(json.dumps({'base': BASE, 'variants': [f'{k}={v}' for k, v in VARIANTS],
                               'alphas': ALPHAS, 'recordings': recs}, indent=1) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
