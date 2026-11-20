#!/usr/bin/env python3
"""Summaries of results/r1_ebssa_analyses.json used by the paper -> results/r1_summary.json."""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / 'results'


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def summ_metrics(ms):
    out = {'auc_mean': mean([m['auc'] for m in ms]), 'auc_sd': float(np.std([m['auc'] for m in ms], ddof=1)),
           'window_ms_median': med([m['window_ms'] for m in ms]),
           'frac_saturated_median': med([m['frac_saturated'] for m in ms]),
           'frac_saturated_max': float(np.max([m['frac_saturated'] for m in ms])), 'quantile_rule': {}}
    for a in ms[0]['quantile_rule']:
        q = [m['quantile_rule'][a] for m in ms]
        out['quantile_rule'][a] = {f'{s}_median': med([x[s] for x in q]) for s in ('rho', 'fpr', 'tpr')}
        out['quantile_rule'][a]['fpr_max'] = float(np.max([x['fpr'] for x in q]))
        out['quantile_rule'][a]['frac_recordings_fpr_le_alpha'] = float(np.mean([x['fpr'] <= float(a) for x in q]))
    return out


def main():
    d = json.load(open(RES / 'r1_ebssa_analyses.json'))
    recs = d['recordings']
    rate = np.array([r['event_rate_hz'] for r in recs])
    groups = {'all': recs, 'activity_low': [r for r in recs if r['event_rate_hz'] <= np.median(rate)],
              'activity_high': [r for r in recs if r['event_rate_hz'] > np.median(rate)]}
    out = {'n_recordings': len(recs), 'median_event_rate_hz': float(np.median(rate)),
           'signal_fraction': {'median': med([r['signal_fraction'] for r in recs]),
                               'min': float(min(r['signal_fraction'] for r in recs)),
                               'max': float(max(r['signal_fraction'] for r in recs))}}
    out['base'] = {g: summ_metrics([r['base'] for r in rs]) for g, rs in groups.items()}
    out['variants'] = {v: {g: summ_metrics([r['variants'][v] for r in rs]) for g, rs in groups.items()}
                       for v in d['variants']}
    # bound: FPR <= alpha / (1 - pi) for the strict quantile rule (rho <= alpha)
    bound = []
    for r in recs:
        pi = r['signal_fraction']
        for a, q in r['base']['quantile_rule'].items():
            bound.append({'alpha': float(a), 'pi': pi, 'fpr': q['fpr'], 'rho': q['rho'],
                          'bound': float(a) / (1 - pi), 'identity_resid': q['rho'] - ((1 - pi) * q['fpr'] + pi * q['tpr'])})
    out['rho_fpr_bound'] = {'all_fpr_le_bound': bool(all(b['fpr'] <= b['bound'] + 1e-12 for b in bound)),
                            'all_rho_le_alpha': bool(all(b['rho'] <= b['alpha'] + 1e-12 for b in bound)),
                            'max_identity_residual': float(max(abs(b['identity_resid']) for b in bound))}
    nc = [r['base']['null_calibration_background'] for r in recs]
    out['null_calibration_background'] = {a: {'median': med([x[a] for x in nc]), 'max': float(max(x[a] for x in nc)),
                                               'frac_recordings_le_a': float(np.mean([x[a] <= float(a) for x in nc]))}
                                          for a in nc[0]}
    fb = [r['base']['fallback'] for r in recs]
    out['fallback'] = {k: {'median': med([x[k] for x in fb]), 'max': (float(max(x[k] for x in fb if x[k] is not None))
                                                                      if any(x[k] is not None for x in fb) else None)}
                       for k in fb[0]}
    po = [r['poisson'] for r in recs]
    out['poisson'] = {
        'n_pixels_tested_median': med([p['n_pixels_tested'] for p in po]),
        'frac_bg_events_on_tested_pixels_median': med([p['frac_bg_events_on_tested_pixels'] for p in po]),
        'duration_s_median': med([p['duration_s'] for p in po]),
        'interval_cv_median': med([p['interval_cv_median'] for p in po]),
        'short_interval': {s: {k: med([p['short_interval'][s][k] for p in po]) for k in ('observed_median', 'expected_poisson_median')}
                           for s in po[0]['short_interval']},
        'fano_median': {b: med([p['fano_median'][b] for p in po]) for b in po[0]['fano_median']},
        'fano_recordings': {b: int(sum(p['fano_median'][b] is not None for p in po)) for b in po[0]['fano_median']},
        'block_rate_cv_median': med([p['block_rate_cv'] for p in po]),
        'block_rate_poisson_cv_median': med([p['block_rate_poisson_cv'] for p in po]),
        'block_rate_max_over_min_median': med([p['block_rate_max_over_min'] for p in po]),
    }
    tm = [r['timing'] for r in recs]
    keys = ['rate_s', 'window_s', 'score_s', 'quantile_s', 'stage1_total_s', 'ynoise_default_s']
    out['timing'] = {k: {'median': med([t[k] for t in tm]), 'mean': mean([t[k] for t in tm])} for k in keys}
    for f in ('0.1', '0.2'):
        out['timing'][f'cascade_{f}'] = {k: med([t[f'cascade_{f}'][k] for t in tm]) for k in ('stage2_s', 'select_s', 'total_s')}
    out['timing']['n_events_median'] = med([t['n_events'] for t in tm])
    out['timing']['ratio_stage1_over_ynoise_median'] = med([t['stage1_total_s'] / t['ynoise_default_s'] for t in tm])
    out['timing']['ratio_cascade0.1_over_ynoise_median'] = med([t['cascade_0.1']['total_s'] / t['ynoise_default_s'] for t in tm])
    (RES / 'r1_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    b = out['base']['all']
    print('base AUC', round(b['auc_mean'], 4), 'W', round(b['window_ms_median']), 'sat', b['frac_saturated_median'], b['frac_saturated_max'])
    for a, q in b['quantile_rule'].items():
        print(' a', a, {k: round(v, 5) for k, v in q.items()})
    print('bound', out['rho_fpr_bound'], 'signal', out['signal_fraction'])
    print('nullcal', out['null_calibration_background'])
    for v, g in out['variants'].items():
        print(f"{v:22s} AUC {g['all']['auc_mean']:.4f} low {g['activity_low']['auc_mean']:.4f} high {g['activity_high']['auc_mean']:.4f} "
              f"W {g['all']['window_ms_median']:.0f} sat {g['all']['frac_saturated_median']:.4f}/{g['all']['frac_saturated_max']:.3f} "
              f"FPR.01 {g['all']['quantile_rule']['0.01']['fpr_median']:.4f} TPR.01 {g['all']['quantile_rule']['0.01']['tpr_median']:.3f}")
    print('lowhigh base', out['base']['activity_low']['auc_mean'], out['base']['activity_high']['auc_mean'])
    print('fallback', json.dumps(out['fallback']))
    print('poisson', json.dumps(out['poisson']))
    print('timing', json.dumps(out['timing']))


if __name__ == '__main__':
    main()
