"""Numbers of the additional analyses in revision_r1/*.py, read from results/r1_*.json."""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / 'results'


def _load(name):
    return json.loads((RES / name).read_text())


def load_r1():
    s = _load('r1_summary.json')
    sem = _load('r1_semantic_audit.json')
    mis = _load('r1_misfit_sims.json')
    dn = _load('r1_dnd21_noise_diagnostics.json')
    cal = sem['calibration']
    alphas = [f'{a:g}' for a in sem['alphas']]
    impl_max_ratio = max(c['implemented_exact_Pr_X_ge_Nprior']['interior_after_first_window'][a] / float(a)
                         for c in cal for a in alphas)
    obo_min_ratio = min(c['off_by_one_Pr_X_ge_Nprior_plus_1']['interior_after_first_window'][a] / float(a)
                        for c in cal for a in alphas if float(a) >= 0.01)
    mis_rows = {}
    for sc, reps in mis['results'].items():
        def mean(f):
            return sum(f(r) for r in reps) / len(reps)
        mis_rows[sc] = {
            'auc': mean(lambda r: r['estimated_rate']['auc_transient_vs_bg']),
            'auc_oracle': mean(lambda r: r['oracle_rate']['auc_transient_vs_bg']),
            'null01': mean(lambda r: r['estimated_rate']['null_calibration_bg']['0.01']),
            'null01_oracle': mean(lambda r: r['oracle_rate']['null_calibration_bg']['0.01']),
            'fpr01': mean(lambda r: r['estimated_rate']['quantile_rule']['0.01']['fpr']),
            'tpr01': mean(lambda r: r['estimated_rate']['quantile_rule']['0.01']['tpr_transient']),
            'pi': mean(lambda r: r['estimated_rate']['signal_prevalence']),
            'window_ms': mean(lambda r: r['window_ms']),
            'rate_ratio': mean(lambda r: r['median_estimated_rate_hz'] / r['median_true_rate_hz']),
        }
        if 'auc_dwell_vs_bg' in reps[0]['estimated_rate']:
            mis_rows[sc]['auc_dwell'] = mean(lambda r: r['estimated_rate']['auc_dwell_vs_bg'])
    return {'s': s, 'sem': sem, 'mis': mis, 'mis_rows': mis_rows, 'dn': dn, 'n_cal_cfg': len(cal),
            'impl_max_ratio': impl_max_ratio, 'obo_min_ratio': obo_min_ratio,
            'score_lam2': sem['hand_cases']['implemented_by_N_lam2'],
            'legacy_lam2': sem['hand_cases']['submitted_legacy_by_N_lam2'],
            'exact_max_excess_us': max(c['implemented_exact_Pr_X_ge_Nprior']['superuniformity_interior']['max_excess']
                                       for c in cal if c['tick_us'] == 1),
            'exact_max_excess_quantised': max(c['implemented_exact_Pr_X_ge_Nprior']['superuniformity_interior']['max_excess']
                                              for c in cal if c['tick_us'] > 1),
            'legacy_max_excess': max(c['submitted_legacy_sf_max_Nprior_minus_1']['superuniformity_interior']['max_excess']
                                     for c in cal),
            'legacy_counterexample': sem['hand_cases']['legacy_counterexample']}
