"""Tables of the paper (main Tables 1-3 and Supplementary Tables S1-S11), from results/*.json.

Generated from the document code by revision_r1/make_public_tree.py in the development repository: the
statements that compute the table rows are kept and the text of the paper is removed. Writes tables.json,
one CSV per table (tables/) and numbers.json (every number read by paper_numbers.py and
paper_numbers_r1.py) into this directory. Usage: python3 paper/make_tables.py
"""
import csv
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_numbers import auc, auc_sd, dauc, dauc_sd, fmt, load_numbers, pct, pfmt  # noqa: E402,F401
from paper_numbers_r1 import load_r1  # noqa: E402

N = load_numbers()
R1 = load_r1()
TABLE_ROWS = []


def _record(label, headers, rows):
    TABLE_ROWS.append({'table': label, 'headers': [no_negative_zero(h) for h in headers],
                       'rows': [[no_negative_zero(c) for c in r] for r in rows]})

ORDER = ['edlr', 'edlr_adaptive', 'bilateral', 'ynoise', 'motion', 'temporal', 'dwf', 'nearest', 'knoise', 'bilateral_ed', 'knn_pp', 'waiting_lr', 'flow_matched', 'dwf_ed', 'recursive', 'plr', 'pi_dc_dvs', 'fano', 'edncnn', 'mlpf']

_NEG_ZERO = re.compile('(?<![0-9.])-(0\\.0+)(?![0-9])')

def no_negative_zero(text):
    """'-0.000' from rounding a small negative value is written as '0.000'."""
    return _NEG_ZERO.sub('\\1', text) if isinstance(text, str) else text

def ci(e):
    return f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"

def label(key):
    return N['methods'][key]['label']

def cond_name(c):
    return c.replace('_', ' ').replace('hz', ' Hz')


def build_tables():
    m = N['methods']
    tests = N['tests']
    es_e = N['es_ebssa']
    es_d = N['es_dnd']
    sc = N['selfcal']
    dm = N['dnd_methods']
    dt = N['dnd_tests_tuned']
    df = N['dnd_tests_free']
    zc = N['zk_config']
    feas_rows = []
    for mm in N['feas_methods']:
        feas_rows.append([mm['name'], mm['type'], mm['input'], mm['training_labels'], mm['extra_sensors_required'], mm['public_code'], mm['executed_here']])
    _record('Table S1', ['Method', 'Type', 'Input', 'Training labels', 'Extra sensors', 'Public code', 'Executed here'], feas_rows)
    zks = N['zk']
    pdf = zks['published_defaults']
    es_rows = []
    for k in ORDER:
        e = es_e.get(k)
        if e is None:
            continue
        es_rows.append([label(k), f"{e['mean_diff']:+.3f}", f"{e['median_diff']:+.3f}", ci(e), f"{e['n_a_better']}/{e['n']}", f"{e['rank_biserial']:+.2f}"])
    _record('Table S3', ['Method', 'Mean ΔAUC', 'Median ΔAUC', '95% CI', 'Proposal ahead', 'r'], es_rows)

    def row(k):
        mm = m[k]
        t = tests.get(k)
        e = es_e.get(k)
        return [label(k), 'training' if mm['supervised'] else 'tuning', auc_sd(N, k), fmt(pdf[k]['auc_mean']) if k in pdf else '-', fmt(mm['f1']['mean']), fmt(mm['nrr']['mean']), fmt(mm['spr']['mean']), '-' if t is None else f"{t['auc_diff_mean']:+.3f}", '-' if e is None else ci(e), '-' if t is None else f"{t['p_holm']:.3f}" if t['p_holm'] >= 0.001 else '<0.001']
    order = ORDER
    t1_rows = [row(k) for k in order]
    _record('Table 1', ['Method', 'Labels used', 'AUC (mean ± SD)', 'AUC, no labels', 'ΔAUC', '95% CI', 'p (Holm)'], [[r[i] for i in (0, 1, 2, 3, 7, 8, 9)] for r in t1_rows])
    _record('Table S2', ['Method', 'F1', 'NRR', 'SPR'], [[r[i] for i in (0, 4, 5, 6)] for r in t1_rows])
    st = N['strata']
    strata_rows = []
    for (key, name) in (('sensor_180x240', '180 x 240-pixel sensor'), ('sensor_240x304', '240 x 304-pixel sensor'), ('activity_low', 'Event rate below median'), ('activity_high', 'Event rate above median')):
        s = st[key]
        strata_rows.append([name, str(s['n_recordings'])] + [fmt(s[k]['mean']) for k in ('edlr', 'bilateral', 'ynoise', 'plr', 'mlpf', 'edncnn')])
    _record('Table S8', ['Stratum', 'n', 'Proposed', 'Bilateral', 'YNoise', 'Voxelised LR', 'MLPF-style', 'EDnCNN-style'], strata_rows)
    cs = N['cost']['scaling']
    cost_rows = []
    for k in ('edlr', 'edlr_zk', 'bilateral_ed', 'ynoise', 'recursive', 'plr', 'bilateral', 'temporal'):
        s = cs[k]
        cost_rows.append(['Event-driven Poisson LR, zero-knowledge form' if k == 'edlr_zk' else label(k).replace(' (proposed)', '').replace(' (frame-based ablation)', ''), s['family'], f"{s['cost_exponent_mean']:.2f} ± {s['cost_exponent_std']:.2f}", f"{s['seconds_dense'] * 1000.0:.1f}", f"{s['seconds_sparse'] * 1000.0:.2f}", f"{s['ns_per_event_dense']:.0f}", f"{s['ns_per_event_sparse']:.0f}", f"{s['peak_bytes_dense'] / 1000000.0:.1f}", f"{s['peak_bytes_sparse'] / 1000000.0:.2f}"])
    _record('Table S9', ['Method', 'Family', 'Exponent β', 'Time dense (ms)', 'Time sparse (ms)', 'ns/event dense', 'ns/event sparse', 'Memory dense (MB)', 'Memory sparse (MB)'], cost_rows)
    ops_rows = []
    for k in ('edlr', 'multiscale', 'ynoise', 'bilateral_ed'):
        r_ = [label(k) if k in m else 'Multiscale Fisher combination (proposed, 3 scales)']
        for a in N['alphas']:
            v = sc[k][f'{a:g}']
            r_ += [f"{v['fpr_median']:.4f}", f"{v['tpr_median']:.3f}"]
        ops_rows.append(r_)
    _record('Table S4', ['Method'] + [f'FPR (α = {a:g})' if i % 2 == 0 else f'TPR (α = {a:g})' for a in N['alphas'] for i in (0, 1)], ops_rows)
    tr = zks['cross_sensor_transfer']
    dfl = zks['published_defaults']
    dirs = list(tr)

    def dname(d):
        return d.replace('x', ' x ').replace('->', ' to ')
    zk_rows = []
    for (k, nm) in (('edlr', 'Proposed, fixed window'), ('edlr_adaptive', 'Proposed, rate-adaptive window'), ('ynoise', 'YNoise'), ('bilateral', 'Event bilateral (frame-based)'), ('bilateral_ed', 'Event bilateral (event-driven)'), ('dwf', 'Double-window filter'), ('knoise', 'kNoise'), ('temporal', 'Background-activity filter'), ('plr', 'Voxelised Poisson LR')):
        zk_rows.append([nm, fmt(dfl[k]['auc_mean'])] + [fmt(tr[d][k]['auc_transferred']) + f" ({-tr[d][k]['drop']:+.3f})" for d in dirs])
    sat = zks['saturation']
    k_curve = {}
    for (c, v) in sat['edlr_adaptive'].items():
        cfg = json.loads(c)
        if cfg['radius'] == zc['radius'] and cfg['interval_quantile'] == zc['interval_quantile']:
            k_curve[cfg['expected_count']] = v
    _record('Table S5', ['Method', 'No tuning (all recordings)'] + [f'Transfer {dname(d)}' for d in dirs], zk_rows)
    hy = N['hyb']
    yf = hy['ynoise_full']
    ps = hy['per_sensor']
    scr = hy['screen']
    f_keys = list(scr)
    ps_names = list(ps)
    hyb_rows = [['Proposed, zero-knowledge form (stage 1 alone)', fmt(hy['stage1']['auc']['mean']), fmt(ps[ps_names[0]]['stage1_auc']['mean']), fmt(ps[ps_names[1]]['stage1_auc']['mean']), 'n/a', 'n/a'], ['YNoise, published default', fmt(yf['default']['auc']['mean']), fmt(ps[ps_names[0]]['ynoise_default_auc']['mean']), fmt(ps[ps_names[1]]['ynoise_default_auc']['mean']), 'n/a', '1.00'], ['YNoise, dt = W handed over', fmt(yf['handoff']['auc']['mean']), fmt(ps[ps_names[0]]['ynoise_handoff_auc']['mean']), fmt(ps[ps_names[1]]['ynoise_handoff_auc']['mean']), 'n/a', '1.00'], ['YNoise, label-tuned CV (Table 1)', fmt(hy['ynoise_cv_reference']['auc_mean']), 'n/a', 'n/a', 'n/a', '1.00']]
    for fk in f_keys:
        e = scr[fk]
        hyb_rows.append([f'Two-stage, f = {fk}, stage 2 dt = W', fmt(e['handoff']['auc']['mean']), 'n/a', 'n/a', f"{e['signal_recall']['mean']:.2f}", f"{e['handoff']['stage2_time_over_ynoise_full']['median']:.2f}"])
    _record('Table 2', ['Configuration', 'AUC, all', f'AUC, {dname(ps_names[0])}', f'AUC, {dname(ps_names[1])}', 'Recall of stage 1', 'Stage-2 time'], hyb_rows)
    hyb_keys = sorted((k for k in dm if k.startswith('hybrid_')))

    def dpair(k):
        return es_d.get(f'edlr_vs_{k}' if k in dt else f'zero_knowledge_vs_{k}')

    def drow(k, name):
        t = dt.get(k) or df.get(k)
        e = dpair(k)
        return [name, dauc_sd(N, k), '-' if t is None else f"{t['auc_diff_mean']:+.3f}", '-' if e is None else ci(e['mixture_level']), '-' if t is None else f"{t['p_holm']:.3f}" if t['p_holm'] >= 0.001 else '<0.001', '-' if e is None else pfmt(e['segment_level']['p_wilcoxon']), '-' if e is None else f"{e['source_level']['n_sources_a_better']}/{e['source_level']['n_sources']}"]
    dnd_rows = [drow('zero_knowledge', 'Proposed, rate-adaptive, fixed deployment (no tuning)'), drow('edlr_adaptive_ebssa', 'Proposed, rate-adaptive, EBSSA-selected k'), drow('edlr_adaptive', 'Proposed, rate-adaptive, nested CV on DND21'), drow('edlr', 'Proposed, fixed window, nested CV on DND21'), drow('ynoise_default', 'YNoise, published default'), drow('ynoise_ebssa', 'YNoise, EBSSA-selected'), drow('ynoise_handoff', 'YNoise, dt = W handed over (no labels)')]
    dnd_rows += [drow(k, f"Two-stage cascade, f = {k.split('_')[1]} (no labels)") for k in hyb_keys]
    dnd_rows += [drow('ynoise', 'YNoise, nested CV'), drow('bilateral', 'Event bilateral, nested CV'), drow('pfd', 'PFD-A (approximation), nested CV'), drow('dwf', 'DWF, nested CV'), drow('knoise', 'kNoise, nested CV'), drow('temporal', 'Background-activity filter, nested CV'), drow('plr', 'Voxelised Poisson LR, nested CV'), drow('mlpf', 'MLPF-style (supervised, other scenes)'), drow('edncnn', 'EDnCNN-style (supervised, other scenes)')]
    _record('Table S6', ['Configuration', 'AUC (mean ± SD)', 'ΔAUC', '95% CI', 'p (Holm)', 'p (segment)', 'Sources'], dnd_rows)
    (ad, ae) = (N['aednet_dnd'], N['aednet_ebssa'])
    (ae_recs, ae_recs_e) = (N['aednet_dnd_recs'], N['aednet_ebssa_recs'])
    es5_rows = []
    for (key, e) in es_d.items():
        (ml, sl, so) = (e['mixture_level'], e['segment_level'], e['source_level'])
        es5_rows.append([e['a'].replace('_', ' '), e['b'].replace('_', ' '), f"{ml['mean_diff']:+.3f}", ci(ml), pfmt(ml['p_wilcoxon']), f"{ml['rank_biserial']:+.2f}", f"{sl['mean_diff']:+.3f}", pfmt(sl['p_wilcoxon']), f"{sl['n_a_better']}/{sl['n']}", f"{so['n_sources_a_better']}/{so['n_sources']}"])
    _record('Table S10', ['A', 'B', 'Mix. Δ', 'Mix. 95% CI', 'Mix. p', 'r', 'Seg. Δ', 'Seg. p', 'Seg. A ahead', 'Sources A ahead'], es5_rows)

    def ae_group(rs):
        return [f"{np.mean([r['aednet_auc_subset'] for r in rs]):.3f}", f"{np.mean([r['zero_knowledge_auc_subset'] for r in rs]):.3f}", f"{np.mean([r['ynoise_default_auc_subset'] for r in rs]):.3f}", f"{np.mean([r['zero_knowledge_auc_full'] for r in rs]):.3f}", f"{np.mean([r['ynoise_default_auc_full'] for r in rs]):.3f}", f"{np.mean([r['n_scored'] for r in rs]):.0f}", str(len(rs))]
    es6_rows = [[f'EBSSA, {hw[0]} x {hw[1]}-pixel sensor', *ae_group([r for r in ae_recs_e if tuple(r['sensor_hw']) == hw])] for hw in ((180, 240), (240, 304))] + [['EBSSA, all recordings', *ae_group(ae_recs_e)]] + [['DND21 scene: ' + s, *ae_group([r for r in ae_recs if r['source'] == s])] for s in N['dnd_sources']] + [['DND21 condition: ' + cond_name(c), *ae_group([r for r in ae_recs if r['condition'] == c])] for c in N['dnd_conditions']] + [['DND21, all mixtures', *ae_group(ae_recs)]]
    for (row, s) in ((es6_rows[2], ae), (es6_rows[-1], ad)):
        assert row[1:6] == [fmt(s['aednet_mean_auc']), fmt(s['zero_knowledge_mean_auc_subset']), fmt(s['ynoise_default_mean_auc_subset']), fmt(s['zero_knowledge_mean_auc_full']), fmt(s['ynoise_default_mean_auc_full'])], 'Table S11 totals must match the summary'
    _record('Table S11', ['Stratum', 'AEDNet (subset)', 'Proposal (subset)', 'YNoise default (subset)', 'Proposal (full)', 'YNoise default (full)', 'Events', 'n'], es6_rows)
    (S1, mr, dn) = (R1['s'], R1['mis_rows'], R1['dn']['recordings'])
    va = S1['variants']
    mis_label = {'poisson': 'Poisson', 'heterogeneous_hot_pixels': 'Heterogeneous (hot pixels)', 'refractory_renewal': 'Refractory renewal (0.3 s dead time)', 'burst_overdispersed': 'Burst (over-dispersed)', 'nonstationary_drift': 'Non-stationary rate (±80%)', 'persistent_source': 'Persistent source (dwell)'}
    _record('Table 3', ['Scenario', 'AUC (est.)', 'AUC (true rate)', 'Null (est.)', 'FPR', 'TPR', 'π'], [[mis_label[k], f"{v['auc']:.3f}", f"{v['auc_oracle']:.3f}", f"{v['null01']:.4f}", f"{v['fpr01']:.4f}", f"{v['tpr01']:.3f}", f"{v['pi']:.3f}"] for (k, v) in mr.items()])
    va = S1['variants']
    _record('Table S7', ['Setting', 'AUC', 'AUC (low act.)', 'AUC (high act.)', 'W (ms)', 'Sat.', 'FPR', 'TPR'], [[name, f"{g['all']['auc_mean']:.3f}", f"{g['activity_low']['auc_mean']:.3f}", f"{g['activity_high']['auc_mean']:.3f}", f"{g['all']['window_ms_median']:.0f}", f"{g['all']['frac_saturated_median']:.3f}", f"{g['all']['quantile_rule']['0.01']['fpr_median']:.4f}", f"{g['all']['quantile_rule']['0.01']['tpr_median']:.3f}"] for (name, g) in [('base', S1['base'])] + list(va.items())])

def _key(t):
    m = re.match(r'Table (S?)(\d+)', t['table'])
    return (m.group(1) == 'S', int(m.group(2)))


def _jsonable(o):
    if hasattr(o, 'item'):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def main():
    build_tables()
    tabs = sorted(TABLE_ROWS, key=_key)
    (HERE / 'tables.json').write_text(json.dumps(tabs, indent=1, ensure_ascii=False) + '\n')
    (HERE / 'tables').mkdir(exist_ok=True)
    for t in tabs:
        with open(HERE / 'tables' / (t['table'].replace(' ', '_') + '.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(t['headers'])
            w.writerows(t['rows'])
    (HERE / 'numbers.json').write_text(json.dumps({'N': N, 'R1': R1}, indent=1, sort_keys=True,
                                                  default=_jsonable, ensure_ascii=False) + '\n')
    print('wrote', len(tabs), 'tables and numbers.json')


if __name__ == '__main__':
    main()
