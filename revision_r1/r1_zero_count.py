"""Zero-count audit: implemented score sf(max(N-1,0)) vs exact upper tail Pr(X>=N) (=1 at N=0)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import event_driven as ed  # noqa: E402
import sp_evaluation as sp  # noqa: E402
from r1_ebssa_analyses import BASE, pixel_rate  # noqa: E402

OUT = HERE.parent / 'results' / 'r1_zero_count.json'
ALPHAS = (0.001, 0.01, 0.05)


def counts_lam(events, shape, radius, window_us, rate):
    n = ed._window_counts(events['x'].astype(np.int64), events['y'].astype(np.int64),
                          events['t'].astype(np.int64), shape[0], shape[1], radius, float(window_us))
    box = ed._neighbourhood_rate(rate, shape, radius)
    lam = np.maximum(box[events['y'].astype(np.intp), events['x'].astype(np.intp)] * window_us, 1e-3)
    return n.astype(np.float64), lam


def scores(n, lam):
    impl = poisson.sf(np.maximum(n - 1.0, 0.0), lam).astype(np.float32).astype(np.float64)
    exact = np.where(n <= 0, 1.0, poisson.sf(n - 1.0, lam)).astype(np.float32).astype(np.float64)
    return impl, exact


def op(p, y):
    out = {}
    for a in ALPHAS:
        thr = np.quantile(p, a)
        s = p < thr
        out[f'{a:g}'] = {'rho': float(s.mean()), 'fpr': float(s[~y].mean()), 'tpr': float(s[y].mean())}
    return out


def counterexample():
    lam = 0.01
    rng = np.random.default_rng(1)
    n = rng.poisson(lam, 200000).astype(np.float64)
    impl, exact = scores(n, np.full_like(n, lam))
    a = float(-np.expm1(-lam))
    a32 = float(np.float32(poisson.sf(0.0, lam)))  # the same value as stored in float32
    legacy64 = poisson.sf(np.maximum(n - 1.0, 0.0), lam)
    return {'lambda': lam, 'a': a, 'Pr_impl_le_a_float32': float((impl <= a32).mean()),
            'Pr_impl_le_a_float64': float((legacy64 <= poisson.sf(0.0, lam)).mean()),
            'Pr_exact_le_a': float((exact <= a32).mean()), 'max_impl': float(impl.max())}


def main():
    if '--counterexample-only' in sys.argv:
        old = json.loads(OUT.read_text())
        old['counterexample'] = counterexample()
        OUT.write_text(json.dumps(old, indent=1) + '\n')
        print(old['counterexample'])
        return
    res = {'counterexample': counterexample(), 'recordings': []}
    print(res['counterexample'], flush=True)
    configs = {'adaptive_k16': None, 'fixed_100ms_r1': (1, 100e3), 'adaptive_k4': 'k4'}
    for idx, events, labels, shape, rec_id in sp.load_recordings():
        y = labels.astype(bool)
        rate, _, _ = pixel_rate(events, shape, BASE['q'], BASE['min_events'], BASE['fallback'])
        r = {'idx': int(idx)}
        for name, c in configs.items():
            if c is None or c == 'k4':
                k = BASE['k'] if c is None else 4.0
                _, w = ed.adaptive_window_us(events, shape, BASE['radius'], k, BASE['q'], rate=rate)
                rad = BASE['radius']
            else:
                rad, w = c
            n, lam = counts_lam(events, shape, rad, w, rate)
            impl, exact = scores(n, lam)
            r[name] = {'frac_N0': float((n == 0).mean()), 'frac_N0_signal': float((n[y] == 0).mean()),
                       'auc_impl': float(roc_auc_score(y, -impl)), 'auc_exact': float(roc_auc_score(y, -exact)),
                       'op_impl': op(impl, y), 'op_exact': op(exact, y),
                       'null_le_0.01_impl': float((impl[~y] <= 0.01).mean()),
                       'null_le_0.01_exact': float((exact[~y] <= 0.01).mean())}
        res['recordings'].append(r)
        print(idx, {k: (round(v['auc_impl'], 4), round(v['auc_exact'], 4), round(v['frac_N0'], 4))
                    for k, v in r.items() if k != 'idx'}, flush=True)
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
