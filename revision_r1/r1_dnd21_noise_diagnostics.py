#!/usr/bin/env python3
"""Poisson diagnostics and null calibration on the two DND21 noise-only DAVIS346 recordings.

Unlike background-labelled EBSSA events, these recordings contain sensor noise only, so the
fraction of events with P_i <= a is a direct check of the null calibration of the tail score.
Segments: the first SEG_S seconds and SEG_S seconds starting at the middle of each recording.
Writes results/r1_dnd21_noise_diagnostics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import dnd21_evaluation as dnd  # noqa: E402
import event_driven as ed  # noqa: E402
import sp_evaluation as sp  # noqa: E402
from r1_ebssa_analyses import poisson_diagnostics, ALPHAS  # noqa: E402

OUT = HERE.parent / 'results' / 'r1_dnd21_noise_diagnostics.json'
SEG_S = 2.0


def main():
    zc = sp.ZERO_KNOWLEDGE_CONFIG
    out = {'segment_s': SEG_S, 'detector': zc, 'recordings': {}}
    for key in ('noise_dark', 'noise_light'):
        ev = dnd.load(key)
        dur = float(ev['t'].max())
        segs = {}
        for name, t0 in (('start', 0.0), ('middle', dur / 2)):
            s = dnd.slice_time(ev, t0, t0 + SEG_S * 1e6)
            shape = dnd.SENSOR
            lab = np.zeros(len(s), bool)
            rate, w = ed.adaptive_window_us(s, shape, zc['radius'], zc['expected_count'], zc['interval_quantile'])
            p = np.asarray(ed.sliding_window_lr(s, shape, radius=zc['radius'], window_us=w, rate=rate), float)
            warm = s['t'] >= s['t'].min() + w
            segs[name] = {'n_events': int(len(s)), 'window_ms': w / 1e3,
                          'mean_rate_hz_per_px': len(s) / SEG_S / (shape[0] * shape[1]),
                          'null_calibration': {f'{a:g}': float((p[warm] <= a).mean()) for a in ALPHAS},
                          'frac_saturated': float((p >= 1).mean()),
                          'poisson': poisson_diagnostics(s, lab, shape)}
            print(key, name, segs[name]['n_events'], segs[name]['null_calibration'],
                  segs[name]['poisson']['fano_median'], segs[name]['poisson']['interval_cv_median'], flush=True)
        out['recordings'][key] = {'file': dnd.FILES[key][1], 'duration_s': dur / 1e6, 'segments': segs}
    OUT.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
