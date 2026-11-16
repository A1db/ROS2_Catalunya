#!/usr/bin/env python3
"""Summarise ftg_debug.csv: why the car crawls, and where. Usage: python3 analyze_debug.py ftg_debug.csv"""
import sys
import numpy as np, csv
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else 'ftg_debug.csv')))
f = lambda k: np.array([float(r[k]) for r in rows])
t, x, y, v = f('t'), f('x'), f('y'), f('speed')
lim = np.array([r['limiter'] for r in rows]); mode = np.array([r['mode'] for r in rows])
print(f'{len(rows)} samples, {t[-1]-t[0]:.0f} s, compute ms: mean {f("compute_ms").mean():.1f} max {f("compute_ms").max():.1f}')
print('\nTime share and mean speed per mode / limiter (DRIVE only):')
for m in sorted(set(mode)):
    print(f'  mode {m:8s} {np.mean(mode==m)*100:5.1f}%')
d = mode == 'DRIVE'
for l in sorted(set(lim[d])):
    k = d & (lim == l)
    print(f'  limiter {l:14s} {k.sum()/d.sum()*100:5.1f}%  mean speed {v[k].mean():.2f} m/s')
print('\nCrawl episodes (speed < 1 m/s for > 0.5 s):')
slow = v < 1.0; i = 0
while i < len(v):
    if slow[i]:
        j = i
        while j < len(v) and slow[j]: j += 1
        if t[j-1] - t[i] > 0.5:
            sl = slice(i, j)
            top = max(set(lim[sl]), key=list(lim[sl]).count)
            print(f'  t={t[i]:.1f}s ({t[j-1]-t[i]:.1f}s) at x={x[i]:.1f}, y={y[i]:.1f} mode={mode[i]} '
                  f'limiter={top} d_arc={f("d_arc")[sl].mean():.2f} d_straight={f("d_straight")[sl].mean():.2f}')
        i = j
    else:
        i += 1
