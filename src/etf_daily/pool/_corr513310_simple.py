import pandas as pd
from pathlib import Path

# Use ablation fullpanel close series (Apr-May 2026) as proxy for recent corr
fp = Path('~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/trend_sleeve_ablation_20260527/2025_regime_switch_ewma_shrink_no_sleeve/new_raw_data_fullpanel_20260401_20260527.csv')
df = pd.read_csv(fp)
df['datetime'] = pd.to_datetime(df['datetime'])
pivot = df.pivot(index='datetime', columns='instrument', values='close').sort_index()
ret = pivot.pct_change().dropna(how='all')
target = 'SH513310'
keeps = ['SH515050', 'SZ159671', 'SZ159761', 'SH562950', 'SZ159967', 'SZ159663']
lines = []
if target in ret.columns:
    for k in keeps:
        if k in ret.columns:
            v = abs(ret[target].corr(ret[k]))
            lines.append(f'{k}\t{v:.4f}\t{"BLOCK>0.70" if v>0.70 else ""}')
    # max among keeps
    if lines:
        best = max(lines, key=lambda s: float(s.split('\t')[1]))
        lines.append('MAX_KEEP=' + best)
else:
    lines.append('target missing in fullpanel')
out = Path('~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/corr_513310_keep_simple.txt')
out.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
