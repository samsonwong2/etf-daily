import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from pool_builder import constants as C
from pool_builder.data_loading import load_fund_universe, load_close_volume
from pool_builder.selector import _resolve_current_cluster_drop_codes, _build_diagnostic_view
from pool_builder.utils import build_normalized_code_set, code_in_normalized_set

cfg = json.loads(Path('shared_filter_config.json').read_text(encoding='utf-8'))
fund_df = load_fund_universe(cfg['fund_list_csv'], C.EXCLUDE_TYPES, cfg['test_period'][1])
close, vol = load_close_volume(cfg['provider_uri'], cfg['market'], fund_df['code'].tolist())
drop_cluster = _resolve_current_cluster_drop_codes(close, vol, C.ALWAYS_DROP_CLUSTER_CODES)
keep = [c for c in close.columns if not code_in_normalized_set(c, build_normalized_code_set(C.ALWAYS_DROP_CODES)) and not code_in_normalized_set(c, build_normalized_code_set(drop_cluster))]
close = close[keep]
corr = _build_diagnostic_view(close, vol)['corr']
target = next(c for c in corr.index if '513310' in c.upper())
keeps = C.ALWAYS_KEEP_CODES
lines = [f'target={target}', f'near_clone_limit={C.FINAL_MAX_ABS_CORR_THRESHOLDS[0]}', '']
rows = []
for k in keeps:
    kc = next((c for c in corr.columns if c.upper().endswith(k[-6:].upper()) or c.upper() == k.upper()), None)
    if kc is None:
        kc = next((c for c in corr.columns if k.replace('SH','').replace('SZ','') in c.upper()), k)
    if kc in corr.columns:
        v = abs(float(corr.loc[target, kc]))
        rows.append((v, k, kc))
rows.sort(reverse=True)
for v, k, kc in rows:
    lines.append(f'{k}\t{kc}\t{v:.6f}\t{"BLOCK" if v > 0.70 else ""}')
Path('~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/corr_513310_keep.txt').write_text('\n'.join(lines), encoding='utf-8')
print('wrote', len(rows), 'rows')
