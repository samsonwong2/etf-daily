"""Hardcoded lists, output paths and numeric thresholds.

Extracted from ``2_filter/0.2_cluster_select_from_dendrogram.py``. Values are
kept byte-identical so the refactored pipeline reproduces the legacy output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from etf_daily.paths import ALL_INSTRUMENTS_TXT, CLUSTER_MAPPING_SELECTED_TXT, TEMP_DIR
except ModuleNotFoundError:
    project_root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    sys.path.insert(0, str(project_root))
    from etf_daily.paths import ALL_INSTRUMENTS_TXT, CLUSTER_MAPPING_SELECTED_TXT, TEMP_DIR

# ── Output paths ────────────────────────────────────────────────────────────
OUT_DIR = str(TEMP_DIR)
CSV_OUT = os.path.join(OUT_DIR, "cluster_mapping.csv")
CSV_SELECTED_OUT = os.path.join(OUT_DIR, "cluster_mapping_selected.csv")


def dated_selected_csv_path(suffix: str) -> str:
    """Machine-generated selected pool; does not overwrite the hand-maintained CSV."""
    return os.path.join(OUT_DIR, f"cluster_mapping_selected_{suffix.strip()}.csv")


# Legacy benchmark exp paths (no longer written by select_pool).
CSV_BENCHMARK_OUT = os.path.join(OUT_DIR, "cluster_mapping_exp.csv")
CSV_SELECTED_BENCHMARK_OUT = os.path.join(OUT_DIR, "cluster_mapping_selected_exp.csv")
METADATA_OUT = os.path.join(OUT_DIR, "cluster_mapping_selected_metadata.json")
IMG_OUT = os.path.join(OUT_DIR, "dendrogram_selected_reps.png")
SVG_OUT = os.path.join(OUT_DIR, "dendrogram_selected_reps.svg")

MANUAL_KEEP_CODES_FILE = os.path.join(OUT_DIR, "etf_filter_checks", "etf_manual_keep_whitelist.csv")
CLUSTER_REFERENCE_MAP_CSV = os.path.join(OUT_DIR, "cluster_mapping.csv")

# Canonical txt output (consumed by pipeline orchestrator as --cluster-mapping-path).
DEFAULT_ALL_TXT = str(ALL_INSTRUMENTS_TXT)
DEFAULT_CANONICAL_TXT = str(CLUSTER_MAPPING_SELECTED_TXT)

# ── Clustering / selection thresholds ───────────────────────────────────────
# DIST_T 说明：distance threshold (dist = 1 - corr).  0.33 → corr>=0.67, 0.40 → corr>=0.60
# A/B 实验支持：可通过环境变量 POOL_DIST_T 覆盖（不改源码切回 Fix#4 参数）
DIST_T = float(os.environ.get("POOL_DIST_T", 0.40))  # 放宽从 0.33 → 0.40，避免把强相关产品切成多簇后在选中池里出现跨簇近克隆
PER_CLUSTER = 1
RULE = "multi_rep_equal_mainline"
BENCHMARK_RULE = ""  # exp mapping export disabled
PRODUCTION_MULTI_REP_MAINLINE_SCHEME = "equal"
PRODUCTION_MULTI_REP_BENCHMARK_SCHEME = "exp"
DEPRECATED_MULTI_REP_WEIGHT_SCHEMES = ("harmonic",)
MULTI_REP_EXP_BASE = 0.65
# A/B 实验支持：POOL_TARGET_COUNT 环境变量可覆盖。默认 200 (Plan B)；Fix#4 用 120。
TARGET_SELECTED_COUNT = int(os.environ.get("POOL_TARGET_COUNT", 200))  # pre-dedup 目标；近克隆 trim 之后实际 ~80-100 只
CLUSTER_LOOKBACK_DAYS = 252
MIN_CLUSTER_HISTORY_DAYS = 120
MIN_PAIR_OVERLAP_DAYS = 120
MIN_SELECTION_HISTORY_DAYS = 180
LOW_OVERLAP_DISTANCE = 0.95
# Final 去相关阈值：从最严 0.70 起步，以便跨簇近克隆在 trim 阶段被剐掉
# (Tier1 整改：0.75→0.70，进一步压制组合内跨簇冗余)
FINAL_MAX_ABS_CORR_THRESHOLDS = (0.70, 0.75, 0.80)

# 簇内（同 cluster）近克隆阈值：比跨簇阈值更严（同簇成员本来就共享主题，
# 保留多个 rank 只应在它们显著分化时才允许）。默认 0.60；
# 环境变量 POOL_INTRA_CLUSTER_CORR_LIMIT 可覆盖；设为 >= 1.0 即等效关闭。
INTRA_CLUSTER_MAX_ABS_CORR = float(os.environ.get("POOL_INTRA_CLUSTER_CORR_LIMIT", 0.60))

# 注：2026-04-24 移除了 dual-tier 收益预筛 + diversifier 分支。
# 结论：先按收益 top-% 切半再聚类会破坏相关性结构（剩余标的多来自同一波强势板块），
# 且 diversifier 路径缺乏前向预测力（OOS Sharpe -1.16）。
# 回归单一路径："层级聚类 → 簇内按 (Sharpe + 12-1 动量) 挑代表"，相关性由聚类保证，
# 收益由簇内打分保证，两层互不干扰。
#EXCLUDE_TYPES = {""} 
EXCLUDE_TYPES = {"指数型-固收"}

# ── Hand-curated whitelists / blacklists ────────────────────────────────────
ALWAYS_KEEP_CODES = [
    "SH588850",  # 科创机械ETF嘉实
    "SH515880",  # 通信ETF国泰
    "SH513310",  # 中韩半导体ETF华泰柏瑞
    "SH588850",  # 科创机械ETF嘉实
    "SH589720",  # 科创创新药ETF国泰
    "SH517120",  # 创新药ETF华泰柏瑞
    "SH513050",  # 中概互联网ETF易方达
    "SH513400",  # 道琼斯ETF鹏华
    "SH513080",  # 法国ETF华安
    "SZ159561",  # 德国ETF嘉实
    "SH562900",  # 农业ETF易方达
    "SH513970",  # 恒生消费ETF景顺
    "SZ159745",  # 建材ETF国泰
    "SH512690",  # 酒ETF鹏华
    "SH563850",  # 食品ETF广发
    "SZ159502",  # 标普生物科技ETF嘉实
    "SH513290",  # 纳指生物科技ETF汇添富
    "SZ159509",  # 纳指科技ETF景顺
    "SH513730",  # 东南亚科技ETF华泰柏瑞
    "SH517400",  # 黄金股ETF国泰
    "SH513650",  # 标普500ETF南方
    "SH513190",  # 港股通金融ETF华夏
    "SH510300",  # 沪深300ETF
    "SH513520",  # 日经ETF华夏
    "SH518880",  # 黄金ETF华安
    "SZ159600",  # 科创债嘉实（固收；覆盖 EXCLUDE_TYPES）
    "SH511010",  # 5年期国债ETF（固收；覆盖 EXCLUDE_TYPES）
    "SH511260",  # 10年期国债ETF（固收；覆盖 EXCLUDE_TYPES）
    "SZ159827",  # 农业ETF银华
    "SZ159275",  # 农牧渔ETF华宝
    "SZ159698",  # 粮食ETF鹏华
    "SZ159980",  # 有色ETF大成
    "SH513100",  # 纳指ETF国泰
    "SH513850",  # 美国50ETF易方达
    "SZ159892",  # 恒生医药ETF华夏
    "SH511090",	#30年国债ETF鹏扬
    "SH511030",	#10年期国债ETF鹏扬
    "SH588170",	#科创半导体ETF华夏
    "SH512530",	#沪深300红利ETF建信
    "SH512700",	#银行ETF南方
    "SZ159611",	#电力ETF广发
    "SH512880",	#证券ETF国泰
    "SH513090",	#香港证券ETF易方达
    "SZ159981",	#能源化工ETF建信
    "SH515210",	#钢铁ETF国泰
    "SH560710",	#船舶ETF富国
    "SH563380",	#航空航天ETF华泰柏瑞
    "SH510880",	#红利ETF华泰柏瑞
    "SH512200",	#房地产ETF南方
    "SH512660",	#金融地产ETF嘉实
    "SH513350",	#标普油气ETF富国
    "SH515220",	#煤炭ETF国泰


]

ALWAYS_DROP_CODES = [
    "SH562000",
    "SH562060",
    "SZ159851",
    "SH512800",
    "SZ159636",
    "SH513770",
    "SH562080",
    "SH520880",
    "SH515980",  # 人工智能ETF华富   单族剔除
    "SH515260",  # 电子ETF华宝  单族剔除
    "SH520580",  # 新兴亚洲ETF招商  单族剔除
    "SZ159363",  # 创业板人工智能ETF华宝  单族剔除
    "SZ159516",  # 半导体设备ETF国泰  单族剔除
    "SZ159583",  # 通信ETF富国  单族剔除
    "SZ159788",  # 港股通100ETF易方达  单族剔除
    "SZ159822",  # 新经济ETF银华  单族剔除
    "SZ159876",  # 有色ETF华宝  单族剔除
    "SZ159913",  # 深价值ETF交银  单族剔除
    "SH589520",  # 科创人工智能ETF华宝
    "SZ159220",  # 港股通红利低波ETF华宝
    "SZ159246",  # 创业板人工智能ETF富国
    "SZ159287",  # 创业板综ETF博时
    "SZ159387",  # 创业板新能源ETF国泰
    "SZ159388",  # 创业板人工智能ETF国泰
    "SZ159543",  # 国证2000ETF工银
    "SZ159567",  # 港股创新药ETF银华
    "SZ159970",  # 深100ETF工银
    "SZ159971",  # 创业板ETF富国
    "SZ159977",  # 创业板ETF天弘
    "SH562030",  # 信创ETF华宝
    "SH588860",  # 科创医药ETF工银  单族剔除
    "SH513750",  # 港股通非银ETF广发
    "SH588070",  # 科创成长ETF万家
    "SZ159203",  # 大盘成长ETF博时  单族剔除
    "SH512980",  # 传媒ETF广发
    "SH512250",  # A50ETF招商
    "SH589010",  # 科创人工智能ETF华夏   单族剔除
    "SH515650",  # 消费50ETF富国
    "SH516190",  # 传媒ETF华夏
    "SZ159728",  # 在线消费ETF南方
    "SZ159725",  # 线上消费ETF工银
    "SH517770",  # 游戏传媒ETF浦银
    "SH510950",  # 上证50ETF广发   单族剔除
    "SZ159612",  # 标普500ETF国泰   单族剔除
    "SH560630",  # 机器人ETF万家   单族剔除
    "SH589300",  # 科创综指ETF嘉实   单族剔除
    "SH510770",  # G60创新ETF申万菱信   单族剔除
    "SZ159202",  # 恒生互联网ETF万家    整个族剔除
    "SH512870",  # 杭州湾区ETF南华    整个族剔除




    '''
    "SH510050",  # 华夏上证50ETF
    "SH517900",  # 招商中证银行AH价格优选ETF
    "SZ159851",  # 华宝中证金融科技主题ETF
    "SZ159822",  # 银华工银南方东英标普中国新经济ETF(QDII)
    "SZ159687",  # 南方基金南方东英富时亚太低碳精选ETF(QDII)
    "SZ159623",  # 博时成渝经济圈ETF
    "SZ159552",  # 招商中证2000增强策略ETF
    "SH560810",  # 融通中证诚通央企ESGETF
    "SH563830",  # 博时中证全指自由现金流ETF
    "SZ159207",  # 广发中证智选高股息策略ETF
    "SH510010",  # 交银上证180公司治理ETF
    "SH515980",  # 华富中证人工智能产业ETF
    "SZ159363",  # 华宝创业板人工智能ETF
    "SZ159971",  # 富国创业板ETF
    "SZ159393",  # 万家沪深300ETF
    "SH517090",  # 国泰富时中国国企开放共赢ETF
    "SH517550",  # 招商中证沪港深消费龙头ETF
    "SZ159913",  # 交银深证300价值ETF
    "SZ159970",  # 工银瑞信深证100ETF
    "SH512250",  # 招商中证A50ETF
    "SH510020",  # 博时上证超大盘ETF
    '''
]

# Put a seed code here to drop the whole cluster before clustering.
ALWAYS_DROP_CLUSTER_CODES = [
    "SH510600",	#上证50ETF申万菱信
    	
    "SH510010",# 180治理ETF交银
    "SH515150",# 一带一路ETF富国
    "SZ159965",  # 央视50ETF国联  消费族剔除
    "SH517090",  # 央企共赢ETF国泰
    "SH563060",  # 央企50ETF易方达
    "SH512240",  # A50ETF鹏华
    "SH515760",  # 浙江国资ETF华夏
    "SH561090",  # A500增强ETF华安   整个族剔除
    "SH515550",  # 中证500ETF国联   整个族剔除
    "SZ159212",  # 深100ETF南方   整个族剔除
    "SZ159351",  # A500ETF嘉实   整个族剔除
    "SH560360",  # 软件ETF万家   整个族剔除
    "SZ159916",  # 基本面ETF建信   整个族剔除
    "SH510130",  # 上证中盘ETF易方达   整个族剔除
    "SH560810",  # 央企ESGETF融通	整个族剔除
    "SZ159207",  # 高股息ETF广发	整个族剔除
    "SH563830",  # 全指现金流ETF博时	整个族剔除
    "SH512640",  # 金融地产ETF嘉实	整个族剔除













  
    '''
    "SH512250",  # 招商中证A50ETF
    "SH562000",  # 华宝中证A100ETF
    "SH563500",  # 华宝中证A500ETF
    "SH588790",  # 博时科创板人工智能ETF
    '''
]

os.makedirs(OUT_DIR, exist_ok=True)
