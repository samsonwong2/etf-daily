"""fund_pool_builder

工程化、模块化重构自旧 `2_filter/`，负责生成 canonical 的
`cluster_mapping_selected.txt` 待选池文件。

核心子包：

- ``pool_builder``：Python 包形式的核心实现（聚类、筛选、导出、审计、对比）。

输入：手工维护的 ``~/etf-daily-output/Documents/stock/etf_strategy_pipeline/temp/fund_list.csv``（code / name / inception_date 等）。
"""
