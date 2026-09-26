"""pool_builder package: generates ``cluster_mapping_selected.txt``.

Modules:

- :mod:`pool_builder.constants` — hardcoded lists, path defaults, thresholds.
- :mod:`pool_builder.config` — JSON config loader + metadata writer.
- :mod:`pool_builder.code_utils` — code normalization / alias helpers.
- :mod:`pool_builder.data_loading` — qlib init + code-name map + close/volume loading.
- :mod:`pool_builder.clustering` — windowing, overlap, multi-rep pool, trimming.
- :mod:`pool_builder.dendrogram` — dendrogram plotting with CJK fonts.
- :mod:`pool_builder.export` — ``cluster_mapping*.csv`` + metadata JSON writer.
- :mod:`pool_builder.selector` — high-level ``select_pool()`` pipeline.
- :mod:`pool_builder.filter_txt` — CSV → canonical ``all.txt``-format txt writer.
- :mod:`pool_builder.audit` — missed-winner audit (was ``0.5_audit_missed_winners.py``).
- :mod:`pool_builder.compare` — two-pool diff report (was ``compare_cluster_pool.py``).
- :mod:`pool_builder.cli` — top-level orchestrator (``select → filter_txt``).
"""
