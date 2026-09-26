"""Dendrogram plotting + CJK font detection."""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.cluster.hierarchy import dendrogram


def detect_and_add_font() -> str | None:
    possible_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for p in possible_files:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                fp = fm.FontProperties(fname=p)
                return fp.get_name()
            except Exception:
                continue
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    for key in ["Noto", "WenQuan", "SimHei", "Microsoft YaHei", "DejaVu"]:
        for name, fname in available.items():
            if key.lower() in name.lower():
                try:
                    fm.fontManager.addfont(fname)
                    fp = fm.FontProperties(fname=fname)
                    return fp.get_name()
                except Exception:
                    continue
    return None


def plot_selected_dendrogram(
    Z,
    codes: Iterable[str],
    selected_set: set[str],
    code_name_map: dict[str, str],
    out_png: str,
    out_svg: str,
) -> None:
    codes = list(codes)
    labels = []
    for c in codes:
        if c in selected_set:
            name = (
                code_name_map.get(str(c).strip())
                or code_name_map.get(str(c).strip().upper())
                or None
            )
            if name is None and str(c).upper().startswith(("SH", "SZ")):
                name = code_name_map.get(str(c)[2:])
            label = f"{name}_x" if name and name.strip() != "" else f"{c}_x"
            labels.append(label)
        else:
            labels.append("")

    num_labels = max(len([c for c in codes if c in selected_set]), 1)
    max_label_len = max((len(label) for label in labels if label), default=10)
    fig_width = max(36, num_labels * 1.0)
    fig_height = max(10, min(20, 8 + max_label_len * 0.18))
    plt.figure(figsize=(fig_width, fig_height))
    font_name = detect_and_add_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    dendrogram(
        Z,
        labels=labels,
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=False,
        leaf_rotation=90,
        leaf_font_size=7,
    )
    plt.title("Dendrogram - selected representatives only")
    plt.subplots_adjust(bottom=0.55)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.4)
    print("Saved selected dendrogram PNG to", out_png)
    try:
        plt.savefig(out_svg, bbox_inches="tight", pad_inches=0.4)
        print("Saved selected dendrogram SVG to", out_svg)
    except Exception:
        pass
    plt.close()
