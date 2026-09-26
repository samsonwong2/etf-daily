import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from runtime_paths import PLOTLY_OUTPUTS_DIR, QLIB_PROVIDER_URI

# 下面是你原来的 Qlib 分析代码
provider_uri = str(QLIB_PROVIDER_URI)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROUTE_A_DIR = BASE_DIR / "outputs_route_a"

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _normalize_date_label(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%Y%m%d")


def build_run_label(benchmark: list[str], start_date: str, end_date: str) -> str:
    benchmark_label = "_".join(str(code).upper() for code in benchmark)
    return f"{benchmark_label}_{_normalize_date_label(start_date)}_{_normalize_date_label(end_date)}"


def default_route_a_params_csv(run_label: str) -> Path:
    return DEFAULT_ROUTE_A_DIR / f"route_a_params_{run_label}.csv"


def default_route_a_quantiles_csv(run_label: str) -> Path:
    return DEFAULT_ROUTE_A_DIR / f"route_a_quantiles_{run_label}.csv"


def load_route_a_forecast(params_csv: str | None, quantiles_csv: str | None) -> pd.DataFrame | None:
    if not params_csv and not quantiles_csv:
        return None

    merged: pd.DataFrame | None = None

    if params_csv:
        params_df = pd.read_csv(params_csv)
        if "target_date" not in params_df.columns:
            raise ValueError("route_a_params.csv must contain target_date column")
        params_df["target_date"] = pd.to_datetime(params_df["target_date"])
        keep_cols = [
            c
            for c in [
                "target_date",
                "forecast_date",
                "mu_raw",
                "mu",
                "mu_smooth",
                "mu_path",
                "pred_mean_20",
                "real_mean_20",
                "scale_20",
                "pred_ann20",
                "real_ann20",
                "sigma",
                "mu_shrink",
                "signal_strength",
                "fit_success",
                "fit_fallback",
                "fit_message",
            ]
            if c in params_df.columns
        ]
        merged = params_df[keep_cols].copy()

    if quantiles_csv:
        quantiles_df = pd.read_csv(quantiles_csv)
        if "target_date" not in quantiles_df.columns:
            raise ValueError("route_a_quantiles.csv must contain target_date column")
        quantiles_df["target_date"] = pd.to_datetime(quantiles_df["target_date"])
        keep_cols = [c for c in ["target_date", "q05", "q25", "q50", "q75", "q95"] if c in quantiles_df.columns]
        quantiles_df = quantiles_df[keep_cols].copy()
        merged = quantiles_df if merged is None else merged.merge(quantiles_df, on="target_date", how="outer")

    if merged is None or merged.empty:
        return None

    merged = merged.rename(columns={"target_date": "datetime"}).sort_values("datetime")

    # Route A outputs use decimal returns; current pct_change plot uses percentage points.
    for col in [
        "mu_raw",
        "mu",
        "mu_smooth",
        "mu_path",
        "pred_mean_20",
        "real_mean_20",
        "pred_ann20",
        "real_ann20",
        "sigma",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].astype(float) * 100.0

    return merged


def resolve_route_a_input(path_str: str | None, fallback_path: Path) -> str | None:
    if path_str:
        return path_str
    if fallback_path.exists():
        return str(fallback_path)
    return None


class KlinePlotter:
    def __init__(
        self,
        df: pd.DataFrame,
        datetime: str = "datetime",
        open_: str = "$open",
        high: str = "$high",
        low: str = "$low",
        close: str = "$close",
        volume: str = "$volume",
        pct_change: str = "$pct_change",
        template: str = "plotly_white",
        renderer: str = "browser",
        html_out_dir: str = str(PLOTLY_OUTPUTS_DIR),
        save_html: bool = True,
        output_suffix: str = "",
    ):
        self.df = df.copy()
        self.dt = datetime
        self.df[self.dt] = pd.to_datetime(self.df[self.dt])
        self.col = dict(open=open_, high=high, low=low, close=close, volume=volume, pct_change=pct_change)
        self.template = template
        self.renderer = renderer
        self.html_out_dir = Path(html_out_dir)
        self.html_out_dir.mkdir(parents=True, exist_ok=True)
        self.save_html = save_html
        self.output_suffix = output_suffix

    def _save_html(self, fig: go.Figure, name: str) -> Path:
        filename = f"{name}_{self.output_suffix}.html" if self.output_suffix else f"{name}.html"
        html_path = self.html_out_dir / filename
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path

    def _safe_show(self, fig: go.Figure, name: str) -> None:
        html_path = None
        if self.save_html:
            html_path = self._save_html(fig, name=name)
            print(f"[INFO] 已导出 HTML: {html_path}")
        try:
            fig.show(renderer=self.renderer)
        except Exception as exc:
            if html_path is None:
                html_path = self._save_html(fig, name=name)
            print(f"[WARN] 图形打开失败（{exc}），已导出 HTML: {html_path}")

    def _build_plot_window_config(self, date_values: pd.Series) -> dict[str, object]:
        plot_dates = pd.to_datetime(date_values).sort_values().dt.normalize().drop_duplicates()
        category_dates = plot_dates.dt.strftime("%Y-%m-%d").tolist()
        last_date = plot_dates.iloc[-1]

        def find_window_start(months: int) -> str:
            threshold = last_date - pd.DateOffset(months=months)
            candidates = plot_dates[plot_dates >= threshold]
            start_ts = candidates.iloc[0] if not candidates.empty else plot_dates.iloc[0]
            return start_ts.strftime("%Y-%m-%d")

        full_start = category_dates[0]
        full_end = category_dates[-1]
        last_6m_start = find_window_start(6)
        last_3m_start = find_window_start(3)
        date_to_idx = {date_str: idx for idx, date_str in enumerate(category_dates)}

        def category_range(start_date_str: str, end_date_str: str) -> list[float]:
            start_idx = date_to_idx.get(start_date_str, 0)
            end_idx = date_to_idx.get(end_date_str, len(category_dates) - 1)
            return [start_idx - 0.5, end_idx + 0.5]

        return {
            "category_dates": category_dates,
            "full_start": full_start,
            "full_end": full_end,
            "last_6m_start": last_6m_start,
            "last_3m_start": last_3m_start,
            "full_x_range": category_range(full_start, full_end),
            "last_6m_x_range": category_range(last_6m_start, full_end),
            "last_3m_x_range": category_range(last_3m_start, full_end),
        }

    def _compute_y_range(self, y_axis_df: pd.DataFrame, start_date_str: str, end_date_str: str) -> list[float]:
        window_df = y_axis_df[
            (y_axis_df["plot_date"] >= start_date_str) & (y_axis_df["plot_date"] <= end_date_str)
        ]
        if window_df.empty:
            window_df = y_axis_df

        y_min = float(window_df["y_value"].min())
        y_max = float(window_df["y_value"].max())
        span = y_max - y_min
        if span <= 0:
            pad = max(abs(y_min) * 0.1, 1.0)
        else:
            pad = max(span * 0.08, 0.5)
        return [y_min - pad, y_max + pad]

    # ------------ K 线 + 成交量 ------------
    def candle_vol(self, start_date=None, end_date=None, title="K线成交量图", height=800, show=True):
        df = self._filter(start_date, end_date)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, "成交量"),
            row_heights=[0.7, 0.3],
        )
        # K 线
        fig.add_trace(
            go.Candlestick(
                x=df[self.dt],
                open=df[self.col["open"]],
                high=df[self.col["high"]],
                low=df[self.col["low"]],
                close=df[self.col["close"]],
                name="K线",
            ),
            row=1,
            col=1,
        )
        # 成交量（只保留一次）
        fig.add_trace(
            go.Bar(
                x=df[self.dt],
                y=df[self.col["volume"]],
                name="成交量",
                marker_color="rgba(0,100,128,0.6)",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            title=title,
            yaxis_title="价格",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            template=self.template,
            height=height,
        )
        fig.update_xaxes(
            title_text="日期",
            tickformat="%Y-%m-%d",
            tickangle=-35,
            showticklabels=True,
            row=2,
            col=1,
        )
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            showticklabels=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="成交量", row=2, col=1)

        if show:
            self._safe_show(fig, name="kline_volume")
        elif self.save_html:
            html_path = self._save_html(fig, name="kline_volume")
            print(f"[INFO] 已导出 HTML: {html_path}")
        return fig

    # ------------ 收盘价折线 ------------
    def close_line(self, start_date=None, end_date=None, title="收盘价走势", height=500, show=True, forecast_df: pd.DataFrame | None = None):
        df = self._filter(start_date, end_date)
        df = df.copy()
        df["_plot_date"] = df[self.dt].dt.strftime("%Y-%m-%d")
        y_axis_sources = [
            pd.DataFrame(
                {
                    "plot_date": df["_plot_date"],
                    "y_value": pd.to_numeric(df[self.col["pct_change"]], errors="coerce"),
                }
            )
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["_plot_date"],
                y=df[self.col["pct_change"]],
                mode="lines+markers",
                name="涨跌幅",
                line=dict(color="royalblue", width=2),
                marker=dict(size=6, color="royalblue", opacity=0.8),
                hovertemplate="日期=%{x}<br>实际涨跌幅=%{y:.3f}%<extra></extra>",
            )
        )

        if forecast_df is not None and not forecast_df.empty:
            forecast_plot = forecast_df.copy()
            forecast_plot[self.dt] = pd.to_datetime(forecast_plot[self.dt])
            if start_date is not None:
                forecast_plot = forecast_plot[forecast_plot[self.dt] >= pd.to_datetime(start_date)]
            if end_date is not None:
                forecast_plot = forecast_plot[forecast_plot[self.dt] <= pd.to_datetime(end_date)]

            if not forecast_plot.empty:
                actual_col = self.col["pct_change"]
                actual_df = df[[self.dt, actual_col]].rename(columns={actual_col: "actual_pct_change"}).copy()
                forecast_plot = forecast_plot.merge(actual_df, on=self.dt, how="left")
                forecast_plot["forecast_date"] = pd.to_datetime(forecast_plot.get("forecast_date"), errors="coerce")
                forecast_plot["target_date"] = pd.to_datetime(forecast_plot[self.dt], errors="coerce")
                forecast_plot["_plot_date"] = forecast_plot[self.dt].dt.strftime("%Y-%m-%d")
                for col in ["mu", "q05", "q25", "q75", "q95", "actual_pct_change"]:
                    if col in forecast_plot.columns:
                        forecast_plot[col] = pd.to_numeric(forecast_plot[col], errors="coerce")
                        y_axis_sources.append(
                            pd.DataFrame(
                                {
                                    "plot_date": forecast_plot["_plot_date"],
                                    "y_value": forecast_plot[col],
                                }
                            )
                        )
                forecast_plot["forecast_error"] = forecast_plot["actual_pct_change"] - forecast_plot["mu"]

                def make_customdata(frame: pd.DataFrame) -> np.ndarray:
                    cols = [
                        frame["forecast_date"].dt.strftime("%Y-%m-%d").fillna(""),
                        frame["target_date"].dt.strftime("%Y-%m-%d").fillna(""),
                        frame.get("actual_pct_change", pd.Series(index=frame.index, dtype=float)),
                        frame.get("mu", pd.Series(index=frame.index, dtype=float)),
                        frame.get("forecast_error", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q05", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q25", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q75", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q95", pd.Series(index=frame.index, dtype=float)),
                    ]
                    return np.column_stack(cols)

                customdata = make_customdata(forecast_plot)
                hover_template = (
                    "target_date=%{customdata[1]}<br>"
                    "forecast_date=%{customdata[0]}<br>"
                    "实际值=%{customdata[2]:.3f}%<br>"
                    "预测均值(mu)=%{customdata[3]:.3f}%<br>"
                    "预测误差(实际-mu)=%{customdata[4]:.3f}%<br>"
                    "q05=%{customdata[5]:.3f}%<br>"
                    "q25=%{customdata[6]:.3f}%<br>"
                    "q75=%{customdata[7]:.3f}%<br>"
                    "q95=%{customdata[8]:.3f}%<extra></extra>"
                )
                hover_template_95 = (
                    "target_date=%{customdata[1]}<br>"
                    "forecast_date=%{customdata[0]}<br>"
                    "q05=%{customdata[5]:.3f}%<br>"
                    "q95=%{customdata[8]:.3f}%<extra></extra>"
                )
                hover_template_50 = (
                    "target_date=%{customdata[1]}<br>"
                    "forecast_date=%{customdata[0]}<br>"
                    "q25=%{customdata[6]:.3f}%<br>"
                    "q75=%{customdata[7]:.3f}%<extra></extra>"
                )

                if {"q05", "q95"}.issubset(forecast_plot.columns):
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q95"],
                            mode="lines",
                            line=dict(color="rgba(255,127,14,0.0)"),
                            customdata=customdata,
                            hoverinfo="skip",
                            showlegend=False,
                            name="95%区间上界",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q05"],
                            mode="lines",
                            line=dict(color="rgba(255,127,14,0.0)"),
                            fill="tonexty",
                            fillcolor="rgba(255,127,14,0.18)",
                            customdata=customdata,
                            hovertemplate=hover_template_95,
                            name="95%预测区间",
                        )
                    )

                if {"q25", "q75"}.issubset(forecast_plot.columns):
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q75"],
                            mode="lines",
                            line=dict(color="rgba(44,160,44,0.0)"),
                            customdata=customdata,
                            hoverinfo="skip",
                            showlegend=False,
                            name="50%区间上界",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q25"],
                            mode="lines",
                            line=dict(color="rgba(44,160,44,0.0)"),
                            fill="tonexty",
                            fillcolor="rgba(44,160,44,0.18)",
                            customdata=customdata,
                            hovertemplate=hover_template_50,
                            name="50%预测区间",
                        )
                    )

                if "mu" in forecast_plot.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["mu"],
                            mode="lines+markers",
                            name="预测均值(mu)",
                            line=dict(color="crimson", width=2, dash="dash"),
                            marker=dict(size=5, color="crimson", opacity=0.8),
                            customdata=customdata,
                            hovertemplate=hover_template,
                        )
                    )

        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="涨跌幅(%)",
            hovermode="x unified",
            template=self.template,
            height=height,
        )

        window_cfg = self._build_plot_window_config(df[self.dt])
        y_axis_df = pd.concat(y_axis_sources, ignore_index=True).dropna(subset=["y_value"])
        full_y_range = self._compute_y_range(y_axis_df, window_cfg["full_start"], window_cfg["full_end"])
        last_6m_y_range = self._compute_y_range(y_axis_df, window_cfg["last_6m_start"], window_cfg["full_end"])
        last_3m_y_range = self._compute_y_range(y_axis_df, window_cfg["last_3m_start"], window_cfg["full_end"])

        fig.update_xaxes(
            type="category",
            tickangle=-35,
            categoryorder="array",
            categoryarray=window_cfg["category_dates"],
            range=window_cfg["last_6m_x_range"],
        )
        fig.update_yaxes(range=last_6m_y_range, autorange=False)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0,
                    y=1.16,
                    showactive=True,
                    buttons=[
                        dict(
                            label="全样本",
                            method="relayout",
                            args=[{"xaxis.range": window_cfg["full_x_range"], "yaxis.range": full_y_range}],
                        ),
                        dict(
                            label="最近半年",
                            method="relayout",
                            args=[{"xaxis.range": window_cfg["last_6m_x_range"], "yaxis.range": last_6m_y_range}],
                        ),
                        dict(
                            label="最近三个月",
                            method="relayout",
                            args=[{"xaxis.range": window_cfg["last_3m_x_range"], "yaxis.range": last_3m_y_range}],
                        ),
                    ],
                )
            ]
        )
        if show:
            self._safe_show(fig, name="pct_change_line")
        elif self.save_html:
            html_path = self._save_html(fig, name="pct_change_line")
            print(f"[INFO] 已导出 HTML: {html_path}")
        return fig

    def close_line_with_cumulative(self, start_date=None, end_date=None, title="涨跌幅、预测分布与累计收益", height=900, show=True, forecast_df: pd.DataFrame | None = None):
        df = self._filter(start_date, end_date).copy()
        df["_plot_date"] = df[self.dt].dt.strftime("%Y-%m-%d")
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.60, 0.40],
            specs=[[{}], [{"secondary_y": True}]],
            subplot_titles=("日涨跌幅、原始mu与平滑路径mu", "累计收益与20日滚动年化收益对比"),
        )

        daily_y_sources = [
            pd.DataFrame(
                {
                    "plot_date": df["_plot_date"],
                    "y_value": pd.to_numeric(df[self.col["pct_change"]], errors="coerce"),
                }
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=df["_plot_date"],
                y=df[self.col["pct_change"]],
                mode="lines+markers",
                name="涨跌幅",
                line=dict(color="royalblue", width=2),
                marker=dict(size=6, color="royalblue", opacity=0.8),
                hovertemplate="日期=%{x}<br>实际涨跌幅=%{y:.3f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        path_df = None
        summary_text = None
        if forecast_df is not None and not forecast_df.empty:
            forecast_plot = forecast_df.copy()
            forecast_plot[self.dt] = pd.to_datetime(forecast_plot[self.dt])
            if start_date is not None:
                forecast_plot = forecast_plot[forecast_plot[self.dt] >= pd.to_datetime(start_date)]
            if end_date is not None:
                forecast_plot = forecast_plot[forecast_plot[self.dt] <= pd.to_datetime(end_date)]

            if not forecast_plot.empty:
                actual_col = self.col["pct_change"]
                actual_df = df[[self.dt, actual_col]].rename(columns={actual_col: "actual_pct_change"}).copy()
                forecast_plot = forecast_plot.merge(actual_df, on=self.dt, how="left")
                forecast_plot["forecast_date"] = pd.to_datetime(forecast_plot.get("forecast_date"), errors="coerce")
                forecast_plot["target_date"] = pd.to_datetime(forecast_plot[self.dt], errors="coerce")
                forecast_plot["_plot_date"] = forecast_plot[self.dt].dt.strftime("%Y-%m-%d")
                main_mu_col = "mu_path" if "mu_path" in forecast_plot.columns else "mu"
                raw_mu_col = "mu_raw" if "mu_raw" in forecast_plot.columns else "mu"
                for col in [raw_mu_col, main_mu_col, "q05", "q25", "q75", "q95", "actual_pct_change"]:
                    if col in forecast_plot.columns:
                        forecast_plot[col] = pd.to_numeric(forecast_plot[col], errors="coerce")
                        daily_y_sources.append(
                            pd.DataFrame(
                                {
                                    "plot_date": forecast_plot["_plot_date"],
                                    "y_value": forecast_plot[col],
                                }
                            )
                        )
                forecast_plot["forecast_error"] = forecast_plot["actual_pct_change"] - forecast_plot[main_mu_col]

                def make_customdata(frame: pd.DataFrame) -> np.ndarray:
                    cols = [
                        frame["forecast_date"].dt.strftime("%Y-%m-%d").fillna(""),
                        frame["target_date"].dt.strftime("%Y-%m-%d").fillna(""),
                        frame.get("actual_pct_change", pd.Series(index=frame.index, dtype=float)),
                        frame.get(raw_mu_col, pd.Series(index=frame.index, dtype=float)),
                        frame.get(main_mu_col, pd.Series(index=frame.index, dtype=float)),
                        frame.get("forecast_error", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q05", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q25", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q75", pd.Series(index=frame.index, dtype=float)),
                        frame.get("q95", pd.Series(index=frame.index, dtype=float)),
                    ]
                    return np.column_stack(cols)

                customdata = make_customdata(forecast_plot)
                hover_template = (
                    "target_date=%{customdata[1]}<br>"
                    "forecast_date=%{customdata[0]}<br>"
                    "实际值=%{customdata[2]:.3f}%<br>"
                    "原始mu=%{customdata[3]:.3f}%<br>"
                    "路径mu=%{customdata[4]:.3f}%<br>"
                    "预测误差(实际-路径mu)=%{customdata[5]:.3f}%<br>"
                    "q05=%{customdata[6]:.3f}%<br>"
                    "q25=%{customdata[7]:.3f}%<br>"
                    "q75=%{customdata[8]:.3f}%<br>"
                    "q95=%{customdata[9]:.3f}%<extra></extra>"
                )
                hover_template_95 = (
                    "target_date=%{customdata[1]}<br>"
                    "forecast_date=%{customdata[0]}<br>"
                    "q05=%{customdata[6]:.3f}%<br>"
                    "q95=%{customdata[9]:.3f}%<extra></extra>"
                )
                hover_template_50 = (
                    "target_date=%{customdata[1]}<br>"
                    "forecast_date=%{customdata[0]}<br>"
                    "q25=%{customdata[7]:.3f}%<br>"
                    "q75=%{customdata[8]:.3f}%<extra></extra>"
                )

                if {"q05", "q95"}.issubset(forecast_plot.columns):
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q95"],
                            mode="lines",
                            line=dict(color="rgba(255,127,14,0.0)"),
                            customdata=customdata,
                            hoverinfo="skip",
                            showlegend=False,
                            name="95%区间上界",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q05"],
                            mode="lines",
                            line=dict(color="rgba(255,127,14,0.0)"),
                            fill="tonexty",
                            fillcolor="rgba(255,127,14,0.18)",
                            customdata=customdata,
                            hovertemplate=hover_template_95,
                            name="95%预测区间",
                        ),
                        row=1,
                        col=1,
                    )

                if {"q25", "q75"}.issubset(forecast_plot.columns):
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q75"],
                            mode="lines",
                            line=dict(color="rgba(44,160,44,0.0)"),
                            customdata=customdata,
                            hoverinfo="skip",
                            showlegend=False,
                            name="50%区间上界",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot["q25"],
                            mode="lines",
                            line=dict(color="rgba(44,160,44,0.0)"),
                            fill="tonexty",
                            fillcolor="rgba(44,160,44,0.18)",
                            customdata=customdata,
                            hovertemplate=hover_template_50,
                            name="50%预测区间",
                        ),
                        row=1,
                        col=1,
                    )

                if raw_mu_col in forecast_plot.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot[raw_mu_col],
                            mode="lines",
                            name="原始mu",
                            line=dict(color="crimson", width=1.4, dash="dot"),
                            customdata=customdata,
                            hovertemplate=hover_template,
                        ),
                        row=1,
                        col=1,
                    )

                if main_mu_col in forecast_plot.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_plot["_plot_date"],
                            y=forecast_plot[main_mu_col],
                            mode="lines+markers",
                            name="平滑路径mu",
                            line=dict(color="darkorange", width=2.2),
                            marker=dict(size=4, color="darkorange", opacity=0.75),
                            customdata=customdata,
                            hovertemplate=hover_template,
                        ),
                        row=1,
                        col=1,
                    )

                path_cols = ["_plot_date", "actual_pct_change", raw_mu_col, main_mu_col]
                for col in ["pred_ann20", "real_ann20"]:
                    if col in forecast_plot.columns:
                        path_cols.append(col)
                path_df = forecast_plot[path_cols].copy()
                path_df = path_df.dropna(subset=["actual_pct_change", main_mu_col]).sort_values("_plot_date")
                if not path_df.empty:
                    path_df["wealth_actual"] = (1.0 + path_df["actual_pct_change"] / 100.0).cumprod()
                    path_df["wealth_path"] = (1.0 + path_df[main_mu_col] / 100.0).cumprod()
                    path_df["wealth_raw"] = (1.0 + path_df[raw_mu_col].fillna(0.0) / 100.0).cumprod()
                    path_df["cum_actual"] = (path_df["wealth_actual"] - 1.0) * 100.0
                    path_df["cum_path"] = (path_df["wealth_path"] - 1.0) * 100.0
                    path_df["cum_raw"] = (path_df["wealth_raw"] - 1.0) * 100.0
                    if "pred_ann20" not in path_df.columns:
                        path_df["pred_ann20"] = path_df[main_mu_col].rolling(20, min_periods=5).mean() * 252.0
                    if "real_ann20" not in path_df.columns:
                        path_df["real_ann20"] = path_df["actual_pct_change"].rolling(20, min_periods=5).mean() * 252.0

                    fig.add_trace(
                        go.Scatter(
                            x=path_df["_plot_date"],
                            y=path_df["cum_actual"],
                            mode="lines+markers",
                            name="累计实际收益",
                            line=dict(color="royalblue", width=2),
                            marker=dict(size=5, color="royalblue", opacity=0.8),
                            hovertemplate="日期=%{x}<br>累计实际收益=%{y:.3f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=path_df["_plot_date"],
                            y=path_df["cum_path"],
                            mode="lines+markers",
                            name="平滑预测累计收益",
                            line=dict(color="darkorange", width=2),
                            marker=dict(size=4, color="darkorange", opacity=0.75),
                            hovertemplate="日期=%{x}<br>平滑预测累计收益=%{y:.3f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=path_df["_plot_date"],
                            y=path_df["cum_raw"],
                            mode="lines",
                            name="原始mu累计收益",
                            line=dict(color="firebrick", width=1.5, dash="dot"),
                            hovertemplate="日期=%{x}<br>原始mu累计收益=%{y:.3f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=path_df["_plot_date"],
                            y=path_df["real_ann20"],
                            mode="lines",
                            name="实际20日滚动年化",
                            line=dict(color="seagreen", width=2),
                            hovertemplate="日期=%{x}<br>实际20日滚动年化=%{y:.2f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                        secondary_y=True,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=path_df["_plot_date"],
                            y=path_df["pred_ann20"],
                            mode="lines",
                            name="预测20日滚动年化",
                            line=dict(color="mediumpurple", width=2, dash="dash"),
                            hovertemplate="日期=%{x}<br>预测20日滚动年化=%{y:.2f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                        secondary_y=True,
                    )

                    final_gap = float((path_df["cum_actual"].iloc[-1] - path_df["cum_path"].iloc[-1]))
                    ann_gap = float((path_df["real_ann20"] - path_df["pred_ann20"]).abs().dropna().mean())
                    summary_text = (
                        f"final_cum_gap={final_gap:.2f}%"
                        f" | mean_ann_gap={ann_gap:.2f}%"
                        f" | end_actual={path_df['cum_actual'].iloc[-1]:.2f}%"
                        f" | end_path={path_df['cum_path'].iloc[-1]:.2f}%"
                    )

        fig.update_layout(
            title=title if not summary_text else f"{title}<br><sup>{summary_text}</sup>",
            hovermode="x unified",
            template=self.template,
            height=height,
        )
        fig.update_yaxes(title_text="日涨跌幅(%)", row=1, col=1)
        fig.update_yaxes(title_text="累计收益(%)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="20日滚动年化收益(%)", row=2, col=1, secondary_y=True)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        window_cfg = self._build_plot_window_config(df[self.dt])
        fig.update_xaxes(
            type="category",
            tickangle=-35,
            categoryorder="array",
            categoryarray=window_cfg["category_dates"],
            range=window_cfg["last_6m_x_range"],
            row=1,
            col=1,
        )
        fig.update_xaxes(
            type="category",
            tickangle=-35,
            categoryorder="array",
            categoryarray=window_cfg["category_dates"],
            range=window_cfg["last_6m_x_range"],
            row=2,
            col=1,
        )
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0,
                    y=1.12,
                    showactive=True,
                    buttons=[
                        dict(
                            label="全样本",
                            method="relayout",
                            args=[
                                {
                                    "xaxis.range": window_cfg["full_x_range"],
                                    "xaxis2.range": window_cfg["full_x_range"],
                                }
                            ],
                        ),
                        dict(
                            label="最近半年",
                            method="relayout",
                            args=[
                                {
                                    "xaxis.range": window_cfg["last_6m_x_range"],
                                    "xaxis2.range": window_cfg["last_6m_x_range"],
                                }
                            ],
                        ),
                        dict(
                            label="最近三个月",
                            method="relayout",
                            args=[
                                {
                                    "xaxis.range": window_cfg["last_3m_x_range"],
                                    "xaxis2.range": window_cfg["last_3m_x_range"],
                                }
                            ],
                        ),
                    ],
                )
            ]
        )

        if show:
            self._safe_show(fig, name="pct_change_line_with_cumulative")
        elif self.save_html:
            html_path = self._save_html(fig, name="pct_change_line_with_cumulative")
            print(f"[INFO] 已导出 HTML: {html_path}")
        return fig

    # ------------ 公用日期过滤 ------------
    def _filter(self, start, end):
        df = self.df
        if start is not None:
            df = df[df[self.dt] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df[self.dt] <= pd.to_datetime(end)]
        if df.empty:
            raise ValueError("起止日期范围内无数据，请检查输入！")
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按命令行参数绘制指定标的K线/成交量/涨跌幅图")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-05-01",
        help="回测起始日期，例如 2024-05-01",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-05-13",
        help="回测结束日期，例如 2026-05-13",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        default=["SH513310"],
        help="标的代码，支持多个，例如 --benchmark SH513310 SH510300",
    )
    parser.add_argument(
        "--provider-uri",
        type=str,
        default=provider_uri,
        help="Qlib 数据目录",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="plotly_white",
        help="Plotly 模板",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="browser",
        help="Plotly renderer，默认 browser",
    )
    parser.add_argument(
        "--html-out-dir",
        type=str,
        default=str(PLOTLY_OUTPUTS_DIR),
        help="图形打开失败时导出的 HTML 目录",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗显示图（默认关闭，直接运行更稳定）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=True,
        help="不弹窗显示图，仅执行数据处理（默认开启）",
    )
    parser.add_argument(
        "--no-save-html",
        action="store_true",
        help="不导出 HTML 文件",
    )
    parser.add_argument(
        "--route-a-params-csv",
        type=str,
        default=None,
        help="Route A 参数文件；默认自动使用当前目录 outputs_route_a 下与标的和时间段匹配的命名文件",
    )
    parser.add_argument(
        "--route-a-quantiles-csv",
        type=str,
        default=None,
        help="Route A 分位数文件；默认自动使用当前目录 outputs_route_a 下与标的和时间段匹配的命名文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    test_period = (args.start_date, args.end_date)
    benchmark_list = args.benchmark
    run_label = build_run_label(benchmark_list, args.start_date, args.end_date)

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    benchmark_old: pd.DataFrame = D.features(
        benchmark_list,
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=test_period[0],
        end_time=test_period[1],
    )

    # 计算方法：当日涨跌幅 = (当日收盘价 - 前一日收盘价) / 前一日收盘价 * 100%
    benchmark_old["$pct_change"] = benchmark_old["$close"].pct_change() * 100

    benchmark_old_test = benchmark_old.reset_index().rename(columns={"index": "datetime"})
    route_a_params_csv = resolve_route_a_input(args.route_a_params_csv, default_route_a_params_csv(run_label))
    route_a_quantiles_csv = resolve_route_a_input(args.route_a_quantiles_csv, default_route_a_quantiles_csv(run_label))
    forecast_df = load_route_a_forecast(route_a_params_csv, route_a_quantiles_csv)
    plotter = KlinePlotter(
        benchmark_old_test,
        template=args.template,
        renderer=args.renderer,
        html_out_dir=args.html_out_dir,
        save_html=not args.no_save_html,
        output_suffix=run_label,
    )

    if forecast_df is None or forecast_df.empty:
        print("[WARN] 未加载到 Route A 预测分布，对应命名的 pct_change_line HTML 将只显示实际涨跌幅。")
    else:
        print(
            f"[INFO] 已加载 Route A 预测分布: {len(forecast_df)} 行"
            f" | params={route_a_params_csv or 'None'}"
            f" | quantiles={route_a_quantiles_csv or 'None'}"
        )

    show_fig = bool(args.show and not args.no_show)
    if not show_fig:
        print("[INFO] 当前为无弹窗模式：仅导出 HTML。若需弹窗显示，请加 --show")

    # 1) K 线 + 成交量（只画一次成交量）
    plotter.candle_vol(start_date=test_period[0], end_date=test_period[1], show=show_fig)

    # 2) 涨跌幅折线
    plotter.close_line(
        start_date=test_period[0],
        end_date=test_period[1],
        show=show_fig,
        title="涨跌幅与Route A预测分布",
        forecast_df=forecast_df,
    )
    plotter.close_line_with_cumulative(
        start_date=test_period[0],
        end_date=test_period[1],
        show=show_fig,
        title="涨跌幅、Route A平滑收益路径与年化收益对比",
        forecast_df=forecast_df,
    )


if __name__ == "__main__":
    main()
