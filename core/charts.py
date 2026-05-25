"""Plotly chart helpers shared across the market-data tabs.

Public API (all return plotly.graph_objects.Figure):
    time_series_chart(wide_df, title, yaxis_title, height)
    percentile_path_chart(series, title, height)
    histogram_with_marker(series, marker_value, title, height)
    term_structure_chart(curves, tenors_ordered, title, yaxis_title, height)
    smile_chart(smile_df, lookback_days, low_pct, high_pct, title, height, ...)
    percentile_heatmap(grid_df, title, height, hover_levels=None)
    time_series_with_quantile_bands(series, quantiles_df, title, ...)
    signal_gauge(value, title, range_min, range_max, height)

Constants:
    COLOR_PALETTE — 10-colour categorical palette for overlaid series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Tableau-like 10-colour categorical palette. Used by time-series and
# overlay charts whenever an arbitrary number of series need distinct
# colours. Cycles via `COLOR_PALETTE[i % len(COLOR_PALETTE)]`.
COLOR_PALETTE: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# -----------------------------------------------------------------------------
# Time-series charts
# -----------------------------------------------------------------------------
def time_series_chart(wide_df: pd.DataFrame,
                          title: str = "",
                          yaxis_title: str = "",
                          height: int = 460) -> go.Figure:
    """Multi-series line chart. `wide_df` columns become individual
    traces, all on a shared y-axis. Date index → x-axis."""
    fig = go.Figure()
    for i, col in enumerate(wide_df.columns):
        ser = wide_df[col].dropna()
        if ser.empty:
            continue
        fig.add_trace(go.Scatter(
            x=ser.index, y=ser.values,
            mode="lines", name=str(col),
            line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=1.6),
        ))
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def percentile_path_chart(series: pd.Series,
                              title: str = "",
                              height: int = 170) -> go.Figure:
    """Single-series percentile chart, 0-100 y-axis with shaded
    extreme bands (10/90 and 1/99). Compact height by default — meant
    to stack under a value chart."""
    fig = go.Figure()
    ser = series.dropna()
    if not ser.empty:
        fig.add_trace(go.Scatter(
            x=ser.index, y=ser.values,
            mode="lines", line=dict(color="#1f77b4", width=1.4),
            showlegend=False, name="Percentile",
            hovertemplate="%{x|%Y-%m-%d}<br>pct: %{y:.1f}<extra></extra>",
        ))
    # Reference bands
    for y_lo, y_hi, color in [
        (10, 90, "rgba(255, 127, 14, 0.10)"),
        (1, 99, "rgba(214, 39, 40, 0.06)"),
    ]:
        fig.add_hrect(y0=y_lo, y1=y_hi, fillcolor=color, line_width=0)
    fig.add_hline(y=50, line_color="#aaa", line_dash="dot", line_width=1)
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 100], title="pct"),
        height=height,
        margin=dict(l=10, r=10, t=30 if title else 5, b=10),
        template="plotly_white",
        showlegend=False,
    )
    return fig


def histogram_with_marker(series: pd.Series,
                                marker_value: float,
                                title: str = "",
                                height: int = 380,
                                bins: int = 50) -> go.Figure:
    """Histogram of `series` with a vertical line at `marker_value`."""
    ser = pd.Series(series).dropna()
    fig = go.Figure()
    if not ser.empty:
        fig.add_trace(go.Histogram(
            x=ser.values, nbinsx=bins,
            marker_color="#1f77b4",
            opacity=0.75, name="Distribution",
            hovertemplate="%{x}<br>count: %{y}<extra></extra>",
        ))
    if marker_value is not None and not pd.isna(marker_value):
        fig.add_vline(x=float(marker_value),
                       line_color="#d62728", line_width=2,
                       annotation_text=f"current = {marker_value:.4g}",
                       annotation_position="top")
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        template="plotly_white", showlegend=False,
        bargap=0.05,
    )
    return fig


def term_structure_chart(curves: dict[str, dict[str, float]],
                              tenors_ordered: list[str],
                              title: str = "",
                              yaxis_title: str = "",
                              height: int = 460) -> go.Figure:
    """Term-structure chart: x-axis = tenor (ordered), one line per
    historical reference point. `curves` is `{label: {tenor: value}}`."""
    fig = go.Figure()
    for i, (label, curve) in enumerate(curves.items()):
        xs = [t for t in tenors_ordered if t in curve]
        ys = [curve[t] for t in xs]
        if not xs:
            continue
        # Highlight "Today" series
        is_today = (label == "Today")
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers", name=label,
            line=dict(
                color="#d62728" if is_today
                      else COLOR_PALETTE[(i + 1) % len(COLOR_PALETTE)],
                width=2.5 if is_today else 1.4,
                dash="solid" if is_today else "dot",
            ),
            marker=dict(size=8 if is_today else 5),
        ))
    fig.update_layout(
        title=title, height=height,
        xaxis=dict(title="Tenor", type="category",
                   categoryorder="array", categoryarray=tenors_ordered),
        yaxis_title=yaxis_title,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


# -----------------------------------------------------------------------------
# Vol-skew chart
# -----------------------------------------------------------------------------
def smile_chart(smile_df: pd.DataFrame,
                    lookback_days: int | None = 1095,
                    low_pct: float = 1.0,
                    high_pct: float = 99.0,
                    title: str = "",
                    height: int = 320,
                    yrange: tuple[float, float] | None = None,
                    show_legend: bool = False) -> go.Figure:
    """One smile chart: 5 delta-strike columns of `smile_df` plotted
    as x-axis, with:
      - shaded band [low_pct, high_pct] over trailing-`lookback_days`
      - dashed line for historical mean smile
      - bold red line for current (last-row) smile

    `smile_df` is the output of compute_smile_panel — DataFrame with
    DELTA_STRIKES columns and DatetimeIndex.
    """
    from core.smile import DELTA_STRIKES
    fig = go.Figure()
    if smile_df is None or smile_df.empty:
        return fig

    cols = [c for c in DELTA_STRIKES if c in smile_df.columns]
    if not cols:
        return fig

    # Trailing window for band calculation
    if isinstance(smile_df.index, pd.DatetimeIndex) and lookback_days:
        cutoff = smile_df.index[-1] - pd.Timedelta(days=int(lookback_days))
        window = smile_df[smile_df.index >= cutoff]
    else:
        window = smile_df

    lo_arr, hi_arr, mean_arr, cur_arr = [], [], [], []
    for col in cols:
        c = window[col].dropna()
        if c.empty:
            lo_arr.append(float("nan"))
            hi_arr.append(float("nan"))
            mean_arr.append(float("nan"))
        else:
            lo_arr.append(float(c.quantile(low_pct / 100.0)))
            hi_arr.append(float(c.quantile(high_pct / 100.0)))
            mean_arr.append(float(c.mean()))
        full_col = smile_df[col].dropna()
        cur_arr.append(float(full_col.iloc[-1]) if not full_col.empty
                          else float("nan"))

    # Shaded band — lower trace + upper-fill trace
    fig.add_trace(go.Scatter(
        x=cols, y=hi_arr, mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="skip", name="hi",
    ))
    fig.add_trace(go.Scatter(
        x=cols, y=lo_arr, mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
        showlegend=show_legend,
        name=f"[{low_pct:g}, {high_pct:g}] pct band",
        hoverinfo="skip",
    ))
    # Historical mean (dashed grey)
    fig.add_trace(go.Scatter(
        x=cols, y=mean_arr, mode="lines+markers",
        line=dict(color="#888", width=1.4, dash="dash"),
        marker=dict(size=5),
        name="Historical mean", showlegend=show_legend,
        hovertemplate="%{x}<br>mean: %{y:.4f}<extra></extra>",
    ))
    # Current (bold red)
    fig.add_trace(go.Scatter(
        x=cols, y=cur_arr, mode="lines+markers",
        line=dict(color="#d62728", width=2.5),
        marker=dict(size=8, color="#d62728"),
        name="Current", showlegend=show_legend,
        hovertemplate="%{x}<br>current: %{y:.4f}<extra></extra>",
    ))

    yaxis_kwargs = {"title": "vol / ATM"}
    if yrange is not None:
        yaxis_kwargs["range"] = list(yrange)

    fig.update_layout(
        title=title, height=height,
        xaxis=dict(type="category", categoryorder="array",
                   categoryarray=cols),
        yaxis=yaxis_kwargs,
        margin=dict(l=10, r=10, t=30 if title else 5, b=10),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1) if show_legend else dict(),
    )
    return fig


# -----------------------------------------------------------------------------
# Percentile heatmap
# -----------------------------------------------------------------------------
def percentile_heatmap(grid_df: pd.DataFrame,
                            title: str = "",
                            height: int = 420,
                            hover_levels: pd.DataFrame | None = None,
                            ) -> go.Figure:
    """Tenor × Strike heatmap of percentiles (0-100), RdYlGn diverging
    palette centred on 50 (median).

    `grid_df` rows are tenors, columns are strikes (or any 2D layout).
    `hover_levels` (optional) is a same-shape DataFrame of underlying
    raw values to show in the hover tooltip alongside the percentile.
    """
    fig = go.Figure()
    if grid_df is None or grid_df.empty:
        return fig

    # Build hover text grid
    if hover_levels is not None:
        hover_text = [
            [f"{grid_df.loc[r, c]:.1f} pct<br>level: {hover_levels.loc[r, c]:.4f}"
             if pd.notna(grid_df.loc[r, c]) and pd.notna(hover_levels.loc[r, c])
             else "—"
             for c in grid_df.columns]
            for r in grid_df.index
        ]
    else:
        hover_text = [
            [f"{grid_df.loc[r, c]:.1f} pct" if pd.notna(grid_df.loc[r, c]) else "—"
             for c in grid_df.columns]
            for r in grid_df.index
        ]

    text_disp = [[f"{v:.0f}" if pd.notna(v) else "" for v in row]
                  for row in grid_df.values]

    fig.add_trace(go.Heatmap(
        z=grid_df.values,
        x=grid_df.columns.tolist(),
        y=grid_df.index.tolist(),
        zmin=0, zmax=100, zmid=50,
        colorscale="RdYlGn_r",  # high pct = red (rich), low = green (cheap)
        text=text_disp, texttemplate="%{text}",
        textfont=dict(size=12),
        customdata=hover_text,
        hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>",
        colorbar=dict(title="pct", thickness=14, len=0.8),
    ))
    fig.update_layout(
        title=title, height=height,
        xaxis=dict(type="category", side="bottom"),
        yaxis=dict(type="category", autorange="reversed"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        template="plotly_white",
    )
    return fig


# -----------------------------------------------------------------------------
# Time-series with quantile bands
# -----------------------------------------------------------------------------
def time_series_with_quantile_bands(series: pd.Series,
                                              quantiles: pd.DataFrame,
                                              title: str = "",
                                              yaxis_title: str = "",
                                              height: int = 240) -> go.Figure:
    """Line chart with reference quantile bands (p10/p25/p50/p75/p90)
    layered behind. `quantiles` is the output of reference_quantiles().
    Bands shown: [p10, p90] (lighter) and [p25, p75] (darker)."""
    fig = go.Figure()
    if quantiles is not None and not quantiles.empty:
        # Align to same date index as series for cleaner plots
        q = quantiles.reindex(series.index).ffill()
        # Outer band p10-p90
        if "p10" in q.columns and "p90" in q.columns:
            fig.add_trace(go.Scatter(
                x=q.index, y=q["p90"], mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=q.index, y=q["p10"], mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tonexty", fillcolor="rgba(31,119,180,0.10)",
                name="p10–p90", showlegend=False, hoverinfo="skip",
            ))
        # Inner band p25-p75
        if "p25" in q.columns and "p75" in q.columns:
            fig.add_trace(go.Scatter(
                x=q.index, y=q["p75"], mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=q.index, y=q["p25"], mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
                name="p25–p75", showlegend=False, hoverinfo="skip",
            ))
        # Median line (dashed)
        if "p50" in q.columns:
            fig.add_trace(go.Scatter(
                x=q.index, y=q["p50"], mode="lines",
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dash"),
                name="Median", hoverinfo="skip", showlegend=False,
            ))

    # Main series on top
    ser = series.dropna()
    if not ser.empty:
        fig.add_trace(go.Scatter(
            x=ser.index, y=ser.values, mode="lines",
            line=dict(color="#1f77b4", width=1.8),
            name="Value", showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, height=height, yaxis_title=yaxis_title,
        margin=dict(l=10, r=10, t=30 if title else 5, b=10),
        template="plotly_white", hovermode="x unified", showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# Time-series with a custom percentile band (user-picked low/high)
# -----------------------------------------------------------------------------
def time_series_with_pct_band(series: pd.Series,
                                       lookback_days: "int | None",
                                       low_pct: float,
                                       high_pct: float,
                                       title: str = "",
                                       yaxis_title: str = "",
                                       height: int = 240) -> go.Figure:
    """Time-series line with a shaded band between the user-selected
    `low_pct` and `high_pct` quantile levels, plus a dashed median line.
    Quantile bands are time-varying (rolling-window trailing quantiles
    over `lookback_days`), so the shading widens/narrows as the
    distribution drifts. `lookback_days=None` means "use full history"
    — the band is the expanding-window quantile cutoffs.

    This is a variant of `time_series_with_quantile_bands` that uses
    the user's chosen percentile cutoffs (typically 1/99 for extremes,
    10/90 for typical range) instead of the hardcoded p10/p25/p50/p75/p90
    set. Use this for "draw me the history with my chosen band" views.
    """
    from core.percentiles import reference_quantiles
    fig = go.Figure()
    ser = series.dropna()
    if ser.empty:
        fig.update_layout(
            title=title, height=height, yaxis_title=yaxis_title,
            margin=dict(l=10, r=10, t=30 if title else 5, b=10),
            template="plotly_white",
        )
        return fig

    # Time-varying quantile cutoffs at the user-picked levels (+ median).
    # Median is included so the dashed reference line still anchors the
    # eye to "middle of typical range".
    q = reference_quantiles(
        ser, lookback_days=lookback_days,
        levels=(low_pct, 50.0, high_pct),
    )
    if not q.empty:
        q = q.reindex(ser.index).ffill()
        lo_col = f"p{int(low_pct)}" if float(low_pct).is_integer() else None
        hi_col = f"p{int(high_pct)}" if float(high_pct).is_integer() else None
        # reference_quantiles uses int() coercion in its column names —
        # fall back to first/last col if the user passed a non-integer
        # level (e.g. 2.5).
        if lo_col is None or lo_col not in q.columns:
            lo_col = q.columns[0]
        if hi_col is None or hi_col not in q.columns:
            hi_col = q.columns[-1]
        # Shaded band: low → high
        fig.add_trace(go.Scatter(
            x=q.index, y=q[hi_col], mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=q.index, y=q[lo_col], mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
            name=f"[{low_pct:g}, {high_pct:g}] pct band",
            showlegend=False, hoverinfo="skip",
        ))
        # Dashed median reference
        if "p50" in q.columns:
            fig.add_trace(go.Scatter(
                x=q.index, y=q["p50"], mode="lines",
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dash"),
                name="Median", hoverinfo="skip", showlegend=False,
            ))

    # Main series — bold red line so it pops against the blue band.
    # Matches the visual language of the Skew tab (current=red, hist=grey).
    fig.add_trace(go.Scatter(
        x=ser.index, y=ser.values, mode="lines",
        line=dict(color="#d62728", width=1.8),
        name="Series", showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
    ))
    # Mark the most recent point so the eye finds "where are we now"
    fig.add_trace(go.Scatter(
        x=[ser.index[-1]], y=[float(ser.iloc[-1])],
        mode="markers",
        marker=dict(color="#d62728", size=8, line=dict(color="white", width=1)),
        name="Current", showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title=title, height=height, yaxis_title=yaxis_title,
        margin=dict(l=10, r=10, t=30 if title else 5, b=10),
        template="plotly_white", hovermode="x unified", showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# Signal gauge (used by app 2 — India)
# -----------------------------------------------------------------------------
def signal_gauge(value: float,
                    title: str = "",
                    range_min: float = -3.0,
                    range_max: float = 3.0,
                    height: int = 300) -> go.Figure:
    """A semi-circular gauge with green-yellow-red colour bands and a
    needle at `value`. Used for composite signals."""
    if pd.isna(value):
        value = 0.0
    val = float(np.clip(value, range_min, range_max))
    span = range_max - range_min
    if span <= 0:
        span = 1.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain=dict(x=[0, 1], y=[0, 1]),
        title=dict(text=title or "", font=dict(size=14)),
        number=dict(font=dict(size=28),
                     valueformat="+.2f"),
        gauge=dict(
            axis=dict(range=[range_min, range_max], tickwidth=1),
            bar=dict(color="#1a1a1a", thickness=0.25),
            steps=[
                dict(range=[range_min, range_min + span * 0.33],
                       color="#2ca02c"),
                dict(range=[range_min + span * 0.33,
                              range_min + span * 0.67], color="#f1c40f"),
                dict(range=[range_min + span * 0.67, range_max],
                       color="#d62728"),
            ],
            threshold=dict(
                line=dict(color="black", width=3),
                thickness=0.85, value=val,
            ),
        ),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 10, b=10),
        template="plotly_white",
    )
    return fig
