"""Visualization functions for dimension quality charts."""

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Chinese font setting (Noto Sans CJK available on Linux)
CJK_FONT = "Noto Sans CJK TC, Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif"


def add_spec_lines(fig: go.Figure, nominal: float, upper: float, lower: float) -> None:
    """Add specification lines and annotations to figure."""
    ann_x = 1.02
    if pd.notna(upper):
        fig.add_hline(y=upper, line_color="red", line_width=2)
        fig.add_annotation(
            x=ann_x,
            xref="paper",
            y=upper,
            yref="y",
            text=f"上限 {upper:.4f}",
            showarrow=False,
            font=dict(family=CJK_FONT, color="red", size=12),
            xanchor="left",
        )
    if pd.notna(nominal):
        fig.add_hline(y=nominal, line_color="red", line_width=1, line_dash="dot")
        fig.add_annotation(
            x=ann_x,
            xref="paper",
            y=nominal,
            yref="y",
            text=f"中值 {nominal:.4f}",
            showarrow=False,
            font=dict(family=CJK_FONT, color="red", size=12),
            xanchor="left",
        )
    if pd.notna(lower):
        fig.add_hline(y=lower, line_color="red", line_width=2)
        fig.add_annotation(
            x=ann_x,
            xref="paper",
            y=lower,
            yref="y",
            text=f"下限 {lower:.4f}",
            showarrow=False,
            font=dict(family=CJK_FONT, color="red", size=12),
            xanchor="left",
        )


def _hover_columns(sub: pd.DataFrame) -> list:
    """Pick traceability columns that exist and contain data (for hover)."""
    candidates = ["pos_tag", "mold", "cavity", "cycle", "file"]
    return [
        c
        for c in candidates
        if c in sub.columns and sub[c].notna().any()
    ]


def build_fig(sub: pd.DataFrame, dim: str, height: int) -> go.Figure:
    """Build box-and-whisker plot figure.

    Points are drawn at category center (jitter=0) so out-of-spec overlay
    markers align exactly with the actual data points. Hover shows
    traceability info (mold/cavity/cycle) and a dashed mean line is shown.
    """
    fig = px.box(
        sub,
        x="group",
        y="value",
        color="group",
        points="all",
        hover_data=_hover_columns(sub),
    )
    fig.update_traces(
        jitter=0, pointpos=0, marker=dict(size=7, opacity=0.85), boxmean=True
    )

    # Show per-group sample size on x labels: small n means the box shape
    # is statistically weak — read the points, not the box.
    counts = sub.groupby("group")["value"].count()
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(counts.index),
        ticktext=[f"{g}<br>(n={c})" for g, c in counts.items()],
    )

    fig.update_layout(
        title=dim,
        xaxis_title=None,
        yaxis_title="量測值",
        showlegend=False,
        height=height,
        margin=dict(l=60, r=140, t=60, b=60),
        font=dict(family=CJK_FONT),
    )
    return fig


def add_spec_band(fig: go.Figure, lower: float, upper: float) -> None:
    """Shade the in-spec zone (LSL~USL) so off-center boxes read at a glance."""
    if pd.notna(lower) and pd.notna(upper) and upper > lower:
        fig.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor="rgba(46, 204, 113, 0.10)",
            line_width=0,
            layer="below",
        )


def apply_y_range(
    fig: go.Figure, sub: pd.DataFrame, lower: float, upper: float, focus_on_data: bool
) -> Tuple[float, float]:
    """Apply Y-axis range based on data and specification limits."""
    data_min = sub["value"].min()
    data_max = sub["value"].max()
    if focus_on_data:
        candidates = [v for v in [data_min, data_max] if pd.notna(v)]
    else:
        candidates = [v for v in [data_min, data_max, lower, upper] if pd.notna(v)]
    if not candidates:
        return np.nan, np.nan
    y_min = min(candidates)
    y_max = max(candidates)
    span = y_max - y_min
    pad = span * 0.15 if span > 0 else (abs(y_max) * 0.02 + 0.02)
    y_min -= pad
    y_max += pad
    fig.update_yaxes(range=[y_min, y_max])
    return y_min, y_max


def add_spec_edge_markers(
    fig: go.Figure, lower: float, upper: float, y_min: float, y_max: float
) -> None:
    """Add markers when spec lines are outside visible range."""
    if pd.notna(upper) and pd.notna(y_max) and upper > y_max:
        fig.add_annotation(
            x=1.02,
            xref="paper",
            y=y_max,
            yref="y",
            text=f"上限 {upper:.4f} (超出視窗)",
            showarrow=False,
            font=dict(family=CJK_FONT, color="red", size=12),
            xanchor="left",
        )
    if pd.notna(lower) and pd.notna(y_min) and lower < y_min:
        fig.add_annotation(
            x=1.02,
            xref="paper",
            y=y_min,
            yref="y",
            text=f"下限 {lower:.4f} (超出視窗)",
            showarrow=False,
            font=dict(family=CJK_FONT, color="red", size=12),
            xanchor="left",
        )


def add_out_of_spec_points(
    fig: go.Figure, sub: pd.DataFrame, lower: float, upper: float
) -> None:
    """Add red markers for out-of-spec points."""
    from .statistics import calc_out_of_spec

    mask = calc_out_of_spec(sub["value"], lower, upper)
    if not mask.any():
        return
    out = sub.loc[mask]

    hover_cols = _hover_columns(out)
    custom = out[hover_cols].astype(str).values if hover_cols else None
    hover_extra = "".join(
        f"<br>{col}: %{{customdata[{i}]}}" for i, col in enumerate(hover_cols)
    )

    fig.add_trace(
        go.Scatter(
            x=out["group"],
            y=out["value"],
            mode="markers",
            marker=dict(color="red", size=12, symbol="circle-open", line=dict(width=2)),
            customdata=custom,
            hovertemplate="超規格: %{y}" + hover_extra + "<extra></extra>",
            showlegend=False,
        )
    )


def build_cpk_heatmap(cpk_df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Build Cpk heatmap visualization.

    Args:
        cpk_df: DataFrame from cpk_with_rating() containing Cpk values and colors
        height: Chart height in pixels

    Returns:
        Plotly figure with Cpk heatmap
    """
    if cpk_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Cpk 熱力圖 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    # Create color mapping
    color_map = {"green": "#2ecc71", "yellow": "#f1c40f", "red": "#e74c3c", "gray": "#95a5a6"}

    # Map colors to numeric values for heatmap
    cpk_values = cpk_df["Cpk"].fillna(0).tolist()
    dimensions = cpk_df["dimension"].tolist()

    # Create bar chart with colors based on Cpk rating
    colors = [color_map.get(c, "#95a5a6") for c in cpk_df["color"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=dimensions,
                y=cpk_values,
                marker_color=colors,
                text=[f"{v:.2f}" if pd.notna(v) and v != 0 else "N/A" for v in cpk_df["Cpk"]],
                textposition="outside",
            )
        ]
    )

    # Add threshold lines
    fig.add_hline(y=1.33, line_color="green", line_width=2, line_dash="dash",
                  annotation_text="良好 (1.33)", annotation_position="right")
    fig.add_hline(y=1.0, line_color="orange", line_width=2, line_dash="dash",
                  annotation_text="可接受 (1.0)", annotation_position="right")

    fig.update_layout(
        title="Cpk 分析圖",
        xaxis_title="維度",
        yaxis_title="Cpk",
        height=height,
        margin=dict(l=60, r=100, t=60, b=100),
        font=dict(family=CJK_FONT),
        xaxis_tickangle=-45,
    )

    return fig


def build_normalized_deviation_chart(
    deviation_df: pd.DataFrame, dim: str, height: int = 400
) -> go.Figure:
    """Build normalized deviation chart for a specific dimension.

    Args:
        deviation_df: DataFrame from calculate_normalized_deviation()
        dim: Dimension name to plot
        height: Chart height in pixels

    Returns:
        Plotly figure with normalized deviation chart
    """
    sub = deviation_df[deviation_df["dimension"] == dim].copy()

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{dim} - 標準化偏離圖 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    # Determine x-axis based on available data
    if "group" in sub.columns and sub["group"].notna().any():
        x_col = "group"
    else:
        sub = sub.reset_index(drop=True)
        sub["index"] = range(1, len(sub) + 1)
        x_col = "index"

    # Color points based on deviation
    colors = []
    for dev in sub["deviation_pct"]:
        if abs(dev) > 100:
            colors.append("red")
        elif abs(dev) > 75:
            colors.append("orange")
        else:
            colors.append("blue")

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=sub[x_col],
            y=sub["deviation_pct"],
            mode="markers",
            marker=dict(color=colors, size=8),
            text=[f"{v:.1f}%" for v in sub["deviation_pct"]],
            hovertemplate="偏離: %{y:.1f}%<br>量測值: %{customdata:.4f}<extra></extra>",
            customdata=sub["value"],
        )
    )

    # Add reference lines
    fig.add_hline(y=0, line_color="green", line_width=2)
    fig.add_hline(y=100, line_color="red", line_width=1, line_dash="dash",
                  annotation_text="上限 +100%", annotation_position="right")
    fig.add_hline(y=-100, line_color="red", line_width=1, line_dash="dash",
                  annotation_text="下限 -100%", annotation_position="right")

    fig.update_layout(
        title=f"{dim} - 標準化偏離圖",
        xaxis_title="群組" if x_col == "group" else "量測點",
        yaxis_title="偏離 %",
        yaxis_range=[-150, 150],
        height=height,
        margin=dict(l=60, r=100, t=60, b=50),
        font=dict(family=CJK_FONT),
    )

    return fig


def build_imr_chart(
    spc_points: pd.DataFrame, dim: str, height: int = 500,
    nelson_points: pd.DataFrame = None,
) -> go.Figure:
    """Build I-MR (Individual-Moving Range) control chart.

    Args:
        spc_points: DataFrame from imr_spc_points() containing SPC data
        dim: Dimension name to plot
        height: Chart height in pixels

    Returns:
        Plotly figure with I-MR control chart (two subplots)
    """
    from plotly.subplots import make_subplots

    sub = spc_points[spc_points["dimension"] == dim].copy()

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{dim} - I-MR 控制圖 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    # Create subplots: I chart on top, MR chart on bottom
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"{dim} - I 圖 (個別值)", f"{dim} - MR 圖 (移動全距)"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # Get control limits (same for all points in this dimension)
    x_cl = sub["X_CL"].iloc[0]
    x_ucl = sub["X_UCL"].iloc[0]
    x_lcl = sub["X_LCL"].iloc[0]
    mr_cl = sub["MR_CL"].iloc[0]
    mr_ucl = sub["MR_UCL"].iloc[0]

    # --- I Chart (Individual Values) ---
    # Data points
    fig.add_trace(
        go.Scatter(
            x=sub["index"],
            y=sub["value"],
            mode="lines+markers",
            name="量測值",
            line=dict(color="blue", width=1),
            marker=dict(size=6),
        ),
        row=1, col=1
    )

    # Center line
    fig.add_hline(
        y=x_cl, line_color="green", line_width=2,
        annotation_text=f"CL={x_cl:.4f}", annotation_position="right",
        row=1, col=1
    )

    # UCL
    if pd.notna(x_ucl):
        fig.add_hline(
            y=x_ucl, line_color="red", line_width=1, line_dash="dash",
            annotation_text=f"UCL={x_ucl:.4f}", annotation_position="right",
            row=1, col=1
        )

    # LCL
    if pd.notna(x_lcl):
        fig.add_hline(
            y=x_lcl, line_color="red", line_width=1, line_dash="dash",
            annotation_text=f"LCL={x_lcl:.4f}", annotation_position="right",
            row=1, col=1
        )

    # Mark Nelson-rule violation points on I chart (orange diamonds)
    if nelson_points is not None and not nelson_points.empty:
        viol = nelson_points[nelson_points["violated"]]
        if not viol.empty:
            fig.add_trace(
                go.Scatter(
                    x=viol["index"],
                    y=viol["value"],
                    mode="markers",
                    name="Nelson 規則異常",
                    marker=dict(color="orange", size=11, symbol="diamond-open", line=dict(width=2)),
                    customdata=[", ".join(r) for r in viol["rules"]],
                    hovertemplate="第 %{x} 點: %{y}<br>違反規則: %{customdata}<extra></extra>",
                ),
                row=1, col=1
            )

    # Mark out-of-control points on I chart
    ooc_mask = (sub["value"] > x_ucl) | (sub["value"] < x_lcl)
    if ooc_mask.any():
        ooc = sub[ooc_mask]
        fig.add_trace(
            go.Scatter(
                x=ooc["index"],
                y=ooc["value"],
                mode="markers",
                name="失控點",
                marker=dict(color="red", size=10, symbol="circle-open", line=dict(width=2)),
            ),
            row=1, col=1
        )

    # --- MR Chart (Moving Range) ---
    mr_data = sub[sub["MR"].notna()]

    # Data points
    fig.add_trace(
        go.Scatter(
            x=mr_data["index"],
            y=mr_data["MR"],
            mode="lines+markers",
            name="移動全距",
            line=dict(color="purple", width=1),
            marker=dict(size=6),
        ),
        row=2, col=1
    )

    # Center line
    if pd.notna(mr_cl):
        fig.add_hline(
            y=mr_cl, line_color="green", line_width=2,
            annotation_text=f"CL={mr_cl:.4f}", annotation_position="right",
            row=2, col=1
        )

    # UCL
    if pd.notna(mr_ucl):
        fig.add_hline(
            y=mr_ucl, line_color="red", line_width=1, line_dash="dash",
            annotation_text=f"UCL={mr_ucl:.4f}", annotation_position="right",
            row=2, col=1
        )

    # LCL (usually 0 for MR chart)
    fig.add_hline(
        y=0, line_color="red", line_width=1, line_dash="dash",
        row=2, col=1
    )

    # Mark out-of-control points on MR chart
    if pd.notna(mr_ucl):
        mr_ooc_mask = mr_data["MR"] > mr_ucl
        if mr_ooc_mask.any():
            mr_ooc = mr_data[mr_ooc_mask]
            fig.add_trace(
                go.Scatter(
                    x=mr_ooc["index"],
                    y=mr_ooc["MR"],
                    mode="markers",
                    name="MR失控點",
                    marker=dict(color="red", size=10, symbol="circle-open", line=dict(width=2)),
                ),
                row=2, col=1
            )

    fig.update_layout(
        height=height,
        margin=dict(l=60, r=120, t=80, b=50),
        font=dict(family=CJK_FONT),
        showlegend=False,
    )

    fig.update_xaxes(title_text="量測點序號", row=2, col=1)
    fig.update_yaxes(title_text="量測值", row=1, col=1)
    fig.update_yaxes(title_text="移動全距", row=2, col=1)

    return fig


def build_position_comparison_chart(
    df: pd.DataFrame, dim: str, height: int = 400
) -> go.Figure:
    """Build position comparison chart (P1, P2, P3...) for multi-mold data.

    Args:
        df: Main data DataFrame with pos_in_mold column
        dim: Dimension name to plot
        height: Chart height in pixels

    Returns:
        Plotly figure with position comparison chart
    """
    from .statistics import pick_spec_values

    sub = df[df["dimension"] == dim].copy()

    if sub.empty or "pos_in_mold" not in sub.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"{dim} - 模次比較圖 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    # Format position labels
    sub["position"] = sub["pos_in_mold"].apply(
        lambda x: f"P{int(x)}" if pd.notna(x) else "P?"
    )

    # Get specification values
    nominal, upper, lower, _ = pick_spec_values(sub)

    # Create box plot by position
    fig = px.box(
        sub,
        x="position",
        y="value",
        color="position",
        points="all",
        hover_data=_hover_columns(sub),
    )
    fig.update_traces(
        jitter=0, pointpos=0, marker=dict(size=7, opacity=0.85), boxmean=True
    )
    add_spec_band(fig, lower, upper)

    # Add specification lines
    if pd.notna(upper):
        fig.add_hline(y=upper, line_color="red", line_width=2,
                      annotation_text=f"上限 {upper:.4f}", annotation_position="right")
    if pd.notna(nominal):
        fig.add_hline(y=nominal, line_color="red", line_width=1, line_dash="dot",
                      annotation_text=f"中值 {nominal:.4f}", annotation_position="right")
    if pd.notna(lower):
        fig.add_hline(y=lower, line_color="red", line_width=2,
                      annotation_text=f"下限 {lower:.4f}", annotation_position="right")

    fig.update_layout(
        title=f"{dim} - 模次比較圖",
        xaxis_title="模次位置",
        yaxis_title="量測值",
        showlegend=False,
        height=height,
        margin=dict(l=60, r=140, t=60, b=50),
        font=dict(family=CJK_FONT),
    )

    return fig


def build_correlation_heatmap(
    corr_matrix: pd.DataFrame, height: int = 600
) -> go.Figure:
    """Build correlation heatmap visualization.

    Args:
        corr_matrix: Correlation matrix from calculate_correlation_matrix()
        height: Chart height in pixels

    Returns:
        Plotly figure with correlation heatmap
    """
    if corr_matrix.empty:
        fig = go.Figure()
        fig.update_layout(
            title="相關性矩陣 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[
                [0.0, "#2166ac"],    # Strong negative - blue
                [0.25, "#67a9cf"],   # Weak negative - light blue
                [0.5, "#f7f7f7"],    # No correlation - white
                [0.75, "#ef8a62"],   # Weak positive - light red
                [1.0, "#b2182b"],    # Strong positive - red
            ],
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="維度1: %{y}<br>維度2: %{x}<br>相關係數: %{z:.3f}<extra></extra>",
            colorbar=dict(
                title="相關係數",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0", "0.5", "1.0"],
            ),
        )
    )

    fig.update_layout(
        title="維度相關性矩陣",
        xaxis_title="",
        yaxis_title="",
        height=height,
        margin=dict(l=100, r=50, t=60, b=100),
        font=dict(family=CJK_FONT),
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    return fig


def build_correlation_scatter(
    pivot_table: pd.DataFrame,
    dim1: str,
    dim2: str,
    height: int = 400
) -> go.Figure:
    """Build scatter plot for two dimensions.

    Args:
        pivot_table: Wide-format data from calculate_correlation_matrix()
        dim1: First dimension name
        dim2: Second dimension name
        height: Chart height in pixels

    Returns:
        Plotly figure with scatter plot and trendline
    """
    if dim1 not in pivot_table.columns or dim2 not in pivot_table.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"{dim1} vs {dim2} (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    # Get paired data
    data = pivot_table[[dim1, dim2]].dropna()

    if data.empty or len(data) < 3:
        fig = go.Figure()
        fig.update_layout(
            title=f"{dim1} vs {dim2} (資料不足)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    x = data[dim1]
    y = data[dim2]

    # Calculate correlation
    corr = x.corr(y)

    # Calculate trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = p(x_line)

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=8, color="blue", opacity=0.7),
            name="量測點",
            hovertemplate=f"{dim1}: %{{x:.4f}}<br>{dim2}: %{{y:.4f}}<extra></extra>",
        )
    )

    # Add trendline
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=f"趨勢線 (r={corr:.3f})",
        )
    )

    fig.update_layout(
        title=f"{dim1} vs {dim2} (r = {corr:.3f})",
        xaxis_title=dim1,
        yaxis_title=dim2,
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
        font=dict(family=CJK_FONT),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def build_histogram(
    sub: pd.DataFrame,
    dim: str,
    nominal: float,
    upper: float,
    lower: float,
    height: int = 400,
) -> go.Figure:
    """Build histogram with normal-curve overlay and spec lines.

    搭配常態性檢定使用：直方圖能看出偏態/雙峰等 Cpk 假設失效的情況。
    """
    values = sub["value"].astype(float).dropna()

    fig = go.Figure()
    if values.empty:
        fig.update_layout(
            title=f"{dim} - 分布直方圖 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    fig.add_trace(
        go.Histogram(
            x=values,
            histnorm="probability density",
            marker_color="#3498db",
            opacity=0.75,
            name="量測值",
        )
    )

    mean = values.mean()
    std = values.std(ddof=1)
    if len(values) >= 3 and pd.notna(std) and std > 0:
        lo_candidates = [values.min()] + ([lower] if pd.notna(lower) else [])
        hi_candidates = [values.max()] + ([upper] if pd.notna(upper) else [])
        lo, hi = min(lo_candidates), max(hi_candidates)
        span = hi - lo if hi > lo else abs(hi) * 0.1 + 0.1
        xs = np.linspace(lo - span * 0.1, hi + span * 0.1, 200)
        pdf = np.exp(-((xs - mean) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pdf,
                mode="lines",
                line=dict(color="#e67e22", width=2),
                name="常態曲線",
            )
        )

    if pd.notna(upper):
        fig.add_vline(x=upper, line_color="red", line_width=2,
                      annotation_text=f"上限 {upper:.4f}", annotation_position="top")
    if pd.notna(nominal):
        fig.add_vline(x=nominal, line_color="red", line_width=1, line_dash="dot",
                      annotation_text=f"中值 {nominal:.4f}", annotation_position="top")
    if pd.notna(lower):
        fig.add_vline(x=lower, line_color="red", line_width=2,
                      annotation_text=f"下限 {lower:.4f}", annotation_position="top")

    fig.update_layout(
        title=f"{dim} - 分布直方圖",
        xaxis_title="量測值",
        yaxis_title="機率密度",
        height=height,
        margin=dict(l=60, r=60, t=80, b=50),
        font=dict(family=CJK_FONT),
        showlegend=False,
        bargap=0.05,
    )
    return fig


def build_cpk_trend(trend_df: pd.DataFrame, height: int = 450) -> go.Figure:
    """Build cross-file Cpk trend chart (one line per dimension).

    Args:
        trend_df: DataFrame with columns: file, dimension, Cpk.
            File order is preserved as given (e.g. upload order = time order).
    """
    if trend_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="跨檔案 Cpk 趨勢 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    file_order = list(dict.fromkeys(trend_df["file"].tolist()))

    fig = px.line(
        trend_df,
        x="file",
        y="Cpk",
        color="dimension",
        markers=True,
    )
    fig.add_hline(y=1.33, line_color="green", line_width=1.5, line_dash="dash",
                  annotation_text="良好 (1.33)", annotation_position="right")
    fig.add_hline(y=1.0, line_color="orange", line_width=1.5, line_dash="dash",
                  annotation_text="可接受 (1.0)", annotation_position="right")

    fig.update_xaxes(categoryorder="array", categoryarray=file_order)
    fig.update_layout(
        title="跨檔案 Cpk 趨勢（依上傳順序）",
        xaxis_title="檔案",
        yaxis_title="Cpk",
        height=height,
        margin=dict(l=60, r=100, t=60, b=80),
        font=dict(family=CJK_FONT),
        legend_title_text="維度",
    )
    return fig


def build_cavity_fingerprint(
    fp_df: pd.DataFrame, group_label: str = "穴號", height: int = 480
) -> go.Figure:
    """Build cavity fingerprint chart: one line per cavity across all dimensions.

    每條線代表一個穴（或位置/分組），Y 軸為標準化偏離%。線整體偏高/偏低
    即代表該穴一致地偏大/偏小 → 修模方向的直接線索。
    """
    if fp_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="穴號指紋圖 (無資料)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        return fig

    dims_order = sorted(fp_df["dimension"].unique().tolist())
    fp_df = fp_df.copy()
    fp_df["group_str"] = fp_df["group_val"].apply(
        lambda x: f"{group_label}{int(x)}" if isinstance(x, (int, float)) and float(x).is_integer() else f"{group_label}{x}"
    )

    fig = px.line(
        fp_df,
        x="dimension",
        y="deviation_pct",
        color="group_str",
        markers=True,
        category_orders={"dimension": dims_order},
    )

    # Spec edges (±100%) and center (0%) reference lines
    fig.add_hline(y=0, line_color="green", line_width=1.5)
    fig.add_hline(y=100, line_color="red", line_width=1, line_dash="dash",
                  annotation_text="上限 +100%", annotation_position="right")
    fig.add_hline(y=-100, line_color="red", line_width=1, line_dash="dash",
                  annotation_text="下限 -100%", annotation_position="right")

    fig.update_layout(
        title=f"{group_label}指紋圖（各{group_label}在各維度的標準化偏離）",
        xaxis_title="維度",
        yaxis_title="標準化偏離 %",
        height=height,
        margin=dict(l=60, r=110, t=60, b=100),
        font=dict(family=CJK_FONT),
        legend_title_text=group_label,
        xaxis_tickangle=-45,
    )
    return fig


def build_pareto_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str = "Pareto 排列圖",
    height: int = 460,
) -> go.Figure:
    """Build a Pareto chart: sorted bars + cumulative percentage line.

    依數量由大到小排序長條，疊上累積百分比折線與 80% 參考線，
    協助依「先解決影響最大的少數」原則決定處理優先序。
    """
    from plotly.subplots import make_subplots

    data = df[[label_col, value_col]].copy()
    data = data[data[value_col] > 0]
    data = data.sort_values(value_col, ascending=False).reset_index(drop=True)

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} (無異常項目)",
            height=height,
            font=dict(family=CJK_FONT),
        )
        fig.add_annotation(
            text="沒有需要排序的異常項目 🎉",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(family=CJK_FONT, size=16),
        )
        return fig

    total = data[value_col].sum()
    data["cum_pct"] = data[value_col].cumsum() / total * 100
    labels = data[label_col].astype(str).tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=labels,
            y=data[value_col],
            marker_color="#e67e22",
            name="數量",
            text=data[value_col],
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=data["cum_pct"],
            mode="lines+markers",
            line=dict(color="#2c3e50", width=2),
            name="累積 %",
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=80, line_color="red", line_width=1, line_dash="dash",
        annotation_text="80%", annotation_position="right",
        secondary_y=True,
    )

    fig.update_yaxes(title_text="數量", secondary_y=False)
    fig.update_yaxes(title_text="累積 %", range=[0, 105], secondary_y=True)
    fig.update_layout(
        title=title,
        xaxis_title="",
        height=height,
        margin=dict(l=60, r=80, t=60, b=110),
        font=dict(family=CJK_FONT),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
