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


def build_fig(sub: pd.DataFrame, dim: str, height: int) -> go.Figure:
    """Build box-and-whisker plot figure."""
    fig = px.box(
        sub,
        x="group",
        y="value",
        color="group",
        points="all",
    )
    fig.update_traces(jitter=0.2, pointpos=0, marker=dict(size=7, opacity=0.85))
    fig.update_layout(
        title=dim,
        xaxis_title=None,
        yaxis_title="量測值",
        showlegend=False,
        height=height,
        margin=dict(l=60, r=140, t=60, b=50),
        font=dict(family=CJK_FONT),
    )
    return fig


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
    fig.add_trace(
        go.Scatter(
            x=out["group"],
            y=out["value"],
            mode="markers",
            marker=dict(color="red", size=7, symbol="circle-open"),
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
    spc_points: pd.DataFrame, dim: str, height: int = 500
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
    )
    fig.update_traces(jitter=0.2, pointpos=0, marker=dict(size=7, opacity=0.85))

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
