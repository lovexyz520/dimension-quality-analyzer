import base64
import io
import os
import zipfile
from typing import Optional, Tuple

# Kaleido 在 Cloud 環境需要關閉 sandbox 模式
os.environ["KALEIDO_DISABLE_SANDBOX"] = "1"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Box-and-Whisker Plot", layout="wide")


def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    for i in range(len(df)):
        row = df.iloc[i].astype(str)
        if row.str.contains("規格", na=False).any() and row.str.contains("球標", na=False).any():
            return i
    return None


def _find_col_index(header_row: pd.Series, label: str) -> Optional[int]:
    for idx, val in header_row.items():
        if str(val).strip() == label:
            return idx
    return None


def _clean_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _detect_focus_sheet(xl: pd.ExcelFile) -> Tuple[Optional[str], Optional[str]]:
    for name in xl.sheet_names:
        if "重點尺寸" in str(name):
            return name, None

    for name in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=name, header=None, engine="openpyxl")
        except Exception:
            continue
        if _find_header_row(df) is not None:
            return name, None

    return None, "找不到包含 '規格' 與 '球標' 的工作表"


def parse_focus_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    header_row = _find_header_row(df)
    if header_row is None:
        raise ValueError("找不到包含 '規格' 與 '球標' 的標題列")

    header = df.iloc[header_row]

    col_spec = _find_col_index(header, "規格")
    col_plus = _find_col_index(header, "正公差")
    col_minus = _find_col_index(header, "負公差")
    col_label = _find_col_index(header, "球標")
    col_method = _find_col_index(header, "量測方式")

    if col_label is None:
        raise ValueError("找不到 '球標' 欄位")

    if col_method is None:
        col_method = df.shape[1]

    meas_cols = [i for i in range(col_label + 1, col_method) if i < df.shape[1]]
    if not meas_cols:
        raise ValueError("找不到量測數值欄位")

    data = df.iloc[header_row + 2 :].copy()

    meas = data.iloc[:, meas_cols].apply(pd.to_numeric, errors="coerce")
    spec_num = (
        pd.to_numeric(data.iloc[:, col_spec], errors="coerce")
        if col_spec is not None
        else pd.Series([np.nan] * len(data), index=data.index)
    )
    plus_num = (
        pd.to_numeric(data.iloc[:, col_plus], errors="coerce")
        if col_plus is not None
        else pd.Series([np.nan] * len(data), index=data.index)
    )
    minus_num = (
        pd.to_numeric(data.iloc[:, col_minus], errors="coerce")
        if col_minus is not None
        else pd.Series([np.nan] * len(data), index=data.index)
    )
    label_series = data.iloc[:, col_label]
    col0 = data.iloc[:, 0] if df.shape[1] > 0 else pd.Series([None] * len(data), index=data.index)

    keep = meas.notna().any(axis=1) | spec_num.notna() | label_series.notna() | col0.notna()
    data = data.loc[keep]
    meas = meas.loc[data.index]
    label_series = label_series.loc[data.index]
    col0 = col0.loc[data.index]
    spec_num = spec_num.loc[data.index]
    plus_num = plus_num.loc[data.index]
    minus_num = minus_num.loc[data.index]

    rows = []
    for idx in data.index:
        base = _clean_cell(label_series.loc[idx])
        prefix = _clean_cell(col0.loc[idx])
        if prefix and base:
            dim_label = f"{prefix} {base}"
        elif base:
            dim_label = base
        elif prefix:
            dim_label = prefix
        else:
            dim_label = ""

        if not dim_label:
            continue

        nominal = spec_num.loc[idx]
        upper = nominal + plus_num.loc[idx] if pd.notna(nominal) and pd.notna(plus_num.loc[idx]) else np.nan
        lower = nominal + minus_num.loc[idx] if pd.notna(nominal) and pd.notna(minus_num.loc[idx]) else np.nan

        values = pd.to_numeric(meas.loc[idx], errors="coerce").dropna().tolist()
        for v in values:
            rows.append(
                {
                    "dimension": dim_label,
                    "value": float(v),
                    "nominal": float(nominal) if pd.notna(nominal) else np.nan,
                    "upper": float(upper) if pd.notna(upper) else np.nan,
                    "lower": float(lower) if pd.notna(lower) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def load_excel(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        xl = pd.ExcelFile(uploaded_file)
        sheet_name, err = _detect_focus_sheet(xl)
        if err:
            return pd.DataFrame(), err
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None, engine="openpyxl")
        out = parse_focus_dimensions(df)
        if out.empty:
            return out, "沒有解析到任何量測數值"
        return out, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def _calc_out_of_spec(values: pd.Series, lower: float, upper: float) -> pd.Series:
    if pd.isna(lower) or pd.isna(upper):
        return pd.Series([False] * len(values), index=values.index)
    return (values < lower) | (values > upper)


def _pick_spec_values(sub: pd.DataFrame) -> Tuple[float, float, float, int]:
    nominal_list = sub["nominal"].dropna().unique().tolist()
    upper_list = sub["upper"].dropna().unique().tolist()
    lower_list = sub["lower"].dropna().unique().tolist()

    nominal = nominal_list[0] if nominal_list else np.nan
    upper = upper_list[0] if upper_list else np.nan
    lower = lower_list[0] if lower_list else np.nan

    spec_versions = max(len(nominal_list), len(upper_list), len(lower_list))
    return float(nominal) if pd.notna(nominal) else np.nan, float(upper) if pd.notna(upper) else np.nan, float(lower) if pd.notna(lower) else np.nan, spec_versions


def _add_spec_lines(fig, nominal: float, upper: float, lower: float) -> None:
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
            font=dict(color="red", size=12),
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
            font=dict(color="red", size=12),
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
            font=dict(color="red", size=12),
            xanchor="left",
        )


def _build_fig(sub: pd.DataFrame, dim: str, height: int) -> go.Figure:
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
    )
    return fig


def _apply_y_range(
    fig, sub: pd.DataFrame, lower: float, upper: float, focus_on_data: bool
) -> Tuple[float, float]:
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


def _add_spec_edge_markers(
    fig, lower: float, upper: float, y_min: float, y_max: float
) -> None:
    if pd.notna(upper) and pd.notna(y_max) and upper > y_max:
        fig.add_annotation(
            x=1.02,
            xref="paper",
            y=y_max,
            yref="y",
            text=f"上限 {upper:.4f} (超出視窗)",
            showarrow=False,
            font=dict(color="red", size=12),
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
            font=dict(color="red", size=12),
            xanchor="left",
        )


def _add_out_of_spec_points(fig, sub: pd.DataFrame, lower: float, upper: float) -> None:
    mask = _calc_out_of_spec(sub["value"], lower, upper)
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


def _download_plot_button(fig, filename: str) -> None:
    try:
        img_bytes = fig.to_image(format="png", scale=3)
    except Exception:
        img_bytes = None

    if img_bytes:
        st.download_button(
            "下載圖表 (PNG)",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    else:
        st.caption("PNG 下載需要 kaleido，請確認已安裝")


def _download_excel_button(df: pd.DataFrame, filename: str) -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button(
        "下載資料 (Excel)",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _stats_table(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(group):
        values = group["value"].astype(float)
        nominal, upper, lower, spec_versions = _pick_spec_values(group)
        out_mask = _calc_out_of_spec(values, lower, upper)
        return pd.Series(
            {
                "count": values.count(),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
                "nominal": nominal,
                "upper": upper,
                "lower": lower,
                "out_of_spec": int(out_mask.sum()),
                "spec_versions": spec_versions,
            }
        )

    return df.groupby("dimension", as_index=False).apply(_agg).reset_index(drop=True)


def _download_stats_excel(stats: pd.DataFrame, filename: str) -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        stats.to_excel(writer, index=False, sheet_name="summary")
    st.download_button(
        "下載統計摘要 (Excel)",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _cp_cpk_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, group in df.groupby("dimension"):
        values = group["value"].astype(float)
        nominal, upper, lower, _ = _pick_spec_values(group)
        mean = values.mean()
        std = values.std(ddof=1)
        if pd.notna(upper) and pd.notna(lower) and std and std > 0:
            cp = (upper - lower) / (6 * std)
            cpu = (upper - mean) / (3 * std)
            cpl = (mean - lower) / (3 * std)
            cpk = min(cpu, cpl)
        else:
            cp = np.nan
            cpk = np.nan
        rows.append(
            {
                "dimension": dim,
                "count": values.count(),
                "mean": mean,
                "std": std,
                "USL": upper,
                "LSL": lower,
                "Cp": cp,
                "Cpk": cpk,
            }
        )
    return pd.DataFrame(rows)


def _imr_spc_points(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    d2 = 1.128
    d3 = 0
    d4 = 3.267

    for dim, group in df.groupby("dimension"):
        values = group["value"].astype(float).reset_index(drop=True)
        if values.empty:
            continue
        mr = values.diff().abs()
        mr_bar = mr[1:].mean()
        sigma = mr_bar / d2 if mr_bar and mr_bar > 0 else np.nan
        x_bar = values.mean()

        x_ucl = x_bar + 3 * sigma if pd.notna(sigma) else np.nan
        x_lcl = x_bar - 3 * sigma if pd.notna(sigma) else np.nan
        mr_cl = mr_bar
        mr_ucl = d4 * mr_bar if pd.notna(mr_bar) else np.nan
        mr_lcl = d3 * mr_bar if pd.notna(mr_bar) else np.nan

        for i, v in enumerate(values, start=1):
            rows.append(
                {
                    "dimension": dim,
                    "index": i,
                    "value": v,
                    "MR": mr.iloc[i - 1] if i > 1 else np.nan,
                    "X_CL": x_bar,
                    "X_UCL": x_ucl,
                    "X_LCL": x_lcl,
                    "MR_CL": mr_cl,
                    "MR_UCL": mr_ucl,
                    "MR_LCL": mr_lcl,
                }
            )

        summary_rows.append(
            {
                "dimension": dim,
                "count": values.count(),
                "X_bar": x_bar,
                "MR_bar": mr_bar,
                "sigma_est": sigma,
                "X_UCL": x_ucl,
                "X_LCL": x_lcl,
                "MR_UCL": mr_ucl,
                "MR_LCL": mr_lcl,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(rows)


def _download_quality_reports(df: pd.DataFrame) -> None:
    cp_cpk = _cp_cpk_summary(df)
    spc_summary, spc_points = _imr_spc_points(df)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        cp_cpk.to_excel(writer, index=False, sheet_name="cp_cpk_summary")
        spc_summary.to_excel(writer, index=False, sheet_name="spc_imr_summary")
        spc_points.to_excel(writer, index=False, sheet_name="spc_imr_points")
    st.download_button(
        "下載 CP/CPK + SPC 報表 (Excel)",
        data=buffer.getvalue(),
        file_name="quality_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _build_report_html(stats: pd.DataFrame, figures: list) -> str:
    table_html = stats.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    images_html = "\n".join(
        [
            f"<h3>{title}</h3><img style='max-width:100%;' src='data:image/png;base64,{img_b64}'/>"
            for title, img_b64 in figures
        ]
    )
    return f"""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>Boxplot Report</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 20px; }}
img {{ margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>盒鬚圖報表</h1>
<h2>統計摘要</h2>
{table_html}
<h2>圖表</h2>
{images_html}
</body>
</html>
"""


st.title("盒鬚圖分析工具")
st.caption("上傳單一或多個 Excel，支援合併或分檔比較，每個維度一張圖。")

uploaded_files = st.file_uploader(
    "上傳 Excel 檔案 (xlsm/xlsx)",
    type=["xlsm", "xlsx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("請先上傳 Excel 檔案")
    st.stop()

combine_files = st.checkbox("合併多檔成單一分佈", value=True)
chart_height = st.slider("圖表高度", min_value=360, max_value=900, value=520, step=20)
focus_on_data = st.checkbox("放大顯示盒鬚圖（以數據為主）", value=True)
dual_view = st.checkbox("雙圖模式（完整規格 + 放大視圖）", value=False)
auto_range = st.checkbox("依規格/數據自動縮放", value=True)

all_frames = []
errors = []
for file in uploaded_files:
    df, err = load_excel(file)
    if err:
        errors.append(f"{file.name}: {err}")
        continue
    df["file"] = file.name
    all_frames.append(df)

if errors:
    st.warning("以下檔案解析失敗或無資料：")
    for msg in errors:
        st.write("-", msg)

if not all_frames:
    st.error("沒有可用資料")
    st.stop()

raw = pd.concat(all_frames, ignore_index=True)

if combine_files:
    raw["group"] = "合併"
else:
    raw["group"] = raw["file"]

all_dimensions = sorted(raw["dimension"].dropna().unique().tolist())

search_text = st.text_input("維度搜尋", value="", placeholder="例如：1-A, 2-B, 2/A1")
if search_text:
    tokens = [t.strip().lower() for t in search_text.replace(";", ",").split(",") if t.strip()]
    if tokens:
        all_dimensions = [d for d in all_dimensions if any(t in d.lower() for t in tokens)]

selected_dimensions = st.multiselect(
    "選擇要顯示的維度 (預設全選)",
    options=all_dimensions,
    default=all_dimensions,
)

max_charts = st.number_input(
    "最多顯示幾張圖 (避免一次渲染太多)",
    min_value=1,
    max_value=max(1, len(all_dimensions)),
    value=min(20, max(1, len(all_dimensions))),
    step=1,
)

stats = _stats_table(raw)

st.download_button(
    "一鍵匯出整理後的長表 CSV",
    data=raw.to_csv(index=False).encode("utf-8-sig"),
    file_name="boxplot_long_table.csv",
    mime="text/csv",
)

_download_stats_excel(stats, "boxplot_summary.xlsx")
_download_quality_reports(raw)

shown = 0
fig_cache = []
for dim in selected_dimensions:
    if shown >= max_charts:
        break
    sub = raw[raw["dimension"] == dim].copy()
    if sub.empty:
        continue

    nominal, upper, lower, spec_versions = _pick_spec_values(sub)

    if dual_view:
        st.markdown("**完整規格視圖**")
        fig_full = _build_fig(sub, dim, chart_height)
        _add_spec_lines(fig_full, nominal, upper, lower)
        _add_out_of_spec_points(fig_full, sub, lower, upper)
        if auto_range:
            _apply_y_range(fig_full, sub, lower, upper, False)
        st.plotly_chart(fig_full, use_container_width=True)

        st.markdown("**放大視圖**")
        fig_zoom = _build_fig(sub, dim, chart_height)
        _add_spec_lines(fig_zoom, nominal, upper, lower)
        _add_out_of_spec_points(fig_zoom, sub, lower, upper)
        if auto_range:
            y_min, y_max = _apply_y_range(fig_zoom, sub, lower, upper, True)
            _add_spec_edge_markers(fig_zoom, lower, upper, y_min, y_max)
        st.plotly_chart(fig_zoom, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _download_plot_button(fig_full, f"{dim}_full.png")
        with c2:
            _download_plot_button(fig_zoom, f"{dim}_zoom.png")
        with c3:
            _download_excel_button(sub, f"{dim}.xlsx")
        with c4:
            st.caption("")

        try:
            img_bytes = fig_full.to_image(format="png", scale=2)
            fig_cache.append((f"{dim} (full)", base64.b64encode(img_bytes).decode("ascii"), img_bytes))
        except Exception:
            pass
        try:
            img_bytes = fig_zoom.to_image(format="png", scale=2)
            fig_cache.append((f"{dim} (zoom)", base64.b64encode(img_bytes).decode("ascii"), img_bytes))
        except Exception:
            pass
    else:
        fig = _build_fig(sub, dim, chart_height)
        _add_spec_lines(fig, nominal, upper, lower)
        _add_out_of_spec_points(fig, sub, lower, upper)
        if auto_range:
            y_min, y_max = _apply_y_range(fig, sub, lower, upper, focus_on_data)
            if focus_on_data:
                _add_spec_edge_markers(fig, lower, upper, y_min, y_max)

        st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            _download_plot_button(fig, f"{dim}.png")
        with col_right:
            _download_excel_button(sub, f"{dim}.xlsx")

        try:
            img_bytes = fig.to_image(format="png", scale=2)
            fig_cache.append((dim, base64.b64encode(img_bytes).decode("ascii"), img_bytes))
        except Exception:
            pass

    if spec_versions > 1:
        st.caption("注意：此維度在不同檔案中規格不一致，已取第一筆規格作為標示。")

    shown += 1

if shown == 0:
    st.info("目前選擇的維度沒有可用資料")

if fig_cache:
    if st.button("產生圖表 ZIP"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for title, _b64, img_bytes in fig_cache:
                zf.writestr(f"{title}.png", img_bytes)
            zf.writestr("summary.csv", stats.to_csv(index=False))
            zf.writestr("long_table.csv", raw.to_csv(index=False))
        st.download_button(
            "下載全部圖表 ZIP",
            data=zip_buffer.getvalue(),
            file_name="boxplot_charts.zip",
            mime="application/zip",
        )

    if st.button("產生報表 (HTML)"):
        figures = [(title, b64) for title, b64, _bytes in fig_cache]
        report_html = _build_report_html(stats, figures)
        st.download_button(
            "下載報表 (HTML)",
            data=report_html.encode("utf-8"),
            file_name="boxplot_report.html",
            mime="text/html",
        )
