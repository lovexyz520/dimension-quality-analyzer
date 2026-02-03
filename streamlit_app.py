"""Dimension Quality Analyzer - Streamlit Web Application."""

import base64
import io
import os
import zipfile

# Kaleido needs sandbox disabled on Cloud environment
os.environ["KALEIDO_DISABLE_SANDBOX"] = "1"

import pandas as pd
import plotly.io as pio
import streamlit as st

# Fix kaleido /dev/shm access issue on Streamlit Cloud
try:
    pio.kaleido.scope.chromium_args = tuple(
        [arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"]
    )
except Exception:
    pass

from core import (
    load_excel,
    pick_spec_values,
    stats_table,
    cpk_with_rating,
    imr_spc_points,
    calculate_normalized_deviation,
    add_spec_lines,
    build_fig,
    apply_y_range,
    add_spec_edge_markers,
    add_out_of_spec_points,
    build_cpk_heatmap,
    build_normalized_deviation_chart,
    build_position_comparison_chart,
    build_imr_chart,
    download_plot_button,
    download_excel_button,
    download_stats_excel,
    download_quality_reports,
    build_report_html,
    download_pdf_report_button,
    assign_groups_vectorized,
)


st.set_page_config(page_title="Box-and-Whisker Plot", layout="wide")

st.title("盒鬚圖分析工具")
st.caption("上傳單一或多個 Excel，支援自動分組或全部合併，每個維度一張圖。")

uploaded_files = st.file_uploader(
    "上傳 Excel 檔案 (xlsm/xlsx)",
    type=["xlsm", "xlsx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("請先上傳 Excel 檔案")
    st.stop()

# Chart settings
chart_height = st.slider("圖表高度", min_value=360, max_value=900, value=520, step=20)
focus_on_data = st.checkbox("放大顯示盒鬚圖（以數據為主）", value=True)
dual_view = st.checkbox("雙圖模式（完整規格 + 放大視圖）", value=False)
auto_range = st.checkbox("依規格/數據自動縮放", value=True)

# Load and parse files
all_frames = []
errors = []

load_progress = st.progress(0, text="正在載入檔案...")
for i, file in enumerate(uploaded_files):
    df, err = load_excel(file)
    if err:
        errors.append(f"{file.name}: {err}")
        continue
    df["file"] = file.name
    all_frames.append(df)
    load_progress.progress((i + 1) / len(uploaded_files), text=f"正在載入檔案... ({i + 1}/{len(uploaded_files)})")
load_progress.empty()

if errors:
    st.warning("以下檔案解析失敗或無資料：")
    for msg in errors:
        st.write("-", msg)

if not all_frames:
    st.error("沒有可用資料")
    st.stop()

raw = pd.concat(all_frames, ignore_index=True)

# Display mode selection
mode = st.radio(
    "顯示模式",
    options=["自動分組", "強制分檔顯示", "全部合併成一張圖"],
    index=0,
    horizontal=True,
)

# Calculate file mold counts
file_list = sorted(raw["file"].dropna().unique().tolist())
file_mold_counts = {}
for fname in file_list:
    sub = raw[raw["file"] == fname]
    molds = [m for m in sub.get("mold", pd.Series(dtype=str)).dropna().unique() if str(m).strip()]
    file_mold_counts[fname] = len(molds)

# Assign groups using vectorized function
raw["group"] = assign_groups_vectorized(raw, mode, file_mold_counts, file_list)

# Dimension selection
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

# Calculate statistics
stats = stats_table(raw)

# Download buttons
col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
with col_dl1:
    st.download_button(
        "匯出長表 CSV",
        data=raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="boxplot_long_table.csv",
        mime="text/csv",
    )
with col_dl2:
    download_stats_excel(stats, "boxplot_summary.xlsx")
with col_dl3:
    download_quality_reports(raw)
with col_dl4:
    # Generate PDF report with all charts
    if st.button("產生完整 PDF 報表", key="gen_pdf"):
        with st.spinner("正在產生 PDF 報表..."):
            cpk_df = cpk_with_rating(raw)
            pdf_figures = []

            # Generate charts for PDF
            pdf_progress = st.progress(0, text="正在產生圖表...")
            dims_to_include = selected_dimensions[:min(len(selected_dimensions), max_charts)]

            for i, dim in enumerate(dims_to_include):
                sub = raw[raw["dimension"] == dim].copy()
                if sub.empty:
                    continue
                nominal, upper, lower, _ = pick_spec_values(sub)
                fig = build_fig(sub, dim, 400)
                add_spec_lines(fig, nominal, upper, lower)
                add_out_of_spec_points(fig, sub, lower, upper)
                try:
                    img_bytes = fig.to_image(format="png", scale=2)
                    pdf_figures.append((dim, img_bytes))
                except Exception:
                    pass
                pdf_progress.progress((i + 1) / len(dims_to_include))

            pdf_progress.empty()

            if pdf_figures:
                download_pdf_report_button(stats, cpk_df, pdf_figures, "quality_report.pdf")
            else:
                st.warning("無法產生圖表，請確認已安裝 kaleido")

# Create tabbed interface
tab1, tab2, tab3, tab4, tab5 = st.tabs(["盒鬚圖", "Cpk 分析", "SPC 控制圖", "標準化偏離", "模次比較"])

# Tab 1: Box-and-Whisker Plots (original functionality)
with tab1:
    st.subheader("盒鬚圖")

    total_charts = min(len(selected_dimensions), max_charts)
    if total_charts > 0:
        progress = st.progress(0, text="正在生成圖表...")
        shown = 0
        fig_cache = []

        for i, dim in enumerate(selected_dimensions):
            if shown >= max_charts:
                break
            sub = raw[raw["dimension"] == dim].copy()
            if sub.empty:
                continue

            nominal, upper, lower, spec_versions = pick_spec_values(sub)

            if dual_view:
                st.markdown("**完整規格視圖**")
                fig_full = build_fig(sub, dim, chart_height)
                add_spec_lines(fig_full, nominal, upper, lower)
                add_out_of_spec_points(fig_full, sub, lower, upper)
                if auto_range:
                    apply_y_range(fig_full, sub, lower, upper, False)
                st.plotly_chart(fig_full, use_container_width=True)

                st.markdown("**放大視圖**")
                fig_zoom = build_fig(sub, dim, chart_height)
                add_spec_lines(fig_zoom, nominal, upper, lower)
                add_out_of_spec_points(fig_zoom, sub, lower, upper)
                if auto_range:
                    y_min, y_max = apply_y_range(fig_zoom, sub, lower, upper, True)
                    add_spec_edge_markers(fig_zoom, lower, upper, y_min, y_max)
                st.plotly_chart(fig_zoom, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    download_plot_button(fig_full, f"{dim}_full.png")
                with c2:
                    download_plot_button(fig_zoom, f"{dim}_zoom.png")
                with c3:
                    download_excel_button(sub, f"{dim}.xlsx")
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
                fig = build_fig(sub, dim, chart_height)
                add_spec_lines(fig, nominal, upper, lower)
                add_out_of_spec_points(fig, sub, lower, upper)
                if auto_range:
                    y_min, y_max = apply_y_range(fig, sub, lower, upper, focus_on_data)
                    if focus_on_data:
                        add_spec_edge_markers(fig, lower, upper, y_min, y_max)

                st.plotly_chart(fig, use_container_width=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    download_plot_button(fig, f"{dim}.png")
                with col_right:
                    download_excel_button(sub, f"{dim}.xlsx")

                try:
                    img_bytes = fig.to_image(format="png", scale=2)
                    fig_cache.append((dim, base64.b64encode(img_bytes).decode("ascii"), img_bytes))
                except Exception:
                    pass

            if spec_versions > 1:
                st.caption("注意：此維度在不同檔案中規格不一致，已取第一筆規格作為標示。")

            shown += 1
            progress.progress((i + 1) / total_charts, text=f"正在生成圖表... ({shown}/{total_charts})")

        progress.empty()

        if shown == 0:
            st.info("目前選擇的維度沒有可用資料")

        if fig_cache:
            if st.button("產生圖表 ZIP", key="zip_boxplot"):
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

            if st.button("產生報表 (HTML)", key="html_boxplot"):
                figures = [(title, b64) for title, b64, _bytes in fig_cache]
                report_html = build_report_html(stats, figures)
                st.download_button(
                    "下載報表 (HTML)",
                    data=report_html.encode("utf-8"),
                    file_name="boxplot_report.html",
                    mime="text/html",
                )
    else:
        st.info("請選擇至少一個維度")

# Tab 2: Cpk Analysis
with tab2:
    st.subheader("Cpk 分析")

    # Calculate Cpk with ratings
    cpk_df = cpk_with_rating(raw)

    if not cpk_df.empty:
        # Show Cpk heatmap
        cpk_fig = build_cpk_heatmap(cpk_df, height=chart_height)
        st.plotly_chart(cpk_fig, use_container_width=True)

        # Show Cpk table with colored ratings
        st.markdown("### Cpk 評級表")
        st.markdown("""
        | Cpk 範圍 | 評級 | 顏色 |
        |----------|------|------|
        | >= 1.33 | 良好 | 🟢 綠色 |
        | 1.0 ~ 1.33 | 可接受 | 🟡 黃色 |
        | < 1.0 | 不良 | 🔴 紅色 |
        """)

        # Format and display table
        display_df = cpk_df[["dimension", "count", "mean", "std", "USL", "LSL", "Cp", "Cpk", "rating"]].copy()
        display_df.columns = ["維度", "數量", "平均值", "標準差", "上限", "下限", "Cp", "Cpk", "評級"]

        # Style the dataframe
        def color_rating(val):
            if val == "良好":
                return "background-color: #2ecc71; color: white"
            elif val == "可接受":
                return "background-color: #f1c40f; color: black"
            elif val == "不良":
                return "background-color: #e74c3c; color: white"
            return ""

        styled_df = display_df.style.applymap(color_rating, subset=["評級"])
        styled_df = styled_df.format({
            "平均值": "{:.4f}",
            "標準差": "{:.4f}",
            "上限": "{:.4f}",
            "下限": "{:.4f}",
            "Cp": "{:.3f}",
            "Cpk": "{:.3f}",
        }, na_rep="N/A")

        st.dataframe(styled_df, use_container_width=True)

        # Download button for Cpk report
        col1, col2 = st.columns(2)
        with col1:
            try:
                cpk_img = cpk_fig.to_image(format="png", scale=3)
                st.download_button(
                    "下載 Cpk 圖表 (PNG)",
                    data=cpk_img,
                    file_name="cpk_analysis.png",
                    mime="image/png",
                )
            except Exception:
                st.caption("PNG 下載需要 kaleido")
        with col2:
            cpk_buffer = io.BytesIO()
            with pd.ExcelWriter(cpk_buffer, engine="openpyxl") as writer:
                cpk_df.to_excel(writer, index=False, sheet_name="cpk_analysis")
            st.download_button(
                "下載 Cpk 資料 (Excel)",
                data=cpk_buffer.getvalue(),
                file_name="cpk_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("無法計算 Cpk，請確認資料包含規格上下限")

# Tab 3: SPC Control Chart
with tab3:
    st.subheader("SPC 控制圖 (I-MR)")

    st.markdown("""
    **說明：** I-MR (Individual-Moving Range) 控制圖用於監控製程穩定性。
    - **I 圖**：顯示個別量測值與控制限
    - **MR 圖**：顯示相鄰量測值的移動全距
    - **紅色圈點**：超出控制限的失控點
    """)

    # Calculate SPC data
    spc_summary, spc_points = imr_spc_points(raw)

    if not spc_points.empty:
        # Dimension selector
        spc_dims = spc_points["dimension"].unique().tolist()
        selected_spc_dim = st.selectbox(
            "選擇維度",
            options=spc_dims,
            key="spc_dim_select",
        )

        if selected_spc_dim:
            # Build and display I-MR chart
            imr_fig = build_imr_chart(spc_points, selected_spc_dim, height=chart_height + 200)
            st.plotly_chart(imr_fig, use_container_width=True)

            # Show SPC summary for selected dimension
            dim_summary = spc_summary[spc_summary["dimension"] == selected_spc_dim]
            if not dim_summary.empty:
                row = dim_summary.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("X̄ (平均值)", f"{row['X_bar']:.4f}")
                with col2:
                    st.metric("MR̄ (平均移動全距)", f"{row['MR_bar']:.4f}" if pd.notna(row['MR_bar']) else "N/A")
                with col3:
                    st.metric("σ 估計值", f"{row['sigma_est']:.4f}" if pd.notna(row['sigma_est']) else "N/A")
                with col4:
                    st.metric("樣本數", f"{int(row['count'])}")

                # Control limits display
                st.markdown("**控制限：**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"I 圖 UCL: {row['X_UCL']:.4f}" if pd.notna(row['X_UCL']) else "I 圖 UCL: N/A")
                    st.write(f"I 圖 LCL: {row['X_LCL']:.4f}" if pd.notna(row['X_LCL']) else "I 圖 LCL: N/A")
                with col2:
                    st.write(f"MR 圖 UCL: {row['MR_UCL']:.4f}" if pd.notna(row['MR_UCL']) else "MR 圖 UCL: N/A")
                    st.write(f"MR 圖 LCL: 0")

            # Check for out-of-control points
            dim_points = spc_points[spc_points["dimension"] == selected_spc_dim]
            x_ucl = dim_points["X_UCL"].iloc[0]
            x_lcl = dim_points["X_LCL"].iloc[0]
            mr_ucl = dim_points["MR_UCL"].iloc[0]

            ooc_x = dim_points[(dim_points["value"] > x_ucl) | (dim_points["value"] < x_lcl)]
            ooc_mr = dim_points[dim_points["MR"] > mr_ucl] if pd.notna(mr_ucl) else pd.DataFrame()

            if not ooc_x.empty or not ooc_mr.empty:
                st.warning(f"⚠️ 發現失控點：I 圖 {len(ooc_x)} 點，MR 圖 {len(ooc_mr)} 點")

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                try:
                    spc_img = imr_fig.to_image(format="png", scale=3)
                    st.download_button(
                        "下載 SPC 圖 (PNG)",
                        data=spc_img,
                        file_name=f"{selected_spc_dim}_spc.png",
                        mime="image/png",
                        key="spc_png",
                    )
                except Exception:
                    st.caption("PNG 下載需要 kaleido")
            with col2:
                spc_buffer = io.BytesIO()
                with pd.ExcelWriter(spc_buffer, engine="openpyxl") as writer:
                    dim_points.to_excel(writer, index=False, sheet_name="spc_data")
                    if not dim_summary.empty:
                        dim_summary.to_excel(writer, index=False, sheet_name="spc_summary")
                st.download_button(
                    "下載 SPC 資料 (Excel)",
                    data=spc_buffer.getvalue(),
                    file_name=f"{selected_spc_dim}_spc.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="spc_xlsx",
                )
    else:
        st.info("無法計算 SPC 控制圖資料")

# Tab 4: Normalized Deviation
with tab4:
    st.subheader("標準化偏離分析")

    deviation_df = calculate_normalized_deviation(raw)

    if not deviation_df.empty:
        st.markdown("""
        **公式說明：**
        - 偏離% = (量測值 - 規格中值) / 公差 × 100%
        - 公差 = (上限 - 下限) / 2
        - ±100% 代表剛好在規格邊界
        """)

        # Dimension selector for deviation chart
        deviation_dims = deviation_df["dimension"].unique().tolist()
        selected_dev_dim = st.selectbox(
            "選擇維度",
            options=deviation_dims,
            key="deviation_dim_select",
        )

        if selected_dev_dim:
            dev_fig = build_normalized_deviation_chart(deviation_df, selected_dev_dim, chart_height)
            st.plotly_chart(dev_fig, use_container_width=True)

            # Show summary statistics for deviation
            dim_dev = deviation_df[deviation_df["dimension"] == selected_dev_dim]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均偏離", f"{dim_dev['deviation_pct'].mean():.1f}%")
            with col2:
                st.metric("最大偏離", f"{dim_dev['deviation_pct'].max():.1f}%")
            with col3:
                st.metric("最小偏離", f"{dim_dev['deviation_pct'].min():.1f}%")
            with col4:
                out_count = (dim_dev['deviation_pct'].abs() > 100).sum()
                st.metric("超規格點數", f"{out_count}")

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                try:
                    dev_img = dev_fig.to_image(format="png", scale=3)
                    st.download_button(
                        "下載偏離圖 (PNG)",
                        data=dev_img,
                        file_name=f"{selected_dev_dim}_deviation.png",
                        mime="image/png",
                        key="dev_png",
                    )
                except Exception:
                    st.caption("PNG 下載需要 kaleido")
            with col2:
                dev_buffer = io.BytesIO()
                with pd.ExcelWriter(dev_buffer, engine="openpyxl") as writer:
                    dim_dev.to_excel(writer, index=False, sheet_name="deviation")
                st.download_button(
                    "下載偏離資料 (Excel)",
                    data=dev_buffer.getvalue(),
                    file_name=f"{selected_dev_dim}_deviation.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dev_xlsx",
                )
    else:
        st.info("無法計算標準化偏離，請確認資料包含完整的規格中值與上下限")

# Tab 5: Position Comparison
with tab5:
    st.subheader("模次比較分析")

    # Check if we have multi-position data
    has_positions = raw["pos_in_mold"].notna().any()

    if has_positions:
        st.markdown("""
        **說明：** 此圖表用於比較不同模次位置 (P1, P2, P3...) 的量測分布。
        適用於多模次檔案或包含位置資訊的資料。
        """)

        # Dimension selector for position comparison
        pos_dims = selected_dimensions if selected_dimensions else all_dimensions
        selected_pos_dim = st.selectbox(
            "選擇維度",
            options=pos_dims,
            key="position_dim_select",
        )

        if selected_pos_dim:
            pos_fig = build_position_comparison_chart(raw, selected_pos_dim, chart_height)
            st.plotly_chart(pos_fig, use_container_width=True)

            # Show position statistics
            pos_sub = raw[raw["dimension"] == selected_pos_dim].copy()
            pos_sub["position"] = pos_sub["pos_in_mold"].apply(
                lambda x: f"P{int(x)}" if pd.notna(x) else "P?"
            )

            pos_stats = pos_sub.groupby("position")["value"].agg(["count", "mean", "std", "min", "max"])
            pos_stats.columns = ["數量", "平均值", "標準差", "最小值", "最大值"]
            st.dataframe(pos_stats.style.format("{:.4f}", subset=["平均值", "標準差", "最小值", "最大值"]), use_container_width=True)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                try:
                    pos_img = pos_fig.to_image(format="png", scale=3)
                    st.download_button(
                        "下載模次比較圖 (PNG)",
                        data=pos_img,
                        file_name=f"{selected_pos_dim}_position.png",
                        mime="image/png",
                        key="pos_png",
                    )
                except Exception:
                    st.caption("PNG 下載需要 kaleido")
            with col2:
                pos_buffer = io.BytesIO()
                with pd.ExcelWriter(pos_buffer, engine="openpyxl") as writer:
                    pos_sub.to_excel(writer, index=False, sheet_name="position_data")
                st.download_button(
                    "下載模次資料 (Excel)",
                    data=pos_buffer.getvalue(),
                    file_name=f"{selected_pos_dim}_position.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="pos_xlsx",
                )
    else:
        st.info("資料中未包含模次位置資訊 (pos_in_mold)，無法進行模次比較分析。")
