"""Dimension Quality Analyzer - Streamlit Web Application."""

import base64
import io
import os
import re
import zipfile
from collections import defaultdict

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


# ============================================================================
# Helper functions for smart grouping
# ============================================================================

def _analyze_tag_structure(tags: list) -> dict:
    """Analyze the structure of pos_tag labels.

    Identifies patterns like #1-1, #2-3 (cavity-cycle format).

    Returns:
        dict with keys: format, cavities, cycles, tags_by_cavity
    """
    result = {
        "format": "unknown",
        "cavities": set(),
        "cycles": set(),
        "tags_by_cavity": defaultdict(list),
        "all_tags": tags,
    }

    # Pattern: #1-1, #2-3, 1-1, 2/3, etc.
    cavity_cycle_pattern = re.compile(r"#?\s*(\d+)\s*[-_/]\s*(\d+)")

    parsed_count = 0
    for tag in tags:
        tag_str = str(tag).strip()
        match = cavity_cycle_pattern.search(tag_str)
        if match:
            cavity = int(match.group(1))
            cycle = int(match.group(2))
            result["cavities"].add(cavity)
            result["cycles"].add(cycle)
            result["tags_by_cavity"][cavity].append(tag_str)
            parsed_count += 1

    if parsed_count > 0 and parsed_count >= len(tags) * 0.5:
        result["format"] = "cavity-cycle"
        result["cavities"] = sorted(result["cavities"])
        result["cycles"] = sorted(result["cycles"])
        # Sort tags within each cavity by cycle number
        for cavity in result["tags_by_cavity"]:
            result["tags_by_cavity"][cavity] = sorted(
                result["tags_by_cavity"][cavity],
                key=lambda t: int(cavity_cycle_pattern.search(t).group(2)) if cavity_cycle_pattern.search(t) else 0
            )
    else:
        result["format"] = "simple"

    return result


def _generate_cavity_cycle_grouping(
    num_cavities: int,
    num_cycles: int,
    arrangement: str,
    start_p: int = 1,
    file_name: str = "",
) -> str:
    """Generate grouping rules based on cavity count, cycle count, and arrangement.

    Args:
        num_cavities: Number of cavities
        num_cycles: Number of cycles/molds
        arrangement: "cavity_first" (穴號優先) or "cycle_first" (模次優先)
        start_p: Starting P number
        file_name: Optional filename to include as header

    Returns:
        Grouping rules string

    Examples:
        cavity_first (4穴3模次): Labels are arranged as cavity1-cycle1,2,3, cavity2-cycle1,2,3...
            P1: 1,2,3    ← cavity 1's cycles 1~3
            P2: 4,5,6    ← cavity 2's cycles 1~3
            P3: 7,8,9    ← cavity 3's cycles 1~3
            P4: 10,11,12 ← cavity 4's cycles 1~3

        cycle_first (4穴3模次): Labels are arranged as cycle1-cavity1,2,3,4, cycle2-cavity1,2,3,4...
            P1: 1,5,9    ← cavity 1 in cycles 1,2,3
            P2: 2,6,10   ← cavity 2 in cycles 1,2,3
            P3: 3,7,11   ← cavity 3 in cycles 1,2,3
            P4: 4,8,12   ← cavity 4 in cycles 1,2,3
    """
    lines = []

    if file_name:
        lines.append(f"# {file_name}")

    for cavity in range(num_cavities):
        p_num = start_p + cavity
        if arrangement == "cavity_first":
            # 穴號優先: labels 1,2,3 are cavity 1's cycles 1~3
            # label = cavity * num_cycles + cycle + 1
            tags = [cavity * num_cycles + cycle + 1 for cycle in range(num_cycles)]
        else:
            # 模次優先: labels 1,2,3,4 are cycle 1's cavities 1~4
            # label = cycle * num_cavities + cavity + 1
            tags = [cycle * num_cavities + cavity + 1 for cycle in range(num_cycles)]

        lines.append(f"P{p_num}: {','.join(map(str, tags))}")

    return "\n".join(lines)


def _generate_grouping_suggestion(raw_df: pd.DataFrame, file_list: list, start_p: int = 1) -> str:
    """Generate smart grouping suggestion based on file structure.

    Args:
        raw_df: DataFrame with file and pos_tag columns
        file_list: List of filenames
        start_p: Starting P number

    Returns:
        Suggested grouping rules as string
    """
    lines = []
    current_p = start_p

    for fname in file_list:
        sub = raw_df[raw_df["file"] == fname]
        if "pos_tag" not in sub.columns:
            continue

        tags = sub["pos_tag"].dropna().unique().tolist()
        if not tags:
            continue

        analysis = _analyze_tag_structure(tags)

        lines.append(f"# {fname}")

        if analysis["format"] == "cavity-cycle" and analysis["tags_by_cavity"]:
            # Group by cavity
            for cavity in sorted(analysis["tags_by_cavity"].keys()):
                cavity_tags = analysis["tags_by_cavity"][cavity]
                lines.append(f"P{current_p}: {','.join(cavity_tags)}")
                current_p += 1
        else:
            # Simple: all tags in one group
            lines.append(f"P{current_p}: {','.join(str(t) for t in sorted(tags))}")
            current_p += 1

        lines.append("")  # Empty line between files

    return "\n".join(lines).strip()


def _get_grouping_details(raw_df: pd.DataFrame) -> dict:
    """Get details of how data is grouped.

    Returns:
        dict mapping group name to dict of {filename: [tags]}
    """
    details = defaultdict(lambda: defaultdict(list))

    for _, row in raw_df.iterrows():
        group = row.get("group", "")
        file_name = row.get("file", "")
        tag = row.get("pos_tag", "")

        if pd.notna(tag) and str(tag).strip():
            if str(tag) not in details[group][file_name]:
                details[group][file_name].append(str(tag))

    # Sort tags within each file
    for group in details:
        for file_name in details[group]:
            details[group][file_name] = sorted(details[group][file_name])

    return dict(details)


# ============================================================================
# Streamlit App
# ============================================================================

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

# Calculate file list and mold counts early (needed for custom grouping UI)
file_list = sorted(raw["file"].dropna().unique().tolist())
file_mold_counts = {}
for fname in file_list:
    sub = raw[raw["file"] == fname]
    molds = [m for m in sub.get("mold", pd.Series(dtype=str)).dropna().unique() if str(m).strip()]
    file_mold_counts[fname] = len(molds)

# Display mode selection
mode = st.radio(
    "顯示模式",
    options=["自動分組", "強制分檔顯示", "全部合併成一張圖"],
    index=0,
    horizontal=True,
)

# Custom grouping rules
use_custom_grouping = st.checkbox("使用自訂分組規則", value=False)
custom_groups = ""
if use_custom_grouping:
    # Show available files and tags for reference
    with st.expander("查看可用的檔案與標籤", expanded=False):
        for fname in file_list:
            sub = raw[raw["file"] == fname]
            tags = sub["pos_tag"].dropna().unique().tolist() if "pos_tag" in sub.columns else []
            if tags:
                analysis = _analyze_tag_structure(tags)
                st.write(f"**{fname}**")
                if analysis["format"] == "cavity-cycle":
                    st.write(f"  - 格式: 穴號-模次")
                    st.write(f"  - 穴數: {len(analysis['cavities'])} ({', '.join(map(str, analysis['cavities']))})")
                    st.write(f"  - 模次: {len(analysis['cycles'])} ({', '.join(map(str, analysis['cycles']))})")
                st.write(f"  - 標籤: {', '.join(str(t) for t in sorted(tags)[:20])}")
                if len(tags) > 20:
                    st.caption(f"    ...還有 {len(tags) - 20} 個標籤")

    # Per-file configuration for cavity/cycle based grouping
    st.markdown("---")
    st.markdown("#### 各檔案獨立配置")
    st.caption("針對每個檔案分別設定排列方式、穴數、模次數")

    # Initialize session state for file configs if not exists
    if "file_configs" not in st.session_state:
        st.session_state["file_configs"] = {}

    # Global starting P number
    start_p_num = st.number_input(
        "起始 P 編號",
        min_value=1,
        value=1,
        step=1,
        help="第一個檔案的 P 編號從幾開始，後續檔案會自動遞增"
    )

    # Configuration for each file
    file_configs = {}
    for i, fname in enumerate(file_list):
        with st.expander(f"📁 {fname}", expanded=True):
            # Get file's tag info
            sub = raw[raw["file"] == fname]
            tags = sub["pos_tag"].dropna().unique().tolist() if "pos_tag" in sub.columns else []
            num_tags = len(tags)

            # Try to detect format
            analysis = _analyze_tag_structure(tags) if tags else {"format": "unknown"}
            detected_format = analysis.get("format", "unknown")

            # Show detected info
            if detected_format == "cavity-cycle":
                st.info(f"偵測到 #穴-模次 格式，共 {num_tags} 個標籤")
            elif num_tags > 0:
                st.info(f"偵測到 {num_tags} 個標籤: {', '.join(str(t) for t in sorted(tags)[:10])}{'...' if num_tags > 10 else ''}")

            col1, col2, col3 = st.columns(3)
            with col1:
                arr = st.radio(
                    "排列方式",
                    options=["穴號優先", "模次優先", "使用智能偵測"],
                    index=2,
                    key=f"arr_{fname}",
                    help="穴號優先：1,2,3 是穴1的模次1~3\n模次優先：1,2,3,4 是模次1的穴1~4\n智能偵測：自動分析 #穴-模次 格式"
                )
            with col2:
                # Try to auto-detect cavity count
                default_cav = len(analysis.get("cavities", [])) if detected_format == "cavity-cycle" else 4
                if default_cav == 0:
                    default_cav = 4
                cav = st.number_input(
                    "穴數",
                    min_value=1,
                    max_value=50,
                    value=default_cav,
                    step=1,
                    key=f"cav_{fname}"
                )
            with col3:
                # Try to auto-detect cycle count
                default_cyc = len(analysis.get("cycles", [])) if detected_format == "cavity-cycle" else 3
                if default_cyc == 0:
                    default_cyc = 3
                cyc = st.number_input(
                    "模次數",
                    min_value=1,
                    max_value=50,
                    value=default_cyc,
                    step=1,
                    key=f"cyc_{fname}"
                )

            file_configs[fname] = {
                "arrangement": arr,
                "cavities": cav,
                "cycles": cyc,
                "tags": tags,
                "analysis": analysis,
            }

    # Generate button
    col_gen, col_smart = st.columns(2)
    with col_gen:
        if st.button("✅ 產生所有檔案的分組規則", help="根據各檔案的配置產生分組規則"):
            parts = []
            current_p = start_p_num

            for fname in file_list:
                cfg = file_configs[fname]

                if cfg["arrangement"] == "使用智能偵測":
                    # Use smart detection for #cavity-cycle format
                    sub_df = raw[raw["file"] == fname]
                    part = _generate_grouping_suggestion(sub_df, [fname], current_p)
                    # Count groups generated
                    num_groups = part.count("\nP") + (1 if part.startswith("P") or part.startswith("# ") else 0)
                    # More accurate count
                    num_groups = len([line for line in part.split("\n") if line.strip().startswith("P")])
                else:
                    arr_key = "cavity_first" if cfg["arrangement"] == "穴號優先" else "cycle_first"
                    part = _generate_cavity_cycle_grouping(
                        cfg["cavities"], cfg["cycles"], arr_key, current_p, fname
                    )
                    num_groups = cfg["cavities"]

                parts.append(part)
                current_p += num_groups

            st.session_state["grouping_suggestion"] = "\n\n".join(parts)
            st.success("已產生分組規則，請查看下方文字區域")

    with col_smart:
        if st.button("🔮 全部使用智能偵測", help="所有檔案都使用智能偵測（適用於 #穴-模次 格式標籤）"):
            suggestion = _generate_grouping_suggestion(raw, file_list, start_p_num)
            st.session_state["grouping_suggestion"] = suggestion
            st.success("已產生分組規則，請查看下方文字區域")

    st.markdown("---")

    # Use suggestion if available
    default_value = st.session_state.get(
        "grouping_suggestion",
        "# 檔案1.xlsm\nP1: #1-1,#1-2,#1-3,#1-4,#1-5\nP2: #2-1,#2-2,#2-3,#2-4,#2-5\n\n# 檔案2.xlsm\nP5: #1-1,#1-2\nP6: #2-1,#2-2"
    )

    custom_groups = st.text_area(
        "自訂分組格式",
        value=default_value,
        help="格式說明：\n"
             "• 簡單格式：群組名稱: 標籤1,標籤2,...\n"
             "• 按檔案分組：先用 # 檔案名稱 指定檔案，接著定義該檔案的分組規則\n"
             "• 當多檔案有相同標籤時，請使用按檔案分組格式避免誤判\n"
             "• 使用上方「各檔案獨立配置」可針對不同檔案設定不同的分組方式",
        height=200,
    )

# Assign groups using vectorized function
def _apply_group_labels(base: pd.Series) -> pd.Series:
    base = base.fillna("").astype(str)
    if mode == "全部合併成一張圖":
        return pd.Series("合併", index=raw.index)
    if mode == "強制分檔顯示":
        return (raw["file"].fillna("") + " " + base.replace("", "合併")).str.strip()
    return base.replace("", "合併")


def _parse_custom_groups(text: str) -> dict:
    """Parse custom grouping rules with optional file-specific syntax.

    Supports two formats:
    1. Simple format (applies to all files):
       P1: #1-1,#1-2,#1-3

    2. File-specific format (use # filename to specify):
       # 6.xlsm
       P1: #1-1,#1-2,#1-3
       # 8.xlsm
       P5: #1-1,#1-2

    Returns:
        dict with keys as (filename, tag) tuples or just tag strings
    """
    mapping = {}
    current_file = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for file header: # filename.xlsm or # filename.xlsx
        if line.startswith("#") and ("." in line or line[1:].strip() in ["", " "]):
            potential_file = line[1:].strip()
            if potential_file and ("." in potential_file):
                current_file = potential_file
                continue

        if ":" not in line:
            continue

        name, values = line.split(":", 1)
        group_name = name.strip()
        for token in values.split(","):
            tag = token.strip()
            if tag:
                if current_file:
                    # File-specific mapping: (filename, tag) -> group
                    mapping[(current_file, tag)] = group_name
                else:
                    # Global mapping: tag -> group
                    mapping[tag] = group_name

    return mapping


def _apply_custom_mapping(raw: pd.DataFrame, mapping: dict) -> pd.Series:
    """Apply custom group mapping considering file-specific rules."""
    result = pd.Series("其他", index=raw.index)

    # Check if mapping has file-specific keys (tuples)
    has_file_keys = any(isinstance(k, tuple) for k in mapping.keys())

    if has_file_keys:
        # File-specific mapping
        for idx, row in raw.iterrows():
            file_name = row.get("file", "")
            tag = row.get("pos_tag", "")
            if pd.isna(tag):
                tag = ""

            # Try file-specific key first
            key = (file_name, str(tag))
            if key in mapping:
                result.loc[idx] = mapping[key]
            # Fall back to global key
            elif str(tag) in mapping:
                result.loc[idx] = mapping[str(tag)]
    else:
        # Simple global mapping
        result = raw["pos_tag"].map(mapping).fillna("其他")

    return result


if use_custom_grouping:
    if "pos_tag" in raw.columns and raw["pos_tag"].notna().any():
        mapping = _parse_custom_groups(custom_groups)
        mapped = _apply_custom_mapping(raw, mapping)
        raw["group"] = _apply_group_labels(mapped)
    else:
        st.warning("找不到量測欄位標籤，已改用自動分組。")
        raw["group"] = assign_groups_vectorized(raw, mode, file_mold_counts, file_list)
else:
    base_group = pd.Series("", index=raw.index, dtype=str)
    for fname in file_list:
        sub = raw[raw["file"] == fname]
        has_pos_tag = "pos_tag" in sub.columns and sub["pos_tag"].notna().any()
        has_cavity = "cavity" in sub.columns and sub["cavity"].notna().any()
        has_pos_in_mold = "pos_in_mold" in sub.columns and sub["pos_in_mold"].notna().any()
        has_arrangement = "arrangement" in sub.columns and sub["arrangement"].notna().any()
        tag_has_cavity_cycle = False
        if has_pos_tag:
            tag_has_cavity_cycle = sub["pos_tag"].astype(str).str.contains(
                r"#\s*\d+\s*[-/]\s*\d+|\d+\s*[-/]\s*\d+", regex=True
            ).any()

        # Get arrangement type for this file
        file_arrangement = sub["arrangement"].iloc[0] if has_arrangement else "unknown"

        if file_arrangement == "cavity_first" and has_cavity:
            # CAV.X format (10.xlsm style): group by cavity number
            # 穴號優先：CAV.1 下的標籤 1,2,3 都屬於 P1
            base_group.loc[sub.index] = sub["cavity"].apply(
                lambda x: f"P{int(x)}" if pd.notna(x) else ""
            )
        elif file_arrangement == "cycle_first" and has_pos_in_mold:
            # 第X模 format (2.xlsm style): group by position within mold
            # 模次優先：第一模的位置1、第二模的位置1、第三模的位置1 都屬於 P1
            base_group.loc[sub.index] = sub["pos_in_mold"].apply(
                lambda x: f"P{int(x)}" if pd.notna(x) else ""
            )
        elif has_cavity and tag_has_cavity_cycle:
            # #穴-模次 format in pos_tag
            base_group.loc[sub.index] = sub["cavity"].apply(
                lambda x: f"P{int(x)}" if pd.notna(x) else ""
            )
        elif has_cavity and sub["cavity"].dropna().nunique() == 1 and has_pos_tag:
            only = sub["cavity"].dropna().unique().tolist()
            p_label = f"P{int(only[0])}" if only else "P1"
            base_group.loc[sub.index] = p_label
        elif has_pos_in_mold:
            base_group.loc[sub.index] = sub["pos_in_mold"].apply(
                lambda x: f"P{int(x)}" if pd.notna(x) else ""
            )
        else:
            fallback = assign_groups_vectorized(sub, mode, file_mold_counts, file_list)
            base_group.loc[sub.index] = fallback

    raw["group"] = _apply_group_labels(base_group)

# Show grouping details
with st.expander("📋 查看分組詳情", expanded=False):
    grouping_details = _get_grouping_details(raw)
    if grouping_details:
        # Sort groups naturally (P1, P2, ... P10, P11)
        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

        sorted_groups = sorted(grouping_details.keys(), key=natural_sort_key)

        st.markdown("**目前分組方式：**")

        # Display mode info
        if mode == "全部合併成一張圖":
            st.info("模式：全部合併成一張圖（所有數據合併顯示）")
        elif mode == "強制分檔顯示":
            st.info("模式：強制分檔顯示（按檔案 + 位置分組）")
        else:
            st.info("模式：自動分組（按穴號/位置自動分配）")

        # Create columns for better layout
        cols = st.columns(min(3, len(sorted_groups)) if sorted_groups else 1)

        for i, group in enumerate(sorted_groups):
            col_idx = i % len(cols)
            with cols[col_idx]:
                file_tags = grouping_details[group]
                st.markdown(f"**{group}**")
                for file_name, tags in sorted(file_tags.items()):
                    if len(file_list) > 1:
                        st.write(f"  `{file_name}`:")
                    st.write(f"  {', '.join(tags[:10])}")
                    if len(tags) > 10:
                        st.caption(f"  ...共 {len(tags)} 個標籤")
                st.markdown("---")

        # Summary statistics
        total_groups = len(sorted_groups)
        total_points = len(raw)
        st.caption(f"共 {total_groups} 個分組，{total_points} 筆數據")
    else:
        st.write("無分組資訊")

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
