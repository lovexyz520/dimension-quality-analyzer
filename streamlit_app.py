"""Dimension Quality Analyzer - Streamlit Web Application."""

import base64
import io
import json
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
    calc_out_of_spec,
    pick_spec_values,
    stats_table,
    cpk_with_rating,
    imr_spc_points,
    calculate_normalized_deviation,
    calculate_correlation_matrix,
    get_high_correlation_pairs,
    nelson_rules_for_dimension,
    normality_test,
    variance_decomposition,
    diagnose_overview,
    cavity_fingerprint_data,
    add_spec_lines,
    add_spec_band,
    build_fig,
    apply_y_range,
    add_spec_edge_markers,
    add_out_of_spec_points,
    build_cpk_heatmap,
    build_normalized_deviation_chart,
    build_position_comparison_chart,
    build_imr_chart,
    build_correlation_heatmap,
    build_correlation_scatter,
    build_histogram,
    build_cpk_trend,
    build_cavity_fingerprint,
    build_pareto_chart,
    DEFAULT_GEMINI_MODEL,
    AVAILABLE_GEMINI_MODELS,
    STAGE_LABELS,
    DEFAULT_STAGE,
    STAGE_TRYOUT,
    STAGE_MONITORING,
    detect_systematic_bias,
    build_analysis_payload,
    format_payload_as_text,
    generate_ai_report,
    download_excel_button,
    download_stats_excel,
    download_quality_reports,
    build_report_html,
    download_pdf_report_button,
    assign_groups_vectorized,
)

# Plotly modebar 內建相機按鈕直接下載高解析 PNG（瀏覽器端轉檔，不經 kaleido）
PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


# ============================================================================
# Cached computation layer
# Streamlit 每次互動都會重跑整個腳本；把解析與統計包進 cache_data，
# 只有資料真正改變時才重算。
# ============================================================================

@st.cache_data(show_spinner=False)
def _load_excel_cached(content: bytes, name: str):
    return load_excel(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def _stats_cached(df: pd.DataFrame) -> pd.DataFrame:
    return stats_table(df)


@st.cache_data(show_spinner=False)
def _cpk_cached(df: pd.DataFrame) -> pd.DataFrame:
    return cpk_with_rating(df)


@st.cache_data(show_spinner=False)
def _spc_cached(df: pd.DataFrame):
    return imr_spc_points(df)


@st.cache_data(show_spinner=False)
def _deviation_cached(df: pd.DataFrame) -> pd.DataFrame:
    return calculate_normalized_deviation(df)


@st.cache_data(show_spinner=False)
def _overview_cached(df: pd.DataFrame) -> pd.DataFrame:
    return diagnose_overview(df)


@st.cache_data(show_spinner=False)
def _variance_cached(df: pd.DataFrame) -> pd.DataFrame:
    return variance_decomposition(df)


@st.cache_data(show_spinner=False)
def _fingerprint_cached(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return cavity_fingerprint_data(df, group_col)


def lazy_png_download(fig, filename: str, key: str, label: str = "產生 PNG 圖檔") -> None:
    """PNG 轉檔（kaleido）延後到使用者點擊時才執行，避免拖慢每次重跑。"""
    if st.button(label, key=f"png_btn_{key}"):
        try:
            with st.spinner("正在轉檔..."):
                img_bytes = fig.to_image(format="png", scale=3)
            st.download_button(
                "⬇️ 下載 PNG",
                data=img_bytes,
                file_name=filename,
                mime="image/png",
                key=f"png_dl_{key}",
            )
        except Exception:
            st.caption("PNG 轉檔需要 kaleido，請確認已安裝")


def _figs_to_png_bytes(figs: list, scale: int = 2) -> list:
    """Convert (title, fig) list to (title, png_bytes), with progress bar."""
    results = []
    if not figs:
        return results
    prog = st.progress(0, text="正在轉檔 PNG...")
    for i, (title, fig) in enumerate(figs):
        try:
            results.append((title, fig.to_image(format="png", scale=scale)))
        except Exception:
            pass
        prog.progress((i + 1) / len(figs), text=f"正在轉檔 PNG... ({i + 1}/{len(figs)})")
    prog.empty()
    return results


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
# Authentication (Google OIDC via st.login)
# ============================================================================

def _require_auth() -> None:
    """Gate the app behind Google login when [auth] is configured in secrets.

    設計為「未設定即不啟用」：本機開發（secrets 無 [auth]）不需登入；
    部署時在 Streamlit Cloud 的 Secrets 設好 [auth] 即自動強制登入。
    可另設 allowed_emails 白名單，只放行指定 Google 帳號。
    """
    try:
        auth_configured = "auth" in st.secrets
    except Exception:
        auth_configured = False
    if not auth_configured:
        return  # 本機開發或未設定登入

    user = getattr(st, "user", None)
    if user is None or not getattr(user, "is_logged_in", False):
        st.title("盒鬚圖分析工具")
        st.info("本系統含機密量測資料，請先登入。")
        st.button("使用 Google 帳號登入", on_click=st.login, type="primary")
        st.stop()

    # 白名單檢查（若有設定）
    try:
        allowed = [str(e).strip().lower() for e in st.secrets.get("allowed_emails", [])]
    except Exception:
        allowed = []
    email = (getattr(user, "email", "") or "").strip().lower()
    if allowed and email not in allowed:
        st.error(f"帳號 {email or '（未知）'} 未獲授權使用本系統，請聯絡管理員。")
        st.button("登出", on_click=st.logout)
        st.stop()

    # 已登入且授權：側欄顯示身分與登出
    with st.sidebar:
        st.caption(f"👤 已登入：{getattr(user, 'name', None) or email}")
        st.button("登出", on_click=st.logout, key="_logout_btn")


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(page_title="Box-and-Whisker Plot", layout="wide")

_require_auth()

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

# Analysis stage — 決定整體解讀框架（不改變統計計算，只改變重點與報告語氣）
analysis_stage = st.radio(
    "分析情境",
    options=STAGE_LABELS,
    index=STAGE_LABELS.index(DEFAULT_STAGE),
    horizontal=True,
    help="試模：重點在偏移補正方向與穴間平衡，淡化 Cpk/SPC；"
         "量產首件：確認符合規格與 Cpk；量產監控：重點在 SPC 穩定性與趨勢。",
)

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
    df, err = _load_excel_cached(file.getvalue(), file.name)
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

    # Import previously saved grouping config (JSON)
    cfg_file = st.file_uploader(
        "匯入分組設定 (JSON)",
        type=["json"],
        key="grouping_cfg_upload",
        help="套用先前匯出的分組設定，下次上傳同格式檔案不必重設",
    )
    if cfg_file is not None:
        cfg_sig = f"{cfg_file.name}_{cfg_file.size}"
        if st.session_state.get("_grouping_cfg_applied") != cfg_sig:
            try:
                cfg = json.loads(cfg_file.getvalue().decode("utf-8"))
                if "start_p" in cfg:
                    st.session_state["start_p_num"] = int(cfg["start_p"])
                for cfg_fname, fc in cfg.get("files", {}).items():
                    if cfg_fname in file_list:
                        if fc.get("arrangement") in ["穴號優先", "模次優先", "使用智能偵測"]:
                            st.session_state[f"arr_{cfg_fname}"] = fc["arrangement"]
                        if "cavities" in fc:
                            st.session_state[f"cav_{cfg_fname}"] = int(fc["cavities"])
                        if "cycles" in fc:
                            st.session_state[f"cyc_{cfg_fname}"] = int(fc["cycles"])
                if cfg.get("grouping_rules"):
                    st.session_state["grouping_suggestion"] = cfg["grouping_rules"]
                st.session_state["_grouping_cfg_applied"] = cfg_sig
                st.success("已套用分組設定")
            except Exception as exc:
                st.error(f"設定檔格式錯誤: {exc}")

    # Global starting P number
    start_p_num = st.number_input(
        "起始 P 編號",
        min_value=1,
        value=1,
        step=1,
        key="start_p_num",
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

    # Export current grouping config for reuse next time
    export_cfg = {
        "start_p": int(start_p_num),
        "files": {
            fname: {
                "arrangement": cfg["arrangement"],
                "cavities": int(cfg["cavities"]),
                "cycles": int(cfg["cycles"]),
            }
            for fname, cfg in file_configs.items()
        },
        "grouping_rules": custom_groups,
    }
    st.download_button(
        "💾 匯出分組設定 (JSON)",
        data=json.dumps(export_cfg, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="grouping_config.json",
        mime="application/json",
        help="儲存目前的排列方式、穴數、模次數與分組規則，下次可直接匯入",
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

# Pre-compute statistics once (cached) — shared by all tabs below
stats = _stats_cached(raw)
cpk_df = _cpk_cached(raw)
overview_df = _overview_cached(raw)

search_text = st.text_input("維度搜尋", value="", placeholder="例如：1-A, 2-B, 2/A1")
filtered_dimensions = all_dimensions
if search_text:
    tokens = [t.strip().lower() for t in search_text.replace(";", ",").split(",") if t.strip()]
    if tokens:
        filtered_dimensions = [d for d in all_dimensions if any(t in d.lower() for t in tokens)]

# Quick selection buttons for the dimension multiselect
_ms_key = "dim_multiselect"


def _set_dim_selection(dims):
    st.session_state[_ms_key] = dims


_oos_dims = stats.loc[stats["out_of_spec"] > 0, "dimension"].tolist()
_bad_cpk_dims = (
    cpk_df.loc[cpk_df["Cpk"] < 1.33, "dimension"].tolist() if not cpk_df.empty else []
)

qc1, qc2, qc3, qc4 = st.columns(4)
with qc1:
    st.button("全選", on_click=_set_dim_selection, args=(filtered_dimensions,), use_container_width=True)
with qc2:
    st.button("清空", on_click=_set_dim_selection, args=([],), use_container_width=True)
with qc3:
    st.button(
        f"只選超規格 ({len(_oos_dims)})",
        on_click=_set_dim_selection,
        args=([d for d in _oos_dims if d in filtered_dimensions],),
        use_container_width=True,
        disabled=not _oos_dims,
    )
with qc4:
    st.button(
        f"只選 Cpk < 1.33 ({len(_bad_cpk_dims)})",
        on_click=_set_dim_selection,
        args=([d for d in _bad_cpk_dims if d in filtered_dimensions],),
        use_container_width=True,
        disabled=not _bad_cpk_dims,
    )

if _ms_key not in st.session_state:
    st.session_state[_ms_key] = filtered_dimensions
else:
    st.session_state[_ms_key] = [
        d for d in st.session_state[_ms_key] if d in filtered_dimensions
    ]

selected_dimensions = st.multiselect(
    "選擇要顯示的維度 (預設全選)",
    options=filtered_dimensions,
    key=_ms_key,
)

max_charts = st.number_input(
    "最多顯示幾張圖 (避免一次渲染太多)",
    min_value=1,
    max_value=max(1, len(filtered_dimensions)),
    value=min(20, max(1, len(filtered_dimensions))),
    step=1,
)

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
                add_spec_band(fig, lower, upper)
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
tab_overview, tab1, tab2, tab3, tab4, tab5, tab6, tab_detail, tab_ai = st.tabs(
    ["📌 診斷總覽", "盒鬚圖", "Cpk 分析", "SPC 控制圖", "標準化偏離", "模次比較", "相關性分析", "🔍 維度詳情", "🤖 AI 報告"]
)

# Tab 0: Diagnostic Overview — 一眼看出哪些維度最需要關注
with tab_overview:
    st.subheader("📌 診斷總覽")
    st.caption(f"目前分析情境：**{analysis_stage}**。彙整 Cpk、超規格、SPC 異常與常態性，依嚴重度排序。細節請到「🔍 維度詳情」分頁。")

    # Stage-aware guidance banner
    if analysis_stage == STAGE_TRYOUT:
        bias = detect_systematic_bias(raw)
        st.info(
            "🔧 **試模模式**：此階段尺寸偏移屬正常，重點是決定修模與調機方向。"
            "請以『偏移方向與量』『穴間平衡』為主，Cpk / SPC / 常態性此時僅供參考、不作良不良判定。"
        )
        if bias.get("message"):
            if bias.get("dominant") and bias["dominant_pct"] >= 50:
                st.warning(f"📐 偏移研判：{bias['message']}")
            else:
                st.caption(f"📐 偏移研判：{bias['message']}")
        st.caption("💡 建議到「模次比較」分頁用『按位置』跑穴號指紋圖，判斷是全穴同偏（調製程）還是某穴獨偏（修該穴模仁）。")
    elif analysis_stage == STAGE_MONITORING:
        st.info(
            "📈 **量產監控模式**：重點在製程穩定性。請優先看 SPC 控制圖的 Nelson 異常模式與跨批 Cpk 趨勢，"
            "偏移與異常點視為製程漂移警訊。"
        )

    if overview_df.empty:
        st.info("無可分析的資料")
    else:
        n_total = len(overview_df)
        n_bad = int((overview_df["rating"] == "不良").sum())
        n_ok = int((overview_df["rating"] == "可接受").sum())
        n_oos_points = int(overview_df["out_of_spec"].sum())
        n_nelson_dims = int((overview_df["nelson_count"] > 0).sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("維度總數", n_total)
        with m2:
            st.metric("Cpk 不良 (<1.0)", n_bad, delta=None if n_bad == 0 else f"{n_bad} 個需處理", delta_color="inverse")
        with m3:
            st.metric("Cpk 可接受 (1.0~1.33)", n_ok)
        with m4:
            st.metric("超規格點總數", n_oos_points)
        with m5:
            st.metric("SPC 異常維度數", n_nelson_dims)

        attention = overview_df[overview_df["priority"] > 0]
        if attention.empty:
            st.success("🎉 所有維度狀態良好：Cpk 達標、無超規格點、無 SPC 異常模式")
        else:
            st.markdown("### ⚠️ 需要關注的維度（依嚴重度排序）")

            disp = attention.copy()
            disp["常態性"] = disp["is_normal"].map(
                {True: "常態", False: "⚠️ 非常態", None: "—"}
            )
            disp = disp[[
                "dimension", "count", "Cpk", "rating", "out_of_spec",
                "nelson_count", "nelson_rules", "常態性", "suggestion",
            ]]
            disp.columns = [
                "維度", "樣本數", "Cpk", "評級", "超規格點",
                "SPC 異常點", "違反規則", "常態性", "調機建議",
            ]

            def _color_rating(val):
                if val == "良好":
                    return "background-color: #2ecc71; color: white"
                if val == "可接受":
                    return "background-color: #f1c40f; color: black"
                if val == "不良":
                    return "background-color: #e74c3c; color: white"
                return ""

            styled = disp.style.map(_color_rating, subset=["評級"])
            styled = styled.format({"Cpk": "{:.3f}"}, na_rep="N/A")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(
                "說明：SPC 異常點為違反 Nelson 規則（R1 超出 3σ、R2 連續 9 點同側、"
                "R3 連續 6 點趨勢、R5 2/3 點超出 2σ）的點數；"
                "非常態分布時 Cpk 數值僅供參考；樣本數 < 25 時 Cpk 估計不確定性大。"
            )

            # Pareto chart: 依「先解決影響最大的少數」原則決定處理優先序
            st.markdown("### 📊 Pareto 排列圖（處理優先序）")
            pareto_metric = st.radio(
                "排序依據",
                options=["超規格點數", "SPC 異常點數"],
                horizontal=True,
                key="pareto_metric",
            )
            metric_col = "out_of_spec" if pareto_metric == "超規格點數" else "nelson_count"
            pareto_fig = build_pareto_chart(
                overview_df,
                label_col="dimension",
                value_col=metric_col,
                title=f"各維度{pareto_metric} — Pareto 排列圖",
                height=chart_height,
            )
            st.plotly_chart(pareto_fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("長條為各維度的異常數量（由大到小），折線為累積百分比；落在 80% 線以左的少數維度通常貢獻大部分問題，建議優先處理。")
            lazy_png_download(pareto_fig, "pareto.png", key="pareto_overview")

        with st.expander("查看全部維度", expanded=False):
            full_disp = overview_df.copy()
            full_disp["常態性"] = full_disp["is_normal"].map(
                {True: "常態", False: "非常態", None: "—"}
            )
            full_disp = full_disp[[
                "dimension", "count", "Cpk", "rating", "out_of_spec", "nelson_count", "常態性",
            ]]
            full_disp.columns = ["維度", "樣本數", "Cpk", "評級", "超規格點", "SPC 異常點", "常態性"]
            st.dataframe(
                full_disp.style.format({"Cpk": "{:.3f}"}, na_rep="N/A"),
                use_container_width=True,
                hide_index=True,
            )

# Tab 1: Box-and-Whisker Plots (original functionality)
with tab1:
    st.subheader("盒鬚圖")
    st.caption("💡 圖表右上角的相機圖示可直接下載單張高解析 PNG（瀏覽器端轉檔，即時完成）")

    total_charts = min(len(selected_dimensions), max_charts)
    if total_charts > 0:
        progress = st.progress(0, text="正在生成圖表...")
        shown = 0
        fig_cache = []  # (title, fig) — PNG 轉檔延後到匯出按鈕才執行

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
                add_spec_band(fig_full, lower, upper)
                add_spec_lines(fig_full, nominal, upper, lower)
                add_out_of_spec_points(fig_full, sub, lower, upper)
                if auto_range:
                    apply_y_range(fig_full, sub, lower, upper, False)
                st.plotly_chart(fig_full, use_container_width=True, config=PLOTLY_CONFIG)

                st.markdown("**放大視圖**")
                fig_zoom = build_fig(sub, dim, chart_height)
                add_spec_band(fig_zoom, lower, upper)
                add_spec_lines(fig_zoom, nominal, upper, lower)
                add_out_of_spec_points(fig_zoom, sub, lower, upper)
                if auto_range:
                    y_min, y_max = apply_y_range(fig_zoom, sub, lower, upper, True)
                    add_spec_edge_markers(fig_zoom, lower, upper, y_min, y_max)
                st.plotly_chart(fig_zoom, use_container_width=True, config=PLOTLY_CONFIG)

                download_excel_button(sub, f"{dim}.xlsx")

                fig_cache.append((f"{dim} (full)", fig_full))
                fig_cache.append((f"{dim} (zoom)", fig_zoom))
            else:
                fig = build_fig(sub, dim, chart_height)
                add_spec_band(fig, lower, upper)
                add_spec_lines(fig, nominal, upper, lower)
                add_out_of_spec_points(fig, sub, lower, upper)
                if auto_range:
                    y_min, y_max = apply_y_range(fig, sub, lower, upper, focus_on_data)
                    if focus_on_data:
                        add_spec_edge_markers(fig, lower, upper, y_min, y_max)

                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                download_excel_button(sub, f"{dim}.xlsx")

                fig_cache.append((dim, fig))

            if spec_versions > 1:
                st.caption("注意：此維度在不同檔案中規格不一致，已取第一筆規格作為標示。")

            shown += 1
            progress.progress((i + 1) / total_charts, text=f"正在生成圖表... ({shown}/{total_charts})")

        progress.empty()

        if shown == 0:
            st.info("目前選擇的維度沒有可用資料")

        if fig_cache:
            st.markdown("---")
            col_zip, col_html = st.columns(2)
            with col_zip:
                if st.button("產生圖表 ZIP", key="zip_boxplot"):
                    png_list = _figs_to_png_bytes(fig_cache)
                    if png_list:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for title, img_bytes in png_list:
                                zf.writestr(f"{title}.png", img_bytes)
                            zf.writestr("summary.csv", stats.to_csv(index=False))
                            zf.writestr("long_table.csv", raw.to_csv(index=False))
                        st.download_button(
                            "⬇️ 下載全部圖表 ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="boxplot_charts.zip",
                            mime="application/zip",
                        )
                    else:
                        st.warning("PNG 轉檔失敗，請確認已安裝 kaleido")
            with col_html:
                if st.button("產生報表 (HTML)", key="html_boxplot"):
                    png_list = _figs_to_png_bytes(fig_cache)
                    if png_list:
                        figures = [
                            (title, base64.b64encode(img_bytes).decode("ascii"))
                            for title, img_bytes in png_list
                        ]
                        report_html = build_report_html(stats, figures)
                        st.download_button(
                            "⬇️ 下載報表 (HTML)",
                            data=report_html.encode("utf-8"),
                            file_name="boxplot_report.html",
                            mime="text/html",
                        )
                    else:
                        st.warning("PNG 轉檔失敗，請確認已安裝 kaleido")
    else:
        st.info("請選擇至少一個維度")

# Tab 2: Cpk Analysis
with tab2:
    st.subheader("Cpk 分析")

    if not cpk_df.empty:
        # Show Cpk heatmap
        cpk_fig = build_cpk_heatmap(cpk_df, height=chart_height)
        st.plotly_chart(cpk_fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Show Cpk table with colored ratings
        st.markdown("### Cpk 評級表")
        st.markdown("""
        | Cpk 範圍 | 評級 | 顏色 |
        |----------|------|------|
        | >= 1.33 | 良好 | 🟢 綠色 |
        | 1.0 ~ 1.33 | 可接受 | 🟡 黃色 |
        | < 1.0 | 不良 | 🔴 紅色 |
        """)

        # Format and display table (with Cpk confidence interval + sample warning)
        display_df = cpk_df[["dimension", "count", "mean", "std", "USL", "LSL", "Cp", "Cpk"]].copy()
        display_df["Cpk 95% CI"] = cpk_df.apply(
            lambda r: f"[{r['Cpk_LCI']:.2f}, {r['Cpk_UCI']:.2f}]"
            if pd.notna(r.get("Cpk_LCI")) else "N/A",
            axis=1,
        )
        display_df["評級"] = cpk_df["rating"]
        display_df["樣本警告"] = cpk_df["low_sample"].map({True: "⚠️ n<25", False: ""})
        display_df.columns = ["維度", "數量", "平均值", "標準差", "上限", "下限", "Cp", "Cpk", "Cpk 95% CI", "評級", "樣本警告"]

        # Style the dataframe
        def color_rating(val):
            if val == "良好":
                return "background-color: #2ecc71; color: white"
            elif val == "可接受":
                return "background-color: #f1c40f; color: black"
            elif val == "不良":
                return "background-color: #e74c3c; color: white"
            return ""

        styled_df = display_df.style.map(color_rating, subset=["評級"])
        styled_df = styled_df.format({
            "平均值": "{:.4f}",
            "標準差": "{:.4f}",
            "上限": "{:.4f}",
            "下限": "{:.4f}",
            "Cp": "{:.3f}",
            "Cpk": "{:.3f}",
        }, na_rep="N/A")

        st.dataframe(styled_df, use_container_width=True)
        st.caption(
            "Cpk 95% CI 為信賴區間：樣本數少時區間很寬，代表 Cpk 點估計不可靠，"
            "判定良莠時建議看區間下限而非點估計。"
        )

        # Cross-file Cpk trend (multi-file = time order comparison)
        if len(file_list) >= 2:
            with st.expander("📈 跨檔案 Cpk 趨勢（依上傳順序，可用於批次/改模前後比較）", expanded=False):
                upload_order = list(dict.fromkeys(f.name for f in uploaded_files))
                trend_rows = []
                for fname in upload_order:
                    file_cpk = _cpk_cached(raw[raw["file"] == fname])
                    for _, r in file_cpk.iterrows():
                        trend_rows.append(
                            {"file": fname, "dimension": r["dimension"], "Cpk": r["Cpk"]}
                        )
                trend_df = pd.DataFrame(trend_rows).dropna(subset=["Cpk"])
                if not trend_df.empty:
                    trend_fig = build_cpk_trend(trend_df, height=chart_height)
                    st.plotly_chart(trend_fig, use_container_width=True, config=PLOTLY_CONFIG)
                    st.caption("點選圖例可隱藏/顯示個別維度")
                else:
                    st.info("各檔案皆無法計算 Cpk")

        # Download button for Cpk report
        col1, col2 = st.columns(2)
        with col1:
            lazy_png_download(cpk_fig, "cpk_analysis.png", key="cpk_tab")
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
    - **橘色菱形**：違反 Nelson 規則的異常模式點（偏移/趨勢等，在超限之前的預警）
    """)

    # Calculate SPC data (cached)
    spc_summary, spc_points = _spc_cached(raw)

    if not spc_points.empty:
        # Dimension selector
        spc_dims = spc_points["dimension"].unique().tolist()
        selected_spc_dim = st.selectbox(
            "選擇維度",
            options=spc_dims,
            key="spc_dim_select",
        )

        if selected_spc_dim:
            # Nelson rules on this dimension (same point order as the I chart)
            dim_points = spc_points[spc_points["dimension"] == selected_spc_dim]
            nelson_df, nelson_findings = nelson_rules_for_dimension(dim_points["value"])

            # Build and display I-MR chart
            imr_fig = build_imr_chart(
                spc_points, selected_spc_dim,
                height=chart_height + 200,
                nelson_points=nelson_df,
            )
            st.plotly_chart(imr_fig, use_container_width=True, config=PLOTLY_CONFIG)

            # Nelson rule findings
            if nelson_findings:
                st.markdown("**Nelson 規則判讀：**")
                for f in nelson_findings:
                    pts_text = ", ".join(map(str, f["points"][:15]))
                    if len(f["points"]) > 15:
                        pts_text += f"...（共 {len(f['points'])} 點）"
                    st.warning(f"**{f['rule']}**｜{f['description']}｜異常點序號：{pts_text}")
            else:
                st.success("✅ 未偵測到 Nelson 規則異常模式（無偏移、趨勢等前兆）")

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
                lazy_png_download(imr_fig, f"{selected_spc_dim}_spc.png", key="spc_tab")
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

    deviation_df = _deviation_cached(raw)

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
            st.plotly_chart(dev_fig, use_container_width=True, config=PLOTLY_CONFIG)

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
                lazy_png_download(dev_fig, f"{selected_dev_dim}_deviation.png", key="dev_tab")
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
            st.plotly_chart(pos_fig, use_container_width=True, config=PLOTLY_CONFIG)

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
                lazy_png_download(pos_fig, f"{selected_pos_dim}_position.png", key="pos_tab")
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

    # Variance decomposition: cavity-to-cavity vs cycle-to-cycle
    st.markdown("---")
    st.markdown("### 🔬 變異來源分解（穴間 vs 模次間）")
    st.caption(
        "判斷改善方向的關鍵：穴間差異大 → 修模/模穴均一性方向；"
        "模次間漂移大 → 成型條件穩定性（調機）方向。百分比為該因子解釋的變異占比。"
    )

    var_df = _variance_cached(raw)
    if not var_df.empty:
        var_disp = var_df.copy()
        var_disp["穴間變異 %"] = var_disp["cavity_pct"].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "—"
        )
        var_disp["模次間變異 %"] = var_disp["cycle_pct"].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "—"
        )
        var_disp = var_disp[["dimension", "穴間變異 %", "cavity_groups", "模次間變異 %", "cycle_groups", "judgment"]]
        var_disp.columns = ["維度", "穴間變異 %", "穴數", "模次間變異 %", "模次數", "判斷"]
        st.dataframe(var_disp, use_container_width=True, hide_index=True)
    else:
        st.info("資料不足（每個維度至少需要 4 筆數據與穴號/模次資訊）")

    # Cavity fingerprint: 每穴在各維度的偏離指紋，找出整體偏大/偏小的穴
    st.markdown("---")
    st.markdown("### 🖐️ 穴號指紋圖（找出整體偏移的穴/位置）")
    st.caption(
        "每條線代表一個穴（或位置/分組），Y 軸為標準化偏離%。整條線偏高/偏低，"
        "代表該穴一致地偏大/偏小 → 修模仁的直接線索。"
    )

    fp_options = []
    if "cavity" in raw.columns and raw["cavity"].dropna().nunique() >= 2:
        fp_options.append(("按穴號 (cavity)", "cavity", "穴"))
    if "pos_in_mold" in raw.columns and raw["pos_in_mold"].dropna().nunique() >= 2:
        fp_options.append(("按位置 (pos_in_mold)", "pos_in_mold", "P"))
    if "group" in raw.columns:
        _fp_valid_groups = [g for g in raw["group"].unique() if g and "合併" not in str(g)]
        if len(_fp_valid_groups) >= 2:
            fp_options.append(("按分組 (group)", "group", ""))

    if fp_options:
        fp_choice = st.radio(
            "分組方式",
            options=[o[0] for o in fp_options],
            horizontal=True,
            key="fingerprint_group_method",
        )
        fp_col, fp_label = next((c, lbl) for (name, c, lbl) in fp_options if name == fp_choice)
        fp_df = _fingerprint_cached(raw, fp_col)
        if not fp_df.empty:
            fp_fig = build_cavity_fingerprint(fp_df, group_label=fp_label or "組", height=chart_height)
            st.plotly_chart(fp_fig, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("提示：若某條線在多數維度都明顯高於/低於其他線，該穴很可能整體偏移，可優先檢查對應模仁。")
            lazy_png_download(fp_fig, "cavity_fingerprint.png", key="fingerprint")
        else:
            st.info("此分組方式下沒有足夠的規格資訊可計算偏離")
    else:
        st.info("需要穴號、位置或分組資訊（且至少 2 組）才能繪製指紋圖")

# Tab 6: Correlation Analysis
with tab6:
    st.subheader("相關性分析")

    st.markdown("""
    **說明：** 分析不同維度之間的相關性，找出連動的尺寸。
    - **強正相關 (r > 0.7)**：兩維度同時增減，可能受同一製程因素影響
    - **強負相關 (r < -0.7)**：一維度增加時另一維度減少
    - **弱相關 (|r| < 0.3)**：兩維度獨立變動
    """)

    if len(all_dimensions) >= 2:
        # Sub-tabs for different analysis modes
        corr_tab1, corr_tab2, corr_tab3 = st.tabs(["📊 相關性矩陣", "📁 按檔案比較", "🔧 按穴號比較"])

        # ============================================================
        # Sub-tab 1: Basic Correlation Matrix
        # ============================================================
        with corr_tab1:
            # Data source selection
            col_source, col_threshold = st.columns([2, 2])
            with col_source:
                source_options = ["全部合併"] + file_list
                selected_source = st.selectbox(
                    "資料範圍",
                    options=source_options,
                    key="corr_source",
                    help="選擇要分析的資料範圍"
                )
            with col_threshold:
                corr_threshold = st.slider(
                    "高相關閾值 |r| >=",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="corr_threshold",
                )

            # Filter data based on selection
            if selected_source == "全部合併":
                corr_data = raw
            else:
                corr_data = raw[raw["file"] == selected_source]

            if len(corr_data) > 0:
                corr_matrix, pivot_table = calculate_correlation_matrix(corr_data)

                if not corr_matrix.empty and len(corr_matrix) >= 2:
                    # Show metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("維度數量", len(corr_matrix.columns))
                    with col_m2:
                        st.metric("樣本數", len(pivot_table))
                    with col_m3:
                        high_pairs = get_high_correlation_pairs(corr_matrix, threshold=corr_threshold)
                        st.metric("高相關對數", len(high_pairs))

                    # Guard: an all-NaN matrix renders as a blank heatmap.
                    # Tell the user why instead of showing an empty image.
                    if bool(corr_matrix.isna().all().all()):
                        st.warning(
                            "⚠️ 相關性矩陣無法計算（熱力圖會全空白）。可能原因：\n"
                            f"- 有效配對樣本數過少（目前僅 {len(pivot_table)} 個，"
                            "每對維度至少需 3 個配對樣本）\n"
                            "- 各維度的量測點無法配對（缺少穴號/模次/位置等可對應的識別資訊）\n\n"
                            "建議：上傳含多個模次或多穴的資料，或改用「全部合併」擴大樣本範圍。"
                        )
                    else:
                        # Display correlation heatmap
                        corr_fig = build_correlation_heatmap(corr_matrix, height=max(400, len(corr_matrix) * 25 + 100))
                        st.plotly_chart(corr_fig, use_container_width=True, config=PLOTLY_CONFIG)

                        # High correlation pairs
                        st.markdown("### 高相關維度對")
                        high_corr_pairs = get_high_correlation_pairs(corr_matrix, threshold=corr_threshold)

                        if not high_corr_pairs.empty:
                            display_pairs = high_corr_pairs.copy()
                            display_pairs["相關性"] = display_pairs["correlation"].apply(
                                lambda x: f"{'🔴' if x > 0 else '🔵'} {x:.3f}"
                            )
                            display_pairs["強度"] = display_pairs["abs_correlation"].apply(
                                lambda x: "強" if x >= 0.8 else "中"
                            )
                            display_pairs = display_pairs.rename(columns={"dim1": "維度 1", "dim2": "維度 2"})

                            st.dataframe(
                                display_pairs[["維度 1", "維度 2", "相關性", "強度"]],
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Scatter plot
                            st.markdown("### 散佈圖")
                            pair_options = [
                                f"{row['dim1']} vs {row['dim2']} (r={row['correlation']:.3f})"
                                for _, row in high_corr_pairs.iterrows()
                            ]
                            selected_pair = st.selectbox("選擇維度對", options=pair_options, key="corr_pair_select")

                            if selected_pair:
                                pair_idx = pair_options.index(selected_pair)
                                dim1 = high_corr_pairs.iloc[pair_idx]["dim1"]
                                dim2 = high_corr_pairs.iloc[pair_idx]["dim2"]
                                scatter_fig = build_correlation_scatter(pivot_table, dim1, dim2, chart_height)
                                st.plotly_chart(scatter_fig, use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.info(f"沒有找到相關係數絕對值 >= {corr_threshold} 的維度對")

                        # Download buttons
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            lazy_png_download(corr_fig, "correlation_heatmap.png", key="corr_tab")
                        with col2:
                            corr_buffer = io.BytesIO()
                            with pd.ExcelWriter(corr_buffer, engine="openpyxl") as writer:
                                corr_matrix.to_excel(writer, sheet_name="correlation_matrix")
                                if not high_corr_pairs.empty:
                                    high_corr_pairs.to_excel(writer, index=False, sheet_name="high_correlation_pairs")
                            st.download_button(
                                "下載相關性資料 (Excel)",
                                data=corr_buffer.getvalue(),
                                file_name="correlation_analysis.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="corr_xlsx",
                            )
                else:
                    st.warning("資料不足以計算相關性矩陣（需要至少 2 個維度）")
            else:
                st.warning("所選資料範圍沒有資料")

        # ============================================================
        # Sub-tab 2: Compare by File
        # ============================================================
        with corr_tab2:
            st.markdown("**比較不同檔案之間的相關性差異**")
            st.caption("若相關性在不同檔案間差異大，可能代表製程變異或條件不同")

            if len(file_list) >= 2:
                # Select dimension pair to compare
                all_dims_sorted = sorted(all_dimensions)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    compare_dim1 = st.selectbox("維度 1", options=all_dims_sorted, key="file_cmp_dim1")
                with col_d2:
                    remaining_dims = [d for d in all_dims_sorted if d != compare_dim1]
                    compare_dim2 = st.selectbox("維度 2", options=remaining_dims, key="file_cmp_dim2")

                if compare_dim1 and compare_dim2:
                    # Calculate correlation for each file
                    file_corrs = []
                    for fname in file_list:
                        file_data = raw[raw["file"] == fname]
                        try:
                            corr_mat, pivot = calculate_correlation_matrix(file_data)
                            if compare_dim1 in corr_mat.columns and compare_dim2 in corr_mat.columns:
                                r = corr_mat.loc[compare_dim1, compare_dim2]
                                n = len(pivot.dropna(subset=[compare_dim1, compare_dim2]))
                                file_corrs.append({"檔案": fname, "相關係數": r, "樣本數": n})
                        except Exception:
                            pass

                    if file_corrs:
                        file_corr_df = pd.DataFrame(file_corrs)

                        # Display as bar chart
                        import plotly.express as px
                        fig_compare = px.bar(
                            file_corr_df,
                            x="檔案",
                            y="相關係數",
                            color="相關係數",
                            color_continuous_scale=["#2166ac", "#f7f7f7", "#b2182b"],
                            range_color=[-1, 1],
                            text="相關係數",
                        )
                        fig_compare.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                        fig_compare.update_layout(
                            title=f"{compare_dim1} vs {compare_dim2} - 各檔案相關性比較",
                            yaxis_range=[-1.2, 1.2],
                            height=400,
                            showlegend=False,
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)

                        # Show table
                        st.dataframe(
                            file_corr_df.style.format({"相關係數": "{:.3f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Statistics
                        corr_values = file_corr_df["相關係數"].dropna()
                        if len(corr_values) >= 2:
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("平均相關係數", f"{corr_values.mean():.3f}")
                            with col_s2:
                                st.metric("標準差", f"{corr_values.std():.3f}")
                            with col_s3:
                                diff = corr_values.max() - corr_values.min()
                                st.metric("最大差異", f"{diff:.3f}",
                                         delta="穩定" if diff < 0.2 else "有變異",
                                         delta_color="normal" if diff < 0.2 else "inverse")
                    else:
                        st.warning("無法計算所選維度對的相關性")
            else:
                st.info("需要上傳至少 2 個檔案才能進行檔案間比較")

        # ============================================================
        # Sub-tab 3: Compare by Cavity/Position
        # ============================================================
        with corr_tab3:
            st.markdown("**比較不同穴號/位置之間的相關性差異**")
            st.caption("若相關性在不同穴號間差異大，可能代表特定模穴有問題")

            # Check available grouping options
            has_cavity_data = "cavity" in raw.columns and raw["cavity"].dropna().nunique() >= 2
            has_pos_in_mold = "pos_in_mold" in raw.columns and raw["pos_in_mold"].dropna().nunique() >= 2

            # Filter out merged groups for comparison
            if "group" in raw.columns:
                valid_groups = [g for g in raw["group"].unique() if g and "合併" not in str(g)]
                has_group_data = len(valid_groups) >= 2
            else:
                has_group_data = False
                valid_groups = []

            # Show data status for debugging
            with st.expander("📋 資料狀態", expanded=False):
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    if has_cavity_data:
                        cavities = raw["cavity"].dropna().unique()
                        st.success(f"穴號: {len(cavities)} 個 ({', '.join(map(lambda x: str(int(x)), sorted(cavities)[:5]))}{'...' if len(cavities) > 5 else ''})")
                    else:
                        st.warning("無穴號資訊")
                with col_info2:
                    if has_pos_in_mold:
                        positions = raw["pos_in_mold"].dropna().unique()
                        st.success(f"位置: {len(positions)} 個")
                    else:
                        st.warning("無位置資訊")
                with col_info3:
                    if has_group_data:
                        st.success(f"分組: {len(valid_groups)} 個")
                    else:
                        st.warning("無有效分組")

            if has_cavity_data or has_pos_in_mold or has_group_data:
                # Let user choose grouping method
                grouping_options = []
                if has_cavity_data:
                    grouping_options.append("按穴號 (cavity)")
                if has_pos_in_mold:
                    grouping_options.append("按位置 (pos_in_mold)")
                if has_group_data:
                    grouping_options.append("按分組 (group)")

                selected_grouping = st.radio(
                    "分組方式",
                    options=grouping_options,
                    horizontal=True,
                    key="cavity_grouping_method"
                )

                # Determine grouping column based on selection
                if "穴號" in selected_grouping:
                    group_col = "cavity"
                    group_label = "穴號"
                    groups = sorted(raw["cavity"].dropna().unique().tolist())
                    group_names = [f"穴{int(g)}" for g in groups]
                elif "位置" in selected_grouping:
                    group_col = "pos_in_mold"
                    group_label = "位置"
                    groups = sorted(raw["pos_in_mold"].dropna().unique().tolist())
                    group_names = [f"P{int(g)}" for g in groups]
                else:
                    group_col = "group"
                    group_label = "分組"
                    groups = sorted(valid_groups)
                    group_names = groups

                # Select dimension pair to compare
                all_dims_sorted = sorted(all_dimensions)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    cavity_dim1 = st.selectbox("維度 1", options=all_dims_sorted, key="cavity_cmp_dim1")
                with col_d2:
                    remaining_dims = [d for d in all_dims_sorted if d != cavity_dim1]
                    cavity_dim2 = st.selectbox("維度 2", options=remaining_dims, key="cavity_cmp_dim2")

                if cavity_dim1 and cavity_dim2:
                    # Show selected groups info
                    st.caption(f"將比較 {len(groups)} 個{group_label}: {', '.join(group_names[:10])}{'...' if len(group_names) > 10 else ''}")

                    # Calculate correlation for each group
                    group_corrs = []
                    skipped_groups = []
                    for g, gname in zip(groups, group_names):
                        group_data = raw[raw[group_col] == g]
                        try:
                            corr_mat, pivot = calculate_correlation_matrix(group_data, min_samples=3)
                            if cavity_dim1 in corr_mat.columns and cavity_dim2 in corr_mat.columns:
                                r = corr_mat.loc[cavity_dim1, cavity_dim2]
                                if pd.notna(r):
                                    n = len(pivot.dropna(subset=[cavity_dim1, cavity_dim2]))
                                    group_corrs.append({group_label: gname, "相關係數": r, "樣本數": n})
                                else:
                                    skipped_groups.append(f"{gname}(資料不足)")
                            else:
                                skipped_groups.append(f"{gname}(維度不存在)")
                        except Exception as e:
                            skipped_groups.append(f"{gname}(錯誤)")

                    if skipped_groups:
                        st.caption(f"⚠️ 跳過的分組: {', '.join(skipped_groups)}")

                    if group_corrs:
                        group_corr_df = pd.DataFrame(group_corrs)

                        # Display as bar chart
                        import plotly.express as px
                        fig_cavity = px.bar(
                            group_corr_df,
                            x=group_label,
                            y="相關係數",
                            color="相關係數",
                            color_continuous_scale=["#2166ac", "#f7f7f7", "#b2182b"],
                            range_color=[-1, 1],
                            text="相關係數",
                        )
                        fig_cavity.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                        fig_cavity.update_layout(
                            title=f"{cavity_dim1} vs {cavity_dim2} - 各{group_label}相關性比較",
                            yaxis_range=[-1.2, 1.2],
                            height=400,
                            showlegend=False,
                        )
                        st.plotly_chart(fig_cavity, use_container_width=True)

                        # Show table
                        st.dataframe(
                            group_corr_df.style.format({"相關係數": "{:.3f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Statistics
                        corr_values = group_corr_df["相關係數"].dropna()
                        if len(corr_values) >= 2:
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("平均相關係數", f"{corr_values.mean():.3f}")
                            with col_s2:
                                st.metric("標準差", f"{corr_values.std():.3f}")
                            with col_s3:
                                diff = corr_values.max() - corr_values.min()
                                st.metric("最大差異", f"{diff:.3f}",
                                         delta="穩定" if diff < 0.2 else "有變異",
                                         delta_color="normal" if diff < 0.2 else "inverse")
                    else:
                        st.warning(f"無法計算所選維度對的相關性。可能原因：\n"
                                  f"- 每個{group_label}的配對樣本數不足（需要至少 3 個）\n"
                                  f"- 所選維度在某些{group_label}中沒有資料")
            else:
                st.info("需要有穴號、位置或分組資訊才能進行比較\n\n"
                       "可能的原因：\n"
                       "- Excel 欄位標題沒有 CAV.X 或 第X模 格式\n"
                       "- 標籤沒有 #穴-模次 格式\n"
                       "- 顯示模式為「全部合併」")
    else:
        st.info("需要至少 2 個維度才能進行相關性分析")

# Tab 7: Dimension Detail — 單一維度的完整檢視
with tab_detail:
    st.subheader("🔍 維度詳情")
    st.caption("一頁看完單一維度的所有視角：盒鬚圖、分布直方圖、I-MR 控制圖與全部統計判讀")

    detail_dim = st.selectbox("選擇維度", options=all_dimensions, key="detail_dim_select")

    if detail_dim:
        detail_sub = raw[raw["dimension"] == detail_dim].copy()
        d_nominal, d_upper, d_lower, _ = pick_spec_values(detail_sub)
        d_values = detail_sub["value"].astype(float)
        d_oos_mask = calc_out_of_spec(d_values, d_lower, d_upper)

        # Key metrics row
        d_cpk_row = cpk_df[cpk_df["dimension"] == detail_dim] if not cpk_df.empty else pd.DataFrame()
        dm1, dm2, dm3, dm4, dm5 = st.columns(5)
        with dm1:
            st.metric("樣本數", int(d_values.count()))
        with dm2:
            st.metric("平均值", f"{d_values.mean():.4f}")
        with dm3:
            st.metric("標準差", f"{d_values.std(ddof=1):.4f}" if d_values.count() >= 2 else "N/A")
        with dm4:
            if not d_cpk_row.empty and pd.notna(d_cpk_row.iloc[0]["Cpk"]):
                r = d_cpk_row.iloc[0]
                st.metric("Cpk", f"{r['Cpk']:.3f}", delta=r["rating"],
                          delta_color="normal" if r["rating"] == "良好" else "inverse")
                if pd.notna(r.get("Cpk_LCI")):
                    st.caption(f"95% CI [{r['Cpk_LCI']:.2f}, {r['Cpk_UCI']:.2f}]")
            else:
                st.metric("Cpk", "N/A")
        with dm5:
            st.metric("超規格點", int(d_oos_mask.sum()),
                      delta="需處理" if d_oos_mask.any() else None, delta_color="inverse")

        # Box plot + histogram side by side
        col_box, col_hist = st.columns(2)
        with col_box:
            d_fig = build_fig(detail_sub, detail_dim, 420)
            add_spec_band(d_fig, d_lower, d_upper)
            add_spec_lines(d_fig, d_nominal, d_upper, d_lower)
            add_out_of_spec_points(d_fig, detail_sub, d_lower, d_upper)
            apply_y_range(d_fig, detail_sub, d_lower, d_upper, False)
            st.plotly_chart(d_fig, use_container_width=True, config=PLOTLY_CONFIG)
        with col_hist:
            d_hist = build_histogram(detail_sub, detail_dim, d_nominal, d_upper, d_lower, 420)
            st.plotly_chart(d_hist, use_container_width=True, config=PLOTLY_CONFIG)

        # I-MR chart with Nelson markers
        _, d_spc_points = _spc_cached(raw)
        d_dim_points = d_spc_points[d_spc_points["dimension"] == detail_dim]
        d_nelson_df, d_nelson_findings = nelson_rules_for_dimension(d_dim_points["value"])
        d_imr = build_imr_chart(d_spc_points, detail_dim, height=560, nelson_points=d_nelson_df)
        st.plotly_chart(d_imr, use_container_width=True, config=PLOTLY_CONFIG)

        # Diagnostic findings
        st.markdown("### 📋 判讀結果")

        d_norm = normality_test(d_values)
        if d_norm["is_normal"] is False:
            st.warning(f"📉 常態性：{d_norm['message']}（建議以直方圖確認分布形狀）")
        elif d_norm["is_normal"] is True:
            st.success(f"📈 常態性：{d_norm['message']}")
        else:
            st.info(f"📈 常態性：{d_norm['message']}")

        if d_nelson_findings:
            for f in d_nelson_findings:
                pts_text = ", ".join(map(str, f["points"][:15]))
                if len(f["points"]) > 15:
                    pts_text += f"...（共 {len(f['points'])} 點）"
                st.warning(f"⚡ SPC {f['rule']}｜{f['description']}｜異常點序號：{pts_text}")
        else:
            st.success("⚡ SPC：未偵測到 Nelson 規則異常模式")

        d_overview_row = overview_df[overview_df["dimension"] == detail_dim]
        if not d_overview_row.empty and d_overview_row.iloc[0]["suggestion"]:
            st.warning(f"🔧 調機建議：{d_overview_row.iloc[0]['suggestion']}")

        d_var = _variance_cached(raw)
        d_var_row = d_var[d_var["dimension"] == detail_dim] if not d_var.empty else pd.DataFrame()
        if not d_var_row.empty:
            vr = d_var_row.iloc[0]
            cav_txt = f"{vr['cavity_pct']:.0f}%" if pd.notna(vr["cavity_pct"]) else "—"
            cyc_txt = f"{vr['cycle_pct']:.0f}%" if pd.notna(vr["cycle_pct"]) else "—"
            st.info(f"🔬 變異分解：穴間 {cav_txt}、模次間 {cyc_txt} — {vr['judgment']}")

        # Out-of-spec point traceability table
        if d_oos_mask.any():
            st.markdown("### ❌ 超規格量測點（溯源）")
            oos_cols = [c for c in ["value", "group", "pos_tag", "mold", "cavity", "cycle", "file"]
                        if c in detail_sub.columns]
            oos_table = detail_sub.loc[d_oos_mask, oos_cols].copy()
            rename_map = {"value": "量測值", "group": "分組", "pos_tag": "標籤",
                          "mold": "模次", "cavity": "穴號", "cycle": "模次序", "file": "檔案"}
            oos_table.columns = [rename_map.get(c, c) for c in oos_table.columns]
            st.dataframe(oos_table, use_container_width=True, hide_index=True)
            st.caption(f"規格範圍：{d_lower:.4f} ~ {d_upper:.4f}" if pd.notna(d_lower) and pd.notna(d_upper) else "")

        download_excel_button(detail_sub, f"{detail_dim}_detail.xlsx")

# Tab 8: AI Report — 把已算好的統計結果交給 Gemini 寫成人類可讀的報告
with tab_ai:
    st.subheader("🤖 AI 分析報告")
    st.caption(
        f"目前分析情境：**{analysis_stage}**。AI 僅根據系統已計算好的統計數字撰寫報告，"
        "不自行運算或捏造數據。免 API 的結構化摘要一定可用；填入 Gemini 金鑰後可額外產生 AI 敘述報告。"
    )

    # Build the deterministic payload once (shared by offline + AI report),
    # 依目前分析情境調整解讀框架
    ai_payload = build_analysis_payload(raw, analysis_stage)
    offline_text = format_payload_as_text(ai_payload)

    with st.expander("📄 結構化摘要（免 API，離線可用）", expanded=True):
        st.text(offline_text)
        st.download_button(
            "下載結構化摘要 (TXT)",
            data=offline_text.encode("utf-8"),
            file_name="analysis_summary.txt",
            mime="text/plain",
            key="offline_summary_dl",
        )

    st.markdown("---")
    st.markdown("### 🤖 Gemini AI 敘述報告")

    # Resolve API key: prefer secrets/env, fall back to a password input
    key_from_secret = ""
    try:
        key_from_secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key_from_secret = ""
    if not key_from_secret:
        key_from_secret = os.environ.get("GEMINI_API_KEY", "")

    col_key, col_model = st.columns([2, 1])
    with col_key:
        if key_from_secret:
            st.success("已從 secrets/環境變數載入 API 金鑰")
            api_key = key_from_secret
        else:
            api_key = st.text_input(
                "Gemini API 金鑰",
                type="password",
                help="於 Google AI Studio (aistudio.google.com) 免費申請；"
                     "或設定 st.secrets['GEMINI_API_KEY'] / 環境變數 GEMINI_API_KEY",
                key="gemini_api_key_input",
            )
    with col_model:
        model_name = st.selectbox(
            "模型",
            options=AVAILABLE_GEMINI_MODELS,
            index=AVAILABLE_GEMINI_MODELS.index(DEFAULT_GEMINI_MODEL),
            key="gemini_model_select",
            help="Flash 系列皆有免費額度；3.5 最新、2.5-lite 最省",
        )

    st.caption(
        "⚠️ 隱私提醒：產生 AI 報告會將上方結構化摘要（含尺寸名稱與統計值）傳送至 Google API。"
        "若量測資料涉及機密，請改用上方離線摘要。"
    )

    # Only call the API on explicit click (avoids cost on every rerun);
    # cache the result in session_state so reruns keep showing it.
    if st.button("✨ 產生 AI 報告", key="gen_ai_report", disabled=not api_key):
        with st.spinner(f"正在以 {model_name} 產生報告..."):
            try:
                report_text = generate_ai_report(offline_text, api_key, model_name, analysis_stage)
                st.session_state["ai_report_text"] = report_text
                st.session_state["ai_report_error"] = None
            except RuntimeError as exc:
                st.session_state["ai_report_text"] = None
                st.session_state["ai_report_error"] = str(exc)

    if st.session_state.get("ai_report_error"):
        st.error(f"產生失敗：{st.session_state['ai_report_error']}")
    if st.session_state.get("ai_report_text"):
        st.markdown(st.session_state["ai_report_text"])
        st.download_button(
            "下載 AI 報告 (Markdown)",
            data=st.session_state["ai_report_text"].encode("utf-8"),
            file_name="ai_quality_report.md",
            mime="text/markdown",
            key="ai_report_dl",
        )
        st.caption("AI 報告可能有誤，請對照實際數據與圖表確認後再作決策。")
