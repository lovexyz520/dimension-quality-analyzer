"""Excel/CSV parsing for dimension quality data.

解析分兩步：先用 `mapping.detect_mapping()` 猜出欄位對映，再用 `apply_mapping()`
套用。自動偵測與 UI 對映精靈共用同一個 `apply_mapping()`，不會有兩套邏輯漂移。
"""

from typing import Optional, Tuple
import re

import numpy as np
import pandas as pd

from .mapping import (
    LAYOUT_LONG,
    LAYOUT_WIDE,
    ColumnMapping,
    detect_mapping,
)

# 標準 schema：下游統計、繪圖、匯出、AI 報告只依賴這些欄位
STANDARD_COLUMNS = [
    "dimension", "value", "nominal", "upper", "lower",
    "mold", "meas_label", "pos_tag", "cavity", "cycle",
    "pos_raw", "pos_in_mold", "arrangement",
]


def _clean_cell(value) -> str:
    """Clean cell value to string, handling NaN and None."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _clean_label(value) -> str:
    """Normalize measurement label to a consistent string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    return text


def _parse_cavity_cycle_from_label(label: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse cavity/cycle from label like #1-3, 4-2, 8(4-2)."""
    if not label:
        return None, None
    match = re.search(r"\((\d+)\s*[-_/]\s*(\d+)\)", label)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"#?\s*(\d+)\s*[-_/]\s*(\d+)", label)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _parse_cavity_from_mold_label(label: str) -> Optional[int]:
    """Parse cavity from mold header like '#4穴'."""
    if not label:
        return None
    match = re.search(r"#?\s*(\d+)\s*穴", label)
    return int(match.group(1)) if match else None


def _parse_cycle_from_mold_label(label: str) -> Optional[int]:
    """Parse cycle from mold header like '第一模'."""
    if not label:
        return None
    match = re.search(r"第\s*(\d+)\s*模", label)
    return int(match.group(1)) if match else None


def _parse_cavity_from_cav_label(label: str) -> Optional[int]:
    """Parse cavity number from CAV.X pattern."""
    if not label:
        return None
    match = re.search(r"(?i)cav[\.\s]*(\d+)", label)
    return int(match.group(1)) if match else None


def _detect_arrangement_from_mold_labels(mold_labels: dict) -> str:
    """Detect data arrangement type from mold labels.

    Returns "cavity_first" (CAV.X), "cycle_first" (第X模), or "unknown".
    """
    has_cav_pattern = False
    has_cycle_pattern = False

    for label in set(mold_labels.values()):
        if not label:
            continue
        if re.search(r"(?i)cav[\.\s]*\d+", label):
            has_cav_pattern = True
        if re.search(r"第\s*\d+\s*模", label):
            has_cycle_pattern = True

    if has_cav_pattern and not has_cycle_pattern:
        return "cavity_first"
    if has_cycle_pattern and not has_cav_pattern:
        return "cycle_first"
    if has_cav_pattern and has_cycle_pattern:
        # 兩種樣式並存時維持原行為
        return "cycle_first"
    return "unknown"


def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    """Locate the header row. Kept for backwards compatibility."""
    from .mapping import detect_header_row

    return detect_header_row(df)


def _extract_mold_and_pos(
    df: pd.DataFrame, header_row: int, meas_cols: list, label_row: Optional[int]
) -> Tuple[dict, dict, dict, dict, dict, dict, str]:
    """Extract mold labels, measurement labels, and position information."""
    header = df.iloc[header_row]

    # 模次標題只寫在群組第一欄，向右填滿。
    mold_labels = {}
    current_label = ""
    for col in meas_cols:
        label = _clean_cell(header.iloc[col])
        if label:
            current_label = label
        mold_labels[col] = current_label

    arrangement = _detect_arrangement_from_mold_labels(mold_labels)

    pos_row = df.iloc[label_row] if label_row is not None and label_row < len(df) else None
    meas_labels, cavities, cycles = {}, {}, {}
    pos_raw, pos_in_mold, mold_counts = {}, {}, {}

    for col in meas_cols:
        raw_val = np.nan
        label = ""
        if pos_row is not None:
            raw_val = pd.to_numeric(pos_row.iloc[col], errors="coerce")
            label = _clean_label(pos_row.iloc[col])
        meas_labels[col] = label

        cav, cyc = _parse_cavity_cycle_from_label(label)
        mold_label = mold_labels.get(col, "")
        if cav is None:
            cav = _parse_cavity_from_mold_label(mold_label)
        if cav is None:
            cav = _parse_cavity_from_cav_label(mold_label)
        if cyc is None:
            cyc = _parse_cycle_from_mold_label(mold_label)

        cavities[col] = cav if cav is not None else np.nan
        cycles[col] = cyc if cyc is not None else np.nan
        pos_raw[col] = raw_val

        mold_counts[mold_label] = mold_counts.get(mold_label, 0) + 1
        pos_in_mold[col] = mold_counts[mold_label]

    return mold_labels, meas_labels, cavities, cycles, pos_raw, pos_in_mold, arrangement


def _num_col(data: pd.DataFrame, col: Optional[int]) -> pd.Series:
    """Numeric column by position, or an all-NaN series when unmapped."""
    if col is None or col >= data.shape[1]:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data.iloc[:, col], errors="coerce")


def _spec_bounds(
    mapping: ColumnMapping, data: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Derive (nominal, upper, lower) from whichever spec columns exist.

    優先用直接給的上下限；否則用標稱值 ± 公差。都沒有時回傳全 NaN，
    下游 Cpk/超規格會自動跳過該維度。
    """
    nominal = _num_col(data, mapping.nominal_col)

    if mapping.upper_col is not None or mapping.lower_col is not None:
        upper = _num_col(data, mapping.upper_col)
        lower = _num_col(data, mapping.lower_col)
        if mapping.nominal_col is None:
            nominal = (upper + lower) / 2
        return nominal, upper, lower

    plus = _num_col(data, mapping.plus_col)
    minus = _num_col(data, mapping.minus_col)
    return nominal, nominal + plus, nominal + minus


def _dimension_names(mapping: ColumnMapping, data: pd.DataFrame) -> pd.Series:
    """Join the mapped name columns with a space, skipping blanks."""
    parts = []
    for col in mapping.dimension_cols:
        if col < data.shape[1]:
            parts.append(data.iloc[:, col].map(_clean_cell))
    if not parts:
        return pd.Series("", index=data.index, dtype=str)
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep=" ").str.strip()
    return joined.str.replace(r"\s+", " ", regex=True).str.strip()


def _apply_wide(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    meas_cols = [c for c in mapping.meas_cols if c < df.shape[1]]
    if not meas_cols:
        raise ValueError("找不到任何量測數值欄位，請在對映設定中指定量測值欄")

    (
        mold_labels, meas_labels, cavities, cycles, pos_raw, pos_in_mold, arrangement
    ) = _extract_mold_and_pos(df, mapping.header_row, meas_cols, mapping.label_row)

    data = df.iloc[mapping.data_start_row :].copy()
    meas = data.iloc[:, meas_cols].apply(pd.to_numeric, errors="coerce")
    names = _dimension_names(mapping, data)
    nominal, upper, lower = _spec_bounds(mapping, data)

    keep = meas.notna().any(axis=1) & names.ne("")
    if not keep.any():
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    rows = []
    for idx in data.index[keep]:
        values = pd.to_numeric(meas.loc[idx], errors="coerce")
        for col, value in zip(meas_cols, values.tolist()):
            if pd.isna(value):
                continue
            rows.append(
                {
                    "dimension": names.loc[idx],
                    "value": float(value),
                    "nominal": float(nominal.loc[idx]) if pd.notna(nominal.loc[idx]) else np.nan,
                    "upper": float(upper.loc[idx]) if pd.notna(upper.loc[idx]) else np.nan,
                    "lower": float(lower.loc[idx]) if pd.notna(lower.loc[idx]) else np.nan,
                    "mold": mold_labels.get(col, ""),
                    "meas_label": meas_labels.get(col, ""),
                    "pos_tag": meas_labels.get(col, ""),
                    "cavity": float(cavities[col]) if pd.notna(cavities[col]) else np.nan,
                    "cycle": float(cycles[col]) if pd.notna(cycles[col]) else np.nan,
                    "pos_raw": float(pos_raw[col]) if pd.notna(pos_raw[col]) else np.nan,
                    "pos_in_mold": float(pos_in_mold[col]) if pd.notna(pos_in_mold[col]) else np.nan,
                    "arrangement": arrangement,
                }
            )

    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def _apply_long(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """One measurement per row (CMM / MES / SPC exports)."""
    data = df.iloc[mapping.data_start_row :].copy()

    names = _dimension_names(mapping, data)
    values = _num_col(data, mapping.value_col)
    nominal, upper, lower = _spec_bounds(mapping, data)
    cavity = _num_col(data, mapping.cavity_col)
    cycle = _num_col(data, mapping.cycle_col)

    if mapping.mold_col is not None and mapping.mold_col < data.shape[1]:
        mold = data.iloc[:, mapping.mold_col].map(_clean_cell)
    else:
        # 沒有明確的模具/組別欄時用模次當 mold：分組邏輯以 mold 的相異數
        # 判定是否為多模次檔案，留空會讓整份資料塌成單一「合併」群組。
        mold = cycle.map(lambda v: f"第{int(v)}模" if pd.notna(v) else "")

    keep = values.notna() & names.ne("")
    out = pd.DataFrame(
        {
            "dimension": names[keep],
            "value": values[keep].astype(float),
            "nominal": nominal[keep],
            "upper": upper[keep],
            "lower": lower[keep],
            "mold": mold[keep],
            "cavity": cavity[keep],
            "cycle": cycle[keep],
        }
    ).reset_index(drop=True)

    if out.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # pos_tag 供分組與相關性配對使用；優先用穴號，其次模次。
    tag = out["cavity"].map(lambda v: f"{int(v)}" if pd.notna(v) else "")
    fallback = out["cycle"].map(lambda v: f"{int(v)}" if pd.notna(v) else "")
    out["pos_tag"] = tag.where(tag.ne(""), fallback)
    out["meas_label"] = out["pos_tag"]
    out["pos_raw"] = out["cavity"]

    # pos_in_mold 決定 P1..Pn 分組：優先用穴號，其次模次，最後退回序號。
    if out["cavity"].notna().any():
        out["pos_in_mold"] = out["cavity"]
        out["arrangement"] = "cavity_first"
    elif out["cycle"].notna().any():
        out["pos_in_mold"] = out["cycle"]
        out["arrangement"] = "cycle_first"
    else:
        out["pos_in_mold"] = out.groupby(["mold", "dimension"]).cumcount() + 1
        out["arrangement"] = "unknown"

    return out[STANDARD_COLUMNS]


def apply_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Turn a raw sheet into the standard schema using an explicit mapping."""
    problems = mapping.validate()
    if problems:
        raise ValueError("；".join(problems))

    if mapping.layout == LAYOUT_LONG:
        return _apply_long(df, mapping)
    return _apply_wide(df, mapping)


def parse_focus_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect the mapping and parse. Raises ValueError when undetectable."""
    mapping = detect_mapping(df)
    if mapping is None:
        raise ValueError(
            "無法自動辨識標題列。請改用「手動對映欄位」指定維度名稱與量測值欄位"
        )
    if mapping.validate():
        raise ValueError(
            "自動辨識的欄位不完整（"
            + "；".join(mapping.validate())
            + "）。請改用「手動對映欄位」"
        )
    return apply_mapping(df, mapping)


# ============================================================================
# File loading
# ============================================================================

def _read_sheets(uploaded_file) -> dict:
    """Read every sheet (or a CSV) as a header-less DataFrame."""
    name = str(getattr(uploaded_file, "name", "") or "").lower()

    if name.endswith(".csv") or name.endswith(".txt"):
        for encoding in ("utf-8-sig", "big5", "cp950", "latin-1"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, encoding=encoding, dtype=object)
                return {"CSV": df}
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 編碼無法辨識（已嘗試 UTF-8 / Big5 / CP950）")

    engine = "xlrd" if name.endswith(".xls") else "openpyxl"
    xl = pd.ExcelFile(uploaded_file, engine=engine)
    return {
        sheet: pd.read_excel(xl, sheet_name=sheet, header=None, engine=engine)
        for sheet in xl.sheet_names
    }


def load_raw_sheets(uploaded_file) -> Tuple[dict, Optional[str]]:
    """Load all sheets untouched, for the manual mapping wizard to preview."""
    try:
        return _read_sheets(uploaded_file), None
    except Exception as exc:
        return {}, f"讀取失敗（{type(exc).__name__}）: {exc}"


def _score_sheet(df: pd.DataFrame) -> int:
    """How parseable a sheet looks: complete mapping beats partial beats none."""
    mapping = detect_mapping(df)
    if mapping is None:
        return 0
    if mapping.validate():
        return 1
    return 3 if mapping.has_spec() else 2


def detect_best_sheet(sheets: dict) -> Tuple[Optional[str], Optional[ColumnMapping]]:
    """Pick the most parseable sheet, preferring a name containing 重點尺寸."""
    for name, df in sheets.items():
        if "重點尺寸" in str(name):
            mapping = detect_mapping(df, sheet_name=name)
            if mapping is not None and not mapping.validate():
                return name, mapping

    best_name, best_score = None, 0
    for name, df in sheets.items():
        score = _score_sheet(df)
        if score > best_score:
            best_name, best_score = name, score

    if best_name is None or best_score < 2:
        return None, None
    return best_name, detect_mapping(sheets[best_name], sheet_name=best_name)


def load_with_mapping(uploaded_file, mapping: ColumnMapping) -> Tuple[pd.DataFrame, Optional[str]]:
    """Parse using a user-confirmed mapping (the wizard / a saved template)."""
    sheets, err = load_raw_sheets(uploaded_file)
    if err:
        return pd.DataFrame(), err

    sheet = mapping.sheet_name
    if sheet not in sheets:
        sheet = next(iter(sheets), None)
    if sheet is None:
        return pd.DataFrame(), "檔案中沒有任何工作表"

    try:
        out = apply_mapping(sheets[sheet], mapping)
    except ValueError as exc:
        return pd.DataFrame(), str(exc)
    except Exception as exc:
        return pd.DataFrame(), f"套用對映失敗（{type(exc).__name__}）: {exc}"

    if out.empty:
        return out, f"工作表「{sheet}」依目前對映沒有解析到任何量測數值"
    return out, None


def load_excel(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    """Auto-detect and parse. Returns (DataFrame, error message)."""
    sheets, err = load_raw_sheets(uploaded_file)
    if err:
        return pd.DataFrame(), err

    sheet_name, mapping = detect_best_sheet(sheets)
    if sheet_name is None or mapping is None:
        names = "、".join(str(s) for s in sheets)
        return pd.DataFrame(), (
            f"無法自動辨識欄位（已檢查工作表：{names}）。"
            "請勾選「手動對映欄位」，指定維度名稱與量測值欄位。"
        )

    try:
        out = apply_mapping(sheets[sheet_name], mapping)
    except ValueError as exc:
        return pd.DataFrame(), f"{exc}"

    if out.empty:
        return out, (
            f"工作表「{sheet_name}」已找到標題列，但沒有解析到任何量測數值。"
            "請確認量測值為數字格式，或改用「手動對映欄位」。"
        )
    return out, None
