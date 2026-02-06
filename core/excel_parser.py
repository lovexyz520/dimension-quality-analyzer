"""Excel parsing functions for dimension quality data."""

from typing import Optional, Tuple
import re

import numpy as np
import pandas as pd


def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    """Find the row containing '規格' and '球標' headers."""
    for i in range(len(df)):
        row = df.iloc[i].astype(str)
        if row.str.contains("規格", na=False).any() and row.str.contains("球標", na=False).any():
            return i
    return None


def _find_col_index(header_row: pd.Series, label: str) -> Optional[int]:
    """Find column index for a given label in header row."""
    for idx, val in header_row.items():
        if str(val).strip() == label:
            return idx
    return None


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
    if match:
        return int(match.group(1))
    return None


def _parse_cycle_from_mold_label(label: str) -> Optional[int]:
    """Parse cycle from mold header like '第一模'."""
    if not label:
        return None
    match = re.search(r"第\s*(\d+)\s*模", label)
    if match:
        return int(match.group(1))
    return None


def _detect_arrangement_from_mold_labels(mold_labels: dict) -> str:
    """Detect data arrangement type from mold labels.

    Args:
        mold_labels: Dict mapping column index to mold label string

    Returns:
        "cavity_first" - if labels contain CAV.X pattern (同穴不同模次)
        "cycle_first" - if labels contain 第X模 pattern (同模次不同穴號)
        "unknown" - if pattern cannot be determined
    """
    labels = set(mold_labels.values())

    has_cav_pattern = False
    has_cycle_pattern = False

    for label in labels:
        if not label:
            continue
        # Check for CAV.X, CAV X, Cav.X patterns (case insensitive)
        if re.search(r"(?i)cav[\.\s]*\d+", label):
            has_cav_pattern = True
        # Check for 第X模 pattern
        if re.search(r"第\s*\d+\s*模", label):
            has_cycle_pattern = True

    if has_cav_pattern and not has_cycle_pattern:
        return "cavity_first"
    elif has_cycle_pattern and not has_cav_pattern:
        return "cycle_first"
    elif has_cav_pattern and has_cycle_pattern:
        # Both patterns present - default to cycle_first (original behavior)
        return "cycle_first"
    else:
        return "unknown"


def _parse_cavity_from_cav_label(label: str) -> Optional[int]:
    """Parse cavity number from CAV.X pattern."""
    if not label:
        return None
    match = re.search(r"(?i)cav[\.\s]*(\d+)", label)
    if match:
        return int(match.group(1))
    return None

def _detect_focus_sheet(xl: pd.ExcelFile) -> Tuple[Optional[str], Optional[str]]:
    """Detect the sheet containing focus dimension data."""
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


def _extract_mold_and_pos(
    df: pd.DataFrame, header_row: int, meas_cols: list
) -> Tuple[dict, dict, dict, dict, dict, dict, str]:
    """Extract mold labels, measurement labels, and position information.

    Returns:
        Tuple of (mold_labels, meas_labels, cavities, cycles, pos_raw, pos_in_mold, arrangement)
        arrangement is one of: "cavity_first", "cycle_first", "unknown"
    """
    header = df.iloc[header_row]
    # Mold labels are in the header row; forward-fill across measurement columns.
    mold_labels = {}
    current_label = ""
    for col in meas_cols:
        label = _clean_cell(header.iloc[col])
        if label:
            current_label = label
        mold_labels[col] = current_label

    # Detect arrangement type from mold labels
    arrangement = _detect_arrangement_from_mold_labels(mold_labels)

    # Position row is usually the next row under the header.
    pos_row = df.iloc[header_row + 1] if header_row + 1 < len(df) else None
    meas_labels = {}
    cavities = {}
    cycles = {}
    pos_raw = {}
    pos_in_mold = {}
    mold_counts = {}
    for col in meas_cols:
        raw_val = np.nan
        if pos_row is not None:
            raw_val = pd.to_numeric(pos_row.iloc[col], errors="coerce")
        label = _clean_label(pos_row.iloc[col]) if pos_row is not None else ""
        meas_labels[col] = label

        cav, cyc = _parse_cavity_cycle_from_label(label)
        if cav is None:
            cav = _parse_cavity_from_mold_label(mold_labels.get(col, ""))
        # Also try to parse cavity from CAV.X pattern
        if cav is None:
            cav = _parse_cavity_from_cav_label(mold_labels.get(col, ""))
        if cyc is None:
            cyc = _parse_cycle_from_mold_label(mold_labels.get(col, ""))
        cavities[col] = cav if cav is not None else np.nan
        cycles[col] = cyc if cyc is not None else np.nan
        pos_raw[col] = raw_val

        mold = mold_labels.get(col, "")
        mold_counts[mold] = mold_counts.get(mold, 0) + 1
        pos_in_mold[col] = mold_counts[mold]

    return mold_labels, meas_labels, cavities, cycles, pos_raw, pos_in_mold, arrangement


def parse_focus_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse focus dimensions from Excel DataFrame."""
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

    mold_labels, meas_labels, cavities, cycles, pos_raw, pos_in_mold, arrangement = _extract_mold_and_pos(
        df, header_row, meas_cols
    )

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

        for col, v in zip(meas_cols, pd.to_numeric(meas.loc[idx], errors="coerce").tolist()):
            if pd.isna(v):
                continue
            rows.append(
                {
                    "dimension": dim_label,
                    "value": float(v),
                    "nominal": float(nominal) if pd.notna(nominal) else np.nan,
                    "upper": float(upper) if pd.notna(upper) else np.nan,
                    "lower": float(lower) if pd.notna(lower) else np.nan,
                    "mold": mold_labels.get(col, ""),
                    "meas_label": meas_labels.get(col, ""),
                    "pos_tag": meas_labels.get(col, ""),
                    "cavity": float(cavities.get(col)) if pd.notna(cavities.get(col, np.nan)) else np.nan,
                    "cycle": float(cycles.get(col)) if pd.notna(cycles.get(col, np.nan)) else np.nan,
                    "pos_raw": float(pos_raw.get(col)) if pd.notna(pos_raw.get(col, np.nan)) else np.nan,
                    "pos_in_mold": float(pos_in_mold.get(col)) if pd.notna(pos_in_mold.get(col, np.nan)) else np.nan,
                    "arrangement": arrangement,
                }
            )

    return pd.DataFrame(rows)


def load_excel(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load and parse Excel file, returning DataFrame and optional error message."""
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
