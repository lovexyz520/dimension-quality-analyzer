"""Column mapping: 讓任意排版的量測報表對映到標準 schema。

原本的解析把「規格」與「球標」兩個字串寫死，任何欄名變體都會整份檔案讀不進來。
這個模組把「哪一欄是什麼」抽成一份 ColumnMapping，並提供：

1. `detect_mapping()` — 用別名表自動猜（第一層容錯）
2. `ColumnMapping` — 使用者可在 UI 覆寫後套用（第二層對映精靈）

`excel_parser.apply_mapping()` 是唯一的套用點，自動偵測與手動對映共用它。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Tuple
import re

import numpy as np
import pandas as pd


LAYOUT_WIDE = "wide"   # 一列一個尺寸，多欄量測值（傳統檢驗報表）
LAYOUT_LONG = "long"   # 一列一筆量測（CMM / MES / SPC 軟體匯出）

# 掃描標題列時最多往下看幾列
_MAX_HEADER_SCAN = 40


def normalize_label(value: Any) -> str:
    """正規化欄位標籤：去空白、去括號與標點、轉小寫。

    保留 '+' 與 '-'，否則「正公差」與「負公差」的符號寫法會撞在一起。
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip().lower()
    return re.sub(r"[\s　_.,:：、（）()\[\]［］]+", "", text)


# 別名表（值皆為 normalize_label() 後的形式）
DIMENSION_ALIASES = [
    "球標", "項目", "項次", "尺寸", "尺寸編號", "編號", "序號", "名稱",
    "特性", "品質特性", "檢驗項目", "量測項目",
    "dimension", "feature", "characteristic", "char", "item", "no",
]
NOMINAL_ALIASES = [
    "規格", "規格值", "標稱", "標稱尺寸", "基準", "基準值", "設計值",
    "中心值", "中值", "目標值",
    "nominal", "target", "spec", "specification", "basic",
]
PLUS_ALIASES = [
    "正公差", "上公差", "公差上限", "+公差", "公差+", "+tol", "tol+",
    "uppertolerance", "uppertol", "utol",
]
MINUS_ALIASES = [
    "負公差", "下公差", "公差下限", "-公差", "公差-", "-tol", "tol-",
    "lowertolerance", "lowertol", "ltol",
]
UPPER_ALIASES = [
    "上限", "規格上限", "上限值", "最大值", "最大", "usl", "max", "maximum",
    "upperlimit", "upperspec",
]
LOWER_ALIASES = [
    "下限", "規格下限", "下限值", "最小值", "最小", "lsl", "min", "minimum",
    "lowerlimit", "lowerspec",
]
METHOD_ALIASES = [
    "量測方式", "檢驗方式", "量測方法", "量具", "檢具", "備註", "判定",
    "method", "gauge", "gage", "instrument", "remark", "note", "judgement",
]
VALUE_ALIASES = [
    "量測值", "實測值", "測定值", "量測結果", "數值", "實際值",
    "value", "measurement", "measured", "actual", "result", "reading",
]
CAVITY_ALIASES = ["穴號", "穴", "模穴", "cavity", "cav"]
CYCLE_ALIASES = ["模次", "模數", "cycle", "shot"]
MOLD_ALIASES = ["模具", "組別", "分組", "群組", "mold", "mould", "group"]


def _contains_safe(alias: str) -> bool:
    """短的 ASCII 別名（no、cav、max…）只做精確比對，避免子字串誤判。

    例如 'no' 會出現在 'nominal' 裡，'min' 會出現在 'minimum' 裡。
    中文別名兩字即可安全做子字串比對。
    """
    return len(alias) >= 4 or not alias.isascii()


def _match_col(
    header: pd.Series, aliases: list[str], used: Optional[set] = None
) -> Optional[int]:
    """在標題列中找出符合別名的欄位索引；先精確、後子字串。"""
    used = used or set()
    norm = {idx: normalize_label(v) for idx, v in header.items()}

    for idx, text in norm.items():
        if idx in used or not text:
            continue
        if text in aliases:
            return idx

    for idx, text in norm.items():
        if idx in used or not text:
            continue
        for alias in aliases:
            if _contains_safe(alias) and alias in text:
                return idx
    return None


_ALIAS_GROUPS = [
    DIMENSION_ALIASES, NOMINAL_ALIASES, PLUS_ALIASES, MINUS_ALIASES,
    UPPER_ALIASES, LOWER_ALIASES, VALUE_ALIASES, METHOD_ALIASES,
]


def _header_score(row: pd.Series) -> int:
    """一列命中幾個不同的別名類別 —— 越多越像標題列。"""
    return sum(1 for group in _ALIAS_GROUPS if _match_col(row, group) is not None)


def detect_header_row(df: pd.DataFrame) -> Optional[int]:
    """找出最像標題列的那一列（至少命中兩個別名類別）。"""
    best_row, best_score = None, 0
    for i in range(min(len(df), _MAX_HEADER_SCAN)):
        score = _header_score(df.iloc[i])
        if score > best_score:
            best_row, best_score = i, score
    return best_row if best_score >= 2 else None


def _numeric_ratio(df: pd.DataFrame, row: int, cols: list[int]) -> float:
    """該列在指定欄位中有多少比例是數字。"""
    if not cols:
        return 0.0
    values = pd.to_numeric(df.iloc[row, cols], errors="coerce")
    return float(values.notna().mean())


@dataclass
class ColumnMapping:
    """描述一份原始表格要如何轉成標準 schema。可序列化成 JSON 重複使用。"""

    layout: str = LAYOUT_WIDE
    sheet_name: Optional[str] = None
    header_row: int = 0
    data_start_row: int = 1
    # wide：量測欄標籤所在列（穴號/模次標籤），None 表示沒有
    label_row: Optional[int] = None
    # 維度名稱欄；多欄時以空白串接（沿用「第一欄前綴 + 球標」的行為）
    dimension_cols: list[int] = field(default_factory=list)
    nominal_col: Optional[int] = None
    plus_col: Optional[int] = None
    minus_col: Optional[int] = None
    upper_col: Optional[int] = None
    lower_col: Optional[int] = None
    # wide 專用
    meas_cols: list[int] = field(default_factory=list)
    # long 專用
    value_col: Optional[int] = None
    cavity_col: Optional[int] = None
    cycle_col: Optional[int] = None
    mold_col: Optional[int] = None

    def has_spec(self) -> bool:
        """是否足以推導出規格上下限。"""
        if self.upper_col is not None and self.lower_col is not None:
            return True
        return self.nominal_col is not None and (
            self.plus_col is not None or self.minus_col is not None
        )

    def validate(self) -> list[str]:
        """回傳阻擋解析的問題清單；空清單代表可以套用。"""
        problems = []
        if not self.dimension_cols:
            problems.append("尚未指定「維度名稱」欄位")
        if self.layout == LAYOUT_WIDE:
            if not self.meas_cols:
                problems.append("尚未指定任何「量測值」欄位")
        elif self.value_col is None:
            problems.append("尚未指定「量測值」欄位")
        if self.data_start_row <= self.header_row:
            problems.append("資料起始列必須在標題列之後")
        return problems

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnMapping":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


def _detect_long(header: pd.Series, mapping: ColumnMapping) -> ColumnMapping:
    used: set = set()
    dim = _match_col(header, DIMENSION_ALIASES, used)
    if dim is not None:
        used.add(dim)
        mapping.dimension_cols = [dim]

    for attr, aliases in [
        ("value_col", VALUE_ALIASES),
        ("upper_col", UPPER_ALIASES),
        ("lower_col", LOWER_ALIASES),
        ("plus_col", PLUS_ALIASES),
        ("minus_col", MINUS_ALIASES),
        ("nominal_col", NOMINAL_ALIASES),
        ("cavity_col", CAVITY_ALIASES),
        ("cycle_col", CYCLE_ALIASES),
        ("mold_col", MOLD_ALIASES),
    ]:
        col = _match_col(header, aliases, used)
        if col is not None:
            used.add(col)
            setattr(mapping, attr, col)
    return mapping


def _detect_meas_cols(
    df: pd.DataFrame, header_row: int, used: set, method_col: Optional[int]
) -> list[int]:
    """量測欄 = 所有已對映欄位的右側、量測方式欄左側。

    起點必須是「已對映欄位的右界」而非維度名欄：品名欄若排在規格欄左邊，
    規格/上下限就會被誤讀成量測值。
    沒有「量測方式」欄時，改用「資料區含數值」判定，而不是吃到表格尾巴。
    """
    start_col = max(used) + 1 if used else 0
    has_method = method_col is not None and method_col > start_col
    end_col = method_col if has_method else df.shape[1]

    candidates = [
        c for c in range(start_col, min(end_col, df.shape[1])) if c not in used
    ]
    if has_method:
        return candidates

    body = df.iloc[header_row + 1 :]
    return [
        col for col in candidates
        if pd.to_numeric(body.iloc[:, col], errors="coerce").notna().any()
    ]


def infer_rows(
    df: pd.DataFrame, header_row: int, dimension_cols: list[int], meas_cols: list[int]
) -> Tuple[int, Optional[int]]:
    """由標題列推導 (資料起始列, 量測欄標籤列)。

    資料起始列 = 標題列下方第一列「維度名稱有值 且 量測欄含數字」。
    其上若還隔了一列，那是量測欄的標籤列（穴號/模次）。
    """
    for row in range(header_row + 1, min(len(df), header_row + 6)):
        has_name = any(
            str(df.iloc[row, c]).strip() not in ("", "nan", "None")
            for c in dimension_cols
            if c < df.shape[1]
        )
        if has_name and _numeric_ratio(df, row, meas_cols) > 0:
            return row, (row - 1 if row > header_row + 1 else None)

    # 找不到就退回原本的固定假設（標題列 +2）
    return header_row + 2, header_row + 1


def _detect_wide(df: pd.DataFrame, header_row: int, mapping: ColumnMapping) -> ColumnMapping:
    header = df.iloc[header_row]
    used: set = set()

    # 順序重要：先抓具體的上下限，「規格上限」才不會被「規格」搶走。
    for attr, aliases in [
        ("upper_col", UPPER_ALIASES),
        ("lower_col", LOWER_ALIASES),
        ("plus_col", PLUS_ALIASES),
        ("minus_col", MINUS_ALIASES),
        ("nominal_col", NOMINAL_ALIASES),
    ]:
        col = _match_col(header, aliases, used)
        if col is not None:
            used.add(col)
            setattr(mapping, attr, col)

    label_col = _match_col(header, DIMENSION_ALIASES, used)
    if label_col is None:
        return mapping
    used.add(label_col)

    # 沿用原行為：第一欄常是分類前綴（如「外觀」），與球標串接成完整維度名。
    mapping.dimension_cols = [0, label_col] if label_col > 0 else [label_col]

    method_col = _match_col(header, METHOD_ALIASES, used)
    mapping.meas_cols = _detect_meas_cols(df, header_row, used, method_col)

    mapping.data_start_row, mapping.label_row = infer_rows(
        df, header_row, mapping.dimension_cols, mapping.meas_cols
    )
    return mapping


def detect_mapping(df: pd.DataFrame, sheet_name: Optional[str] = None) -> Optional[ColumnMapping]:
    """自動猜測欄位對映；完全認不出標題列時回傳 None。"""
    header_row = detect_header_row(df)
    if header_row is None:
        return None

    header = df.iloc[header_row]
    mapping = ColumnMapping(header_row=header_row, sheet_name=sheet_name)

    is_long = (
        _match_col(header, VALUE_ALIASES) is not None
        and _match_col(header, DIMENSION_ALIASES) is not None
    )
    if is_long:
        mapping.layout = LAYOUT_LONG
        mapping.data_start_row = header_row + 1
        return _detect_long(header, mapping)

    return _detect_wide(df, header_row, mapping)
