"""Grouping logic for dimension data with vectorized operations."""

import numpy as np
import pandas as pd


def format_pos(x) -> str:
    """Format position number as P1, P2, etc."""
    if pd.isna(x):
        return "P?"
    return f"P{int(float(x))}"


def assign_groups_vectorized(
    df: pd.DataFrame, mode: str, file_mold_counts: dict, file_list: list
) -> pd.Series:
    """Assign groups to rows using vectorized operations.

    Args:
        df: DataFrame with 'file' and 'pos_in_mold' columns
        mode: Display mode - "全部合併成一張圖", "強制分檔顯示", or "自動分組"
        file_mold_counts: Dict mapping filename to mold count
        file_list: List of unique filenames

    Returns:
        Series of group labels
    """
    if mode == "全部合併成一張圖":
        return pd.Series("合併", index=df.index)

    # Vectorized check for multi-mold files
    multi_mold = df["file"].map(lambda f: file_mold_counts.get(f, 0) >= 2)

    # Vectorized position formatting
    pos_formatted = df["pos_in_mold"].apply(format_pos)

    if mode == "強制分檔顯示":
        # Multi-mold: "filename Pn", Single-mold: "filename 合併"
        return pd.Series(
            np.where(
                multi_mold,
                df["file"] + " " + pos_formatted,
                df["file"] + " 合併"
            ),
            index=df.index
        )

    # 自動分組 mode
    single_file = len(file_list) <= 1

    # Build conditions and choices
    # Priority: multi_mold -> single_file & not multi_mold -> not multi_mold (multiple files)
    conditions = [
        multi_mold,
        ~multi_mold & single_file,
        ~multi_mold & ~single_file,
    ]

    choices = [
        pos_formatted,
        pd.Series("合併", index=df.index),
        df["file"] + " 合併",
    ]

    # Use numpy select for vectorized conditional assignment
    result = np.select(
        conditions,
        [c.values if isinstance(c, pd.Series) else c for c in choices],
        default="合併"
    )

    return pd.Series(result, index=df.index)
