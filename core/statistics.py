"""Statistical calculation functions for dimension quality analysis."""

from typing import Tuple

import numpy as np
import pandas as pd


def calc_out_of_spec(values: pd.Series, lower: float, upper: float) -> pd.Series:
    """Calculate which values are out of specification limits."""
    if pd.isna(lower) or pd.isna(upper):
        return pd.Series([False] * len(values), index=values.index)
    return (values < lower) | (values > upper)


def pick_spec_values(sub: pd.DataFrame) -> Tuple[float, float, float, int]:
    """Pick specification values from a subset of data."""
    nominal_list = sub["nominal"].dropna().unique().tolist()
    upper_list = sub["upper"].dropna().unique().tolist()
    lower_list = sub["lower"].dropna().unique().tolist()

    nominal = nominal_list[0] if nominal_list else np.nan
    upper = upper_list[0] if upper_list else np.nan
    lower = lower_list[0] if lower_list else np.nan

    spec_versions = max(len(nominal_list), len(upper_list), len(lower_list))
    return (
        float(nominal) if pd.notna(nominal) else np.nan,
        float(upper) if pd.notna(upper) else np.nan,
        float(lower) if pd.notna(lower) else np.nan,
        spec_versions,
    )


def stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate statistics summary table for all dimensions."""

    def _agg(group):
        values = group["value"].astype(float)
        nominal, upper, lower, spec_versions = pick_spec_values(group)
        out_mask = calc_out_of_spec(values, lower, upper)
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


def cp_cpk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cp and Cpk summary for all dimensions."""
    rows = []
    for dim, group in df.groupby("dimension"):
        values = group["value"].astype(float)
        nominal, upper, lower, _ = pick_spec_values(group)
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


def cpk_with_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cp/Cpk with rating and color information.

    Cpk Rating:
    - >= 1.33: Good (green)
    - 1.0 ~ 1.33: Acceptable (yellow)
    - < 1.0: Poor (red)
    """
    cp_cpk = cp_cpk_summary(df)

    def get_rating(cpk):
        if pd.isna(cpk):
            return "N/A", "gray"
        if cpk >= 1.33:
            return "良好", "green"
        if cpk >= 1.0:
            return "可接受", "yellow"
        return "不良", "red"

    ratings = cp_cpk["Cpk"].apply(get_rating)
    cp_cpk["rating"] = ratings.apply(lambda x: x[0])
    cp_cpk["color"] = ratings.apply(lambda x: x[1])

    return cp_cpk


def imr_spc_points(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate I-MR SPC control chart data."""
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


def calculate_normalized_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate normalized deviation for each measurement.

    Formula:
    deviation% = (measurement - nominal) / tolerance * 100%
    tolerance = (upper - lower) / 2
    """
    rows = []
    for dim, group in df.groupby("dimension"):
        nominal, upper, lower, _ = pick_spec_values(group)

        if pd.isna(nominal) or pd.isna(upper) or pd.isna(lower):
            continue

        tolerance = (upper - lower) / 2
        if tolerance <= 0:
            continue

        for _, row in group.iterrows():
            value = row["value"]
            deviation_pct = (value - nominal) / tolerance * 100

            rows.append(
                {
                    "dimension": dim,
                    "value": value,
                    "nominal": nominal,
                    "upper": upper,
                    "lower": lower,
                    "tolerance": tolerance,
                    "deviation_pct": deviation_pct,
                    "file": row.get("file", ""),
                    "group": row.get("group", ""),
                    "pos_in_mold": row.get("pos_in_mold", np.nan),
                }
            )

    return pd.DataFrame(rows)


def calculate_correlation_matrix(df: pd.DataFrame, min_samples: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate correlation matrix between dimensions.

    Args:
        df: DataFrame with dimension and value columns
        min_samples: Minimum number of paired samples required for correlation

    Returns:
        Tuple of (correlation_matrix, pivot_table)
        - correlation_matrix: Square DataFrame with correlation coefficients
        - pivot_table: Wide-format data used for calculation
    """
    # Create a unique identifier for each measurement point
    # Use file + pos_tag or file + mold as the sample identifier
    df = df.copy()

    # Get file prefix (empty string if no file column)
    file_prefix = df["file"].fillna("") if "file" in df.columns else ""
    if isinstance(file_prefix, str):
        file_prefix = pd.Series([file_prefix] * len(df), index=df.index)

    # Check for valid pos_tag (non-null and non-empty strings)
    has_pos_tag = ("pos_tag" in df.columns and
                   df["pos_tag"].notna().any() and
                   df["pos_tag"].astype(str).str.strip().ne("").any())

    # Check for valid mold column (multiple unique values)
    has_mold = ("mold" in df.columns and
                df["mold"].notna().any() and
                df["mold"].nunique() > 1)

    if has_pos_tag:
        df["sample_id"] = file_prefix + "_" + df["pos_tag"].astype(str)
    elif has_mold:
        df["sample_id"] = file_prefix + "_mold" + df["mold"].astype(str)
    else:
        # Fallback: use row index within each dimension
        df["sample_id"] = file_prefix + "_idx" + df.groupby("dimension").cumcount().astype(str)

    # Pivot to wide format: rows = samples, columns = dimensions
    pivot = df.pivot_table(
        index="sample_id",
        columns="dimension",
        values="value",
        aggfunc="mean"
    )

    # Calculate correlation matrix
    corr_matrix = pivot.corr(method="pearson", min_periods=min_samples)

    return corr_matrix, pivot


def get_high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.7
) -> pd.DataFrame:
    """Extract pairs of dimensions with high correlation.

    Args:
        corr_matrix: Correlation matrix from calculate_correlation_matrix()
        threshold: Minimum absolute correlation to include

    Returns:
        DataFrame with columns: dim1, dim2, correlation, abs_correlation
    """
    pairs = []
    dims = corr_matrix.columns.tolist()

    for i, dim1 in enumerate(dims):
        for j, dim2 in enumerate(dims):
            if i >= j:  # Skip diagonal and lower triangle
                continue
            corr = corr_matrix.loc[dim1, dim2]
            if pd.notna(corr) and abs(corr) >= threshold:
                pairs.append({
                    "dim1": dim1,
                    "dim2": dim2,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                })

    result = pd.DataFrame(pairs)
    if not result.empty:
        result = result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    return result
