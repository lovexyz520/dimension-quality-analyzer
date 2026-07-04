"""Statistical calculation functions for dimension quality analysis."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Nelson rules 判讀說明（用於 SPC 控制圖異常模式偵測）
NELSON_RULE_DESCRIPTIONS = {
    "R1": "單點超出 3σ 控制限（異常單點）",
    "R2": "連續 9 點落在中心線同一側（製程平均已偏移）",
    "R3": "連續 6 點持續上升或下降（趨勢：可能為模溫漂移、刀具磨耗等）",
    "R5": "連續 3 點中有 2 點落在同側 2σ 之外（變異增大或偏移前兆）",
}


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
        n = int(values.count())
        cpk_lci, cpk_uci = cpk_confidence_interval(cpk, n)
        rows.append(
            {
                "dimension": dim,
                "count": n,
                "mean": mean,
                "std": std,
                "USL": upper,
                "LSL": lower,
                "Cp": cp,
                "Cpk": cpk,
                "Cpk_LCI": cpk_lci,
                "Cpk_UCI": cpk_uci,
                "low_sample": n < 25,
            }
        )
    return pd.DataFrame(rows)


def cpk_confidence_interval(
    cpk: float, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Approximate confidence interval for Cpk (Bissell's method).

    SE(Cpk) ≈ sqrt(1/(9n) + Cpk²/(2(n-1)))
    """
    if pd.isna(cpk) or n is None or n < 2:
        return np.nan, np.nan
    try:
        from scipy import stats as scipy_stats

        z = float(scipy_stats.norm.ppf(0.5 + confidence / 2))
    except Exception:
        z = 1.96
    se = np.sqrt(1.0 / (9.0 * n) + cpk**2 / (2.0 * (n - 1)))
    return cpk - z * se, cpk + z * se


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
    # Build a sample_id that identifies each *physical* measurement point so
    # that the same part measured for different dimensions is paired together.
    #
    # A physical part is identified by the combination of all available
    # identity columns (file + mold + cavity + cycle + position). Using the
    # finest available key avoids collapsing many rows into one sample via the
    # pivot's aggfunc (the previous mold-only / pos_tag-only logic averaged
    # away most rows, leaving too few samples → an all-NaN, blank heatmap).
    df = df.copy()

    identity_cols = [
        c
        for c in ["file", "mold", "cavity", "cycle", "pos_in_mold", "pos_tag"]
        if c in df.columns and df[c].notna().any()
    ]

    if identity_cols:
        sample_id = df[identity_cols[0]].fillna("").astype(str)
        for c in identity_cols[1:]:
            sample_id = sample_id + "|" + df[c].fillna("").astype(str)
        df["sample_id"] = sample_id
    else:
        df["sample_id"] = df.groupby("dimension").cumcount().astype(str)

    # If the identity key is still not unique per (sample, dimension) — i.e.
    # multiple measurements share the same physical id — disambiguate with a
    # per-key running counter so repeated shots become separate samples
    # instead of being averaged together.
    dup = df.groupby(["sample_id", "dimension"]).cumcount()
    df["sample_id"] = df["sample_id"] + "#" + dup.astype(str)

    # Pivot to wide format: rows = samples, columns = dimensions
    pivot = df.pivot_table(
        index="sample_id",
        columns="dimension",
        values="value",
        aggfunc="mean",
    )

    # Adapt the min_periods floor to how many paired samples actually exist:
    # never require more than are available (but keep a floor of 3, below
    # which a correlation coefficient is not meaningful).
    n_samples = len(pivot)
    effective_min = max(3, min(min_samples, n_samples))

    corr_matrix = pivot.corr(method="pearson", min_periods=effective_min)

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


def check_nelson_rules(values: pd.Series, cl: float, sigma: float) -> pd.DataFrame:
    """Check Nelson rules (R1/R2/R3/R5) on a series of individual values.

    Args:
        values: Measurement values in process order
        cl: Center line (process mean)
        sigma: Estimated process sigma (MR-bar / d2)

    Returns:
        DataFrame with columns: index (1-based), value, rules (list of rule
        codes violated at that point), violated (bool)
    """
    v = pd.Series(values).dropna().astype(float).reset_index(drop=True)
    n = len(v)
    rules: List[List[str]] = [[] for _ in range(n)]

    if n > 0 and pd.notna(sigma) and sigma > 0 and pd.notna(cl):
        # R1: single point beyond 3 sigma
        for i in range(n):
            if abs(v[i] - cl) > 3 * sigma:
                rules[i].append("R1")

        # R2: 9 consecutive points on the same side of CL
        run = 0
        prev_side = 0
        for i in range(n):
            side = np.sign(v[i] - cl)
            if side != 0 and side == prev_side:
                run += 1
            else:
                run = 1 if side != 0 else 0
            prev_side = side
            if run >= 9:
                rules[i].append("R2")

        # R3: 6 consecutive points steadily increasing or decreasing
        run = 0
        prev_dir = 0
        for i in range(1, n):
            direction = np.sign(v[i] - v[i - 1])
            if direction != 0 and direction == prev_dir:
                run += 1
            elif direction != 0:
                run = 1
            else:
                run = 0
            prev_dir = direction
            if run >= 5:  # 5 consecutive diffs = 6 points
                rules[i].append("R3")

        # R5: 2 of 3 consecutive points beyond 2 sigma on the same side
        for i in range(2, n):
            window = v[i - 2 : i + 1]
            above = int(((window - cl) > 2 * sigma).sum())
            below = int(((cl - window) > 2 * sigma).sum())
            if above >= 2 or below >= 2:
                rules[i].append("R5")

    return pd.DataFrame(
        {
            "index": range(1, n + 1),
            "value": v,
            "rules": rules,
            "violated": [len(r) > 0 for r in rules],
        }
    )


def nelson_rules_for_dimension(values) -> Tuple[pd.DataFrame, List[dict]]:
    """Run Nelson rules on one dimension's values using I-MR sigma estimate.

    Returns:
        Tuple of (per-point DataFrame from check_nelson_rules, findings list).
        Each finding: {"rule", "description", "points" (1-based indices)}
    """
    v = pd.Series(values).dropna().astype(float).reset_index(drop=True)
    if len(v) < 2:
        return pd.DataFrame(columns=["index", "value", "rules", "violated"]), []

    d2 = 1.128
    mr_bar = v.diff().abs()[1:].mean()
    sigma = mr_bar / d2 if pd.notna(mr_bar) and mr_bar > 0 else np.nan
    cl = v.mean()

    point_df = check_nelson_rules(v, cl, sigma)
    findings = []
    for rule, desc in NELSON_RULE_DESCRIPTIONS.items():
        pts = point_df.loc[
            point_df["rules"].apply(lambda r: rule in r), "index"
        ].tolist()
        if pts:
            findings.append({"rule": rule, "description": desc, "points": pts})
    return point_df, findings


def normality_test(values, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normality test.

    Cpk 假設數據為常態分布；非常態時 Cpk 估計會失真。

    Returns:
        dict with keys: n, statistic, p_value, is_normal (True/False/None),
        message (診斷說明)
    """
    v = pd.Series(values).dropna().astype(float)
    n = len(v)
    result = {
        "n": n,
        "statistic": np.nan,
        "p_value": np.nan,
        "is_normal": None,
        "message": "",
    }
    if n < 3 or v.nunique() < 3:
        result["message"] = "樣本數不足，無法進行常態性檢定"
        return result
    try:
        from scipy import stats as scipy_stats

        stat, p = scipy_stats.shapiro(v)
    except ImportError:
        result["message"] = "需要安裝 scipy 套件"
        return result
    except Exception as exc:
        result["message"] = f"檢定失敗: {exc}"
        return result

    result["statistic"] = float(stat)
    result["p_value"] = float(p)
    result["is_normal"] = bool(p >= alpha)
    if p < alpha:
        result["message"] = f"非常態分布 (p={p:.3f} < {alpha})，Cpk 數值僅供參考"
    else:
        result["message"] = f"未拒絕常態假設 (p={p:.3f})"
    if n < 15:
        result["message"] += "；樣本數偏少，檢定力有限"
    return result


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose variance into cavity-to-cavity vs cycle-to-cycle components.

    對射出成型的關鍵判斷：穴間差異大 → 修模方向；模次間漂移大 → 調機方向。
    以組間平方和 / 總平方和 (R²) 估計各因子的變異貢獻百分比。

    Returns:
        DataFrame: dimension, cavity_pct, cavity_groups, cycle_pct,
        cycle_groups, judgment
    """

    def _between_pct(group: pd.DataFrame, col: str) -> Tuple[float, int]:
        if col not in group.columns:
            return np.nan, 0
        valid = group[col].notna()
        if valid.sum() < max(4, len(group) * 0.5):
            return np.nan, 0
        g = group.loc[valid]
        vals = g["value"].astype(float)
        grand = vals.mean()
        total_ss = ((vals - grand) ** 2).sum()
        grouped = g.groupby(col)["value"]
        k = grouped.ngroups
        if k < 2 or total_ss <= 0:
            return np.nan, k
        # 每組平均至少 2 筆樣本才有意義：每組僅 1 筆時組間變異
        # 恆為 100%，無法與組內變異區分
        if len(g) / k < 2:
            return np.nan, k
        between_ss = sum(
            len(x) * (x.astype(float).mean() - grand) ** 2 for _, x in grouped
        )
        return float(between_ss / total_ss * 100), k

    rows = []
    for dim, group in df.groupby("dimension"):
        values = group["value"].astype(float)
        if len(values) < 4:
            continue

        cavity_pct, n_cav = _between_pct(group, "cavity")
        cycle_col = (
            "cycle"
            if "cycle" in group.columns and group["cycle"].notna().any()
            else "mold"
        )
        cycle_pct, n_cyc = _between_pct(group, cycle_col)

        if pd.isna(cavity_pct) and pd.isna(cycle_pct):
            judgment = "缺少穴號/模次資訊，無法分解"
        elif pd.notna(cavity_pct) and cavity_pct >= 50 and (
            pd.isna(cycle_pct) or cavity_pct >= cycle_pct
        ):
            judgment = "變異主要來自穴間差異 → 建議往修模/模穴均一性方向檢討"
        elif pd.notna(cycle_pct) and cycle_pct >= 50 and (
            pd.isna(cavity_pct) or cycle_pct > cavity_pct
        ):
            judgment = "變異主要來自模次間漂移 → 建議檢查成型條件穩定性（調機方向）"
        else:
            judgment = "穴間與模次間變異皆不顯著，變異多來自隨機/量測"

        rows.append(
            {
                "dimension": dim,
                "cavity_pct": cavity_pct,
                "cavity_groups": n_cav,
                "cycle_pct": cycle_pct,
                "cycle_groups": n_cyc,
                "judgment": judgment,
            }
        )

    return pd.DataFrame(rows)


def center_offset_suggestion(
    mean: float, upper: float, lower: float, cp: float, cpk: float
) -> Tuple[float, float, str]:
    """Compute process-center offset and a tuning suggestion.

    Returns:
        Tuple of (shift, shift_pct_of_tolerance, suggestion text)
    """
    if pd.isna(upper) or pd.isna(lower) or upper <= lower or pd.isna(mean):
        return np.nan, np.nan, ""

    mid = (upper + lower) / 2
    tol = (upper - lower) / 2
    shift = mean - mid
    shift_pct = shift / tol * 100

    suggestion = ""
    if pd.notna(cpk) and cpk < 1.33:
        if pd.notna(cp) and cp >= 1.33 and abs(shift_pct) >= 10:
            direction = "負向（調低）" if shift > 0 else "正向（調高）"
            suggestion = (
                f"製程能力足夠 (Cp={cp:.2f}) 但中心偏{'高' if shift > 0 else '低'} "
                f"{shift:+.4f}（公差的 {shift_pct:+.0f}%），建議將製程中心往{direction}調整"
            )
        elif pd.notna(cp) and cp < 1.0:
            suggestion = "Cp 亦不足，變異過大，調整中心無法根治，需縮小製程變異"
        elif abs(shift_pct) >= 10:
            direction = "負向（調低）" if shift > 0 else "正向（調高）"
            suggestion = (
                f"中心偏{'高' if shift > 0 else '低'}（公差的 {shift_pct:+.0f}%），"
                f"建議往{direction}調整，並同時關注變異"
            )
    return float(shift), float(shift_pct), suggestion


def detect_systematic_bias(df: pd.DataFrame, centered_pct: float = 15.0) -> dict:
    """Detect whether dimensions are systematically biased to one side.

    對試模最有價值的訊號：多數尺寸一致偏大/偏小，通常代表可用單一製程槓桿
    （保壓、料溫、縮水補償）整體修正，而非逐一修模。

    Args:
        df: 標準長格式 DataFrame
        centered_pct: 偏移絕對值小於此百分比（相對公差）視為「已置中」

    Returns:
        dict: n_total, n_high, n_low, n_centered, dominant, dominant_pct, message
    """
    highs = lows = centered = 0
    total = 0
    for _dim, group in df.groupby("dimension"):
        nominal, upper, lower, _ = pick_spec_values(group)
        if pd.isna(nominal) or pd.isna(upper) or pd.isna(lower):
            continue
        tol = (upper - lower) / 2
        if tol <= 0:
            continue
        total += 1
        off_pct = (group["value"].astype(float).mean() - nominal) / tol * 100
        if abs(off_pct) < centered_pct:
            centered += 1
        elif off_pct > 0:
            highs += 1
        else:
            lows += 1

    result = {
        "n_total": total,
        "n_high": highs,
        "n_low": lows,
        "n_centered": centered,
        "dominant": None,
        "dominant_pct": 0.0,
        "message": "",
    }
    if total == 0:
        result["message"] = "缺少規格資訊，無法評估偏移方向"
        return result

    off_side = highs + lows
    if off_side == 0:
        result["message"] = "多數尺寸已置中，無明顯系統性偏移"
        return result

    if highs >= lows:
        result["dominant"] = "high"
        result["dominant_pct"] = highs / total * 100
    else:
        result["dominant"] = "low"
        result["dominant_pct"] = lows / total * 100

    # 系統性：偏移方向一面倒（同向 >= 偏移尺寸的 70% 且占總數過半）
    dom_count = max(highs, lows)
    if dom_count >= off_side * 0.7 and dom_count >= total * 0.5:
        side_txt = "偏大" if result["dominant"] == "high" else "偏小"
        lever = "降低保壓/縮短保壓時間或檢查縮水補償" if result["dominant"] == "high" \
            else "提高保壓/延長保壓時間、或提高料溫/模溫改善充填"
        result["message"] = (
            f"系統性{side_txt}：{dom_count}/{total} 個尺寸一致{side_txt}。"
            f"建議優先以製程條件整體修正（{lever}），可能一次拉回多個尺寸，"
            "再處理個別殘餘偏差。"
        )
    else:
        result["message"] = (
            f"偏移方向分散（偏大 {highs}、偏小 {lows}、已置中 {centered}），"
            "以個別尺寸補正為主。"
        )
    return result


def cavity_fingerprint_data(
    df: pd.DataFrame, group_col: str = "cavity"
) -> pd.DataFrame:
    """Per-group normalized deviation across every dimension (cavity fingerprint).

    對每個維度，計算各穴（或位置/分組）的平均量測值相對規格中值的標準化偏離%，
    讓「某一穴整體偏大/偏小」的模具指紋一眼可見。

    Args:
        df: 標準長格式 DataFrame
        group_col: 分組欄位，通常為 "cavity"、"pos_in_mold" 或 "group"

    Returns:
        DataFrame: group_val, dimension, deviation_pct, n
    """
    if group_col not in df.columns:
        return pd.DataFrame(columns=["group_val", "dimension", "deviation_pct", "n"])

    rows = []
    for dim, group in df.groupby("dimension"):
        nominal, upper, lower, _ = pick_spec_values(group)
        if pd.isna(nominal) or pd.isna(upper) or pd.isna(lower):
            continue
        tolerance = (upper - lower) / 2
        if tolerance <= 0:
            continue

        valid = group[group[group_col].notna()]
        for gval, sub in valid.groupby(group_col):
            mean_val = sub["value"].astype(float).mean()
            deviation_pct = (mean_val - nominal) / tolerance * 100
            rows.append(
                {
                    "group_val": gval,
                    "dimension": dim,
                    "deviation_pct": deviation_pct,
                    "n": int(len(sub)),
                }
            )

    return pd.DataFrame(rows)


def diagnose_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Build the diagnostic overview: rank dimensions by how much attention they need.

    Combines Cpk rating, out-of-spec counts, Nelson rule violations,
    normality and center-offset suggestions into one table sorted worst-first.
    """
    cpk_df = cpk_with_rating(df)
    cpk_map = cpk_df.set_index("dimension") if not cpk_df.empty else pd.DataFrame()

    rows = []
    for dim, group in df.groupby("dimension"):
        values = group["value"].astype(float)
        nominal, upper, lower, _ = pick_spec_values(group)
        oos = int(calc_out_of_spec(values, lower, upper).sum())

        _, findings = nelson_rules_for_dimension(values)
        nelson_points = sorted({p for f in findings for p in f["points"]})
        nelson_rules = "、".join(f["rule"] for f in findings)

        norm = normality_test(values)

        cp = cpk = np.nan
        rating = "N/A"
        if not cpk_map.empty and dim in cpk_map.index:
            cp = cpk_map.loc[dim, "Cp"]
            cpk = cpk_map.loc[dim, "Cpk"]
            rating = cpk_map.loc[dim, "rating"]

        shift, shift_pct, suggestion = center_offset_suggestion(
            values.mean(), upper, lower, cp, cpk
        )

        priority = 0.0
        if pd.notna(cpk):
            if cpk < 1.0:
                priority += 3
            elif cpk < 1.33:
                priority += 1.5
        if oos > 0:
            priority += 2 + min(oos, 5) * 0.2
        if nelson_points:
            priority += 1
        if norm["is_normal"] is False:
            priority += 0.5

        rows.append(
            {
                "dimension": dim,
                "count": int(values.count()),
                "Cpk": cpk,
                "rating": rating,
                "out_of_spec": oos,
                "nelson_count": len(nelson_points),
                "nelson_rules": nelson_rules,
                "is_normal": norm["is_normal"],
                "shift": shift,
                "shift_pct": shift_pct,
                "suggestion": suggestion,
                "priority": priority,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["priority", "Cpk"], ascending=[False, True]
        ).reset_index(drop=True)
    return result
