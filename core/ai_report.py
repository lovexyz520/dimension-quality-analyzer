"""AI-assisted report generation.

設計原則：AI 只是「解讀層」——所有統計數字都由確定性的程式（core.statistics）
先算好，這裡只把結構化結果整理成 prompt，交給大型語言模型寫成人類可讀的報告。
模型被嚴格限制「只能根據提供的數字撰寫，不得自行推算或捏造」，以壓低幻覺風險。

本模組不匯入 streamlit，保持可獨立測試；API 金鑰由呼叫端傳入。
"""

from typing import Optional

import pandas as pd

from .statistics import (
    diagnose_overview,
    variance_decomposition,
    detect_systematic_bias,
)

# ---------------------------------------------------------------------------
# 分析情境（階段）：只改變「解讀框架與報告重點」，不改變底層統計計算。
# ---------------------------------------------------------------------------
STAGE_TRYOUT = "試模 (T0/T1/T2)"
STAGE_FIRST_ARTICLE = "量產首件"
STAGE_MONITORING = "量產監控"
STAGE_LABELS = [STAGE_TRYOUT, STAGE_FIRST_ARTICLE, STAGE_MONITORING]
DEFAULT_STAGE = STAGE_FIRST_ARTICLE

# 各階段給模型的重點指引（附加在系統指令後）
_STAGE_GUIDANCE = {
    STAGE_TRYOUT: (
        "【分析情境：射出試模 T0/T1/T2】此階段模具與成型條件尚未定案，"
        "尺寸偏移是正常且預期的，目的是決定如何修模與調機。分析重點依序為："
        "(1) 各尺寸中心偏移的『方向與量』，作為補正依據；"
        "(2) 系統性偏移（多數尺寸同向偏移）優先以製程條件（保壓、料溫、模溫、縮水補償）整體修正，而非逐一修模；"
        "(3) 穴／位置間平衡（某穴獨偏→修該穴模仁；全穴同偏→調製程）；"
        "(4) 提醒鋼料安全邊觀念（偏小常為刻意留料、待加工修正的安全方向，需與模具端確認）。"
        "請【務必淡化】Cpk、SPC 控制圖與常態性判定——此階段製程未穩、樣本少，這些指標僅供參考，"
        "不得據此下良/不良判定。用語請以『待補正／待修正』取代『不良』。"
    ),
    STAGE_FIRST_ARTICLE: (
        "【分析情境：量產首件】模具與條件應已定案，重點在確認是否符合規格與製程能力（Cpk）。"
        "超規格屬異常警訊，需追查原因。"
    ),
    STAGE_MONITORING: (
        "【分析情境：量產監控】重點在製程穩定性：SPC 控制圖與 Nelson 規則異常模式、"
        "跨批趨勢、製程能力（Cpk/Ppk）是否維持。偏移與異常點視為製程漂移的警訊。"
    ),
}

# 各階段的報告段落結構
_STAGE_SECTIONS = {
    STAGE_TRYOUT: (
        "1. 【整體收斂狀況】：本批距離規格的整體狀況、變異是否夠緊（模具體質）。\n"
        "2. 【全域製程調整建議】：若有系統性偏移，說明優先調整的製程條件。\n"
        "3. 【個別尺寸補正清單】：列出需個別補正的尺寸，明確標示偏移方向與量。\n"
        "4. 【穴／位置平衡】：是否有特定穴／位置獨自偏移。\n"
        "5. 【待確認事項】：殘料類量測、量測系統、樣本數等提醒。"
    ),
    STAGE_FIRST_ARTICLE: (
        "1. 【整體總結】。\n"
        "2. 【優先處理】：依嚴重度列出最需處理的尺寸。\n"
        "3. 【可能原因與建議措施】：結合變異分解與調機建議給出檢查方向。\n"
        "4. 【製程整體觀察】：常態性、樣本數等提醒。"
    ),
    STAGE_MONITORING: (
        "1. 【製程穩定性】：SPC／Nelson 異常模式整體評估。\n"
        "2. 【異常尺寸】：列出出現失控或異常模式的尺寸。\n"
        "3. 【能力維持】：Cpk 是否維持在目標水準。\n"
        "4. 【趨勢與提醒】。"
    ),
}

# 預設使用 Google 免費的 Gemini Flash 系列（可於 UI 覆寫）
# gemini-3.5-flash：2026-05 發表的最新 Flash（gemini-2.0-flash 已於 2026-06 停用）
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
# 供 UI 下拉選單使用的可選模型（皆為 Flash 級、含免費額度）
AVAILABLE_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

SYSTEM_INSTRUCTION_BASE = (
    "你是一位製造業（射出成型）尺寸品質分析助手。"
    "你只能根據使用者提供的『已計算好的統計數據』撰寫分析報告，"
    "嚴禁自行推算、猜測或捏造任何數字；若資料未提供某項資訊，就明說『資料未提供』。"
    "所有建議請以『建議檢查方向』的語氣表達，不得下最終良/不良判定。"
    "一律使用繁體中文，語氣專業、精簡、條理分明。"
)
# 向後相容別名
SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION_BASE

REPORT_PROMPT_TEMPLATE = """以下是本批量測資料經統計程式計算後的結果（分析情境：{stage}），請據此撰寫一份品質分析報告。

報告請包含以下段落：
{sections}

請勿加入資料中沒有的數字。

=== 已計算的分析數據 ===
{payload}
"""


def _system_instruction_for(stage: str) -> str:
    guidance = _STAGE_GUIDANCE.get(stage, "")
    return SYSTEM_INSTRUCTION_BASE + ("\n\n" + guidance if guidance else "")


def build_analysis_payload(df: pd.DataFrame, stage: str = DEFAULT_STAGE) -> dict:
    """Assemble a compact, deterministic summary of the computed analysis.

    只包含程式算好的數字，作為 AI 報告與離線文字摘要的共同資料來源。
    stage 只影響摘要中附帶的解讀提示（如系統性偏移），不改變任何統計值。
    """
    overview = diagnose_overview(df)
    variance = variance_decomposition(df)
    var_map = variance.set_index("dimension") if not variance.empty else pd.DataFrame()

    total = len(overview)
    n_bad = int((overview["rating"] == "不良").sum()) if not overview.empty else 0
    n_ok_rating = int((overview["rating"] == "可接受").sum()) if not overview.empty else 0
    n_good = int((overview["rating"] == "良好").sum()) if not overview.empty else 0
    n_oos_points = int(overview["out_of_spec"].sum()) if not overview.empty else 0
    n_nelson_dims = int((overview["nelson_count"] > 0).sum()) if not overview.empty else 0

    problem_dims = []
    if not overview.empty:
        for _, row in overview[overview["priority"] > 0].iterrows():
            dim = row["dimension"]
            entry = {
                "dimension": dim,
                "count": int(row["count"]),
                "Cpk": None if pd.isna(row["Cpk"]) else round(float(row["Cpk"]), 3),
                "rating": row["rating"],
                "out_of_spec": int(row["out_of_spec"]),
                "nelson_count": int(row["nelson_count"]),
                "nelson_rules": row["nelson_rules"] or "",
                "is_normal": row["is_normal"],
                "shift_pct": None if pd.isna(row["shift_pct"]) else round(float(row["shift_pct"]), 1),
                "suggestion": row["suggestion"] or "",
            }
            if not var_map.empty and dim in var_map.index:
                v = var_map.loc[dim]
                entry["cavity_pct"] = None if pd.isna(v["cavity_pct"]) else round(float(v["cavity_pct"]), 0)
                entry["cycle_pct"] = None if pd.isna(v["cycle_pct"]) else round(float(v["cycle_pct"]), 0)
                entry["variance_judgment"] = v["judgment"]
            problem_dims.append(entry)

    bias = detect_systematic_bias(df)

    return {
        "stage": stage,
        "summary": {
            "total_dimensions": total,
            "good": n_good,
            "acceptable": n_ok_rating,
            "poor": n_bad,
            "out_of_spec_points": n_oos_points,
            "spc_abnormal_dimensions": n_nelson_dims,
        },
        "systematic_bias": bias,
        "problem_dimensions": problem_dims,
    }


def format_payload_as_text(payload: dict) -> str:
    """Render the payload as a readable Traditional-Chinese text summary.

    這同時是「免 API 的結構化報告」——沒有金鑰或網路時可直接使用。
    """
    s = payload["summary"]
    stage = payload.get("stage", DEFAULT_STAGE)
    lines = [
        f"【分析情境】{stage}",
        "",
        "【整體統計】",
        f"- 維度總數：{s['total_dimensions']}（良好 {s['good']}、可接受 {s['acceptable']}、不良 {s['poor']}）",
        f"- 超規格點總數：{s['out_of_spec_points']}",
        f"- SPC 異常維度數：{s['spc_abnormal_dimensions']}",
    ]

    bias = payload.get("systematic_bias")
    if bias and bias.get("message"):
        lines.append(
            f"- 偏移分布：偏大 {bias['n_high']}、偏小 {bias['n_low']}、"
            f"已置中 {bias['n_centered']}（共 {bias['n_total']}）"
        )
        lines.append(f"- 偏移研判：{bias['message']}")
    lines.append("")

    problems = payload["problem_dimensions"]
    if not problems:
        lines.append("【需關注維度】無 —— 所有維度 Cpk 達標、無超規格點、無 SPC 異常模式。")
        return "\n".join(lines)

    lines.append("【需關注維度（依嚴重度排序）】")
    for i, d in enumerate(problems, 1):
        cpk_txt = "N/A" if d["Cpk"] is None else f"{d['Cpk']:.3f}"
        norm_txt = {True: "常態", False: "非常態", None: "未檢定"}.get(d["is_normal"], "未檢定")
        lines.append(
            f"{i}. {d['dimension']}｜Cpk={cpk_txt}（{d['rating']}）｜"
            f"超規格 {d['out_of_spec']} 點｜SPC 異常 {d['nelson_count']} 點"
            f"{('（' + d['nelson_rules'] + '）') if d['nelson_rules'] else ''}｜{norm_txt}｜n={d['count']}"
        )
        if d.get("variance_judgment"):
            cav = "—" if d.get("cavity_pct") is None else f"{d['cavity_pct']:.0f}%"
            cyc = "—" if d.get("cycle_pct") is None else f"{d['cycle_pct']:.0f}%"
            lines.append(f"   變異分解：穴間 {cav}、模次間 {cyc} → {d['variance_judgment']}")
        if d["suggestion"]:
            lines.append(f"   調機建議：{d['suggestion']}")
    return "\n".join(lines)


def generate_ai_report(
    payload_text: str,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    stage: str = DEFAULT_STAGE,
    timeout: int = 90,
) -> str:
    """Call the Gemini API to turn the computed summary into a narrative report.

    Args:
        payload_text: 由 format_payload_as_text() 產生的結構化摘要
        api_key: Google AI Studio API 金鑰
        model: Gemini 模型名稱（預設 gemini-3.5-flash）
        stage: 分析情境（試模／量產首件／量產監控），決定報告重點與語氣

    Returns:
        報告文字

    Raises:
        RuntimeError: 金鑰缺失、套件缺失、API 錯誤或回應被安全機制攔截時
    """
    if not api_key:
        raise RuntimeError("尚未提供 API 金鑰")

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 requests 套件，請執行 pip install requests") from exc

    sections = _STAGE_SECTIONS.get(stage, _STAGE_SECTIONS[DEFAULT_STAGE])
    prompt = REPORT_PROMPT_TEMPLATE.format(
        stage=stage, sections=sections, payload=payload_text
    )

    url = GEMINI_ENDPOINT.format(model=model)
    body = {
        "systemInstruction": {"parts": [{"text": _system_instruction_for(stage)}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            # 需給足額度：Gemini Flash 為推理模型，思考 token 也計入 maxOutputTokens，
            # 若太低會被思考吃光導致正文截斷（finishReason=MAX_TOKENS）。
            "maxOutputTokens": 8192,
            "topP": 0.9,
            # 本任務僅需依已算好的數字撰寫報告，不需深度推理；關閉思考
            # 可避免截斷、並讓速度與費用最省。
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        resp = requests.post(
            url,
            params={"key": api_key},
            json=body,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"連線失敗：{exc}") from exc

    if resp.status_code != 200:
        # 盡量給出可讀的錯誤訊息，但不外洩金鑰
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:300]
        if resp.status_code == 400 and "API key" in detail:
            raise RuntimeError("API 金鑰無效，請確認金鑰正確")
        if resp.status_code == 429:
            raise RuntimeError("已達免費額度上限（rate limit），請稍後再試")
        if resp.status_code == 404:
            raise RuntimeError(f"找不到模型「{model}」，請確認模型名稱正確")
        raise RuntimeError(f"API 錯誤 (HTTP {resp.status_code})：{detail}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        block = data.get("promptFeedback", {}).get("blockReason")
        if block:
            raise RuntimeError(f"回應被安全機制攔截（{block}）")
        raise RuntimeError("模型未回傳任何內容")

    finish = candidates[0].get("finishReason", "")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError(
            f"模型回傳空內容（finishReason={finish}）；"
            "若為 MAX_TOKENS 代表報告過長被截斷，請減少維度數或分批分析"
        )
    if finish == "MAX_TOKENS":
        text += "\n\n> ⚠️ 註：報告可能因長度上限被截斷（維度數過多），以上為部分內容。"
    return text


def resolve_report(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    stage: str = DEFAULT_STAGE,
) -> tuple:
    """Convenience wrapper: build payload, return (offline_text, ai_text_or_None, error).

    - offline_text 一定會有（免 API 的結構化摘要）
    - 若提供 api_key，會嘗試產生 AI 報告；失敗時 ai_text=None 且 error 有訊息
    """
    payload = build_analysis_payload(df, stage)
    offline_text = format_payload_as_text(payload)

    if not api_key:
        return offline_text, None, None

    try:
        ai_text = generate_ai_report(offline_text, api_key, model, stage)
        return offline_text, ai_text, None
    except RuntimeError as exc:
        return offline_text, None, str(exc)
