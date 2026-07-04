# CLAUDE.md

本文件提供 Claude Code 開發本專案時的指引與背景資訊。

## 專案概述

Dimension Quality Analyzer 是一個製造業品質分析 Web 應用程式，用於分析零件尺寸量測數據，生成盒鬚圖與統計報表。

## 技術棧

- **Python 3.10+**
- **Streamlit** - Web 應用框架
- **Pandas** - 數據處理
- **Plotly** - 互動式圖表
- **openpyxl** - Excel 讀寫
- **NumPy** - 數值計算
- **SciPy** - 統計檢定（Shapiro-Wilk 常態性檢定、常態分位數）
- **Kaleido** - Plotly 圖表轉 PNG
- **fpdf2** - PDF 報表產生

## 專案結構

```
dimension-quality-analyzer/
├── streamlit_app.py          # UI 主程式
├── core/                     # 核心模組
│   ├── __init__.py           # 模組匯出
│   ├── excel_parser.py       # Excel 解析
│   ├── statistics.py         # 統計計算
│   ├── visualization.py      # 圖表繪製
│   ├── export.py             # 匯出功能
│   └── grouping.py           # 分組邏輯
├── requirements.txt          # Python 依賴
├── pyproject.toml            # 專案配置（UV 套件管理）
├── .streamlit/
│   └── config.toml           # Streamlit 配置
└── sample_data/
    └── template.xlsx         # Excel 範本
```

## 模組架構

### core/excel_parser.py
| 函數 | 功能 |
|------|------|
| `_find_header_row()` | 尋找含「規格」與「球標」的標題列 |
| `_find_col_index()` | 尋找指定標籤的欄位索引 |
| `_clean_cell()` | 清理儲存格值 |
| `_detect_focus_sheet()` | 自動偵測「重點尺寸」工作表 |
| `_detect_arrangement_from_mold_labels()` | 偵測資料排列方式 (CAV.X / 第X模) |
| `_parse_cavity_from_cav_label()` | 從 CAV.X 格式解析穴號 |
| `_extract_mold_and_pos()` | 提取模次與位置資訊 |
| `parse_focus_dimensions()` | 解析 Excel 數據為標準格式 |
| `load_excel()` | Excel 檔案載入入口 |

### core/statistics.py
| 函數 | 功能 |
|------|------|
| `calc_out_of_spec()` | 計算超規格點 |
| `pick_spec_values()` | 取得規格值 |
| `stats_table()` | 計算統計摘要 |
| `cp_cpk_summary()` | 計算 Cp/Cpk 製程能力 |
| `cpk_with_rating()` | Cpk 加上評級與顏色 |
| `cpk_confidence_interval()` | 計算 Cpk 95% 信賴區間 (Bissell 法) |
| `imr_spc_points()` | 計算 I-MR 控制圖數據 |
| `calculate_normalized_deviation()` | 計算標準化偏離 |
| `calculate_correlation_matrix()` | 計算維度間相關性矩陣 |
| `get_high_correlation_pairs()` | 取得高相關維度對 |
| `check_nelson_rules()` | Nelson 規則檢查 (R1/R2/R3/R5) |
| `nelson_rules_for_dimension()` | 單一維度的 Nelson 規則判讀 |
| `normality_test()` | Shapiro-Wilk 常態性檢定 |
| `variance_decomposition()` | 穴間 vs 模次間變異分解 |
| `center_offset_suggestion()` | 中心偏移量與調機建議 |
| `diagnose_overview()` | 診斷總覽彙總表（依嚴重度排序） |
| `cavity_fingerprint_data()` | 各穴/位置在各維度的標準化偏離（指紋圖資料） |
| `detect_systematic_bias()` | 偵測多數尺寸同向偏移（試模用，指向製程槓桿） |

### core/visualization.py
| 函數 | 功能 |
|------|------|
| `add_spec_lines()` | 新增規格線標註 |
| `add_spec_band()` | 規格區間 (LSL~USL) 底色帶 |
| `build_fig()` | 建立 Plotly 盒鬚圖（含 hover 溯源、平均值標記、n 標籤） |
| `apply_y_range()` | 套用 Y 軸範圍 |
| `add_spec_edge_markers()` | 新增規格邊界標記 |
| `add_out_of_spec_points()` | 標記超規格點（與資料點對位、hover 溯源） |
| `build_cpk_heatmap()` | 建立 Cpk 熱力圖 |
| `build_normalized_deviation_chart()` | 建立標準化偏離圖 |
| `build_position_comparison_chart()` | 建立模次比較圖 |
| `build_imr_chart()` | 建立 I-MR SPC 控制圖（可標記 Nelson 異常點） |
| `build_correlation_heatmap()` | 建立相關性熱力圖 |
| `build_correlation_scatter()` | 建立相關性散佈圖 |
| `build_histogram()` | 建立分布直方圖（常態曲線 + 規格線） |
| `build_cpk_trend()` | 建立跨檔案 Cpk 趨勢圖 |
| `build_cavity_fingerprint()` | 建立穴號指紋圖（每穴一條偏離折線） |
| `build_pareto_chart()` | 建立 Pareto 排列圖（長條 + 累積百分比） |

### core/export.py
| 函數 | 功能 |
|------|------|
| `download_plot_button()` | 圖表 PNG 下載按鈕 |
| `download_excel_button()` | Excel 資料下載按鈕 |
| `download_stats_excel()` | 統計摘要 Excel 下載 |
| `download_quality_reports()` | CP/CPK + SPC 報表下載 |
| `build_report_html()` | 建立 HTML 報表 |
| `generate_pdf_report()` | 產生 PDF 報表 |
| `download_pdf_report_button()` | PDF 報表下載按鈕 |

### core/grouping.py
| 函數 | 功能 |
|------|------|
| `format_pos()` | 格式化位置標籤 (P1, P2...) |
| `assign_groups_vectorized()` | 向量化分組邏輯 |

## UI 分頁結構

應用程式使用分頁式介面：

1. **📌 診斷總覽** - 彙整 Cpk/超規格/SPC 異常/常態性，依嚴重度排序；含 Pareto 排列圖（處理優先序）
2. **盒鬚圖** - 盒鬚圖分析（規格帶底色、平均值標記、n 標籤、hover 溯源）
3. **Cpk 分析** - Cpk 熱力圖、評級表（含 95% CI 與小樣本警告）、跨檔案 Cpk 趨勢
4. **SPC 控制圖** - I-MR 控制圖 + Nelson 規則異常模式判讀
5. **標準化偏離** - 標準化偏離圖（相對於規格公差的偏離百分比）
6. **模次比較** - 多模次位置比較圖 + 變異來源分解（穴間 vs 模次間）+ 穴號指紋圖
7. **相關性分析** - 維度間相關性矩陣與散佈圖
8. **🔍 維度詳情** - 單一維度完整檢視（盒鬚 + 直方圖 + I-MR + 全部判讀）
9. **🤖 AI 報告** - 將已算好的統計結果交給 Gemini 產生敘述報告（含免 API 的離線摘要）

## 分析情境（階段）

頂部「分析情境」選擇器提供三種階段，**只改變解讀框架與報告重點，不改變任何統計計算**：

| 階段 | 重點 | 淡化 |
|------|------|------|
| 試模 (T0/T1/T2) | 偏移方向與補正量、穴間平衡、系統性偏移的製程槓桿、鋼料安全邊提醒 | Cpk/SPC/常態性（不作良不良判定，用語改「待補正」） |
| 量產首件 | 是否符合規格、Cpk；超規格為警訊 | — |
| 量產監控 | SPC 穩定性、Nelson 異常、跨批趨勢、Cpk/Ppk 維持 | — |

- 常數定義於 `core/ai_report.py`（`STAGE_TRYOUT` 等、`_STAGE_GUIDANCE`、`_STAGE_SECTIONS`）
- `detect_systematic_bias()`（`core/statistics.py`）偵測「多數尺寸同向偏移」，試模模式在診斷總覽顯示、並寫入 AI 報告摘要
- 診斷總覽與 AI 報告分頁都會依所選階段調整提示與 AI 語氣

## AI 報告 (core/ai_report.py)

AI 只是「解讀層」：統計數字全由 `core.statistics` 確定性算好，模型只負責寫成人話，
並被系統指令嚴格限制「只能依提供的數字撰寫，不得自行運算或捏造」。
系統指令與報告段落結構會依上述分析情境切換。

| 函數 | 功能 |
|------|------|
| `build_analysis_payload()` | 從 diagnose_overview + variance 組出結構化摘要 dict |
| `format_payload_as_text()` | 摘要轉繁中文字（＝免 API 的離線報告） |
| `generate_ai_report()` | 呼叫 Gemini API 產生敘述報告 |
| `resolve_report()` | 一次取得 (離線摘要, AI 報告, 錯誤) |

- 使用 Google Gemini（REST，`requests`）；預設模型 `gemini-3.5-flash`
  （`gemini-2.0-flash` 已於 2026-06 停用），可選 `gemini-2.5-flash` / `gemini-2.5-flash-lite`
- 金鑰來源優先序：`st.secrets['GEMINI_API_KEY']` → 環境變數 `GEMINI_API_KEY` → UI 密碼輸入
- API 呼叫只在按鈕點擊時執行（避免每次 rerun 產生費用），結果存 `st.session_state`
- 隱私：產生 AI 報告會將摘要（含尺寸名稱與統計值）送至 Google API，機密資料請改用離線摘要

## 效能設計

- 所有 Excel 解析與統計計算皆以 `st.cache_data` 快取（`streamlit_app.py` 開頭的 `_*_cached` 函數），
  Streamlit 每次互動重跑腳本時不會重算
- **PNG 轉檔（kaleido）一律延後到使用者點擊匯出按鈕才執行**（`lazy_png_download()`、
  `_figs_to_png_bytes()`），渲染迴圈內不得呼叫 `fig.to_image()`
- 單張圖表的即時 PNG 下載改用 Plotly modebar 相機按鈕（瀏覽器端轉檔，`PLOTLY_CONFIG` 設定 3x scale）

## 資料流

```
Excel 上傳 → load_excel() → parse_focus_dimensions()
    → DataFrame (dimension, value, nominal, upper, lower, mold, pos_in_mold, cavity, cycle, arrangement)
    → 自動偵測 arrangement → 分組
    → 各分頁視覺化與統計
```

## 開發指令

```bash
# 啟動開發伺服器
streamlit run streamlit_app.py

# 使用 UV 安裝依賴
uv sync

# 使用 pip 安裝依賴
pip install -r requirements.txt
```

## Excel 解析邏輯

應用程式預期的 Excel 結構：

1. 工作表名稱含「重點尺寸」，或內容含「規格」+「球標」欄位
2. 標題列格式：`[空] | 規格 | 正公差 | 負公差 | 圖示位置 | 球標 | 量測1 | 量測2 | ... | 量測方式`
3. 數據從標題列 +2 行開始
4. 量測值位於「球標」與「量測方式」欄位之間

## 注意事項

- 所有 UI 文字使用繁體中文
- 超規格點以紅色標記
- 規格線使用紅色實線（上下限）與紅色虛線（中值）
- 匯出 PNG 使用 3x 縮放（高解析度）
- 支援多檔案合併或分檔比較模式
- 顯示模式包含：自動分組、強制分檔顯示、全部合併成一張圖
- 多模次檔案會依位置列自動分組成 P1..Pn

## 自動偵測資料排列方式

系統會根據 Excel 欄位標題自動偵測資料排列方式：

| 欄位格式 | 偵測結果 | 分組方式 | 說明 |
|----------|----------|----------|------|
| `CAV.1`, `CAV.2`... | `cavity_first` | 按 `cavity` 分組 | 穴號優先：同穴不同模次 |
| `第一模`, `第二模`... | `cycle_first` | 按 `pos_in_mold` 分組 | 模次優先：同模次不同穴號 |
| `#1-1`, `#2-3`... | 原有邏輯 | 按 `cavity` 分組 | 從標籤解析穴號 |

### 範例 (4穴3模次)

**穴號優先 (CAV.X 格式)**：
```
P1: 1,2,3    ← CAV.1 的模次 1~3
P2: 4,5,6    ← CAV.2 的模次 1~3
P3: 7,8,9    ← CAV.3 的模次 1~3
P4: 10,11,12 ← CAV.4 的模次 1~3
```

**模次優先 (第X模 格式)**：
```
P1: 1,5,9    ← 各模次的位置 1
P2: 2,6,10   ← 各模次的位置 2
P3: 3,7,11   ← 各模次的位置 3
P4: 4,8,12   ← 各模次的位置 4
```

## 自訂分組功能

### 各檔案獨立配置
- 每個上傳的檔案可獨立設定排列方式、穴數、模次數
- 支援預覽生成的分組規則
- 多檔案時 P 編號自動遞增

### 快速配置工具
- 根據穴數/模次數參數快速生成分組規則
- 支援「穴號優先」與「模次優先」兩種排列方式

## Cpk 評級標準

| Cpk 範圍 | 評級 | 顏色 |
|----------|------|------|
| >= 1.33 | 良好 | 綠色 |
| 1.0 ~ 1.33 | 可接受 | 黃色 |
| < 1.0 | 不良 | 紅色 |

- Cpk 另附 95% 信賴區間（Bissell 法）；樣本數 < 25 時顯示「⚠️ n<25」警告
- 常態性以 Shapiro-Wilk 檢定；非常態時提示 Cpk 僅供參考

## Nelson 規則 (SPC 異常模式)

實作於 `check_nelson_rules()`，共四條規則：

| 規則 | 條件 | 意義 |
|------|------|------|
| R1 | 單點超出 3σ | 異常單點 |
| R2 | 連續 9 點在中心線同側 | 製程平均偏移 |
| R3 | 連續 6 點持續上升/下降 | 趨勢（模溫漂移、磨耗等） |
| R5 | 3 點中 2 點超出同側 2σ | 變異增大或偏移前兆 |

SPC 分頁與維度詳情分頁會以橘色菱形在 I 圖上標記違規點並列出文字判讀。

## 變異來源分解

`variance_decomposition()` 以組間平方和占比 (R²) 估計穴間與模次間的變異貢獻：

- 穴間變異 ≥ 50% 且大於模次間 → 建議往修模/模穴均一性方向檢討
- 模次間變異 ≥ 50% 且大於穴間 → 建議檢查成型條件穩定性（調機）
- 每組平均樣本數 < 2 時不計算（組間變異無意義）

## 調機建議邏輯

`center_offset_suggestion()`：Cp 足夠但 Cpk 不足且中心偏移 ≥ 公差 10% 時，
給出偏移量與調整方向；Cp 亦不足時提示需縮小變異而非調中心。

## 分組設定重用

自訂分組介面支援將「排列方式/穴數/模次數/分組規則」匯出成 JSON
（`grouping_config.json`），下次上傳同格式檔案可直接匯入套用。

## 標準化偏離公式

```
偏離% = (量測值 - 規格中值) / 公差 × 100%
公差 = (上限 - 下限) / 2
```

## SPC 控制圖 (I-MR)

I-MR 控制圖用於監控製程穩定性：

- **I 圖 (Individual Chart)**：顯示個別量測值
  - CL = X̄ (平均值)
  - UCL = X̄ + 3σ
  - LCL = X̄ - 3σ

- **MR 圖 (Moving Range Chart)**：顯示相鄰量測值的差異
  - CL = MR̄ (平均移動全距)
  - UCL = D4 × MR̄ (D4 = 3.267)
  - LCL = 0

失控點（超出控制限）以紅色圈點標記。

## 相關性分析

相關性分析用於找出不同維度之間的關聯：

- **Pearson 相關係數**：衡量兩維度間的線性關係
  - r > 0.7：強正相關（兩維度同時增減）
  - r < -0.7：強負相關（一維度增加，另一維度減少）
  - |r| < 0.3：弱相關（兩維度獨立變動）

### 子分頁

| 分頁 | 功能 |
|------|------|
| 📊 相關性矩陣 | 基本相關性熱力圖，可選擇資料範圍 |
| 📁 按檔案比較 | 比較不同檔案的相關性差異 |
| 🔧 按穴號比較 | 比較不同穴號/分組的相關性差異 |

### 樣本配對邏輯

系統依序使用以下欄位作為樣本識別：
1. `pos_tag`（位置標籤）
2. `mold`（模次）
3. 維度內的行索引（fallback）

## 部署

本應用設計用於 Streamlit Cloud 部署：

1. 推送至 GitHub
2. 連接 Streamlit Cloud
3. 選擇 `streamlit_app.py` 作為主檔案
4. 自動從 `requirements.txt` 安裝依賴
