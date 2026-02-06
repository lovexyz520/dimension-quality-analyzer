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
| `imr_spc_points()` | 計算 I-MR 控制圖數據 |
| `calculate_normalized_deviation()` | 計算標準化偏離 |

### core/visualization.py
| 函數 | 功能 |
|------|------|
| `add_spec_lines()` | 新增規格線標註 |
| `build_fig()` | 建立 Plotly 盒鬚圖 |
| `apply_y_range()` | 套用 Y 軸範圍 |
| `add_spec_edge_markers()` | 新增規格邊界標記 |
| `add_out_of_spec_points()` | 標記超規格點 |
| `build_cpk_heatmap()` | 建立 Cpk 熱力圖 |
| `build_normalized_deviation_chart()` | 建立標準化偏離圖 |
| `build_position_comparison_chart()` | 建立模次比較圖 |
| `build_imr_chart()` | 建立 I-MR SPC 控制圖 |

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

1. **盒鬚圖** - 原有的盒鬚圖分析功能
2. **Cpk 分析** - Cpk 熱力圖與評級表
3. **SPC 控制圖** - I-MR 控制圖（製程穩定性監控）
4. **標準化偏離** - 標準化偏離圖（相對於規格公差的偏離百分比）
5. **模次比較** - 多模次位置比較圖

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

## 部署

本應用設計用於 Streamlit Cloud 部署：

1. 推送至 GitHub
2. 連接 Streamlit Cloud
3. 選擇 `streamlit_app.py` 作為主檔案
4. 自動從 `requirements.txt` 安裝依賴
