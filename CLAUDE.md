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

## 專案結構

```
dimension-quality-analyzer/
├── streamlit_app.py      # 主應用程式（所有邏輯在此檔案）
├── requirements.txt      # Python 依賴
├── pyproject.toml        # 專案配置（UV 套件管理）
├── .streamlit/
│   └── config.toml       # Streamlit 配置
└── sample_data/
    └── template.xlsx     # Excel 範本
```

## 程式碼架構 (streamlit_app.py)

### 核心函數

| 函數 | 行數 | 功能 |
|------|------|------|
| `_find_header_row()` | 16-21 | 尋找含「規格」與「球標」的標題列 |
| `_detect_focus_sheet()` | 39-52 | 自動偵測「重點尺寸」工作表 |
| `parse_focus_dimensions()` | 55-140 | 解析 Excel 數據為標準格式 |
| `load_excel()` | 143-155 | Excel 檔案載入入口 |
| `_build_fig()` | 217-234 | 建立 Plotly 盒鬚圖 |
| `_add_spec_lines()` | 177-214 | 新增規格線標註 |
| `_stats_table()` | 330-351 | 計算統計摘要 |
| `_cp_cpk_summary()` | 366-393 | 計算 Cp/Cpk 製程能力 |
| `_imr_spc_points()` | 396-448 | 計算 I-MR 控制圖數據 |

### 資料流

```
Excel 上傳 → _detect_focus_sheet() → parse_focus_dimensions()
    → DataFrame (dimension, value, nominal, upper, lower)
    → _build_fig() → Plotly 圖表
    → _stats_table() / _cp_cpk_summary() → 統計報表
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

## 部署

本應用設計用於 Streamlit Cloud 部署：

1. 推送至 GitHub
2. 連接 Streamlit Cloud
3. 選擇 `streamlit_app.py` 作為主檔案
4. 自動從 `requirements.txt` 安裝依賴
