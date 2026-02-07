# Dimension Quality Analyzer

製造業尺寸品質分析工具 - 基於盒鬚圖的量測數據可視化與統計分析平台

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dimension-quality-analyzer-rib2iuacnie46eat67mwfr.streamlit.app/)

## 功能特色

- **Excel 自動解析** - 自動識別「重點尺寸」工作表，智能定位標題列與數據欄位
- **自動偵測排列方式** - 根據欄位標題 (CAV.X / 第X模) 自動判斷穴號優先或模次優先
- **盒鬚圖可視化** - 使用 Plotly 繪製互動式盒鬚圖，直觀呈現數據分佈
- **規格線標示** - 自動顯示上限、下限、中值規格線
- **超規格標記** - 紅色標記超出規格的量測點
- **統計分析** - 計算 Count、Mean、Std、Min、Median、Max
- **製程能力指標** - 計算 Cp、Cpk 評估製程穩定性
- **SPC 控制圖** - I-MR 控制圖分析與管制界限計算
- **相關性分析** - 維度間相關性矩陣、熱力圖與散佈圖
- **多檔案支援** - 自動分組、強制分檔或全部合併成單一盒鬚圖
- **各檔案獨立配置** - 每個檔案可獨立設定排列方式、穴數、模次數
- **多格式匯出** - 支援 PNG、Excel、HTML、PDF、ZIP 等格式

## 快速開始

### 線上使用

👉 **[立即使用 Dimension Quality Analyzer](https://dimension-quality-analyzer-rib2iuacnie46eat67mwfr.streamlit.app/)**

### 本地安裝

```bash
# 克隆專案
git clone https://github.com/lovexyz520/dimension-quality-analyzer.git
cd dimension-quality-analyzer

# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 執行應用程式
streamlit run streamlit_app.py
```

## Excel 檔案格式要求

應用程式會自動偵測包含「重點尺寸」的工作表，或尋找含有「規格」與「球標」欄位的工作表。

### 分組規則（自動分組）

系統會自動偵測 Excel 欄位標題格式來決定分組方式：

| 欄位格式 | 偵測結果 | 分組方式 |
|----------|----------|----------|
| `CAV.1`, `CAV.2`... | 穴號優先 | P1=1,2,3 \| P2=4,5,6... |
| `第一模`, `第二模`... | 模次優先 | P1=1,5,9 \| P2=2,6,10... |
| `#1-1`, `#2-3`... | 自動解析 | 從標籤解析穴號 |

- 單一檔案若存在多個模次（例如：第一模/第二模/第三模），會依位置列自動分組成 P1..Pn
- 單模次檔案會合併為單一盒鬚圖
- 多檔上傳時，能分組的檔案會依 P 分組；無法分組的檔案會獨立顯示
- 支援自訂分組規則，可針對每個檔案獨立設定

### 必要欄位

| 欄位名稱 | 說明 |
|---------|------|
| 規格 | 標稱值 (Nominal) |
| 正公差 | 上公差值 |
| 負公差 | 下公差值 |
| 球標 | 尺寸標識/編號 |
| 量測值欄位 | 位於球標之後，量測方式之前 |
| 量測方式 | (選填) 量測方法說明 |

### 範本檔案

請參考 `sample_data/template.xlsx` 了解正確的檔案格式。

## 使用流程

1. **上傳檔案** - 拖放或選擇 `.xlsm` / `.xlsx` 檔案
2. **選擇模式** - 自動分組 / 強制分檔顯示 / 全部合併成一張圖
3. **調整參數** - 圖表高度、視圖模式、縮放設定
4. **搜尋維度** - 使用關鍵字篩選特定尺寸
5. **查看分析** - 瀏覽盒鬚圖與統計數據
6. **匯出結果** - 下載 PNG、Excel 或 HTML 報表

## 技術架構

| 元件 | 技術 |
|------|------|
| Web 框架 | Streamlit |
| 數據處理 | Pandas, NumPy |
| 視覺化 | Plotly |
| Excel 讀寫 | openpyxl |
| 圖片匯出 | Kaleido |

## 匯出選項

- **CSV 長表** - 整理後的量測數據
- **Excel 統計摘要** - 各維度統計資訊
- **CP/CPK + SPC 報表** - 製程能力與控制圖分析
- **PNG 圖表** - 高解析度盒鬚圖
- **HTML 報表** - 包含圖表的完整報告
- **ZIP 打包** - 批次下載所有圖表

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案
