# Google 帳號登入設定教學（Streamlit Cloud）

本文件說明如何為本 app 開啟 Google 登入（OIDC）與 email 白名單保護。
**在完成本設定之前，線上 app 是公開的，任何人皆可存取——請勿上傳機密資料。**

流程共五步：
1. Google Cloud Console 建立 OAuth 憑證
2. 產生 cookie_secret
3. 在 Streamlit Cloud 填入 Secrets
4. 驗證登入是否生效
5.（選用）本機測試登入

---

## 步驟 1：Google Cloud Console 建立 OAuth 憑證

### 1-1. 建立/選擇專案
1. 前往 <https://console.cloud.google.com/>
2. 頂端專案選單 → 「新增專案」（或選現有專案），例如命名 `dimension-analyzer`

### 1-2. 設定「OAuth 同意畫面」(OAuth consent screen)
1. 左側選單 → **APIs 和服務 → OAuth 同意畫面**
2. User Type 選 **外部 (External)**（若你是 Google Workspace 組織且只給內部人用，可選 Internal）→ 建立
3. 填寫：
   - 應用程式名稱：例如「尺寸品質分析工具」
   - 使用者支援電子郵件：你的 email
   - 開發人員聯絡資訊：你的 email
4. 「範圍 (Scopes)」：**不用加**，預設的 `openid` / `email` / `profile` 就夠 → 儲存並繼續
5. 「測試使用者 (Test users)」→ **新增**所有要登入的 Google 帳號（包含你自己）
   - ⚠️ 只要維持在「測試中 (Testing)」狀態，就只有這裡列出的帳號能登入，且**不需要 Google 審核**。對內部小工具這是最簡單的做法。
   - 不要為了這個去申請「發布/驗證」，那是給公開大量使用者用的，反而麻煩。

### 1-3. 建立「OAuth 用戶端 ID」
1. 左側選單 → **APIs 和服務 → 憑證 (Credentials)**
2. 上方 **建立憑證 → OAuth 用戶端 ID**
3. 應用程式類型：**網頁應用程式 (Web application)**
4. 名稱：例如 `streamlit-app`
5. ⚠️ 頁面上有**兩個**欄位，別填錯：
   - 「**已授權的 JavaScript 來源**」→ **留空**（此欄不能有路徑，填 `/oauth2callback` 會報「來源無效」）
   - 「**已授權的重新導向 URI (Authorized redirect URIs)**」→ 填這裡，新增兩個（本機 + 雲端）：
   ```
   http://localhost:8501/oauth2callback
   https://你的app名稱.streamlit.app/oauth2callback
   ```
   - ⚠️ 雲端那個要用**你實際的 app 網址**（在 Streamlit Cloud 打開 app 時瀏覽器上那串），結尾一定要加 `/oauth2callback`
   - ⚠️ 必須**完全一致**：`https`（不是 http）、沒有多餘的斜線、大小寫相同
6. 建立後跳出視窗，複製 **用戶端 ID (Client ID)** 與 **用戶端密鑰 (Client secret)** 備用

---

## 步驟 2：產生 cookie_secret

在專案目錄執行（用來簽署登入 cookie，需一組強隨機字串）：

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

複製輸出的那串（64 個十六進位字元）備用。
⚠️ 之後不要隨意更換，換掉會讓所有人被登出。

---

## 步驟 3：在 Streamlit Cloud 填入 Secrets

1. 前往 <https://share.streamlit.io/>，進入你的 app
2. 右下角 **Manage app → ⋮ → Settings → Secrets**（或 App 頁面的 Settings → Secrets）
3. 貼上以下內容，換成你自己的值：

   ```toml
   # Gemini AI 報告金鑰（選用）
   GEMINI_API_KEY = "你的-gemini-金鑰"

   [auth]
   redirect_uri = "https://你的app名稱.streamlit.app/oauth2callback"
   cookie_secret = "步驟2產生的64字元字串"
   client_id = "步驟1的用戶端ID.apps.googleusercontent.com"
   client_secret = "步驟1的用戶端密鑰"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

   # 白名單：只有這些帳號能進（強烈建議設定）
   allowed_emails = [
     "你@gmail.com",
     "同事@公司.com",
   ]
   ```

4. **Save** → app 會自動重新啟動

> `redirect_uri` 必須和步驟 1-3 在 Google 填的雲端 URI **一模一樣**，否則會出現 `redirect_uri_mismatch` 錯誤。

---

## 步驟 4：驗證是否生效

1. 開一個**無痕視窗**，打開你的 app 網址
2. 應該先看到「**使用 Google 帳號登入**」畫面（而不是直接進入）
   - 若仍直接進入、或看到黃色「登入保護尚未啟用」警告 → 表示 `[auth]` 沒吃到，回步驟 3 檢查
3. 用**白名單內**的帳號登入 → 順利進入
4. （可選）用**不在白名單**的帳號登入 → 應顯示「未獲授權」並被擋下
5. 側欄應出現「已登入：<你的名字>」與「登出」按鈕

驗證通過後，這個站就只有白名單帳號能用，機密資料才可上傳。

---

## 步驟 5（選用）：本機也要測登入

平常本機開發不需要登入。若想在本機測登入流程：

1. 把 `[auth]` 區塊也貼進本機 `.streamlit/secrets.toml`，但 `redirect_uri` 改成
   `http://localhost:8501/oauth2callback`
2. `streamlit run streamlit_app.py`
3. 測完可把 `[auth]` 從本機 secrets 移除，恢復免登入開發

（`.streamlit/secrets.toml` 已被 gitignore，不會進版控。）

---

## 常見問題

| 症狀 | 原因與解法 |
|------|-----------|
| `來源無效：URI 不得包含路徑或以「/」結尾` | 把含 `/oauth2callback` 的網址填錯到「**已授權的 JavaScript 來源**」欄了。該欄只收網域（無路徑）。請改填到下方「**已授權的重新導向 URI**」欄；JavaScript 來源欄留空即可。 |
| `Error 400: redirect_uri_mismatch` | Google 憑證裡的重新導向 URI 與 secrets 的 `redirect_uri` 不一致。逐字比對：https、結尾 `/oauth2callback`、無多餘斜線。 |
| `Access blocked: app not verified` / 無法登入 | 同意畫面在「測試中」，但登入帳號不在測試使用者清單。回步驟 1-2 加入該帳號。 |
| 登入後仍被擋「未獲授權」 | 該 email 不在 `allowed_emails`。加入清單（大小寫不拘）。 |
| 設定後仍直接進入、無登入畫面 | `[auth]` 未正確存入 Secrets（TOML 格式錯、或存在錯的地方）。確認貼在 app 的 Secrets、格式為 `[auth]` 區塊。 |
| `StreamlitMissingAuthlibError` / `No module named 'httpx'` | 登入需要 `Authlib` 與 `httpx`（`requirements.txt` 已含）。若雲端仍報錯，代表 app 未以最新 `requirements.txt` 重新部署：Manage app → Reboot，或確認已抓到最新 commit 重新 build。 |
| 大家突然都被登出 | `cookie_secret` 被改動。保持固定不要換。 |
| 找不到 app 的正確網址 | 在 Streamlit Cloud 開啟 app，複製瀏覽器網址列那串 `https://xxx.streamlit.app`。 |

---

## 安全提醒

- `client_secret`、`cookie_secret`、`GEMINI_API_KEY` 只放在 Streamlit Cloud 的 Secrets，**絕不要**寫進 repo。
- 白名單 `allowed_emails` 建議務必設定；留空的話任何能登入的 Google 帳號都進得來。
- 機密客戶資料：Community Cloud 為第三方（美國）主機，AI 報告會將摘要送至 Google。高度機密的批次建議只用「離線結構化摘要」，或改為公司內網自架。
