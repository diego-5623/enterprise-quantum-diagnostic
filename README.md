# Enterprise Quantum Utility Diagnostic — Financial PoC

> **企業量子技術導入評估與實效診斷框架**（橫跨量子最佳化演算法 × QML × 量子生成之資料適配度評測提案，附台美股兩年真實市場先導實證 PoC）。

## 核心文件
- **正式提案企劃書（Markdown）**：[`ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md`](./ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md)
- **圖文並茂網頁版（含全部實測大圖，可用瀏覽器開啟或列印 PDF）**：[`ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.html`](./ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.html)

## 專案結構與執行程式

```text
├── ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md    # 最新版企業量子健檢提案企劃書
├── ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.html  # 內嵌圖表之網頁版企劃書
├── graph_utils.py                               # 圖結構生成、精確 MaxCut 求解與拓撲指標
├── qaoa_core.py                                 # Goemans-Williamson SDP 熱啟動、對齊混合器與 IBM 雜訊模型
├── run_financial_warmstart_study.py             # Yahoo Finance 台美股真實資料抓取、6 網路評測、Regime Shift 與 Markowitz QUBO
├── run_full_3way_benchmark.py                   # 5 大 QAOA 範式（Standard / Pure CVaR / Warm-Start / Hybrid / R-QAOA）同場實測
└── results/                                     # 實跑數據 CSV 與 300-DPI 圖表（results/plots/）
```

## 重現實測數據（PowerShell）

```powershell
# 1. 執行台美股真實資產網路評測、市場突變（Regime Shift）與 Markowitz 12 選 6 投資組合實驗
python run_financial_warmstart_study.py

# 2. 執行五大 QAOA 架構（Standard vs. Pure CVaR vs. Warm-Start vs. Hybrid vs. R-QAOA）同場基準測試
python run_full_3way_benchmark.py
```
