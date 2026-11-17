# Enterprise Quantum Utility Diagnostic — Financial PoC

> **企業量子技術導入評估與實效診斷框架**（第一階段實作：賽道 A 量子最佳化演算法之台美股真實資料與嚴格控制實驗；賽道 B QML 與賽道 C QGen 為第二階段提案協議）。

## 核心文件與誠實聲明（Anti-Hype）
- **正式提案企劃書**：[`ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md`](./ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md)
- **關鍵結論**：
  1. **小規模 ($N \le 12$) 與準二分結構皆判 🔴 紅燈（維持古典）**：12 qubits 解空間僅 $2^{12}=4096$（均勻隨機盲猜 4,096 次即有 `86.47%` 機率至少中 1 次最佳解），且古典窮舉（`0.67 ms`）、模擬退火（`2.37 ms`）與 1-flip 爬山法（`AR = 0.9975–1.0000`）皆能毫秒級求解；二分／準二分圖在結構上對古典二著色與 SDP 本來就極容易，無須動用量子硬體。
  2. **演算法機制與指標公平對齊**：以提示來源消融（`exp_c1`）證明 Warm-Start 的高命中率主要源於古典初態編碼與對齊混合器；並以 $\text{CVaR}_{0.25}$ 指標公平呈現 `Pure CVaR-QAOA` 在前 25% 尾部確實優於冷啟動標準 QAOA（理想 `0.8462 → 0.8886`、合成雜訊 `0.8183 → 0.8481`），消除單看全體均值 AR 的指標錯配。

## 專案結構與執行程式

```text
├── ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md    # 最新版企業量子健檢提案企劃書（嚴格校訂版）
├── graph_utils.py                               # 圖結構生成、精確 MaxCut 與精確自同構軌道對稱度（對齊 Group 9 指標）
├── qaoa_core.py                                 # Goemans-Williamson SDP 熱啟動、對齊混合器與 IBM 量級合成雜訊模型
├── run_financial_warmstart_study.py             # 台美股真實資料（TW-Core-12 / US-Macro-12）、6 網路評測與 Markowitz QUBO
├── run_full_3way_benchmark.py                   # 5 大 QAOA 架構（Standard / Pure CVaR / Warm-Start / Hybrid / R-QAOA）實測
├── run_diagnostic_validation_upgrades.py        # 強古典基準、H1→H2 跨期與 IBM 19-qubit Heavy-Hex SWAP 路由驗證
├── run_control_experiments.py                   # 控制實驗 C1（含 4096 次均勻盲猜之提示消融）與 C2（全部 14 滾動視窗 + 爬山法）
├── tests/test_core.py                           # 核心單元測試（Symmetry / MaxCut / Warm-Start 邊界）
├── requirements.txt                             # Python 套件版本需求
├── LICENSE                                      # MIT License
└── results/                                     # 實跑數據 CSV（含 Demo 用邊清單 tw_core_12_edges.csv）與圖表
```

## 重現實測與測試（PowerShell / Bash）

```bash
# 執行核心單元測試（5 項測試）
python tests/test_core.py

# 執行提示消融（含均勻盲猜基線）與全部 14 滾動視窗控制實驗
python run_control_experiments.py

# 執行台美股基準、5 大架構對決與 Heavy-Hex 驗證
python run_financial_warmstart_study.py
python run_full_3way_benchmark.py
python run_diagnostic_validation_upgrades.py
```
