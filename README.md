# Enterprise Quantum Utility Diagnostic — Financial PoC

> **企業量子技術導入評估與實效診斷框架**（第一階段實作：賽道 A 量子最佳化演算法之台美股真實資料對照驗證；賽道 B QML 與賽道 C QGen 為第二階段提案協議）。

## 核心文件與誠實聲明
- **正式提案企劃書**：[`ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md`](./ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md)
- **關鍵結論（Anti-Hype）**：在目前可精確驗證的 $N \le 12$ 金融資產規模下，古典窮舉（`0.67 ms`）、模擬退火（`2.37 ms`）與 1-flip 爬山法（`AR≈1.0`）已能毫秒級求解，因此本健檢對小規模實務導入判為 **🔴 紅燈（維持古典）**；Warm-Start QAOA 的價值在於透過控制實驗拆解不同量子架構在合成雜訊與 Heavy-Hex 拓撲下的擴展表現。

## 專案結構與執行程式

```text
├── ENTERPRISE_QUANTUM_DIAGNOSTIC_PROPOSAL.md    # 最新版企業量子健檢提案企劃書（誠實校訂版）
├── graph_utils.py                               # 圖結構生成、精確 MaxCut 與精確自同構軌道對稱度（對齊 Group 9 指標）
├── qaoa_core.py                                 # Goemans-Williamson SDP 熱啟動、對齊混合器與 IBM 量級合成雜訊模型
├── run_financial_warmstart_study.py             # 台美股真實資料（TW-Core-12 / US-Macro-12）、6 網路評測與 Markowitz QUBO
├── run_full_3way_benchmark.py                   # 5 大 QAOA 架構（Standard / Pure CVaR / Warm-Start / Hybrid / R-QAOA）實測
├── run_diagnostic_validation_upgrades.py        # 強古典基準、H1→H2 跨期與 IBM 19-qubit Heavy-Hex SWAP 路由驗證
├── run_control_experiments.py                   # 控制實驗 C1（提示來源消融）與 C2（全部 14 個真實滾動視窗 + 古典爬山法對照）
├── tests/test_core.py                           # 核心單元測試（Symmetry / MaxCut / Warm-Start 邊界）
├── requirements.txt                             # Python 套件版本需求
├── LICENSE                                      # MIT License
└── results/                                     # 實跑數據 CSV 與圖表（results/plots/）
```

## 重現實測與測試（PowerShell / Bash）

```bash
# 執行核心單元測試
python tests/test_core.py

# 執行提示消融與全部 14 滾動視窗控制實驗（支援 --quick 快速測試）
python run_control_experiments.py

# 執行台美股基準、5 大架構對決與 Heavy-Hex 驗證
python run_financial_warmstart_study.py
python run_full_3way_benchmark.py
python run_diagnostic_validation_upgrades.py
```
