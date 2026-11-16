"""
Real-World Financial Warm-Start QAOA Study & Competition Benchmark
1. Fetches 2 years of real daily return data from Yahoo Finance for TW-Core-12 & US-Macro-12.
2. Constructs 6 real-market diversification/hedging networks (plus Markowitz portfolio QUBOs).
3. Executes:
   - Exp F1: 5-seed benchmark (Ideal & IBM Noisy Aer) across 6 real financial networks
             comparing Standard QAOA, Init-Only (Unaligned X-Mixer), and Aligned Warm-Start QAOA.
   - Exp F2: Out-of-Sample Market Regime Shift & Frozen Dynamics Recovery stress test.
   - Exp F3: Binary Markowitz Mean-Variance Portfolio Optimization (K=6 of N=12) via Warm-Start QAOA.
4. Saves all raw CSVs, median/IQR summaries, and 3 publication-grade plots.
"""

import os
import time
import json
import urllib.request
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from graph_utils import solve_exact_maxcut, compute_graph_metrics
from qaoa_core import (
    compute_warm_start_angles,
    build_qaoa_circuit,
    create_realistic_noise_model,
    evaluate_circuit_metrics,
    optimize_qaoa,
    execute_noisy_simulation,
)

# -----------------------------------------------------------------------------
# Tickers & Asset Universes
# -----------------------------------------------------------------------------
TW_TICKERS = [
    "2330.TW", "2454.TW", "2317.TW", "2382.TW", "2308.TW", "3711.TW",  # Tech/AI Supply Chain
    "2881.TW", "2882.TW", "2412.TW", "1216.TW", "00679B.TWO", "00635U.TW",  # Fin/Def/Bonds/Gold
]
TW_LABELS = {
    "2330.TW": "2330 TSMC",
    "2454.TW": "2454 MediaTek",
    "2317.TW": "2317 HonHai",
    "2382.TW": "2382 Quanta",
    "2308.TW": "2308 Delta",
    "3711.TW": "3711 ASE",
    "2881.TW": "2881 Fubon",
    "2882.TW": "2882 Cathay",
    "2412.TW": "2412 Chunghwa",
    "1216.TW": "1216 UniPres",
    "00679B.TWO": "00679B US20Y",
    "00635U.TW": "00635U Gold",
}
TW_SUBSET_10 = [
    "2330.TW", "2454.TW", "2317.TW", "2382.TW", "2308.TW",
    "2881.TW", "2412.TW", "1216.TW", "00679B.TWO", "00635U.TW",
]

US_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "AVGO",  # Tech/Growth
    "JPM", "XOM", "JNJ", "PG", "TLT", "GLD",         # Fin/Energy/Def/Bonds/Gold
]
US_SUBSET_10 = [
    "NVDA", "AAPL", "MSFT", "AMZN", "META",
    "JPM", "XOM", "JNJ", "TLT", "GLD",
]


def fetch_or_load_returns(tickers: list[str], cache_csv: str, corr_csv: str) -> pd.DataFrame:
    """Fetch 2y daily close returns from Yahoo Finance (or load cached CSV)."""
    if os.path.exists(cache_csv):
        ret_df = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
        if list(ret_df.columns) == tickers and len(ret_df) > 200:
            return ret_df

    series = {}
    for t in tickers:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=2y&interval=1d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            r = json.loads(resp.read())["chart"]["result"][0]
        dt = pd.to_datetime(r["timestamp"], unit="s").normalize()
        closes = r["indicators"]["quote"][0]["close"]
        s = pd.Series(closes, index=dt).dropna()
        s = s[~s.index.duplicated(keep="last")]
        series[t] = s

    price_df = pd.DataFrame(series).dropna()
    ret_df = price_df.pct_change().dropna()
    ret_df.to_csv(cache_csv)
    ret_df.corr().to_csv(corr_csv)
    return ret_df


def build_hedging_graph(ret_df: pd.DataFrame, tickers: list[str], threshold: float = 0.75) -> nx.Graph:
    """
    Construct weighted diversification/hedging MaxCut graph from asset returns.
    Edge weight w_ij = 1 - corr(i, j) when 1 - corr(i, j) > threshold.
    """
    sub_ret = ret_df[tickers]
    corr = sub_ret.corr().values
    n = len(tickers)
    G = nx.Graph()
    for i, t in enumerate(tickers):
        G.add_node(i, ticker=t)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(1.0 - corr[i, j])
            if w > threshold:
                G.add_edge(i, j, weight=round(w, 4))
    # Ensure graph connectivity if needed
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for k in range(len(comps) - 1):
            u = min(comps[k])
            v = min(comps[k + 1])
            w = max(0.25, float(1.0 - corr[u, v]))
            G.add_edge(u, v, weight=round(w, 4))
    return G


# -----------------------------------------------------------------------------
# Variant Circuit & Optimization Helpers (Aligned vs Init-Only vs Custom Hint)
# -----------------------------------------------------------------------------
def build_variant_circuit(
    G: nx.Graph,
    p: int,
    gammas: np.ndarray,
    betas: np.ndarray,
    variant: str,
    thetas: np.ndarray = None,
    measure: bool = False,
) -> QuantumCircuit:
    """
    variant in {"standard", "init_only", "warm_start"}:
      - "standard": |+>^n + standard Rx(2*beta) mixer
      - "init_only": Ry(theta_i)|0> + standard Rx(2*beta) mixer (mismatched!)
      - "warm_start": Ry(theta_i)|0> + Egger et al. aligned mixer Ry(th) Rz(-2*beta) Ry(-th)
    """
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    if variant == "standard" or thetas is None:
        qc.h(range(n))
    else:
        for i in range(n):
            qc.ry(float(thetas[i]), i)

    for layer in range(p):
        g_val = float(gammas[layer])
        b_val = float(betas[layer])
        for u, v, d in G.edges(data=True):
            qc.rzz(float(d.get("weight", 1.0)) * g_val, u, v)
        if variant in ("standard", "init_only") or thetas is None:
            for i in range(n):
                qc.rx(2.0 * b_val, i)
        else:
            for i in range(n):
                th = float(thetas[i])
                qc.ry(-th, i)
                qc.rz(-2.0 * b_val, i)
                qc.ry(th, i)

    if measure:
        qc.measure_all()
    return qc


def run_single_maxcut_trial(
    G: nx.Graph,
    exact_info: dict,
    variant: str,
    p: int = 1,
    epsilon: float = 0.15,
    custom_c_star: np.ndarray = None,
    maxiter: int = 100,
    seed: int = 42,
    noise_model=None,
    shots: int = 4096,
) -> dict:
    """Run optimization + ideal evaluation + optional IBM noisy Aer evaluation."""
    rng = np.random.default_rng(seed)
    cut_values = exact_info["cut_values"]
    c_max = exact_info["c_max"]
    opt_mask = exact_info["opt_mask"]

    if variant == "standard":
        thetas = None
    elif custom_c_star is not None:
        c_tilde = np.clip(custom_c_star, epsilon, 1.0 - epsilon)
        thetas = 2.0 * np.arcsin(np.sqrt(c_tilde))
    else:
        thetas, _ = compute_warm_start_angles(G, epsilon=epsilon, seed=seed)

    history = []

    def obj(params):
        qc = build_variant_circuit(G, p, params[:p], params[p:], variant=variant, thetas=thetas)
        probs = Statevector.from_instruction(qc).probabilities()
        exp_cut = float(np.dot(probs, cut_values))
        history.append(exp_cut / c_max)
        return -exp_cut

    avg_deg = max(1.0, 2.0 * G.number_of_edges() / G.number_of_nodes())
    g_scale = 0.6 / np.sqrt(avg_deg)
    if variant == "standard":
        init_g = np.linspace(0.20, 0.45, p) * g_scale + rng.normal(0, 0.06, size=p)
        init_b = np.linspace(0.15, 0.35, p) + rng.normal(0, 0.05, size=p)
    else:
        init_g = np.linspace(0.15, 0.35, p) * g_scale + rng.normal(0, 0.05, size=p)
        init_b = np.linspace(0.15, 0.30, p) + rng.normal(0, 0.05, size=p)
    x0 = np.concatenate([init_g, init_b])

    res = minimize(obj, x0, method="COBYLA", options={"maxiter": maxiter, "tol": 1e-4, "rhobeg": 0.25})
    best_g, best_b = res.x[:p], res.x[p:]

    qc_opt = build_variant_circuit(G, p, best_g, best_b, variant=variant, thetas=thetas, measure=False)
    probs = Statevector.from_instruction(qc_opt).probabilities()
    ar_ideal = float(np.dot(probs, cut_values) / c_max)
    p_opt_ideal = float(np.sum(probs[opt_mask]))

    hist_arr = np.array(history)
    best_so_far = np.maximum.accumulate(hist_arr)
    reached = np.where(best_so_far >= best_so_far[-1] * 0.995)[0]
    n_iter_eff = int(reached[0] + 1) if len(reached) > 0 else len(hist_arr)

    # Transpiled hardware metrics
    t_qc = transpile(qc_opt, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=42)
    ops = t_qc.count_ops()
    depth = t_qc.depth()
    two_q = int(ops.get("cx", 0))

    ar_noisy, p_opt_noisy = np.nan, np.nan
    if noise_model is not None:
        qc_m = build_variant_circuit(G, p, best_g, best_b, variant=variant, thetas=thetas, measure=True)
        sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        t_m = transpile(qc_m, sim, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=seed)
        counts = sim.run(t_m, shots=shots).result().get_counts()
        tot_cut = 0.0
        opt_hits = 0
        for bstr, cnt in counts.items():
            idx = int(bstr.replace(" ", ""), 2)
            tot_cut += cnt * cut_values[idx]
            if opt_mask[idx]:
                opt_hits += cnt
        ar_noisy = float((tot_cut / shots) / c_max)
        p_opt_noisy = float(opt_hits / shots)

    return {
        "ar_ideal": ar_ideal,
        "p_opt_ideal": p_opt_ideal,
        "ar_noisy": ar_noisy,
        "p_opt_noisy": p_opt_noisy,
        "noise_loss": float(ar_ideal - ar_noisy) if not np.isnan(ar_noisy) else np.nan,
        "n_iter": n_iter_eff,
        "circuit_depth": depth,
        "two_qubit_gates": two_q,
    }


# -----------------------------------------------------------------------------
# Markowitz Mean-Variance Binary Portfolio Solver & QAOA
# -----------------------------------------------------------------------------
def run_markowitz_experiment(ret_df: pd.DataFrame, tickers: list[str], market_name: str, k_target: int = 6) -> list[dict]:
    """
    Binary Markowitz Portfolio Selection:
      Maximize F(x) = lam * mu^T x - (1 - lam) * x^T Sigma x - penalty * (sum(x) - K)^2
    over x in {0, 1}^N, using continuous box-constrained QP warm-start (c* in [0, 1], sum c* = K).
    """
    sub = ret_df[tickers]
    mu = sub.mean().values * 252.0         # Annualized expected return
    sigma = sub.cov().values * 252.0       # Annualized covariance matrix
    n = len(tickers)

    lam = 0.35
    penalty = 0.15

    # Brute-force exact diagonal utilities over all 2^N portfolios (N <= 12 -> 4096 states)
    states = np.arange(1 << n, dtype=np.uint32)
    bits = ((states[None, :] >> np.arange(n, dtype=np.uint32)[:, None]) & 1).astype(np.float64)  # (N, 2^N)
    card = np.sum(bits, axis=0)
    ret_vec = mu @ bits
    # Vectorized x^T Sigma x across all states
    risk_vec = np.sum(bits * (sigma @ bits), axis=0)
    raw_util = lam * ret_vec - (1.0 - lam) * risk_vec - penalty * ((card - k_target) ** 2)

    u_max = float(np.max(raw_util))
    feasible_mask = np.isclose(card, k_target)
    u_feas_min = float(np.min(raw_util[feasible_mask]))
    u_opt_feas_max = float(np.max(raw_util[feasible_mask]))
    assert np.isclose(u_max, u_opt_feas_max), "Penalty too small to enforce exact feasibility"

    # Normalized utility score relative to global min-to-max range for positive QAOA AR comparison
    u_min = float(np.min(raw_util))
    norm_util = (raw_util - u_min) / (u_max - u_min)

    # Top 1% of feasible portfolios threshold
    feas_utils = raw_util[feasible_mask]
    top1_thresh = np.percentile(feas_utils, 99.0)
    top1_mask = feasible_mask & (raw_util >= top1_thresh - 1e-12)
    opt_mask = np.isclose(raw_util, u_max)

    # Map F(x) to Ising coefficients: x_i = (1 - Z_i) / 2
    # Linear Z_i coeff h_i and Quadratic Z_i Z_j coeff J_ij for exp(-i * gamma * H_cost)
    # Using cost Hamiltonian H_cost = -F(x):
    #   x_i = 0.5 - 0.5 Z_i
    #   x_i x_j (i != j) = 0.25 (1 - Z_i - Z_j + Z_i Z_j)
    #   x_i^2 = x_i = 0.5 - 0.5 Z_i
    Q = -(1.0 - lam) * sigma - penalty * np.ones((n, n))
    np.fill_diagonal(Q, np.diag(Q))
    lin_c = lam * mu + 2.0 * penalty * k_target * np.ones(n)
    # Separate diagonal x_i^2 = x_i
    total_lin = lin_c + np.diag(Q)
    Q_off = Q.copy()
    np.fill_diagonal(Q_off, 0.0)

    # H_cost = -F(x) = const + sum_i h_z[i] Z_i + sum_{i<j} J_zz[i,j] Z_i Z_j
    h_z = 0.5 * total_lin + 0.5 * np.sum(Q_off, axis=1)
    J_zz = -0.5 * Q_off  # since Q_off is symmetric, sum_{i<j} 2*Q_off[i,j] * (0.25 Z_i Z_j) -> -0.5 Q_off[i,j]

    # Solve Continuous Markowitz Box-Constrained QP (c in [0, 1], sum c = K)
    def cont_loss(c):
        return -(lam * np.dot(mu, c) - (1.0 - lam) * float(c @ sigma @ c))

    res_qp = minimize(
        cont_loss,
        np.full(n, k_target / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda c: np.sum(c) - k_target},
        options={"ftol": 1e-9, "maxiter": 300},
    )
    c_star = np.clip(res_qp.x, 0.0, 1.0)
    eps = 0.18
    c_tilde = np.clip(c_star, eps, 1.0 - eps)
    thetas_ws = 2.0 * np.arcsin(np.sqrt(c_tilde))

    def build_portfolio_circ(gammas, betas, p_layers, variant, thetas=None):
        qc = QuantumCircuit(n)
        if variant == "standard" or thetas is None:
            qc.h(range(n))
        else:
            for i in range(n):
                qc.ry(float(thetas[i]), i)
        for lyr in range(p_layers):
            g = float(gammas[lyr])
            b = float(betas[lyr])
            for i in range(n):
                if abs(h_z[i]) > 1e-9:
                    qc.rz(2.0 * g * float(h_z[i]), i)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(J_zz[i, j]) > 1e-9:
                        qc.rzz(2.0 * g * float(J_zz[i, j]), i, j)
            if variant in ("standard", "init_only") or thetas is None:
                for i in range(n):
                    qc.rx(2.0 * b, i)
            else:
                for i in range(n):
                    th = float(thetas[i])
                    qc.ry(-th, i)
                    qc.rz(-2.0 * b, i)
                    qc.ry(th, i)
        return qc

    rows = []
    configs = [
        ("Standard QAOA", "standard", 1),
        ("Standard QAOA", "standard", 2),
        ("Init-Only QAOA (Unaligned)", "init_only", 1),
        ("Warm-Start QAOA (ε=0.18)", "warm_start", 1),
        ("Warm-Start QAOA (ε=0.18)", "warm_start", 2),
    ]
    for label, var, p_val in configs:
        th = None if var == "standard" else thetas_ws
        for seed in [42, 43, 44, 45, 46]:
            rng = np.random.default_rng(seed)

            def obj(params):
                qc = build_portfolio_circ(params[:p_val], params[p_val:], p_val, var, th)
                probs = Statevector.from_instruction(qc).probabilities()
                return -float(np.dot(probs, raw_util))

            x0 = np.concatenate([
                np.linspace(0.2, 0.4, p_val) + rng.normal(0, 0.05, size=p_val),
                np.linspace(0.15, 0.30, p_val) + rng.normal(0, 0.04, size=p_val),
            ])
            res = minimize(obj, x0, method="COBYLA", options={"maxiter": 100, "tol": 1e-4, "rhobeg": 0.25})
            probs = Statevector.from_instruction(
                build_portfolio_circ(res.x[:p_val], res.x[p_val:], p_val, var, th)
            ).probabilities()

            rows.append({
                "market": market_name,
                "method": label,
                "variant": var,
                "p": p_val,
                "seed": seed,
                "norm_utility_ratio": float(np.dot(probs, norm_util)),
                "feasible_prob": float(np.sum(probs[feasible_mask])),
                "p_top1_pct": float(np.sum(probs[top1_mask])),
                "p_opt_exact": float(np.sum(probs[opt_mask])),
                "u_feas_min": u_feas_min,
                "u_max": u_max,
            })
    return rows


# -----------------------------------------------------------------------------
# Plotting Helpers
# -----------------------------------------------------------------------------
def generate_plots(
    ret_tw, ret_us, G_tw, G_us, exact_tw, exact_us,
    df_f1_summary, df_f2, plots_dir
):
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

    # Fig F1: TW-Core-12 & US-Macro-12 Correlation Heatmaps & Quasi-Bipartite Structure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    tw_short = [t.replace(".TWO", "").replace(".TW", "") for t in TW_TICKERS]
    corr_tw = ret_tw[TW_TICKERS].corr()
    corr_tw.index = tw_short
    corr_tw.columns = tw_short
    sns.heatmap(corr_tw, ax=axes[0], cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.9,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.8}, annot=True, fmt=".2f", annot_kws={"size": 7.5})
    axes[0].set_title(f"TW-Core-12 Return Correlation (MaxCut Edges={G_tw.number_of_edges()}, $C_{{max}}$={exact_tw['c_max']:.2f})",
                      fontweight="bold", fontsize=11.5)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    corr_us = ret_us[US_TICKERS].corr()
    sns.heatmap(corr_us, ax=axes[1], cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.9,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.8}, annot=True, fmt=".2f", annot_kws={"size": 7.5})
    axes[1].set_title(f"US-Macro-12 Return Correlation (MaxCut Edges={G_us.number_of_edges()}, $C_{{max}}$={exact_us['c_max']:.2f})",
                      fontweight="bold", fontsize=11.5)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.suptitle("Real Market Asset Correlation Diagnosis: Block-Anticorrelated Quasi-Bipartite Structure",
                 fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig_f1_financial_network_diagnosis.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig F2: 6 Real Financial Networks Benchmark (Median AR & P_opt at p=1, Ideal & Noisy)
    sub_p1 = df_f1_summary[df_f1_summary["p"] == 1].copy()
    markets_order = list(sub_p1["market"].drop_duplicates())
    methods_order = [
        "Standard QAOA",
        "Init-Only QAOA (Unaligned)",
        "Warm-Start QAOA (ε=0.25)",
        "Warm-Start QAOA (ε=0.15)",
    ]
    colors = ["#64748B", "#F59E0B", "#3B82F6", "#10B981"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    x = np.arange(len(markets_order))
    width = 0.19

    for idx, (m_name, col) in enumerate(zip(methods_order, colors)):
        m_df = sub_p1[sub_p1["method"] == m_name].set_index("market").loc[markets_order]
        offset = (idx - 1.5) * width
        # Left: Ideal AR + Noisy AR overlay marker
        axes[0].bar(x + offset, m_df["ar_ideal_median"], width=width, color=col, alpha=0.9, label=m_name)
        axes[0].scatter(x + offset, m_df["ar_noisy_median"], color="#1E293B", marker="D", s=22, zorder=5,
                        label="IBM Noisy Aer (p=1)" if idx == 0 else "")
        # Right: Optimal Partition Hit Probability P_opt
        axes[1].bar(x + offset, m_df["p_opt_ideal_median"], width=width, color=col, alpha=0.9, label=m_name)
        axes[1].scatter(x + offset, m_df["p_opt_noisy_median"], color="#1E293B", marker="D", s=22, zorder=5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(markets_order, rotation=22, ha="right", fontsize=9.5)
    axes[0].set_ylim(0.55, 1.03)
    axes[0].set_ylabel("Approximation Ratio (Median, 5 Seeds)", fontweight="bold")
    axes[0].set_title("Hedging Cut Quality at Depth p=1 (Bars=Ideal, Diamonds=IBM Noisy)", fontweight="bold", fontsize=11.5)
    axes[0].legend(loc="lower right", fontsize=8.5, ncol=2)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(markets_order, rotation=22, ha="right", fontsize=9.5)
    axes[1].set_ylim(0.0, 0.98)
    axes[1].set_ylabel("Optimal Partition Hit Prob. $P_{opt}$ (Median)", fontweight="bold")
    axes[1].set_title("Exact Hedging Partition Hit Probability $P_{opt}$ at Depth p=1", fontweight="bold", fontsize=11.5)
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Real TW & US Asset Hedging Benchmark: Warm-Start QAOA vs. Standard & Unaligned Init-Only (p=1)",
                 fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig_f2_real_market_benchmark.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig F3: Market Regime Shift Stress Test (TW-Core-12 & US-Macro-12)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    method_styles = [
        ("Stale Classical Hint (p=0)", "#6B7280", "--", "o"),
        ("Frozen WS-QAOA (ε=0.01, p=1)", "#EF4444", ":", "X"),
        ("Standard QAOA (Cold-Start, p=1)", "#64748B", "-.", "s"),
        ("Regularized WS-QAOA (ε=0.20, p=1)", "#3B82F6", "-", "^"),
        ("Regularized WS-QAOA (ε=0.20, p=2)", "#10B981", "-", "D"),
    ]
    for ax, mkt in zip(axes, ["TW-Core-12", "US-Macro-12"]):
        sub_m = df_f2[df_f2["market"] == mkt]
        for m_lbl, col, ls, mk in method_styles:
            grp = sub_m[sub_m["method"] == m_lbl].groupby("flipped_assets")["ar_ideal"].agg(["mean", "std"]).reset_index()
            ax.plot(grp["flipped_assets"], grp["mean"], color=col, linestyle=ls, marker=mk, linewidth=2.2, markersize=7, label=m_lbl)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Number of Assets Flipped by Market Regime Shift (k)", fontweight="bold")
        ax.set_ylabel("Post-Shift Hedging Cut AR", fontweight="bold")
        ax.set_ylim(0.58, 1.01)
        ax.set_title(f"{mkt}: Out-of-Sample Regime Shift Stress Test", fontweight="bold", fontsize=11.5)
        ax.legend(loc="lower left", fontsize=8.5)

    plt.suptitle("Overcoming the Frozen Dynamics Trap Under Out-of-Sample Market Regime Shifts",
                 fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig_f3_regime_shift_quantum_gain.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    t_start = time.time()
    out_dir = "results"
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("[1/5] Fetching / Loading Real Market Data (TW-Core-12 & US-Macro-12, 2y daily returns)...")
    ret_tw = fetch_or_load_returns(
        TW_TICKERS,
        os.path.join(out_dir, "data_tw_core_12_returns.csv"),
        os.path.join(out_dir, "data_tw_core_12_corr.csv"),
    )
    ret_us = fetch_or_load_returns(
        US_TICKERS,
        os.path.join(out_dir, "data_us_macro_12_returns.csv"),
        os.path.join(out_dir, "data_us_macro_12_corr.csv"),
    )
    print(f"      TW-Core-12 trading days: {len(ret_tw)} ({ret_tw.index.min().date()} to {ret_tw.index.max().date()})")
    print(f"      US-Macro-12 trading days: {len(ret_us)} ({ret_us.index.min().date()} to {ret_us.index.max().date()})")

    # Build 6 Real Financial Diversification/Hedging Graphs
    mid_tw = len(ret_tw) // 2
    mid_us = len(ret_us) // 2
    market_specs = [
        ("TW-Core-12 (Full 2Y)", build_hedging_graph(ret_tw, TW_TICKERS, threshold=0.75)),
        ("TW-Core-12 (H2 Window)", build_hedging_graph(ret_tw.iloc[mid_tw:], TW_TICKERS, threshold=0.75)),
        ("TW-Semi-Def-10", build_hedging_graph(ret_tw, TW_SUBSET_10, threshold=0.70)),
        ("US-Macro-12 (Full 2Y)", build_hedging_graph(ret_us, US_TICKERS, threshold=0.75)),
        ("US-Macro-12 (H1 Window)", build_hedging_graph(ret_us.iloc[:mid_us], US_TICKERS, threshold=0.75)),
        ("US-Tech-Def-10", build_hedging_graph(ret_us, US_SUBSET_10, threshold=0.70)),
    ]

    exact_map = {}
    metrics_rows = []
    for m_name, G in market_specs:
        ex = solve_exact_maxcut(G)
        exact_map[m_name] = ex
        gm = compute_graph_metrics(G, ex)
        gm["market"] = m_name
        metrics_rows.append(gm)
        print(f"      {m_name:24s} | N={gm['n']:2d}, E={gm['edges']:2d}, dens={gm['edge_density']:.3f}, tri={gm['triangles']:2d}, C_max={gm['c_max']:.3f}")
    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, "data_financial_graph_metrics.csv"), index=False)

    # =========================================================================
    # EXP F1: 5-Seed Benchmark Across 6 Real Market Networks (Ideal & IBM Noisy)
    # =========================================================================
    print("[2/5] Running Exp F1: 5-Seed Benchmark across 6 Real Financial Networks (p=1, 2)...")
    noise_model = create_realistic_noise_model()
    f1_rows = []
    f1_configs = [
        ("Standard QAOA", "standard", 0.50),
        ("Init-Only QAOA (Unaligned)", "init_only", 0.15),
        ("Warm-Start QAOA (ε=0.25)", "warm_start", 0.25),
        ("Warm-Start QAOA (ε=0.15)", "warm_start", 0.15),
    ]

    for m_name, G in market_specs:
        ex = exact_map[m_name]
        for p_val in [1, 2]:
            for method_lbl, var, eps in f1_configs:
                for seed in [42, 43, 44, 45, 46]:
                    res = run_single_maxcut_trial(
                        G, ex, variant=var, p=p_val, epsilon=eps,
                        maxiter=100, seed=seed,
                        noise_model=noise_model, shots=4096,
                    )
                    res.update({
                        "market": m_name,
                        "method": method_lbl,
                        "variant": var,
                        "epsilon": eps,
                        "p": p_val,
                        "seed": seed,
                        "n": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "c_max": ex["c_max"],
                    })
                    f1_rows.append(res)

    df_f1 = pd.DataFrame(f1_rows)
    df_f1.to_csv(os.path.join(out_dir, "exp_f1_real_market_benchmark.csv"), index=False)

    def iqr_fn(s):
        return float(np.percentile(s, 75) - np.percentile(s, 25))

    df_f1_sum = (
        df_f1.groupby(["market", "method", "p"], sort=False)
        .agg(
            ar_ideal_median=("ar_ideal", "median"),
            ar_ideal_iqr=("ar_ideal", iqr_fn),
            p_opt_ideal_median=("p_opt_ideal", "median"),
            p_opt_ideal_iqr=("p_opt_ideal", iqr_fn),
            ar_noisy_median=("ar_noisy", "median"),
            ar_noisy_iqr=("ar_noisy", iqr_fn),
            p_opt_noisy_median=("p_opt_noisy", "median"),
            n_iter_median=("n_iter", "median"),
            circuit_depth=("circuit_depth", "median"),
            two_qubit_gates=("two_qubit_gates", "median"),
        )
        .reset_index()
    )
    df_f1_sum.to_csv(os.path.join(out_dir, "exp_f1_summary_median_iqr.csv"), index=False)

    # =========================================================================
    # EXP F2: Out-of-Sample Market Regime Shift & Frozen Dynamics Stress Test
    # =========================================================================
    print("[3/5] Running Exp F2: Out-of-Sample Market Regime Shift Stress Test...")
    f2_rows = []
    for mkt_key, G_base, ex_base in [
        ("TW-Core-12", market_specs[0][1], exact_map["TW-Core-12 (Full 2Y)"]),
        ("US-Macro-12", market_specs[3][1], exact_map["US-Macro-12 (Full 2Y)"]),
    ]:
        n_nodes = G_base.number_of_nodes()
        best_bits = ex_base["best_bits"]
        c_max = ex_base["c_max"]

        for k_flip in [0, 1, 2, 3]:
            for trial in range(4):
                rng = np.random.default_rng(700 + k_flip * 20 + trial)
                hint_bits = best_bits.copy()
                if k_flip > 0:
                    flip_idx = rng.choice(n_nodes, size=k_flip, replace=False)
                    hint_bits[flip_idx] = 1.0 - hint_bits[flip_idx]

                state_int = sum((int(hint_bits[i]) << i) for i in range(n_nodes))
                hint_ar = float(ex_base["cut_values"][state_int] / c_max)
                hint_popt = 1.0 if ex_base["opt_mask"][state_int] else 0.0

                f2_rows.append({
                    "market": mkt_key, "flipped_assets": k_flip, "trial": trial,
                    "method": "Stale Classical Hint (p=0)",
                    "ar_ideal": hint_ar, "p_opt_ideal": hint_popt, "quantum_gain": 0.0,
                })

                # Frozen WS-QAOA (eps=0.01, p=1)
                r_frz = run_single_maxcut_trial(
                    G_base, ex_base, variant="warm_start", p=1, epsilon=0.01,
                    custom_c_star=hint_bits, maxiter=80, seed=42 + trial
                )
                f2_rows.append({
                    "market": mkt_key, "flipped_assets": k_flip, "trial": trial,
                    "method": "Frozen WS-QAOA (ε=0.01, p=1)",
                    "ar_ideal": r_frz["ar_ideal"], "p_opt_ideal": r_frz["p_opt_ideal"],
                    "quantum_gain": r_frz["ar_ideal"] - hint_ar,
                })

                # Standard Cold-Start QAOA (p=1)
                r_std = run_single_maxcut_trial(
                    G_base, ex_base, variant="standard", p=1, maxiter=80, seed=42 + trial
                )
                f2_rows.append({
                    "market": mkt_key, "flipped_assets": k_flip, "trial": trial,
                    "method": "Standard QAOA (Cold-Start, p=1)",
                    "ar_ideal": r_std["ar_ideal"], "p_opt_ideal": r_std["p_opt_ideal"],
                    "quantum_gain": r_std["ar_ideal"] - hint_ar,
                })

                # Regularized WS-QAOA (eps=0.20, p=1 and p=2)
                for p_val in [1, 2]:
                    r_reg = run_single_maxcut_trial(
                        G_base, ex_base, variant="warm_start", p=p_val, epsilon=0.20,
                        custom_c_star=hint_bits, maxiter=80, seed=42 + trial
                    )
                    f2_rows.append({
                        "market": mkt_key, "flipped_assets": k_flip, "trial": trial,
                        "method": f"Regularized WS-QAOA (ε=0.20, p={p_val})",
                        "ar_ideal": r_reg["ar_ideal"], "p_opt_ideal": r_reg["p_opt_ideal"],
                        "quantum_gain": r_reg["ar_ideal"] - hint_ar,
                    })

    df_f2 = pd.DataFrame(f2_rows)
    df_f2.to_csv(os.path.join(out_dir, "exp_f2_regime_shift_stress_test.csv"), index=False)
    df_f2_sum = (
        df_f2.groupby(["market", "flipped_assets", "method"], sort=False)
        .agg(
            ar_mean=("ar_ideal", "mean"),
            ar_median=("ar_ideal", "median"),
            p_opt_mean=("p_opt_ideal", "mean"),
            quantum_gain_mean=("quantum_gain", "mean"),
        )
        .reset_index()
    )
    df_f2_sum.to_csv(os.path.join(out_dir, "exp_f2_summary.csv"), index=False)

    # =========================================================================
    # EXP F3: Markowitz Binary Portfolio Optimization (K=6 of N=12)
    # =========================================================================
    print("[4/5] Running Exp F3: Markowitz Mean-Variance Binary Portfolio Selection (K=6 of 12)...")
    f3_rows = []
    f3_rows.extend(run_markowitz_experiment(ret_tw, TW_TICKERS, "TW-Core-12", k_target=6))
    f3_rows.extend(run_markowitz_experiment(ret_us, US_TICKERS, "US-Macro-12", k_target=6))
    df_f3 = pd.DataFrame(f3_rows)
    df_f3.to_csv(os.path.join(out_dir, "exp_f3_markowitz_portfolio_qubo.csv"), index=False)
    df_f3_sum = (
        df_f3.groupby(["market", "method", "p"], sort=False)
        .agg(
            norm_util_median=("norm_utility_ratio", "median"),
            norm_util_iqr=("norm_utility_ratio", iqr_fn),
            feasible_prob_median=("feasible_prob", "median"),
            p_top1_pct_median=("p_top1_pct", "median"),
            p_opt_exact_median=("p_opt_exact", "median"),
        )
        .reset_index()
    )
    df_f3_sum.to_csv(os.path.join(out_dir, "exp_f3_summary.csv"), index=False)

    # =========================================================================
    # Plots
    # =========================================================================
    print("[5/5] Generating publication-grade plots...")
    generate_plots(
        ret_tw, ret_us,
        market_specs[0][1], market_specs[3][1],
        exact_map["TW-Core-12 (Full 2Y)"], exact_map["US-Macro-12 (Full 2Y)"],
        df_f1_sum, df_f2, plots_dir
    )

    print(f"\nCompleted Real Financial Warm-Start QAOA Study in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
