"""
Diagnostic Validation Upgrades (Addressing Peer-Review & Senior Feedback)
=======================================================================
1. Exp F5: Strong Classical Baselines (Exact Brute-Force, Simulated Annealing, Pure GW-SDP p=0)
           vs. QAOA on the 6 real TW/US financial networks (honest runtime & AR comparison).
2. Exp F6: True Out-of-Sample Rolling-Window Market Regime Shift (H1 Hint -> H2 Market)
           + Epsilon Sensitivity Sweep on TW-Core-12 & US-Macro-12.
3. Exp F7: Real IBM Heavy-Hex Coupling Map (19-qubit SWAP Routing) & Heterogeneous
           Per-Qubit Noise Simulation on TW-Core-12 & US-Macro-12.

Outputs:
  - results/exp_f5_classical_vs_quantum_baselines.csv
  - results/exp_f6_real_rolling_regime_shift.csv
  - results/exp_f6_epsilon_sensitivity.csv
  - results/exp_f7_heavy_hex_hardware_routing.csv
  - results/plots/fig_f5_diagnostic_validation_upgrades.png
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from graph_utils import solve_exact_maxcut
from qaoa_core import compute_warm_start_angles
from run_financial_warmstart_study import (
    TW_TICKERS,
    TW_SUBSET_10,
    US_TICKERS,
    US_SUBSET_10,
    fetch_or_load_returns,
    build_hedging_graph,
    build_variant_circuit,
    run_single_maxcut_trial,
)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


# -----------------------------------------------------------------------------
# Helper: Classical Baselines (Simulated Annealing & Pure GW-SDP p=0)
# -----------------------------------------------------------------------------
def run_simulated_annealing(G: nx.Graph, exact_info: dict, n_sweeps: int = 120, seed: int = 42) -> dict:
    """Fast classical Simulated Annealing solver for Weighted MaxCut."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G)
    best_bits = None
    best_cut = -1.0

    for _ in range(5):  # 5 restarts
        bits = rng.integers(0, 2, size=n, dtype=np.int8)
        spins = 1.0 - 2.0 * bits
        cur_cut = 0.25 * np.sum(adj * (1.0 - np.outer(spins, spins)))
        for step in range(n_sweeps):
            temp = max(0.01, 1.5 * (0.95 ** step))
            i = int(rng.integers(0, n))
            # Delta cut when flipping node i
            nbr_sum = float(np.dot(adj[i], spins))
            delta = spins[i] * nbr_sum
            if delta >= 0 or rng.random() < np.exp(delta / temp):
                bits[i] ^= 1
                spins[i] = -spins[i]
                cur_cut += delta
                if cur_cut > best_cut:
                    best_cut = cur_cut
                    best_bits = bits.copy()

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    ar = float(best_cut / exact_info["c_max"])
    p_opt = 1.0 if np.isclose(ar, 1.0) else 0.0
    return {"ar": ar, "p_opt": p_opt, "runtime_ms": runtime_ms}


def run_pure_sdp_p0(G: nx.Graph, exact_info: dict, seed: int = 42) -> dict:
    """Evaluates pure Classical GW-SDP rounding (p=0, no quantum circuit)."""
    t0 = time.perf_counter()
    _, c_tilde = compute_warm_start_angles(G, epsilon=0.0, seed=seed)
    bits = (c_tilde >= 0.5).astype(np.int8)
    cut_val = sum(
        d.get("weight", 1.0) * (bits[u] ^ bits[v])
        for u, v, d in G.edges(data=True)
    )
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    ar = float(cut_val / exact_info["c_max"])
    p_opt = 1.0 if np.isclose(ar, 1.0) else 0.0
    return {"ar": ar, "p_opt": p_opt, "runtime_ms": runtime_ms, "c_star": bits.astype(float)}


def create_heterogeneous_heavy_hex_noise_model(n_qubits: int = 19, seed: int = 42) -> NoiseModel:
    """
    Creates an inhomogeneous per-qubit IBM-calibrated noise model (with realistic spatial
    variation across qubits and edges) for Heavy-Hex SWAP-routed simulation.
    """
    rng = np.random.default_rng(seed)
    nm = NoiseModel()
    for q in range(n_qubits):
        p1 = float(np.clip(rng.normal(0.0005, 0.00015), 0.0002, 0.0012))
        pro = float(np.clip(rng.normal(0.015, 0.004), 0.006, 0.030))
        nm.add_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"], [q])
        nm.add_readout_error(ReadoutError([[1.0 - pro, pro], [pro, 1.0 - pro]]), [q])

    # Pairwise 2-qubit CNOT errors matching Heavy-Hex edges
    cm = CouplingMap.from_heavy_hex(3)
    seen_pairs = set()
    for u, v in cm.get_edges():
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        p2 = float(np.clip(rng.normal(0.0085, 0.0025), 0.004, 0.018))
        err2 = depolarizing_error(p2, 2)
        nm.add_quantum_error(err2, ["cx"], [u, v])
        nm.add_quantum_error(err2, ["cx"], [v, u])
    return nm


def evaluate_heavy_hex_circuit(
    G: nx.Graph,
    exact_info: dict,
    variant: str,
    p: int,
    epsilon: float = 0.15,
    seed: int = 42,
    shots: int = 4096,
) -> dict:
    """Optimizes on Statevector then transpiles onto 19-qubit IBM Heavy-Hex with SWAP routing."""
    res_opt = run_single_maxcut_trial(G, exact_info, variant=variant, p=p, epsilon=epsilon, maxiter=100, seed=seed)
    thetas = None
    if variant == "warm_start":
        thetas, _ = compute_warm_start_angles(G, epsilon=epsilon, seed=seed)

    # Re-optimize briefly to extract best params for transpilation
    rng = np.random.default_rng(seed)
    cut_values = exact_info["cut_values"]
    c_max = exact_info["c_max"]
    opt_mask = exact_info["opt_mask"]

    def obj(params):
        qc = build_variant_circuit(G, p, params[:p], params[p:], variant=variant, thetas=thetas)
        probs = Statevector.from_instruction(qc).probabilities()
        return -float(np.dot(probs, cut_values))

    avg_deg = max(1.0, 2.0 * G.number_of_edges() / G.number_of_nodes())
    g_scale = 0.6 / np.sqrt(avg_deg)
    x0 = np.concatenate([
        np.linspace(0.15, 0.35, p) * g_scale + rng.normal(0, 0.05, size=p),
        np.linspace(0.15, 0.30, p) + rng.normal(0, 0.05, size=p),
    ])
    res = minimize(obj, x0, method="COBYLA", options={"maxiter": 100, "tol": 1e-4, "rhobeg": 0.25})
    qc_m = build_variant_circuit(G, p, res.x[:p], res.x[p:], variant=variant, thetas=thetas, measure=True)

    cm = CouplingMap.from_heavy_hex(3)
    nm = create_heterogeneous_heavy_hex_noise_model(n_qubits=cm.size(), seed=42)
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)

    t_qc = transpile(
        qc_m,
        coupling_map=cm,
        basis_gates=["cx", "rz", "sx", "x"],
        optimization_level=3,
        seed_transpiler=seed,
    )
    ops = t_qc.count_ops()
    routed_cx = int(ops.get("cx", 0))
    routed_depth = int(t_qc.depth())

    counts = sim.run(t_qc, shots=shots).result().get_counts()
    tot_cut = 0.0
    opt_hits = 0
    for bstr, cnt in counts.items():
        # Measured bits correspond to G's n nodes in little-endian
        idx = int(bstr.replace(" ", ""), 2)
        tot_cut += cnt * cut_values[idx]
        if opt_mask[idx]:
            opt_hits += cnt

    return {
        "ar_ideal": res_opt["ar_ideal"],
        "p_opt_ideal": res_opt["p_opt_ideal"],
        "ar_heavyhex_noisy": float((tot_cut / shots) / c_max),
        "p_opt_heavyhex_noisy": float(opt_hits / shots),
        "all2all_cx": res_opt["two_qubit_gates"],
        "heavyhex_routed_cx": routed_cx,
        "heavyhex_routed_depth": routed_depth,
    }


def main():
    t_all = time.time()
    out_dir = "results"
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    ret_tw = fetch_or_load_returns(TW_TICKERS, os.path.join(out_dir, "data_tw_core_12_returns.csv"), os.path.join(out_dir, "data_tw_core_12_corr.csv"))
    ret_us = fetch_or_load_returns(US_TICKERS, os.path.join(out_dir, "data_us_macro_12_returns.csv"), os.path.join(out_dir, "data_us_macro_12_corr.csv"))

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

    # =========================================================================
    # EXP F5: Strong Classical Baselines (Exact, SA, Pure SDP p=0)
    # =========================================================================
    print("[1/3] Running Exp F5: Classical Baselines (Exact Brute-Force, Simulated Annealing, GW-SDP p=0)...")
    f5_rows = []
    for m_name, G in market_specs:
        t0 = time.perf_counter()
        ex = solve_exact_maxcut(G)
        exact_ms = (time.perf_counter() - t0) * 1000.0
        f5_rows.append({"market": m_name, "method": "Classical Exact Brute-Force", "ar": 1.0, "p_opt": 1.0, "runtime_ms": exact_ms})

        for seed in [42, 43, 44, 45, 46]:
            sa = run_simulated_annealing(G, ex, n_sweeps=120, seed=seed)
            f5_rows.append({"market": m_name, "method": "Classical Simulated Annealing", "ar": sa["ar"], "p_opt": sa["p_opt"], "runtime_ms": sa["runtime_ms"], "seed": seed})
            sdp = run_pure_sdp_p0(G, ex, seed=seed)
            f5_rows.append({"market": m_name, "method": "Classical GW-SDP (p=0)", "ar": sdp["ar"], "p_opt": sdp["p_opt"], "runtime_ms": sdp["runtime_ms"], "seed": seed})

    df_f5 = pd.DataFrame(f5_rows)
    df_f5.to_csv(os.path.join(out_dir, "exp_f5_classical_vs_quantum_baselines.csv"), index=False)
    print(df_f5.groupby("method")[["ar", "p_opt", "runtime_ms"]].mean().to_string())

    # =========================================================================
    # EXP F6: Real Out-of-Sample Rolling-Window Regime Shift + Epsilon Sweep
    # =========================================================================
    print("\n[2/3] Running Exp F6: Real Rolling-Window Regime Shift (H1 Hint -> H2 Market) & Epsilon Sweep...")
    f6_rows = []
    rolling_configs = [
        ("TW-Core-12", ret_tw, TW_TICKERS, 0.75),
        ("US-Macro-12", ret_us, US_TICKERS, 0.75),
    ]
    for m_lbl, ret_df, tickers, thr in rolling_configs:
        mid = len(ret_df) // 2
        q60 = int(len(ret_df) * 0.60)
        splits = [
            ("H1->H2 (50%/50%)", ret_df.iloc[:mid], ret_df.iloc[mid:]),
            ("First60%->Last40%", ret_df.iloc[:q60], ret_df.iloc[q60:]),
        ]
        for sp_name, df_past, df_future in splits:
            G_past = build_hedging_graph(df_past, tickers, threshold=thr)
            G_future = build_hedging_graph(df_future, tickers, threshold=thr)
            ex_past = solve_exact_maxcut(G_past)
            ex_future = solve_exact_maxcut(G_future)

            _, c_star_past = compute_warm_start_angles(G_past, epsilon=0.0, seed=42)
            # Evaluate stale past hint directly on future market (p=0)
            past_bits = (c_star_past >= 0.5).astype(np.int8)
            stale_cut = sum(d.get("weight", 1.0) * (past_bits[u] ^ past_bits[v]) for u, v, d in G_future.edges(data=True))
            stale_ar = float(stale_cut / ex_future["c_max"])
            stale_popt = 1.0 if np.isclose(stale_ar, 1.0) else 0.0

            # Also check how many assets differed between past SDP and future exact optimum
            hamming_diff = int(min(
                np.sum(past_bits != ex_future["best_bits"]),
                len(tickers) - np.sum(past_bits != ex_future["best_bits"]),
            ))

            for seed in [42, 43, 44, 45, 46]:
                f6_rows.append({"market": m_lbl, "split": sp_name, "method": "Stale Past Hint (p=0)", "hamming_shift": hamming_diff, "seed": seed, "ar_ideal": stale_ar, "p_opt_ideal": stale_popt})
                for m_name, var, p_v, eps_v, hint in [
                    ("Standard QAOA (Cold, p=1)", "standard", 1, 0.50, None),
                    ("Frozen WS-QAOA (ε=0.01, p=1)", "warm_start", 1, 0.01, c_star_past),
                    ("Reg. WS-QAOA (ε=0.20, p=1)", "warm_start", 1, 0.20, c_star_past),
                    ("Reg. WS-QAOA (ε=0.20, p=2)", "warm_start", 2, 0.20, c_star_past),
                ]:
                    r = run_single_maxcut_trial(G_future, ex_future, variant=var, p=p_v, epsilon=eps_v, custom_c_star=hint, maxiter=100, seed=seed)
                    f6_rows.append({"market": m_lbl, "split": sp_name, "method": m_name, "hamming_shift": hamming_diff, "seed": seed, "ar_ideal": r["ar_ideal"], "p_opt_ideal": r["p_opt_ideal"]})

    df_f6 = pd.DataFrame(f6_rows)
    df_f6.to_csv(os.path.join(out_dir, "exp_f6_real_rolling_regime_shift.csv"), index=False)
    print(df_f6.groupby(["market", "split", "method"])[["hamming_shift", "ar_ideal", "p_opt_ideal"]].mean().to_string())

    # Epsilon Sensitivity Sweep on TW-Core-12 & US-Macro-12 (Full 2Y)
    eps_rows = []
    for m_lbl, ret_df, tickers, thr in rolling_configs:
        G_full = build_hedging_graph(ret_df, tickers, threshold=thr)
        ex_full = solve_exact_maxcut(G_full)
        for eps in [0.01, 0.05, 0.10, 0.15, 0.25, 0.50]:
            for seed in [42, 43, 44]:
                r = run_single_maxcut_trial(G_full, ex_full, variant="warm_start", p=1, epsilon=eps, maxiter=100, seed=seed)
                eps_rows.append({"market": m_lbl, "epsilon": eps, "seed": seed, "ar_ideal": r["ar_ideal"], "p_opt_ideal": r["p_opt_ideal"]})
    df_eps = pd.DataFrame(eps_rows)
    df_eps.to_csv(os.path.join(out_dir, "exp_f6_epsilon_sensitivity.csv"), index=False)

    # =========================================================================
    # EXP F7: IBM Heavy-Hex 19-Qubit SWAP Routing + Heterogeneous Noise
    # =========================================================================
    print("\n[3/3] Running Exp F7: IBM Heavy-Hex 19-Qubit SWAP Routing & Heterogeneous Noise...")
    f7_rows = []
    for m_lbl, ret_df, tickers, thr in rolling_configs:
        G_full = build_hedging_graph(ret_df, tickers, threshold=thr)
        ex_full = solve_exact_maxcut(G_full)
        for method_lbl, var, p_v in [
            ("Standard QAOA (p=1)", "standard", 1),
            ("Standard QAOA (p=2)", "standard", 2),
            ("Warm-Start QAOA (p=1)", "warm_start", 1),
            ("Warm-Start QAOA (p=2)", "warm_start", 2),
        ]:
            for seed in [42, 43, 44]:
                r = evaluate_heavy_hex_circuit(G_full, ex_full, variant=var, p=p_v, epsilon=0.15, seed=seed, shots=4096)
                r.update({"market": m_lbl, "method": method_lbl, "seed": seed})
                f7_rows.append(r)

    df_f7 = pd.DataFrame(f7_rows)
    df_f7.to_csv(os.path.join(out_dir, "exp_f7_heavy_hex_hardware_routing.csv"), index=False)
    print(df_f7.groupby(["market", "method"])[["all2all_cx", "heavyhex_routed_cx", "ar_ideal", "ar_heavyhex_noisy", "p_opt_heavyhex_noisy"]].mean().to_string())

    # =========================================================================
    # Plot Fig F5: 3-Panel Diagnostic Validation Summary
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))

    # Panel (a): Epsilon Sensitivity on TW-Core-12 & US-Macro-12
    ax = axes[0]
    for m_lbl, col, mk in [("TW-Core-12", "#2563EB", "o"), ("US-Macro-12", "#10B981", "s")]:
        sub = df_eps[df_eps["market"] == m_lbl].groupby("epsilon")[["ar_ideal", "p_opt_ideal"]].mean().reset_index()
        ax.plot(sub["epsilon"], sub["ar_ideal"], color=col, marker=mk, linewidth=2.2, label=f"{m_lbl} AR")
        ax.plot(sub["epsilon"], sub["p_opt_ideal"], color=col, marker=mk, linestyle="--", linewidth=1.8, alpha=0.7, label=f"{m_lbl} $P_{{opt}}$")
    ax.axvspan(0.15, 0.25, color="#10B981", alpha=0.12, label="Recommended ε ∈ [0.15, 0.25]")
    ax.set_xlabel("Regularization Boundary ε (0.5 = Standard QAOA)", fontweight="bold")
    ax.set_ylabel("Ideal Score (AR & $P_{opt}$)", fontweight="bold")
    ax.set_title("(a) Financial ε Sensitivity Sweep (p=1)", fontweight="bold", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=8)

    # Panel (b): Real Out-of-Sample Rolling Regime Shift (H1 Hint -> H2 Market)
    ax = axes[1]
    sub_roll = df_f6[df_f6["split"] == "H1->H2 (50%/50%)"]
    m_order = ["Stale Past Hint (p=0)", "Frozen WS-QAOA (ε=0.01, p=1)", "Standard QAOA (Cold, p=1)", "Reg. WS-QAOA (ε=0.20, p=1)", "Reg. WS-QAOA (ε=0.20, p=2)"]
    short_m = ["Stale H1\nHint (p=0)", "Frozen WS\n(ε=0.01)", "Standard\n(Cold)", "Reg. WS\n(p=1)", "Reg. WS\n(p=2)"]
    x_r = np.arange(len(m_order))
    w_r = 0.36
    tw_means = [sub_roll[(sub_roll["market"] == "TW-Core-12") & (sub_roll["method"] == m)]["ar_ideal"].mean() for m in m_order]
    us_means = [sub_roll[(sub_roll["market"] == "US-Macro-12") & (sub_roll["method"] == m)]["ar_ideal"].mean() for m in m_order]
    b1 = ax.bar(x_r - w_r / 2, tw_means, width=w_r, color="#2563EB", alpha=0.9, label="TW-Core-12 (H1→H2)")
    b2 = ax.bar(x_r + w_r / 2, us_means, width=w_r, color="#10B981", alpha=0.9, label="US-Macro-12 (H1→H2)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x_r)
    ax.set_xticklabels(short_m, fontsize=8.5)
    ax.set_ylim(0.60, 1.06)
    ax.set_ylabel("Out-of-Sample H2 Market AR", fontweight="bold")
    ax.set_title("(b) Real Rolling Regime Shift (H1 Hint → H2)", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel (c): IBM Heavy-Hex 19-Qubit Routed CNOT vs Noisy AR
    ax = axes[2]
    f7_sum = df_f7.groupby(["market", "method"])[["heavyhex_routed_cx", "ar_heavyhex_noisy"]].mean().reset_index()
    for m_lbl, col, mk in [("TW-Core-12", "#2563EB", "o"), ("US-Macro-12", "#10B981", "s")]:
        for m_type, ls in [("Standard QAOA", "--"), ("Warm-Start QAOA", "-")]:
            sub = f7_sum[(f7_sum["market"] == m_lbl) & (f7_sum["method"].str.startswith(m_type))].sort_values("heavyhex_routed_cx")
            ax.plot(sub["heavyhex_routed_cx"], sub["ar_heavyhex_noisy"], color=col, linestyle=ls, marker=mk, linewidth=2.2, markersize=8,
                    label=f"{m_lbl} {m_type.split()[0]} (p=1→2)")
    ax.set_xlabel("IBM Heavy-Hex Post-SWAP Routed CNOT Count", fontweight="bold")
    ax.set_ylabel("Heavy-Hex Heterogeneous Noisy AR", fontweight="bold")
    ax.set_ylim(0.60, 0.95)
    ax.set_title("(c) IBM Heavy-Hex SWAP Routing & Noise Resilience", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Diagnostic Validation Upgrades: ε Sensitivity, Real Rolling Market Shift & IBM Heavy-Hex Routing", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig_f5_diagnostic_validation_upgrades.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nCompleted all validation upgrades in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
