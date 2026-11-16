"""
Full Head-to-Head 5-Way QAOA Benchmark & Financial Showdown
===========================================================
Benchmarks all 3 major QAOA paradigms + hybrid across BOTH:
  - Dataset A: 6 Real TW/US Stock Market Hedging Networks (TW-Core-12 & US-Macro-12, Yahoo Finance 2024-2026)
  - Dataset B: 8 Canonical Graph Topologies (N=10: Path, Star, Cycle, Tree, Complete, Bipartite, Community, Random)

Algorithms Compared (at circuit depth p=1, Ideal Statevector + IBM Noisy AerSimulator 4,096 shots):
  1. Standard QAOA            (Cold-start |+>^n, X-mixer, Mean Energy <C>)
  2. Pure CVaR-QAOA (a=0.25)  (Cold-start |+>^n, X-mixer, Tail Expectation CVaR_0.25 — Barkoutsos et al., 2020)
  3. Warm-Start QAOA (e=0.15) (SDP Ry(theta)|0>, Aligned Mixer, Mean Energy <C> — Egger et al., 2021)
  4. Hybrid WS + CVaR         (SDP Warm-Start + Aligned Mixer + Composite 0.5*CVaR_0.25 + 0.5*<C>)
  5. Recursive QAOA (R-QAOA)  (Iterative <Z_u Z_v> variable elimination down to n=4 — Bravyi et al., 2020)
     (+ "Init-Only (Unaligned X)" in Dataset B to isolate aligned mixer necessity)

Outputs:
  - results/exp_f4_full_5way_financial_benchmark.csv
  - results/exp_f4_summary_median_iqr.csv
  - results/exp6_architecture_ablation_rqaoa.csv
  - results/exp6_summary_eight_graphs.csv
  - results/plots/fig_f4_three_pillars_financial_showdown.png
  - results/plots/fig8_ablation_and_rqaoa_showdown.png
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

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from graph_utils import generate_eight_graph_families, solve_exact_maxcut
from qaoa_core import compute_warm_start_angles, create_realistic_noise_model
from run_financial_warmstart_study import (
    TW_TICKERS,
    TW_SUBSET_10,
    US_TICKERS,
    US_SUBSET_10,
    fetch_or_load_returns,
    build_hedging_graph,
    build_variant_circuit,
)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def compute_cvar(probs: np.ndarray, cut_values: np.ndarray, alpha: float = 0.25) -> float:
    """Exact Conditional Value-at-Risk (top alpha tail expectation) of a cut distribution."""
    desc_order = np.argsort(-cut_values)
    sorted_cuts = cut_values[desc_order]
    s_probs = probs[desc_order]
    cum_p = np.cumsum(s_probs)
    idx_cut = int(np.searchsorted(cum_p, alpha))
    if idx_cut == 0:
        return float(sorted_cuts[0])
    if idx_cut >= len(sorted_cuts):
        return float(np.dot(s_probs, sorted_cuts))
    head_mass = cum_p[idx_cut - 1]
    rem_mass = max(0.0, alpha - head_mass)
    return float((np.dot(s_probs[:idx_cut], sorted_cuts[:idx_cut]) + rem_mass * sorted_cuts[idx_cut]) / alpha)


def run_single_variational_qaoa(
    G: nx.Graph,
    exact_info: dict,
    method_name: str,
    variant: str,
    objective_mode: str = "mean",
    p: int = 1,
    epsilon: float = 0.15,
    cvar_alpha: float = 0.25,
    maxiter: int = 100,
    seed: int = 42,
    noise_model=None,
    shots: int = 4096,
) -> dict:
    """
    Runs one single-round variational QAOA trial (Standard, Pure CVaR, Init-Only, Warm-Start, or WS+CVaR)
    on both Ideal Statevector and IBM Calibrated Noisy AerSimulator.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    cut_values = exact_info["cut_values"]
    c_max = exact_info["c_max"]
    opt_mask = exact_info["opt_mask"]

    if variant == "standard":
        thetas = None
    else:
        thetas, _ = compute_warm_start_angles(G, epsilon=epsilon, seed=seed)

    desc_order = np.argsort(-cut_values)
    sorted_cuts = cut_values[desc_order]
    eval_count = 0

    def eval_cvar_fast(probs: np.ndarray) -> float:
        s_probs = probs[desc_order]
        cum_p = np.cumsum(s_probs)
        idx_cut = int(np.searchsorted(cum_p, cvar_alpha))
        if idx_cut == 0:
            return float(sorted_cuts[0])
        if idx_cut >= len(sorted_cuts):
            return float(np.dot(s_probs, sorted_cuts))
        head_mass = cum_p[idx_cut - 1]
        rem_mass = max(0.0, cvar_alpha - head_mass)
        return float((np.dot(s_probs[:idx_cut], sorted_cuts[:idx_cut]) + rem_mass * sorted_cuts[idx_cut]) / cvar_alpha)

    def obj(params):
        nonlocal eval_count
        eval_count += 1
        qc = build_variant_circuit(G, p, params[:p], params[p:], variant=variant, thetas=thetas)
        probs = Statevector.from_instruction(qc).probabilities()
        mean_cut = float(np.dot(probs, cut_values))
        if objective_mode == "cvar":
            return -eval_cvar_fast(probs)
        elif objective_mode == "hybrid_cvar":
            cvar_val = eval_cvar_fast(probs)
            return -(0.5 * cvar_val + 0.5 * mean_cut)
        else:
            return -mean_cut

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
    cvar25_ideal = float(eval_cvar_fast(probs) / c_max)

    t_qc = transpile(qc_opt, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=42)
    ops = t_qc.count_ops()
    depth = int(t_qc.depth())
    two_q = int(ops.get("cx", 0))

    ar_noisy, p_opt_noisy, cvar25_noisy = np.nan, np.nan, np.nan
    if noise_model is not None:
        qc_m = build_variant_circuit(G, p, best_g, best_b, variant=variant, thetas=thetas, measure=True)
        sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        t_m = transpile(qc_m, sim, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=seed)
        counts = sim.run(t_m, shots=shots).result().get_counts()
        tot_cut = 0.0
        opt_hits = 0
        noisy_probs = np.zeros(len(cut_values), dtype=np.float64)
        for bstr, cnt in counts.items():
            idx = int(bstr.replace(" ", ""), 2)
            noisy_probs[idx] = cnt / shots
            tot_cut += cnt * cut_values[idx]
            if opt_mask[idx]:
                opt_hits += cnt
        ar_noisy = float((tot_cut / shots) / c_max)
        p_opt_noisy = float(opt_hits / shots)
        cvar25_noisy = float(compute_cvar(noisy_probs, cut_values, alpha=0.25) / c_max)

    runtime_s = time.time() - t0
    return {
        "method": method_name,
        "ar_ideal": ar_ideal,
        "p_opt_ideal": p_opt_ideal,
        "cvar25_ideal": cvar25_ideal,
        "ar_noisy": ar_noisy,
        "p_opt_noisy": p_opt_noisy,
        "cvar25_noisy": cvar25_noisy,
        "circuit_evals": int(eval_count),
        "qpu_rounds": 1,
        "circuit_depth": depth,
        "two_qubit_gates": two_q,
        "runtime_s": float(runtime_s),
    }


def _execute_single_rqaoa_pass(
    G_orig: nx.Graph,
    exact_info: dict,
    p: int = 1,
    n_cutoff: int = 4,
    seed: int = 42,
    noise_model=None,
    shots: int = 4096,
) -> dict:
    """
    Runs Recursive QAOA (Bravyi et al., PRL 2020) either in Ideal Statevector mode (noise_model=None)
    or in IBM Noisy Shot-Based correlation measurement mode (noise_model!=None).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    active_G = nx.Graph()
    for u in G_orig.nodes():
        active_G.add_node(u)
    for u, v, d in G_orig.edges(data=True):
        active_G.add_edge(u, v, J=float(d.get("weight", 1.0)))

    relations = []
    total_evals = 0
    qpu_rounds = 0
    peak_two_q = 0
    peak_depth = 0
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed) if noise_model is not None else None

    while active_G.number_of_nodes() > n_cutoff and active_G.number_of_edges() > 0:
        qpu_rounds += 1
        nodes_list = list(active_G.nodes())
        m_nodes = len(nodes_list)
        node_to_idx = {nd: i for i, nd in enumerate(nodes_list)}

        states = np.arange(1 << m_nodes, dtype=np.uint32)
        bits = ((states[None, :] >> np.arange(m_nodes, dtype=np.uint32)[:, None]) & 1).astype(np.int8)
        spins = 1.0 - 2.0 * bits  # +1 for |0>, -1 for |1>

        ising_energy = np.zeros(1 << m_nodes, dtype=np.float64)
        for u, v, d in active_G.edges(data=True):
            iu, iv = node_to_idx[u], node_to_idx[v]
            ising_energy += d["J"] * (spins[iu] * spins[iv])

        def build_r_qc(gammas, betas, measure=False):
            qc = QuantumCircuit(m_nodes)
            qc.h(range(m_nodes))
            for lyr in range(p):
                for u, v, d in active_G.edges(data=True):
                    iu, iv = node_to_idx[u], node_to_idx[v]
                    qc.rzz(float(d["J"] * gammas[lyr]), iu, iv)
                for iq in range(m_nodes):
                    qc.rx(2.0 * float(betas[lyr]), iq)
            if measure:
                qc.measure_all()
            return qc

        def rqaoa_obj(params):
            nonlocal total_evals
            total_evals += 1
            qc = build_r_qc(params[:p], params[p:], measure=False)
            probs = Statevector.from_instruction(qc).probabilities()
            return float(np.dot(probs, ising_energy))

        x0 = np.concatenate([rng.uniform(0.15, 0.40, size=p), rng.uniform(0.15, 0.35, size=p)])
        res = minimize(rqaoa_obj, x0, method="COBYLA", options={"maxiter": 55, "tol": 1e-3, "rhobeg": 0.25})
        best_g, best_b = res.x[:p], res.x[p:]

        if qpu_rounds == 1:
            t_qc = transpile(build_r_qc(best_g, best_b, measure=False), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=42)
            peak_two_q = int(t_qc.count_ops().get("cx", 0))
            peak_depth = int(t_qc.depth())

        if sim is None:
            probs_corr = Statevector.from_instruction(build_r_qc(best_g, best_b, measure=False)).probabilities()
        else:
            qc_m = build_r_qc(best_g, best_b, measure=True)
            t_m = transpile(qc_m, sim, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=seed + qpu_rounds)
            counts = sim.run(t_m, shots=shots).result().get_counts()
            probs_corr = np.zeros(1 << m_nodes, dtype=np.float64)
            for bstr, cnt in counts.items():
                probs_corr[int(bstr.replace(" ", ""), 2)] = cnt / shots

        best_abs_corr = -1.0
        best_edge = None
        best_sign = -1.0
        for u, v, _ in active_G.edges(data=True):
            iu, iv = node_to_idx[u], node_to_idx[v]
            corr = float(np.dot(probs_corr, spins[iu] * spins[iv]))
            if abs(corr) > best_abs_corr:
                best_abs_corr = abs(corr)
                best_edge = (u, v)
                best_sign = 1.0 if corr >= 0 else -1.0

        u_keep, v_elim = best_edge
        relations.append((v_elim, u_keep, best_sign))
        active_G.remove_edge(u_keep, v_elim)
        for nbr in list(active_G.neighbors(v_elim)):
            j_val = active_G[v_elim][nbr]["J"] * best_sign
            if nbr != u_keep:
                if active_G.has_edge(u_keep, nbr):
                    active_G[u_keep][nbr]["J"] += j_val
                    if abs(active_G[u_keep][nbr]["J"]) < 1e-9:
                        active_G.remove_edge(u_keep, nbr)
                else:
                    active_G.add_edge(u_keep, nbr, J=j_val)
        active_G.remove_node(v_elim)

    rem_nodes = list(active_G.nodes())
    m_rem = len(rem_nodes)
    best_e = 1e18
    best_rem_spins = {}
    for s in range(1 << m_rem):
        sp = {rem_nodes[i]: (1.0 if ((s >> i) & 1) == 0 else -1.0) for i in range(m_rem)}
        e_val = sum(d["J"] * sp[u] * sp[v] for u, v, d in active_G.edges(data=True))
        if e_val < best_e:
            best_e = e_val
            best_rem_spins = sp

    full_spins = dict(best_rem_spins)
    for v_elim, u_keep, sgn in reversed(relations):
        full_spins[v_elim] = sgn * full_spins[u_keep]

    cut_achieved = sum(
        d.get("weight", 1.0) * (0.5 * (1.0 - full_spins[u] * full_spins[v]))
        for u, v, d in G_orig.edges(data=True)
    )
    ar = float(cut_achieved / exact_info["c_max"])
    p_opt = 1.0 if np.isclose(ar, 1.0) else 0.0
    return {
        "ar": ar,
        "p_opt": p_opt,
        "total_evals": int(total_evals),
        "qpu_rounds": int(qpu_rounds),
        "two_qubit_gates": peak_two_q,
        "circuit_depth": peak_depth,
        "runtime_s": float(time.time() - t0),
    }


def run_rqaoa_trial(
    G: nx.Graph,
    exact_info: dict,
    p: int = 1,
    n_cutoff: int = 4,
    seed: int = 42,
    noise_model=None,
    shots: int = 4096,
) -> dict:
    """Runs R-QAOA in both Ideal Statevector and IBM Noisy Shot-Based Correlation modes."""
    res_ideal = _execute_single_rqaoa_pass(G, exact_info, p=p, n_cutoff=n_cutoff, seed=seed, noise_model=None)
    res_noisy = _execute_single_rqaoa_pass(G, exact_info, p=p, n_cutoff=n_cutoff, seed=seed, noise_model=noise_model, shots=shots)
    return {
        "method": "Recursive QAOA (R-QAOA)",
        "ar_ideal": res_ideal["ar"],
        "p_opt_ideal": res_ideal["p_opt"],
        "cvar25_ideal": res_ideal["ar"],
        "ar_noisy": res_noisy["ar"],
        "p_opt_noisy": res_noisy["p_opt"],
        "cvar25_noisy": res_noisy["ar"],
        "circuit_evals": res_ideal["total_evals"],
        "qpu_rounds": res_ideal["qpu_rounds"],
        "circuit_depth": res_ideal["circuit_depth"],
        "two_qubit_gates": res_ideal["two_qubit_gates"],
        "runtime_s": res_ideal["runtime_s"] + res_noisy["runtime_s"],
    }


def summarize_with_iqr(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Computes median, IQR, mean, and std across random seeds."""
    metrics = ["ar_ideal", "p_opt_ideal", "cvar25_ideal", "ar_noisy", "p_opt_noisy", "circuit_evals", "qpu_rounds", "two_qubit_gates", "runtime_s"]
    rows = []
    for keys, grp in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in metrics:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                continue
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            row[f"{col}_median"] = float(q50)
            row[f"{col}_iqr"] = float(q75 - q25)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def generate_benchmark_plots(
    df_f4_summary: pd.DataFrame,
    df_f4_overall: pd.DataFrame,
    df_exp6_summary: pd.DataFrame,
    plots_dir: str,
):
    """Generates fig_f4_three_pillars_financial_showdown.png and fig8_ablation_and_rqaoa_showdown.png."""
    methods_fin = [
        "Standard QAOA",
        "Pure CVaR-QAOA (α=0.25)",
        "Warm-Start QAOA (ε=0.15)",
        "Hybrid WS + CVaR (ε=0.15, α=0.25)",
        "Recursive QAOA (R-QAOA)",
    ]
    short_fin = {
        "Standard QAOA": "Standard\nQAOA",
        "Pure CVaR-QAOA (α=0.25)": "Pure CVaR\n(Cold-Start)",
        "Warm-Start QAOA (ε=0.15)": "Warm-Start\n(ε=0.15)",
        "Hybrid WS + CVaR (ε=0.15, α=0.25)": "Hybrid\nWS + CVaR",
        "Recursive QAOA (R-QAOA)": "R-QAOA\n(Recursive)",
    }
    colors_fin = ["#64748B", "#8B5CF6", "#10B981", "#059669", "#2563EB"]

    # -------------------------------------------------------------------------
    # Figure F4: 4-Panel Full 5-Way Financial Showdown
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.5))

    markets_order = list(df_f4_summary["market"].drop_duplicates())
    x_m = np.arange(len(markets_order))
    w = 0.165

    # Panel (a): Ideal vs Noisy AR across all 6 real markets
    ax = axes[0, 0]
    for i, (m_lbl, col) in enumerate(zip(methods_fin, colors_fin)):
        sub = df_f4_summary[df_f4_summary["method"] == m_lbl].set_index("market").loc[markets_order]
        off = (i - 2.0) * w
        ax.bar(x_m + off, sub["ar_ideal_median"], width=w, color=col, alpha=0.92, label=m_lbl)
        ax.scatter(x_m + off, sub["ar_noisy_median"], color="#0F172A", marker="D", s=18, zorder=5,
                   label="IBM Noisy Aer (4,096 shots)" if i == 0 else "")
    ax.set_xticks(x_m)
    ax.set_xticklabels(markets_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.55, 1.12)
    ax.set_ylabel("Approximation Ratio (Median, 5 Seeds)", fontweight="bold")
    ax.set_title("(a) Hedging Cut Quality on 6 Real Markets (Bars=Ideal, Diamonds=IBM Noisy)", fontweight="bold", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=7.6, ncol=3, framealpha=0.92)

    # Panel (b): Optimal Hedging Partition Hit Rate P_opt (Overall Median across 6 Markets x 5 Seeds)
    ax = axes[0, 1]
    ov = df_f4_overall.set_index("method").loc[methods_fin]
    x_k = np.arange(len(methods_fin))
    bw = 0.36
    bars_id = ax.bar(x_k - bw / 2, ov["p_opt_ideal_median"] * 100.0, width=bw, color=colors_fin, alpha=0.95, label="Ideal Statevector")
    bars_ns = ax.bar(x_k + bw / 2, ov["p_opt_noisy_median"] * 100.0, width=bw, color=colors_fin, alpha=0.45, hatch="//", edgecolor="#0F172A", label="IBM Noisy Aer (4,096 shots)")
    for b_id, b_ns in zip(bars_id, bars_ns):
        ax.text(b_id.get_x() + b_id.get_width() / 2, b_id.get_height() + 1.5, f"{b_id.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(b_ns.get_x() + b_ns.get_width() / 2, b_ns.get_height() + 1.5, f"{b_ns.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x_k)
    ax.set_xticklabels([short_fin[m] for m in methods_fin], fontsize=9.2)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Exact Optimal Partition Hit Rate $P_{opt}$ (%)", fontweight="bold")
    ax.set_title("(b) Exact Optimal Hedging Hit Rate $P_{opt}$ (6 Markets Median)", fontweight="bold", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel (c): QPU Circuit Evaluation & Recursion Overhead
    ax = axes[1, 0]
    bars_ev = ax.bar(x_k, ov["circuit_evals_mean"], width=0.52, color=colors_fin, alpha=0.92)
    for b_ev, rnd in zip(bars_ev, ov["qpu_rounds_mean"]):
        ax.text(b_ev.get_x() + b_ev.get_width() / 2, b_ev.get_height() + 6,
                f"{b_ev.get_height():.0f} evals\n({rnd:.1f} QPU round{'s' if rnd > 1.1 else ''})",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x_k)
    ax.set_xticklabels([short_fin[m] for m in methods_fin], fontsize=9.2)
    ax.set_ylim(0, ov["circuit_evals_mean"].max() * 1.24)
    ax.set_ylabel("Total QPU Circuit Evaluations (Mean)", fontweight="bold")
    ax.set_title("(c) Circuit Evaluation & Sequential QPU Round Cost (p=1)", fontweight="bold", fontsize=11)

    # Panel (d): Ideal vs Noisy AR & Cost-Accuracy Pareto View
    ax = axes[1, 1]
    ann_offsets = {
        "Standard QAOA": (10, -2),
        "Pure CVaR-QAOA (α=0.25)": (10, -6),
        "Warm-Start QAOA (ε=0.15)": (10, 10),
        "Hybrid WS + CVaR (ε=0.15, α=0.25)": (10, -14),
        "Recursive QAOA (R-QAOA)": (-125, -4),
    }
    for m_lbl, col in zip(methods_fin, colors_fin):
        r = ov.loc[m_lbl]
        ax.scatter(
            r["circuit_evals_mean"],
            r["ar_noisy_median"],
            s=max(140, r["p_opt_noisy_median"] * 900 + 140),
            color=col,
            alpha=0.88,
            edgecolors="#0F172A",
            linewidths=1.4,
            label=f"{m_lbl} (Noisy AR={r['ar_noisy_median']:.3f})",
        )
        dx, dy = ann_offsets[m_lbl]
        ax.annotate(
            short_fin[m_lbl].replace("\n", " "),
            (r["circuit_evals_mean"], r["ar_noisy_median"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlabel("Total QPU Circuit Evaluations (Lower = Faster Rebalancing)", fontweight="bold")
    ax.set_ylabel("IBM Noisy Hardware AR (Higher = Better)", fontweight="bold")
    ax.set_xlim(15, ov["circuit_evals_mean"].max() * 1.15)
    ax.set_ylim(0.58, 1.05)
    ax.set_title("(d) Real-Market Engineering Trade-off (Bubble Size = Noisy $P_{opt}$)", fontweight="bold", fontsize=11)
    ax.legend(loc="center right", fontsize=7.8, framealpha=0.92, markerscale=0.55)

    plt.suptitle("5-Way Real Financial Market Benchmark: Standard vs. Pure CVaR vs. Warm-Start vs. Hybrid vs. R-QAOA (p=1)",
                 fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig_f4_three_pillars_financial_showdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 8: Updated 8 Canonical Topologies Full Showdown (Including Pure CVaR & R-QAOA Noisy)
    # -------------------------------------------------------------------------
    methods_8 = [
        "Standard QAOA",
        "Pure CVaR-QAOA (α=0.25)",
        "Init-Only (Unaligned X)",
        "Warm-Start QAOA (ε=0.15)",
        "Hybrid WS + CVaR (ε=0.15, α=0.25)",
        "Recursive QAOA (R-QAOA)",
    ]
    colors_8 = ["#64748B", "#8B5CF6", "#F59E0B", "#10B981", "#059669", "#2563EB"]
    graphs_order = ["Path", "Star", "Cycle", "Tree", "Complete", "Bipartite", "Community", "Random"]

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.4))
    xg = np.arange(len(graphs_order))
    wg = 0.135

    for idx, (m_lbl, col) in enumerate(zip(methods_8, colors_8)):
        sub = df_exp6_summary[df_exp6_summary["method"] == m_lbl].set_index("graph").loc[graphs_order]
        off = (idx - 2.5) * wg
        axes[0].bar(xg + off, sub["ar_ideal_median"], width=wg, color=col, alpha=0.92, label=m_lbl)
        axes[0].scatter(xg + off, sub["ar_noisy_median"], color="#0F172A", marker="D", s=16, zorder=5,
                        label="IBM Noisy Aer" if idx == 0 else "")
        axes[1].bar(xg + off, sub["p_opt_ideal_median"] * 100.0, width=wg, color=col, alpha=0.92, label=m_lbl)
        axes[1].scatter(xg + off, sub["p_opt_noisy_median"] * 100.0, color="#0F172A", marker="D", s=16, zorder=5)

    axes[0].set_xticks(xg)
    axes[0].set_xticklabels(graphs_order, rotation=20, ha="right", fontsize=9.5)
    axes[0].set_ylim(0.45, 1.13)
    axes[0].set_ylabel("Approximation Ratio (Median)", fontweight="bold")
    axes[0].set_title("(a) 8 Canonical Topologies: Ideal (Bars) vs. IBM Noisy (Diamonds) AR", fontweight="bold", fontsize=11.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=7.6, ncol=3, framealpha=0.92)

    axes[1].set_xticks(xg)
    axes[1].set_xticklabels(graphs_order, rotation=20, ha="right", fontsize=9.5)
    axes[1].set_ylim(0.0, 116.0)
    axes[1].set_ylabel("Exact Optimal Hit Probability $P_{opt}$ (%)", fontweight="bold")
    axes[1].set_title("(b) 8 Canonical Topologies: $P_{opt}$ (%) Across 6 QAOA Architectures", fontweight="bold", fontsize=11.5)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=7.6, ncol=3, framealpha=0.92)

    plt.suptitle("8-Graph Structural Ablation & Multi-Paradigm Benchmark: Why Aligned Warm-Start + CVaR Excels Without Multi-Round Overhead",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig8_ablation_and_rqaoa_showdown.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    t_all = time.time()
    out_dir = "results"
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    noise_model = create_realistic_noise_model()

    # =========================================================================
    # PART A: 6 Real Financial Market Networks (5 Seeds x 5 Algorithms, p=1)
    # =========================================================================
    print("[1/4] Loading cached 2-year TW & US daily return datasets...")
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

    fin_rows = []
    print("[2/4] Running 5-Way Real Financial Network Benchmark (6 markets x 5 methods x 5 seeds)...")
    for m_name, G in market_specs:
        ex = solve_exact_maxcut(G)
        for seed in [42, 43, 44, 45, 46]:
            configs = [
                ("Standard QAOA", "standard", "mean", 0.50),
                ("Pure CVaR-QAOA (α=0.25)", "standard", "cvar", 0.50),
                ("Warm-Start QAOA (ε=0.15)", "warm_start", "mean", 0.15),
                ("Hybrid WS + CVaR (ε=0.15, α=0.25)", "warm_start", "hybrid_cvar", 0.15),
            ]
            for m_lbl, var, obj_m, eps in configs:
                r = run_single_variational_qaoa(
                    G, ex, method_name=m_lbl, variant=var, objective_mode=obj_m,
                    p=1, epsilon=eps, cvar_alpha=0.25, maxiter=100, seed=seed,
                    noise_model=noise_model, shots=4096,
                )
                r.update({"market": m_name, "seed": seed, "n": G.number_of_nodes(), "edges": G.number_of_edges(), "c_max": ex["c_max"]})
                fin_rows.append(r)

            r_rq = run_rqaoa_trial(G, ex, p=1, n_cutoff=4, seed=seed, noise_model=noise_model, shots=4096)
            r_rq.update({"market": m_name, "seed": seed, "n": G.number_of_nodes(), "edges": G.number_of_edges(), "c_max": ex["c_max"]})
            fin_rows.append(r_rq)
        print(f"      Finished 5-way benchmark for {m_name}")

    df_f4 = pd.DataFrame(fin_rows)
    df_f4.to_csv(os.path.join(out_dir, "exp_f4_full_5way_financial_benchmark.csv"), index=False)

    df_f4_summary = summarize_with_iqr(df_f4, ["market", "method"])
    df_f4_summary.to_csv(os.path.join(out_dir, "exp_f4_summary_median_iqr.csv"), index=False)

    df_f4_overall = summarize_with_iqr(df_f4, ["method"])
    df_f4_overall.to_csv(os.path.join(out_dir, "exp_f4_overall_median_iqr.csv"), index=False)

    print("\n--- Overall 6-Market Financial Benchmark Summary (p=1, 30 trials per method) ---")
    for _, row in df_f4_overall.iterrows():
        print(
            f"  {row['method']:34s} | Ideal AR={row['ar_ideal_median']:.4f} | "
            f"Noisy AR={row['ar_noisy_median']:.4f} | P_opt(Id)={row['p_opt_ideal_median']*100:5.2f}% | "
            f"P_opt(Ns)={row['p_opt_noisy_median']*100:5.2f}% | Evals={row['circuit_evals_mean']:5.1f}"
        )

    # =========================================================================
    # PART B: 8 Canonical Graph Topologies (3 Seeds x 6 Methods, p=1)
    # =========================================================================
    print("\n[3/4] Running 6-Way Benchmark across 8 Canonical Topologies (N=10, 3 seeds)...")
    graphs_8 = generate_eight_graph_families(n=10, seed=42)
    exp6_rows = []
    for g_name, G in graphs_8.items():
        ex = solve_exact_maxcut(G)
        for seed in [42, 43, 44]:
            configs_8 = [
                ("Standard QAOA", "standard", "mean", 0.50),
                ("Pure CVaR-QAOA (α=0.25)", "standard", "cvar", 0.50),
                ("Init-Only (Unaligned X)", "init_only", "mean", 0.15),
                ("Warm-Start QAOA (ε=0.15)", "warm_start", "mean", 0.15),
                ("Hybrid WS + CVaR (ε=0.15, α=0.25)", "warm_start", "hybrid_cvar", 0.15),
            ]
            for m_lbl, var, obj_m, eps in configs_8:
                r = run_single_variational_qaoa(
                    G, ex, method_name=m_lbl, variant=var, objective_mode=obj_m,
                    p=1, epsilon=eps, cvar_alpha=0.25, maxiter=100, seed=seed,
                    noise_model=noise_model, shots=4096,
                )
                r.update({"graph": g_name, "seed": seed, "n": G.number_of_nodes(), "edges": G.number_of_edges(), "c_max": ex["c_max"]})
                exp6_rows.append(r)

            r_rq = run_rqaoa_trial(G, ex, p=1, n_cutoff=4, seed=seed, noise_model=noise_model, shots=4096)
            r_rq.update({"graph": g_name, "seed": seed, "n": G.number_of_nodes(), "edges": G.number_of_edges(), "c_max": ex["c_max"]})
            exp6_rows.append(r_rq)
        print(f"      Finished 8-graph benchmark for {g_name}")

    df_exp6 = pd.DataFrame(exp6_rows)
    df_exp6.to_csv(os.path.join(out_dir, "exp6_architecture_ablation_rqaoa.csv"), index=False)

    df_exp6_summary = summarize_with_iqr(df_exp6, ["graph", "method"])
    df_exp6_summary.to_csv(os.path.join(out_dir, "exp6_summary_eight_graphs.csv"), index=False)

    df_exp6_overall = summarize_with_iqr(df_exp6, ["method"])
    df_exp6_overall.to_csv(os.path.join(out_dir, "exp6_overall_median_iqr.csv"), index=False)

    print("\n--- Overall 8-Graph Structural Benchmark Summary (p=1, 24 trials per method) ---")
    for _, row in df_exp6_overall.iterrows():
        print(
            f"  {row['method']:34s} | Ideal AR={row['ar_ideal_median']:.4f} | "
            f"Noisy AR={row['ar_noisy_median']:.4f} | P_opt(Id)={row['p_opt_ideal_median']*100:5.2f}% | "
            f"P_opt(Ns)={row['p_opt_noisy_median']*100:5.2f}% | Evals={row['circuit_evals_mean']:5.1f}"
        )

    # =========================================================================
    # PART C: Generate Benchmark Charts
    # =========================================================================
    print("\n[4/4] Generating publication-grade plots...")
    generate_benchmark_plots(df_f4_summary, df_f4_overall, df_exp6_summary, plots_dir)
    print(f"Done! Total benchmark runtime: {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
