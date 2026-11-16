"""
Standard QAOA vs. Warm-Start QAOA Core Implementation in Qiskit
Follows Egger et al., "Warm-starting quantum optimization", Quantum 5, 479 (2021)
and aligns with all metrics in NTU IBM Qiskit Hackathon Group 9's benchmark.
"""

import numpy as np
import networkx as nx
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def compute_warm_start_angles(G: nx.Graph, epsilon: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the Goemans-Williamson SDP Relaxation (via CVXPY) and hyperplane rounding
    (Egger et al., Quantum 2021, Section 3) to obtain vertex assignments c_i^* in [0, 1],
    then apply regularization epsilon in (0, 0.5] to obtain initial rotation angles theta_i:
        theta_i = 2 * arcsin(sqrt(c_tilde_i))
    where c_tilde_i in [epsilon, 1 - epsilon].

    Note: When epsilon = 0.5, c_tilde_i = 0.5 and theta_i = pi/2 for all i,
          which exactly recovers Standard QAOA!
    """
    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    adj = nx.to_numpy_array(G)

    c_star = None
    if HAS_CVXPY:
        try:
            X = cp.Variable((n, n), symmetric=True)
            constraints = [X >> 0, cp.diag(X) == 1]
            obj = 0.25 * cp.sum(cp.multiply(adj, 1 - X))
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)

            if X.value is not None:
                eigvals, eigvecs = np.linalg.eigh(X.value)
                eigvals = np.maximum(eigvals, 0.0)
                V = eigvecs @ np.diag(np.sqrt(eigvals))

                best_cut = -1.0
                best_signs = None
                for trial in range(80):
                    if trial == 0:
                        r = eigvecs[:, -1]
                    else:
                        r = rng.normal(size=n)
                    r_norm = np.linalg.norm(r)
                    if r_norm > 1e-9:
                        r = r / r_norm
                    proj = V @ r
                    signs = np.where(proj >= 0, 1.0, -1.0)
                    cut_val = 0.25 * np.sum(adj * (1.0 - np.outer(signs, signs)))
                    if cut_val > best_cut:
                        best_cut = cut_val
                        best_signs = signs

                c_star = 0.5 * (1.0 + best_signs)
        except Exception:
            c_star = None

    if c_star is None:
        # Fallback: BMZ rank-2 continuous angle relaxation on unit circle
        def bmz_loss(alphas):
            diff = alphas[:, None] - alphas[None, :]
            return -0.25 * np.sum(adj * (1.0 - np.cos(diff)))

        best_res = None
        for _ in range(10):
            x0 = rng.uniform(0, 2 * np.pi, size=n)
            res = minimize(bmz_loss, x0, method="BFGS")
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        signs = np.where(np.cos(best_res.x) >= 0, 1.0, -1.0)
        c_star = 0.5 * (1.0 + signs)

    # Apply Egger et al. regularization with parameter epsilon in [0, 0.5]
    c_tilde = np.clip(c_star, epsilon, 1.0 - epsilon)
    thetas = 2.0 * np.arcsin(np.sqrt(c_tilde))
    return thetas, c_tilde


def build_qaoa_circuit(
    G: nx.Graph,
    p: int,
    gammas: np.ndarray,
    betas: np.ndarray,
    mode: str = "standard",
    thetas: np.ndarray = None,
    measure: bool = False,
) -> QuantumCircuit:
    """
    Construct the QAOA or Warm-Start QAOA quantum circuit for MaxCut on graph G.
    """
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)

    # 1. Initial State Preparation
    if mode == "standard" or thetas is None:
        qc.h(range(n))
    elif mode == "warm_start":
        for i in range(n):
            qc.ry(float(thetas[i]), i)
    else:
        raise ValueError(f"Unknown QAOA mode: {mode}")

    # 2. Alternating p layers of Cost Unitary U_C(gamma_layer) and Mixer Unitary U_M(beta_layer)
    for layer in range(p):
        gamma = float(gammas[layer])
        beta = float(betas[layer])

        # Cost Hamiltonian evolution for MaxCut: exp(-i * gamma * Z_u Z_v / 2) -> RZZ(gamma)
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            qc.rzz(w * gamma, u, v)

        # Mixer Hamiltonian evolution
        if mode == "standard" or thetas is None:
            for i in range(n):
                qc.rx(2.0 * beta, i)
        elif mode == "warm_start":
            # Egger et al. (2021) custom Warm-Start Mixer:
            # Preserves |phi^*> = bigotimes_i Ry(theta_i)|0> as ground state of H_M^(ws)
            for i in range(n):
                th = float(thetas[i])
                qc.ry(-th, i)
                qc.rz(-2.0 * beta, i)
                qc.ry(th, i)

    if measure:
        qc.measure_all()

    return qc


def create_realistic_noise_model(
    p_1q: float = 0.0005,
    p_2q: float = 0.008,
    p_ro: float = 0.015,
) -> NoiseModel:
    """
    Build an IBM-calibrated realistic depolarizing + readout error noise model
    for noisy QAOA execution comparisons.
    """
    noise_model = NoiseModel()
    err_1q = depolarizing_error(p_1q, 1)
    err_2q = depolarizing_error(p_2q, 2)
    ro_err = ReadoutError([[1.0 - p_ro, p_ro], [p_ro, 1.0 - p_ro]])

    noise_model.add_all_qubit_quantum_error(err_1q, ["rx", "ry", "rz", "h", "sx", "x"])
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "rzz"])
    noise_model.add_all_qubit_readout_error(ro_err)
    return noise_model


def evaluate_circuit_metrics(G: nx.Graph, p: int, mode: str = "standard", thetas: np.ndarray = None) -> dict:
    """
    Decompose the QAOA circuit into standard hardware basis gates (cx, rz, sx, x)
    and return circuit depth and 2-qubit gate count.
    """
    dummy_gammas = np.ones(p) * 0.5
    dummy_betas = np.ones(p) * 0.5
    qc = build_qaoa_circuit(G, p, dummy_gammas, dummy_betas, mode=mode, thetas=thetas, measure=False)
    t_qc = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=42)
    ops = t_qc.count_ops()
    return {
        "depth": t_qc.depth(),
        "two_qubit_gates": ops.get("cx", 0),
    }


def optimize_qaoa(
    G: nx.Graph,
    exact_info: dict,
    p: int = 1,
    mode: str = "standard",
    epsilon: float = 0.25,
    maxiter: int = 100,
    seed: int = 42,
) -> dict:
    """
    Optimize QAOA or Warm-Start QAOA parameters (gammas, betas) using COBYLA
    (maxiter=100, matching Group 9's slide definition S = 1 - N_iter / 100),
    and evaluate ideal statevector metrics.
    """
    rng = np.random.default_rng(seed)
    cut_values = exact_info["cut_values"]
    c_max = exact_info["c_max"]
    opt_mask = exact_info["opt_mask"]

    thetas = None
    if mode == "warm_start":
        thetas, _ = compute_warm_start_angles(G, epsilon=epsilon, seed=seed)

    history = []

    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        qc = build_qaoa_circuit(G, p, gammas, betas, mode=mode, thetas=thetas, measure=False)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        exp_cut = float(np.dot(probs, cut_values))
        history.append(exp_cut / c_max)
        return -exp_cut

    avg_deg = max(1.0, 2.0 * G.number_of_edges() / G.number_of_nodes())
    gamma_scale = 0.6 / np.sqrt(avg_deg)

    if mode == "warm_start":
        init_gammas = np.linspace(0.15, 0.35, p) * gamma_scale + rng.normal(0, 0.05, size=p)
        init_betas = np.linspace(0.15, 0.30, p) + rng.normal(0, 0.05, size=p)
    else:
        init_gammas = np.linspace(0.20, 0.45, p) * gamma_scale + rng.normal(0, 0.06, size=p)
        init_betas = np.linspace(0.15, 0.35, p) + rng.normal(0, 0.05, size=p)
    x0 = np.concatenate([init_gammas, init_betas])

    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": maxiter, "tol": 1e-4, "rhobeg": 0.25},
    )

    best_params = res.x
    best_gammas = best_params[:p]
    best_betas = best_params[p:]

    # Final evaluation on Ideal Statevector
    qc_opt = build_qaoa_circuit(G, p, best_gammas, best_betas, mode=mode, thetas=thetas, measure=False)
    probs = Statevector.from_instruction(qc_opt).probabilities()

    expected_cut = float(np.dot(probs, cut_values))
    ar_ideal = expected_cut / c_max
    p_opt_ideal = float(np.sum(probs[opt_mask]))

    # Number of iterations to reach within 99.5% of final converged AR
    history_arr = np.array(history)
    best_so_far = np.maximum.accumulate(history_arr)
    target_ar = best_so_far[-1] * 0.995
    reached_indices = np.where(best_so_far >= target_ar)[0]
    n_iter_effective = int(reached_indices[0] + 1) if len(reached_indices) > 0 else len(history_arr)
    n_iter_total = min(int(res.nfev), maxiter)
    # Group 9's Convergence Speed formula: S := 1 - N_iter / 100
    conv_speed_s = max(0.0, 1.0 - (n_iter_effective / 100.0))

    return {
        "mode": mode,
        "p": p,
        "epsilon": epsilon if mode == "warm_start" else 0.5,
        "ar_ideal": ar_ideal,
        "p_opt_ideal": p_opt_ideal,
        "expected_cut_ideal": expected_cut,
        "n_iter": n_iter_effective,
        "n_fev": n_iter_total,
        "conv_speed_s": conv_speed_s,
        "history": history_arr,
        "best_gammas": best_gammas,
        "best_betas": best_betas,
        "thetas": thetas,
    }


def execute_noisy_simulation(
    G: nx.Graph,
    exact_info: dict,
    opt_result: dict,
    noise_model: NoiseModel,
    shots: int = 4096,
    seed: int = 42,
) -> dict:
    """
    Execute the optimized QAOA / Warm-Start QAOA circuit on IBM-calibrated Noisy Aer
    with 4,096 shots (matching Group 9's exact setup).
    """
    p = opt_result["p"]
    mode = opt_result["mode"]
    thetas = opt_result["thetas"]
    gammas = opt_result["best_gammas"]
    betas = opt_result["best_betas"]

    qc = build_qaoa_circuit(G, p, gammas, betas, mode=mode, thetas=thetas, measure=True)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    t_qc = transpile(qc, sim, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=seed)
    result = sim.run(t_qc, shots=shots).result()
    counts = result.get_counts()

    cut_values = exact_info["cut_values"]
    c_max = exact_info["c_max"]
    opt_mask = exact_info["opt_mask"]

    total_cut = 0.0
    opt_hits = 0
    for bitstr, count in counts.items():
        state_idx = int(bitstr.replace(" ", ""), 2)
        total_cut += count * cut_values[state_idx]
        if opt_mask[state_idx]:
            opt_hits += count

    ar_noisy = (total_cut / shots) / c_max
    p_opt_noisy = opt_hits / shots

    return {
        "ar_noisy": float(ar_noisy),
        "p_opt_noisy": float(p_opt_noisy),
        "noise_loss": float(opt_result["ar_ideal"] - ar_noisy),
    }
