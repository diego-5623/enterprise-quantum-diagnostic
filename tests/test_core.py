import os
import sys
import unittest
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_utils import solve_exact_maxcut, compute_graph_metrics
from qaoa_core import compute_warm_start_angles, build_qaoa_circuit


class TestCore(unittest.TestCase):
    def test_complete_graph_symmetry(self):
        G = nx.complete_graph(10)
        metrics = compute_graph_metrics(G)
        self.assertTrue(np.isclose(metrics["symmetry"], 1.0), f"Expected 1.0 for K10, got {metrics['symmetry']}")

    def test_star_graph_symmetry(self):
        G = nx.star_graph(9)  # 10 nodes -> 2 orbits -> (10-2)/9 = 8/9
        metrics = compute_graph_metrics(G)
        self.assertTrue(np.isclose(metrics["symmetry"], 8.0 / 9.0), f"Expected 8/9 for Star10, got {metrics['symmetry']}")

    def test_exact_maxcut_bipartite(self):
        G = nx.complete_bipartite_graph(4, 4)
        ex = solve_exact_maxcut(G)
        self.assertTrue(np.isclose(ex["c_max"], 16.0))
        self.assertEqual(ex["degeneracy"], 2)

    def test_warm_start_regularization_bounds(self):
        G = nx.cycle_graph(6)
        eps = 0.15
        thetas, c_tilde = compute_warm_start_angles(G, epsilon=eps, seed=42)
        self.assertTrue(np.all(c_tilde >= eps - 1e-9) and np.all(c_tilde <= 1.0 - eps + 1e-9))
        self.assertEqual(thetas.shape, (6,))

    def test_aligned_circuit_zero_extra_cx(self):
        G = nx.cycle_graph(6)
        thetas, _ = compute_warm_start_angles(G, epsilon=0.15, seed=42)
        qc_std = build_qaoa_circuit(G, p=1, gammas=np.array([0.3]), betas=np.array([0.2]), mode="standard")
        qc_ws = build_qaoa_circuit(G, p=1, gammas=np.array([0.3]), betas=np.array([0.2]), mode="warm_start", thetas=thetas)
        self.assertEqual(qc_std.count_ops().get("rzz", 0), qc_ws.count_ops().get("rzz", 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
