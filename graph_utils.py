"""
Graph Generation, Exact MaxCut Solver, and Structural Metrics Module
Aligned with NTU IBM Qiskit Hackathon Group 9 Benchmark & Warm-Start QAOA Extension.
"""

import numpy as np
import networkx as nx


def generate_eight_graph_families(n: int = 10, seed: int = 42) -> dict[str, nx.Graph]:
    """
    Generate the 8 representative graph classes (n=10 nodes) used in Group 9's benchmark:
    1. Path Graph
    2. Star Graph
    3. Cycle Graph
    4. Tree Graph (Random Prüfer / degree-varied tree)
    5. Complete Graph
    6. Bipartite Graph (Balanced 5-5 with dense edge probability p=0.75)
    7. Community Graph (Stochastic Block Model, two communities of 5)
    8. Random Graph (Erdős-Rényi G(n, p=0.5))
    """
    graphs = {}

    graphs["Path"] = nx.path_graph(n)
    graphs["Star"] = nx.star_graph(n - 1)
    graphs["Cycle"] = nx.cycle_graph(n)
    graphs["Tree"] = nx.random_labeled_tree(n, seed=seed)
    graphs["Complete"] = nx.complete_graph(n)

    n1 = n // 2
    n2 = n - n1
    G_bip = nx.bipartite.random_graph(n1, n2, p=0.75, seed=seed)
    if not nx.is_connected(G_bip):
        for i in range(n1):
            G_bip.add_edge(i, n1 + (i % n2))
    graphs["Bipartite"] = G_bip

    sizes = [n // 2, n - n // 2]
    probs = [[0.8, 0.25], [0.25, 0.8]]
    G_comm = nx.stochastic_block_model(sizes, probs, seed=seed)
    if not nx.is_connected(G_comm):
        G_comm.add_edge(0, n // 2)
    graphs["Community"] = G_comm

    G_rand = nx.erdos_renyi_graph(n, p=0.5, seed=seed)
    while not nx.is_connected(G_rand):
        seed += 1
        G_rand = nx.erdos_renyi_graph(n, p=0.5, seed=seed)
    graphs["Random"] = G_rand

    return graphs


def solve_exact_maxcut(G: nx.Graph) -> dict:
    """
    Vectorized brute-force Exact MaxCut solver for small graphs (n <= 18).
    Returns:
        c_max: Maximum cut value
        cut_values: Array of shape (2^n,) with cut value for each integer state 0..(2^n - 1)
                    Note: Qiskit uses little-endian bit ordering where qubit i corresponds to bit (state >> i) & 1.
        opt_mask: Boolean mask of shape (2^n,) where cut_values == c_max
        degeneracy: Number of optimal bitstrings D = #{x : C(x) = C_max}
        best_bits: One optimal bit assignments array of shape (n,) in {0, 1}
    """
    n = G.number_of_nodes()
    states = np.arange(1 << n, dtype=np.uint32)
    bits = ((states[None, :] >> np.arange(n, dtype=np.uint32)[:, None]) & 1).astype(np.int8)

    cut_values = np.zeros(1 << n, dtype=np.float64)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        cut_values += w * (bits[u] ^ bits[v])

    c_max = float(np.max(cut_values))
    opt_mask = np.isclose(cut_values, c_max)
    degeneracy = int(np.sum(opt_mask))
    best_idx = int(np.argmax(cut_values))
    best_bits = bits[:, best_idx].astype(float)

    return {
        "c_max": c_max,
        "cut_values": cut_values,
        "opt_mask": opt_mask,
        "degeneracy": degeneracy,
        "best_bits": best_bits,
    }


def compute_graph_metrics(G: nx.Graph, exact_info: dict = None) -> dict:
    """
    Compute the 6 Advanced Graph Structural Metrics defined in Group 9's slides:
    1. Edge Density = 2|E| / (n(n-1))
    2. Degree Variance = Var(deg(v))
    3. Clustering Coefficient = 3 * #triangles / sum_v binom(deg(v), 2)
    4. Odd-cycle Structure (#triangles)
    5. Symmetry = (n - #orbits) / (n - 1) via exact automorphism orbit pinning
    6. Optimal-solution Degeneracy D = #{x : C(x) = C_max}
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edge_density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0

    degrees = np.array([d for _, d in G.degree()], dtype=float)
    degree_variance = float(np.var(degrees))

    tri_dict = nx.triangles(G)
    num_triangles = int(sum(tri_dict.values()) // 3)
    clustering_coeff = float(nx.transitivity(G))

    # Exact automorphism orbits: u ~ v iff some automorphism maps u -> v.
    # (Pin u in one copy and v in the other, then test isomorphism.)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nodes = list(G.nodes())
    deg = dict(G.degree())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if find(u) == find(v) or deg[u] != deg[v]:
                continue
            G1, G2 = G.copy(), G.copy()
            nx.set_node_attributes(G1, {x: int(x == u) for x in nodes}, "pin")
            nx.set_node_attributes(G2, {x: int(x == v) for x in nodes}, "pin")
            if nx.is_isomorphic(G1, G2, node_match=lambda a, b: a["pin"] == b["pin"]):
                parent[find(u)] = find(v)
    num_orbits = len(set(find(i) for i in nodes))
    symmetry = float(n - num_orbits) / (n - 1) if n > 1 else 0.0

    if exact_info is None:
        exact_info = solve_exact_maxcut(G)

    return {
        "n": n,
        "edges": m,
        "edge_density": edge_density,
        "degree_variance": degree_variance,
        "clustering_coeff": clustering_coeff,
        "triangles": num_triangles,
        "symmetry": symmetry,
        "degeneracy": exact_info["degeneracy"],
        "c_max": exact_info["c_max"],
    }
