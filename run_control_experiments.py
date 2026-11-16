"""
Control experiments for the Warm-Start QAOA claims (no information leakage).

Exp C1  Hint-source ablation on the full-2Y graphs.
        Warm-start with: (a) exact/SDP-best hint (the original setup),
        (b) a weak greedy hint, (c) a random hint (zero information),
        versus cold-start QAOA and a purely classical "4096 uniform samples" baseline.
        Question: how much of the P_opt gain comes from the classical hint itself?

Exp C2  Real regime shift with a *changed graph*.
        Rolling windows: hint = optimum of the train window's graph,
        evaluated on the NEXT window's graph (its own exact optimum / c_max).
        Baselines: stale hint, 1-flip hill climbing from the stale hint (classical, ~free),
        classical re-solve (AR=1 by construction), cold QAOA, regularised WS-QAOA p=1/2.
        ALL windows are reported (no subset selection).

Usage:  python run_control_experiments.py            # full
        python run_control_experiments.py --quick    # tiny smoke test
Outputs: results/exp_c1_hint_ablation.csv, results/exp_c2_rolling_shift.csv
Ideal statevector only (no noise) to keep this fast; the question here is the hint, not the noise.
"""

import os
import sys
import numpy as np
import pandas as pd

from graph_utils import solve_exact_maxcut
from run_financial_warmstart_study import (
    TW_TICKERS, US_TICKERS, build_hedging_graph, run_single_maxcut_trial,
)

QUICK = "--quick" in sys.argv
OUT = "results"
SHOTS = 4096


def bits_to_state(bits):
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def cut_of_bits(ex, bits):
    return float(ex["cut_values"][bits_to_state(bits)])


def greedy_hint(G, rng):
    """Single-pass randomised greedy MaxCut (weak classical heuristic)."""
    n = G.number_of_nodes()
    side = -np.ones(n)
    for u in rng.permutation(n):
        gain0 = sum(d.get("weight", 1.0) for v, d in G[u].items() if side[v] == 1)
        gain1 = sum(d.get("weight", 1.0) for v, d in G[u].items() if side[v] == 0)
        side[u] = 0 if gain0 >= gain1 else 1
    return side


def hill_climb(ex, bits):
    """Steepest-ascent 1-flip local search using the exact cut table (pure classical)."""
    bits = bits.copy().astype(int)
    n = len(bits)
    cur = cut_of_bits(ex, bits)
    while True:
        best_i, best_v = None, cur
        for i in range(n):
            bits[i] ^= 1
            v = cut_of_bits(ex, bits)
            bits[i] ^= 1
            if v > best_v + 1e-12:
                best_i, best_v = i, v
        if best_i is None:
            return bits, cur
        bits[best_i] ^= 1
        cur = best_v


def hit_prob(p_single, shots=SHOTS):
    """Probability that at least one of `shots` samples is optimal."""
    return 1.0 - (1.0 - p_single) ** shots


def load_returns():
    tw = pd.read_csv(os.path.join(OUT, "data_tw_core_12_returns.csv"), index_col=0, parse_dates=True)
    us = pd.read_csv(os.path.join(OUT, "data_us_macro_12_returns.csv"), index_col=0, parse_dates=True)
    return {"TW-Core-12": (tw, TW_TICKERS), "US-Macro-12": (us, US_TICKERS)}


def exp_c1(markets):
    rows = []
    seeds = [42] if QUICK else [42, 43, 44, 45, 46]
    for name, (ret, tick) in markets.items():
        G = build_hedging_graph(ret, tick)
        ex = solve_exact_maxcut(G)
        n = G.number_of_nodes()
        deg = ex["degeneracy"]
        p_uniform = deg / (2 ** n)
        rows.append(dict(market=name, method="Classical: uniform random sampling", seed=-1,
                         ar=np.nan, p_opt=p_uniform, hit_in_4096=hit_prob(p_uniform)))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            hints = {
                "WS hint = exact optimum (original setup)": ex["best_bits"],
                "WS hint = single greedy pass": greedy_hint(G, rng),
                "WS hint = random bits (no information)": rng.integers(0, 2, size=n).astype(float),
            }
            r = run_single_maxcut_trial(G, ex, variant="standard", p=1, maxiter=80, seed=seed)
            rows.append(dict(market=name, method="Cold-start QAOA p=1", seed=seed,
                             ar=r["ar_ideal"], p_opt=r["p_opt_ideal"], hit_in_4096=hit_prob(r["p_opt_ideal"])))
            for label, h in hints.items():
                hint_ar = cut_of_bits(ex, h) / ex["c_max"]
                r = run_single_maxcut_trial(G, ex, variant="warm_start", p=1, epsilon=0.15,
                                            custom_c_star=h, maxiter=80, seed=seed)
                rows.append(dict(market=name, method=f"{label} | QAOA p=1", seed=seed,
                                 ar=r["ar_ideal"], p_opt=r["p_opt_ideal"], hit_in_4096=hit_prob(r["p_opt_ideal"])))
                rows.append(dict(market=name, method=f"{label} | hint alone (no QAOA)", seed=seed,
                                 ar=hint_ar, p_opt=float(hint_ar >= 1 - 1e-9), hit_in_4096=float(hint_ar >= 1 - 1e-9)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp_c1_hint_ablation.csv"), index=False)
    return df


def exp_c2(markets):
    rows = []
    win, step = 120, 40
    seeds = [42] if QUICK else [42, 43]
    for name, (ret, tick) in markets.items():
        starts = list(range(0, len(ret) - 2 * win + 1, step))
        if QUICK:
            starts = starts[:1]
        for w, s in enumerate(starts):
            tr, te = ret.iloc[s:s + win], ret.iloc[s + win:s + 2 * win]
            G_tr, G_te = build_hedging_graph(tr, tick), build_hedging_graph(te, tick)
            ex_tr, ex_te = solve_exact_maxcut(G_tr), solve_exact_maxcut(G_te)
            stale = ex_tr["best_bits"]
            stale_ar = cut_of_bits(ex_te, stale) / ex_te["c_max"]
            ham = int(min(np.sum(stale != ex_te["best_bits"]), np.sum(stale == ex_te["best_bits"])))
            _, hc_cut = hill_climb(ex_te, stale)
            base = dict(market=name, window=w, hamming_to_new_opt=ham,
                        test_start=str(te.index[0].date()))
            rows.append({**base, "method": "Stale hint (p=0)", "seed": -1, "ar": stale_ar})
            rows.append({**base, "method": "Classical: 1-flip hill climb from stale hint", "seed": -1,
                         "ar": hc_cut / ex_te["c_max"]})
            rows.append({**base, "method": "Classical: re-solve on new graph", "seed": -1, "ar": 1.0})
            for seed in seeds:
                r = run_single_maxcut_trial(G_te, ex_te, variant="standard", p=1, maxiter=80, seed=seed)
                rows.append({**base, "method": "Cold-start QAOA p=1", "seed": seed, "ar": r["ar_ideal"]})
                for p in (1, 2):
                    r = run_single_maxcut_trial(G_te, ex_te, variant="warm_start", p=p, epsilon=0.20,
                                                custom_c_star=stale, maxiter=80, seed=seed)
                    rows.append({**base, "method": f"Reg. WS-QAOA eps=0.20 p={p}", "seed": seed, "ar": r["ar_ideal"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp_c2_rolling_shift.csv"), index=False)
    return df


def main():
    markets = load_returns()
    print("[C1] hint-source ablation ...")
    c1 = exp_c1(markets)
    print(c1.groupby(["market", "method"])[["ar", "p_opt", "hit_in_4096"]].mean().round(4).to_string())
    print("\n[C2] real rolling regime shift (changed graph, all windows) ...")
    c2 = exp_c2(markets)
    summ = c2.groupby(["market", "method"]).ar.agg(["mean", "std", "count"]).round(4)
    print(summ.to_string())
    stale = c2[c2.method == "Stale hint (p=0)"].groupby(["market", "window"]).ar.mean()
    for m in c2.method.unique():
        if m.startswith("Reg.") or m.startswith("Cold") or m.startswith("Classical: 1-flip"):
            g = c2[c2.method == m].groupby(["market", "window"]).ar.mean()
            gain = (g - stale).dropna()
            print(f"gain vs stale | {m}: mean={gain.mean():+.4f}  windows>0: {(gain > 0).sum()}/{len(gain)}")


if __name__ == "__main__":
    main()
