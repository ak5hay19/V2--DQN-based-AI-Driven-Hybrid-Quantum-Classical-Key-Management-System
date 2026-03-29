"""
qberskr.py
==========
key.csv + network_features.csv -> state_vectors.csv

Derives QBER, SKR from QKD quadrature data, computes PQC overhead,
carries threat_category through from feature_extraction.
"""

import pandas as pd
import numpy as np

net = pd.read_csv("network_features.csv")
N   = len(net)
print(f"Loaded network_features.csv  ({N:,} rows)")

if "threat_category" not in net.columns:
    print("  WARNING: threat_category not found - run feature_extraction.py first")
    net["threat_category"] = 1

key = pd.read_csv("key.csv")
print(f"Loaded key.csv  ({len(key):,} rows)")
key = key.sample(n=N, replace=True, random_state=42).reset_index(drop=True)

# Rolling correlation for per-row QBER variation
WINDOW = 50
rho = pd.Series(key["x_pe"].values).rolling(WINDOW, min_periods=10).corr(
    pd.Series(key["y_pe"].values))
rho = rho.fillna(method="bfill").fillna(0.5).values
rho = np.clip(rho, 0.01, 0.99)

excess = (1 - rho) + np.random.normal(0, 0.015, N)
qber = np.clip(0.5 * np.clip(excess, 0.005, 0.20), 0.005, 0.12)

def H(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

skr = np.maximum(0, 1 - 2 * H(qber))
skr = skr / (skr.max() + 1e-6)

pkt_norm = (net["packet_load"] - net["packet_load"].min()) / \
           (net["packet_load"].max() - net["packet_load"].min() + 1e-6)
pqc_overhead = np.clip(
    (0.40 + 0.45 * pkt_norm + 0.18 * net["threat_score"]) *
    np.random.normal(1.0, 0.15, N), 0.3, 0.9)

state = pd.DataFrame({
    "QBER": qber, "SKR": skr,
    "avg_latency": net["avg_latency"], "packet_load": net["packet_load"],
    "threat_score": net["threat_score"], "PQC_overhead": pqc_overhead,
    "threat_category": net["threat_category"].astype(int),
})

state.to_csv("state_vectors.csv", index=False)
print(f"\nSaved state_vectors.csv  ({len(state):,} rows)")
for col in state.columns:
    print(f"  {col:18s}  min={state[col].min():.4f}  max={state[col].max():.4f}  "
          f"mean={state[col].mean():.4f}")
