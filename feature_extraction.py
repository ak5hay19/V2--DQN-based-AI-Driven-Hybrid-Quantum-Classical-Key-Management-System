"""
feature_extraction.py
=====================
RT_IOT2022.csv -> network_features.csv

Output: avg_latency, packet_load, threat_score, threat_category
"""

import pandas as pd
import numpy as np

CATEGORY_NAMES = {0: "benign", 1: "recon", 2: "active_attack"}

HIGH_KEYWORDS   = ["ddos", "dos", "flood", "syn_hping", "exploit",
                   "ransomware", "botnet", "backdoor", "injection",
                   "brute", "password"]
MEDIUM_KEYWORDS = ["scan", "nmap", "recon", "probe", "portscan",
                   "fingerprint", "discovery", "arp", "spoof",
                   "mitm", "os_", "wipro", "bulb"]
LOW_KEYWORDS    = ["mqtt", "thing_speak", "weather", "normal", "benign"]


def map_attack_to_category(attack_type: str) -> int:
    val = str(attack_type).strip().lower().replace(" ", "_")
    if any(kw in val for kw in LOW_KEYWORDS):    return 0
    elif any(kw in val for kw in HIGH_KEYWORDS):  return 2
    elif any(kw in val for kw in MEDIUM_KEYWORDS): return 1
    else: return 1


df = pd.read_csv("RT_IOT2022.csv")
print(f"Loaded RT_IOT2022.csv  ({len(df):,} rows, {len(df.columns)} columns)")

features = pd.DataFrame()
features["avg_latency"]  = np.log1p(df["payload_bytes_per_second"])
features["packet_load"]  = df["fwd_pkts_tot"] + df["bwd_pkts_tot"]
features["threat_score"] = np.log1p(df["flow_pkts_per_sec"])

for col in ["avg_latency", "packet_load", "threat_score"]:
    mn, mx = features[col].min(), features[col].max()
    features[col] = (features[col] - mn) / (mx - mn + 1e-9)

features["threat_category"] = df["Attack_type"].apply(map_attack_to_category)

print(f"\nThreat category mapping:")
for at in df["Attack_type"].unique():
    cat = map_attack_to_category(at)
    cnt = (df["Attack_type"] == at).sum()
    print(f"  {str(at):30s} -> {cat} ({CATEGORY_NAMES[cat]})  ({cnt:,} rows)")

features.to_csv("network_features.csv", index=False)
print(f"\nSaved network_features.csv  ({len(features):,} rows)")
print(f"Columns: {list(features.columns)}")
