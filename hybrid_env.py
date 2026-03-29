"""
hybrid_env.py — Data-driven RL environment for adaptive key distribution.

Rewards computed from crypto_profiles.json benchmarks, scaled by network
state, with security based on threat vulnerability matrix.
"""

import json, os
import numpy as np
import pandas as pd

ACTION_CLASSICAL, ACTION_PQC, ACTION_QKD, ACTION_REUSE, ACTION_REFRESH = 0, 1, 2, 3, 4
NUM_ACTIONS = 5
ACTION_NAMES = {0: "Classical", 1: "PQC", 2: "QKD", 3: "Reuse", 4: "Refresh"}
_PROFILE_KEY = {0: "classical", 1: "pqc", 2: "qkd", 3: "reuse", 4: "refresh"}
_CAT_NAMES = {0: "benign", 1: "recon", 2: "active_attack"}

FORCE_THREAT = None
REWARD_ALPHA, REWARD_BETA, REWARD_GAMMA = 1.2, 0.4, 0.4

def _load_profiles(path="crypto_profiles.json"):
    for p in [path, os.path.join(os.path.dirname(__file__), path)]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return None

PROFILES = _load_profiles()

def _infer_threat_category(threat_score):
    if threat_score < 0.25: return "benign"
    elif threat_score < 0.60: return "recon"
    else: return "active_attack"

def _compute_latency(action, pkt_load, avg_lat, pqc_oh):
    if PROFILES is None:
        return {0:0.16, 1:pqc_oh, 2:0.80, 3:0.04, 4:0.40}.get(action, 0.2)
    P = PROFILES
    ml = P["normalization"]["max_latency_ms"]
    if action == 0:   base = P["classical"]["key_exchange_ms"]
    elif action == 1: base = P["pqc"]["key_exchange_ms"] + pqc_oh * 5.0
    elif action == 2: base = P["qkd"]["key_exchange_ms"]
    elif action == 3: return 0.02
    elif action == 4: base = P["pqc"]["key_exchange_ms"] * 3.0
    else: base = 1.0
    nf = 1.0 + pkt_load * 1.5 + avg_lat * 1.0
    return float(np.clip(base * nf / ml, 0, 1))

def _compute_energy(action, pkt_load, avg_lat):
    if PROFILES is None:
        return {0:0.16, 1:0.40, 2:0.90, 3:0.04, 4:0.40}.get(action, 0.2)
    P = PROFILES
    me = P["normalization"]["max_energy_cycles"]
    if action == 0:   base = P["classical"]["cpu_cycles_keygen"]
    elif action == 1: base = P["pqc"]["cpu_cycles_keygen"]
    elif action == 2: base = P["qkd"]["cpu_cycles_keygen"] + 500000
    elif action == 3: return 0.02
    elif action == 4: base = P["pqc"]["cpu_cycles_keygen"] * 1.5
    else: base = 100000
    tf = 1.0 + avg_lat * 2.0 + pkt_load * 1.0
    return float(np.clip(base * tf / me, 0, 1))

def _compute_security(action, threat, threat_cat, qber, skr, key_age):
    if PROFILES is None:
        if action == 0: return 0.65 * (1.0 - threat) ** 1.3
        if action == 1: return 0.55
        if action == 2: return max(0, threat * skr * 1.5 - (0.3 if qber > 0.08 else 0))
        if action == 3: return max(0, 0.55 - 0.06*key_age) * ((1-threat) if threat>0.4 else 1)
        return 0.50
    P = PROFILES
    vuln = P["threat_vulnerability"]
    pk = _PROFILE_KEY[action]
    if action in (0,1,2):
        base = P[pk]["nist_security_level"] / P[pk]["max_security_level"]
    elif action == 3: base = 0.5
    else: base = 0.55
    vf = vuln[pk].get(threat_cat, 0.5)
    if action == 2:
        qp = max(0, (qber - 0.03) * 5.0)
        quality = (0.5 + 0.5 * skr) * max(0, 1.0 - qp)
    else: quality = 1.0
    if action == 3: age_d = max(0, 1.0 - 0.08 * key_age)
    else: age_d = 1.0
    return float(np.clip(base * vf * quality * age_d, 0, 1))


class HybridKeyEnv:
    def __init__(self, state_csv):
        self.data = pd.read_csv(state_csv)
        self.N = len(self.data)
        self.predicted_threat = None
        self.reset()

    def reset(self):
        self.idx = 0
        self.key_age = 0
        self.current_key_type = 0
        self.done = False
        self.action_counts = {i: 0 for i in range(NUM_ACTIONS)}
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.idx]
        threat = row["threat_score"] if FORCE_THREAT is None else FORCE_THREAT
        state = np.array([float(threat), float(row["QBER"]), float(row["SKR"]),
                          float(row["packet_load"]), float(row["avg_latency"]),
                          float(row["PQC_overhead"]),
                          min(self.key_age / 10.0, 1.0)], dtype=np.float32)
        if self.predicted_threat is not None:
            state = np.append(state, float(self.predicted_threat))
        return state

    def step(self, action):
        if self.done: raise RuntimeError("Episode finished.")
        self.action_counts[action] += 1
        row = self.data.iloc[self.idx]
        threat = float(row["threat_score"] if FORCE_THREAT is None else FORCE_THREAT)
        qber, skr = float(row["QBER"]), float(row["SKR"])
        pkt_load, avg_lat, pqc_oh = float(row["packet_load"]), float(row["avg_latency"]), float(row["PQC_overhead"])

        if FORCE_THREAT is not None:
            threat_cat = _infer_threat_category(threat)
        elif "threat_category" in self.data.columns:
            threat_cat = _CAT_NAMES.get(int(row["threat_category"]), "recon")
        else:
            threat_cat = _infer_threat_category(threat)

        lat = _compute_latency(action, pkt_load, avg_lat, pqc_oh)
        eng = _compute_energy(action, pkt_load, avg_lat)
        sec = _compute_security(action, threat, threat_cat, qber, skr, self.key_age)

        reward = REWARD_ALPHA * sec - REWARD_BETA * lat - REWARD_GAMMA * eng

        if action == ACTION_REFRESH: self.key_age = 0
        else: self.key_age += 1
        self.current_key_type = action
        self.idx += 1
        if self.idx >= self.N - 1: self.done = True

        info = {"action_name": ACTION_NAMES[action], "key_age": self.key_age,
                "threat": threat, "threat_cat": threat_cat, "QBER": qber, "SKR": skr,
                "security_norm": sec, "latency_norm": lat, "energy_norm": eng}
        return self._get_state(), reward, self.done, info

    def print_episode_summary(self, epsilon=None):
        total = sum(self.action_counts.values())
        print("\n================ EPISODE SUMMARY ================")
        for a, c in self.action_counts.items():
            print(f"  {ACTION_NAMES[a]:30s}: {c:4d}  ({c/total*100:5.1f}%)")
        if epsilon: print(f"  Epsilon: {epsilon:.4f}")
        print("=================================================\n")
