"""
realtime_pipeline.py — Real-time threat detection and key switching.

Modes: --replay (CSV simulation) / --live (scapy capture)
Agent: auto-detects dqn_model.pt or q_table.pkl
"""

import argparse, time, os, numpy as np, pandas as pd, pickle, sys
from collections import deque
from hybrid_env import ACTION_NAMES, NUM_ACTIONS

parser = argparse.ArgumentParser(description="SmartKeyNet Real-Time Pipeline")
parser.add_argument("--replay", action="store_true", default=True)
parser.add_argument("--live", action="store_true")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--speed", type=float, default=0.0)
parser.add_argument("--duration", type=int, default=60)
parser.add_argument("--csv", type=str, default="RT_IOT2022.csv")
parser.add_argument("--use-lstm", action="store_true")
parser.add_argument("--dqn", action="store_true", help="Force DQN")
parser.add_argument("--qtable", action="store_true", help="Force Q-table")
args = parser.parse_args()
if args.live: args.replay = False


class ThreatScorer:
    def __init__(self, path="threat_model.pkl"):
        with open(path,"rb") as f: b = pickle.load(f)
        self.clf, self.scaler = b["classifier"], b["scaler"]
        self.feature_names, self.rl_norm = b["feature_names"], b["rl_norm_params"]
        self.n_classes = b.get("n_classes",2); self._checked = False
        print(f"  ThreatScorer ({len(self.feature_names)} features, {self.n_classes}-class)")

    def score(self, row):
        if not self._checked:
            self._checked = True
            miss = [f for f in self.feature_names if f not in row]
            if miss: print(f"  WARNING: {len(miss)} features missing: {miss[:5]}")
            else: print(f"  Feature check OK: all {len(self.feature_names)} matched")
        feats = {}
        for f in self.feature_names:
            v = row.get(f,0.0)
            feats[f] = [float(0.0 if v is None or (isinstance(v,float) and np.isnan(v)) else v)]
        proba = self.clf.predict_proba(self.scaler.transform(pd.DataFrame(feats)))[0]
        if self.n_classes == 2: return float(np.clip(proba[1],0,1))
        return float(np.clip(np.dot(proba, np.linspace(0,1,self.n_classes)),0,1))


class RLAgent:
    def __init__(self, state_dim=7, use_dqn=None):
        self.state_dim = state_dim
        # Auto-detect model
        if use_dqn or (use_dqn is None and os.path.exists("dqn_model.pt")):
            from dqn_agent import DQNAgent
            self.agent = DQNAgent(state_dim=state_dim, num_actions=NUM_ACTIONS)
            self.agent.load("dqn_model.pt")
            self.use_dqn = True
            print(f"  RLAgent: DQN ({state_dim}-dim)")
        elif os.path.exists("q_table.pkl"):
            with open("q_table.pkl","rb") as f: self.Q = pickle.load(f)
            self.bins = [np.linspace(0,1,10)] * state_dim
            self.use_dqn = False
            print(f"  RLAgent: Q-table ({len(self.Q):,} states, {state_dim}-dim)")
        else:
            print("  WARNING: No model found, using heuristic only")
            self.use_dqn = False; self.Q = {}; self.bins = [np.linspace(0,1,10)] * state_dim

    def act(self, state):
        if self.use_dqn:
            return self.agent.act(state, training=False), True
        sd = tuple(int(np.clip(np.digitize(state[i],self.bins[i])-1,0,9)) for i in range(min(len(state),self.state_dim)))
        if sd in self.Q: return int(np.argmax(self.Q[sd])), True
        t = state[0]
        return (0 if t<0.25 else 1 if t<0.55 else 2), False


class FeatureEng:
    def __init__(self, norm):
        self.lat_min,self.lat_max = norm["latency_min"],norm["latency_max"]
        self.pkt_min,self.pkt_max = norm["pktload_min"],norm["pktload_max"]
    def compute(self, row):
        lat = np.clip((np.log1p(float(row.get("payload_bytes_per_second",0)))-self.lat_min)/(self.lat_max-self.lat_min+1e-9),0,1)
        pkt = np.clip((float(row.get("fwd_pkts_tot",0))+float(row.get("bwd_pkts_tot",0))-self.pkt_min)/(self.pkt_max-self.pkt_min+1e-9),0,1)
        return {"avg_latency":float(lat),"packet_load":float(pkt)}


def replay_gen(csv, steps):
    df = pd.read_csv(csv).sample(frac=1, random_state=np.random.randint(10000)).reset_index(drop=True)
    for i in range(min(steps, len(df))):
        r = df.iloc[i].to_dict(); r["_ground_truth"] = r.get("Attack_type","?"); yield r


def main():
    print(f"\n{'='*70}\n  SmartKeyNet Real-Time Pipeline\n{'='*70}\nLoading:")
    scorer = ThreatScorer(); feat = FeatureEng(scorer.rl_norm)
    sd = 8 if args.use_lstm else 7
    predictor = None; th = deque(maxlen=25)
    if args.use_lstm:
        try:
            from lstm_threat_predictor import ThreatPredictor
            predictor = ThreatPredictor("lstm_threat.pt"); print(f"  LSTM loaded")
        except: pass
    use_dqn = True if args.dqn else (False if args.qtable else None)
    agent = RLAgent(sd, use_dqn)
    data = replay_gen(args.csv, args.steps) if args.replay else None

    print(f"\n{'Step':>4}  {'Threat':>6}  {'Action':>30}  {'Known':>5}  {'KeyAge':>6}  {'Label'}")
    print("-"*95)
    ac = {i:0 for i in range(NUM_ACTIONS)}; threats=[]; key_age=0; total=0

    for row in data:
        net = feat.compute(row); threat = scorer.score(row); threats.append(threat); th.append(threat)
        qber = float(np.clip(0.04+np.random.normal(0,0.008),0.005,0.12))
        skr = float(np.clip(0.85-2*(qber-0.03)+np.random.normal(0,0.02),0.3,1.0))
        pqc = float(np.clip((0.4+0.45*net["packet_load"]+0.18*threat)*np.random.normal(1,0.08),0.3,0.9))
        pred = predictor.predict(list(th)) if predictor and len(th)>=5 else None
        state = np.array([threat,qber,skr,net["packet_load"],net["avg_latency"],pqc,min(key_age/10,1)],dtype=np.float32)
        if pred is not None: state = np.append(state, float(pred))
        action, known = agent.act(state)
        ac[action]+=1; total+=1
        if action==4: key_age=0
        else: key_age+=1
        gt = str(row.get("_ground_truth",""))[:20]
        print(f"{total:>4}  {threat:>6.3f}  {ACTION_NAMES[action]:>30}  {'Y' if known else 'N':>5}  {key_age:>6}  {gt}")
        if args.speed>0: time.sleep(args.speed)

    print(f"\n{'='*70}\n  SUMMARY ({total} steps)\n{'='*70}")
    print(f"  Avg threat: {np.mean(threats):.3f}  Range: [{min(threats):.3f}, {max(threats):.3f}]")
    for a,c in ac.items():
        pct = c/total*100 if total>0 else 0
        print(f"  {ACTION_NAMES[a]:30s}: {c:4d} ({pct:5.1f}%)  {'█'*int(pct/2)}")

if __name__=="__main__": main()
