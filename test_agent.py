"""
test_agent.py — Evaluate trained agent across four threat regimes.
Supports both Q-table and DQN models.

Usage:
    python test_agent.py                # auto-detects dqn_model.pt or q_table.pkl
    python test_agent.py --dqn          # force DQN
    python test_agent.py --qtable       # force Q-table
    python test_agent.py --use-lstm     # with LSTM
    python test_agent.py --steps 100    # more steps per regime
"""

import argparse, os, numpy as np, pickle
import hybrid_env
from hybrid_env import HybridKeyEnv, ACTION_NAMES, NUM_ACTIONS

parser = argparse.ArgumentParser()
parser.add_argument("--dqn", action="store_true", help="Force DQN model")
parser.add_argument("--qtable", action="store_true", help="Force Q-table")
parser.add_argument("--use-lstm", action="store_true")
parser.add_argument("--steps", type=int, default=50)
args = parser.parse_args()

STATE_DIM = 7
predictor = None
if args.use_lstm:
    try:
        from lstm_threat_predictor import ThreatPredictor
        predictor = ThreatPredictor("lstm_threat.pt"); STATE_DIM = 8
        print("LSTM loaded (8-dim)")
    except Exception as e: print(f"LSTM not available: {e}")

# --- Load agent ---
use_dqn = False
if args.dqn or (not args.qtable and os.path.exists("dqn_model.pt")):
    from dqn_agent import DQNAgent
    agent = DQNAgent(state_dim=STATE_DIM, num_actions=NUM_ACTIONS)
    agent.load("dqn_model.pt")
    use_dqn = True
    print(f"Using DQN model ({STATE_DIM}-dim)")
elif os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f: Q = pickle.load(f)
    NUM_BINS = 10
    bins = [np.linspace(0,1,NUM_BINS)] * STATE_DIM
    def discretize(s):
        return tuple(int(np.clip(np.digitize(s[i],bins[i])-1,0,NUM_BINS-1)) for i in range(min(len(s),STATE_DIM)))
    print(f"Using Q-table ({len(Q):,} states, {STATE_DIM}-dim)")
else:
    print("ERROR: No model found. Run Train_dqn.py or Train_qlearning.py first.")
    exit(1)

def _fallback(threat):
    if threat < 0.25: return 0
    elif threat < 0.55: return 1
    else: return 2

THREAT_LEVELS = [0.05, 0.30, 0.60, 0.85]
results = []

for forced in THREAT_LEVELS:
    print(f"\n{'='*72}\n  FORCED THREAT = {forced:.2f}\n{'='*72}")
    hybrid_env.FORCE_THREAT = forced
    env = HybridKeyEnv("state_vectors.csv")
    state = env.reset()
    threat_history = [float(state[0])]
    if predictor:
        env.predicted_threat = predictor.predict(threat_history)
        state = env._get_state()

    total_reward = 0.0
    action_counts = {i:0 for i in range(NUM_ACTIONS)}
    unknown = 0

    for step in range(args.steps):
        if use_dqn:
            action = agent.act(state, training=False)
            known = True
        else:
            sd = discretize(state)
            if sd in Q:
                action = int(np.argmax(Q[sd])); known = True
            else:
                action = _fallback(forced); known = False; unknown += 1

        ns, reward, done, info = env.step(action)
        if predictor:
            threat_history.append(float(ns[0]))
            env.predicted_threat = predictor.predict(threat_history)
            ns = env._get_state()

        total_reward += reward
        action_counts[action] += 1
        print(f"Step {step+1:02d} | {ACTION_NAMES[action]:30s} | "
              f"Reward: {reward:6.3f} | Threat: {forced:.2f} | "
              f"Sec: {info['security_norm']:.2f} | KeyAge: {info['key_age']}")
        state = ns
        if done: break

    print(f"\nSummary (threat={forced:.2f})  Total: {total_reward:.3f}  "
          f"Unknown: {unknown}/{args.steps}")
    for a,c in action_counts.items():
        print(f"  {ACTION_NAMES[a]:30s}: {c:3d}  {'█'*c}")
    results.append({"threat":forced, "total":total_reward, "actions":action_counts.copy(), "unknown":unknown})

hybrid_env.FORCE_THREAT = None
print(f"\n{'='*72}\n  CROSS-REGIME SUMMARY\n{'='*72}")
print(f"  {'Threat':>8}  {'Reward':>10}  {'Unknown':>8}  Top Action")
print("  " + "-"*55)
for r in results:
    top = max(r["actions"], key=r["actions"].get)
    print(f"  {r['threat']:.2f}     {r['total']:>10.3f}    {r['unknown']:>3}/{args.steps}    {ACTION_NAMES[top]}")
print(f"\n  Expected: 0.05->Classical/Reuse  0.30->PQC  0.60->PQC+someQKD  0.85->QKD+PQC\n")
