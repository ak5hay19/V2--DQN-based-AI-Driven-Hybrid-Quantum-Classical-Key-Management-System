"""
Train_qlearning.py — Tabular Q-learning (kept for comparison with DQN).
"""

import argparse, numpy as np, pickle
import hybrid_env
from hybrid_env import HybridKeyEnv, NUM_ACTIONS

parser = argparse.ArgumentParser()
parser.add_argument("--use-lstm", action="store_true")
parser.add_argument("--episodes", type=int, default=1000)
args = parser.parse_args()

STATE_DIM = 7
predictor = None
if args.use_lstm:
    try:
        from lstm_threat_predictor import ThreatPredictor
        predictor = ThreatPredictor("lstm_threat.pt")
        STATE_DIM = 8
        print("LSTM loaded (8-dim)")
    except: pass

NUM_BINS = 10; ALPHA = 0.15; GAMMA = 0.95
EPSILON = 1.0; EPS_MIN = 0.05; EPS_DECAY = 0.995
MAX_STEPS = 200; OPTIMISTIC = 0.1
AUG_LOW, AUG_MID, AUG_HIGH = 0.15, 0.25, 0.35

bins = [np.linspace(0,1,NUM_BINS)] * STATE_DIM
def discretize(s):
    return tuple(int(np.clip(np.digitize(s[i],bins[i])-1,0,NUM_BINS-1)) for i in range(min(len(s),STATE_DIM)))

env = HybridKeyEnv("state_vectors.csv"); Q = {}
episode_rewards = []; threat_history = []

print(f"Training Q-learning... ({args.episodes} eps, {STATE_DIM}-dim)")
print(f"{'Ep':>6}  {'Reward':>8}  {'Eps':>6}  {'Q-size':>8}  {'Avg50':>8}")
print("-" * 46)

for ep in range(args.episodes):
    state = env.reset(); threat_history.clear()
    r = np.random.rand()
    if r < AUG_LOW: hybrid_env.FORCE_THREAT = np.random.uniform(0.05,0.25)
    elif r < AUG_LOW+AUG_MID: hybrid_env.FORCE_THREAT = np.random.uniform(0.25,0.55)
    elif r < AUG_LOW+AUG_MID+AUG_HIGH: hybrid_env.FORCE_THREAT = np.random.uniform(0.55,1.0)
    else: hybrid_env.FORCE_THREAT = None
    if predictor:
        threat_history.append(float(state[0]))
        env.predicted_threat = predictor.predict(threat_history)
        state = env._get_state()
    sd = discretize(state); total = 0.0
    for step in range(MAX_STEPS):
        if sd not in Q: Q[sd] = np.full(NUM_ACTIONS, OPTIMISTIC)
        action = np.random.randint(NUM_ACTIONS) if np.random.rand() < EPSILON else int(np.argmax(Q[sd]))
        ns, reward, done, _ = env.step(action)
        if predictor:
            threat_history.append(float(ns[0]))
            env.predicted_threat = predictor.predict(threat_history)
            ns = env._get_state()
        nd = discretize(ns)
        if nd not in Q: Q[nd] = np.full(NUM_ACTIONS, OPTIMISTIC)
        Q[sd][action] += ALPHA * (reward + GAMMA * np.max(Q[nd]) - Q[sd][action])
        sd = nd; total += reward
        if done: break
    EPSILON = max(EPSILON * EPS_DECAY, EPS_MIN)
    episode_rewards.append(total)
    if ep % 100 == 0 or ep == args.episodes-1:
        avg = np.mean(episode_rewards[-50:]) if len(episode_rewards)>=50 else np.mean(episode_rewards)
        print(f"{ep:>6}  {total:>8.1f}  {EPSILON:>6.3f}  {len(Q):>8}  {avg:>8.1f}")

hybrid_env.FORCE_THREAT = None
with open("q_table.pkl","wb") as f: pickle.dump(Q,f)
np.save("rewards_log.npy", np.array(episode_rewards))
print(f"\nDone. Q-table: {len(Q):,} states. Saved q_table.pkl + rewards_log.npy")
