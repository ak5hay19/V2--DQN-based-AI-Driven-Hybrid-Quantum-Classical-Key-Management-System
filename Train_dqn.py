"""
Train_dqn.py
============
Trains a DQN agent on HybridKeyEnv.

Usage:
    python Train_dqn.py                      # 7-dim state
    python Train_dqn.py --use-lstm           # 8-dim with LSTM
    python Train_dqn.py --episodes 2000      # more episodes
"""

import argparse
import numpy as np
import hybrid_env
from hybrid_env import HybridKeyEnv, NUM_ACTIONS, ACTION_NAMES
from dqn_agent import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument("--use-lstm", action="store_true")
parser.add_argument("--episodes", type=int, default=1500)
args = parser.parse_args()

STATE_DIM = 7
predictor = None
if args.use_lstm:
    try:
        from lstm_threat_predictor import ThreatPredictor
        predictor = ThreatPredictor("lstm_threat.pt")
        STATE_DIM = 8
        print("LSTM predictor loaded (8-dim state)")
    except Exception as e:
        print(f"LSTM not available ({e}), using 7-dim")

MAX_STEPS = 200
AUG_LOW, AUG_MID, AUG_HIGH = 0.15, 0.25, 0.35

env   = HybridKeyEnv("state_vectors.csv")
agent = DQNAgent(state_dim=STATE_DIM, num_actions=NUM_ACTIONS,
                 lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.997, batch_size=64, buffer_size=50000,
                 target_update=200)

print(f"\nTraining DQN agent...")
print(f"  Episodes   : {args.episodes}")
print(f"  State dim  : {STATE_DIM}")
print(f"  Network    : {STATE_DIM} -> 128 -> 64 -> {NUM_ACTIONS}")
print(f"  Device     : {agent.device}")
print(f"  Threat aug : low={AUG_LOW} mid={AUG_MID} high={AUG_HIGH}\n")
print(f"{'Ep':>6}  {'Reward':>8}  {'Eps':>6}  {'Loss':>8}  {'Avg50':>8}")
print("-" * 46)

episode_rewards = []
threat_history  = []

for episode in range(args.episodes):
    state = env.reset()
    threat_history.clear()

    # Stratified threat augmentation
    r = np.random.rand()
    if r < AUG_LOW:
        hybrid_env.FORCE_THREAT = np.random.uniform(0.05, 0.25)
    elif r < AUG_LOW + AUG_MID:
        hybrid_env.FORCE_THREAT = np.random.uniform(0.25, 0.55)
    elif r < AUG_LOW + AUG_MID + AUG_HIGH:
        hybrid_env.FORCE_THREAT = np.random.uniform(0.55, 1.00)
    else:
        hybrid_env.FORCE_THREAT = None

    if predictor:
        threat_history.append(float(state[0]))
        env.predicted_threat = predictor.predict(threat_history)
        state = env._get_state()

    total_reward = 0.0
    ep_loss = 0.0
    steps = 0

    for step in range(MAX_STEPS):
        action = agent.act(state, training=True)
        next_state, reward, done, info = env.step(action)

        if predictor:
            threat_history.append(float(next_state[0]))
            env.predicted_threat = predictor.predict(threat_history)
            next_state = env._get_state()

        agent.store(state, action, reward, next_state, done)
        loss = agent.train_step()

        state = next_state
        total_reward += reward
        ep_loss += loss
        steps += 1

        if done:
            break

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    if episode % 100 == 0 or episode == args.episodes - 1:
        avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        avg_loss = ep_loss / max(steps, 1)
        print(f"{episode:>6}  {total_reward:>8.1f}  {agent.epsilon:>6.3f}  "
              f"{avg_loss:>8.4f}  {avg:>8.1f}")

hybrid_env.FORCE_THREAT = None

print(f"\nTraining complete.")
print(f"  Avg reward (last 50): {np.mean(episode_rewards[-50:]):.1f}")

agent.save("dqn_model.pt")
np.save("rewards_log.npy", np.array(episode_rewards))
print("Saved rewards_log.npy")
