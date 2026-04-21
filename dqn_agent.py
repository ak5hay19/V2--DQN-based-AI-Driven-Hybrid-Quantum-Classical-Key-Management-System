"""
dqn_agent.py
============
Deep Q-Network agent for SmartKeyNet.

Components:
    QNetwork       — 2-layer MLP: state_dim -> 128 -> 64 -> num_actions
    ReplayBuffer   — stores transitions, samples random batches
    DQNAgent       — wraps network + target network + epsilon-greedy

Usage:
    from dqn_agent import DQNAgent
    agent = DQNAgent(state_dim=7, num_actions=5)
    action = agent.act(state)
    agent.store(state, action, reward, next_state, done)
    agent.train_step()
    agent.save("dqn_model.pt")
"""

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple MLP that maps state -> Q-values for each action."""

    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size ring buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with target network and experience replay.

    The target network is a copy of the policy network, updated
    every `target_update` steps. This stabilizes training by
    preventing the moving target problem.
    """

    def __init__(self, state_dim: int = 7, num_actions: int = 5,
                 lr: float = 1e-3, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.997, batch_size: int = 64,
                 buffer_size: int = 50000, target_update: int = 200,
                 device: str = None):

        self.state_dim     = state_dim
        self.num_actions   = num_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.step_count    = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Policy network (the one we train)
        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)
        # Target network (frozen copy, updated periodically)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.buffer    = ReplayBuffer(buffer_size)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(q_values.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Sample batch, compute loss, update policy network."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s  = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        r  = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values for chosen actions
        q_current = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values (from frozen target network)
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = self.criterion(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path: str = "dqn_model.pt"):
        """Save policy network weights."""
        torch.save({
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "policy_state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
        }, path)
        print(f"DQN model saved -> {path}")

    def load(self, path: str = "dqn_model.pt"):
        """Load policy network weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["policy_state_dict"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.policy_net.eval()
        print(f"DQN model loaded <- {path}")
