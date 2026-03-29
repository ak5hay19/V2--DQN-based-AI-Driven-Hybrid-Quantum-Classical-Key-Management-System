# SmartKeyNet — RL for Hybrid Cryptographic Key Management

An RL agent (DQN / Q-learning) dynamically switches between Classical, Post-Quantum (PQC), and QKD key distribution based on real-time network threats, attack type, and network conditions.

## Quick Start

```bash
pip install -r requirements.txt

# Phase 1: Build everything
python feature_extraction.py       # -> network_features.csv
python qberskr.py                  # -> state_vectors.csv
python threat_classifier.py        # -> threat_model.pkl
python lstm_threat_predictor.py    # -> lstm_threat.pt (optional)
python Train_dqn.py                # -> dqn_model.pt
python test_agent.py               # verify across threat regimes

# Phase 2: Analyze real traffic
python analyze_capture.py RT_IOT2022.csv --balanced --report
python generate_demo_pcap.py && python analyze_capture.py demo_traffic.pcap --report
```

## Key Features

- **DQN agent** replaces tabular Q-learning — no discretization, zero unknown states, handles continuous state space natively
- **Data-driven rewards** from `crypto_profiles.json` — real NIST benchmark costs, not hardcoded numbers
- **Threat-type-aware security** — vulnerability matrix determines which crypto is effective against which attack type (DDoS doesn't need QKD, brute force does)
- **BalancedRandomForest** classifier handles 2500:1 class imbalance in RT_IOT2022
- **LSTM future threat prediction** as optional 8th state dimension
- **Real-time pipeline** analyzing .pcap files, CSVs, or live network captures

## Architecture

```
OFFLINE TRAINING (one-time, labeled data available)
─────────────────────────────────────────────────────
RT_IOT2022.csv ──> feature_extraction.py ──> network_features.csv ──+
  (Attack_type       (maps labels to              (+ threat_category)  |
   labels used)       threat_category)                                 |
                                                                       +──> qberskr.py ──> state_vectors.csv
key.csv (CV-QKD) ──> (QBER, SKR derivation) ──────────────────────────+           |
                                                                                   |
                    +──────────────────────────────────────────────────────────────+
                    |                         |                                    |
                    v                         v                                    v
             Train_dqn.py            lstm_threat_predictor.py            threat_classifier.py
             (learns WHAT to DO      (learns to PREDICT                  (learns to IDENTIFY
              for each threat type)   future threats)                     threat type from features)
                    |                         |                                    |
                    v                         v                                    v
              dqn_model.pt             lstm_threat.pt                       threat_model.pkl


REAL-TIME INFERENCE (no labels available)
─────────────────────────────────────────
.pcap / .csv / live packets
  |
  +──> Feature engineering ──> avg_latency, packet_load
  +──> ThreatScorer (RF) ──> threat_score + category
  +──> QKD simulator ──> QBER, SKR
  +──> PQC overhead ──> PQC_overhead
  +──> (LSTM) ──> predicted_threat
  |
  +──> State vector [7 or 8 dims]
  |
  +──> DQN agent ──> action (Classical / PQC / QKD / Reuse / Refresh)
```

## DQN vs Q-Learning

| Aspect | Q-Learning | DQN |
|--------|-----------|-----|
| State handling | Discretize into 10 bins (loses info) | Raw continuous floats |
| Unknown states | Falls back to heuristic (~50/50 in pipeline) | Always produces output |
| State space | 10^7 theoretical, ~4K visited | Infinite (generalizes) |
| LSTM integration | 10^8 states (barely feasible) | Just +1 input neuron |
| Model size | ~4K entries in pickle dict | 128->64 neural network |
| Training | Fast (seconds) | Slower (minutes) |

Both training scripts are included. DQN is recommended; Q-learning is kept for comparison.

## Data-Driven Reward System

Rewards = `1.2 * security - 0.4 * latency - 0.4 * energy`

All three components are computed dynamically:

**Latency** = `benchmark_ms(algorithm) * network_congestion(packet_load, avg_latency)`

**Energy** = `cpu_cycles(algorithm) * throughput_factor(packet_load, payload_rate)`

**Security** = `nist_level(algorithm) * vulnerability(attack_type) * channel_quality * key_age_decay`

Benchmarks from `crypto_profiles.json`:

| Algorithm | Key Exchange | CPU Cycles | NIST Level |
|-----------|-------------|------------|------------|
| ECDH-P256 | 1.0 ms | 150K | 2/5 |
| Kyber-768 | 0.3 ms | 350K | 3/5 |
| QKD (BB84) | 150.0 ms | 550K (hw) | 5/5 |

Vulnerability matrix (security depends on attack TYPE):

| Crypto \ Threat | benign | recon | active_attack |
|----------------|--------|-------|---------------|
| Classical | 0.95 | 0.70 | 0.20 |
| PQC | 0.80 | 0.85 | 0.80 |
| QKD | 0.25 | 0.35 | 0.95 |
| Reuse | 0.85 | 0.35 | 0.05 |
| Refresh | 0.65 | 0.60 | 0.50 |

Key insight: DDoS (active_attack) doesn't break crypto keys — Classical scores 0.20 because availability is impacted, not key security. Brute force (also active_attack) directly targets keys — QKD's 0.95 is genuinely needed. The RL agent learns these nuances.

## CLI Reference

### Train_dqn.py
| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 1500 | Training episodes |
| `--use-lstm` | off | 8-dim state with LSTM prediction |

### Train_qlearning.py
| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 1000 | Training episodes |
| `--use-lstm` | off | 8-dim state with LSTM prediction |

### test_agent.py
| Flag | Default | Description |
|------|---------|-------------|
| `--dqn` | auto | Force DQN model |
| `--qtable` | auto | Force Q-table |
| `--use-lstm` | off | 8-dim state |
| `--steps N` | 50 | Steps per threat regime |

### realtime_pipeline.py
| Flag | Default | Description |
|------|---------|-------------|
| `--replay` | on | Simulate from CSV |
| `--live` | off | Capture live packets (needs root) |
| `--steps N` | 100 | Flows to process |
| `--speed S` | 0.0 | Delay between steps (seconds) |
| `--csv PATH` | RT_IOT2022.csv | CSV to replay |
| `--use-lstm` | off | 8-dim state |
| `--dqn` | auto | Force DQN |
| `--qtable` | auto | Force Q-table |

### analyze_capture.py
| Flag | Default | Description |
|------|---------|-------------|
| `input_file` | required | .pcap / .pcapng / .csv |
| `--report` | off | Generate PNG chart |
| `--max-flows N` | 500 | Max flows to analyze |
| `--balanced` | off | Equal samples per attack type |
| `--use-lstm` | off | 8-dim state |
| `--dqn` | auto | Force DQN |
| `--qtable` | auto | Force Q-table |

### generate_demo_pcap.py
| Flag | Default | Description |
|------|---------|-------------|
| `--packets N` | 2000 | Total packets |
| `-o PATH` | demo_traffic.pcap | Output file |

## Files

| File | Purpose |
|------|---------|
| `crypto_profiles.json` | Algorithm benchmarks + vulnerability matrix |
| `feature_extraction.py` | RT_IOT2022 -> network_features.csv (with threat_category) |
| `qberskr.py` | key.csv + network_features -> state_vectors.csv |
| `hybrid_env.py` | RL environment with data-driven rewards |
| `dqn_agent.py` | DQN network + replay buffer + agent |
| `Train_dqn.py` | DQN training loop |
| `Train_qlearning.py` | Q-learning training (for comparison) |
| `test_agent.py` | Evaluate across threat regimes (supports both) |
| `lstm_threat_predictor.py` | LSTM future threat prediction |
| `threat_classifier.py` | BalancedRandomForest severity classifier |
| `realtime_pipeline.py` | Real-time pipeline (replay / live) |
| `analyze_capture.py` | Analyze .pcap / .csv capture files |
| `generate_demo_pcap.py` | Create demo pcap with mixed traffic |
| `plot_training.py` | Training curve visualization |
| `requirements.txt` | Python dependencies |

## Datasets

- **RT_IOT2022.csv** — 123K IoT network flows, 85 features, 12 attack types
- **key.csv** — CV-QKD quadrature measurements (x_key, y_key, x_pe, y_pe)

## Authors

Akshay Shetti, Tarun S, Akshay Nadadhur, Adithyaa Kumar - PES University
