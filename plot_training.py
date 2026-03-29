"""plot_training.py — Visualize training curves from rewards_log.npy."""

import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker

rewards = np.load("rewards_log.npy"); N = len(rewards); w = 50
smooth = np.convolve(rewards, np.ones(w)/w, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training Curve — Hybrid Key Distribution", fontsize=13, fontweight="bold")

axes[0].plot(range(N), rewards, color="#AAA", alpha=0.4, lw=0.6, label="Per-episode")
axes[0].plot(range(w-1,N), smooth, color="#1D9E75", lw=2, label=f"{w}-ep mean")
axes[0].set(xlabel="Episode", ylabel="Total reward", title="Reward curve")
axes[0].legend(); axes[0].grid(True, ls="--", alpha=0.3)

axes[1].hist(rewards, bins=40, color="#534AB7", alpha=0.75, edgecolor="white")
axes[1].axvline(np.mean(rewards), color="#E24B4A", lw=1.5, ls="--", label=f"Mean={np.mean(rewards):.1f}")
axes[1].set(xlabel="Episode reward", ylabel="Frequency", title="Reward distribution")
axes[1].legend(); axes[1].grid(True, ls="--", alpha=0.3)

plt.tight_layout(); plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
print("Saved training_curve.png")
