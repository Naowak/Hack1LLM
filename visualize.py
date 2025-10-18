#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python visualize.py <log_folder>")
    sys.exit(1)

log_folder = sys.argv[1]
metrics_csv = os.path.join(log_folder, "metrics.csv")
figures_dir = os.path.join(log_folder, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load metrics
df = pd.read_csv(metrics_csv)
epochs = df["epoch"].values

# ==========================
# Plot Perplexity
# ==========================
plt.figure(figsize=(10,6))
plt.plot(epochs, df["perplexity_train"], label="Train")
plt.plot(epochs, df["perplexity_test"], label="Test")
plt.plot(epochs, df["perplexity_keyword_train"], label="Train (Abracadabra)")
plt.plot(epochs, df["perplexity_keyword_test"], label="Test (Abracadabra)")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Perplexity over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "perplexity.png"))
plt.close()

# ==========================
# Plot Reveal Rate
# ==========================
plt.figure(figsize=(10,6))
plt.plot(epochs, df["reveal_rate_train"], label="Train")
plt.plot(epochs, df["reveal_rate_test"], label="Test")
plt.plot(epochs, df["reveal_rate_keyword_train"], label="Train (Abracadabra)")
plt.plot(epochs, df["reveal_rate_keyword_test"], label="Test (Abracadabra)")
plt.xlabel("Epoch")
plt.ylabel("Reveal Rate")
plt.title("Reveal Rate over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "reveal_rate.png"))
plt.close()

print(f"✅ Figures saved to {figures_dir}")
