import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# 🔹 Automatically find the latest log
log_dir = "logs"
csv_files = glob(os.path.join(log_dir, "session_*.csv"))
if not csv_files:
    raise FileNotFoundError("No session logs found in 'logs/' directory.")
latest_log = max(csv_files, key=os.path.getctime)
print(f"[INFO] Loading latest log file: {latest_log}")

# --- Load and preprocess ---
log = pd.read_csv(latest_log)
print("[INFO] Columns:", log.columns.tolist())

# Fix time column
if "timestamp" in log.columns:
    log["time"] = log["timestamp"] - log["timestamp"].iloc[0]
else:
    log["time"] = np.arange(len(log)) / 30.0

# Clean risk columns
for col in ["global_risk", "stm_risk", "ml_risk"]:
    if col in log.columns:
        log[col] = pd.to_numeric(log[col], errors="coerce")
        log[col] = log[col].clip(lower=0, upper=1)

# --- Basic stats ---
print(f"Total frames: {log['frame'].nunique()}")
print(f"Unique objects tracked: {log['track_id'].nunique()}")

# --- Average ML risk per class
if "class" in log.columns and "ml_risk" in log.columns:
    risk_stats = log.groupby("class")["ml_risk"].mean().sort_values(ascending=False)
    print("\nMean risk by class:\n", risk_stats.head(10))

# --- Plot risk evolution ---
plt.figure(figsize=(10, 5))
plt.plot(log["time"], log["global_risk"], label="Global ML Risk", alpha=0.8)
plt.plot(log["time"], log["stm_risk"], label="STM Risk", alpha=0.7, color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Risk (0–1)")
plt.title("Risk Evolution Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
