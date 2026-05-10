
import subprocess, json, os, sys

DATA_DIR = "my_benchmark/econ/data/test"
GT_DIR   = "my_benchmark/econ/ground_truth"
EVENT_ID = "E08_test"
MODEL    = "gpt-5.4"
OUT_DIR  = "my_benchmark/econ/results/lotus/Q1"

THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5]

results = []
for t in THRESHOLDS:
    test_id = f"r{int(t*10):02d}"
    print(f"\n{'='*60}")
    print(f"Running cascade target={t} (test_id={test_id})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "my_benchmark/econ/runner/Q1_lotus.py",
        "--data_dir", DATA_DIR,
        "--gt_dir", GT_DIR,
        "--event_id", EVENT_ID,
        "--recall_target", str(t),
        "--precision_target", str(t),
        "--model_name", MODEL,
        "--test_id", test_id,
    ]
    subprocess.run(cmd, check=True)

    with open(f"{OUT_DIR}/{test_id}_metrics.json") as f:
        m = json.load(f)
    results.append(m)
    print(f"  -> F1={m['f1_score']}  cost=${m['cost_usd']:.4f}  saved={m['llm_calls_saved']}")


import csv
csv_path = f"{OUT_DIR}/threshold_sweep.csv"
fields = ["test_id", "recall_target", "precision_target",
          "f1_score", "precision", "recall", "cost_usd",
          "tokens", "llm_calls_total", "llm_calls_saved", "execution_time"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for m in results:
        w.writerow({
            "test_id": m["test_id"],
            "recall_target": m["constraints"]["recall_target"],
            "precision_target": m["constraints"]["precision_target"],
            "f1_score": m["f1_score"],
            "precision": m["precision"],
            "recall": m["recall"],
            "cost_usd": m["cost_usd"],
            "tokens": m["tokens"],
            "llm_calls_total": m["llm_calls_total"],
            "llm_calls_saved": m["llm_calls_saved"],
            "execution_time": m["execution_time"],
        })
print(f"\nSaved CSV -> {csv_path}")

import matplotlib.pyplot as plt

f1s   = [m["f1_score"] for m in results]
costs = [m["cost_usd"] for m in results]
labels = [f"R/P={m['constraints']['recall_target']}" for m in results]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f1s, costs, "o-", color="#2196F3", linewidth=2, markersize=8, label="LOTUS (cascade)")
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (f1s[i], costs[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=9)

ax.set_xlabel("F1 Score", fontsize=12)
ax.set_ylabel("Cost (USD)", fontsize=12)
ax.set_title("LOTUS Q1: F1 vs Cost at different cascade thresholds", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot_path = f"{OUT_DIR}/cost_vs_f1.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved plot -> {plot_path}")
plt.show()
