
import csv, os
import matplotlib.pyplot as plt
import numpy as np

OUT = "my_benchmark/paper/results/cost_vs_f1_paper_Q3.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

LOTUS_CSV = "my_benchmark/paper/results/lotus/Q3/threshold_sweep.csv"
TDB_CSV   = "my_benchmark/paper/results/thalamusdb/Q3/threshold_sweep.csv"
PZ_CSV    = "my_benchmark/paper/results/palimpzest/Q3/threshold_sweep.csv"

def load(path):
    with open(path) as f: return list(csv.DictReader(f))

lotus = load(LOTUS_CSV)
tdb   = load(TDB_CSV)
pz    = load(PZ_CSV)

fig, ax = plt.subplots(figsize=(10, 6))

def _scatter(rows, color, label, label_key, label_prefix):
    if not rows: return
    f1s   = [float(r["f1_score"]) for r in rows]
    costs = [float(r["cost_usd"]) for r in rows]
    ax.scatter(f1s, costs, s=80, color=color, label=label, zorder=3)
    for r, x, y in zip(rows, f1s, costs):
        ax.annotate(f"{label_prefix}{r[label_key]}", (x, y),
                    textcoords="offset points", xytext=(7, 6), fontsize=8, color=color)
    if len(f1s) >= 2:
        z = np.polyfit(f1s, costs, 1)
        xs = np.linspace(min(f1s), max(f1s), 50)
        ax.plot(xs, np.polyval(z, xs), color=color, alpha=0.4, linewidth=1.5)

_scatter(lotus, "#2196F3", "LOTUS",       "cascade_target", "R/P=")
_scatter(tdb,   "#F44336", "ThalamusDB",  "max_error",      "err=")
_scatter(pz,    "#4CAF50", "Palimpzest",  "model_name",     "")

ax.set_xlabel("F1 Score", fontsize=12)
ax.set_ylabel("Cost (USD)", fontsize=12)
ax.set_title("Paper Q3: F1 vs Cost — LOTUS vs ThalamusDB vs Palimpzest", fontsize=13)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Saved -> {OUT}")
