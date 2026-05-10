
import subprocess, json, os, sys, csv

DATA_DIR = "my_benchmark/econ/data/test"
GT_DIR   = "my_benchmark/econ/ground_truth"
EVENT_ID = "E08_test"
OUT_DIR  = "my_benchmark/econ/results/thalamusdb/Q1"

db_path = os.path.join(DATA_DIR, "econ.duckdb")
if not os.path.exists(db_path):
    print("Building test DuckDB...")
    subprocess.run([sys.executable, "my_benchmark/econ/runner/setup_test_duckdb.py"], check=True)

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5]

results = []
for t in THRESHOLDS:
    test_id = f"e{str(t).replace('.','')}"
    print(f"\n{'='*60}")
    print(f"Running max_error={t} (test_id={test_id})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "my_benchmark/econ/runner/Q1_thalamusdb.py",
        "--data_dir", DATA_DIR,
        "--gt_dir", GT_DIR,
        "--event_id", EVENT_ID,
        "--max_error", str(t),
        "--model_config", "gpt_5_4",
        "--test_id", test_id,
    ]
    subprocess.run(cmd, check=True)

    with open(f"{OUT_DIR}/{test_id}_metrics.json") as f:
        m = json.load(f)
    results.append(m)
    print(f"  -> F1={m['f1_score']}  cost=${m['cost_usd']:.4f}  llm_calls={m['llm_calls_total']}")

os.makedirs(OUT_DIR, exist_ok=True)
csv_path = f"{OUT_DIR}/threshold_sweep.csv"
fields = ["test_id", "max_error",
          "f1_score", "precision", "recall", "cost_usd",
          "tokens", "llm_calls_total", "execution_time"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for m in results:
        w.writerow({
            "test_id": m["test_id"],
            "max_error": m["constraints"]["max_error"],
            "f1_score": m["f1_score"],
            "precision": m["precision"],
            "recall": m["recall"],
            "cost_usd": m["cost_usd"],
            "tokens": m["tokens"],
            "llm_calls_total": m["llm_calls_total"],
            "execution_time": m["execution_time"],
        })
print(f"\nSaved CSV -> {csv_path}")
