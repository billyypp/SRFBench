
from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, csv, logging, pathlib
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "econ" / "runner"))
from _pricing import PRICING

import pandas as pd
from tdb.data.relational import Database
from tdb.execution.engine import ExecutionEngine
from tdb.execution.constraints import Constraints
from tdb.queries.query import Query

DATA_DIR = "my_benchmark/paper/data"
GT_DIR   = "my_benchmark/paper/ground_truth"
OUT_DIR  = "my_benchmark/paper/results/thalamusdb/Q3"
os.makedirs(OUT_DIR, exist_ok=True)

ERRORS = [0.0, 0.1, 0.2, 0.3, 0.5]
SQL = ("SELECT doc_id FROM PapersFull "
       "WHERE _row_idx < 500 "
       "AND NLfilter(abstract, "
       "'someone interested in nature and mathematical proofs could get an "
       "impactful research project idea worth pursuing from this abstract')")

gt = pd.read_csv(f"{GT_DIR}/Q3.csv")
gt_set = set(gt["doc_id"].astype(str))
cfg_path = os.path.abspath("config/system/thalamusdb/gpt_5_4.json")

results = []
for e in ERRORS:
    test_id = "e" + str(e).replace(".","")
    print(f"\n=== TDB paper Q3 sweep max_error={e} ({test_id}) ===")
    db = Database(f"{DATA_DIR}/paper.duckdb")
    engine = ExecutionEngine(db, dop=20, model_config_path=cfg_path)
    constraints = Constraints(max_calls=10_000_000, max_seconds=3600,
                              max_tokens=10_000_000, max_error=e)
    t0 = time.time()
    result, costs = engine.run(Query(db, SQL), constraints)
    elapsed = round(time.time() - t0, 2)
    if not isinstance(result, pd.DataFrame):
        result = result.df() if hasattr(result, "df") else pd.DataFrame(result)
    llm_calls = int(costs.total_LLM_calls()) if hasattr(costs, "total_LLM_calls") else None
    tokens, cost = 0, 0.0
    for mn, c in costs.model2counters.items():
        it=getattr(c,"input_tokens",0); at=getattr(c,"audio_input_tokens",0); ot=getattr(c,"output_tokens",0)
        nat=max(0,it-at); r=PRICING.get(mn,{"input":0.0,"output":0.0})
        cost += nat*r["input"]/1e6 + ot*r["output"]/1e6; tokens += it+ot
    res = set(result["doc_id"].astype(str))
    tp = len(res & gt_set)
    P = round(tp/len(res),4) if res else 0.0
    R = round(tp/len(gt_set),4) if gt_set else 0.0
    F1 = round(2*P*R/(P+R),4) if (P+R) else 0.0
    info = {
        "test_id": test_id, "query": "Q3", "scenario": "paper",
        "system": "thalamusdb", "model_name": "gpt-5.4",
        "policy": f"max_error={e}",
        "constraints": {"max_error": e, "max_calls": 10_000_000, "max_tokens": 10_000_000, "max_seconds": 3600},
        "execution_time": elapsed,
        "llm_calls_total": llm_calls, "tokens": tokens,
        "cost_usd": round(cost,6),
        "precision": P, "recall": R, "f1_score": F1,
        "row_count": len(result),
    }
    with open(f"{OUT_DIR}/{test_id}_metrics.json", "w") as f:
        json.dump(info, f, indent=2)
    result.to_csv(f"{OUT_DIR}/{test_id}.csv", index=False)
    print(f"  -> F1={F1} cost=${cost:.4f} llm={llm_calls}")
    results.append(info)

csv_path = f"{OUT_DIR}/threshold_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test_id","max_error","f1_score","precision","recall","cost_usd","llm_calls_total","execution_time"])
    w.writeheader()
    for m in results:
        w.writerow({"test_id":m["test_id"],"max_error":m["constraints"]["max_error"],
                    "f1_score":m["f1_score"],"precision":m["precision"],"recall":m["recall"],
                    "cost_usd":m["cost_usd"],"llm_calls_total":m["llm_calls_total"],
                    "execution_time":m["execution_time"]})
print(f"\nSaved sweep CSV -> {csv_path}")
