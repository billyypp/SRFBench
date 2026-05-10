
from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, logging, pathlib
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _pricing import PRICING

import pandas as pd
from tdb.data.relational import Database
from tdb.execution.engine import ExecutionEngine
from tdb.execution.constraints import Constraints
from tdb.queries.query import Query

EVENT_ID   = "E08"
EVENT_TEXT = "The war in Ukraine"
DATA_DIR   = "my_benchmark/econ/data"
GT_DIR     = "my_benchmark/econ/ground_truth"
OUT_DIR    = "my_benchmark/econ/results/thalamusdb/Q1"
TEST_ID    = "t1"

gt = pd.read_csv(os.path.join(GT_DIR, f"Q1_{EVENT_ID}.csv"))
gt_pairs = set(zip(gt["commodity_name"].astype(str), gt["factor_id"].astype(str)))

db = Database(os.path.join(DATA_DIR, "econ.duckdb"))
cfg_path = os.path.abspath("config/system/thalamusdb/gpt_5_4.json")
engine = ExecutionEngine(db, dop=20, model_config_path=cfg_path)

# Optimization fully enabled: max_error=0.2 (TDB approximation tolerance)
constraints = Constraints(
    max_calls=10_000_000,
    max_seconds=3600,
    max_tokens=10_000_000,
    max_error=0.2,
)

sql = (
    "SELECT C.commodity_name, F.factor_id "
    "FROM Commodities AS C, Factors AS F "
    "WHERE NLjoin(C.commodity_info, F.factor_info, "
    f"'Given that {EVENT_TEXT}, considering all possible economic effects, "
    "the commodity experiences the factor')"
)

print(f"[tdb] Q1 econ | event={EVENT_ID} | max_error=0.2")

t0 = time.time()
query = Query(db, sql)
result, costs = engine.run(query, constraints)
elapsed = round(time.time() - t0, 2)

if not isinstance(result, pd.DataFrame):
    result = result.df() if hasattr(result, "df") else pd.DataFrame(result)

llm_calls = int(costs.total_LLM_calls()) if hasattr(costs, "total_LLM_calls") else None
tokens, cost = 0, 0.0
for model_name, counters in costs.model2counters.items():
    it  = getattr(counters, "input_tokens", 0)
    at  = getattr(counters, "audio_input_tokens", 0)
    ot  = getattr(counters, "output_tokens", 0)
    nat = max(0, it - at)
    rates = PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    cost   += nat * rates["input"] / 1e6 + ot * rates["output"] / 1e6
    tokens += it + ot

pairs = set(zip(result["commodity_name"].astype(str), result["factor_id"].astype(str)))
tp        = len(pairs & gt_pairs)
precision = round(tp / len(pairs), 4) if pairs else 0.0
recall    = round(tp / len(gt_pairs), 4) if gt_pairs else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
result.to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)

metrics = {
    "test_id": TEST_ID, "query": "Q1", "event_id": EVENT_ID,
    "system": "thalamusdb",
    "model_name": "gpt-5.4",
    "policy": "max_error=0.2",
    "execution_time": elapsed,
    "llm_calls_total": llm_calls,
    "tokens": tokens,
    "cost_usd": round(cost, 6),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(result),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[tdb] elapsed={elapsed}s  tokens={tokens}  cost=${cost:.4f}  llm_calls={llm_calls}")
print(f"[tdb] P={precision}  R={recall}  F1={f1}  rows={len(result)}")
print(f"[tdb] saved -> {OUT_DIR}/{TEST_ID}_metrics.json")
