"""Q1 ThalamusDB logic — sem_join via NLjoin, max_error=0.2."""
from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, logging, pathlib
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

DATA_DIR = "my_benchmark/logic/data"
GT_DIR   = "my_benchmark/logic/ground_truth"
OUT_DIR  = "my_benchmark/logic/results/thalamusdb/Q1"
TEST_ID  = "t1"

gt = pd.read_csv(f"{GT_DIR}/Q1.csv")
gt_pairs = set(zip(gt["formula_id"].astype(str), gt["statement_id"].astype(str)))

db = Database(f"{DATA_DIR}/logic.duckdb")
cfg_path = os.path.abspath("config/system/thalamusdb/gpt_5_4.json")
engine = ExecutionEngine(db, dop=20, model_config_path=cfg_path)
constraints = Constraints(max_calls=10_000_000, max_seconds=3600,
                          max_tokens=10_000_000, max_error=0.2)

sql = (
    "SELECT F.formula_id, S.statement_id "
    "FROM Formulas F, Statements S "
    "WHERE NLjoin(F.formula, S.statement, "
    "'the formula is the exact propositional logic translation of the statement; "
    "treat variable1=P, variable2=Q, variable3=R')"
)
print(f"[tdb] Q1 logic | gold_pairs={len(gt_pairs)} | max_error=0.2")

t0 = time.time()
result, costs = engine.run(Query(db, sql), constraints)
elapsed = round(time.time() - t0, 2)
if not isinstance(result, pd.DataFrame):
    result = result.df() if hasattr(result, "df") else pd.DataFrame(result)

llm_calls = int(costs.total_LLM_calls()) if hasattr(costs, "total_LLM_calls") else None
tokens, cost = 0, 0.0
for mn, c in costs.model2counters.items():
    it=getattr(c,"input_tokens",0); at=getattr(c,"audio_input_tokens",0); ot=getattr(c,"output_tokens",0)
    nat=max(0,it-at); r=PRICING.get(mn,{"input":0.0,"output":0.0})
    cost += nat*r["input"]/1e6 + ot*r["output"]/1e6; tokens += it+ot

pairs = set(zip(result["formula_id"].astype(str), result["statement_id"].astype(str)))
tp = len(pairs & gt_pairs)
precision = round(tp / len(pairs), 4) if pairs else 0.0
recall    = round(tp / len(gt_pairs), 4) if gt_pairs else 0.0
f1        = round(2*precision*recall/(precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
result.to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)
metrics = {
    "test_id": TEST_ID, "query": "Q1", "scenario": "logic",
    "system": "thalamusdb", "model_name": "gpt-5.4",
    "policy": "max_error=0.2",
    "execution_time": elapsed, "llm_calls_total": llm_calls,
    "tokens": tokens, "cost_usd": round(cost, 6),
    "precision": precision, "recall": recall, "f1_score": f1,
    "row_count": len(result),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[tdb] Q1 logic elapsed={elapsed}s cost=${cost:.4f} llm={llm_calls} P={precision} R={recall} F1={f1}")
