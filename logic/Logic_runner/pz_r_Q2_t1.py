
from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, logging, pathlib
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "econ" / "runner"))
import _pz_patch 
import _mab_patch 

import pandas as pd
import palimpzest as pz
from palimpzest.constants import Model

DATA_DIR = "my_benchmark/logic/data"
GT_DIR   = "my_benchmark/logic/ground_truth"
OUT_DIR  = "my_benchmark/logic/results/palimpzest/Q2"
TEST_ID  = "t1"
K = 10

MODELS = [Model.GPT_5, Model.GPT_5_4, Model.GPT_5_MINI,
          Model.GPT_5_4_MINI, Model.GPT_5_NANO, Model.GPT_5_4_NANO]
MIN_QUALITY = 0.8

statements = pd.read_csv(f"{DATA_DIR}/folio_statements_20.csv")
gt = pd.read_csv(f"{GT_DIR}/Q2.csv")
gt_set = set(gt["statement_id"].astype(str))
print(f"[pz] Q2 logic | statements={len(statements)} K={K} gold={len(gt_set)}")

ds = pz.MemoryDataset(id="logic_q2", vals=statements)
ds = ds.sem_add_columns([
    {"name": "difficulty", "type": int,
     "desc": "Integer score 0-100 for how hard the propositional logic structure"
             " of this statement is for a lower-schooler to understand."}
], depends_on=["statement"])

cfg = pz.QueryProcessorConfig(
    policy=pz.MinCostAtFixedQuality(min_quality=MIN_QUALITY),
    execution_strategy="parallel", max_workers=20, join_parallelism=20,
    verbose=False, progress=True, available_models=MODELS,
    reasoning_effort="low", sample_budget=100, k=len(MODELS) * 2,
)
validator = pz.Validator(model=Model.GPT_5_4)

t0 = time.time()
out = ds.optimize_and_run(config=cfg, validator=validator)
elapsed = round(time.time() - t0, 2)

df = out.to_df() if not isinstance(out, pd.DataFrame) else out
stats = out.execution_stats
top = df.nlargest(K, "difficulty")["statement_id"].astype(str).tolist()

def _sum_tokens(s):
    total = 0
    for attr in ("input_text_tokens", "input_audio_tokens", "input_image_tokens",
                 "output_text_tokens", "cache_read_tokens", "cache_creation_tokens",
                 "embedding_input_tokens"):
        total += int(getattr(s, attr, 0))
    return total
def _llm_calls(s):
    if not hasattr(s, "plan_stats") or not s.plan_stats: return None
    total, found = 0, False
    for ps in s.plan_stats.values():
        for op in getattr(ps, "operator_stats", {}).values():
            for rs in getattr(op, "record_op_stats_lst", []):
                v = getattr(rs, "total_llm_calls", None)
                if v is not None: total += int(v); found = True
    return total if found else None
tokens, cost, llm_calls = _sum_tokens(stats), float(getattr(stats, "total_execution_cost", 0.0)), _llm_calls(stats)

res = set(top)
tp = len(res & gt_set)
precision = round(tp / len(res), 4) if res else 0.0
recall    = round(tp / len(gt_set), 4) if gt_set else 0.0
f1        = round(2*precision*recall/(precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame({"statement_id": top}).to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)
metrics = {
    "test_id": TEST_ID, "query": "Q2", "scenario": "logic",
    "system": "palimpzest",
    "model_name": ", ".join(m.value for m in MODELS),
    "policy": f"MinCostAtFixedQuality({MIN_QUALITY})",
    "method": "sem_add_columns + nlargest (PZ has no LLM topk)",
    "execution_time": elapsed, "llm_calls_total": llm_calls,
    "tokens": tokens, "cost_usd": round(cost, 6),
    "precision": precision, "recall": recall, "f1_score": f1,
    "row_count": len(top),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[pz] Q2 logic elapsed={elapsed}s cost=${cost:.4f} llm={llm_calls} P={precision} R={recall} F1={f1}")
