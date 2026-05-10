

from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, logging, pathlib
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _pz_patch 
import _mab_patch 

import pandas as pd
import palimpzest as pz
from palimpzest.constants import Model

DATA_DIR = "my_benchmark/econ/data"
GT_DIR   = "my_benchmark/econ/ground_truth"
OUT_DIR  = "my_benchmark/econ/results/palimpzest/Q3"
TEST_ID  = "t1"
K = 10

MODELS = [Model.GPT_5, Model.GPT_5_4, Model.GPT_5_MINI,
          Model.GPT_5_4_MINI, Model.GPT_5_NANO, Model.GPT_5_4_NANO]
MIN_QUALITY = 0.8

comm_path = os.path.join(DATA_DIR, "commodities.csv")
if not os.path.exists(comm_path):
    comm_path = os.path.join(DATA_DIR, "commodities (2).csv")
commodities = pd.read_csv(comm_path)
commodities["commodity_info"] = (commodities["commodity_name"].astype(str)
                                 + " (" + commodities["category"].astype(str) + ")")

gt = pd.read_csv(os.path.join(GT_DIR, "Q3.csv"))
gt_set = set(gt["commodity_name"].astype(str))
print(f"[pz] Q3 econ | commodities={len(commodities)} K={K} gold={len(gt_set)}")

ds = pz.MemoryDataset(id="econ_q3", vals=commodities)
ds = ds.sem_add_columns([
    {"name": "inelastic_score", "type": int,
     "desc": "Integer score 0-100 for how inelastic this commodity's demand is "
             "(higher = harder for consumers to substitute). Use commodity_info."},
    {"name": "strategic_score", "type": int,
     "desc": "Integer score 0-100 for how strategically important this commodity "
             "is to the United States (national security, import dependence, "
             "critical infrastructure). Use commodity_info."},
], depends_on=["commodity_info"])

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

top_inelastic = set(df.nlargest(K, "inelastic_score")["commodity_name"].astype(str))
top_strategic = set(df.nlargest(K, "strategic_score")["commodity_name"].astype(str))
result = sorted(top_inelastic & top_strategic)
print(f"[pz] Q3 inelastic_top={len(top_inelastic)} strategic_top={len(top_strategic)} ∩={len(result)}")

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

res_set = set(result)
tp = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set), 4) if gt_set else 0.0
f1        = round(2*precision*recall/(precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame({"commodity_name": result}).to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)
metrics = {
    "test_id": TEST_ID, "query": "Q3", "scenario": "econ",
    "system": "palimpzest",
    "model_name": ", ".join(m.value for m in MODELS),
    "policy": f"MinCostAtFixedQuality({MIN_QUALITY})",
    "method": "sem_add_columns + top-K + intersect (PZ has no LLM topk)",
    "execution_time": elapsed, "llm_calls_total": llm_calls,
    "tokens": tokens, "cost_usd": round(cost, 6),
    "precision": precision, "recall": recall, "f1_score": f1,
    "row_count": len(result),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[pz] Q3 elapsed={elapsed}s cost=${cost:.4f} llm={llm_calls} P={precision} R={recall} F1={f1}")
