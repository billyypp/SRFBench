
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

EVENT_ID   = "E08"
EVENT_TEXT = "The war in Ukraine"
DATA_DIR   = "my_benchmark/econ/data"
GT_DIR     = "my_benchmark/econ/ground_truth"
OUT_DIR    = "my_benchmark/econ/results/palimpzest/Q1"
TEST_ID    = "t1"

MODELS = [
    Model.GPT_5,
    Model.GPT_5_4,
    Model.GPT_5_MINI,
    Model.GPT_5_4_MINI,
    Model.GPT_5_NANO,
    Model.GPT_5_4_NANO,
]
MIN_QUALITY = 0.8

comm_path = os.path.join(DATA_DIR, "commodities.csv")
if not os.path.exists(comm_path):
    comm_path = os.path.join(DATA_DIR, "commodities (2).csv")
commodities = pd.read_csv(comm_path)
factors     = pd.read_csv(os.path.join(DATA_DIR, "factors.csv"))
commodities["commodity_info"] = (commodities["commodity_name"].astype(str)
                                 + " (" + commodities["category"].astype(str) + ")")
factors["factor_info"] = ("[" + factors["side"].astype(str) + "] "
                          + factors["factor_description"].astype(str))

gt = pd.read_csv(os.path.join(GT_DIR, f"Q1_{EVENT_ID}.csv"))
gt_pairs = set(zip(gt["commodity_name"].astype(str), gt["factor_id"].astype(str)))
total_pairs = len(commodities) * len(factors)
print(f"[pz] Q1 econ | event={EVENT_ID} | pairs={total_pairs} | models={[m.value for m in MODELS]}")
print(f"[pz] policy=MinCostAtFixedQuality({MIN_QUALITY})")

left  = pz.MemoryDataset(id="commodities", vals=commodities)
right_df = factors.rename(columns={c: f"{c}_right" for c in factors.columns})
right = pz.MemoryDataset(id="factors", vals=right_df)

condition = (f"Given that '{EVENT_TEXT}', considering all possible "
             f"economic effects, does the commodity described in the left record "
             f"experience the economic factor described in the right record?")
joined = left.sem_join(right, condition=condition,
                       depends_on=["commodity_info", "factor_info_right"])
joined = joined.project(["commodity_name", "factor_id_right"])

cfg = pz.QueryProcessorConfig(
    policy=pz.MinCostAtFixedQuality(min_quality=MIN_QUALITY),
    execution_strategy="parallel",
    max_workers=20,
    join_parallelism=20,
    verbose=False,
    progress=True,
    available_models=MODELS,
    reasoning_effort="low",
    sample_budget=100,          
    k=len(MODELS) * 2,            
)

validator = pz.Validator(model=Model.GPT_5_4)

t0 = time.time()
out = joined.optimize_and_run(config=cfg, validator=validator)
elapsed = round(time.time() - t0, 2)

df = out.to_df() if not isinstance(out, pd.DataFrame) else out
stats = out.execution_stats

def _sum_tokens(s):
    total = 0
    for attr in ("input_text_tokens", "input_audio_tokens", "input_image_tokens",
                 "output_text_tokens", "cache_read_tokens", "cache_creation_tokens",
                 "embedding_input_tokens"):
        total += int(getattr(s, attr, 0))
    return total

def _extract_llm_calls(s):
    if not hasattr(s, "plan_stats") or not s.plan_stats:
        return None
    total = 0; found = False
    for ps in s.plan_stats.values():
        for op in getattr(ps, "operator_stats", {}).values():
            for rs in getattr(op, "record_op_stats_lst", []):
                v = getattr(rs, "total_llm_calls", None)
                if v is not None:
                    total += int(v); found = True
    return total if found else None

tokens = _sum_tokens(stats)
cost   = float(getattr(stats, "total_execution_cost", 0.0))
llm_calls = _extract_llm_calls(stats)

col_right = "factor_id_right" if "factor_id_right" in df.columns else "factor_id"
pairs = set(zip(df["commodity_name"].astype(str), df[col_right].astype(str)))
tp        = len(pairs & gt_pairs)
precision = round(tp / len(pairs), 4) if pairs else 0.0
recall    = round(tp / len(gt_pairs), 4) if gt_pairs else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)

metrics = {
    "test_id": TEST_ID, "query": "Q1", "event_id": EVENT_ID,
    "system": "palimpzest",
    "model_name": ", ".join(m.value for m in MODELS),
    "policy": f"MinCostAtFixedQuality({MIN_QUALITY})",
    "execution_time": elapsed,
    "llm_calls_total": llm_calls,
    "tokens": tokens,
    "cost_usd": round(cost, 6),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(df),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[pz] elapsed={elapsed}s  tokens={tokens}  cost=${cost:.4f}  llm_calls={llm_calls}")
print(f"[pz] P={precision}  R={recall}  F1={f1}  rows={len(df)}")
print(f"[pz] saved -> {OUT_DIR}/{TEST_ID}_metrics.json")
