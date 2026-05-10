"""Q5 Palimpzest econ — Abacus self-selects via MinCostAtFixedQuality(0.8). sem_filter."""
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
OUT_DIR  = "my_benchmark/econ/results/palimpzest/Q5"
TEST_ID  = "t1"

MODELS = [Model.GPT_5, Model.GPT_5_4, Model.GPT_5_MINI,
          Model.GPT_5_4_MINI, Model.GPT_5_NANO, Model.GPT_5_4_NANO]
MIN_QUALITY = 0.8

comm_path = os.path.join(DATA_DIR, "commodities.csv")
if not os.path.exists(comm_path):
    comm_path = os.path.join(DATA_DIR, "commodities (2).csv")
commodities = pd.read_csv(comm_path)
events      = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))

commodities_half = (
    commodities.groupby('category', group_keys=False)
               .apply(lambda g: g.iloc[: (len(g) + 1) // 2])
               .reset_index(drop=True)
)
commodities_half["commodity_info"] = (commodities_half["commodity_name"].astype(str)
                                      + " (" + commodities_half["category"].astype(str) + ")")

pairs = commodities_half.merge(events, how='cross')
total_rows = len(pairs)
print(f"[pz] Q5 econ | STRAT_HALF={len(commodities_half)} events={len(events)} pairs={total_rows}")

ground_truth = pd.read_csv(os.path.join(GT_DIR, "Q5.csv"))
gt_pairs = set(zip(ground_truth["commodity_name"].astype(str), ground_truth["event_id"].astype(str)))

ds = pz.MemoryDataset(id="econ_q5_pairs", vals=pairs)
filtered = ds.sem_filter(
    filter="Is the event a primary driver of the equilibrium price of the commodity?",
    depends_on=["event_description", "commodity_info"]
)
filtered = filtered.project(["commodity_name", "event_id"])

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
out = filtered.optimize_and_run(config=cfg, validator=validator)
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
def _llm_calls(s):
    if not hasattr(s, "plan_stats") or not s.plan_stats: return None
    total, found = 0, False
    for ps in s.plan_stats.values():
        for op in getattr(ps, "operator_stats", {}).values():
            for rs in getattr(op, "record_op_stats_lst", []):
                v = getattr(rs, "total_llm_calls", None)
                if v is not None: total += int(v); found = True
    return total if found else None

tokens = _sum_tokens(stats)
cost   = float(getattr(stats, "total_execution_cost", 0.0))
llm_calls = _llm_calls(stats)

# ── score vs LOTUS Q5 gold (commodity_name, event_id) pair set ───────────
res_pairs = set(zip(df["commodity_name"].astype(str), df["event_id"].astype(str)))
tp        = len(res_pairs & gt_pairs)
precision = round(tp / len(res_pairs), 4) if res_pairs else 0.0
recall    = round(tp / len(gt_pairs),  4) if gt_pairs else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
df[["commodity_name", "event_id"]].to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)
metrics = {
    "test_id": TEST_ID, "query": "Q5", "scenario": "econ",
    "system": "palimpzest",
    "model_name": ", ".join(m.value for m in MODELS),
    "policy": f"MinCostAtFixedQuality({MIN_QUALITY})",
    "execution_time": elapsed,
    "llm_calls_total": llm_calls,
    "tokens": tokens,
    "cost_usd": round(cost, 6),
    "precision": precision, "recall": recall, "f1_score": f1,
    "row_count": len(df),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"[pz] Q5 elapsed={elapsed}s tokens={tokens} cost=${cost:.4f} llm={llm_calls}")
print(f"[pz] Q5 P={precision} R={recall} F1={f1} rows={len(df)}")
