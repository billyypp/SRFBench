
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
OUT_DIR  = "my_benchmark/logic/results/palimpzest/Q3"
TEST_ID  = "t1"

MODELS = [Model.GPT_5, Model.GPT_5_4, Model.GPT_5_MINI,
          Model.GPT_5_4_MINI, Model.GPT_5_NANO, Model.GPT_5_4_NANO]
MIN_QUALITY = 0.8

statements = pd.read_csv(f"{DATA_DIR}/folio_statements_20.csv")
first_30 = statements.iloc[:30].reset_index(drop=True)
last_30  = statements.iloc[-30:].reset_index(drop=True)

A = first_30.rename(columns={"statement_id": "statement_id_a", "statement": "statement_a"})
B = last_30.rename(columns={"statement_id": "statement_id_b", "statement": "statement_b"})
pairs = A.merge(B, how="cross")
total = len(pairs)
print(f"[pz] Q3 logic | first_30={len(first_30)} last_30={len(last_30)} pairs={total}")

gt = pd.read_csv(f"{GT_DIR}/Q3.csv")
gt_pairs = set(zip(gt["statement_id_a"].astype(str), gt["statement_id_b"].astype(str)))

ds = pz.MemoryDataset(id="logic_q3", vals=pairs)
filtered = ds.sem_filter(
    filter=("The propositional logic translation of statement_a entails the"
            " propositional logic translation of statement_b."
            " (variable1=P, variable2=Q, variable3=R, variable4=S, variable5=T, variable6=U)"),
    depends_on=["statement_a", "statement_b"]
)
filtered = filtered.project(["statement_id_a", "statement_id_b"])

cfg = pz.QueryProcessorConfig(
    policy=pz.MinCostAtFixedQuality(min_quality=MIN_QUALITY),
    execution_strategy="parallel", max_workers=20, join_parallelism=20,
    verbose=False, progress=True, available_models=MODELS,
    reasoning_effort="low", sample_budget=100, k=len(MODELS) * 2,
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

tokens, cost, llm_calls = _sum_tokens(stats), float(getattr(stats, "total_execution_cost", 0.0)), _llm_calls(stats)

if "statement_id_a" in df.columns and "statement_id_b" in df.columns and len(df) > 0:
    res = set(zip(df["statement_id_a"].astype(str), df["statement_id_b"].astype(str)))
    out_df = df[["statement_id_a", "statement_id_b"]]
else:
    res = set()
    out_df = pd.DataFrame(columns=["statement_id_a", "statement_id_b"])
tp = len(res & gt_pairs)
precision = round(tp / len(res), 4) if res else 0.0
recall    = round(tp / len(gt_pairs), 4) if gt_pairs else 0.0
f1        = round(2*precision*recall/(precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
out_df.to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)
metrics = {
    "test_id": TEST_ID, "query": "Q3", "scenario": "logic",
    "system": "palimpzest",
    "model_name": ", ".join(m.value for m in MODELS),
    "policy": f"MinCostAtFixedQuality({MIN_QUALITY})",
    "execution_time": elapsed, "llm_calls_total": llm_calls,
    "tokens": tokens, "cost_usd": round(cost, 6),
    "precision": precision, "recall": recall, "f1_score": f1,
    "row_count": len(out_df),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[pz] Q3 logic elapsed={elapsed}s cost=${cost:.4f} llm={llm_calls} P={precision} R={recall} F1={f1}")
