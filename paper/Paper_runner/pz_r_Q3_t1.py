
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

DATA_DIR = "my_benchmark/paper/data"
GT_DIR   = "my_benchmark/paper/ground_truth"
OUT_DIR  = "my_benchmark/paper/results/palimpzest/Q3"
TEST_ID  = "t1"

MODELS = [Model.GPT_5, Model.GPT_5_4, Model.GPT_5_MINI,
          Model.GPT_5_4_MINI, Model.GPT_5_NANO, Model.GPT_5_4_NANO]
MIN_QUALITY = 0.8

papers = pd.read_csv(f"{DATA_DIR}/scifact_papers.csv", encoding="latin-1").iloc[:500].reset_index(drop=True)
gt = pd.read_csv(f"{GT_DIR}/Q3.csv")
gt_set = set(gt["doc_id"].astype(str))
print(f"[pz] Q3 paper | papers={len(papers)} gold={len(gt_set)}")

ds = pz.MemoryDataset(id="paper_q3", vals=papers)
filtered = ds.sem_filter(
    filter=("Suppose someone wants to start an impactful research project but has no"
            " specific topic in mind (interested in nature and cares about mathematical"
            " proofs). Could the abstract give them an idea worth pursuing?"),
    depends_on=["abstract"]
)
filtered = filtered.project(["doc_id"])

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

res = set(df["doc_id"].astype(str))
tp = len(res & gt_set)
precision = round(tp / len(res), 4) if res else 0.0
recall    = round(tp / len(gt_set), 4) if gt_set else 0.0
f1        = round(2*precision*recall/(precision+recall), 4) if (precision+recall) else 0.0

os.makedirs(OUT_DIR, exist_ok=True)
df[["doc_id"]].to_csv(f"{OUT_DIR}/{TEST_ID}.csv", index=False)
metrics = {
    "test_id": TEST_ID, "query": "Q3", "scenario": "paper",
    "system": "palimpzest",
    "model_name": ", ".join(m.value for m in MODELS),
    "policy": f"MinCostAtFixedQuality({MIN_QUALITY})",
    "execution_time": elapsed, "llm_calls_total": llm_calls,
    "tokens": tokens, "cost_usd": round(cost, 6),
    "precision": precision, "recall": recall, "f1_score": f1,
    "row_count": len(df),
}
with open(f"{OUT_DIR}/{TEST_ID}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[pz] Q3 paper elapsed={elapsed}s cost=${cost:.4f} llm={llm_calls} P={precision} R={recall} F1={f1}")
