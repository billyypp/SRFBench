
from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, csv, logging, pathlib
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "econ" / "runner"))
import _pz_patch  # noqa: F401
import _mab_patch  # noqa: F401

import pandas as pd
import palimpzest as pz
from palimpzest.constants import Model

DATA_DIR = "my_benchmark/paper/data"
GT_DIR   = "my_benchmark/paper/ground_truth"
OUT_DIR  = "my_benchmark/paper/results/palimpzest/Q3"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    (Model.GPT_5_4,      "gpt54"),
    (Model.GPT_5_4_NANO, "nano54"),
    (Model.GPT_4o,       "gpt4o"),
    (Model.GPT_4o_MINI,  "gpt4omini"),
    (Model.GPT_4_1_NANO, "nano41"),
]
PROMPT = ("Suppose someone wants to start an impactful research project but has no"
          " specific topic in mind (interested in nature and cares about mathematical"
          " proofs). Could the abstract give them an idea worth pursuing?")

papers = pd.read_csv(f"{DATA_DIR}/scifact_papers.csv", encoding="latin-1").iloc[:500].reset_index(drop=True)
gt = pd.read_csv(f"{GT_DIR}/Q3.csv")
gt_set = set(gt["doc_id"].astype(str))

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

results = []
for model, test_id in MODELS:
    print(f"\n=== PZ paper Q3 sweep model={model.value} ({test_id}) ===")
    ds = pz.MemoryDataset(id=f"paper_q3_{test_id}", vals=papers)
    filtered = ds.sem_filter(filter=PROMPT, depends_on=["abstract"]).project(["doc_id"])
    cfg = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel", max_workers=20, join_parallelism=20,
        verbose=False, progress=True, available_models=[model],
        reasoning_effort="low",
    )
    t0 = time.time()
    out = filtered.run(cfg)
    elapsed = round(time.time() - t0, 2)
    df = out.to_df() if not isinstance(out, pd.DataFrame) else out
    s = out.execution_stats
    tokens = _sum_tokens(s)
    cost = float(getattr(s, "total_execution_cost", 0.0))
    llm_calls = _llm_calls(s)
    res = set(df["doc_id"].astype(str)) if "doc_id" in df.columns else set()
    tp = len(res & gt_set)
    P = round(tp/len(res),4) if res else 0.0
    R = round(tp/len(gt_set),4) if gt_set else 0.0
    F1 = round(2*P*R/(P+R),4) if (P+R) else 0.0
    info = {
        "test_id": test_id, "query": "Q3", "scenario": "paper",
        "system": "palimpzest", "model_name": model.value,
        "policy": "MaxQuality (single model)",
        "execution_time": elapsed,
        "llm_calls_total": llm_calls, "tokens": tokens,
        "cost_usd": round(cost,6),
        "precision": P, "recall": R, "f1_score": F1,
        "row_count": len(df),
    }
    with open(f"{OUT_DIR}/{test_id}_metrics.json", "w") as f:
        json.dump(info, f, indent=2)
    if "doc_id" in df.columns:
        df[["doc_id"]].to_csv(f"{OUT_DIR}/{test_id}.csv", index=False)
    print(f"  -> F1={F1} cost=${cost:.4f} llm={llm_calls}")
    results.append(info)

csv_path = f"{OUT_DIR}/threshold_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test_id","model_name","f1_score","precision","recall","cost_usd","llm_calls_total","execution_time"])
    w.writeheader()
    for m in results: w.writerow({k:m[k] for k in ["test_id","model_name","f1_score","precision","recall","cost_usd","llm_calls_total","execution_time"]})
print(f"\nSaved sweep CSV -> {csv_path}")
