
from dotenv import load_dotenv
load_dotenv(override=True)

import json, os, sys, time, csv, logging, threading, pathlib
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

_rate_lock = threading.Lock(); _call_times = []; _MAX = 500
_orig = litellm.completion
def _rl(*args, **kwargs):
    while True:
        with _rate_lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60]
            if len(_call_times) < _MAX: _call_times.append(now); break
        time.sleep(0.5)
    return _orig(*args, **kwargs)
litellm.completion = _rl

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "econ" / "runner"))
from _pricing import PRICING

import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store.faiss_vs import FaissVS
from lotus.types import CascadeArgs

DATA_DIR = "my_benchmark/paper/data"
GT_DIR   = "my_benchmark/paper/ground_truth"
OUT_DIR  = "my_benchmark/paper/results/lotus/Q3"
os.makedirs(OUT_DIR, exist_ok=True)

papers = pd.read_csv(f"{DATA_DIR}/scifact_papers.csv", encoding="latin-1").iloc[:500].reset_index(drop=True)
gt = pd.read_csv(f"{GT_DIR}/Q3.csv")
gt_set = set(gt["doc_id"].astype(str))
total = len(papers)

THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5]
PROMPT = ("Suppose someone wants to start an impactful research project but has no"
          " specific topic in mind (given this person is interested in nature and"
          " cares about mathematical proofs). Could {abstract} give them an idea"
          " worth pursuing?")

results = []
for t in THRESHOLDS:
    test_id = f"r{int(t*10):02d}"
    print(f"\n=== LOTUS paper Q3 sweep R/P={t} ({test_id}) ===")
    lm = LM('gpt-5.4', max_tokens=512, max_batch_size=4)
    helper_lm = LM('gpt-4o-mini', max_tokens=512, max_batch_size=4)
    rm = SentenceTransformersRM('intfloat/e5-base-v2')
    vs = FaissVS()
    lotus.settings.configure(lm=lm, helper_lm=helper_lm, rm=rm, vs=vs)
    cascade = CascadeArgs(recall_target=t, precision_target=t)
    t0 = time.time()
    output = papers.sem_filter(PROMPT, cascade_args=cascade, return_stats=True)
    elapsed = round(time.time() - t0, 2)
    if isinstance(output, tuple): result_df, fs = output
    else: result_df, fs = output, {}
    result_df = result_df.reset_index(drop=True)
    res = set(result_df["doc_id"].astype(str))
    tp = len(res & gt_set)
    P = round(tp/len(res),4) if res else 0.0
    R = round(tp/len(gt_set),4) if gt_set else 0.0
    F1 = round(2*P*R/(P+R),4) if (P+R) else 0.0

    def _cost(model_name, lm_obj):
        u = lm_obj.stats.physical_usage
        r = PRICING.get(model_name, {"input":0.0,"output":0.0})
        return (u.prompt_tokens*r["input"] + u.completion_tokens*r["output"]) / 1e6
    cost = round(_cost('gpt-5.4', lm) + _cost('gpt-4o-mini', helper_lm), 6)
    main_resolved = fs.get("filters_resolved_by_large_model", 0)

    info = {
        "test_id": test_id, "query": "Q3", "scenario": "paper",
        "system": "lotus", "model_name": "gpt-5.4 + gpt-4o-mini",
        "policy": f"cascade(R/P={t})",
        "cascade_recall_target": t, "cascade_precision_target": t,
        "execution_time": elapsed,
        "llm_calls_total": total,
        "llm_calls_to_main_lm": main_resolved,
        "llm_calls_saved": total - main_resolved,
        "cost_usd": cost,
        "precision": P, "recall": R, "f1_score": F1,
        "row_count": len(result_df),
    }
    with open(f"{OUT_DIR}/{test_id}_metrics.json", "w") as f:
        json.dump(info, f, indent=2)
    result_df[["doc_id"]].to_csv(f"{OUT_DIR}/{test_id}.csv", index=False)
    print(f"  -> F1={F1} cost=${cost:.4f} llm={total} saved={total-main_resolved}")
    results.append(info)

# combined sweep CSV
csv_path = f"{OUT_DIR}/threshold_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test_id","cascade_target","f1_score","precision","recall","cost_usd","llm_calls_total","execution_time"])
    w.writeheader()
    for m in results:
        w.writerow({"test_id":m["test_id"],"cascade_target":m["cascade_recall_target"],
                    "f1_score":m["f1_score"],"precision":m["precision"],"recall":m["recall"],
                    "cost_usd":m["cost_usd"],"llm_calls_total":m["llm_calls_total"],
                    "execution_time":m["execution_time"]})
print(f"\nSaved sweep CSV -> {csv_path}")
