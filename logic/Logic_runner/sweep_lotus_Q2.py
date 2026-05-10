
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

DATA_DIR = "my_benchmark/logic/data"
GT_DIR   = "my_benchmark/logic/ground_truth"
OUT_DIR  = "my_benchmark/logic/results/lotus/Q2"
os.makedirs(OUT_DIR, exist_ok=True)

statements = pd.read_csv(f"{DATA_DIR}/folio_statements_20.csv")
gt = pd.read_csv(f"{GT_DIR}/Q2.csv")
gt_set = set(gt["statement_id"].astype(str))
K = 10

METHODS = [
    ("naive",    "naive"),     # full pairwise — most expensive
    ("quick",    "quick"),     # default quicksort
    ("heap",     "heap"),      # heapsort
    ("quick-sem","quicksem"),  # embedding-presort quicksort
]
PROMPT = ("Which statement's logic is harder for a lower schooler"
          " to understand? {statement}")

results = []
for method, test_id in METHODS:
    print(f"\n=== LOTUS logic Q2 sweep method={method} ({test_id}) ===")
    lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
    rm = SentenceTransformersRM('intfloat/e5-base-v2')
    vs = FaissVS()
    lotus.settings.configure(lm=lm, rm=rm, vs=vs)
    t0 = time.time()
    output = statements.sem_topk(PROMPT, K=K, method=method, return_stats=True)
    elapsed = round(time.time() - t0, 2)
    if isinstance(output, tuple): result_df, st = output
    else: result_df, st = output, {}
    result_df = result_df.reset_index(drop=True)
    res = set(result_df["statement_id"].astype(str))
    tp = len(res & gt_set)
    P = round(tp/len(res),4) if res else 0.0
    R = round(tp/len(gt_set),4) if gt_set else 0.0
    F1 = round(2*P*R/(P+R),4) if (P+R) else 0.0

    def _cost(model_name, lm_obj):
        u = lm_obj.stats.physical_usage
        r = PRICING.get(model_name, {"input":0.0,"output":0.0})
        return (u.prompt_tokens*r["input"] + u.completion_tokens*r["output"]) / 1e6
    cost = round(_cost('gpt-5.4', lm), 6)
    llm_calls = st.get("total_llm_calls", 0)

    info = {
        "test_id": test_id, "query": "Q2", "scenario": "logic",
        "system": "lotus", "model_name": "gpt-5.4",
        "policy": f"method={method}",
        "cascade_target": method,
        "execution_time": elapsed,
        "llm_calls_total": llm_calls,
        "cost_usd": cost,
        "precision": P, "recall": R, "f1_score": F1,
        "row_count": len(result_df),
    }
    with open(f"{OUT_DIR}/{test_id}_metrics.json", "w") as f:
        json.dump(info, f, indent=2)
    result_df[["statement_id"]].to_csv(f"{OUT_DIR}/{test_id}.csv", index=False)
    print(f"  -> F1={F1} cost=${cost:.4f} llm={llm_calls}")
    results.append(info)

csv_path = f"{OUT_DIR}/threshold_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["test_id","cascade_target","f1_score","precision","recall","cost_usd","llm_calls_total","execution_time"])
    w.writeheader()
    for m in results:
        w.writerow({"test_id":m["test_id"],"cascade_target":m["cascade_target"],
                    "f1_score":m["f1_score"],"precision":m["precision"],"recall":m["recall"],
                    "cost_usd":m["cost_usd"],"llm_calls_total":m["llm_calls_total"],
                    "execution_time":m["execution_time"]})
print(f"\nSaved sweep CSV -> {csv_path}")
