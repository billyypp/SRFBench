from dotenv import load_dotenv
load_dotenv()

import time, json, os, logging, threading

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True

_rate_lock = threading.Lock()
_call_times: list = []
_MAX_CALLS_PER_MIN = 2000

_orig_completion = litellm.completion
def _rate_limited_completion(*args, **kwargs):
    while True:
        with _rate_lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60]
            if len(_call_times) < _MAX_CALLS_PER_MIN:
                _call_times.append(now)
                break
        time.sleep(0.5)
    return _orig_completion(*args, **kwargs)

litellm.completion = _rate_limited_completion

import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store.faiss_vs import FaissVS

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
rm = SentenceTransformersRM('intfloat/e5-base-v2')
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

os.makedirs('my_benchmark/econ/results/lotus/Q3', exist_ok=True)
commodities  = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')
ground_truth = pd.read_csv('my_benchmark/econ/ground_truth/Q3.csv')
with open('my_benchmark/econ/ground_truth/Q3_stats.json') as f:
    oracle_stats = json.load(f)

K = 10

print(f"[t1] Q3 | approximate (method=quick-sem)")
print(f"Commodities: {len(commodities)}, K={K}\n")
t0 = time.time()

commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})", axis=1
)

# Ranking 1: demand inelasticity
print("[1/2] Ranking by demand inelasticity")
output1 = commodities.sem_topk(
    "Which commodity has more inelastic demand"
    " (harder for consumers to substitute)?"
    " {commodity_info}",
    K=K,
    method="quick-sem",
    return_stats=True
)
inelastic, stats1 = output1 if isinstance(output1, tuple) else (output1, {})
inelastic = inelastic.reset_index(drop=True)
print(f"  stats: {stats1}")

# Ranking 2: strategic importance to the United States
print("[2/2] Ranking by US strategic importance")
output2 = commodities.sem_topk(
    "Which commodity is more strategically important"
    " to the United States, considering national security,"
    " import dependence, and critical infrastructure?"
    " {commodity_info}",
    K=K,
    method="quick-sem",
    return_stats=True
)
us_strategic, stats2 = output2 if isinstance(output2, tuple) else (output2, {})
us_strategic = us_strategic.reset_index(drop=True)
print(f"  stats: {stats2}")

# Set intersection
output = pd.merge(
    inelastic[["commodity_name"]],
    us_strategic[["commodity_name"]],
    on="commodity_name"
)
print(f"\nIntersection: {len(output)} commodities")

elapsed = round(time.time() - t0, 2)

# Evaluation vs ground truth
res_set = set(output["commodity_name"])
gt_set  = set(ground_truth["commodity_name"])
tp        = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set),  4) if gt_set  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

output.to_csv('my_benchmark/econ/results/lotus/Q3/t1.csv', index=False)

optimized_calls = stats1.get("total_llm_calls", 0) + stats2.get("total_llm_calls", 0)
oracle_calls = oracle_stats["llm_calls_total"]

print(f"  Oracle calls: {oracle_calls} | Optimized calls: {optimized_calls} | Saved: {oracle_calls - optimized_calls}")

info = {
    "test_id": "t1", "query": "Q3", "event_id": "none",
    "model_name": "gpt-5.4",
    "policy": "optimized",
    "method": "quick-sem",
    "execution_time": elapsed,
    "llm_calls_total": oracle_calls,
    "llm_calls_to_main_lm": optimized_calls,
    "llm_calls_saved": oracle_calls - optimized_calls,
    "llm_calls_calibration": 0,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(output),
}
with open('my_benchmark/econ/results/lotus/Q3/t1_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/econ/results/lotus/Q3/t1.csv ({elapsed}s)")
