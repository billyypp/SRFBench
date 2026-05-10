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
from lotus.types import CascadeArgs

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
rm = SentenceTransformersRM('intfloat/e5-base-v2')
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)
cascade = CascadeArgs(recall_target=0.8, precision_target=0.8)

os.makedirs('my_benchmark/econ/results/lotus/Q1', exist_ok=True)
commodities  = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')
factors      = pd.read_csv('my_benchmark/econ/data/factors.csv')
ground_truth = pd.read_csv('my_benchmark/econ/ground_truth/Q1_E08.csv')

EVENT = "The war in Ukraine"

commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})", axis=1
)
factors["factor_info"] = factors.apply(
    lambda r: f"[{r['side']}] {r['factor_description']}", axis=1
)

total_pairs = len(commodities) * len(factors)
print(f"[t1] Q1 | E08 | approximate (recall=0.8, precision=0.8)")
print(f"Pairs: {len(commodities)} x {len(factors)} = {total_pairs}\n")

t0 = time.time()
output = commodities.sem_join(
    factors,
    f'Given that {EVENT}, considering all possible economic effects: does {{commodity_info}} experience {{factor_info}}?',
    cascade_args=cascade,
    return_stats=True
)
elapsed = round(time.time() - t0, 2)

if isinstance(output, tuple):
    result, join_stats = output
else:
    result, join_stats = output, {}
result = result.reset_index(drop=True)

print(f"  stats: {join_stats}")

res_pairs = set(zip(result["commodity_name"], result["factor_id"]))
gt_pairs  = set(zip(ground_truth["commodity_name"], ground_truth["factor_id"]))
tp        = len(res_pairs & gt_pairs)
precision = round(tp / len(res_pairs), 4) if res_pairs else 0.0
recall    = round(tp / len(gt_pairs),  4) if gt_pairs  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

result[["commodity_name", "factor_id"]].to_csv('my_benchmark/econ/results/lotus/Q1/t1.csv', index=False)

info = {
    "test_id": "t1", "query": "Q1", "event_id": "E08",
    "model_name": "gpt-5.4",
    "policy": "approximate",
    "cascade_recall_target": 0.8,
    "cascade_precision_target": 0.8,
    "execution_time": elapsed,
    "llm_calls_total": total_pairs,
    "llm_calls_to_main_lm": join_stats.get("join_resolved_by_large_model", total_pairs),
    "llm_calls_saved": join_stats.get("join_resolved_by_helper_model", 0),
    "llm_calls_calibration": join_stats.get("optimized_join_cost", 0),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(result),
}
with open('my_benchmark/econ/results/lotus/Q1/t1_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/econ/results/lotus/Q1/t1.csv ({elapsed}s)")
