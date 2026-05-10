from dotenv import load_dotenv
load_dotenv(override=True)
import os
for _k in ("OPENAI_API_KEY", "OPENAI_ORGANIZATION"):
    if _k in os.environ:
        os.environ[_k] = os.environ[_k].strip().strip("\r\n")

import time, json, logging, threading
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True

_rate_lock = threading.Lock()
_call_times: list = []
_MAX_CALLS_PER_MIN = 450

def _fix_kwargs(kwargs):
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    return kwargs

def _throttle():
    while True:
        with _rate_lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60]
            if len(_call_times) < _MAX_CALLS_PER_MIN:
                _call_times.append(now)
                return
        time.sleep(0.5)

_orig_completion = litellm.completion
def _rate_limited_completion(*args, **kwargs):
    _throttle()
    return _orig_completion(*args, **_fix_kwargs(kwargs))
litellm.completion = _rate_limited_completion

_orig_batch = litellm.batch_completion
def _rate_limited_batch(*args, **kwargs):
    return _orig_batch(*args, **_fix_kwargs(kwargs))
litellm.batch_completion = _rate_limited_batch

import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
import lotus.models.lm as _lotus_lm_mod
_lotus_lm_mod.batch_completion = _rate_limited_batch
from lotus.vector_store.faiss_vs import FaissVS
from lotus.types import CascadeArgs

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=4)
helper_lm = LM('gpt-4o-mini', max_tokens=512, max_batch_size=4)
rm = SentenceTransformersRM('intfloat/e5-base-v2')
vs = FaissVS()
lotus.settings.configure(lm=lm, helper_lm=helper_lm, rm=rm, vs=vs)
cascade = CascadeArgs(recall_target=0.8, precision_target=0.8)

os.makedirs('my_benchmark/econ/results/lotus/Q4', exist_ok=True)
commodities  = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')
events       = pd.read_csv('my_benchmark/econ/data/events.csv')
ground_truth = pd.read_csv('my_benchmark/econ/ground_truth/Q4.csv')

EVENT_IDS = ["E12", "E13", "E14", "E15", "E16"]

total_rows = len(commodities)
print(f"[t1] Q4 | {'+'.join(EVENT_IDS)} | approximate (recall=0.8, precision=0.8)")
print(f"Commodities: {total_rows}\n")

# Preprocessing: construct filter key
commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})",
    axis=1
)

# Build event context from the 5 selected events
event_set = events[events["event_id"].isin(EVENT_IDS)]
event_context = "; ".join(event_set["event_description"].tolist())

t0 = time.time()
# Semantic filter with cascade
output = commodities.sem_filter(
    f"Given the following concurrent events: [{event_context}], "
    f"is {{commodity_info}} heavily impacted by the combined "
    f"effect of these events?",
    cascade_args=cascade,
    return_stats=True
)
elapsed = round(time.time() - t0, 2)

if isinstance(output, tuple):
    result, filter_stats = output
else:
    result, filter_stats = output, {}
result = result.reset_index(drop=True)

# Projection
result_out = result[["commodity_name"]]

print(f"  -> {len(result_out)} matched")
print(f"  stats: {filter_stats}")

res_set = set(result_out["commodity_name"])
gt_set  = set(ground_truth["commodity_name"])
tp        = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set),  4) if gt_set  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

result_out.to_csv('my_benchmark/econ/results/lotus/Q4/t1.csv', index=False)

info = {
    "test_id": "t1", "query": "Q4", "event_id": "+".join(EVENT_IDS),
    "model_name": "gpt-5.4",
    "policy": "approximate",
    "cascade_recall_target": 0.8,
    "cascade_precision_target": 0.8,
    "execution_time": elapsed,
    "llm_calls_total": total_rows,
    "llm_calls_to_main_lm": filter_stats.get("filter_resolved_by_large_model", total_rows),
    "llm_calls_saved": filter_stats.get("filter_resolved_by_helper_model", 0),
    "llm_calls_calibration": filter_stats.get("optimized_filter_cost", 0),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(result_out),
}
with open('my_benchmark/econ/results/lotus/Q4/t1_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/econ/results/lotus/Q4/t1.csv ({elapsed}s)")
