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
_calls_by_model: dict = {}

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

def _record(kwargs, n=1):
    name = kwargs.get("model", "?")
    with _rate_lock:
        _calls_by_model[name] = _calls_by_model.get(name, 0) + n

_orig_completion = litellm.completion
def _rate_limited_completion(*args, **kwargs):
    _throttle()
    _record(kwargs, 1)
    return _orig_completion(*args, **_fix_kwargs(kwargs))
litellm.completion = _rate_limited_completion

_orig_batch = litellm.batch_completion
def _rate_limited_batch(*args, **kwargs):
    msgs = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
    _record(kwargs, max(1, len(msgs)))
    return _orig_batch(*args, **_fix_kwargs(kwargs))
litellm.batch_completion = _rate_limited_batch

import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
import lotus.models.lm as _lotus_lm_mod
_lotus_lm_mod.batch_completion = _rate_limited_batch
from lotus.vector_store.faiss_vs import FaissVS
from lotus.types import CascadeArgs
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'econ', 'runner'))
from _pricing import PRICING

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=4)
helper_lm = LM('gpt-4o-mini', max_tokens=512, max_batch_size=4)
rm = SentenceTransformersRM('intfloat/e5-base-v2')
vs = FaissVS()
lotus.settings.configure(lm=lm, helper_lm=helper_lm, rm=rm, vs=vs)
cascade = CascadeArgs(recall_target=0.8, precision_target=0.8)

os.makedirs('my_benchmark/paper/results/lotus/Q3', exist_ok=True)
papers = pd.read_csv('my_benchmark/paper/data/scifact_papers.csv', encoding='latin-1')
ground_truth = pd.read_csv('my_benchmark/paper/ground_truth/Q3.csv')

# First 500 papers (same as ground)
papers = papers.iloc[:500].reset_index(drop=True)
total_rows = len(papers)

print(f"[t1] Paper Q3 | approximate (recall=0.8, precision=0.8)")
print(f"Papers: {total_rows}\n")

t0 = time.time()
output = papers.sem_filter(
    "Suppose someone wants to start an impactful research project"
    " but has no specific topic in mind (given this person is interested"
    " in nature and cares about mathematical proofs)."
    " Could {abstract} give them an idea worth pursuing?",
    cascade_args=cascade,
    return_stats=True
)
elapsed = round(time.time() - t0, 2)

if isinstance(output, tuple):
    result, filter_stats = output
else:
    result, filter_stats = output, {}
result = result.reset_index(drop=True)

result_out = result[["doc_id"]]
print(f"  -> {len(result_out)} matched")
print(f"  stats: {filter_stats}")
print(f"  calls_by_model: {_calls_by_model}")

res_set = set(result_out["doc_id"])
gt_set  = set(ground_truth["doc_id"])
tp        = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set),  4) if gt_set  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

result_out.to_csv('my_benchmark/paper/results/lotus/Q3/t1.csv', index=False)

helper_resolved = filter_stats.get("filters_resolved_by_helper_model", 0)
main_resolved   = filter_stats.get("filters_resolved_by_large_model", 0)
total_main_calls   = _calls_by_model.get("gpt-5.4", 0)
total_helper_calls = _calls_by_model.get("gpt-4o-mini", 0)

def _cost(model_name, lm_obj):
    u = lm_obj.stats.physical_usage
    r = PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    return (u.prompt_tokens * r["input"] + u.completion_tokens * r["output"]) / 1e6

cost_usd = round(_cost('gpt-5.4', lm) + _cost('gpt-4o-mini', helper_lm), 6)

info = {
    "test_id": "t1", "query": "Q3", "scenario": "paper",
    "model_name": "gpt-5.4",
    "policy": "approximate",
    "cascade_recall_target": 0.8,
    "cascade_precision_target": 0.8,
    "execution_time": elapsed,
    "llm_calls_total": total_rows,
    "llm_calls_to_main_lm": main_resolved,
    "llm_calls_saved": total_rows - main_resolved,
    "llm_calls_calibration": total_helper_calls + max(0, total_main_calls - main_resolved),
    "cost_usd": cost_usd,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(result_out),
}
with open('my_benchmark/paper/results/lotus/Q3/t1_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/paper/results/lotus/Q3/t1.csv ({elapsed}s)")
