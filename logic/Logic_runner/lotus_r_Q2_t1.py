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
_calls_by_model: dict = {}

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
    name = kwargs.get("model", "?")
    with _rate_lock:
        _calls_by_model[name] = _calls_by_model.get(name, 0) + 1
    return _orig_completion(*args, **kwargs)

litellm.completion = _rate_limited_completion

import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store.faiss_vs import FaissVS
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'econ', 'runner'))
from _pricing import PRICING

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
rm = SentenceTransformersRM('intfloat/e5-base-v2')
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

os.makedirs('my_benchmark/logic/results/lotus/Q2', exist_ok=True)
nl_statements = pd.read_csv('my_benchmark/logic/data/folio_statements_20.csv')
ground_truth  = pd.read_csv('my_benchmark/logic/ground_truth/Q2.csv')
with open('my_benchmark/logic/ground_truth/Q2_stats.json') as f:
    oracle_stats = json.load(f)

K = 10
total_rows = len(nl_statements)

print(f"[t1] Logic Q2 | approximate (method=quick-sem)")
print(f"Statements: {total_rows}, K={K}\n")

t0 = time.time()
output = nl_statements.sem_topk(
    "Which statement's logic is harder for a lower schooler"
    " to understand? {statement}",
    K=K,
    method="quick-sem",
    return_stats=True
)
elapsed = round(time.time() - t0, 2)

result, stats = output if isinstance(output, tuple) else (output, {})
result = result.reset_index(drop=True)
print(f"  -> top {len(result)}")
print(f"  stats: {stats}")

# Eval vs ground truth (set-based P/R/F1)
res_set = set(result["statement_id"])
gt_set  = set(ground_truth["statement_id"])
tp        = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set),  4) if gt_set  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

result[["statement_id"]].to_csv('my_benchmark/logic/results/lotus/Q2/t1.csv', index=False)

optimized_calls = stats.get("total_llm_calls", 0)
oracle_calls    = oracle_stats["llm_calls_total"]

def _cost(model_name, lm_obj):
    u = lm_obj.stats.physical_usage
    r = PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    return (u.prompt_tokens * r["input"] + u.completion_tokens * r["output"]) / 1e6

cost_usd = round(_cost('gpt-5.4', lm), 6)

info = {
    "test_id": "t1", "query": "Q2", "scenario": "logic",
    "model_name": "gpt-5.4",
    "policy": "optimized",
    "method": "quick-sem",
    "execution_time": elapsed,
    "llm_calls_total": oracle_calls,
    "llm_calls_to_main_lm": optimized_calls,
    "llm_calls_saved": oracle_calls - optimized_calls,
    "llm_calls_calibration": 0,
    "cost_usd": cost_usd,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(result),
}
with open('my_benchmark/logic/results/lotus/Q2/t1_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"  Oracle calls: {oracle_calls} | Optimized calls: {optimized_calls} | Saved: {oracle_calls - optimized_calls}")
print(f"Saved -> my_benchmark/logic/results/lotus/Q2/t1.csv ({elapsed}s)")
