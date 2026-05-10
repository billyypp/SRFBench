from dotenv import load_dotenv
load_dotenv()

import os, json, time, logging, threading
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
from lotus.models import LM
from lotus.types import ReasoningStrategy

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
lotus.settings.configure(lm=lm)

os.makedirs('my_benchmark/logic/results/lotus/Q3', exist_ok=True)
nl_statements = pd.read_csv('my_benchmark/logic/data/folio_statements_20.csv')
ground_truth  = pd.read_csv('my_benchmark/logic/ground_truth/Q3.csv')

first_30 = nl_statements.iloc[:30].reset_index(drop=True)
last_30  = nl_statements.iloc[-30:].reset_index(drop=True)
print(f"first_30: {len(first_30)} | last_30: {len(last_30)}")

A = first_30.rename(columns={'statement_id': 'statement_id_a', 'statement': 'statement_a'})
B = last_30.rename(columns={'statement_id': 'statement_id_b', 'statement': 'statement_b'})
pairs = A.merge(B, how='cross')
total_pairs = len(pairs)
print(f"Cross-joined pairs: {total_pairs}\n")

t0 = time.time()
result = pairs.sem_filter(
    "The propositional logic translation of {statement_a}"
    " entails the propositional logic translation of {statement_b}."
    " (variable1=P, variable2=Q, variable3=R, variable4=S, variable5=T, variable6=U)",
    strategy=ReasoningStrategy.COT
).reset_index(drop=True)
elapsed = round(time.time() - t0, 2)

llm_out = result[['statement_id_a', 'statement_id_b']]
llm_out.to_csv('my_benchmark/logic/results/lotus/Q3/ground.csv', index=False)
print(f"  -> {len(llm_out)} entailed pairs (LLM-judged, exact mode)")

# Compare LLM output vs data-derived gold (Q3.csv built by generate_ground_truth.py)
res_set = set(map(tuple, llm_out.values))
gt_set  = set(map(tuple, ground_truth[['statement_id_a', 'statement_id_b']].values))
tp        = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set),  4) if gt_set  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

info = {
    "test_id": "ground", "query": "Q3", "scenario": "logic",
    "model_name": "gpt-5.4",
    "policy": "exact",
    "execution_time": elapsed,
    "llm_calls_total": total_pairs,
    "llm_entailed_count": len(llm_out),
    "gold_entailed_count": len(gt_set),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(llm_out),
}
with open('my_benchmark/logic/results/lotus/Q3/ground_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)
# Also write to ground_truth/ for the t1 runner to read (matches econ Q3 pattern)
with open('my_benchmark/logic/ground_truth/Q3_stats.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nLLM (exact) vs gold:  P={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/logic/results/lotus/Q3/ground.csv ({elapsed}s)")
