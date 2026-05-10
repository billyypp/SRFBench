
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

# 10 from start of first_30, 10 from start of last_30
first_10 = nl_statements.iloc[:10].reset_index(drop=True)
last_10  = nl_statements.iloc[-30:-20].reset_index(drop=True)
print(f"[smoke] first_10: {len(first_10)} | last_10: {len(last_10)}")

A = first_10.rename(columns={'statement_id': 'statement_id_a', 'statement': 'statement_a'})
B = last_10.rename(columns={'statement_id': 'statement_id_b', 'statement': 'statement_b'})
pairs = A.merge(B, how='cross')
total_pairs = len(pairs)
print(f"Pairs: {total_pairs}\n")

t0 = time.time()
result = pairs.sem_filter(
    "The propositional logic translation of {statement_a}"
    " entails the propositional logic translation of {statement_b}."
    " (variable1=P, variable2=Q, variable3=R, variable4=S, variable5=T, variable6=U)",
    strategy=ReasoningStrategy.COT
).reset_index(drop=True)
elapsed = round(time.time() - t0, 2)

llm_out = result[['statement_id_a', 'statement_id_b']]
print(f"  -> {len(llm_out)} entailed pairs (LLM-judged)")

# Restrict gold to the 10x10 subset
sids_a = set(first_10['statement_id'])
sids_b = set(last_10['statement_id'])
gt_subset = ground_truth[
    ground_truth['statement_id_a'].isin(sids_a)
    & ground_truth['statement_id_b'].isin(sids_b)
]

res_set = set(map(tuple, llm_out.values))
gt_set  = set(map(tuple, gt_subset[['statement_id_a', 'statement_id_b']].values))
tp        = len(res_set & gt_set)
precision = round(tp / len(res_set), 4) if res_set else 0.0
recall    = round(tp / len(gt_set),  4) if gt_set  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

print(f"\nLLM (smoke) vs gold subset:")
print(f"  pairs: {total_pairs} | gold: {len(gt_set)} | llm: {len(res_set)} | tp: {tp}")
print(f"  P={precision}  R={recall}  F1={f1}")
print(f"  elapsed: {elapsed}s")

llm_out.to_csv('my_benchmark/logic/results/lotus/Q3/smoke.csv', index=False)
info = {
    "test_id": "smoke", "query": "Q3", "scenario": "logic",
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
with open('my_benchmark/logic/results/lotus/Q3/smoke_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f"Saved -> my_benchmark/logic/results/lotus/Q3/smoke.csv")
