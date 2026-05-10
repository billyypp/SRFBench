from dotenv import load_dotenv
load_dotenv()

import os, json, logging, threading, time
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

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
lotus.settings.configure(lm=lm)

os.makedirs('my_benchmark/logic/ground_truth', exist_ok=True)
os.makedirs('my_benchmark/logic/results/lotus/Q2', exist_ok=True)
nl_statements = pd.read_csv('my_benchmark/logic/data/folio_statements_20.csv')

K = 10

print(f"Generating ground truth | Logic Q2 | Top-K Difficulty")
print(f"Statements: {len(nl_statements)}, K={K}\n")

t0 = time.time()
output = nl_statements.sem_topk(
    "Which statement's logic is harder for a lower schooler"
    " to understand? {statement}",
    K=K,
    return_stats=True
)
elapsed = round(time.time() - t0, 2)

result, stats = output if isinstance(output, tuple) else (output, {})
result = result.reset_index(drop=True)
print(f"  -> top {len(result)} | llm_calls: {stats.get('total_llm_calls', 0)}")

result[["statement_id"]].to_csv('my_benchmark/logic/ground_truth/Q2.csv', index=False)

oracle_stats = {
    "llm_calls_total": stats.get("total_llm_calls", 0),
    "lotus_stats": stats,
}
with open('my_benchmark/logic/ground_truth/Q2_stats.json', 'w') as f:
    json.dump(oracle_stats, f, indent=2)

print(f"\nSaved {len(result)} rows -> my_benchmark/logic/ground_truth/Q2.csv ({elapsed}s)")
print(f"Oracle LLM calls: {oracle_stats['llm_calls_total']}")
