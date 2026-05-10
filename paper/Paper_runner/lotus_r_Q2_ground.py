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

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
lotus.settings.configure(lm=lm)

os.makedirs('my_benchmark/paper/ground_truth', exist_ok=True)
os.makedirs('my_benchmark/paper/results/lotus/Q2', exist_ok=True)
claims = pd.read_csv('my_benchmark/paper/data/scifact_claims_30.csv')
papers = pd.read_csv('my_benchmark/paper/data/scifact_papers_30.csv', encoding='latin-1')

K = 10
N_PAIRS = 50

print(f"Generating ground truth | Paper Q2 | Top-K Claim-Paper Support")
print(f"Claims: {len(claims)}, Papers: {len(papers)}, K={K}\n")

# Cross product, then take a fixed sample of N_PAIRS pairs
pairs = claims.merge(papers, how='cross')
pairs = pairs.sample(n=N_PAIRS, random_state=42).reset_index(drop=True)
print(f"PAIRS sampled: {len(pairs)} (seed=42)")

t0 = time.time()
output = pairs.sem_topk(
    "Which pair shows stronger evidential support of the claim by the paper?"
    " claim = {claim} abstract = {abstract}",
    K=K,
    return_stats=True
)
elapsed = round(time.time() - t0, 2)

result, stats = output if isinstance(output, tuple) else (output, {})
result = result.reset_index(drop=True)
print(f"  -> top {len(result)} | llm_calls: {stats.get('total_llm_calls', 0)}")

result[["claim_id", "doc_id"]].to_csv('my_benchmark/paper/ground_truth/Q2.csv', index=False)

oracle_stats = {
    "llm_calls_total": stats.get("total_llm_calls", 0),
    "lotus_stats": stats,
}
with open('my_benchmark/paper/ground_truth/Q2_stats.json', 'w') as f:
    json.dump(oracle_stats, f, indent=2)

print(f"\nSaved {len(result)} rows -> my_benchmark/paper/ground_truth/Q2.csv ({elapsed}s)")
print(f"Oracle LLM calls: {oracle_stats['llm_calls_total']}")
