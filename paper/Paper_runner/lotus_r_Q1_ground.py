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
from lotus.models import LM
from lotus.types import ReasoningStrategy

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
lotus.settings.configure(lm=lm)

claims       = pd.read_csv('my_benchmark/paper/data/scifact_claims.csv')
papers       = pd.read_csv('my_benchmark/paper/data/scifact_papers_60.csv', encoding='latin-1')
ground_truth = pd.read_csv('my_benchmark/paper/ground_truth/Q1.csv')

total_pairs = len(claims) * len(papers)
print(f"[ground] Paper Q1 | exact (oracle)")
print(f"Pairs: {len(claims)} x {len(papers)} = {total_pairs}\n")

t0 = time.time()
result = claims.sem_join(
    papers,
    "Should the research of {claim} cite {abstract}? And {abstract} support {claim}?",
    strategy=ReasoningStrategy.ZS_COT
).reset_index(drop=True)
elapsed = round(time.time() - t0, 2)

print(f"  -> {len(result)} matched")

res_pairs = set(zip(result["claim_id"], result["doc_id"]))
gt_pairs  = set(zip(ground_truth["claim_id"], ground_truth["doc_id"]))
tp        = len(res_pairs & gt_pairs)
precision = round(tp / len(res_pairs), 4) if res_pairs else 0.0
recall    = round(tp / len(gt_pairs),  4) if gt_pairs  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

result = result[["claim_id", "doc_id"]].drop_duplicates().reset_index(drop=True)
result.to_csv('my_benchmark/paper/results/lotus/Q1/ground.csv', index=False)

info = {
    "test_id": "ground", "query": "Q1", "scenario": "paper",
    "model_name": "gpt-5.4",
    "policy": "exact",
    "execution_time": elapsed,
    "llm_calls_total": total_pairs,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(res_pairs),
}
with open('my_benchmark/paper/results/lotus/Q1/ground_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/paper/results/lotus/Q1/ground.csv ({elapsed}s)")
