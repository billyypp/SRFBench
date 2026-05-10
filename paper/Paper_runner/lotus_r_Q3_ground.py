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
os.makedirs('my_benchmark/paper/results/lotus/Q3', exist_ok=True)
papers = pd.read_csv('my_benchmark/paper/data/scifact_papers.csv', encoding='latin-1')

# First 500 papers
papers = papers.iloc[:500].reset_index(drop=True)
total_rows = len(papers)

print(f"Generating ground truth | Paper Q3 | Inspiration Filter")
print(f"Papers: {total_rows}\n")

t0 = time.time()
result = papers.sem_filter(
    "Suppose someone wants to start an impactful research project"
    " but has no specific topic in mind (given this person is interested"
    " in nature and cares about mathematical proofs)."
    " Could {abstract} give them an idea worth pursuing?"
).reset_index(drop=True)
elapsed = round(time.time() - t0, 2)

output = result[["doc_id"]]
output.to_csv('my_benchmark/paper/ground_truth/Q3.csv', index=False)
print(f"  -> {len(output)} kept (LLM-judged, exact mode)")

info = {
    "test_id": "ground", "query": "Q3", "scenario": "paper",
    "model_name": "gpt-5.4",
    "policy": "exact",
    "execution_time": elapsed,
    "llm_calls_total": total_rows,
    "row_count": len(output),
}
with open('my_benchmark/paper/ground_truth/Q3_stats.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nSaved -> my_benchmark/paper/ground_truth/Q3.csv ({elapsed}s)")
