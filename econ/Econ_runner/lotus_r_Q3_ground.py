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

os.makedirs('my_benchmark/econ/ground_truth', exist_ok=True)
commodities = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')

K = 10

print(f"Generating ground truth | Q3")
print(f"Commodities: {len(commodities)}, K={K}\n")

commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})", axis=1
)

# Ranking 1: demand inelasticity (oracle — random pivots)
print("[1/2] Ranking by demand inelasticity")
output1 = commodities.sem_topk(
    "Which commodity has more inelastic demand"
    " (harder for consumers to substitute)?"
    " {commodity_info}",
    K=K,
    return_stats=True
)
inelastic, stats1 = output1 if isinstance(output1, tuple) else (output1, {})
inelastic = inelastic.reset_index(drop=True)
print(f"  -> top {len(inelastic)} | llm_calls: {stats1.get('total_llm_calls', 0)}")

# Ranking 2: strategic importance to the United States
print("[2/2] Ranking by US strategic importance")
output2 = commodities.sem_topk(
    "Which commodity is more strategically important"
    " to the United States, considering national security,"
    " import dependence, and critical infrastructure?"
    " {commodity_info}",
    K=K,
    return_stats=True
)
us_strategic, stats2 = output2 if isinstance(output2, tuple) else (output2, {})
us_strategic = us_strategic.reset_index(drop=True)
print(f"  -> top {len(us_strategic)} | llm_calls: {stats2.get('total_llm_calls', 0)}")

# Set intersection
output = pd.merge(
    inelastic[["commodity_name"]],
    us_strategic[["commodity_name"]],
    on="commodity_name"
)
print(f"\nIntersection: {len(output)} commodities")

output.to_csv('my_benchmark/econ/ground_truth/Q3.csv', index=False)

# Save oracle call counts so t1 can reference them
oracle_stats = {
    "llm_calls_ranking1": stats1.get("total_llm_calls", 0),
    "llm_calls_ranking2": stats2.get("total_llm_calls", 0),
    "llm_calls_total": stats1.get("total_llm_calls", 0) + stats2.get("total_llm_calls", 0),
    "lotus_stats_ranking1": stats1,
    "lotus_stats_ranking2": stats2,
}
with open('my_benchmark/econ/ground_truth/Q3_stats.json', 'w') as f:
    json.dump(oracle_stats, f, indent=2)

print(f"Saved {len(output)} rows -> my_benchmark/econ/ground_truth/Q3.csv")
print(f"Oracle LLM calls: {oracle_stats['llm_calls_total']}")
