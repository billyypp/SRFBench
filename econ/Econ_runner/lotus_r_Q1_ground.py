from dotenv import load_dotenv
load_dotenv()

import os, logging, threading, time
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
factors     = pd.read_csv('my_benchmark/econ/data/factors.csv')

EVENT = "The war in Ukraine"

print(f"Generating ground truth | Q1 | E08")
print(f"Pairs: {len(commodities)} x {len(factors)} = {len(commodities)*len(factors)}\n")


commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})", axis=1
)

factors["factor_info"] = factors.apply(
    lambda r: f"[{r['side']}] {r['factor_description']}", axis=1
)

result = commodities.sem_join(
    factors,
    f'Given that {EVENT}, considering all possible economic effects: does {{commodity_info}} experience {{factor_info}}?'
).reset_index(drop=True)

result[["commodity_name", "factor_id"]].to_csv('my_benchmark/econ/ground_truth/Q1_E08.csv', index=False)
print(f"Saved {len(result)} pairs -> my_benchmark/econ/ground_truth/Q1_E08.csv")
