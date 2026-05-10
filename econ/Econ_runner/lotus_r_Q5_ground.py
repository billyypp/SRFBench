from dotenv import load_dotenv
load_dotenv(override=True)
import os
for _k in ("OPENAI_API_KEY", "OPENAI_ORGANIZATION"):
    if _k in os.environ:
        os.environ[_k] = os.environ[_k].strip().strip("\r\n")

import time, json, os, logging, threading
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True

_rate_lock = threading.Lock()
_call_times: list = []
_MAX_CALLS_PER_MIN = 450

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

_orig_completion = litellm.completion
def _rate_limited_completion(*args, **kwargs):
    _throttle()
    return _orig_completion(*args, **_fix_kwargs(kwargs))
litellm.completion = _rate_limited_completion

_orig_batch = litellm.batch_completion
def _rate_limited_batch(*args, **kwargs):
    return _orig_batch(*args, **_fix_kwargs(kwargs))
litellm.batch_completion = _rate_limited_batch

import pandas as pd
import lotus
from lotus.models import LM
import lotus.models.lm as _lotus_lm_mod
_lotus_lm_mod.batch_completion = _rate_limited_batch

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=4)
lotus.settings.configure(lm=lm)

os.makedirs('my_benchmark/econ/ground_truth', exist_ok=True)
commodities = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')
events      = pd.read_csv('my_benchmark/econ/data/events.csv')

print(f"Generating ground truth | Q5 | Event-Commodity Primary-Driver Filter")
print(f"Commodities: {len(commodities)}, Events: {len(events)}")

# STRAT_HALF: stratified half-sample of commodities (one slice per category,
# preserving category proportions). With 68 commodities -> |STRAT_HALF| = 34.
commodities_half = (
    commodities
    .groupby('category', group_keys=False)
    .apply(lambda g: g.iloc[: (len(g) + 1) // 2])
    .reset_index(drop=True)
)
print(f"STRAT_HALF size: {len(commodities_half)}\n")

# Constructed filter key
commodities_half["commodity_info"] = commodities_half.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})", axis=1
)

# Cross join: every sampled commodity x every event
pairs = commodities_half.merge(events, how='cross')
print(f"Cross-joined pairs: {len(pairs)}")

t0 = time.time()
# Semantic filter: primary-driver predicate
result = pairs.sem_filter(
    "Is {event_description} a primary driver of"
    " the equilibrium price of {commodity_info}?"
).reset_index(drop=True)
elapsed = round(time.time() - t0, 2)

# Projection
output = result[["commodity_name", "event_id"]]

output.to_csv('my_benchmark/econ/ground_truth/Q5.csv', index=False)
print(f"Saved {len(output)} rows -> my_benchmark/econ/ground_truth/Q5.csv ({elapsed}s)")
