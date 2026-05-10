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

EVENT_IDS = ["E12", "E13", "E14", "E15", "E16"]

print(f"Generating ground truth | Q4 | {'+'.join(EVENT_IDS)}")
print(f"Commodities: {len(commodities)}\n")

# Preprocessing: construct filter key
commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']})",
    axis=1
)

# Build event context from the 5 selected events
event_set = events[events["event_id"].isin(EVENT_IDS)]
event_context = "; ".join(event_set["event_description"].tolist())

t0 = time.time()
# Semantic filter
result = commodities.sem_filter(
    f"Given the following concurrent events: [{event_context}], "
    f"is {{commodity_info}} heavily impacted by the combined "
    f"effect of these events?"
).reset_index(drop=True)
elapsed = round(time.time() - t0, 2)

# Projection
output = result[["commodity_name"]]

output.to_csv('my_benchmark/econ/ground_truth/Q4.csv', index=False)
print(f"Saved {len(output)} rows -> my_benchmark/econ/ground_truth/Q4.csv ({elapsed}s)")
