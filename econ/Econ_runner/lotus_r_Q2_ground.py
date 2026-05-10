from dotenv import load_dotenv
load_dotenv()

import os, logging, threading, time, re
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
equilibrium = pd.read_csv('my_benchmark/econ/data/equilibrium.csv')
predictions = pd.read_csv('my_benchmark/econ/data/predictions.csv')

EVENT = "Armed conflict erupted in the Middle East, threatening major shipping routes through the Strait of Hormuz"

print(f"Generating ground truth | Q2 | E15")

# --- Step 1: Commodity x Factor (sem_join) ---
commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']}): {r['description']}", axis=1
)
factors["factor_info"] = factors.apply(
    lambda r: f"[{r['side']}] {r['factor_description']}", axis=1
)
print(f"\n[1/3] Commodity x Factor: {len(commodities)} x {len(factors)} = {len(commodities)*len(factors)} pairs")
cf = commodities.sem_join(
    factors,
    f'Given that {EVENT}, considering all its economic effects: does {{commodity_info}} experience {{factor_info}}?'
).reset_index(drop=True)
print(f"  -> {len(cf)} matched")

# --- Step 2: Classify each commodity to its single best equilibrium (sem_map) ---
cf_agg = cf.groupby("commodity_name").agg(
    description=("description", "first"),
    factor_descriptions=("factor_description", lambda x: "; ".join(x))
).reset_index()
cf_agg["cf_context"] = cf_agg.apply(
    lambda r: f"{r['commodity_name']} ({r['description']}) experiencing: {r['factor_descriptions']}", axis=1
)
eq_options = "\n".join([
    f"{r['outcome_id']}: supply shift={r['supply_shift']}, demand shift={r['demand_shift']}"
    for _, r in equilibrium.iterrows()
])
print(f"\n[2/3] Classify {len(cf_agg)} commodities to best equilibrium")
mapped2 = cf_agg.sem_map(
    f'Given that {EVENT} and {{cf_context}}: which equilibrium outcome best describes the net market result?\nOptions:\n{eq_options}\nReturn only the outcome_id (e.g. EQ01).'
)
cf_agg["outcome_id"] = mapped2["_map"].apply(
    lambda x: m.group(0) if (m := re.search(r'EQ\d+', str(x))) else None
)
cfe_agg = cf_agg.merge(
    equilibrium[["outcome_id", "supply_shift", "demand_shift", "equilibrium_price", "equilibrium_quantity"]],
    on="outcome_id"
)
cfe = cfe_agg.merge(cf[["commodity_name", "factor_id"]], on="commodity_name")
print(f"  -> {len(cfe_agg)} equilibrium assignments, {len(cfe)} rows after factor merge")

# --- Step 3: Classify each commodity to its single best price prediction (sem_map) ---
cfe["cfe_context"] = cfe.apply(
    lambda r: f"{r['commodity_name']} ({r['description']}) experiencing {r['factor_descriptions']}, "
              f"supply shift: {r['supply_shift']}, demand shift: {r['demand_shift']}, "
              f"equilibrium price: {r['equilibrium_price']}, quantity: {r['equilibrium_quantity']}", axis=1
)
pred_options = "\n".join([
    f"{r['prediction_id']}: {r['equilibrium_price_change']} by {r['magnitude']}"
    for _, r in predictions.iterrows()
])
# classify at commodity level (all rows share the same context per commodity)
cfe_per_commodity = cfe.groupby("commodity_name").first().reset_index()
print(f"\n[3/3] Classify {len(cfe_per_commodity)} commodities to best price prediction")
mapped3 = cfe_per_commodity.sem_map(
    f'Given that {EVENT} and {{cfe_context}}: which price prediction is most accurate?\nOptions:\n{pred_options}\nReturn only the prediction_id (e.g. P01).'
)
cfe_per_commodity["prediction_id"] = mapped3["_map"].apply(
    lambda x: m.group(0) if (m := re.search(r'P\d+', str(x))) else None
)
cfep = cfe.merge(cfe_per_commodity[["commodity_name", "prediction_id"]], on="commodity_name")
cfep = cfep.dropna(subset=["prediction_id"])
print(f"  -> {len(cfep)} rows")

cfep[["commodity_name", "factor_id", "outcome_id", "prediction_id"]].to_csv(
    'my_benchmark/econ/ground_truth/Q2_E15.csv', index=False
)
print(f"\nSaved {len(cfep)} rows -> my_benchmark/econ/ground_truth/Q2_E15.csv")
