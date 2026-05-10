from dotenv import load_dotenv
load_dotenv()

import time, json, os, logging, threading, re
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
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store.faiss_vs import FaissVS
from lotus.types import CascadeArgs, ProxyModel

lm = LM('gpt-5.4', max_tokens=512, max_batch_size=5)
rm = SentenceTransformersRM('intfloat/e5-base-v2')
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)
cascade = CascadeArgs(recall_target=0.8, precision_target=0.8, proxy_model=ProxyModel.EMBEDDING_MODEL)

os.makedirs('my_benchmark/econ/results/lotus/Q2', exist_ok=True)
commodities  = pd.read_csv('my_benchmark/econ/data/commodities (2).csv')
factors      = pd.read_csv('my_benchmark/econ/data/factors.csv')
equilibrium  = pd.read_csv('my_benchmark/econ/data/equilibrium.csv')
predictions  = pd.read_csv('my_benchmark/econ/data/predictions.csv')
ground_truth = pd.read_csv('my_benchmark/econ/ground_truth/Q2_E15.csv')

EVENT = "Armed conflict erupted in the Middle East, threatening major shipping routes through the Strait of Hormuz"

print(f"[t1] Q2 | E15 | gpt-4o + embedding cascade")
t0 = time.time()

# --- Step 1: Commodity x Factor (sem_join with cascade) ---
commodities["commodity_info"] = commodities.apply(
    lambda r: f"{r['commodity_name']} ({r['category']}): {r['description']}", axis=1
)
factors["factor_info"] = factors.apply(
    lambda r: f"[{r['side']}] {r['factor_description']}", axis=1
)
total = len(commodities) * len(factors)
print(f"\n[1/3] Commodity x Factor: {len(commodities)} x {len(factors)} = {total} pairs")
lm.reset_stats()
output = commodities.sem_join(
    factors,
    f'Given that {EVENT}, considering all its economic effects: does {{commodity_info}} experience {{factor_info}}?',
    cascade_args=cascade,
    return_stats=True
)
cf, stats = output if isinstance(output, tuple) else (output, {})
cf = cf.reset_index(drop=True)
print(f"  -> {len(cf)} matched | main_lm: {stats.get('join_resolved_by_large_model', total)} | saved: {stats.get('join_resolved_by_helper_model', 0)}")

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

elapsed = round(time.time() - t0, 2)

# --- Final evaluation vs ground truth ---
res_pairs = set(zip(cfep["commodity_name"], cfep["factor_id"], cfep["outcome_id"], cfep["prediction_id"]))
gt_pairs  = set(zip(ground_truth["commodity_name"], ground_truth["factor_id"], ground_truth["outcome_id"], ground_truth["prediction_id"]))
tp        = len(res_pairs & gt_pairs)
precision = round(tp / len(res_pairs), 4) if res_pairs else 0.0
recall    = round(tp / len(gt_pairs),  4) if gt_pairs  else 0.0
f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) else 0.0

cfep[["commodity_name", "factor_id", "outcome_id", "prediction_id"]].to_csv(
    'my_benchmark/econ/results/lotus/Q2/t1.csv', index=False
)

info = {
    "test_id": "t1", "query": "Q2", "event_id": "E15",
    "model_name": "gpt-5.4",
    "cascade": "embedding (recall=0.8, precision=0.8)",
    "execution_time": elapsed,
    "llm_calls_total": stats.get("join_resolved_by_large_model", total) + stats.get("join_resolved_by_helper_model", 0),
    "llm_calls_to_main_lm": stats.get("join_resolved_by_large_model", total),
    "llm_calls_saved": stats.get("join_resolved_by_helper_model", 0),
    "llm_calls_calibration": stats.get("optimized_join_cost", 0),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "row_count": len(cfep),
}
with open('my_benchmark/econ/results/lotus/Q2/t1_metrics.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\nP={precision}  R={recall}  F1={f1}")
print(f"Saved -> my_benchmark/econ/results/lotus/Q2/t1.csv ({elapsed}s)")
