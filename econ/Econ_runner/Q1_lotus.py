
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse, json, os, sys, time, logging, pathlib, threading

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

_rate_lock = threading.Lock()
_call_times: list = []
_MAX_CALLS_PER_MIN = 500

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
from lotus.types import CascadeArgs

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _pricing import PRICING


def parse_args():
    p = argparse.ArgumentParser(description="Q1 LOTUS econ benchmark")
    p.add_argument("--event_id",   default="E08")
    p.add_argument("--event_text", default="The war in Ukraine")
    p.add_argument("--data_dir",   default="my_benchmark/econ/data",
                   help="Directory containing commodities and factors CSVs")
    p.add_argument("--gt_dir",     default="my_benchmark/econ/ground_truth",
                   help="Directory containing Q1_<event_id>.csv ground truth")
    p.add_argument("--model_name", default="gpt-5.4")
    p.add_argument("--max_batch_size", type=int, default=5)
    p.add_argument("--embedding_model", default="intfloat/e5-base-v2")
   
    p.add_argument("--recall_target",    type=float, default=0.8,
                   help="Cascade embedding pre-filter recall target (default: 0.8)")
    p.add_argument("--precision_target", type=float, default=0.8,
                   help="Cascade embedding pre-filter precision target (default: 0.8)")
    p.add_argument("--no_cascade", action="store_true",
                   help="Disable cascade; every pair goes to the LLM (exact mode)")
    
    p.add_argument("--max_cost",   type=float, default=None,
                   help="[post-hoc] Flag OVER_BUDGET if cost exceeds this USD amount")
    p.add_argument("--f1_floor",   type=float, default=None,
                   help="[post-hoc] Flag BELOW_TARGET if F1 falls below threshold")
    p.add_argument("--test_id",    default="t1")
    return p.parse_args()


def main():
    args = parse_args()

   
    lm = LM(args.model_name, max_tokens=512, max_batch_size=args.max_batch_size)
    rm = SentenceTransformersRM(args.embedding_model)
    vs = FaissVS()
    lotus.settings.configure(lm=lm, rm=rm, vs=vs)

    cascade = None
    if not args.no_cascade:
        cascade = CascadeArgs(
            recall_target=args.recall_target,
            precision_target=args.precision_target,
        )

   
    comm_path = os.path.join(args.data_dir, "commodities.csv")
    if not os.path.exists(comm_path):
        comm_path = os.path.join(args.data_dir, "commodities (2).csv")
    commodities  = pd.read_csv(comm_path)
    factors      = pd.read_csv(os.path.join(args.data_dir, "factors.csv"))
    ground_truth = pd.read_csv(os.path.join(args.gt_dir, f"Q1_{args.event_id}.csv"))

    commodities["commodity_info"] = commodities.apply(
        lambda r: f"{r['commodity_name']} ({r['category']})", axis=1)
    factors["factor_info"] = factors.apply(
        lambda r: f"[{r['side']}] {r['factor_description']}", axis=1)

    total_pairs = len(commodities) * len(factors)
    gt_pairs = set(zip(ground_truth["commodity_name"].astype(str),
                       ground_truth["factor_id"].astype(str)))

    mode = "exact" if args.no_cascade else f"cascade(R={args.recall_target}, P={args.precision_target})"
    print(f"[lotus] event={args.event_id}  model={args.model_name}  mode={mode}")
    print(f"[lotus] pairs={total_pairs}  gt={len(gt_pairs)}")
    print(f"[lotus] post-hoc labels: max_cost=${args.max_cost}  f1_floor={args.f1_floor}")

    join_kwargs = {
        "cascade_args": cascade,
        "return_stats": True,
    }

    t0 = time.time()
    output = commodities.sem_join(
        factors,
        (f"Given that {args.event_text}, considering all possible economic "
         f"effects: does {{commodity_info}} experience {{factor_info}}?"),
        **{k: v for k, v in join_kwargs.items() if v is not None},
    )
    elapsed = round(time.time() - t0, 2)

    if isinstance(output, tuple):
        result, join_stats = output
    else:
        result, join_stats = output, {}
    result = result.reset_index(drop=True)
    usage = lotus.settings.lm.stats.physical_usage
    tokens = int(getattr(usage, "total_tokens", 0))
    cost   = float(getattr(usage, "total_cost", 0.0))

    if cost == 0.0 and tokens > 0:
        prompt_t = int(getattr(usage, "prompt_tokens", 0))
        compl_t  = int(getattr(usage, "completion_tokens", 0))
        rates = PRICING.get(args.model_name, {"input": 0.0, "output": 0.0})
        cost = (prompt_t * rates["input"] + compl_t * rates["output"]) / 1e6

    llm_calls_total  = total_pairs
    llm_calls_to_lm  = join_stats.get("join_resolved_by_large_model", total_pairs)
    llm_calls_saved  = join_stats.get("join_resolved_by_helper_model", 0)
    llm_calibration  = join_stats.get("optimized_join_cost", 0)

    res_pairs = set(zip(result["commodity_name"].astype(str),
                        result["factor_id"].astype(str)))
    tp        = len(res_pairs & gt_pairs)
    precision = round(tp / len(res_pairs), 4) if res_pairs else 0.0
    recall    = round(tp / len(gt_pairs),  4) if gt_pairs  else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    flags = []
    if args.max_cost is not None and cost > args.max_cost:
        flags.append(f"OVER_BUDGET (${cost:.4f} > ${args.max_cost:.4f})")
    if args.f1_floor is not None and f1 < args.f1_floor:
        flags.append(f"BELOW_TARGET (F1={f1:.4f} < {args.f1_floor:.4f})")

    status = "success" if not flags else "warning"
    if flags:
        for f_ in flags:
            print(f"[lotus] WARNING: {f_}")

    out_dir = "my_benchmark/econ/results/lotus/Q1"
    os.makedirs(out_dir, exist_ok=True)
    result[["commodity_name", "factor_id"]].to_csv(
        f"{out_dir}/{args.test_id}.csv", index=False)

    metrics = {
        "test_id":       args.test_id,
        "query":         "Q1",
        "event_id":      args.event_id,
        "system":        "lotus",
        "model_name":    args.model_name,
        "policy":        mode,
        "reasoning_effort": None,
        "constraints": {
            "recall_target":    None if args.no_cascade else args.recall_target,
            "precision_target": None if args.no_cascade else args.precision_target,
            "max_cost_usd":     args.max_cost,
            "f1_floor":         args.f1_floor,
        },
        "execution_time":    elapsed,
        "tokens":            tokens,
        "cost_usd":          round(cost, 6),
        "llm_calls_total":   llm_calls_total,
        "llm_calls_saved":   llm_calls_saved,
        "precision":         precision,
        "recall":            recall,
        "f1_score":          f1,
        "row_count":         len(result),
        "status":            status,
        "flags":             flags,
    }
    with open(f"{out_dir}/{args.test_id}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[lotus] elapsed={elapsed}s  tokens={tokens}  cost=${cost:.4f}")
    print(f"[lotus] llm_calls={llm_calls_to_lm} + saved={llm_calls_saved} "
          f"+ calibration={llm_calibration}")
    print(f"[lotus] P={precision}  R={recall}  F1={f1}  rows={len(result)}")
    print(f"[lotus] saved -> {out_dir}/{args.test_id}_metrics.json")


if __name__ == "__main__":
    main()
