
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse, json, os, sys, time, logging, pathlib


logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _pricing import PRICING

import pandas as pd
from tdb.data.relational import Database
from tdb.execution.engine import ExecutionEngine
from tdb.execution.constraints import Constraints

def parse_args():
    p = argparse.ArgumentParser(description="Q1 ThalamusDB econ benchmark")
    p.add_argument("--event_id",   default="E08")
    p.add_argument("--event_text", default="The war in Ukraine")
    p.add_argument("--data_dir",   default="my_benchmark/econ/data",
                   help="Directory containing econ.duckdb (or commodities/factors CSVs)")
    p.add_argument("--gt_dir",     default="my_benchmark/econ/ground_truth",
                   help="Directory containing Q1_<event_id>.csv ground truth")
    p.add_argument("--model_config", default="gpt_5_4",
                   help="Config file basename in config/system/thalamusdb/ "
                        "(without .json)")
    p.add_argument("--dop", type=int, default=20,
                   help="Degree of parallelism")
   
    p.add_argument("--max_calls",   type=int,   default=10_000_000)
    p.add_argument("--max_seconds", type=int,   default=3600)
    p.add_argument("--max_tokens",  type=int,   default=10_000_000)
    p.add_argument("--max_error",   type=float, default=0.2,
                   help="Bound-width tolerance (u-l)/(u+l).  Lower = more "
                        "LLM calls but higher confidence.  NOT distance-from-"
                        "truth.  Default: 0.2")
    
    p.add_argument("--max_cost",    type=float, default=None,
                   help="[post-hoc] Flag OVER_BUDGET if cost exceeds this USD amount")
    p.add_argument("--f1_floor",    type=float, default=None,
                   help="[post-hoc] Flag BELOW_TARGET if F1 falls below threshold")
    p.add_argument("--test_id",     default="t1")
    return p.parse_args()



def main():
    args = parse_args()

    
    gt = pd.read_csv(os.path.join(args.gt_dir, f"Q1_{args.event_id}.csv"))
    gt_pairs = set(zip(gt["commodity_name"].astype(str),
                       gt["factor_id"].astype(str)))

    
    db_path = os.path.join(args.data_dir, "econ.duckdb")
    db = Database(db_path)
    cfg_path = os.path.abspath(
        f"config/system/thalamusdb/{args.model_config}.json")
    engine = ExecutionEngine(db, dop=args.dop, model_config_path=cfg_path)

    constraints = Constraints(
        max_calls=args.max_calls,
        max_seconds=args.max_seconds,
        max_tokens=args.max_tokens,
        max_error=args.max_error,
    )

    sql = (
        "SELECT C.commodity_name, F.factor_id "
        "FROM Commodities AS C, Factors AS F "
        "WHERE NLjoin(C.commodity_info, F.factor_info, "
        f"'Given that {args.event_text}, considering all possible economic "
        "effects, the commodity experiences the factor')"
    )

    print(f"[tdb] event={args.event_id}  config={args.model_config}  dop={args.dop}")
    print(f"[tdb] native constraints: max_calls={args.max_calls}  "
          f"max_tokens={args.max_tokens}  max_error={args.max_error}  "
          f"max_seconds={args.max_seconds}")
    print(f"[tdb] post-hoc labels: max_cost=${args.max_cost}  f1_floor={args.f1_floor}")

    
    t0 = time.time()
    query = Query(db, sql)
    result, costs = engine.run(query, constraints)
    elapsed = round(time.time() - t0, 2)

    if not isinstance(result, pd.DataFrame):
        result = result.df() if hasattr(result, "df") else pd.DataFrame(result)

   
    llm_calls = int(costs.total_LLM_calls()) if hasattr(costs, "total_LLM_calls") else None
    tokens, cost = 0, 0.0
    for model_name, counters in costs.model2counters.items():
        it  = getattr(counters, "input_tokens", 0)
        at  = getattr(counters, "audio_input_tokens", 0)
        ot  = getattr(counters, "output_tokens", 0)
        nat = max(0, it - at)
        rates = PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        cost   += nat * rates["input"] / 1e6 + ot * rates["output"] / 1e6
        tokens += it + ot

   
    pairs = set(zip(result["commodity_name"].astype(str),
                    result["factor_id"].astype(str)))
    tp        = len(pairs & gt_pairs)
    precision = round(tp / len(pairs), 4) if pairs else 0.0
    recall    = round(tp / len(gt_pairs), 4) if gt_pairs else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    
    flags = []
    if args.max_cost is not None and cost > args.max_cost:
        flags.append(f"OVER_BUDGET (${cost:.4f} > ${args.max_cost:.4f})")
    if args.f1_floor is not None and f1 < args.f1_floor:
        flags.append(f"BELOW_TARGET (F1={f1:.4f} < {args.f1_floor:.4f})")

    status = "success" if not flags else "warning"
    if flags:
        for f_ in flags:
            print(f"[tdb] WARNING: {f_}")

   
    out_dir = "my_benchmark/econ/results/thalamusdb/Q1"
    os.makedirs(out_dir, exist_ok=True)
    result.to_csv(f"{out_dir}/{args.test_id}.csv", index=False)

    metrics = {
        "test_id":       args.test_id,
        "query":         "Q1",
        "event_id":      args.event_id,
        "system":        "thalamusdb",
        "model_name":    args.model_config,
        "policy":        f"max_error={args.max_error}",
        "reasoning_effort": None,
        "constraints": {
            "max_calls":     args.max_calls,
            "max_tokens":    args.max_tokens,
            "max_seconds":   args.max_seconds,
            "max_error":     args.max_error,
            "max_cost_usd":  args.max_cost,
            "f1_floor":      args.f1_floor,
        },
        "execution_time":    elapsed,
        "tokens":            tokens,
        "cost_usd":          round(cost, 6),
        "llm_calls_total":   llm_calls,
        "llm_calls_saved":   None,
        "precision":         precision,
        "recall":            recall,
        "f1_score":          f1,
        "row_count":         len(result),
        "status":            status,
        "flags":             flags,
    }
    with open(f"{out_dir}/{args.test_id}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[tdb] elapsed={elapsed}s  tokens={tokens}  cost=${cost:.4f}  "
          f"llm_calls={llm_calls}")
    print(f"[tdb] P={precision}  R={recall}  F1={f1}  rows={len(result)}")
    print(f"[tdb] saved -> {out_dir}/{args.test_id}_metrics.json")


if __name__ == "__main__":
    main()
