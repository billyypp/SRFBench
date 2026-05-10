
from dotenv import load_dotenv
load_dotenv(override=True)

import argparse, json, os, sys, time, logging, pathlib

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
import litellm; litellm.suppress_debug_info = True; litellm.drop_params = True

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _pz_patch  # noqa: F401
import _mab_patch  # noqa: F401  — force-sample all operators before MAB pruning

import pandas as pd
import palimpzest as pz
from palimpzest.constants import Model


def parse_args():
    p = argparse.ArgumentParser(description="Q1 Palimpzest econ benchmark")
    p.add_argument("--event_id",  default="E08")
    p.add_argument("--event_text", default="The war in Ukraine")
    p.add_argument("--data_dir",  default="my_benchmark/econ/data")
    p.add_argument("--gt_dir",    default="my_benchmark/econ/ground_truth")
    # ── optimization mode (pick ONE) ─────────────────────────────────────
    p.add_argument("--max_cost",    type=float, default=None,
                   help="MaxQualityAtFixedCost: USD budget for the query")
    p.add_argument("--min_quality", type=float, default=None,
                   help="MinCostAtFixedQuality: minimum quality score 0-1")
    p.add_argument("--policy", default=None, choices=["MaxQuality", "MinCost"],
                   help="Unconstrained policy (used if --max_cost and --min_quality are both unset)")
    # ── model / execution config ─────────────────────────────────────────
    p.add_argument("--models", nargs="+", default=["GPT_5_4"],
                   help="Model names (attributes on palimpzest.constants.Model)")
    p.add_argument("--max_workers", type=int, default=20)
    p.add_argument("--sample_budget", type=int, default=10,
                   help="Max operator-input pairs to sample during optimization "
                        "(default: 10 for test data, use 100+ for full data)")
    p.add_argument("--reasoning_effort", default="low",
                   choices=["minimal", "low", "medium", "high"])
    p.add_argument("--f1_floor", type=float, default=None,
                   help="[post-hoc] Flag BELOW_TARGET if F1 falls below threshold")
    p.add_argument("--test_id", default="t1")
    return p.parse_args()


def _sum_tokens(stats) -> int:
    total = 0
    for attr in ("input_text_tokens", "input_audio_tokens",
                 "input_image_tokens", "output_text_tokens",
                 "cache_read_tokens", "cache_creation_tokens",
                 "embedding_input_tokens"):
        total += int(getattr(stats, attr, 0))
    return total


def _extract_llm_calls(stats):
    if not hasattr(stats, "plan_stats") or not stats.plan_stats:
        return None
    total = 0
    found_any = False
    for ps in stats.plan_stats.values():
        for os_ in getattr(ps, "operator_stats", {}).values():
            for rs in getattr(os_, "record_op_stats_lst", []):
                val = getattr(rs, "total_llm_calls", None)
                if val is not None:
                    total += int(val)
                    found_any = True
    return total if found_any else None


def main():
    args = parse_args()

    # ── load data ────────────────────────────────────────────────────────
    comm_path = os.path.join(args.data_dir, "commodities.csv")
    if not os.path.exists(comm_path):
        comm_path = os.path.join(args.data_dir, "commodities (2).csv")
    commodities = pd.read_csv(comm_path)
    factors     = pd.read_csv(os.path.join(args.data_dir, "factors.csv"))
    commodities["commodity_info"] = (commodities["commodity_name"].astype(str)
                                     + " (" + commodities["category"].astype(str) + ")")
    factors["factor_info"] = ("[" + factors["side"].astype(str) + "] "
                              + factors["factor_description"].astype(str))

    gt = pd.read_csv(os.path.join(args.gt_dir, f"Q1_{args.event_id}.csv"))
    gt_pairs = set(zip(gt["commodity_name"].astype(str),
                       gt["factor_id"].astype(str)))
    total_pairs = len(commodities) * len(factors)
    print(f"[pz] event={args.event_id}  commodities={len(commodities)}  "
          f"factors={len(factors)}  pairs={total_pairs}  gt={len(gt_pairs)}")

    # ── build query plan ─────────────────────────────────────────────────
    left  = pz.MemoryDataset(id="commodities", vals=commodities)
    right_df = factors.rename(columns={c: f"{c}_right" for c in factors.columns})
    right = pz.MemoryDataset(id="factors", vals=right_df)

    condition = (f"Given that '{args.event_text}', considering all possible "
                 f"economic effects, does the commodity described in the left record "
                 f"experience the economic factor described in the right record?")
    joined = left.sem_join(right, condition=condition,
                           depends_on=["commodity_info", "factor_info_right"])
    joined = joined.project(["commodity_name", "factor_id_right"])

    # ── resolve policy ───────────────────────────────────────────────────
    if args.max_cost is not None:
        policy = pz.MaxQualityAtFixedCost(max_cost=args.max_cost)
        policy_label = f"MaxQualityAtFixedCost(${args.max_cost})"
    elif args.min_quality is not None:
        policy = pz.MinCostAtFixedQuality(min_quality=args.min_quality)
        policy_label = f"MinCostAtFixedQuality({args.min_quality})"
    elif args.policy == "MinCost":
        policy = pz.MinCost()
        policy_label = "MinCost"
    else:
        policy = pz.MaxQuality()
        policy_label = "MaxQuality"

    # ── resolve models ───────────────────────────────────────────────────
    models = [getattr(Model, m) for m in args.models]

    cfg = pz.QueryProcessorConfig(
        policy=policy,
        execution_strategy="parallel",
        max_workers=args.max_workers,
        join_parallelism=args.max_workers,
        verbose=False,
        progress=True,
        available_models=models,
        reasoning_effort=args.reasoning_effort,
        sample_budget=args.sample_budget,
        k=len(models) * 2,  # ensure all models fit in frontier (2 join types per model)
    )

    print(f"[pz] policy={policy_label}  models={[m.value for m in models]}")

    # ── validator uses gpt-5.4 as judge (same model that generated GT) ────
    validator = pz.Validator(model=Model.GPT_5_4)

    t0 = time.time()
    if args.max_cost is not None or args.min_quality is not None:
        out = joined.optimize_and_run(config=cfg, validator=validator)
    else:
        out = joined.run(cfg)
    elapsed = round(time.time() - t0, 2)

    df = out.to_df() if not isinstance(out, pd.DataFrame) else out
    stats = out.execution_stats
    tokens = _sum_tokens(stats)
    cost   = float(getattr(stats, "total_execution_cost", 0.0))
    llm_calls = _extract_llm_calls(stats)

    # ── score against ground truth ───────────────────────────────────────
    col_right = "factor_id_right" if "factor_id_right" in df.columns else "factor_id"
    pairs = set(zip(df["commodity_name"].astype(str), df[col_right].astype(str)))
    tp = len(pairs & gt_pairs)
    precision = round(tp / len(pairs), 4) if pairs else 0.0
    recall    = round(tp / len(gt_pairs), 4) if gt_pairs else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    # ── post-hoc flags ───────────────────────────────────────────────────
    flags = []
    if args.f1_floor is not None and f1 < args.f1_floor:
        flags.append(f"BELOW_TARGET (F1={f1:.4f} < {args.f1_floor:.4f})")
    status = "success" if not flags else "warning"

    # ── save ─────────────────────────────────────────────────────────────
    out_dir = "my_benchmark/econ/results/palimpzest/Q1"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/{args.test_id}.csv", index=False)

    metrics = {
        "test_id":       args.test_id,
        "query":         "Q1",
        "event_id":      args.event_id,
        "system":        "palimpzest",
        "model_name":    ", ".join(m.value for m in models),
        "policy":        policy_label,
        "reasoning_effort": args.reasoning_effort,
        "constraints": {
            "max_cost_usd":  args.max_cost,
            "min_quality":   args.min_quality,
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
        "row_count":         len(df),
        "status":            status,
        "flags":             flags,
    }
    with open(f"{out_dir}/{args.test_id}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[pz] elapsed={elapsed}s  tokens={tokens}  cost=${cost:.4f}  "
          f"llm_calls={llm_calls}")
    print(f"[pz] P={precision}  R={recall}  F1={f1}  rows={len(df)}")
    print(f"[pz] saved -> {out_dir}/{args.test_id}_metrics.json")


if __name__ == "__main__":
    main()
