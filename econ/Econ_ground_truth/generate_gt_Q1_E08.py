"""
Run once to generate ground truth for Q1 (E08).
Uses gpt-4o as oracle. Saves to ground_truth/Q1_E08.csv
"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import lotus
from lotus.models import LM
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT      = Path(__file__).parent / "Q1_E08.csv"

lm = LM("gpt-4o", max_tokens=512, max_batch_size=5)
lotus.settings.configure(lm=lm)

commodities = pd.read_csv(DATA_DIR / "commodities (2).csv")
factors     = pd.read_csv(DATA_DIR / "factors.csv")
events      = pd.read_csv(DATA_DIR / "events.csv")

event_desc = events[events["event_id"] == "E08"].iloc[0]["event_description"]
print(f"Generating ground truth for E08...")
print(f"{len(commodities)} x {len(factors)} = {len(commodities)*len(factors)} pairs\n")

result = commodities.sem_join(
    factors,
    f'Given the following event: "{event_desc}" — '
    'does {commodity_name} ({description}) directly experience {factor_description} as a result?'
)

out = result[["commodity_name", "factor_id"]]
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} pairs -> {OUT}")
