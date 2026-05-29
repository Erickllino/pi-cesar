import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


HF_API_KEY = os.getenv("HF_API_KEY")

PROJECT_ROOT = Path(__file__).parent.parent.parent
ORIGINAL_DIR = PROJECT_ROOT / "Data" / "Hackaprompt" / "Original"
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("hf://datasets/hackaprompt/hackaprompt-dataset/hackaprompt.parquet")

print(df.head())

df.to_parquet(ORIGINAL_DIR / "hackaprompt.parquet", index=False)