"""Download the source PIArena splits into the local ignored data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Data" / "PIArena" / "Original"
REPOSITORY = "sleeepeer/PIArena"
DATASETS = (
    "dolly_closed_qa",
    "dolly_information_extraction",
    "dolly_summarization",
    "gov_report_long",
    "hotpotqa_long",
    "hotpotqa_rag",
    "hotpotqa_rag_knowledge_corruption",
    "lcc_long",
    "msmarco_rag",
    "msmarco_rag_knowledge_corruption",
    "multi_news_long",
    "nq_rag",
    "nq_rag_knowledge_corruption",
    "passage_retrieval_en_long",
    "qasper_long",
    "squad_v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baixa os datasets-fonte do PIArena.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--revision", default="main", help="Revision ou commit do dataset no Hugging Face")
    parser.add_argument("--restart", action="store_true", help="Substitui arquivos locais ja existentes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        destination = OUTPUT_DIR / f"{name}.json"
        if destination.exists() and not args.restart:
            print(f"[{name}] ja existe; use --restart para baixar novamente")
            continue

        dataset = load_dataset(REPOSITORY, split=name, revision=args.revision)
        destination.write_text(
            json.dumps(dataset.to_list(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{name}] salvo: {len(dataset)} exemplos")


if __name__ == "__main__":
    main()
