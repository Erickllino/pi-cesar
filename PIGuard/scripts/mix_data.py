"""
Une os dados ORIGINAIS (en) com os dados TRADUZIDOS (pt-BR, e futuramente
outros idiomas) do PIGuard, gerando um dataset bilíngue por split em
Data/PIGuard/Mixed/{model}/.

Cada split combinado recebe um campo "lang" em cada registro (en, pt_br, ...)
para permitir filtrar por idioma depois, se necessário.

Uso:
  python mix_data.py --model gpt-4o
  python mix_data.py --model gpt-4o --langs pt_br es
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "Data" / "PIGuard"
ORIGINAL_DIR = DATA_DIR / "Original"
TRANSLATED_DIR = DATA_DIR / "Translated"
MIXED_DIR = DATA_DIR / "Mixed"

# Splits em formato lista de registros (prompt/label/...), disponíveis como
# .json (original e a maioria dos modelos) ou .parquet (NotInject em alguns modelos).
LIST_SPLITS = ["train", "valid", "wildguard", "NotInject_one", "NotInject_two", "NotInject_three"]

# Splits em formato dict {categoria: [prompt, ...]} (injeções BIPIA).
DICT_SPLITS = ["BIPIA_code", "BIPIA_text"]

# Splits que só existem traduzidos (não há original en salvo localmente).
TRANSLATED_ONLY_SPLITS = ["injections"]


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return json.loads(path.read_text(encoding="utf-8"))


def tag_lang(records: list[dict], lang: str) -> list[dict]:
    return [{**r, "lang": lang} for r in records]


def find_translated_list_path(model: str, split: str, lang: str) -> Path | None:
    candidates = [
        TRANSLATED_DIR / model / f"{split}_{lang}.json",
        TRANSLATED_DIR / model / f"{split}_{lang}.parquet",
    ]
    if lang == "pt_br":
        # Alguns runs (ex.: gpt-4o) salvaram o NotInject pt-BR já sobrescrevendo
        # o nome do original, sem o sufixo "_pt_br".
        candidates.append(TRANSLATED_DIR / model / f"{split}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def mix_list_split(model: str, split: str, langs: list[str]) -> list[dict]:
    mixed: list[dict] = []

    orig_path = ORIGINAL_DIR / f"{split}.json"
    if orig_path.exists():
        mixed += tag_lang(load_records(orig_path), "en")
    else:
        print(f"  [aviso] original não encontrado: {orig_path}")

    for lang in langs:
        path = find_translated_list_path(model, split, lang)
        if path is None:
            print(f"  [aviso] {split} ({lang}) não encontrado para o modelo {model}, pulando.")
            continue
        mixed += tag_lang(load_records(path), lang)

    return mixed


def mix_dict_split(model: str, split: str, langs: list[str]) -> list[dict]:
    mixed: list[dict] = []

    def flatten(data: dict, lang: str) -> list[dict]:
        return [
            {"category": category, "prompt": prompt, "lang": lang}
            for category, prompts in data.items()
            for prompt in prompts
        ]

    orig_path = ORIGINAL_DIR / f"{split}.json"
    if orig_path.exists():
        mixed += flatten(json.loads(orig_path.read_text(encoding="utf-8")), "en")
    else:
        print(f"  [aviso] original não encontrado: {orig_path}")

    for lang in langs:
        path = TRANSLATED_DIR / model / f"{split}_{lang}.json"
        if not path.exists():
            print(f"  [aviso] {split} ({lang}) não encontrado para o modelo {model}, pulando.")
            continue
        mixed += flatten(json.loads(path.read_text(encoding="utf-8")), lang)

    return mixed


def mix_translated_only_split(model: str, split: str, langs: list[str]) -> list[dict]:
    mixed: list[dict] = []
    for lang in langs:
        path = TRANSLATED_DIR / model / f"{split}_{lang}.json"
        if not path.exists():
            print(f"  [aviso] {split} ({lang}) não encontrado para o modelo {model}, pulando.")
            continue
        mixed += tag_lang(load_records(path), lang)
    return mixed


def save_json(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter

    by_lang = Counter(r["lang"] for r in records)
    print(f"  Salvo: {path} | {len(records)} amostras | {dict(by_lang)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Une dados originais (en) e traduzidos do PIGuard por split.")
    parser.add_argument("--model", default="gpt-4o", help="Modelo de tradução (subpasta em Data/PIGuard/Translated/)")
    parser.add_argument("--langs", nargs="+", default=["pt_br"], help="Idiomas traduzidos a incluir (default: pt_br)")
    args = parser.parse_args()

    out_dir = MIXED_DIR / args.model

    print(f"\nModelo: {args.model} | Idiomas: {args.langs}")

    print("\n=== Splits em lista (valid, wildguard, NotInject) ===")
    for split in LIST_SPLITS:
        mixed = mix_list_split(args.model, split, args.langs)
        save_json(mixed, out_dir / f"{split}.json")

    print("\n=== Splits em dicionário por categoria (BIPIA) ===")
    for split in DICT_SPLITS:
        mixed = mix_dict_split(args.model, split, args.langs)
        save_json(mixed, out_dir / f"{split}.json")

    print("\n=== Splits só traduzidos (sem original local) ===")
    for split in TRANSLATED_ONLY_SPLITS:
        mixed = mix_translated_only_split(args.model, split, args.langs)
        save_json(mixed, out_dir / f"{split}.json")

    print(f"\nPronto! Datasets mistos em: {out_dir}")


if __name__ == "__main__":
    main()
