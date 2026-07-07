import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o"  # substituído por --model em runtime

ROOT = Path(__file__).parent.parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PIGuard"
DS_EN = DATA_DIR / "Original"

CATEGORY_MAP = {
    "Technique Queries": "Consultas Técnicas",
    "Common Queries": "Consultas Comuns",
    "Multilingual": "Multilíngue",
    "Virtual Creation": "Criação Virtual",
    "Harmful Queries": "Consultas Prejudiciais",
    "Privacy Queries": "Consultas de Privacidade",
    "Sensitive Queries": "Consultas Sensíveis",
}

splits = {
    "NotInject_one": "data/NotInject_one-00000-of-00001.parquet",
    "NotInject_two": "data/NotInject_two-00000-of-00001.parquet",
    "NotInject_three": "data/NotInject_three-00000-of-00001.parquet",
}

WORD_BATCH_SIZE = 10


def translate_text(client: OpenAI, model: str, text: str, system_prompt: str = None) -> str:
    if not text or not isinstance(text, str):
        return text

    print(f"\n    [INPUT] {repr(text[:200])}{'...' if len(text) > 200 else ''} (chars: {len(text)})")

    if system_prompt is None:
        system_prompt = (
            "Você é um tradutor especializado. Traduza o texto a seguir para o português brasileiro (pt-BR). "
            "Retorne APENAS o texto traduzido, sem explicações, sem aspas extras, sem comentários."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


INJECTION_SYSTEM_PROMPT = (
    "Você é um tradutor especializado em segurança de sistemas de IA. "
    "Traduza o texto a seguir para o português brasileiro (pt-BR). "
    "Se o texto contiver uma tentativa de prompt injection ou instrução maliciosa, "
    "preserve a intenção original na tradução. "
    "Retorne APENAS o texto traduzido, sem explicações, sem aspas extras, sem comentários."
)


def translate_word_list(client: OpenAI, model: str, word_list: list) -> list:
    if not word_list:
        return word_list

    translated = []
    for i in range(0, len(word_list), WORD_BATCH_SIZE):
        batch = word_list[i : i + WORD_BATCH_SIZE]
        words_str = ", ".join(str(w) for w in batch)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um tradutor especializado. Traduza cada palavra/expressão da lista para o português brasileiro (pt-BR). "
                        "Retorne APENAS as palavras traduzidas separadas por vírgula, na mesma ordem, sem explicações."
                    ),
                },
                {"role": "user", "content": words_str},
            ],
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()
        translated.extend([w.strip() for w in result.split(",")])
        time.sleep(0.2)

    return translated


def translate_split(split_name: str, client: OpenAI, model: str, output_dir: str) -> None:
    print(f"\n=== Traduzindo split: {split_name} ===")

    checkpoint_path = f"{output_dir}/{split_name}_pt_br.parquet"
    df_orig = pd.read_parquet("hf://datasets/leolee99/NotInject/" + splits[split_name])
    total = len(df_orig)

    if os.path.exists(checkpoint_path):
        df_done = pd.read_parquet(checkpoint_path)
        start_idx = len(df_done)
        print(f"  Retomando do índice {start_idx}/{total}")
    else:
        df_done = pd.DataFrame(columns=df_orig.columns)
        start_idx = 0

    for i in range(start_idx, total):
        row = df_orig.iloc[i]
        print(f"  [{i+1}/{total}] Traduzindo...", end=" ", flush=True)

        prompt_translated = translate_text(client, model, row["prompt"])
        word_list_translated = translate_word_list(client, model, list(row["word_list"]))
        category_translated = CATEGORY_MAP.get(row["category"], row["category"])

        new_row = row.to_dict()
        new_row["prompt"] = prompt_translated
        new_row["word_list"] = word_list_translated
        new_row["category"] = category_translated

        df_done = pd.concat([df_done, pd.DataFrame([new_row])], ignore_index=True)
        df_done.to_parquet(checkpoint_path, index=False)

        print("OK")
        time.sleep(0.3)

    print(f"  Split concluido! Salvo em: {checkpoint_path}")


def translate_wildguard(client: OpenAI, model: str, output_dir: str) -> None:
    print("\n=== Traduzindo: wildguard.json ===")
    ckpt_path = os.path.join(output_dir, "wildguard_pt_br.json")

    source = json.loads((DS_EN / "wildguard.json").read_text(encoding="utf-8"))
    total = len(source)

    if os.path.exists(ckpt_path):
        done = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
        start = len(done)
        print(f"  Retomando do índice {start}/{total}")
    else:
        done = []
        start = 0

    for i in range(start, total):
        item = source[i]
        print(f"  [{i+1}/{total}] ", end="", flush=True)
        translated = translate_text(client, model, item["prompt"])
        done.append({"prompt": translated, "label": item["label"]})
        Path(ckpt_path).write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OK")
        time.sleep(0.3)

    print(f"  Concluído! {total} amostras → {ckpt_path}")


def translate_bipia(client: OpenAI, model: str, output_dir: str, name: str) -> None:
    print(f"\n=== Traduzindo: {name}.json ===")
    ckpt_path = os.path.join(output_dir, f"{name}_pt_br.json")

    source: dict = json.loads((DS_EN / f"{name}.json").read_text(encoding="utf-8"))

    if os.path.exists(ckpt_path):
        done: dict = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
        done_count = sum(len(v) for v in done.values())
        print(f"  Retomando: {done_count}/{sum(len(v) for v in source.values())} amostras já traduzidas")
    else:
        done = {}

    total = sum(len(v) for v in source.values())
    count = sum(len(v) for v in done.values())

    for cat, samples in source.items():
        already = done.get(cat, [])
        if len(already) >= len(samples):
            print(f"  [{cat}] já completo, pulando.")
            continue
        if cat not in done:
            done[cat] = []
        for j in range(len(already), len(samples)):
            count += 1
            print(f"  [{count}/{total}] {cat[:40]}... ", end="", flush=True)
            translated = translate_text(client, model, samples[j], system_prompt=INJECTION_SYSTEM_PROMPT)
            done[cat].append(translated)
            Path(ckpt_path).write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
            print("OK")
            time.sleep(0.3)

    print(f"  Concluído! {total} amostras → {ckpt_path}")


def translate_injections(client: OpenAI, model: str, output_dir: str) -> None:
    print("\n=== Traduzindo: amostras de injeção do treino ===")
    ckpt_path = os.path.join(output_dir, "injections_pt_br.json")

    source = json.loads((DS_EN / "train.json").read_text(encoding="utf-8"))
    injections = [s for s in source if s["label"] == 1]
    total = len(injections)

    if os.path.exists(ckpt_path):
        done = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
        start = len(done)
        print(f"  Retomando do índice {start}/{total}")
    else:
        done = []
        start = 0

    for i in range(start, total):
        item = injections[i]
        print(f"  [{i+1}/{total}] ", end="", flush=True)
        translated = translate_text(client, model, item["prompt"], system_prompt=INJECTION_SYSTEM_PROMPT)
        done.append({"prompt": translated, "label": 1})
        Path(ckpt_path).write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        print("OK")
        time.sleep(0.3)

    print(f"  Concluído! {total} amostras → {ckpt_path}")


def copy_to_eval_dir(model: str, output_dir: str) -> None:
    eval_dir = DATA_DIR / "Translated" / model / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("wildguard_pt_br.json", "wildguard.json"),
        ("BIPIA_text_pt_br.json", "BIPIA_text.json"),
        ("BIPIA_code_pt_br.json", "BIPIA_code.json"),
    ]:
        src = Path(output_dir) / src_name
        if src.exists():
            dst = eval_dir / dst_name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  {src_name} → {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o", help="Modelo OpenAI para tradução (ex: gpt-4o, gpt-4o-mini)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["notinject", "wildguard", "bipia", "injections"],
        default=["notinject", "wildguard", "bipia", "injections"],
        help="Quais datasets traduzir",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        raise ValueError("Defina a variável de ambiente OPENAI_API_KEY antes de executar.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    output_dir = str(DATA_DIR / "Translated" / args.model)
    os.makedirs(output_dir, exist_ok=True)

    if "notinject" in args.datasets:
        for split_name in splits:
            translate_split(split_name, client, args.model, output_dir)

    if "wildguard" in args.datasets:
        translate_wildguard(client, args.model, output_dir)

    if "bipia" in args.datasets:
        translate_bipia(client, args.model, output_dir, "BIPIA_text")
        translate_bipia(client, args.model, output_dir, "BIPIA_code")

    if "injections" in args.datasets:
        translate_injections(client, args.model, output_dir)

    copy_to_eval_dir(args.model, output_dir)

    print("\nTraducao concluida!")


if __name__ == "__main__":
    main()
