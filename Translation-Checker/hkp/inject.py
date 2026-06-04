import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).parent.parent
PROJECT_ROOT = ROOT.parent
MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"

DEFAULT_CONCURRENCY = 16   # requisicoes simultaneas ao vLLM (o servidor batcheia sozinho)
DEFAULT_CHUNK_SIZE = 200   # linhas processadas antes de cada checkpoint (salvar + retomada)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_env_file(PROJECT_ROOT / ".env")

HOST = "127.0.0.1"
PORT = 8000
VLLM_BASE_URL = f"http://{HOST}:{PORT}/v1"
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY").strip() or "EMPTY"
INPUT_PATH = PROJECT_ROOT / "Data" / "Hackaprompt" / "Original" / "hackaprompt_pt_br_Qwen__Qwen2.5-72B-Instruct-AWQ.parquet"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "Hackaprompt" / "Translated"



def inject(client: OpenAI, model: str, user_input: str, sys_prompt: str ) -> str:
    """Executa o ataque: roda sys_prompt + user_input no modelo e devolve a resposta.

    Função pura (sem prints) para rodar em threads. Usa temperature=0 para
    tornar a avaliação determinística e reproduzível.
    """
    if not user_input or not isinstance(user_input, str):
        return user_input
    
    if not sys_prompt or not isinstance(sys_prompt, str):
        sys_prompt = ""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def inject_row(client: OpenAI, model: str, idx: int, row: pd.Series) -> dict:
    """Executa a injeção de uma linha. Captura erros para não derrubar o run inteiro."""
    result = {"index": idx, "row": row, "error": None}
    try:
        result["completion_tr"] = inject(client, model, row["user_input"], row["sys_prompt"])
    except Exception as exc:  # noqa: BLE001
        # Em caso de falha persistente, registra o erro e deixa a resposta como None
        # (não inventa conteúdo) para refazer/excluir depois.
        result["error"] = str(exc)
        result["completion_tr"] = None
    return result


def check_injection(completion: str, completion_tr: str) -> bool:
    """True se a diferença absoluta de tamanho entre `completion` e `completion_tr`
    for grande demais (sinaliza possível divergência relevante na resposta)."""
    if not completion or not completion_tr or not isinstance(completion, str) or not isinstance(completion_tr, str):
        return False
    a, b = len(completion), len(completion_tr)
    return abs(a - b) > 0.5 * a


def _preview(s, n: int = 160) -> str:
    if not isinstance(s, str):
        return str(s)
    s = s.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def inject_dataset(client: OpenAI, model: str, output_path: Path,
                      limit: int | None, concurrency: int, chunk_size: int) -> None:
    print(f"\n=== Executando injeção: {INPUT_PATH.name} ===")
    df_orig = pd.read_parquet(INPUT_PATH)
    if limit is not None:
        df_orig = df_orig.head(limit).reset_index(drop=True)
    total = len(df_orig)
    print(f"  Total de amostras: {total}")
    print(f"  Concorrência: {concurrency} req simultâneas | checkpoint a cada {chunk_size} linhas")

    warnings_path = OUTPUT_DIR / "translation_warnings.parquet"
    # Preserva warnings de execuções anteriores (importante na retomada)
    warning_records = pd.read_parquet(warnings_path).to_dict("records") if warnings_path.exists() else []

    if output_path.exists():
        df_done = pd.read_parquet(output_path)
        start_idx = len(df_done)
        print(f"  Retomando do índice {start_idx}/{total}")
    else:
        df_done = pd.DataFrame(columns=df_orig.columns.tolist() + ["completion_tr"])  # colunas originais + tradução
        start_idx = 0

    if start_idx >= total:
        print("  Nada a fazer — já está completo.")
        return

    # Um único pool reaproveitado por todos os chunks
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for chunk_start in range(start_idx, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            indices = list(range(chunk_start, chunk_end))

            # 1) dispara o chunk em paralelo — o vLLM batcheia do lado servidor
            futures = [pool.submit(inject_row, client, model, i, df_orig.iloc[i]) for i in indices]
            results = {f.result()["index"]: f.result() for f in futures}

            # 2) grava em ORDEM (mantém a retomada por contagem válida)
            new_rows = []
            for i in indices:
                r = results[i]
                row = r["row"]
                label = row["correct"]
                pct = (i + 1) / total * 100

                if r["error"]:
                    print(f"  [{pct:5.1f}% | {i + 1}/{total}] label={label}  [ERRO] {r['error'][:120]}")
                    warning_records.append({"index": i, "label": label,
                                            "original": row["prompt"], "translated": None, "type": "error"})
                else:
                    print(f"  [{pct:5.1f}% | {i + 1}/{total}] label={label}")
                    print(f"      in : {_preview(row['user_input'])}")
                    print(f"      out: {_preview(r['completion_tr'])}")
                    # Flag de tamanho: diferença absoluta grande entre completion e completion_tr
                    if check_injection(row["completion"], r["completion_tr"]):
                        a, b = len(row["completion"]), len(r["completion_tr"])
                        print(f"      [AVISO] diferença de tamanho grande (completion={a} vs completion_tr={b})")
                        warning_records.append({"index": i, "label": label,
                                                "original": row["completion"],
                                                "translated": r["completion_tr"], "type": "length"})

                new_row = row.to_dict()
                new_row["completion_tr"] = r.get("completion_tr")  # None em caso de erro
                new_rows.append(new_row)

            # 3) checkpoint: anexa o chunk e salva (resume parte daqui se cair)
            df_done = pd.concat([df_done, pd.DataFrame(new_rows)], ignore_index=True)
            df_done.to_parquet(output_path, index=False)
            if warning_records:
                pd.DataFrame(warning_records).to_parquet(warnings_path, index=False)
            print(f"  -- checkpoint: {len(df_done)}/{total} salvos\n")

    print(f"\n  Concluído! Salvo em: {output_path}")
    n_err = sum(1 for w in warning_records if w.get("type") == "error")
    if warning_records:
        print(f"  {len(warning_records)} avisos ({n_err} erros) em: {warnings_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Executa os prompts (sys_prompt + user_input) num servidor vLLM local "
                    "(API OpenAI-compatible), em paralelo, para checar a efetividade da injeção.")
    parser.add_argument("--model", default=MODEL, help="Modelo servido pelo vLLM")
    parser.add_argument("--limit", type=int, default=None, help="Limita número de amostras (útil para teste)")
    parser.add_argument("--output", default=None, help="Caminho do parquet de saída")
    parser.add_argument("--base-url", default=VLLM_BASE_URL, help=f"Base URL do vLLM (default: {VLLM_BASE_URL})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Requisições simultâneas (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Linhas por checkpoint (default: {DEFAULT_CHUNK_SIZE})")
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset RAW não encontrado em: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "__")
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"hackaprompt_injection_{model_slug}.parquet"

    # timeout/max_retries dão resiliência a engasgos transitórios do servidor
    client = OpenAI(api_key=VLLM_API_KEY, base_url=args.base_url, timeout=120.0, max_retries=3)

    inject_dataset(client, args.model, output_path, args.limit, args.concurrency, args.chunk_size)

    print("\nInjeção concluída!")


if __name__ == "__main__":
    main()