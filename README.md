# PiGuard

Pipeline de preparação e tradução dos datasets usados no treinamento e na avaliação do PiGuard.

## Estrutura

```text
PIGuard/
├── EDA.ipynb
└── scripts/
    ├── translate.py
    ├── translate_train.py
    ├── translate_train_v2.py
    ├── translate_train_v3.py
    ├── translate_v4.py
    ├── translate_v5.py
    ├── translate_v6.py
    ├── convert_to_piguard.py
    └── prepare_training.py
```

Os datasets não são versionados. Por padrão, os scripts leem e gravam em:

```text
Data/PIGuard/
├── Original/
└── Translated/
```

## Instalação

Requer Python 3.13 ou superior e, preferencialmente, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Também é possível instalar com pip:

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz quando necessário:

```dotenv
OPENAI_API_KEY=...
VLLM_API_KEY=EMPTY
```

O arquivo `.env` e a pasta `Data/` são ignorados pelo Git.

## Uso

Consulte todas as opções de cada pipeline com `--help`:

```bash
uv run python PIGuard/scripts/translate.py --help
uv run python PIGuard/scripts/translate_v5.py --help
uv run python PIGuard/scripts/translate_train_v3.py --help
```

Exemplo usando a API da OpenAI:

```bash
uv run python PIGuard/scripts/translate.py --model gpt-4o
```

Exemplo traduzindo o conjunto de treino com um servidor vLLM compatível com a API da OpenAI:

```bash
uv run python PIGuard/scripts/translate_train_v3.py \
  --model Qwen/Qwen2.5-72B-Instruct-AWQ \
  --base-url http://127.0.0.1:8000/v1 \
  --language pt_br
```

Use `--limit` e um diretório de saída separado antes de executar o pipeline completo.
