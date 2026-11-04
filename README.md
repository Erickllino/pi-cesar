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

## Interface de revisão de prompts

Para revisar os prompts dos arquivos JSON em `Data/`, inicie a interface local (sem dependências adicionais):

```bash
python PIGuard/review_app.py
```

Abra **http://127.0.0.1:8765** no navegador. É possível escolher outro diretório ou porta com `--data-dir` e `--port`.

A interface permite filtrar por dataset, status, rótulo original (`0` = non-malicious; `1` = malicious) e texto; marcar **Malicious**, **Non-malicious** ou **Incerto**; selecionar múltiplas categorias de técnica; adicionar novas categorias e registrar observações. A catalogação original é apenas exibida, sem alteração. Para arquivos BIPIA e NotInject sem `label`, o rótulo original exibido é inferido do nome do dataset (BIPIA = `1`, NotInject = `0`); os JSONs não são modificados e isso não cria uma revisão humana. As avaliações e categorias personalizadas ficam em `Data/reviews.sqlite3` (ignorado pelo Git). O botão **Exportar CSV** gera um arquivo com as avaliações salvas, incluindo os metadados e o texto original de cada prompt. O CSV contém apenas prompts revisados. Há um veredito humano compartilhado por prompt; a última revisão salva prevalece.

> A interface escuta apenas em `127.0.0.1` por padrão. Não a exponha na rede sem adicionar autenticação.

### Comparar as três avaliações

Abra **http://127.0.0.1:8765/compare** para ver, lado a lado, o prompt com seu rótulo/categoria/fonte originais, o resultado já salvo do Jev (probabilidades, confiança, técnicas e modelo) e a revisão humana (veredito, técnicas e observações). Filtre por dataset, rótulo, texto, divergências por par, avaliações ausentes e prompts com as três decisões. Os contadores de concordância e divergência total consideram apenas prompts com **três vereditos binários**; revisão manual `Incerto` e rótulos originais ausentes não são comparados como se fossem decisões binárias. A página é somente leitura e não chama a API da TypeSafe. A interface e as rotas para executar o juiz não são servidas.

### Hospedagem

Esta versão ainda usa SQLite local e arquivos em `Data/`; **não está pronta para implantação na Vercel**. Funções da Vercel não oferecem armazenamento SQLite persistente, e os datasets ignorados pelo Git não acompanham o deploy. Para uso compartilhado será necessário mover os prompts, as avaliações e os resultados históricos do Jev para armazenamento persistente externo, além de configurar autenticação e backups. Não publique o servidor local diretamente na internet sem essas adaptações.
