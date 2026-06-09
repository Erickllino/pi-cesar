# PI-Cesar

Tradução, validação e benchmark de datasets de **prompt injection** e **jailbreak** do inglês para o **português brasileiro (pt-BR)**.

O projeto parte de uma lacuna documentada na literatura: praticamente não existem benchmarks de segurança de LLMs em português, apesar de a língua ter mais de 250 milhões de falantes. Entradas inseguras traduzidas para línguas de menor recurso contornam as proteções dos modelos com frequência muito maior do que em inglês (ver [`docs/survey.md`](docs/survey.md)). Aqui construímos a base de dados pt-BR necessária para estudar — e futuramente defender contra — esses ataques.

## Objetivos

1. **Traduzir** datasets consagrados de prompt injection/jailbreak para pt-BR, preservando a *intenção adversarial* (o tradutor traduz o ataque literalmente, sem obedecer nem recusar).
2. **Validar** as traduções e medir a efetividade dos ataques traduzidos contra modelos-alvo.
3. **Treinar e avaliar** guardrails (PIGuard) sobre o corpus pt-BR.
4. Servir de base para um **benchmark público pt-BR** de segurança de LLMs.

## Datasets

Cada dataset vive em sua própria pasta com `scripts/` (download/tradução) e uma análise exploratória em notebook. Os dados ficam em [`Data/`](Data/) (fora do versionamento — ver `.gitignore`), separados em `Original/` e `Translated/`.

| Pasta | Dataset de origem | Notas |
|---|---|---|
| [`Hackaprompt/`](Hackaprompt/) | [hackaprompt/hackaprompt-dataset](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | Prompts de competição de prompt injection. Colunas `prompt`, `user_input`, `sys_prompt`, `correct`. |
| [`Rogue Security/`](Rogue%20Security/) | `rs-dataset` (jailbreak/benign) | Traduzido com DeepSeek. |
| [`ArtPrompt/`](ArtPrompt/) | `harmful_behaviors_custom` | Ataques baseados em ASCII art. |
| [`PI-Guard/`](PI-Guard/) | NotInject, WildGuard, BIPIA, injeções de treino | Pipeline completo de treino + avaliação do guardrail. |
| [`Translation-Checker/`](Translation-Checker/) | — | Validação das traduções e execução das injeções (`inject.py`). |

Outros datasets levantados e avaliados quanto à viabilidade de tradução estão em [`docs/data_analysis.md`](docs/data_analysis.md) e [`docs/data_gather2.md`](docs/data_gather2.md) (Qualifire, PINT, BIPIA, Open-Prompt-Injection, NotInject, WAInjectBench, etc.).

## Metodologia de tradução

O princípio central é tratar **todo conteúdo do dataset como dado, nunca como instrução**. As estratégias evoluíram entre os pipelines:

- **System prompt especializado em segurança**, com regras absolutas: traduzir jailbreaks/injeções *literalmente*, sem obedecer nem recusar.
- **Envelopamento do input** em tags `<texto_para_traduzir>...</texto_para_traduzir>` para reforçar o estatuto de "dado".
- **Lembrete pós-conteúdo** contra gatilhos de pré-preenchimento (ex.: `let's start with: "..."`), que devem ser traduzidos como string e não continuados.
- **Few-shot** com exemplos de injeção corretamente traduzida (ver `Hackaprompt/scripts/translate.py`).
- **Checkpoint incremental** por arquivo (parquet/JSON), permitindo retomar runs interrompidos.
- **Flags de qualidade** (`translation_warnings.parquet`) quando a diferença de tamanho entre original e tradução é suspeita.

O relatório completo de prompts, modelos e arquivos gerados está em [`docs/relatorio_traducoes.md`](docs/relatorio_traducoes.md).

### Modelos de tradução usados

- **OpenAI** `gpt-4o`, `gpt-4o-mini` — pipeline PIGuard
- **DeepSeek** `deepseek-chat` — Rogue Security / ArtPrompt
- **Qwen2.5-72B-Instruct-AWQ** via **vLLM local** (API OpenAI-compatível) — Hackaprompt

## Estrutura do repositório

```
PI-Cesar/
├── Hackaprompt/         download.py + translate.py (vLLM) + EDA.ipynb
├── Rogue Security/      translate.py (DeepSeek) + EDA.ipynb
├── ArtPrompt/           translate.py (DeepSeek) + EDO.ipynb
├── PI-Guard/            translate.py, convert_to_piguard.py, prepare_training.py, EDA.ipynb
├── Translation-Checker/ validação (validate.ipynb) + inject.py
├── Data/                Original/ e Translated/ por dataset (não versionado)
├── docs/                survey, análise de datasets e relatórios
├── pyproject.toml       dependências (gerenciado por uv)
└── requirements.txt
```

## Instalação

Requer **Python 3.13+**. O projeto usa [`uv`](https://github.com/astral-sh/uv).

```bash
# com uv (recomendado)
uv sync

# ou com pip
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz com as chaves necessárias (lidas automaticamente pelos scripts):

```bash
OPENAI_API_KEY=...      # pipeline PIGuard
DEEPSEEK_API_KEY=...    # Rogue Security / ArtPrompt
HF_API_KEY=...          # download de datasets do HuggingFace
VLLM_API_KEY=EMPTY      # servidor vLLM local (opcional)
```

## Uso

### 1. Baixar um dataset

```bash
uv run python Hackaprompt/scripts/download.py
```

### 2. Traduzir

```bash
# Hackaprompt — requer um servidor vLLM em http://127.0.0.1:8000/v1
uv run python Hackaprompt/scripts/translate.py --model Qwen/Qwen2.5-72B-Instruct-AWQ
uv run python Hackaprompt/scripts/translate.py --limit 50   # teste rápido

# Rogue Security — DeepSeek
uv run python "Rogue Security/scripts/translate.py" --model deepseek-chat

# PIGuard — OpenAI
uv run python PI-Guard/scripts/translate.py --model gpt-4o
uv run python PI-Guard/scripts/translate.py --model gpt-4o-mini --datasets notinject injections
```

Flags comuns: `--limit` (amostras), `--concurrency` (requisições simultâneas), `--chunk-size` (linhas por checkpoint), `--output` (caminho de saída).

### 3. Validar e medir efetividade da injeção

```bash
# Roda sys_prompt + user_input traduzidos no modelo-alvo e compara
uv run python Translation-Checker/hkp/inject.py --model Qwen/Qwen2.5-72B-Instruct-AWQ
```

A validação manual e a inspeção das traduções ficam nos notebooks `Translation-Checker/*/validate.ipynb`.

## PIGuard em pt-BR — resultados

Após traduzir o corpus, o guardrail PIGuard foi treinado e avaliado sobre os dados pt-BR (detalhes em [`docs/relatorio_traducoes.md`](docs/relatorio_traducoes.md)):

| Métrica | Valor |
|---|---|
| Over-defense ACC (NotInject) | 0,9233 |
| Benign ACC (WildGuard) | 0,8599 |
| Injection ACC (BIPIA) | 0,7133 |
| **Overall ACC** | **0,8322** |

O ponto fraco é **BIPIA_text** (0,47, contra 0,96 em BIPIA_code) — injeções em texto natural traduzido perdem mais sinal do que injeções embutidas em código, principal candidato para iteração futura.

## Contexto acadêmico

Trabalho desenvolvido no contexto de pesquisa em segurança de LLMs (CISSA/UFPE). A fundamentação teórica e o levantamento bibliográfico (Yong et al. NeurIPS 2023, Deng et al. ICLR 2024, Geng et al. CMC 2026, entre outros) estão em [`docs/survey.md`](docs/survey.md).
