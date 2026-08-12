# Dados do PIArena

## Escopo

Este repositório usa o PIArena somente como fonte de dados para tradução. Não inclui nem executa a implementação upstream de benchmark, ataques, defesas, agentes ou interface web.

## Origem

Os dados vêm de [`sleeepeer/PIArena`](https://huggingface.co/datasets/sleeepeer/PIArena). O script `scripts/download.py` grava os JSONs em `Data/PIArena/Original/`. Como `Data/` é ignorado pelo Git, cada execução deve registrar a revisão ou commit do dataset utilizado.

## Arquivos

A fonte possui 16 arquivos JSON. Todos são traduzidos por padrão, inclusive os
três arquivos de corrupção de conhecimento. O pipeline apenas traduz os dados;
qualquer adaptação posterior das métricas do benchmark é responsabilidade da
etapa de avaliação.

Cada exemplo possui `context`, `target_inst`, `injected_task`, `target_task_answer`, `injected_task_answer` e `category`.

`category` é metadado e permanece em inglês. `injected_task_answer` é vazio na
maioria dos datasets, mas é traduzido quando estiver preenchido.

## Datasets de Corrupção de Conhecimento

Nos datasets `*_knowledge_corruption`, `injected_task` costuma ficar vazio e
`injected_task_answer` contém a resposta incorreta que representa o ataque. Por
isso, os dois campos de resposta são traduzidos: `target_task_answer` continua
representando a resposta correta, enquanto `injected_task_answer` representa a
resposta induzida pelo ataque.

Antes de executar a avaliação original desses datasets com `substring_match`,
revise os dois campos de resposta no idioma-alvo. Cada um precisa preservar o
fato correspondente e usar uma forma compatível com a resposta esperada do
modelo. Eles não devem ser iguais entre si; são gabaritos para resultados
distintos. A adaptação ou normalização posterior da métrica pertence à etapa de
avaliação, não a este pipeline de tradução.

## Comandos

Baixar os JSONs:

```cmd
python scripts\download.py --revision main
```

Execução padrão pela API da OpenAI. Sem `--limit`, traduz todos os registros;
o modelo padrão é `gpt-4o` e `--max-tokens` não é imposto pelo script:

```cmd
set OPENAI_API_KEY=SUA_CHAVE

python scripts\translate.py --language pt_br
```

Para testar somente algumas linhas, acrescente `--limit`. Para outro idioma,
informe somente a língua desejada:

```cmd
python scripts\translate.py ^
  --language de ^
  --limit 5
```

Use `--model` apenas quando quiser trocar o modelo padrão `gpt-4o`.

### Paralelismo

O padrão é `--concurrency 1`, portanto uma execução local com Ollama continua
sequencial. Em uma API ou servidor vLLM com capacidade para múltiplas requisições,
é possível traduzir registros em paralelo:

```cmd
python scripts\translate.py ^
  --language de ^
  --concurrency 8
```

Comece com `--concurrency 4` ou `8` e ajuste conforme os limites de taxa, memória
e estabilidade do servidor. Para Ollama local, mantenha `1`. Nos datasets `_long`,
prefira uma concorrência menor, como `2` ou `4`.

Teste local com Ollama:

```cmd
set OPENAI_API_KEY=ollama

python scripts\translate.py ^
  --language pt_br ^
  --model qwen2.5:7b ^
  --base-url "http://localhost:11434/v1" ^
  --datasets dolly_closed_qa hotpotqa_rag lcc_long ^
  --limit 2 ^
  --max-tokens 512
```

As saídas ficam em `Data/PIArena/Translated/<modelo>/<idioma>/`.
O script retoma automaticamente o checkpoint existente. Para refazer uma
execução, remova manualmente a saída correspondente ou informe outro
`--output-dir`.
