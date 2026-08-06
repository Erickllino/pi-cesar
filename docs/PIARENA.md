# Dados do PIArena

## Escopo

Este repositório usa o PIArena somente como fonte de dados para tradução. Não inclui nem executa a implementação upstream de benchmark, ataques, defesas, agentes ou interface web.

## Origem

Os dados vêm de [`sleeepeer/PIArena`](https://huggingface.co/datasets/sleeepeer/PIArena). O script `scripts/download.py` grava os JSONs em `Data/PIArena/Original/`. Como `Data/` é ignorado pelo Git, cada execução deve registrar a revisão ou commit do dataset utilizado.

## Arquivos

A fonte possui 16 arquivos JSON. Treze são traduzidos por padrão; os três arquivos abaixo ficam de fora porque medem corrupção de conhecimento e não preservam a mesma métrica após tradução:

- `hotpotqa_rag_knowledge_corruption`
- `msmarco_rag_knowledge_corruption`
- `nq_rag_knowledge_corruption`

Cada exemplo possui `context`, `target_inst`, `injected_task`, `target_task_answer`, `injected_task_answer` e `category`.

`category` é metadado e permanece em inglês. `injected_task_answer` está vazio nos dados de origem e não é traduzido.

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
