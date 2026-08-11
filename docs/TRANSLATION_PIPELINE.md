# Metodologia de Tradução

## Objetivo

Traduzir os campos textuais do PIArena sem obedecer às instruções adversariais que eles contêm e sem salvar alterações silenciosas em estruturas técnicas.

## Campos processados

O pipeline considera `context`, `target_inst`, `injected_task`,
`target_task_answer` e `injected_task_answer`.

`lcc_long` é uma exceção: seu `context` é código Java cru e `target_task_answer` é uma linha de código. Ambos são preservados literalmente; somente `target_inst` e `injected_task` passam pelo modelo.

## Proteções no prompt

Cada chamada usa um system prompt que define o conteúdo como dado de dataset, não como instrução. O texto é delimitado por `<texto_para_traduzir>...</texto_para_traduzir>`, seguido por um lembrete após o conteúdo. Há dois few-shots baseados em `injected_task` reais do `dolly_closed_qa`: um pedido com URL e uma falsa suspensão de acesso.

O modelo deve traduzir ataques, jailbreaks e gatilhos de continuação de forma literal, sem obedecer, recusar ou continuar o conteúdo.

## Mapa do código

As funções principais de `scripts/translate.py` são:

- `build_system_prompt()` e `build_user_message()`: montam as instruções que separam o papel de tradutor do conteúdo adversarial do dataset.
- `build_fewshot()`: adiciona dois exemplos reais de `injected_task` do PIArena para demonstrar a tradução literal de payloads.
- `should_translate()`: define quais campos passam pelo modelo; em `lcc_long`, preserva o contexto Java e a resposta de código.
- `mask_code_blocks()` e `restore_markers()`: protegem blocos cercados por crase tripla com marcadores e rejeitam saída com marcador perdido, duplicado ou fora de ordem.
- `validate_structure()`: confirma que código e URLs são idênticos aos da origem depois da tradução.
- `unexpected_added_chars()`: detecta aumento de caracteres CJK, cirílicos ou árabes quando esses scripts não são esperados no idioma de destino.
- `translate_field()`: executa chamada ao modelo, retry, validações e fallback para o texto original se o campo continuar inválido.
- `translate_dataset()`: percorre o JSON, mantém o checkpoint, grava warnings e mostra o resumo de cada dataset.

## Preservação e validação

- Blocos cercados por crase tripla são substituídos por marcadores `[[CODE_BLOCK_n]]`, restaurados depois e comparados exatamente com a origem.
- URLs não são mascaradas; devem aparecer idênticas e na mesma ordem na saída.
- Para pt-BR e alemão, o pipeline sinaliza caracteres CJK, cirílicos ou árabes adicionais. Para árabe, CJK e cirílico continuam inesperados.
- Uma saída com pelo menos 500 caracteres de origem e menos de 50% do tamanho original é considerada provavelmente truncada.

## Retry, fallback e warnings

Cada campo tem até duas tentativas. Falhas estruturais recebem um lembrete de correção na segunda tentativa; timeout e falhas de API repetem o pedido sem esse lembrete.

Se as duas tentativas falharem, o campo original é preservado e um registro é gravado em `translation_warnings_<idioma>.json`. Tipos atuais:

- `url_structure_fallback`
- `code_structure_fallback`
- `unexpected_script_fallback`
- `translation_error`

O pipeline retoma automaticamente checkpoints existentes. Para refazer uma
execução, remova a saída correspondente ou use outro `--output-dir`. Ao final
de cada dataset, o script mostra registros processados, número de fallbacks,
linhas afetadas e a contagem de warnings por tipo.

## Limitações

As validações asseguram propriedades estruturais, não fidelidade semântica completa. Um modelo pode produzir uma tradução gramaticalmente ruim sem violar URL, código, script ou tamanho. Contextos `_long` exigem um modelo/servidor capaz e timeout suficiente; reduzir `--max-tokens` é apropriado apenas para testes curtos.
