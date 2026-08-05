# Metodologia de Tradução

## Objetivo

Traduzir os campos textuais do PIArena sem obedecer às instruções adversariais que eles contêm e sem salvar alterações silenciosas em estruturas técnicas.

## Campos processados

O pipeline considera `context`, `target_inst`, `injected_task` e `target_task_answer`.

`lcc_long` é uma exceção: seu `context` é código Java cru e `target_task_answer` é uma linha de código. Ambos são preservados literalmente; somente `target_inst` e `injected_task` passam pelo modelo.

## Proteções no prompt

Cada chamada usa um system prompt que define o conteúdo como dado de dataset, não como instrução. O texto é delimitado por `<texto_para_traduzir>...</texto_para_traduzir>`, seguido por um lembrete após o conteúdo. Há few-shots de uma injeção e de uma URL preservada.

O modelo deve traduzir ataques, jailbreaks e gatilhos de continuação de forma literal, sem obedecer, recusar ou continuar o conteúdo.

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

O pipeline retoma automaticamente checkpoints existentes. `--restart` apaga apenas a saída e os warnings dos datasets selecionados. Ao final de cada dataset, o script mostra registros processados, número de fallbacks, linhas afetadas e a contagem de warnings por tipo.

## Limitações

As validações asseguram propriedades estruturais, não fidelidade semântica completa. Um modelo pode produzir uma tradução gramaticalmente ruim sem violar URL, código, script ou tamanho. Contextos `_long` exigem um modelo/servidor capaz e timeout suficiente; reduzir `--max-tokens` é apropriado apenas para testes curtos.
