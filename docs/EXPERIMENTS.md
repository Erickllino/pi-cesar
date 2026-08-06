# Registro de Execuções

Registre uma seção por execução relevante. Não registre como resultado de referência testes locais pequenos feitos apenas para verificar o pipeline.

## Modelo de registro

```text
Data:
Responsável:
Idioma:
Modelo:
Endpoint:
Revisão do dataset PIArena:
Datasets processados:
Parâmetros: --timeout, --max-tokens, --limit, --output-dir

Linhas de entrada:
Linhas concluídas:
Warnings por tipo:
Campos em fallback:

Observações de qualidade:
Limitações ou falhas conhecidas:
Caminho local da saída:
```

## Referência de interpretação

- Um warning/fallback protege contra uma saída inválida conhecida; não é prova de tradução de baixa qualidade em todos os demais campos.
- Ausência de warning não garante fidelidade semântica; amostras devem ser revisadas manualmente, sobretudo em `*_long`.
- Compare execuções somente quando idioma, revisão do dataset, modelo e parâmetros estiverem registrados.
