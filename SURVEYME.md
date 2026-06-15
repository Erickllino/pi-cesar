# SURVEYME

Guia de organização da documentação do **PI-Cesar** para a escrita de um **artigo no formato survey** sobre *prompt injection* em LLMs com foco em **português brasileiro (pt-BR) e contextos multilíngues**.

Este arquivo tem dois propósitos:

1. **Explicar o que cada documento em [`docs/`](docs/) contém** e como ele alimenta o artigo.
2. **Propor a estrutura do survey** seção a seção, já mapeando cada referência de [`docs/survey.md`](docs/survey.md) e cada análise de dataset para o lugar onde ela deve entrar.

---

## 1. Mapa da documentação

Todos os insumos do artigo estão em [`docs/`](docs/). Abaixo, o que cada um é e para que serve na escrita.

| Documento | O que é | Papel no survey |
|---|---|---|
| [`docs/survey.md`](docs/survey.md) | **Bibliografia anotada** — 10 referências principais + bônus pt-BR/locais, cada uma com resumo e indicação de em quais seções entra. | Espinha dorsal do artigo. Define a narrativa e o estado da arte. |
| [`docs/data_analysis.md`](docs/data_analysis.md) | Levantamento de **6 datasets** (Qualifire, PINT, BIPIA, Open-Prompt-Injection, NotInject, WAInjectBench) com links, existência de pt-BR e dificuldade de adaptação. | Matéria-prima da seção de **datasets/benchmarks**. |
| [`docs/data_analysis_v2.md`](docs/data_analysis_v2.md) | Versão **revisada e mais rigorosa** da anterior: tabela comparativa, 3 categorias conceituais, descrição precisa de cada base e matriz de diferenças por dimensão. | **Use esta como fonte canônica** para a seção de datasets (substitui a v1). |
| [`docs/data_gather2.md`](docs/data_gather2.md) | Lista curta de **datasets adicionais** de injeção indireta/agêntica (BIPIA+GPT 70k, LLMail-Inject, InjecAgent, PoisonedRAG). | Amplia a cobertura de **ataques indiretos e agênticos**. |
| [`docs/relatorio_traducoes.md`](docs/relatorio_traducoes.md) | Relatório técnico do **trabalho de tradução** do PI-Cesar (prompts, modelos, pipelines, resultados do PIGuard em pt-BR). | Base da seção de **metodologia/contribuição própria** e dos **resultados experimentais**. |

> **Recomendação:** ao escrever, trate `data_analysis_v2.md` como a fonte oficial sobre datasets e considere `data_analysis.md` apenas histórico. Cite `data_gather2.md` para cobrir o eixo indireto/agêntico que a v2 não detalha.

---

## 2. O documento central: `survey.md`

O `survey.md` é uma **bibliografia anotada já pré-organizada por seção**. Cada entrada indica explicitamente "→ Seção X". Consolidando essas indicações, o esqueleto do artigo emerge naturalmente:

| Ref. | Trabalho (ano) | Contribuição-chave | Seção(ões) alvo |
|---|---|---|---|
| 1 | Yong, Menghini & Bach — *Low-Resource Languages Jailbreak GPT-4* (NeurIPS 2023) | Traduzir para línguas de baixo recurso contorna proteções (79% de respostas acionáveis). **Paper fundacional.** | 1, 6 |
| 2 | Deng et al. — *Multilingual Jailbreak Challenges* (ICLR 2024) | Dataset MultiJail; cenários "não-intencional" vs "intencional"; baixo recurso ~3× mais nocivo. | 3, 6 |
| 3 | Wang et al. — *All Languages Matter* (ACL 2024 Findings) | XSAFETY, 1º benchmark de segurança multilíngue em larga escala. | 5 |
| 4 | Shen et al. — *The Language Barrier* (2024) | Disseca **por que** a segurança falha fora do inglês. | 6 |
| 5 | Geng et al. — *Prompt Injection Attacks on LLMs: A Survey* (CMC 2026) | **Survey-âncora.** Taxonomia + defesas; já aponta dimensão cross-lingual como frente de pesquisa. | 4, 5 |
| 6 | Abbasi et al. — *Multilingual Prompt Injection Detection* (SSRN 2025) | Molde metodológico: traduz EN→ES, avalia 19 modelos. **O mais próximo do desenho do PI-Cesar.** | 3, 5 |
| 7 | MIPIAD — *Multilingual Indirect PI Defense* (2026) | Construído sobre BIPIA; gap EN–bangla em injeção indireta. | 4, 5 |
| 8 | Theocharopoulos et al. — *Hidden PI on Academic Reviewing* (2025) | Injeção indireta "escondida"; o idioma do ataque muda o efeito. | 4 |
| 9 | Marx & Dunaiski — *Jailbreaking Using Low-Resource Languages* (2026) | Red-teaming nativo multi-turno > tradução single-turn. | 6, 7 |
| 10 | Queiroz (UFJF) — *Adversarial Versification in Portuguese* (2025) | **Item mais próximo de pt-BR**; argumenta a lacuna crítica do português. | 6 |
| B1 | SecBERT (Amorim, TCC CIn/UFPE) | Classificação de jailbreak em pt-BR com BERTimbau. **Conexão local (UFPE).** | 6 (relacionados) |
| B2 | MiJaBench (2026) | Inclui testes em português para consistência cross-lingual. | 5 |
| B3 | *Jailbreaking and Mitigation of Vulnerabilities* (2024) | Survey com taxonomia que inclui jailbreak multilíngue. | 4 (comparar taxonomias) |

---

## 3. Estrutura proposta do survey

A seguir, a estrutura do artigo derivada das indicações "→ Seção X" do `survey.md`, com os insumos de cada documento mapeados para cada seção.

### Seção 1 — Introdução e motivação
- **Tese central:** segurança de LLMs degrada fora do inglês; pt-BR é uma lacuna crítica (250M+ falantes, sem benchmarks dedicados).
- **Fontes:** ref. 1 (Yong et al. — gatilho da motivação), ref. 10 (Queiroz — formulação explícita da lacuna pt-BR).
- **Gancho do PI-Cesar:** apontar que este trabalho constrói o corpus pt-BR que falta.

### Seção 2 — Fundamentos e definições
- Diferenciar **prompt injection** (direto vs indireto), **jailbreak**, **over-defense** (falsos positivos).
- **Fonte:** as 3 categorias conceituais de [`docs/data_analysis_v2.md`](docs/data_analysis_v2.md) (classificação binária / detecção em contexto / toolkits de ataque-defesa) servem perfeitamente como base taxonômica introdutória.

### Seção 3 — Metodologia de construção de datasets multilíngues
- Como se constroem datasets de segurança cross-lingual: tradução vs criação nativa; intencional vs não-intencional.
- **Fontes:** ref. 2 (MultiJail), ref. 6 (Abbasi — molde EN→ES, espelha o pipeline do PI-Cesar).
- **Contribuição própria:** descrever aqui a metodologia de tradução do PI-Cesar a partir de [`docs/relatorio_traducoes.md`](docs/relatorio_traducoes.md) (system prompt anti-obediência, envelopamento em tags, resistência a pré-preenchimento, few-shot, checkpoint).

### Seção 4 — Taxonomia de ataques (com ênfase em indireto/agêntico)
- Taxonomia de métodos de ataque; injeção indireta via RAG/contexto externo; injeção em pipelines de agentes.
- **Fontes:** ref. 5 (Geng — taxonomia-âncora), ref. 7 (MIPIAD), ref. 8 (injeção escondida), B3 (taxonomia comparativa).
- **Datasets de apoio:** BIPIA e WAInjectBench de [`docs/data_analysis_v2.md`](docs/data_analysis_v2.md); BIPIA+GPT 70k, LLMail-Inject, InjecAgent, PoisonedRAG de [`docs/data_gather2.md`](docs/data_gather2.md).

### Seção 5 — Benchmarks, avaliação e defesas
- Panorama de benchmarks de segurança multilíngue e de detecção de injeção; estratégias de defesa.
- **Fontes:** ref. 3 (XSAFETY), ref. 5 (defesas), ref. 6 (avaliação de 19 modelos), ref. 7, B2 (MiJaBench).
- **Tabela-mestra:** a matriz comparativa "Resumo de Diferenças por Dimensão" de [`docs/data_analysis_v2.md`](docs/data_analysis_v2.md) (o que mede / tipo de ataque / modalidade / multilíngue?) é o material pronto para a tabela de benchmarks do survey.
- **Resultados próprios:** acurácias do PIGuard em pt-BR de [`docs/relatorio_traducoes.md`](docs/relatorio_traducoes.md) (Overall 0,8322; ponto fraco BIPIA_text 0,47).

### Seção 6 — O caso do português e o gap cross-lingual
- Por que a segurança falha fora do inglês; especificidades do pt-BR; tradução vs dados nativos.
- **Fontes:** ref. 4 (Shen — *por que* falha), ref. 1 e 2 (evidência de degradação), ref. 9 (nativo > tradução), ref. 10 (lacuna pt-BR), B1 (SecBERT/UFPE como trabalho local relacionado).

### Seção 7 — Direções futuras
- Coleta nativa pt-BR (red-teaming com falantes nativos), ataques multi-turno, benchmark público pt-BR.
- **Fontes:** ref. 9 (multi-turno e red-teaming nativo), mais a agenda própria do PI-Cesar.

### Seção 8 — Conclusão

---

## 4. Lacunas a preencher antes de submeter

Itens que **não** estão cobertos pelos docs atuais e precisam ser produzidos:

- **Trabalhos relacionados puramente em pt-BR** além de SecBERT e Queiroz — a base local ainda é fina.
- **Metodologia de avaliação própria** formalizada (métricas, modelos-alvo, protocolo) — hoje só há os números do PIGuard.
- **Verificação das citações** (anos, IDs arXiv/DOI) do `survey.md` antes de virar bibliografia formal.
- **Posicionamento vs. o survey-âncora** (Geng et al., ref. 5): deixar claro o delta do PI-Cesar — foco pt-BR + corpus traduzido + benchmark.

---

## 5. Fluxo de trabalho sugerido

1. Ler `survey.md` e fixar o esqueleto de 8 seções acima.
2. Para cada seção, puxar as referências da tabela do item 2 e os datasets de `data_analysis_v2.md` / `data_gather2.md`.
3. Inserir a contribuição própria (metodologia + resultados) a partir de `relatorio_traducoes.md` nas Seções 3 e 5.
4. Fechar as lacunas do item 4.
5. Converter a bibliografia anotada em referências formais (BibTeX).
