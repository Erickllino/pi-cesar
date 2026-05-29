# Análise das Bases de Dados Prompt Injection
## Descrição, Diferenças e Viabilidade de Adaptação para PT-BR
### Ordenado por complexidade crescente de criação de versão em português

---

## Visão Geral Comparativa

| #   | Base                  | Tipo de Ataque                      | Modalidade     | Tamanho                 | Acesso                     | PT-BR   | Complexidade PT-BR |
| --- | --------------------- | ----------------------------------- | -------------- | ----------------------- | -------------------------- | ------- | ------------------ |
| 1   | NotInject             | Over-defense (falsos positivos)     | Texto          | 339                     | Público (HuggingFace)      | Parcial | 🟢 Baixa-Moderada   |
| 2   | Qualifire             | Jailbreak / Direto                  | Texto          | 5.000                   | Público (HuggingFace)      | ❌ Não   | 🟡 Baixa-Moderada   |
| 3   | PINT                  | Direto + Jailbreak + Hard Negatives | Texto          | 4.314                   | Fechado (solicitar acesso) | Parcial | 🟡 Moderada         |
| 4   | Open-Prompt-Injection | Direto (combinado entre tarefas)    | Texto          | Gerado dinamicamente    | Público (GitHub)           | ❌ Não   | 🟠 Moderada-Alta    |
| 5   | BIPIA                 | Indireto (RAG / conteúdo externo)   | Texto          | ~5 tarefas × N amostras | Público (GitHub)           | ❌ Não   | 🔴 Alta             |
| 6   | WAInjectBench         | Direto + Indireto (web agents)      | Texto + Imagem | ~centenas por categoria | Público (GitHub)           | ❌ Não   | 🔴 Alta             |

---

## Diferenças Conceituais Principais

Antes das descrições individuais, é importante entender as **três grandes categorias** em que esses datasets se encaixam, pois elas explicam as diferenças de complexidade:

**1. Datasets de classificação binária** (Qualifire, NotInject): formato mais simples cada amostra é um texto com rótulo `benign` ou `malicious`. Fáceis de traduzir e estender.

**2. Benchmarks de detecção em contexto** (PINT, BIPIA): o ataque está embutido dentro de um documento ou conversa maior. Não basta traduzir o ataque o contexto ao redor também precisa fazer sentido em português.

**3. Toolkits/frameworks de ataque e defesa** (Open-Prompt-Injection, WAInjectBench): não são datasets estáticos. Os ataques são gerados programaticamente combinando tarefas. Adaptar para PT-BR exige criar as tarefas-base em português do zero.

---

## 1. 🟢 NotInject / PIGuard
**Complexidade PT-BR: Baixa a Moderada**

**🔗 Dataset:** https://huggingface.co/datasets/leolee99/NotInject
**🔗 Código:** https://github.com/leolee99/PIGuard

**Descrição precisa:** O NotInject é um dataset de avaliação de *over-defense*, ou seja, ele não testa se um modelo detecta ataques, mas sim se ele **bloqueia indevidamente prompts legítimos** que contêm palavras associadas a ataques (ex: "ignore", "override", "system"). Contém **339 amostras 100% benignas**, construídas para conter deliberadamente essas palavras-gatilho em contextos inocentes. É dividido em 3 subsets pelo número de trigger words por frase (1, 2 ou 3 gatilhos), e em 4 tópicos: *Common Queries*, *Technique Queries*, *Virtual Creation* e **Multilingual Queries** (onde pode haver algum português). Foi publicado junto ao modelo PIGuard no ACL 2025.

**Diferencial em relação aos outros:** É o único dataset focado exclusivamente em **falsos positivos** (over-defense). Todos os outros medem falsos negativos (ataques não detectados). Isso o torna complementar não substituto dos demais.

**Dificuldade PT-BR:** A tarefa de criação é conceitualmente simples: escrever frases benignas em português que contenham as mesmas trigger words. Não exige traduzir ataques complexos nem recriar contextos externos. A licença MIT permite uso livre.

---

## 2. 🟡 Qualifire Prompt-Injection Benchmark
**Complexidade PT-BR: Baixa a Moderada**

**🔗 Dataset:** https://huggingface.co/datasets/qualifire/Qualifire-prompt-injection-benchmark

**Descrição precisa:** Dataset de classificação binária com **5.000 prompts** rotulados como `jailbreak` (instruções maliciosas que tentam subverter o comportamento do modelo) ou `benign` (uso legítimo). É o dataset mais direto do conjunto cada amostra é um único prompt isolado, sem contexto externo, sem tarefas combinadas. Foi criado pela empresa Rogue Security e é adequado para treinar e avaliar classificadores simples de detecção. Requer aceite de termos para acesso no HuggingFace.


**Dificuldade PT-BR:** A tradução automática (DeepL, GPT) com revisão humana é viável para os prompts benignos. O desafio está nos jailbreaks: muitos usam construções linguísticas específicas do inglês (roleplay, personas, estruturas de "DAN") que perdem naturalidade ou eficácia ao ser traduzidas literalmente. Uma versão PT-BR de qualidade exigiria reescrever os ataques nativamente, não apenas traduzir. Licença CC-BY-NC-4.0 (uso não-comercial).

---

## 3. 🟡 PINT Benchmark (Prompt Injection Test)
**Complexidade PT-BR: Moderada**

**🔗 Código:** https://github.com/lakeraai/pint-benchmark
**🔗 Solicitar acesso ao dataset:** https://share-eu1.hsforms.com/1TwiBEvLXRrCjJSdnbnHpLwfdfs3

**Descrição precisa:** Benchmark de avaliação de sistemas de detecção de prompt injectio. Contém **4.314 amostras** distribuídas em 5 categorias: `prompt_injection` (5,2%), `jailbreak` (0,9%), `hard_negatives` (20,9% prompts benignos que parecem ataques), `chat` (36,5% conversas usuário-agente) e `documents` (36,5% documentos públicos). O diferencial técnico é o alto percentual de **hard negatives**, que testa a robustez do classificador contra falsos positivos em texto real. O dataset é **fechado** (não público), e a Lakera mantém controle para evitar que soluções comerciais sejam treinadas diretamente nele.



**Dificuldade PT-BR:** Português já existe no dataset, mas como idioma minoritário dentro dos 1.298 não-ingleses. Criar uma versão PT-BR dedicada exigiria acesso ao dataset completo (via formulário) e expandir as amostras nas 5 categorias para o português, mantendo o equilíbrio entre categorias.

---

## 4. 🟠 Open-Prompt-Injection
**Complexidade PT-BR: Moderada a Alta**

**🔗 Código:** https://github.com/liu00222/Open-Prompt-Injection

**Descrição precisa:** Não é um dataset estático é um **framework/toolkit** que gera ataques de prompt injection dinamicamente. O conceito central é o ataque *combinado*: um modelo LLM está executando uma **tarefa-alvo** (ex: analisar sentimento de uma review) e o atacante injeta instruções de uma **tarefa secundária** (ex: classificar spam) dentro dos dados de entrada. O framework suporta 7 datasets de NLP como base (SST-2, SMS Spam, HSOL, MRPC, RTE, Gigaword, JFLEG) e múltiplos LLMs (GPT-4, PaLM2, Llama 2, Flan-T5). A métrica principal é o **ASV (Attack Success Rate vs. Victim)** percentual em que o modelo executa a tarefa injetada em vez da original.

**Diferencial em relação aos outros:** É o único framework que modela o ataque como uma **competição entre duas tarefas NLP**, o que reflete cenários reais de LLMs integrados em pipelines. Também inclui detecção (DataSentinel) e localização do ataque (PromptLocate). Os outros datasets tratam o ataque como um texto binário (malicioso/benigno); aqui o ataque tem estrutura semântica complexa.

**Dificuldade PT-BR:** Exige criar datasets equivalentes em PT-BR para cada tarefa suportada (ex: um dataset de análise de sentimento em português no lugar do SST-2). A parte técnica do framework é acessível; o esforço está na curadoria dos dados de base. **Obs:** É possível fazer fine-tuning dos modelos de detecção da biblioteca com dados em português.

---

## 5. 🔴 BIPIA Benchmark of Indirect Prompt Injection Attacks
**Complexidade PT-BR: Alta**

**🔗 Código:** https://github.com/microsoft/BIPIA

**Descrição precisa:** Desenvolvido pela Microsoft Research, o BIPIA é o principal benchmark para **ataques indiretos** onde o ataque não vem diretamente do usuário, mas está embutido em **conteúdo externo** que o LLM processa (como um e-mail recebido, uma página web, uma tabela ou um trecho de código). O modelo trabalha em 5 tarefas: *Email QA* (responder perguntas sobre e-mails), *Web QA* (responder sobre páginas web), *Table QA* (responder sobre tabelas), *Summarization* (resumir textos) e *Code QA* (responder sobre código). Em cada tarefa, um ataque está injetado no contexto externo. Avalia tanto a **taxa de sucesso do ataque (ASR)** quanto a eficácia de defesas (meta-prompting e fine-tuning).

**Diferencial em relação aos outros:** É o único dataset focado exclusivamente em **ataques indiretos via RAG/contexto externo**. Enquanto os outros datasets tratam o prompt do usuário como vetor de ataque, o BIPIA simula o cenário onde o ataque vem de uma fonte externa processada pelo modelo o cenário mais realista para aplicações empresariais com RAG.

**Dificuldade PT-BR:** Alta porque não basta traduzir os prompts de ataque é preciso recriar os contextos externos (e-mails, páginas web, tabelas) em português de forma que façam sentido culturalmente, e os ataques precisam estar semanticamente integrados nesses contextos. Uma tradução mecânica quebraria a coerência entre contexto e ataque.

---

## 6. 🔴 WAInjectBench
**Complexidade PT-BR: Alta**

**🔗 Código:** https://github.com/Norrrrrrr-lyn/WAInjectBench

**Descrição precisa:** Benchmark focado em detecção de prompt injection especificamente em **web agents** LLMs que navegam na web e executam ações. Cobre **6 tipos de ataques** em **duas modalidades**: texto (JSONL com amostras benignas e maliciosas) e **imagem** (screenshots de páginas web com ataques injetados visualmente). Para modalidade de texto, avalia detectores como KAD, PromptArmor, DataSentinel e PromptGuard. Para imagem, avalia GPT-4o, LLaVA-1.5-7B e JailGuard. É o dataset mais recente e especializado do conjunto, voltado ao cenário de agentes autônomos.

**Diferencial em relação aos outros:** É o único benchmark que trata **ataques em imagem** como vetor. Os outros cobrem apenas texto. Também é o único focado no contexto de *web agents*, onde o modelo não só lê mas age (clica, preenche formulários, executa scripts) o que eleva drasticamente o impacto de um ataque bem-sucedido.

**Dificuldade PT-BR:** A mais alta do conjunto. A parte textual pode ser traduzida com esforço moderado, mas os **ataques em imagem** (texto injetado visualmente em screenshots de páginas web) precisariam ser re-gerados com páginas em português o que requer criação de novos artefatos visuais, não apenas tradução de texto.

---

## Resumo de Diferenças por Dimensão

| Dimensão              | NotInject            | Qualifire            | PINT               | Open-PI                   | BIPIA                        | WAInjectBench          |
| --------------------- | -------------------- | -------------------- | ------------------ | ------------------------- | ---------------------------- | ---------------------- |
| **O que mede?**       | Over-defense         | Detecção binária     | Detecção geral     | Taxa de sucesso de ataque | Robustez a ataques indiretos | Detecção em web agents |
| **Tipo de ataque**    | Nenhum (só benignos) | Jailbreak direto     | Direto + jailbreak | Combinado entre tarefas   | Indireto via RAG             | Direto + indireto      |
| **Modalidade**        | Texto                | Texto                | Texto              | Texto                     | Texto                        | Texto + Imagem         |
| **Dataset estático?** | ✅ Sim                | ✅ Sim                | ✅ Sim              | ❌ Gerado dinamicamente    | ✅ Sim                        | ✅ Sim                  |
| **Acesso**            | Público              | Público (com aceite) | Fechado            | Público                   | Público                      | Público                |
| **Contexto externo?** | ❌                    | ❌                    | Parcial            | ❌                         | ✅ Sempre                     | ✅ Sempre               |
| **Multilíngue?**      | Parcial              | ❌                    | ✅ ~20 idiomas      | ❌                         | ❌                            | ❌                      |
