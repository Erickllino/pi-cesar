

# Base de dados:
### ➢ Qualifire Prompt-Injection Benchmark
○ Conjunto de prompts rotulados como benign vs jailbreak (instruções maliciosas). Adequado para avaliação básica de classificadores.
○ https://huggingface.co/datasets/qualifire/Qualifire-prompt-injection-benchmark
### ➢ PINT Benchmark (Prompt Injection Test)
○ Benchmark abrangendo múltiplos tipos de ataques (hard negatives, jailbreaks, multilíngues).
○ https://github.com/lakeraai/pint-benchmark
### ➢ BIPIA (Benchmark of Indirect Prompt Injection Attacks)
○ Focado em ataques indiretos (conteúdo externo ou RAG). Referência para detecção contextual.
○ https://github.com/microsoft/BIPIA
### ➢ Open-Prompt-Injection Benchmark
○ Repositório aberto a ataques e defesas, com scripts de benchmarking.
○ https://github.com/liu00222/Open-Prompt-Injection
### ➢ NotInject (via InjecGuard)
○ Dataset voltado a medir over-defense (falsos positivos em prompts benignos). Útil para avaliação de guardrails.
○ https://github.com/SaFoLab-WISC/InjecGuard
### ➢ WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents
○ WAInjectBench is a comprehensive benchmark for prompt injection detection in web agents. It covers 6 types of attacks, across two
modalities: text and image.
○ https://github.com/Norrrrrrr-lyn/WAInjectBench


## 1. Qualifire Prompt-Injection Benchmark

**🔗 Link:** https://huggingface.co/datasets/qualifire/Qualifire-prompt-injection-benchmark

**Existe PT-BR?** Não. O dataset contém 5.000 prompts rotulados como `jailbreak` ou `benign`, com apenas duas colunas: `text` e `label`. Todo é em inglês.

**Dificuldade de traduzir/criar versão PT-BR:** 🟡 **Baixa a Moderada**

A estrutura é simples (texto + rótulo binário). Tradução automática com revisão humana é viável, mas ataques de jailbreak têm nuances culturais e linguísticas, uma tradução literal pode perder a eficácia do ataque, o que exige curadoria manual. A licença **CC-BY-NC-4.0** permite uso não-comercial adaptado.



## 2. PINT Benchmark

**🔗 Link:** https://github.com/lakeraai/pint-benchmark

**Existe PT-BR?** ⚠️ Parcialmente. O dataset abrange múltiplos idiomas do grupo Indo-Europeu, incluindo pt-br, francês, alemão, espanhol, russo, entre outros. Portanto, o português já está representado.

O dataset possui **4.314 amostras no total**, sendo 3.016 em inglês e 1.298 em outros idiomas.

**Dificuldade de criar versão PT-BR dedicada:** 🟡 **Moderada**

O dataset não é publico, necessita pedir acesso https://share-eu1.hsforms.com/1TwiBEvLXRrCjJSdnbnHpLwfdfs3

## 3. BIPIA — Benchmark of Indirect Prompt Injection Attacks (Microsoft)

**🔗 Link:** https://github.com/microsoft/BIPIA

**Existe PT-BR?** Não. O BIPIA foca em ataques indiretos com cinco tarefas: Web QA, Email QA, Table QA, Summarization e Code QA — todas em inglês. Os dados de contexto vêm de fontes como OpenAI Evals, WikiTableQuestions e Stack Exchange, predominantemente em inglês.

**Dificuldade de traduzir/criar versão PT-BR:** 🔴 **Alta**

Não se trata apenas de traduzir prompts, mas também de traduzir os **contextos externos** (e-mails, páginas web, tabelas, código) nos quais os ataques estão embutidos.

## 4. Open-Prompt-Injection

**🔗 Link:** https://github.com/liu00222/Open-Prompt-Injection

**Existe PT-BR?** Não. O repositório fornece um toolkit para ataques e defesas, utilizando tarefas como análise de sentimento (SST2) e detecção de spam — todas em inglês.

**Dificuldade de criar versão PT-BR:** 🟠 **Moderada a Alta**

O framework é modular e extensível — é possível criar novas tarefas e datasets em português via arquivos de configuração JSON. Contudo, seria necessário encontrar ou criar datasets equivalentes em PT-BR para as tarefas-alvo (ex: análise de sentimento em português) e construir os ataques correspondentes. A parte técnica é acessível; o esforço está na curadoria dos dados.

**Obs:** Pode ser fazer fine tuning da biblioteca

## 5. NotInject / PIGuard (InjecGuard)

**🔗 Link:** https://huggingface.co/datasets/leolee99/NotInject

**Existe PT-BR?** ⚠️ Parcialmente. O NotInject contém **339 amostras benignas** com *trigger words* comuns em ataques de prompt injection, divididas em quatro tópicos: Common Queries, Technique Queries, Virtual Creation e **Multilingual Queries**. O tópico multilíngue pode conter algum português, mas o dataset não é somente pt-br.

**Dificuldade de criar versão PT-BR:** 🟢 **Baixa a Moderada**

Este é provavelmente o **mais fácil de adaptar** para o português, pois o objetivo é justamente testar *over-defense* com prompts benignos que contêm palavras-gatilho. Criar amostras benignas em português com essas palavras é relativamente simples e tem alto valor prático para avaliar guardrails em contextos lusófonos.



## 6. WAInjectBench

**🔗 Link:** https://github.com/Norrrrrrr-lyn/WAInjectBench

**Existe PT-BR?** Não. O WAInjectBench cobre 6 tipos de ataques em duas modalidades — texto e imagem — com dados benignos e maliciosos armazenados em formato JSONL. Não há menção a idiomas além do inglês.

**Dificuldade de criar versão PT-BR:** 🔴 **Alta**

É o benchmark mais complexo do grupo por combinar modalidades de texto e imagem. A parte textual pode ser traduzida, mas os **ataques em imagem** (texto injetado em screenshots de páginas web, por exemplo) exigiriam re-geração das imagens com conteúdo em português — o que representa um esforço significativo de produção de dados.



