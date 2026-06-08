1. Low-Resource Languages Jailbreak GPT-4 — Yong, Menghini & Bach (NeurIPS 2023 SoLaR). [arXiv:2310.02446]
O paper fundacional da sua tese. Mostra que traduzir entradas inseguras para línguas de menor recurso contorna as proteções do GPT-4, que engaja com o conteúdo e dá respostas acionáveis 79% das vezes — à altura de ataques de jailbreak estado-da-arte. → Seções 1 e 6 (motivação + caso pt-BR). arXiv


2. Multilingual Jailbreak Challenges in Large Language Models — Deng et al. (ICLR 2024). [arXiv:2310.06474]
Introduz o dataset MultiJail e separa cenários "não-intencional" vs. "intencional". Achado-chave: línguas de baixo recurso têm cerca de três vezes mais chance de gerar conteúdo nocivo que as de alto recurso. → Seções 3 e 6 (metodologia de dataset + evidência de degradação). arxiv


3. All Languages Matter: On the Multilingual Safety of LLMs — Wang et al. (ACL 2024 Findings).
Apresenta o XSAFETY, primeiro benchmark de segurança multilíngue em larga escala; mostra que os LLMs têm desempenho de segurança significativamente menor em línguas não-inglesas. → Seção 5 (avaliação/benchmarks). Nature


4. The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts — Shen et al. (2024). [arXiv:2401.13136]
Disseca por que a segurança falha fora do inglês — útil para a fundamentação teórica da Seção 6. → Seção 6.


5. Prompt Injection Attacks on LLMs: A Survey of Attack Methods, Root Causes, and Defense Strategies — Geng et al. (CMC 2026). [DOI:10.32604/cmc.2025.074081]
Seu survey-âncora. Note que ele já tem uma seção sobre dimensões cross-lingual e defende detecção comportamental agnóstica de língua, treino adversarial cross-lingual e datasets de segurança multilíngues como frentes de pesquisa. → Seções 4 e 5 (taxonomia + defesas). ScienceDirect


6. Multilingual Prompt Injection Attacks Detection — Abbasi et al. (SSRN 2025). [abstract 5244151]
PI direto + multilíngue, o mais próximo do seu desenho: avalia 19 modelos pré-treinados e traduz datasets do inglês para o espanhol para medir robustez cross-lingual, cobrindo injeções diretas, prompts ofuscados e jailbreak. → Seções 3 e 5 (é praticamente um molde metodológico). SSRN


7. MIPIAD: Multilingual Indirect Prompt Injection Attack Defense — (2026). [arXiv:2605.07269]
Relevante porque é construído sobre o BIPIA (que você já traduziu) e avalia o gap cross-lingual inglês–bangla em injeção indireta. → Seção 4 (injeção indireta) + Seção 5 (defesa). arXiv


8. Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing — Theocharopoulos et al. (2025). [arXiv:2512.23684]
Injeção indireta "escondida" em documentos, com variação por língua: injeções em inglês, japonês e chinês mudam substancialmente notas e decisões de aceite/rejeição, enquanto as em árabe quase não têm efeito — exemplo concreto de que o idioma do ataque importa. → Seção 4 (indireto/agêntico). arXiv


9. Multilingual Jailbreaking of LLMs Using Low-Resource Languages — Marx & Dunaiski (2026). [arXiv:2605.18239]
Metodologicamente rico para o seu plano de coleta nativa: usa multi-turno e red-teaming humano com falantes nativos de línguas africanas de baixo recurso, descobrindo que ataques de tradução em turno único foram ineficazes enquanto conversas multi-turno tiveram taxas altas. → Seções 6 e 7 (dados nativos vs. tradução). arXiv


10. Adversarial Versification in Portuguese as a Jailbreak Operator in LLMs — Queiroz, UFJF (2025). [arXiv:2512.15353]
O item mais próximo de pt-BR. Argumenta que a ausência de avaliações em português — língua de alta complexidade morfossintática e mais de 250 milhões de falantes — constitui uma lacuna crítica. → Seção 6 (justamente a lacuna que você quer preencher). arxiv

Bônus pt-BR / local (vale citar, especialmente o primeiro, que é da sua própria casa):

SecBERT (Amorim, TCC CIn/UFPE) — classificação de prompts de jailbreak em português com BERTimbau; posicionado como um avanço na segurança de LLMs em português. É um TCC, então tier menor, mas é da UFPE — boa conexão local e mostra que o tema já germina aí. Ufpe
MiJaBench [arXiv:2601.04389] — inclui testes em português para checar consistência cross-lingual de modos de falha.
Jailbreaking and Mitigation of Vulnerabilities in LLMs [arXiv:2410.15236] — survey com uma taxonomia que inclui explicitamente jailbreak multilíngue; bom para comparar com a árvore do Geng et al.