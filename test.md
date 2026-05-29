 Tive uma ideia que queria saber oque o senhor acha dela.

Pensando no trabalho de prompt injection, percebi que não existe nenhuma biblioteca open-source com foco em segurança contra esses ataques para pipelines de agentes LLM e especialmente nenhuma com suporte a português.

A ideia seria criar uma lib em Rust que funciona como uma camada de guarda em pipelines de agentes: detecta tentativas de prompt injection antes que o input chegue ao LLM, usando detectores em cascata (regex/heurística → similaridade por embedding → LLM judge). O diferencial técnico principal é usar o sistema de tipos do Rust para tornar a segurança obrigatória em nível de compilação algo impossível em Python.

Os datasets em PT-BR do nosso trabalho no CISSA poderiam virar um benchmark público junto com a lib, o que daria visibilidade acadêmica pra pesquisa e contribuiria pro ecossistema de segurança de LLMs.

Isso serviria como open-source com potencial de publicação, poderia ser um paper sobre o benchmark em PT-BR + a metodologia dos detectores.
