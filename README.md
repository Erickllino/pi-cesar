# Tradução Multilíngue do PIArena

Este projeto usa o PIArena exclusivamente como fonte de datasets para
tradução para português brasileiro, espanhol, alemão e árabe. Ele não inclui nem executa
os ataques, defesas, agentes ou avaliadores do repositório upstream.

## Estrutura

```text
Data/PIArena/Original/                 # JSONs de origem, ignorados pelo Git
Data/PIArena/Translated/<modelo>/<idioma>/
scripts/download.py                    # baixa os dados-fonte do Hugging Face
scripts/translate.py                   # executa a tradução
docs/                                  # origem, metodologia e experimentos
```

## Documentação

- [Origem e schema dos dados](docs/PIARENA.md)
- [Metodologia do pipeline de tradução](docs/TRANSLATION_PIPELINE.md)
- [Registro de execuções](docs/EXPERIMENTS.md)

## Instalação rápida

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download.py --revision main
```
