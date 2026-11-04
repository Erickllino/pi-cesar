# PiGuard — revisão de prompts

Interface de revisão humana dos prompts públicos do PiGuard. Os dados de origem ficam em `Data/*.json` (versionados); as avaliações humanas e os resultados históricos do Jev ficam em `db/*.sqlite3` (ignorados pelo Git). O juiz não pode ser executado pelo site.

## Estrutura

```text
Data/                 # JSONs públicos usados pela interface
db/                   # SQLite local; nunca publicar no Git
src/review_app.py     # servidor e API de revisão
src/judge.py          # leitura dos resultados históricos do Jev
src/comparison.py     # comparação entre rótulos
src/review_static/    # páginas e arquivos estáticos
src/test_*.py         # testes
```

## Uso local

Requer Python 3.13 ou superior, sem dependências adicionais:

```bash
python3 src/review_app.py
```

Abra <http://127.0.0.1:8765>. Para usar outras pastas, passe `--data-dir` (JSONs) e `--db-dir` (bancos SQLite). Uma instalação nova cria os bancos locais em `db/` automaticamente; para manter as avaliações anteriores, copie os bancos antigos para `db/` **sem colocá-los no Git**.

Na revisão, cada prompt tem um único veredito compartilhado: a última avaliação salva prevalece. Para BIPIA e NotInject sem campo `label`, o rótulo original é inferido pelo nome do arquivo (BIPIA = `1`/malicious; NotInject = `0`/non-malicious), sem modificar os JSONs ou preencher a revisão humana. A página `/compare` mostra resultados históricos do Jev, se `db/jev_reviews.sqlite3` estiver disponível. Rotas de execução do juiz e a página `/judge` não são disponibilizadas.

Para rodar os testes:

```bash
PYTHONPATH=src python3 -m unittest discover -s src -p 'test_*.py'
```

## Hospedagem

**Este servidor local ainda não está pronto para a Vercel.** Os JSONs públicos já podem acompanhar o código, mas as funções da Vercel não mantêm bancos SQLite persistentes entre execuções e o `ThreadingHTTPServer` precisa ser adaptado para o runtime de funções. Para uma revisão compartilhada online, é necessário um banco externo para avaliações humanas (e para resultados históricos do Jev, caso se queira manter a comparação), autenticação da equipe e backups externos a cada dois dias. Não publique `db/` nem segredos no Git e não exponha o servidor local diretamente na internet.
