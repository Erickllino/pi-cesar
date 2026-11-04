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

Requer Python 3.13 ou superior. Instale a dependência PostgreSQL com `pip install -e .` (ou `uv sync`). Sem `DATABASE_URL`, o modo local usa apenas a biblioteca padrão e continua com SQLite:

```bash
python3 src/review_app.py
```

Abra <http://127.0.0.1:8765>. Para usar outras pastas, passe `--data-dir` (JSONs) e `--db-dir` (bancos SQLite). Uma instalação nova cria os bancos locais em `db/` automaticamente; para manter as avaliações anteriores, copie os bancos antigos para `db/` **sem colocá-los no Git**.

Na revisão, cada prompt tem um único veredito compartilhado: a última avaliação salva prevalece. Para BIPIA e NotInject sem campo `label`, o rótulo original é inferido pelo nome do arquivo (BIPIA = `1`/malicious; NotInject = `0`/non-malicious), sem modificar os JSONs ou preencher a revisão humana. A página `/compare` mostra resultados históricos do Jev, se `db/jev_reviews.sqlite3` estiver disponível. Rotas de execução do juiz e a página `/judge` não são disponibilizadas.

Para rodar os testes:

```bash
PYTHONPATH=src python3 -m unittest discover -s src -p 'test_*.py'
```

## Banco compartilhado (PostgreSQL)

Crie um banco PostgreSQL externo, por exemplo no Neon, e configure a URL de conexão com TLS como variável de ambiente **`DATABASE_URL`**. Não a coloque no repositório nem a envie por mensagem. O modo PostgreSQL cria as tabelas `categories`, `reviews` e `jev_reviews` na primeira inicialização; os prompts públicos permanecem nos JSONs versionados. Em produção, use a URL de conexão com pool do provedor para evitar excesso de conexões. Todas as pessoas compartilham um único veredito por prompt, e a última gravação vence.

Para importar as avaliações locais e os resultados já gerados pelo Jev **uma vez, antes de abrir o site para a equipe**:

```bash
# Com DATABASE_URL configurada no ambiente e os SQLite em db/:
python3 src/migrate_sqlite.py --apply
```

O importador só lê `db/reviews.sqlite3` e `db/jev_reviews.sqlite3`, insere apenas IDs presentes nos JSONs atuais e **não substitui** registros que já existam no PostgreSQL; repetir o comando é seguro. `db/judge_reviews.sqlite3` é legado e não é usado pelo site. Depois da importação, mantenha os arquivos locais como cópia inicial e confira os totais antes de compartilhar a página. Com `DATABASE_URL` configurada, `python3 src/review_app.py` usa o PostgreSQL; sem ela, usa SQLite local.

**Backup:** exporte o PostgreSQL para um local seguro fora da Vercel a cada dois dias (por exemplo, com `pg_dump` e agendamento no sistema ou no provedor). Uma cópia local de `db/` não inclui as revisões feitas posteriormente no banco remoto. Proteja também a credencial e os arquivos de backup; o histórico de restauração do plano gratuito não substitui esse backup.

## Hospedagem

**A integração do banco não é um deploy completo na Vercel.** Os JSONs públicos já acompanham o código e `DATABASE_URL` permite armazenamento compartilhado, mas `ThreadingHTTPServer` ainda precisa ser adaptado para funções da Vercel e é necessária autenticação para a equipe. Não publique `db/` nem segredos no Git e não exponha o servidor local diretamente na internet.
