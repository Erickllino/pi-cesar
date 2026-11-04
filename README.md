# PiGuard — revisão de prompts

Interface de revisão humana dos prompts públicos do PiGuard. Os dados de origem ficam em `Data/*.json` (versionados); as avaliações humanas e os resultados históricos do Jev ficam em `db/*.sqlite3` (ignorados pelo Git). O juiz não pode ser executado pelo site.

## Estrutura

```text
Data/                 # JSONs públicos usados pela interface
db/                   # SQLite local; nunca publicar no Git
app.py                # entrada Flask usada pela Vercel
src/web_app.py        # login e API protegida do site compartilhado
src/review_app.py     # lógica de revisão e servidor local legado
src/judge.py          # leitura dos resultados históricos do Jev
src/comparison.py     # comparação entre rótulos
src/review_static/    # páginas e arquivos estáticos
src/test_*.py         # testes
```

## Uso local

Requer Python 3.13 ou superior. Use `uv sync --python 3.13`. Sem `DATABASE_URL`, o servidor local legado usa SQLite:

```bash
python3 src/review_app.py
```

Abra <http://127.0.0.1:8765>. Para usar outras pastas, passe `--data-dir` (JSONs) e `--db-dir` (bancos SQLite). Uma instalação nova cria os bancos locais em `db/` automaticamente; para manter as avaliações anteriores, copie os bancos antigos para `db/` **sem colocá-los no Git**.

Na revisão, cada prompt tem um único veredito compartilhado: a última avaliação salva prevalece. Para BIPIA e NotInject sem campo `label`, o rótulo original é inferido pelo nome do arquivo (BIPIA = `1`/malicious; NotInject = `0`/non-malicious), sem modificar os JSONs ou preencher a revisão humana. A página `/compare` mostra resultados históricos do Jev, se `db/jev_reviews.sqlite3` estiver disponível. Rotas de execução do juiz e a página `/judge` não são disponibilizadas.

**Não exponha `src/review_app.py` na internet:** esse servidor legado não exige senha. Para testar localmente a versão protegida, configure `DATABASE_URL`, `REVIEW_PASSWORD` e `SESSION_SECRET` em `.env` e rode `uv run --env-file .env python app.py` (abra <http://127.0.0.1:8765>). Nesse modo local apenas, o cookie pode trafegar por HTTP; na Vercel ele exige HTTPS.

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

O `app.py` na raiz é o ponto de entrada Flask detectado pela Vercel. Na Vercel, importe este repositório com **Root Directory `./`**, preset Python/Flask e escolha `feature/review-site-vercel` como branch de produção em **Project → Settings → Environments → Production → Branch Tracking**. Em **Project → Settings → Environment Variables**, defina para o ambiente Production:

- `DATABASE_URL`: **connection string PostgreSQL com pool** copiada de **Neon → Connect**, começando com `postgresql://` (não é a URL HTTPS `/rest/v1`). A string do ambiente local não é enviada automaticamente à Vercel.
- `REVIEW_PASSWORD`: senha forte compartilhada com a equipe, com pelo menos 16 caracteres.
- `SESSION_SECRET`: segredo independente, aleatório, com pelo menos 32 caracteres. Gere um com `uv run python -c 'import secrets; print(secrets.token_urlsafe(48))'` no seu terminal.

Não use o mesmo valor para senha e segredo; nunca cole esses valores no repositório, no chat ou em `vercel.json`. Se também usar deploys Preview, configure as variáveis para esse ambiente ou deixe os previews indisponíveis por padrão. **Depois de salvar as variáveis, faça Redeploy do branch de produção**. Confirme que `/login` abre, que `/api/meta` não é acessível sem login, e que as 107 revisões e 78.314 avaliações históricas aparecem após entrar. As páginas, JS e APIs só são entregues após autenticação; a sessão dura até 12 horas e trocar a senha invalida as sessões antigas.

Ainda é necessário agendar o backup externo do PostgreSQL a cada dois dias. Não publique `db/` nem segredos no Git.
