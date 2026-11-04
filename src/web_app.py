"""Password-protected WSGI app for the shared review site."""

import hashlib
import hmac
import json
import os
import secrets
from datetime import timedelta

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, session

from comparison import FILTERS, filter_prompts, summary
from review_app import DEFAULT_DATA, DEFAULT_DB, STATIC, create_stores


LOGIN_PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PiGuard · Acesso</title><style>
body{font-family:system-ui,sans-serif;background:#10121b;color:#f4f4f7;min-height:100vh;display:grid;place-items:center;margin:0}
main{background:#1e2230;padding:2rem;border-radius:14px;width:min(370px,calc(100vw - 4rem))}
h1{font-size:1.4rem}label{display:block;margin:1.5rem 0 .5rem}input,button{box-sizing:border-box;width:100%;padding:.85rem;border-radius:7px;font:inherit}
button{margin-top:1rem;background:#8257e6;color:white;border:0;cursor:pointer}input{border:1px solid #858795}
p{color:#ffb3b3}small{color:#c4c6d4}
</style></head><body><main><h1>PiGuard · Revisão de prompts</h1><small>Acesso reservado à equipe</small>
{% if error %}<p role="alert">Senha incorreta.</p>{% endif %}
<form method="post" action="/login"><label for="password">Senha da equipe</label>
<input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
<button type="submit">Entrar</button></form></main></body></html>"""

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/compare": ("compare.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/compare.js": ("compare.js", "application/javascript; charset=utf-8"),
    "/compare.css": ("compare.css", "text/css; charset=utf-8"),
}


def create_app(store=None, judge_store=None, password=None, secret_key=None, secure_cookie=True):
    password = password if password is not None else os.environ.get("REVIEW_PASSWORD", "")
    secret_key = secret_key if secret_key is not None else os.environ.get("SESSION_SECRET", "")
    if not isinstance(password, str) or len(password) < 8:
        raise RuntimeError("Configure REVIEW_PASSWORD com pelo menos 8 caracteres antes de iniciar o site.")
    if not isinstance(secret_key, str) or len(secret_key) < 32:
        raise RuntimeError("Configure SESSION_SECRET com pelo menos 32 caracteres antes de iniciar o site.")
    if store is None or judge_store is None:
        if not os.environ.get("DATABASE_URL"):
            raise RuntimeError("Configure DATABASE_URL para o site compartilhado; SQLite local não é persistente na Vercel.")
        store, judge_store = create_stores(DEFAULT_DATA, DEFAULT_DB, os.environ["DATABASE_URL"])

    app = Flask(__name__)
    app.secret_key = secret_key
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SECURE=secure_cookie,
                      SESSION_COOKIE_SAMESITE="Lax", PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
                      MAX_CONTENT_LENGTH=50000)
    auth_value = hmac.new(secret_key.encode(), password.encode(), hashlib.sha256).hexdigest()

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.before_request
    def require_login():
        if request.path == "/login":
            return None
        if not hmac.compare_digest(session.get("auth", ""), auth_value):
            return (jsonify(error="Acesso não autorizado."), 401) if request.path.startswith("/api/") else redirect("/login")
        if request.method == "POST" and not hmac.compare_digest(
            request.headers.get("X-CSRF-Token", ""), session.get("csrf", "-"),
        ):
            return jsonify(error="Sessão inválida. Atualize a página e tente novamente."), 403
        return None

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template_string(LOGIN_PAGE, error=False)
        entered = request.form.get("password", "")
        if hmac.compare_digest(entered, password):
            session.clear()
            session["auth"] = auth_value
            session["csrf"] = secrets.token_urlsafe(32)
            session.permanent = True
            return redirect("/")
        return render_template_string(LOGIN_PAGE, error=True), 401

    @app.get("/api/session")
    def session_info():
        return jsonify(csrf=session["csrf"])

    @app.get("/api/meta")
    def meta():
        return jsonify(datasets=store.datasets, categories=store.categories(), stats=store.stats())

    @app.get("/api/prompts")
    def prompts():
        page = pagination()
        if page is None:
            return jsonify(error="Paginação inválida."), 400
        items = store.filtered(*(request.args.get(key, "") for key in ("dataset", "status", "label", "search")))
        offset, limit = page
        return jsonify(total=len(items), items=items[offset:offset + limit])

    @app.get("/api/export")
    def export():
        return send_file_bytes(store.export_csv(), "revisoes_prompts.csv")

    @app.post("/api/reviews")
    def save_review():
        body = json_body()
        if body is None:
            return jsonify(error="Envie um objeto JSON válido."), 400
        try:
            result = store.save_review(body.get("id"), body.get("verdict"), body.get("categories", []), body.get("notes", ""))
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(result), 201

    @app.post("/api/categories")
    def add_category():
        body = json_body()
        if body is None:
            return jsonify(error="Envie um objeto JSON válido."), 400
        try:
            result = store.add_category(body.get("name", ""))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(result), 201

    @app.get("/api/compare/meta")
    def compare_meta():
        return jsonify(datasets=store.datasets, summary=summary(store.prompts, judge_store.reviews(), store.reviews()))

    @app.get("/api/compare/prompts")
    def compare_prompts():
        page = pagination()
        if page is None:
            return jsonify(error="Paginação inválida."), 400
        dataset, status, label, search = (request.args.get(key, "") for key in ("dataset", "status", "label", "search"))
        if dataset not in ("", *store.datasets) or status not in FILTERS or label not in ("", "0", "1"):
            return jsonify(error="Filtro inválido."), 400
        offset, limit = page
        return jsonify(filter_prompts(store.prompts, judge_store.reviews(), store.reviews(),
                                      dataset, status, label, search, offset, limit))

    @app.get("/")
    @app.get("/compare")
    @app.get("/app.js")
    @app.get("/style.css")
    @app.get("/compare.js")
    @app.get("/compare.css")
    def static_page():
        filename, mime = STATIC_FILES[request.path]
        return send_file(STATIC / filename, mimetype=mime)

    return app


def pagination():
    try:
        return max(0, int(request.args.get("offset", "0"))), min(50, max(1, int(request.args.get("limit", "12"))))
    except ValueError:
        return None


def json_body():
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else None


def send_file_bytes(content, filename):
    from io import BytesIO

    return send_file(BytesIO(content), mimetype="text/csv; charset=utf-8", as_attachment=True, download_name=filename)
