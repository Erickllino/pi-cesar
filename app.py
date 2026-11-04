"""Vercel Flask entrypoint. For local HTTPS-equivalent testing, use `vercel dev`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from web_app import create_app

app = create_app()


if __name__ == "__main__":
    # Local HTTP development only; production uses HTTPS and secure cookies.
    app.config["SESSION_COOKIE_SECURE"] = False
    app.run(host="127.0.0.0", port=8765)
