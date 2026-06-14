"""Dash app construction: server, layout, callbacks."""

from __future__ import annotations

import dash

from visualizer.callbacks import register_callbacks
from visualizer.data import discover_owl_files
from visualizer.layouts import build_layout


def create_app() -> dash.Dash:
    app = dash.Dash(__name__, title="Ontology Viewer", suppress_callback_exceptions=True)
    files = discover_owl_files()
    app.layout = build_layout(files)
    register_callbacks(app)
    return app


def run() -> None:
    app = create_app()
    # Dash 3.x renamed `app.run_server` -> `app.run`; both still work on 2.18+.
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    run()
