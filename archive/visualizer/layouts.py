"""Dash UI layout: sidebar (file picker, filters, layout, stats) + main pane
(cytoscape graph) + modal popup overlay for node details."""

from __future__ import annotations

from collections.abc import Sequence

import dash_cytoscape as cyto
from dash import dcc, html

from visualizer.data import DiscoveredFile

NODE_TYPE_OPTIONS = [
    {"label": "Classes", "value": "class"},
    {"label": "Object properties", "value": "object_property"},
    {"label": "Data properties", "value": "data_property"},
    {"label": "Individuals", "value": "individual"},
]

LAYOUT_OPTIONS = [
    {"label": "Force-directed (cose)", "value": "cose"},
    {"label": "Tree (breadth-first)", "value": "breadthfirst"},
    {"label": "Concentric rings", "value": "concentric"},
    {"label": "Circle", "value": "circle"},
    {"label": "Grid", "value": "grid"},
    {"label": "Random", "value": "random"},
]
DEFAULT_LAYOUT = "cose"

CYTO_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "data(color)",
            "shape": "data(shape)",
            "width": 18,
            "height": 18,
            "font-size": 9,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "color": "#333",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": 1,
            "line-color": "#999",
            "target-arrow-color": "#999",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "label": "data(label)",
            "font-size": 8,
            "color": "#777",
            "text-rotation": "autorotate",
        },
    },
    {"selector": "edge[type = 'subClassOf']",     "style": {"line-color": "#7da7d9"}},
    {"selector": "edge[type = 'domain-range']",   "style": {"line-color": "#92c47d"}},
    {"selector": "edge[type = 'rdf:type']",       "style": {"line-color": "#c89bd4"}},
    {"selector": "node[type = 'class']",            "style": {"background-color": "#3f6db8"}},
    {"selector": "node[type = 'object_property']",  "style": {"background-color": "#62a85a"}},
    {"selector": "node[type = 'data_property']",    "style": {"background-color": "#d99a3a"}},
    {"selector": "node[type = 'individual']",       "style": {"background-color": "#9e5fc6"}},
]


def _file_options(files: Sequence[DiscoveredFile]) -> list[dict]:
    """Format DiscoveredFile entries for the dcc.Dropdown options field."""
    return [
        {
            "label": f"[{f.group}] {f.label}  ({f.size_bytes / 1024:.0f} KB)",
            "value": str(f.path),
        }
        for f in files
    ]


def build_layout(files: Sequence[DiscoveredFile]) -> html.Div:
    return html.Div(
        style={"display": "flex", "fontFamily": "sans-serif", "height": "100vh"},
        children=[
            # Sidebar
            html.Div(
                style={
                    "width": "340px",
                    "padding": "16px",
                    "borderRight": "1px solid #ddd",
                    "overflowY": "auto",
                    "fontSize": "13px",
                },
                children=[
                    html.H3("Ontology Viewer", style={"marginTop": 0}),
                    html.Label("File:"),
                    dcc.Dropdown(
                        id="file-dropdown",
                        options=_file_options(files),
                        placeholder="Select an ontology file...",
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label("Custom path:"),
                    dcc.Input(
                        id="custom-path",
                        type="text",
                        placeholder="/absolute/path/to/file.owl",
                        debounce=True,
                        style={"width": "100%"},
                    ),
                    html.Div(id="custom-path-error", style={"color": "#c00", "fontSize": "11px"}),
                    html.Hr(),
                    html.Label("Show node types:"),
                    dcc.Checklist(
                        id="type-filter",
                        options=NODE_TYPE_OPTIONS,
                        value=["class"],
                        labelStyle={"display": "block"},
                    ),
                    html.Hr(),
                    html.Label("Graph layout:"),
                    dcc.Dropdown(
                        id="layout-name",
                        options=LAYOUT_OPTIONS,
                        value=DEFAULT_LAYOUT,
                        clearable=False,
                    ),
                    html.Hr(),
                    html.Label("Search:"),
                    html.Div(
                        style={"display": "flex", "gap": "6px", "marginTop": "4px"},
                        children=[
                            dcc.Input(
                                id="name-filter",
                                type="text",
                                placeholder="e.g. Person",
                                debounce=False,
                                n_submit=0,
                                style={"flex": 1, "minWidth": 0},
                            ),
                            html.Button("Search", id="search-button", n_clicks=0),
                        ],
                    ),
                    html.Div(
                        "Substring match on label or IRI · case-insensitive · "
                        "press Enter or click Search",
                        style={"fontSize": "11px", "color": "#666", "marginTop": "4px"},
                    ),
                    html.Br(),
                    html.Label("Hops from matches:"),
                    dcc.Slider(
                        id="hops-slider",
                        min=0,
                        max=3,
                        step=1,
                        value=1,
                        marks={i: str(i) for i in range(4)},
                    ),
                    html.Hr(),
                    html.Div(id="stats", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),
                    html.Hr(),
                    html.Button("Fit / reset view", id="fit-button", n_clicks=0),
                    html.Div(id="status", style={"marginTop": "12px", "color": "#666"}),
                ],
            ),
            # Main pane: graph fills the available space; node details show
            # in a modal overlay (see below) instead of a bottom panel.
            html.Div(
                style={"flex": 1, "display": "flex", "flexDirection": "column"},
                children=[
                    cyto.Cytoscape(
                        id="graph",
                        layout={"name": DEFAULT_LAYOUT, "animate": False},
                        style={"flex": "1", "width": "100%", "height": "100%"},
                        stylesheet=CYTO_STYLESHEET,
                        elements=[],
                    ),
                ],
            ),
            # Modal overlay for node details. Hidden by default; opens on
            # node tap, closes via the Close button.
            html.Div(
                id="modal-backdrop",
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100vw",
                    "height": "100vh",
                    "background": "rgba(0,0,0,0.45)",
                    "zIndex": 1000,
                    "justifyContent": "center",
                    "alignItems": "center",
                },
                children=html.Div(
                    id="modal-content",
                    style={
                        "background": "#fff",
                        "padding": "20px",
                        "borderRadius": "6px",
                        "maxWidth": "640px",
                        "width": "90%",
                        "maxHeight": "80vh",
                        "overflowY": "auto",
                        "boxShadow": "0 8px 24px rgba(0,0,0,0.2)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "10px",
                                "gap": "10px",
                            },
                            children=[
                                html.Strong(
                                    id="modal-title",
                                    style={"fontFamily": "sans-serif", "fontSize": "14px"},
                                ),
                                html.Button("Close", id="modal-close", n_clicks=0),
                            ],
                        ),
                        html.Pre(
                            id="modal-body",
                            style={
                                "fontFamily": "monospace",
                                "fontSize": "12px",
                                "whiteSpace": "pre-wrap",
                                "margin": 0,
                            },
                        ),
                    ],
                ),
            ),
        ],
    )
