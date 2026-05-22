"""Dash UI layout: sidebar (file picker, filters, stats) + main pane
(cytoscape graph + selected-node detail card)."""

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
                    html.Label("Name filter (substring; matches label or IRI):"),
                    dcc.Input(
                        id="name-filter",
                        type="text",
                        placeholder="e.g. Person",
                        debounce=True,
                        style={"width": "100%"},
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
            # Main pane
            html.Div(
                style={"flex": 1, "display": "flex", "flexDirection": "column"},
                children=[
                    cyto.Cytoscape(
                        id="graph",
                        layout={"name": "cose", "animate": False},
                        style={"flex": "3", "width": "100%"},
                        stylesheet=CYTO_STYLESHEET,
                        elements=[],
                    ),
                    html.Div(
                        id="detail-card",
                        style={
                            "flex": "1",
                            "padding": "12px",
                            "borderTop": "1px solid #ddd",
                            "overflowY": "auto",
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                            "whiteSpace": "pre-wrap",
                        },
                        children="Click a node to see its details.",
                    ),
                ],
            ),
        ],
    )
