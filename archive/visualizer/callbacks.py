"""Reactive callbacks for the Dash visualizer: load file -> filter ->
N-hop expand -> build Cytoscape elements + summary stats. Node clicks
populate a modal overlay (driven by separate open/close callbacks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dash import Input, Output, State, ctx, no_update

from backend.app.helpers.ontology_pruning import (
    collect_related_class_iris,
    safe_get_iri,
)
from visualizer.data import (
    compute_stats,
    file_size_mb,
    is_too_large,
    load_ontology,
    resolve_custom_path,
)

NODE_TYPES = {"class", "object_property", "data_property", "individual"}

SHAPE_BY_TYPE = {
    "class": "ellipse",
    "object_property": "diamond",
    "data_property": "round-rectangle",
    "individual": "triangle",
}


def _short_label(entity: dict[str, Any], fallback_iri: str) -> str:
    labels = entity.get("labels") or []
    for lab in labels:
        if isinstance(lab, str) and lab.strip():
            return lab.strip()
        if isinstance(lab, dict):
            value = lab.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    name = entity.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    # Last resort: trim the IRI to the local fragment.
    if "#" in fallback_iri:
        return fallback_iri.rsplit("#", 1)[-1] or fallback_iri
    if "/" in fallback_iri:
        return fallback_iri.rsplit("/", 1)[-1] or fallback_iri
    return fallback_iri


def _matches_name_filter(label: str, iri: str, needle: str) -> bool:
    if not needle:
        return True
    needle_low = needle.lower()
    return needle_low in label.lower() or needle_low in iri.lower()


def _filter_class_iris(classes_dict: dict, name_filter: str, hops: int) -> set[str]:
    """Pick seed IRIs by name-filter match, then BFS-expand `hops` away
    using the existing build_class_graph / collect_related helpers."""
    seeds = []
    for iri, entity in classes_dict.items():
        label = _short_label(entity, iri)
        if _matches_name_filter(label, iri, name_filter):
            seeds.append(iri)
    if not seeds:
        return set()
    if hops == 0:
        return set(seeds)
    return collect_related_class_iris(classes_dict, seeds, max_hops=hops)


def _build_class_nodes_edges(
    classes_dict: dict,
    keep_iris: set[str],
) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_edge: set[tuple[str, str, str]] = set()
    for iri in keep_iris:
        entity = classes_dict.get(iri)
        if entity is None:
            continue
        label = _short_label(entity, iri)
        nodes.append({
            "data": {
                "id": iri,
                "label": label,
                "type": "class",
                "shape": SHAPE_BY_TYPE["class"],
                "color": "#3f6db8",
                "iri": iri,
                "kind": "class",
            }
        })
        # subClassOf edges (only within the visible set).
        for sc in entity.get("superclasses", []):
            sc_iri = safe_get_iri(sc)
            if sc_iri and sc_iri in keep_iris:
                key = (iri, sc_iri, "subClassOf")
                if key not in seen_edge:
                    seen_edge.add(key)
                    edges.append({
                        "data": {
                            "source": iri,
                            "target": sc_iri,
                            "type": "subClassOf",
                            "label": "subClassOf",
                        }
                    })
    return nodes, edges


def _build_property_nodes_edges(
    properties_dict: dict,
    classes_visible: set[str],
    type_value: str,
    include_class_edges: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Render properties as their own nodes plus domain -> property -> range
    arrows (when classes_visible covers the endpoints)."""
    nodes: list[dict] = []
    edges: list[dict] = []
    for iri, entity in properties_dict.items():
        label = _short_label(entity, iri)
        nodes.append({
            "data": {
                "id": iri,
                "label": label,
                "type": type_value,
                "shape": SHAPE_BY_TYPE[type_value],
                "color": "#62a85a" if type_value == "object_property" else "#d99a3a",
                "iri": iri,
                "kind": type_value,
            }
        })
        if not include_class_edges:
            continue
        for d in entity.get("domain", []):
            d_iri = safe_get_iri(d)
            if not d_iri or d_iri not in classes_visible:
                continue
            edges.append({
                "data": {
                    "source": d_iri,
                    "target": iri,
                    "type": "domain-range",
                    "label": "domain",
                }
            })
        for r in entity.get("range", []):
            r_iri = safe_get_iri(r)
            if not r_iri or r_iri not in classes_visible:
                continue
            edges.append({
                "data": {
                    "source": iri,
                    "target": r_iri,
                    "type": "domain-range",
                    "label": "range",
                }
            })
    return nodes, edges


def _build_individual_nodes_edges(
    instances_dict: dict,
    classes_visible: set[str],
) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    for iri, entity in instances_dict.items():
        label = _short_label(entity, iri)
        nodes.append({
            "data": {
                "id": iri,
                "label": label,
                "type": "individual",
                "shape": SHAPE_BY_TYPE["individual"],
                "color": "#9e5fc6",
                "iri": iri,
                "kind": "individual",
            }
        })
        for t in entity.get("types", []):
            t_iri = safe_get_iri(t)
            if t_iri and t_iri in classes_visible:
                edges.append({
                    "data": {
                        "source": iri,
                        "target": t_iri,
                        "type": "rdf:type",
                        "label": "type",
                    }
                })
    return nodes, edges


def _render(
    loaded: dict,
    types: list[str],
    name_filter: str,
    hops: int,
) -> tuple[list[dict], str, str]:
    """Produce (elements, stats_text, status_text) for the current selection."""
    classes_dict = loaded.get("classes_dict", {})
    obj_props = loaded.get("object_properties_dict", {})
    data_props = loaded.get("data_properties_dict", {})
    instances = loaded.get("instances_dict", {})

    classes_visible = _filter_class_iris(classes_dict, name_filter, hops)

    elements: list[dict] = []
    if "class" in types:
        nodes, edges = _build_class_nodes_edges(classes_dict, classes_visible)
        elements.extend(nodes)
        elements.extend(edges)
    if "object_property" in types:
        nodes, edges = _build_property_nodes_edges(
            obj_props, classes_visible, "object_property",
            include_class_edges="class" in types,
        )
        elements.extend(nodes)
        elements.extend(edges)
    if "data_property" in types:
        nodes, edges = _build_property_nodes_edges(
            data_props, classes_visible, "data_property",
            include_class_edges="class" in types,
        )
        elements.extend(nodes)
        elements.extend(edges)
    if "individual" in types:
        nodes, edges = _build_individual_nodes_edges(
            instances, classes_visible if "class" in types else set()
        )
        elements.extend(nodes)
        elements.extend(edges)

    stats = compute_stats(loaded)
    stats_text = (
        f"Classes:           {stats['classes']}\n"
        f"Object properties: {stats['object_properties']}\n"
        f"Data properties:   {stats['data_properties']}\n"
        f"Individuals:       {stats['instances']}\n\n"
        f"Visible nodes:     {len([e for e in elements if 'source' not in e['data']])}\n"
        f"Visible edges:     {len([e for e in elements if 'source' in e['data']])}"
    )

    node_count = len([e for e in elements if "source" not in e["data"]])
    if node_count > 500:
        status = (
            f"WARNING: rendering {node_count} nodes -- the graph may be slow. "
            "Add a name filter to narrow the view."
        )
    else:
        status = ""
    return elements, stats_text, status


def register_callbacks(app) -> None:
    @app.callback(
        Output("graph", "elements"),
        Output("stats", "children"),
        Output("status", "children"),
        Output("custom-path-error", "children"),
        Input("file-dropdown", "value"),
        Input("custom-path", "value"),
        Input("type-filter", "value"),
        Input("search-button", "n_clicks"),
        Input("name-filter", "n_submit"),
        Input("hops-slider", "value"),
        State("name-filter", "value"),
    )
    def _on_inputs(file_value, custom_text, types, _clicks, _submits, hops, name_filter):
        # Search trigger: re-renders happen on Search-button click, name-filter
        # Enter press (n_submit), or any of the file/type/hops changes. The
        # name-filter VALUE comes through as State so typing it doesn't re-render
        # on every keystroke.

        # Pick which path wins: custom path overrides dropdown when valid.
        path: Path | None = None
        custom_error = ""
        if custom_text:
            resolved = resolve_custom_path(custom_text)
            if resolved is None:
                custom_error = "Path must point to an existing .owl/.rdf/.ttl file."
            else:
                path = resolved
        if path is None and file_value:
            path = Path(file_value)

        if path is None:
            return [], "Select an ontology file to begin.", "", custom_error

        if is_too_large(path):
            return (
                [],
                f"File is {file_size_mb(path):.0f} MB -- too large to safely "
                "parse in this viewer. Use the CLI `merge` to produce a smaller "
                "derived ontology and view that instead.",
                "Aborted load (file too large).",
                custom_error,
            )

        loaded = load_ontology(path)
        if loaded is None:
            return [], "Could not load file.", "Load failed.", custom_error

        elements, stats_text, status = _render(loaded, types or [], name_filter or "", int(hops))
        return elements, stats_text, status, custom_error

    @app.callback(
        Output("graph", "layout"),
        Input("layout-name", "value"),
        Input("fit-button", "n_clicks"),
        State("layout-name", "value"),
    )
    def _on_layout(layout_name, _fit_clicks, current):
        # Both layout-dropdown changes and Fit-button clicks rebuild the layout
        # config. animate=False keeps the rebuild snappy on dense graphs.
        chosen = layout_name or current or "cose"
        return {"name": chosen, "animate": False}

    @app.callback(
        Output("modal-backdrop", "style"),
        Output("modal-title", "children"),
        Output("modal-body", "children"),
        Input("graph", "tapNodeData"),
        Input("modal-close", "n_clicks"),
        State("file-dropdown", "value"),
        State("custom-path", "value"),
        State("modal-backdrop", "style"),
    )
    def _modal(node_data, _close_clicks, file_value, custom_text, current_style):
        triggered = ctx.triggered_id
        base_style = dict(current_style or {})
        if triggered == "modal-close":
            base_style["display"] = "none"
            return base_style, "", ""
        if not node_data:
            # No tap yet -- keep modal hidden, leave content empty.
            base_style["display"] = "none"
            return base_style, "", ""
        path = resolve_custom_path(custom_text or "") or (Path(file_value) if file_value else None)
        if path is None:
            return no_update, no_update, no_update
        loaded = load_ontology(path)
        if loaded is None:
            return no_update, no_update, no_update
        kind = node_data.get("kind") or node_data.get("type")
        iri = node_data.get("iri") or node_data.get("id")
        if not iri:
            return no_update, no_update, no_update
        dict_name = {
            "class": "classes_dict",
            "object_property": "object_properties_dict",
            "data_property": "data_properties_dict",
            "individual": "instances_dict",
        }.get(kind, "classes_dict")
        entity = loaded.get(dict_name, {}).get(iri)
        title = node_data.get("label") or iri
        base_style["display"] = "flex"
        if entity is None:
            return base_style, title, f"{iri}\n(entity not found in {dict_name})"
        body = json.dumps(_summarize_entity(entity), indent=2, default=str)
        return base_style, title, body


def _summarize_entity(entity: dict) -> dict:
    """Trim huge fields (raw_axiom_triples) so the detail card stays readable."""
    keep = {
        "iri",
        "name",
        "namespace",
        "labels",
        "comments",
        "descriptions",
        "annotations",
        "superclasses",
        "equivalent_to",
        "disjoints",
        "domain",
        "range",
        "inverse_property",
        "characteristics",
        "types",
        "sources",
    }
    return {k: v for k, v in entity.items() if k in keep and v not in (None, [], {})}
