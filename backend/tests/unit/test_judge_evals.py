"""LLM-as-judge evaluations: the parts that need no database or model."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from backend.app.evals import judge_evals as je
from backend.app.services.prompts import eval_definition_judge, eval_typing_judge
from backend.app.services.prompts_deepseek import (
    deepseek_eval_definition_judge,
    deepseek_eval_typing_judge,
)


def test_verdicts_keep_only_valid_answers():
    batch = [{}, {}, {}]
    got = [{"id": 0, "verdict": "Yes", "reason": "fine"}, {"id": 1, "verdict": "maybe"},
           {"id": 7, "verdict": "no"}, {"id": "2", "verdict": "no", "reason": "wrong"}, "x"]
    assert je.parse_verdicts(got, batch) == {0: ("yes", "fine"), 2: ("no", "wrong")}


def test_tally_gives_both_percentages():
    items = [{"verdict": "yes", "k": 1}, {"verdict": "no", "k": 1},
             {"verdict": "no_definition", "k": 0}, {"verdict": "unanswered", "k": 0}]
    t = je.tally(items, {"k1": lambda it: it["k"] == 1})
    assert t["all"] == {"items": 4, "yes": 1, "no": 1, "unanswered": 1, "no_definition": 1,
                        "judged": 2, "yes_pct_of_judged": 50.0, "yes_pct_of_all": 25.0}
    assert t["k1"]["yes_pct_of_all"] == 50.0


def test_snippet_centres_on_the_mention():
    text = "x " * 400 + "Samsung SDI reported revenue. " + "y " * 400
    s = je.snippet(text, "samsung sdi", 60)
    assert "Samsung SDI" in s and s.startswith("...") and len(s) <= 66


class _Router:
    total_cost_usd = 0.0

    def __init__(self, fail_first=False):
        self.calls, self.fail_first = 0, fail_first

    def task_spec(self, task):
        return {"items_per_call": 2, "concurrency": 1, "model": "m"}

    async def chat(self, task, *, system, user):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("boom")
        n = user.count("\n   ") or 1
        ids = [int(line.split(".")[0]) for line in user.splitlines() if line[:1].isdigit()]
        return SimpleNamespace(text=json.dumps(
            {"decisions": [{"id": i, "verdict": "yes", "reason": "ok"} for i in ids][:n + 2]}))


def test_judge_items_sets_verdicts_and_marks_failed_calls():
    items = [{"name": f"e{i}"} for i in range(4)]
    run = asyncio.run(je.judge_items(
        _Router(fail_first=True), "t", items, eval_typing_judge,
        lambda it: {"name": it["name"], "kind": "entity", "class": "C",
                    "class_definition": "", "parents": "", "passage": ""}))
    assert run["failed_calls"] == 1 and run["retried"] == 2
    assert [it["verdict"] for it in items] == ["yes", "yes", "yes", "yes"]  # retried


def _folder(tmp_path: Path) -> Path:
    def cls(name, parents, definition="", **kw):
        iri = f"https://x#{name}"
        return iri, {"name": name, "iri": iri, "labels": [name], "comments": [],
                     "descriptions": [definition] if definition else [],
                     "superclasses": [{"name": p, "iri": f"https://x#{p}"} for p in parents],
                     **kw}
    classes = dict([
        cls("Quantity", ["Thing"], "An amount.", sources=["dc.owl"]),
        cls("Revenue", ["Quantity"], "Income from sales.", sources=["dc.owl"]),
        cls("Organization", ["Thing"], "", sources=["foaf"]),
        cls("Association", ["Organization"], "", sources=["fibo"], annotations={
            je.SKOS_DEFINITION: ["not-for-profit organization owned by its members"]}),
        cls("ChipRevenue", ["Revenue"], "Revenue from chips.",
            semantic_role={"role": "proposed_data_class"}),
    ])
    (tmp_path / "merged.json").write_text(json.dumps({"classes_dict": classes}))
    return tmp_path


def test_class_definitions_are_grouped(tmp_path):
    items = {it["class"]: it for it in je.load_class_definitions(_folder(tmp_path), ["Quantity"])}
    assert items["Revenue"]["concept"] and items["ChipRevenue"]["concept"]
    assert not items["Quantity"]["concept"]  # the root itself is not a concept class
    assert items["ChipRevenue"]["new"] and not items["Revenue"]["new"]
    assert items["Organization"]["verdict"] == "no_definition"
    assert items["ChipRevenue"]["parents"] == "Revenue"
    assert items["Association"]["definition"].startswith("not-for-profit")  # skos:definition


def test_ontology_eval_sends_only_defined_classes(tmp_path):
    router = _Router()
    report, _items = asyncio.run(je.ontology_eval(_folder(tmp_path), router, ["Quantity"]))
    g = report["groups"]
    assert g["all"]["no_definition"] == 1 and g["all"]["yes"] == 4
    assert g["all"]["yes_pct_of_judged"] == 100.0 and g["all"]["yes_pct_of_all"] == 80.0
    assert router.calls == 2  # 4 defined classes, 2 per call


def test_scores_and_text():
    report = {"evaluation": "typing", "model": "m", "cost_usd": 0.01, "failed_calls": 0,
              "groups": je.tally([{"verdict": "yes"}], {})}
    assert je.judge_scores(report)["typing.all.yes_pct_of_all"] == 100.0
    assert "yes/judged 100.0%" in je.format_judge_eval(report)


def test_prompts_list_every_item_and_ask_for_json():
    item = {"id": 0, "name": "Samsung SDI", "kind": "entity", "class": "Organization",
            "class_definition": "", "parents": "Agent", "passage": "Samsung SDI said"}
    for build in (eval_typing_judge, deepseek_eval_typing_judge):
        _system, user = build([item])
        assert "0. name: Samsung SDI" in user and '"verdict"' in user
    _, user = eval_definition_judge([{"id": 3, "class": "Revenue", "definition": "d",
                                      "parents": ""}])
    assert "3. class: Revenue" in user
    assert "ONE JSON" in deepseek_eval_definition_judge([])[0]
