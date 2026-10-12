import json

from src.scoring import answer_ok, contains
from src.tasks import NOT_FOUND, build_tasks
from src.tools import load_invoices


def test_committed_tasks_match_the_generator():
    with open("data/tasks.jsonl", encoding="utf-8") as f:
        committed = [json.loads(line) for line in f]
    assert committed == build_tasks(load_invoices())


def test_numbers_match_as_whole_numbers_only():
    assert contains("The fee is $345.00", "345")
    assert contains("total: 15,875.50 USD", "15875.5")
    assert not contains("It is 2026-10-18", "18")
    assert not contains("about 1500", "150")
    assert contains("There are 4 open invoices", "4")
    assert not contains("about 4.5 invoices", "4")
    assert not contains("3450", "345")


def test_answer_ok_requires_all_expected_and_any_phrase():
    task = {"expect": ["Acme Corp", "Globex"], "expect_any": []}
    assert answer_ok(task, "Globex and Acme Corp")
    assert not answer_ok(task, "Globex")
    assert not answer_ok(task, None)
    nf = {"expect": [], "expect_any": NOT_FOUND}
    assert answer_ok(nf, "Invoice INV-2000 was not found.")
    assert not answer_ok(nf, "The amount is 500")


def test_every_task_is_answerable_from_the_tools():
    tasks = build_tasks(load_invoices())
    assert len(tasks) == 16 and len({t["id"] for t in tasks}) == 16
    assert all(t["expect"] or t["expect_any"] for t in tasks)
