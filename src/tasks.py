"""Evaluation tasks with ground truth computed from data/invoices.json, so the expected answers cannot drift from the data.

    python -m src.tasks            # rewrite data/tasks.jsonl (CI fails if the committed file differs)
"""
import json
import os
from datetime import date

from src.tools import load_invoices

NOT_FOUND = ["not found", "no invoice", "does not exist", "doesn't exist", "no record", "not exist", "could not find",
             "couldn't find", "cannot find", "can't find", "unable to find"]


def fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def build_tasks(invoices: list[dict]) -> list[dict]:
    by_id = {i["id"]: i for i in invoices}
    open_ = [i for i in invoices if i["status"] == "open"]
    paid = [i for i in invoices if i["status"] == "paid"]
    as_of = date(2026, 10, 12)
    tasks = []

    def add(category, question, expect=None, expect_any=None, must_call=()):
        tasks.append({"id": f"t{len(tasks) + 1:02d}", "category": category, "question": question,
                      "expect": expect or [], "expect_any": expect_any or [], "must_call": list(must_call)})

    add("lookup", "What is the amount of invoice INV-1005?", [fmt(by_id["INV-1005"]["amount"])], must_call=["lookup_invoice"])
    add("lookup", "Which customer is invoice INV-1008 for?", [by_id["INV-1008"]["customer"]], must_call=["lookup_invoice"])
    add("lookup", "What is the due date of INV-1003?", [by_id["INV-1003"]["due"]], must_call=["lookup_invoice"])
    add("lookup", "What is the status of INV-1006?", [by_id["INV-1006"]["status"]], must_call=["lookup_invoice"])

    add("filter", "How many invoices are currently open?", [str(len(open_))], must_call=["list_invoices"])
    disputed = sorted({i["customer"] for i in invoices if i["status"] == "disputed"})
    add("filter", "Which customers have a disputed invoice?", disputed, must_call=["list_invoices"])
    add("filter", "How many invoices does Acme Corp have in total?",
        [str(sum(i["customer"] == "Acme Corp" for i in invoices))], must_call=["list_invoices"])

    add("calc", "What is the total amount of all open invoices?", [fmt(sum(i["amount"] for i in open_))],
        must_call=["list_invoices", "calculator"])
    add("calc", "If a 15% late fee is added to INV-1004, how much is the fee?", [fmt(by_id["INV-1004"]["amount"] * 0.15)],
        must_call=["lookup_invoice", "calculator"])
    add("calc", "What is the combined amount of INV-1001 and INV-1007?",
        [fmt(by_id["INV-1001"]["amount"] + by_id["INV-1007"]["amount"])], must_call=["lookup_invoice", "calculator"])
    add("calc", "What is the average amount of the paid invoices?", [fmt(sum(i["amount"] for i in paid) / len(paid))],
        must_call=["list_invoices", "calculator"])

    add("date", "As of 2026-10-12, how many days past its due date is INV-1004?",
        [str((as_of - date.fromisoformat(by_id["INV-1004"]["due"])).days)], must_call=["lookup_invoice", "days_between"])
    add("date", "As of 2026-10-12, how many days past its due date is INV-1003?",
        [str((as_of - date.fromisoformat(by_id["INV-1003"]["due"])).days)], must_call=["lookup_invoice", "days_between"])
    add("date", "Counting from 2026-10-12, how many days until INV-1008 is due?",
        [str((date.fromisoformat(by_id["INV-1008"]["due"]) - as_of).days)], must_call=["lookup_invoice", "days_between"])

    add("not-found", "What is the amount of invoice INV-2000?", expect_any=NOT_FOUND, must_call=["lookup_invoice"])
    add("not-found", "Which customer is invoice INV-1042 for?", expect_any=NOT_FOUND, must_call=["lookup_invoice"])
    return tasks


def main(path: str = os.path.join("data", "tasks.jsonl")):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in build_tasks(load_invoices()):
            f.write(json.dumps(t) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
