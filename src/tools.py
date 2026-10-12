"""The tools the agent may call. Every tool takes keyword arguments and returns a short string.

A tool that cannot do what was asked raises ToolError; the agent loop turns that message into an observation so the model
can correct itself instead of the whole run crashing.
"""
import ast
import json
import operator
import os
from dataclasses import dataclass
from datetime import date

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "invoices.json")


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: callable


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ToolError("only numbers and + - * / ( ) are allowed")


def calculator(expression: str) -> str:
    """Evaluate arithmetic with an AST walk, never eval(), so model-written input cannot run code."""
    try:
        value = _eval_node(ast.parse(str(expression).replace(",", ""), mode="eval").body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {expression!r}") from exc
    return f"{round(value, 6):.6f}".rstrip("0").rstrip(".")


def load_invoices(path: str = DATA_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_tools(invoices: list[dict]) -> dict[str, Tool]:
    by_id = {inv["id"]: inv for inv in invoices}

    def lookup_invoice(invoice_id: str) -> str:
        inv = by_id.get(str(invoice_id).strip().upper())
        if inv is None:
            raise ToolError(f"no invoice with id {invoice_id}")
        return json.dumps(inv)

    def list_invoices(status: str | None = None, customer: str | None = None) -> str:
        rows = [i for i in invoices
                if (status is None or i["status"] == str(status).lower())
                and (customer is None or i["customer"].lower() == str(customer).lower())]
        return json.dumps([{k: i[k] for k in ("id", "customer", "amount", "status")} for i in rows])

    def days_between(start: str, end: str) -> str:
        try:
            return str((date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days)
        except ValueError as exc:
            raise ToolError("dates must look like YYYY-MM-DD") from exc

    tools = [
        Tool("lookup_invoice", 'lookup_invoice(invoice_id) - full record of one invoice: id, customer, amount, status '
                               '(paid/open/disputed), issued date, due date', lookup_invoice),
        Tool("list_invoices", 'list_invoices(status?, customer?) - id, customer, amount and status of invoices, optionally '
                              'filtered by status and/or exact customer name', list_invoices),
        Tool("days_between", "days_between(start, end) - number of days from start to end, dates as YYYY-MM-DD "
                             "(negative if end is earlier)", days_between),
        Tool("calculator", "calculator(expression) - evaluates arithmetic such as '2300 * 0.15 + 12'. Use it for every calculation",
             calculator),
    ]
    return {t.name: t for t in tools}
