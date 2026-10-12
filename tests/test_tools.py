import json

import pytest

from src.tools import ToolError, calculator, load_invoices, make_tools


@pytest.fixture(scope="module")
def tools():
    return make_tools(load_invoices())


def test_calculator_arithmetic():
    assert calculator("2300 * 0.15") == "345"
    assert calculator("(100 + 250.5) / 2") == "175.25"
    assert calculator("-3 + 10") == "7"
    assert calculator("1,250 + 3,340") == "4590"


@pytest.mark.parametrize("bad", ["__import__('os').system('echo hi')", "2 ** 10", "abs(-1)", "open('x')", "1 +"])
def test_calculator_rejects_anything_but_arithmetic(bad):
    with pytest.raises(ToolError):
        calculator(bad)


def test_calculator_division_by_zero():
    with pytest.raises(ToolError, match="division by zero"):
        calculator("1 / 0")


def test_lookup_invoice(tools):
    inv = json.loads(tools["lookup_invoice"].fn(invoice_id="inv-1005"))
    assert inv["customer"] == "Umbrella Ltd" and inv["amount"] == 15200.0


def test_lookup_missing_invoice_raises(tools):
    with pytest.raises(ToolError, match="no invoice with id INV-9999"):
        tools["lookup_invoice"].fn(invoice_id="INV-9999")


def test_list_invoices_filters(tools):
    open_ids = [i["id"] for i in json.loads(tools["list_invoices"].fn(status="open"))]
    assert open_ids == ["INV-1002", "INV-1003", "INV-1004", "INV-1008"]
    acme = json.loads(tools["list_invoices"].fn(customer="acme corp"))
    assert len(acme) == 3
    assert len(json.loads(tools["list_invoices"].fn())) == 10


def test_days_between(tools):
    assert tools["days_between"].fn(start="2026-09-24", end="2026-10-12") == "18"
    assert tools["days_between"].fn(start="2026-10-12", end="2026-10-05") == "-7"
    with pytest.raises(ToolError):
        tools["days_between"].fn(start="yesterday", end="2026-10-12")
