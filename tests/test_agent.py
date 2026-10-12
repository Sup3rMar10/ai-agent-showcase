import json

from src.agent import parse_action, run_agent, system_prompt
from src.llm import ScriptedLLM
from src.tools import load_invoices, make_tools

TOOLS = make_tools(load_invoices())


def call(action, **args):
    return json.dumps({"action": action, "args": args})


def final(answer):
    return json.dumps({"action": "final", "answer": answer})


def test_parse_action_tolerates_prose_and_code_fences():
    assert parse_action('Sure!\n```json\n{"action": "final", "answer": "x"}\n```')["answer"] == "x"


def test_multi_step_run_feeds_observations_back():
    llm = ScriptedLLM([call("lookup_invoice", invoice_id="INV-1004"), call("calculator", expression="2300 * 0.15"), final("345")])
    res = run_agent("late fee?", llm, TOOLS)
    assert res.answer == "345" and res.finished
    assert res.tools_called == ["lookup_invoice", "calculator"]
    assert res.n_errors == 0
    last_prompt = llm.calls[-1]
    assert last_prompt[-1] == {"role": "user", "content": "Observation: 345"}
    assert "Acme Corp" in last_prompt[-3]["content"]  # the invoice record from step 1 is still in the model's context


def test_malformed_reply_is_reported_and_the_model_can_recover():
    llm = ScriptedLLM(["I think the answer is 5", final("5")])
    res = run_agent("q", llm, TOOLS)
    assert res.answer == "5"
    assert res.n_errors == 1
    assert "no JSON object" in llm.calls[1][-1]["content"]


def test_unknown_tool_and_bad_arguments_become_observations():
    llm = ScriptedLLM([call("delete_everything"), call("lookup_invoice", wrong_name="x"), call("lookup_invoice", invoice_id="INV-9"),
                       final("INV-9 does not exist")])
    res = run_agent("q", llm, TOOLS)
    assert [s.ok for s in res.steps] == [False, False, False, True]
    assert "unknown tool" in res.steps[0].observation
    assert "bad arguments" in res.steps[1].observation
    assert "no invoice with id INV-9" in res.steps[2].observation
    assert res.answer == "INV-9 does not exist"


def test_gives_up_after_max_steps():
    llm = ScriptedLLM([call("list_invoices")] * 10)
    res = run_agent("q", llm, TOOLS, max_steps=3)
    assert not res.finished and res.answer is None and len(res.steps) == 3
    assert len(llm.calls) == 3


def test_system_prompt_lists_every_tool():
    prompt = system_prompt(TOOLS)
    assert all(name in prompt for name in TOOLS)


def test_reply_is_cut_where_the_model_starts_inventing_observations():
    fake = call("lookup_invoice", invoice_id="INV-1006") + '\nObservation: {"customer": "Made Up"}\n' + final("made up")
    llm = ScriptedLLM([fake, final("disputed")])
    res = run_agent("q", llm, TOOLS)
    assert res.answer == "disputed"
    assert "Made Up" not in json.dumps(llm.calls[1])
