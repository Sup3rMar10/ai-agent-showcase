"""A small tool-using agent: the model replies with one JSON action per turn, the loop runs it and feeds back the result.

    question -> model -> {"action": "lookup_invoice", "args": {...}} -> tool -> Observation -> model -> ... -> {"action": "final", ...}

The loop is deliberately plain Python so every decision is visible: it parses the reply, validates it against the tool
registry, runs the tool, and stops on a final answer or after `max_steps`. Mistakes (bad JSON, unknown tool, wrong arguments,
tool errors) become observations so the model gets a chance to correct itself; they are counted so they can be measured.
"""
import inspect
import json
from dataclasses import dataclass, field

from src.tools import Tool, ToolError

MAX_OBSERVATION_CHARS = 1500

SYSTEM_TEMPLATE = """You are an invoice assistant. Answer the user's question by calling tools, then give a short final answer.

Reply with exactly one JSON object per turn and nothing else.
To call a tool: {{"action": "<tool name>", "args": {{...}}}}
To finish:      {{"action": "final", "answer": "<short answer>"}}

After each tool call you will receive an Observation. Rules:
- Never guess invoice data or do arithmetic in your head. Look it up and use the calculator.
- If a tool reports that something does not exist, say so in the final answer instead of guessing.
- Dates are YYYY-MM-DD. Amounts are in USD.

Tools:
{tools}

Example
User: What is the total of INV-2001 and INV-2002?
{{"action": "lookup_invoice", "args": {{"invoice_id": "INV-2001"}}}}
Observation: {{"id": "INV-2001", "customer": "Foo Co", "amount": 100, "status": "open", "issued": "2026-01-01", "due": "2026-01-31"}}
{{"action": "lookup_invoice", "args": {{"invoice_id": "INV-2002"}}}}
Observation: {{"id": "INV-2002", "customer": "Bar Co", "amount": 250.5, "status": "paid", "issued": "2026-01-05", "due": "2026-02-04"}}
{{"action": "calculator", "args": {{"expression": "100 + 250.5"}}}}
Observation: 350.5
{{"action": "final", "answer": "The total is 350.50 USD."}}"""


class ParseError(Exception):
    pass


@dataclass
class Step:
    raw: str
    action: str | None
    args: dict
    observation: str
    ok: bool  # False when the reply was malformed, the tool was unknown or misused, or the tool raised an error


@dataclass
class Result:
    question: str
    answer: str | None
    steps: list[Step] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.answer is not None

    @property
    def tools_called(self) -> list[str]:
        return [s.action for s in self.steps if s.ok and s.action not in (None, "final")]

    @property
    def n_errors(self) -> int:
        return sum(not s.ok for s in self.steps)


def system_prompt(tools: dict[str, Tool]) -> str:
    return SYSTEM_TEMPLATE.format(tools="\n".join(f"- {t.description}" for t in tools.values()))


def parse_action(text: str) -> dict:
    """Pull the first JSON object out of a model reply (models sometimes add prose or a code fence around it)."""
    start = text.find("{")
    if start == -1:
        raise ParseError("reply contained no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str):
        raise ParseError('JSON must be an object with a string "action"')
    return obj


def run_agent(question: str, llm, tools: dict[str, Tool], max_steps: int = 8) -> Result:
    messages = [{"role": "system", "content": system_prompt(tools)}, {"role": "user", "content": question}]
    result = Result(question=question, answer=None)
    for _ in range(max_steps):
        raw = llm.chat(messages).split("\nObservation:")[0].strip()  # small models sometimes write the tool's reply themselves
        messages.append({"role": "assistant", "content": raw})
        try:
            act = parse_action(raw)
        except ParseError as exc:
            obs = f'Error: {exc}. Reply with one JSON object such as {{"action": "final", "answer": "..."}}.'
            result.steps.append(Step(raw, None, {}, obs, ok=False))
            messages.append({"role": "user", "content": "Observation: " + obs})
            continue
        name = act["action"]
        if name == "final":
            result.answer = str(act.get("answer", "")).strip()
            result.steps.append(Step(raw, "final", {}, "", ok=True))
            return result
        args = act.get("args") or {}
        ok, obs = _call_tool(tools, name, args)
        result.steps.append(Step(raw, name, args if isinstance(args, dict) else {}, obs, ok))
        messages.append({"role": "user", "content": "Observation: " + obs[:MAX_OBSERVATION_CHARS]})
    return result


def _call_tool(tools: dict[str, Tool], name: str, args) -> tuple[bool, str]:
    tool = tools.get(name)
    if tool is None:
        return False, f"Error: unknown tool {name!r}. Available: {', '.join(tools)}."
    if not isinstance(args, dict):
        return False, 'Error: "args" must be a JSON object.'
    try:
        inspect.signature(tool.fn).bind(**args)
    except TypeError as exc:
        return False, f"Error: bad arguments for {name}: {exc}."
    try:
        return True, tool.fn(**args)
    except ToolError as exc:
        return False, f"Error: {exc}."
