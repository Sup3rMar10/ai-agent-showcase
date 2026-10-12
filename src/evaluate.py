"""Run the agent on every task and write per-task results plus a summary.

    python -m src.evaluate --model Qwen/Qwen2.5-1.5B-Instruct          # -> results/agent.Qwen2.5-1.5B-Instruct.json
"""
import argparse
import json
import os
import time
from collections import defaultdict

from src.agent import run_agent
from src.scoring import answer_ok
from src.tools import load_invoices, make_tools


def read_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> dict:
    def rate(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else None
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "n_tasks": len(rows),
        "accuracy": rate(rows, "correct"),
        "used_expected_tools": rate(rows, "used_expected_tools"),
        "finished_rate": rate(rows, "finished"),
        "avg_steps": sum(r["n_steps"] for r in rows) / len(rows),
        "avg_errors_per_task": sum(r["n_errors"] for r in rows) / len(rows),
        "avg_seconds_per_task": sum(r["seconds"] for r in rows) / len(rows),
        "by_category": {c: {"n": len(rs), "accuracy": rate(rs, "correct")} for c, rs in sorted(by_cat.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", default=os.path.join("data", "tasks.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    from src.llm import LocalChatModel
    llm = LocalChatModel(args.model)
    tools = make_tools(load_invoices())
    out = args.out or os.path.join("results", f"agent.{args.model.split('/')[-1]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for task in read_tasks(args.tasks)[: args.limit]:
        t0 = time.time()
        res = run_agent(task["question"], llm, tools, max_steps=args.max_steps)
        rows.append({
            "id": task["id"], "category": task["category"], "question": task["question"], "answer": res.answer,
            "correct": answer_ok(task, res.answer), "finished": res.finished, "n_steps": len(res.steps),
            "n_errors": res.n_errors, "tools_called": res.tools_called,
            "used_expected_tools": all(t in res.tools_called for t in task["must_call"]),
            "seconds": round(time.time() - t0, 1),
            "trace": [{"raw": s.raw, "observation": s.observation[:300], "ok": s.ok} for s in res.steps]})
        print(f"{task['id']} {'OK ' if rows[-1]['correct'] else 'BAD'} steps={rows[-1]['n_steps']} "
              f"err={rows[-1]['n_errors']} {rows[-1]['seconds']}s  {res.answer!r}", flush=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:  # saved after every task so an interrupted run keeps its progress
            json.dump({"model": args.model, "summary": summarize(rows), "tasks": rows}, f, indent=2)
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
