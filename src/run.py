"""Ask the agent a question and watch it work.

    python -m src.run "How many days past due is INV-1004 as of 2026-10-12?"
"""
import argparse

from src.agent import run_agent
from src.tools import load_invoices, make_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    from src.llm import LocalChatModel
    result = run_agent(args.question, LocalChatModel(args.model), make_tools(load_invoices()), max_steps=args.max_steps)
    for i, s in enumerate(result.steps, 1):
        print(f"[{i}] model: {s.raw}")
        if s.observation:
            print(f"    tool : {s.observation[:200]}")
    print("\nANSWER:", result.answer if result.finished else f"(no final answer after {len(result.steps)} steps)")


if __name__ == "__main__":
    main()
