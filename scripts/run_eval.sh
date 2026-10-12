#!/usr/bin/env bash
# Evaluate both local models on the 16 tasks (about 15-30 minutes on a 4-core CPU).
set -e
for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct; do
  python -m src.evaluate --model "$m" 2>&1 | tee "results/$(basename "$m").log"
done
python -m src.report
