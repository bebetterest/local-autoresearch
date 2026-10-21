"""List tasks or check a baseline in the same sandbox used for research."""

import argparse
import json
from pathlib import Path

from runner.evaluate import evaluate
from tasks import TASKS

parser = argparse.ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest="command", required=True)
commands.add_parser("list", help="Show available tasks")
check = commands.add_parser("check", help="Run a task's baseline and independent evaluator")
check.add_argument("task", choices=list(TASKS))
args = parser.parse_args()

if args.command == "list":
    for name, task in TASKS.items():
        print(f"{name:12} {task.__doc__}")
else:
    path = Path(TASKS[args.task].__file__).with_name("baseline.py")
    source = path.read_text()
    result = evaluate(source, task=args.task)
    if result["valid"]:
        print(
            f"{args.task}: valid baseline · score {result['mean_score']:.6g}"
            f" · {result['trials'][0]['cpu_seconds']:.3f} CPU seconds"
        )
    else:
        print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)
