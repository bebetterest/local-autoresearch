"""Execute one candidate inside the OS sandbox; acceptance happens in the parent."""

import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

cpu_seconds = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
spec = importlib.util.spec_from_file_location("candidate", "solver.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
inputs = json.loads(Path("input.json").read_text())
solution = candidate.solve(
    inputs["problem"], inputs["seed"], max(0, cpu_seconds - time.process_time() - 0.5)
)
Path("solution.json").write_text(json.dumps(solution, allow_nan=False))
