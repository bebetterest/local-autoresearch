"""Fixed CPU problems. Scores are always maximized."""

from importlib import import_module
from pathlib import Path

TASKS = {
    path.name: import_module(f"tasks.{path.name}")
    for path in sorted(Path(__file__).parent.iterdir())
    if path.is_dir() and (path / "baseline.py").is_file()
}
