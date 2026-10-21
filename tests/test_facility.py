import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tasks import facility


def test_facility_weighted_distance(monkeypatch):
    monkeypatch.setattr(
        facility, "PROBLEM", {"points": [[0, 0], [3, 0], [3, 4]], "demand": [1, 2, 3], "depots": 1}
    )
    checked = facility.check([1])
    assert checked["distance"] == (3 + 3 * 4) / 6
    assert checked["score"] == -checked["distance"]


@pytest.mark.parametrize(
    "answer", [[], [0] * 8, list(range(7)), [True, *range(1, 8)], list(range(93, 101))]
)
def test_facility_rejects_wrong_number_duplicate_or_out_of_range_depots(answer):
    with pytest.raises(ValueError):
        facility.check(answer)


def test_copied_task_is_discovered_checked_and_reported_without_core_edits(tmp_path):
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / "tasks"
    package.mkdir()
    for filename in ("__init__.py", "__main__.py"):
        shutil.copyfile(root / "tasks" / filename, package / filename)
    shutil.copytree(
        root / "tasks/facility", package / "my_task", ignore=shutil.ignore_patterns("__pycache__")
    )
    env = {**os.environ, "PYTHONPATH": str(root)}
    subprocess.run(
        [sys.executable, "-m", "tasks", "check", "my_task"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    script = """
import json
from pathlib import Path
from tasks import TASKS
from tasks.my_task.baseline import solve
from report import report
assert set(TASKS) == {'my_task'}
task = TASKS['my_task']
trial = {'valid': True, 'seed': 0, **task.check(solve(task.PROBLEM, 0, 15))}
root = Path('example')
(root / '000').mkdir(parents=True)
(root / 'manifest.json').write_text(json.dumps({'task': 'my_task', 'seeds': [0], 'model': 'flint'}))
(root / '000/evaluation.json').write_text(json.dumps({'valid': True, 'mean_score': trial['score'], 'trials': [trial]}))
(root / '000/decision.json').write_text(json.dumps({'decision': 'baseline'}))
report(root)
assert (root / 'progress.png').stat().st_size > 0
assert 'Mean weighted distance' in (root / 'report.html').read_text()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
