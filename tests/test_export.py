import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools import export_run


def test_export_redacts_paths_and_preserves_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(export_run, "api_key", lambda inference: "k" * 32)
    monkeypatch.setattr(export_run, "report", lambda root: (root / "report.html").write_text(""))
    source, target = tmp_path / "source", tmp_path / "target"
    attempt = source / "001"
    attempt.mkdir(parents=True)
    (source / "manifest.json").write_text("{}")
    paired_baseline = '{"valid": false, "trials": []}\n'
    (source / "baseline-final-test.json").write_text(paired_baseline)
    raw = json.dumps({"text": f"File {Path.home()}/code/example.py failed"}) + "\n"
    (attempt / "stream.jsonl").write_text(raw)
    (attempt / "diagnostic.json").write_text(raw)
    (attempt / "http-error.txt").write_text(raw)
    export_run.export(source, target)
    exported = gzip.decompress((target / "001/stream.jsonl.gz").read_bytes()).decode()
    assert exported == raw.replace(str(Path.home()), "<HOME>")
    assert (target / "001/diagnostic.json").read_text() == exported
    assert (target / "001/http-error.txt").read_text() == exported
    assert (target / "baseline-final-test.json").read_text() == paired_baseline
    record = next(
        entry
        for entry in json.loads((target / "export.json").read_text())["files"]
        if entry["file"] == "001/stream.jsonl.gz"
    )
    assert record["original_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert record["export_sha256"] == hashlib.sha256(exported.encode()).hexdigest()


def test_export_refuses_local_token(tmp_path, monkeypatch):
    token = "secret-for-test-1234567890"
    monkeypatch.setattr(export_run, "api_key", lambda inference: token)
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"leaked": token}))
    with pytest.raises(RuntimeError, match="Credential detected"):
        export_run.export(source, tmp_path / "target")
