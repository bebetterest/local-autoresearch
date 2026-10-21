"""Export a run with path redaction, credential checks and compressed streams."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from report import report
from research import api_key

ROOT_FILES = {
    "manifest.json",
    "summary.json",
    "best.py",
    "validation.json",
    "notes.json",
    "annotations.json",
    "continuation.json",
    "progress.json",
    "final-test-started.json",
    "final-test.json",
    "baseline-final-test.json",
}
ATTEMPT_FILES = {
    "request.json",
    "response.json",
    "response.txt",
    "solver.py",
    "evaluation.json",
    "decision.json",
    "stream.jsonl",
    "diagnostic.json",
    "http-error.txt",
}


def export(source, target):
    manifest = json.loads((source / "manifest.json").read_text())
    # The first circle run predates saved server details and used rec authentication.
    key = api_key(manifest["inference"] if "inference" in manifest else {})
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    paths = [source / name for name in ROOT_FILES if (source / name).exists()]
    paths += [
        p
        for d in source.iterdir()
        if d.is_dir() and d.name.isdigit()
        for p in d.iterdir()
        if p.name in ATTEMPT_FILES
    ]
    for path in sorted(paths):
        original = path.read_text()
        if (key is not None and key in original) or re.search(
            r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}", original
        ):
            raise RuntimeError(f"Credential detected in {path.name}; export stopped")
        text = original.replace(str(Path.home()), "<HOME>")
        text = re.sub(
            r"/private/var/folders/[^\s\"'\\]+/flint-packing-[^/\s\"'\\]+", "<WORKSPACE>", text
        )
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "stream.jsonl":
            destination = destination.with_suffix(".jsonl.gz")
            destination.write_bytes(gzip.compress(text.encode(), mtime=0))
        else:
            destination.write_text(text)
        entries.append(
            {
                "file": str(destination.relative_to(target)),
                "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "export_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "path_redacted": text != original,
            }
        )
    (target / "export.json").write_text(
        json.dumps(
            {
                "transformations": [
                    "Home directory → <HOME>",
                    "Candidate temporary directory → <WORKSPACE>",
                    "Raw stream losslessly gzip-compressed after path redaction",
                ],
                "files": entries,
            },
            indent=2,
        )
        + "\n"
    )
    report(target)
    print(f"Exported {len(entries)} files to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    export(args.source.resolve(), args.target.resolve())
