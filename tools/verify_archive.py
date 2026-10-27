"""Verify this code-only archive without requiring competition inputs."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/"provenance/code_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    known = {v["path"] for v in files}
    for item in files:
        path = ROOT/item["path"]
        ast.parse(path.read_text(encoding="utf-8"), filename=item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["export_sha256"]:
            raise ValueError("Archived code hash mismatch: "+item["path"])
        for dependency in item["local_imports"]:
            if dependency not in known:
                raise ValueError("Unresolved local import: "+dependency)
    for path in ROOT.rglob("*.py"):
        if any(p in {".git", ".venv", "__pycache__"} for p in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
    ledger = json.loads((ROOT/"provenance/submissions.json").read_text(encoding="utf-8"))
    for item in ledger["submissions"]:
        for code in item["code"]:
            if code not in known:
                raise ValueError("Submission references missing code: "+code)
    for path in ROOT.rglob("*.md"):
        if any(p in {".git", ".venv"} for p in path.parts):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent/target).exists():
                raise ValueError(f"Broken document link: {path.name}: {target}")
    checks = json.loads((ROOT/"provenance/evidence.json").read_text(encoding="utf-8"))["historical_code_hash_checks"]
    if not all(v["matches_recorded"] and v["recorded_sha256"] == v["source_sha256"] for v in checks):
        raise ValueError("Historical hash evidence failed")
    print(f"PASS: {len(files)} archived code files, {len(ledger['submissions'])} submission records, "
          f"{len(checks)} historical code hashes, local dependencies and document links")


if __name__ == "__main__":
    main()
