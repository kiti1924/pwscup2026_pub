"""Build reproducible S0/S1 server probes from the target routing manifest."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "attack_phase/src"))
from target_router import route_aia, route_mia, TARGETS
from build_attack_candidates import build_one, DEFAULT_DIST, DEFAULT_TOKEN


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_probe(name: str, manifest_path: Path, output_root: Path, submissions_root: Path,
                token_path: Path, dist_path: Path, validate: bool = True) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe = manifest["probes"][name]
    resolved = json.loads(json.dumps(probe))
    for kind in ("mia", "aia"):
        for key, value in probe["sources"][kind].items():
            resolved["sources"][kind][key] = str((ROOT / value).resolve())
    mia = route_mia(resolved); aia = route_aia(resolved)
    run_name = probe.get("run_name") or ("20260906_server_" + name + ("_v3" if name == "S3_orthogonal_probe" else ""))
    run = output_root / run_name
    if run.exists(): raise FileExistsError(f"refusing to overwrite {run}")
    with tempfile.TemporaryDirectory(prefix="server-probe-") as temp_name:
        temp = Path(temp_name)
        mia_path, aia_path = temp / "F_mia.csv", temp / "F_aia.csv"
        # Preserve source bytes for the no-override S0/S1 anchor probes.
        mia_source = Path(resolved["sources"]["mia"][resolved["routing"]["mia"]["default"]])
        aia_source = Path(resolved["sources"]["aia"][resolved["routing"]["aia"]["default"]])
        if not probe["routing"]["mia"].get("target_overrides"):
            shutil.copyfile(mia_source, mia_path)
        else: mia.to_csv(mia_path, index=False)
        if not probe["routing"]["aia"].get("target_overrides"):
            shutil.copyfile(aia_source, aia_path)
        else: aia.to_csv(aia_path, index=False)
        paired = probe.get("comparison_probe")
        other_aia = "" if not paired else sha(output_root / ("20260906_server_" + paired) / "F_aia.csv")
        meta = build_one(run_name=run_name, method=name,
                         mia_path=mia_path, aia_path=aia_path,
                         other_aia_sha256=other_aia, output_root=output_root,
                         submissions_root=submissions_root, token_path=token_path,
                         dist_path=dist_path, seed=20260906, validate=validate,
                         zip_filename=probe.get("submission_filename"))
    meta["manifest"] = str(manifest_path.relative_to(ROOT)).replace("\\", "/")
    meta["inputs"]["mia"] = probe["sources"]["mia"][resolved["routing"]["mia"]["default"]]
    meta["inputs"]["aia"] = probe["sources"]["aia"][resolved["routing"]["aia"]["default"]]
    meta["routing"] = probe["routing"]
    meta["hypothesis"] = probe["hypothesis"]
    meta["comparison_probe"] = paired
    meta["target_count"] = len(TARGETS)
    meta["probe_name"] = name
    meta["submission_filename"] = probe.get("submission_filename")
    (run / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("probe_name")
    p.add_argument("--manifest", type=Path, default=ROOT / "attack_phase/config/server_probe_manifest.json")
    p.add_argument("--no-validate", action="store_true"); p.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    args = p.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.probe_name not in manifest["probes"]:
            raise ValueError(f"unknown probe in manifest: {args.probe_name}")
        print(json.dumps(build_probe(args.probe_name, args.manifest, ROOT / "attack_phase/outputs", ROOT / "attack_phase/submissions", args.token, DEFAULT_DIST, not args.no_validate), ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
