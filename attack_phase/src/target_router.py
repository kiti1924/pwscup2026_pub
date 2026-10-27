"""Target-wise composition of already validated attack candidates."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 24, 26, 29)


def _sources(manifest: dict, kind: str) -> dict[str, Path]:
    return {name: Path(value) for name, value in manifest["sources"][kind].items()}


def route_mia(manifest: dict) -> pd.DataFrame:
    sources = _sources(manifest, "mia")
    default = manifest["routing"]["mia"]["default"]
    overrides = manifest["routing"]["mia"].get("target_overrides", {})
    frames = {name: pd.read_csv(path) for name, path in sources.items()}
    base = frames[default].copy()
    if list(map(str, base.columns[1:])) != [str(t) for t in TARGETS]:
        raise ValueError("MIA target columns do not match the official 21 targets")
    for target in TARGETS:
        source = frames[overrides.get(str(target), default)]
        if list(base["record_id"]) != list(source["record_id"]):
            raise ValueError("MIA record_id order differs between sources")
        base[str(target)] = source[str(target)].to_numpy()
    return base


def route_aia(manifest: dict) -> pd.DataFrame:
    sources = _sources(manifest, "aia")
    default = manifest["routing"]["aia"]["default"]
    overrides = manifest["routing"]["aia"].get("target_overrides", {})
    frames = {name: pd.read_csv(path) for name, path in sources.items()}
    base = frames[default].copy()
    expected = set(zip(base["target_k"], base["challenge_row_id"]))
    for target in TARGETS:
        key = str(target)
        source = frames[overrides.get(key, default)]
        if set(zip(source["target_k"], source["challenge_row_id"])) != expected:
            raise ValueError("AIA challenge key set differs between sources")
        mask = base["target_k"].eq(target)
        source_rows = source.loc[source["target_k"].eq(target)].sort_values("challenge_row_id")
        base.loc[mask, "time_hat"] = source_rows["time_hat"].to_numpy()
    return base
