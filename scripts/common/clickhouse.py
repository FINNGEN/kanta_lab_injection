"""ClickHouse access shared by the injection scripts, and the default engine-output location.

Every query runs as `clickhouse -q` on the engine-output parquet (the data VM has the
`clickhouse` binary on PATH). Parameters are passed as ClickHouse query parameters
(`{name:String}` in the query, `name=...` here), never formatted into the SQL.
"""

import subprocess
import sys
from pathlib import Path

# Engine runs sit in <ENGINE_DIR>/<prefix>_<YYYY_MM_DD>/<prefix>_<YYYY_MM_DD>.parquet
ENGINE_DIR = Path("/mnt/disks/data/kanta/engine")


def clickhouse(query, **params):
    """Run one query with `clickhouse -q` and return its stdout."""
    cmd = ["clickhouse", "-q", query]
    for k, v in params.items():
        cmd.append(f"--param_{k}={v}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("ClickHouse query failed")
    return proc.stdout


def latest_engine_output(engine_dir=ENGINE_DIR):
    """Most recent engine output: the run whose folder name carries the latest date
    (ties: the newest file), or None if there is none."""
    engine_dir = Path(engine_dir)
    runs = [(d.name[-10:], (d / f"{d.name}.parquet").stat().st_mtime, d / f"{d.name}.parquet")
            for d in engine_dir.glob("*_[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]")
            if (d / f"{d.name}.parquet").is_file()]
    return max(runs)[2] if runs else None
