"""Derive every bar size from the collector's 1min and D files. Loops forever, rebuilding a symbol
only when its source file is newer than its output. Outputs go to nse_derived/ (not uploaded, can
always be rebuilt). Run from Trading/:  .venv_fetch/bin/python -u nse-collector/derive.py  [once]"""
import sys
import time
from pathlib import Path

import pandas as pd

SRC, OUT = Path("nse_data"), Path("nse_derived")
AGG = dict(open="first", high="max", low="min", close="last", volume="sum")
INTRADAY = ["3min", "5min", "10min", "15min", "30min", "60min", "240min"]
DAILY = {"1W": "W-FRI", "1M": "MS"}


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(path.with_suffix(".tmp"), index=False, compression="zstd")
    path.with_suffix(".tmp").replace(path)  # atomic


def stale(src, path):
    return not path.exists() or src.stat().st_mtime > path.stat().st_mtime


def derive_intraday(src, sym):
    # pre-open bars (09:08 etc) are not tradable, keep the 09:15-15:29 session only, same as research loaders
    d = pd.read_parquet(src).set_index("timestamp").between_time("09:15", "15:29")
    for rule in INTRADAY:
        write(d.resample(rule, origin="start_day", offset="9h15min").agg(AGG).dropna(), OUT / rule / f"{sym}.parquet")


def derive_daily(src, sym):
    d = pd.read_parquet(src).set_index("timestamp")
    for name, rule in DAILY.items():
        write(d.resample(rule).agg(AGG).dropna(), OUT / name / f"{sym}.parquet")


def sweep():
    n = 0
    for src in sorted((SRC / "1min").glob("*.parquet")):
        if stale(src, OUT / INTRADAY[-1] / src.name):
            derive_intraday(src, src.stem)
            n += 1
    for src in sorted((SRC / "D").glob("*.parquet")):
        if stale(src, OUT / "1M" / src.name):
            derive_daily(src, src.stem)
            n += 1
    return n


if __name__ == "__main__":
    while True:
        t = time.time()
        n = sweep()
        log(f"sweep: {n} files rebuilt in {time.time() - t:.0f}s")
        if "once" in sys.argv:
            break
        time.sleep(600)  # ponytail: poll every 10 min, no filesystem-event dependency
