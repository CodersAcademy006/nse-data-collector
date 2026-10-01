"""
nse_collector.py -- NSE data collector (Kotak historical API, consumer key only).

  backfill (Mac, once):  5yr of 1min + D per instrument -> nse_data/{1min,D}/SYM.parquet,
                         pushed to the private HF dataset hourly (only changed files upload),
                         one Kaggle snapshot at the end, then exit. Re-run resumes.
  daily (GitHub Actions): last 5 days of 1min + D for every instrument ->
                         daily/{1min,D}/YYYY/MM/YYYY-MM-DD.parquet (all symbols, `symbol` column) -> HF.
                         Overlapping windows self-heal missed days; dedupe on (symbol, timestamp).
3/5/10/15/30/60min and W/M are not fetched: resample 1min/D locally.
Order: indices, 5-stock basket, Nifty 500, then every other EQ/BE/SM/ST stock.

Mac (from Trading/): PYTHONPATH=. nohup caffeinate -i .venv_fetch/bin/python -u nse-collector/nse_collector.py backfill > logs/collector.log 2>&1 &
"""
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import certifi
import pandas as pd
from huggingface_hub import HfApi
from neo_api_client import NeoAPI

os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # python.org build has no system CA bundle
KEY = os.environ.get("KOTAK_CONSUMER_KEY") or __import__("kotak_bridge").CONSUMER_KEY

OUT = Path("nse_data")  # cwd-relative: Trading/ on the Mac, repo root in Actions
HF_REPO = "Srijan-Upadhyay/nse-ohlcv-kotak"
KAGGLE = "srijan1upadhyay/nse-ohlcv-kotak"
CHUNK = {"1min": 28, "D": 175}
COLS = ["timestamp", "open", "high", "low", "close", "volume"]
BASKET = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "SBIN"]
INDICES = {"NIFTY50": "Nifty 50", "BANKNIFTY": "Nifty Bank"}  # history API wants index names, not tokens
hf = HfApi()

client = NeoAPI(consumer_key=KEY, environment="prod")
gap = 3.0  # seconds between calls, adapts to 429s


def log(m):
    print(f"[{dt.datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)


def universe():
    url = client.scrip_master("nse_cm")
    sm = pd.read_csv(url, low_memory=False, dtype={"pSymbol": str})
    eq = sm[sm["pGroup"].isin(["EQ", "BE", "SM", "ST"])].drop_duplicates("pSymbolName")
    tok = dict(zip(eq["pSymbolName"], eq["pSymbol"]))
    try:
        req = urllib.request.Request("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
                                     headers={"User-Agent": "Mozilla/5.0"})
        n500 = pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=20).read()))["Symbol"].tolist()
    except Exception as e:
        log(f"nifty500 list unavailable ({e}), using scrip master order")
        n500 = []
    order = BASKET + n500 + sorted(eq[eq["pGroup"] == "EQ"]["pSymbolName"]) + sorted(tok)
    out = dict(INDICES)
    for s in order:
        if s in tok and s not in out:
            out[s] = tok[s]
    return out


def fetch(token, interval, fd, td):
    global gap
    for _ in range(12):
        time.sleep(gap)
        try:
            r = client.historical_data(neosymbol=f"nse_cm|{token}", interval=interval, from_date=fd, to_date=td)
        except Exception as e:
            log(f"  net error {e}")
            gap = min(gap * 1.5, 60)
            continue
        if isinstance(r, dict) and r.get("status") == "SUCCESS":
            gap = max(gap * 0.97, 1.0)
            return r["data"]["candles"]
        if isinstance(r, dict) and r.get("fault", {}).get("code") == 422:
            return None  # past the 5yr wall
        gap = min(gap * 1.5, 60)
    log(f"  gave up {token} {interval} {fd}->{td}")
    return []


def update(sym, token, interval):
    path = OUT / interval / f"{sym}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    today = dt.date.today()
    start = old["timestamp"].max().date() if old is not None else today - dt.timedelta(days=int(365.25 * 5) - 2)
    rows, end = [], today
    while end >= start:
        s = max(start, end - dt.timedelta(days=CHUNK[interval]))
        got = fetch(token, interval, s.isoformat(), end.isoformat())
        if got is None:
            break
        if not got and old is None and rows:
            break  # walked back past the listing date
        if got:
            log(f"   {sym} {interval} {s} -> {end}: {len(got)} bars")
        rows += got
        end = s - dt.timedelta(days=1)
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=COLS)
    new["timestamp"] = pd.to_datetime(new["timestamp"])
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path.with_suffix(".tmp"), index=False, compression="zstd")
    path.with_suffix(".tmp").replace(path)  # atomic: a kill mid-write never corrupts the file
    return len(new)


def hf_push(folder, path_in_repo=""):
    hf.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=str(folder),
                     path_in_repo=path_in_repo, allow_patterns=["*.parquet"], commit_message=f"auto {dt.date.today()}")
    log(f"hf push ok: {folder}")


def kaggle_snapshot():
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "NSE OHLCV 1min and Daily 5yr", "id": KAGGLE, "licenses": [{"name": "CC0-1.0"}]}))
    v = subprocess.run(["kaggle", "datasets", "version", "-p", str(OUT), "-m", f"snapshot {dt.date.today()}",
                        "--dir-mode", "zip"], capture_output=True, text=True)
    if v.returncode != 0:  # first run: create it (private by default)
        v = subprocess.run(["kaggle", "datasets", "create", "-p", str(OUT), "--dir-mode", "zip"],
                           capture_output=True, text=True)
    log(f"kaggle rc={v.returncode} {(v.stdout + v.stderr).strip()[-300:]}")


def backfill():
    uni, last = universe(), time.time()
    log(f"backfill start: {len(uni)} instruments")
    for i, (sym, token) in enumerate(uni.items(), 1):
        for interval in ("D", "1min"):
            try:
                log(f"{i}/{len(uni)} {sym} {interval}: +{update(sym, token, interval)} rows (gap {gap:.1f}s)")
            except Exception as e:
                log(f"{i}/{len(uni)} {sym} {interval}: FAILED {type(e).__name__} {e}")
        if time.time() - last > 3600:
            try:
                hf_push(OUT)
            except Exception as e:
                log(f"hf push failed, retry next hour: {e}")
            last = time.time()
    hf_push(OUT)
    kaggle_snapshot()
    log("BACKFILL DONE")


def daily():
    uni, today = universe(), dt.date.today()
    limit = int(os.environ.get("LIMIT", 0)) or len(uni)  # LIMIT=5 for a smoke run
    fd = (today - dt.timedelta(days=5)).isoformat()
    for interval in ("D", "1min"):
        frames = []
        for i, (sym, token) in enumerate(list(uni.items())[:limit], 1):
            got = fetch(token, interval, fd, today.isoformat()) or []
            if got:
                f = pd.DataFrame(got, columns=COLS)
                f.insert(0, "symbol", sym)
                frames.append(f)
            if i % 250 == 0:
                log(f"{interval} {i}/{limit} (gap {gap:.1f}s)")
        df = pd.concat(frames)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        path = OUT / "daily" / interval / f"{today:%Y}" / f"{today:%m}" / f"{today}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="zstd")
        log(f"daily {interval}: {len(df)} rows, {df['symbol'].nunique()} symbols")
    hf_push(OUT / "daily", "daily")


if __name__ == "__main__":
    {"backfill": backfill, "daily": daily}[sys.argv[1]]()
