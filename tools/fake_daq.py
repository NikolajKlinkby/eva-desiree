"""Fake DAQ: replays example RA<det>_<date>.<time>.csv files as a live run.

Lines are written as in the source (CRLF, padding, trailing ';'), paced by the
'Sync Time' column (default) or 'Event time'. One thread per detector, all
started together. When a file is exhausted it loops. On later passes
(--on-loop append) 'Event time', 'Sync Time' and 'Event ID' are shifted by one
run length per pass, so each detector's clock stays continuous (pulse spacing,
cross-detector clock offset and drift continue smoothly through the wrap).

  python fake_daq.py --src ./examples --out ./live --speed 20
  python fake_daq.py --src ./examples --out ./live --mode event --tear
"""
import argparse
import os
import re
import threading
import time
from pathlib import Path
from datetime import datetime

import numpy as np

SRC_RE = re.compile(r"^RA(\d+)_(\d{8}[._]\d{6})\.csv$")


def load_source(path, mode):
    with open(path, "rb") as f:
        lines = f.readlines()                       # keeps b"\r\n"
    header, body = lines[0], lines[1:]
    if not body[-1].endswith(b"\n"):
        body[-1] += b"\r\n"
    names = [t.strip() for t in header.decode().strip().split(";") if t.strip()]
    i_id, i_ev, i_sy = (names.index(n) for n in ("Event ID", "Event time", "Sync Time"))
    sync = np.array([float(l.split(b";")[i_sy]) for l in body])
    event = np.array([float(l.split(b";")[i_ev]) for l in body])
    t = sync if mode == "sync" else event
    t = np.maximum.accumulate(t - t[0])             # monotonic, starts at 0
    shift = dict(
        cols=(i_id, i_ev, i_sy),
        dt=float(sync[-1] - sync[0] + np.median(np.diff(np.unique(sync)))),
        did=int(body[-1].split(b";")[i_id]) - int(body[0].split(b";")[i_id]) + 1)
    return header, body, t, shift


def shift_line(line, k, shift):
    i_id, i_ev, i_sy = shift["cols"]
    p = line.split(b";")
    p[i_id] = b"%d" % (int(p[i_id]) + k * shift["did"])
    p[i_ev] = b"%.8f" % (float(p[i_ev]) + k * shift["dt"])
    p[i_sy] = b"%.8f" % (float(p[i_sy]) + k * shift["dt"])
    return b";".join(p)


def make_groups(t, tick):
    """Group lines released together: same time bucket -> one burst."""
    key = np.floor(t / tick).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, np.diff(key) != 0])
    ends = np.r_[starts[1:], len(t)]
    return starts, ends, t[starts]


def run_detector(path_out, header, body, t, shift, args, t0, stop, stats, det):
    starts, ends, rel = make_groups(t, args.tick)
    gap = float(np.median(np.diff(rel))) if len(rel) > 1 else 1.0
    period = rel[-1] + gap                           # loop length (source time)
    loop = 0
    f = open(path_out, "wb")
    f.write(header)
    f.flush()
    rng = np.random.default_rng(int(det))
    while not stop.is_set():
        k = loop if args.on_loop == "append" else 0
        for s, e, r in zip(starts, ends, rel):
            due = t0 + (loop * period + r) / args.speed
            wait = due - time.monotonic()
            if wait > 0 and stop.wait(wait):
                break
            lines = body[s:e]
            if k:
                lines = [shift_line(l, k, shift) for l in lines]
            data = b"".join(lines)
            if args.tear and rng.random() < 0.1 and len(data) > 20:
                cut = int(rng.integers(1, len(data) - 1))   # mid-line split
                f.write(data[:cut]); f.flush()
                time.sleep(0.002)
                f.write(data[cut:])
            else:
                f.write(data)
            f.flush()
            stats[det] += e - s
        else:                                        # finished a full pass
            loop += 1
            if args.on_loop == "restart":
                f.close()
                f = open(path_out, "wb")             # truncate, like a new file
                f.write(header); f.flush()
            continue
        break
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1] / "examples" / "test_data"),
                    help="dir with example RA*.csv files")
    ap.add_argument("--out", default="./live", help="dir to write live files into")
    ap.add_argument("--run", default="latest",
                    help="source run stamp, e.g. 20260624.130138 (default: newest in --src)")
    ap.add_argument("--mode", choices=["sync", "event"], default="sync",
                    help="pace by 'Sync Time' (bursts ~1/s) or 'Event time' (smooth)")
    ap.add_argument("--speed", type=float, default=1.0, help="time compression factor")
    ap.add_argument("--tick", type=float, default=0.02,
                    help="lines within one tick (source seconds) are written together")
    ap.add_argument("--on-loop", choices=["append", "restart"], default="append",
                    help="append: keep growing the file (clocks continue); "
                         "restart: truncate and rewrite header")
    ap.add_argument("--tear", action="store_true",
                    help="occasionally split a write mid-line (tests partial-line handling)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    runs = {}                                        # run stamp -> {det: path}
    for n in os.listdir(args.src):
        if m := SRC_RE.match(n):
            runs.setdefault(m[2], {})[m[1]] = os.path.join(args.src, n)
    if not runs:
        raise SystemExit(f"no RA<det>_<date>.<time>.csv files in {args.src}")
    run = max(runs) if args.run == "latest" else args.run
    if run not in runs:
        raise SystemExit(f"run {run} not found; available: {sorted(runs)}")
    print(f"source run {run} (available: {sorted(runs)})")
    sources = sorted(runs[run].items())

    stop, stats, threads = threading.Event(), {}, []
    t0 = time.monotonic() + 0.2
    for det, path in sources:
        header, body, t, shift = load_source(path, args.mode)
        out = os.path.join(args.out, f"RA{det}_{stamp}.csv")
        stats[det] = 0
        print(f"RA{det}: {len(body)} lines, {t[-1]:.0f} s of source time -> {out}")
        th = threading.Thread(target=run_detector, daemon=True,
                              args=(out, header, body, t, shift, args, t0, stop, stats, det))
        th.start(); threads.append(th)
    try:
        while True:
            time.sleep(2)
            print("written:", {f"RA{d}": int(n) for d, n in stats.items()})
    except KeyboardInterrupt:
        stop.set()
        for th in threads:
            th.join()
        print("stopped")


if __name__ == "__main__":
    main()
