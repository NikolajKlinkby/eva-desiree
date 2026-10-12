"""Append-only CSV tail loader (single file) + multi-detector set.

Design rules
------------
* Files are only ever appended to. We remember a byte offset and read
  only complete lines (up to the last newline) on each poll().
* Data is stored column-major in one preallocated array (ncols, capacity),
  so every column is a contiguous 1D view.
* State (epoch, buffer, n) is published as ONE tuple assignment, so readers
  get a consistent view without any lock. Rows below n are never modified.
* Growth allocates a new buffer and swaps it in; old snapshots stay valid.

File format handled: ';' delimited, CRLF, header without trailing ';',
data rows WITH trailing ';' and space padding (extra empty field ignored).
"""
import io
import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np

RUN_RE = re.compile(r"^RA(?P<det>\d+)_(?P<run>\d{8}[._]\d{6})\.csv$")


@dataclass(frozen=True)
class Snapshot:
    epoch: int                 # increments if the file was truncated/restarted
    n: int                     # number of valid rows
    columns: dict = field(default_factory=dict)  # name -> 1D view (read-only use!)

    @property
    def version(self):
        return (self.epoch, self.n)


@dataclass(frozen=True)
class SetSnapshot:
    run: str                   # run stamp, e.g. "20260624.130138" (None if no files)
    detectors: dict            # detector id (int) -> Snapshot

    @property
    def version(self):
        return (self.run, tuple((d, s.version) for d, s in sorted(self.detectors.items())))


class Poller:
    """Shared background-polling helper."""
    _thread = None
    last_error = None            # repr of the last exception raised in the worker thread

    @property
    def running(self):
        return self._thread is not None

    def start(self, interval=0.2, on_update=None):
        """Poll in a background thread. on_update(n_new) runs in that thread."""
        if self._thread is not None:
            return
        self._stop = threading.Event()

        def run():
            while not self._stop.is_set():
                try:
                    n_new = self.poll()
                except Exception as exc:       # keep the thread alive
                    self.last_error = f"{type(self).__name__}: {exc!r}"
                    print("worker error:", self.last_error)
                    n_new = 0
                if n_new and on_update:
                    on_update(n_new)
                self._stop.wait(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None


class EventLoader(Poller):
    def __init__(self, path, delimiter=";", columns=None, has_header="auto",
                 dtype=np.float64, initial_capacity=200_000):
        self.path = path
        self.delimiter = delimiter
        self._user_columns = columns
        self._has_header = has_header
        self.dtype = dtype
        self._capacity = initial_capacity

        self.n_bad_lines = 0
        self._offset = 0
        self._names = None
        self._header_done = False
        self._state = (0, None, 0)        # (epoch, buffer, n_valid)

    # ------------------------------------------------------------ public
    def snapshot(self) -> Snapshot:
        epoch, buf, n = self._state       # single atomic read
        if buf is None:
            return Snapshot(epoch, 0, {})
        cols = {name: buf[i, :n] for i, name in enumerate(self._names)}
        return Snapshot(epoch, n, cols)

    def poll(self) -> int:
        """Read any new complete lines. Returns number of rows added."""
        try:
            size = os.path.getsize(self.path)
        except FileNotFoundError:
            return 0
        if size < self._offset:           # file truncated or replaced
            self._reset()
        if size == self._offset:
            return 0

        with open(self.path, "rb") as f:
            f.seek(self._offset)
            raw = f.read(size - self._offset)

        end = raw.rfind(b"\n")
        if end < 0:                       # no complete line yet
            return 0
        self._offset += end + 1
        return self._ingest(raw[: end + 1])

    # ----------------------------------------------------------- internal
    def _reset(self):
        epoch = self._state[0] + 1
        self._offset = 0
        self._header_done = False
        self._names = None
        self.n_bad_lines = 0
        self._state = (epoch, None, 0)

    def _tokens(self, line: bytes):
        toks = line.decode(errors="replace").strip().split(self.delimiter)
        while toks and not toks[-1].strip():       # drop trailing empty fields
            toks.pop()
        return [t.strip() for t in toks]

    def _ingest(self, chunk: bytes) -> int:
        if not self._header_done:
            first, _, rest = chunk.partition(b"\n")
            tokens = self._tokens(first)
            if self._has_header == "auto":
                try:
                    [float(t) for t in tokens]
                    header = False
                except ValueError:
                    header = True
            else:
                header = bool(self._has_header)
            if header:
                self._names = tokens
                chunk = rest
            elif self._user_columns:
                self._names = list(self._user_columns)
            else:
                self._names = [f"col{i}" for i in range(len(tokens))]
            if len(self._names) != len(tokens):
                raise ValueError(f"{len(self._names)} column names but "
                                 f"{len(tokens)} fields in file")
            self._header_done = True
            self._state = (self._state[0],
                           np.empty((len(tokens), self._capacity), self.dtype), 0)

        arr = self._parse(chunk)
        m = arr.shape[0]
        if m == 0:
            return 0

        epoch, buf, n = self._state
        if n + m > buf.shape[1]:
            new = np.empty((buf.shape[0], max(2 * buf.shape[1], n + m)), self.dtype)
            new[:, :n] = buf[:, :n]
            buf = new
        buf[:, n:n + m] = arr.T           # write BEYOND the published count
        self._state = (epoch, buf, n + m)  # publish last
        return m

    def _parse(self, chunk: bytes) -> np.ndarray:
        ncols = len(self._names)
        if not chunk.strip():
            return np.empty((0, ncols), self.dtype)
        try:  # fast path: C parser; usecols ignores the trailing empty field
            return np.loadtxt(io.BytesIO(chunk), delimiter=self.delimiter,
                              dtype=self.dtype, ndmin=2, usecols=range(ncols))
        except ValueError:
            pass
        rows = []                          # slow path: skip bad lines one by one
        for line in chunk.splitlines():
            if not line.strip():
                continue
            parts = self._tokens(line)
            if len(parts) != ncols:
                self.n_bad_lines += 1
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                self.n_bad_lines += 1
        return np.array(rows, self.dtype).reshape(-1, ncols)


class DetectorSet(Poller):
    """All detector files (RA<det>_<YYYYMMDD>.<HHMMSS>.csv) of one run.

    run="latest" follows the newest run stamp found in the directory; if a
    newer run appears, everything is dropped and reloaded (run changes).
    Pass an explicit stamp to pin one run. Files may appear at any time.
    """
    def __init__(self, directory, run="latest", **loader_kwargs):
        self.directory = directory
        self.run = run
        self._kw = loader_kwargs
        self._state = (None, {})          # (run stamp, {det: EventLoader})

    def snapshot(self) -> SetSnapshot:
        run, loaders = self._state
        return SetSnapshot(run, {d: l.snapshot() for d, l in loaders.items()})

    def poll(self) -> int:
        self._discover()
        return sum(l.poll() for l in list(self._state[1].values()))

    def _discover(self):
        found = {}                         # run -> {det: path}
        try:
            names = os.listdir(self.directory)
        except FileNotFoundError:
            return
        for name in names:
            m = RUN_RE.match(name)
            if m:
                found.setdefault(m["run"], {})[int(m["det"])] = os.path.join(
                    self.directory, name)
        if not found:
            return
        run = max(found) if self.run == "latest" else self.run
        if run not in found:
            return
        cur_run, loaders = self._state
        if run != cur_run:                 # new run: start from scratch
            loaders = {}
        new = {d: p for d, p in found[run].items() if d not in loaders}
        if new or run != cur_run:
            loaders = dict(loaders)        # copy-on-write, then publish
            for d, p in new.items():
                loaders[d] = EventLoader(p, **self._kw)
            self._state = (run, loaders)