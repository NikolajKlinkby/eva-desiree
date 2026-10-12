"""Background saving of the analysis file (Qt-free): one write at a time, skipped when nothing changed."""
import threading
import time

from storage import capture, h5file


def session_key(session):
    """Changes whenever the data, a setting or a result changed (cheap: versions only)."""
    s = session
    snap = s.ds.snapshot()
    boots = []
    for k in ("a", "b"):
        r = s.abel.bootstrap_result(k)
        boots.append(None if r is None else (r.n_done, r.running))
    return (snap.version, s.sel.settings.version, s.pos.settings.version, s.img.settings.version,
            s.polar.settings.version, s.abel.settings.version, tuple(boots), s.flt.cycle_shift,
            s.flt._window, tuple(s.flt.bkg_shots))


class Saver:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._key = None
        self._created = None
        self._n = 0
        self.last_time = None            # time.time() of the last successful save
        self.last_path = None
        self.last_error = None
        self.last_seconds = None
        self.busy = False

    def reset(self):
        """A new session: the next file starts afresh."""
        self._key, self._created, self._n = None, None, 0
        self.last_time = self.last_error = None

    def needs_save(self, session, path):
        """True if something changed since the last save (or the target is another file)."""
        return path != self.last_path or session_key(session) != self._key

    def save(self, session, path, gui=None, comment="", source=None, force=False, wait=False):
        """Write the file. force=False: only if something changed since the last save. wait=False: in a
        background thread (returns at once; a save still running makes this call a no-op).
        Returns True if a write was started / done."""
        with self._lock:
            if self.busy:
                return False
            key = session_key(session)
            if not force and key == self._key and path == self.last_path:
                return False
            self.busy = True
        # everything that needs the Qt thread has been collected by the caller (gui); the rest is thread-safe
        snap = capture.capture(session, gui=gui, comment=comment, created=self._created, n_saves=self._n + 1,
                               source=source)

        def job():
            t0 = time.time()
            try:
                h5file.write_file(path, snap)
                with self._lock:
                    self._key, self._created, self._n = key, snap["meta"]["created"], self._n + 1
                    self.last_time, self.last_path, self.last_error = time.time(), path, None
                    self.last_seconds = time.time() - t0
            except Exception as exc:                             # shown in the status line, never fatal
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self.busy = False
        if wait:
            job()
        else:
            self._thread = threading.Thread(target=job, daemon=True)
            self._thread.start()
        return True

    def join(self, timeout=None):
        t = self._thread
        if t is not None:
            t.join(timeout)

    def status(self):
        with self._lock:
            if self.busy:
                return "saving ..."
            if self.last_error:
                return "SAVE FAILED: " + self.last_error
            if self.last_time is None:
                return "not saved yet"
            return f"saved {time.strftime('%H:%M:%S', time.localtime(self.last_time))} ({self.last_seconds:.1f} s)"
