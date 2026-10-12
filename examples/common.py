"""Helpers shared by the examples (no GUI). Importing this module makes the project packages (analysis, storage, session, ...) importable from the project's src/."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paths import EXAMPLE_DATA, scan_detectors          # noqa: E402
from session import Session, SessionConfig               # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "output"          # the examples write their files here
OUTPUT.mkdir(exist_ok=True)


def example_detectors(folder=EXAMPLE_DATA):
    """(A, B): the detector that is filtered and imaged first, and its partner (the two found in the folder)."""
    dets = scan_detectors(folder)
    if len(dets) < 2:
        raise SystemExit(f"need DAQ files of two detectors in {folder}, found {dets}")
    return dets[-1], dets[0]


def start_session(folder=EXAMPLE_DATA, det_a=None, det_b=None, bins=128, **config):
    """A running Session on a folder of DAQ files. Stop it with session.stop()."""
    a, b = example_detectors(folder)
    s = Session(SessionConfig(directory=str(folder), det_a=det_a or a, det_b=det_b or b, bins=bins, **config))
    s.start()
    return s


def abel_ready(s):
    r = s.abel_result()
    return r is not None and r.a.abel is not None and r.a.n_events > 0 and r.b.abel is not None


def peak_index(abel):
    """Index of the strongest I(rho) bin where beta is defined (beta is NaN where the intensity is too low)."""
    import numpy as np
    ok = np.isfinite(abel.beta).all(axis=0) if getattr(abel.beta, "size", 0) else np.ones(len(abel.intensity), bool)
    return int(np.argmax(np.where(ok, abel.intensity, -np.inf)))


def wait_until(predicate, timeout=300.0, poll=0.5, what="results"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return
        time.sleep(poll)
    raise TimeoutError(f"no {what} after {timeout:.0f} s")


def _versions(s):
    r = s.abel_result()
    return (s.ds.snapshot().version, s.sel.settings.version, s.pos.settings.version, s.polar.settings.version,
            s.abel.settings.version, None if r is None else r.version)


def wait_settled(s, quiet=4.0, timeout=600.0):
    """Wait until the Abel result exists and nothing has changed for `quiet` seconds. With files that
    are complete (offline analysis) the pipeline is then done; with live data this is only a pause."""
    wait_until(lambda: abel_ready(s), timeout, what="Abel result")
    last, since = _versions(s), time.time()
    while time.time() - since < quiet:
        time.sleep(0.5)
        now = _versions(s)
        if now != last:
            last, since = now, time.time()
        if time.time() - since > timeout:
            break


def report(s, title=""):
    """The numbers behind the Detectors, Coincidence and Analysis tabs, as text."""
    print(f"--- {title}" if title else "---")
    snap = s.ds.snapshot()
    print("rows read:", {f"det {d}": sn.n for d, sn in snap.detectors.items()})
    coin = s.coincidence()
    if coin is not None:
        print(coin.summary())
    sel = s.selection()
    if sel is not None:
        print(sel.settings.describe(sel.det_a, sel.det_b))
        for label, na, nb in sel.cutflow:
            print(f"  cut flow: {label:<28s} A {na:>9}   B {nb:>9}")
    ab = s.abel_result()
    if ab is not None:
        for k in ("a", "b"):
            a = ab[k]
            line = f"Abel, detector {a.det}: {a.n_events} events"
            if a.abel is not None and a.n_events:
                i = peak_index(a.abel)
                betas = ", ".join(f"beta{n} = {a.abel.beta[j, i]:+.3f}" for j, n in enumerate(a.abel.beta_orders))
                line += f", I(rho) peaks at rho = {a.rho[i]:.4g} ({a.abel.intensity[i]:.4g} per unit radius), {betas}"
                line += f"; loss terms: {a.basis.loss_text()}" if a.basis is not None else ""
            print(line)
