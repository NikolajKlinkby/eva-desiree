"""Offline analysis from an analysis file: load it, change the analysis, save a new file.

The file contains the raw data and every setting. A Session opened on it re-runs the whole pipeline (no DAQ
files needed), so anything can be changed after the fact - cuts, coincidence window, centre, Abel corrections -
exactly as in the GUI after "open saved file". This script tries maximum entropy and a tighter Sum cut.

  python examples/06_reanalyse_h5.py examples/output/example_run.h5            (made by 04_save_h5.py)
"""
import sys

import common
from session import Session, SessionConfig
from storage import capture, h5file


def main(path, out):
    info = h5file.read_meta(path)                              # settings, configuration, widget state
    print("file saved", info["meta"]["saved"], "| run", info["meta"]["run"], "| comment:", info["meta"].get("comment") or "-")
    c = info["config"]
    s = Session(SessionConfig(directory="", det_a=c["det_a"], det_b=c["det_b"], bins=c["bins"],
                              search_window_us=c["search_window_us"], shot_period_us=c["shot_period_us"],
                              offline_file=path))             # the raw data come from the file
    s.preset = info["settings"]                               # applied before the stages start
    s.start()
    try:
        common.wait_settled(s)
        common.report(s, "as saved")

        s.abel.set_loss("a", entropy=0.3)                       # a different Abel correction ...
        s.sel.set_sum(s.config.det_a, 1500, None)               # ... and a different Sum cut
        common.wait_settled(s)
        common.report(s, "re-analysed")

        h5file.write_file(out, capture.capture(s, comment="re-analysed from " + path, source=path))
        print("written", out)
    finally:
        s.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    main(src, sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + "_reanalysed.h5")
