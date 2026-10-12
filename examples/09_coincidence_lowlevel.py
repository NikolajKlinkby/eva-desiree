"""The building blocks without a Session: read DAQ files, find coincidences, apply cuts, make an image.

DetectorSet reads the CSV files (it can also follow growing ones); CoincidenceFilter matches the two detectors'
events using the laser sync pulses; SelectionFilter applies Sum / time cuts and subtracts the accidentals;
Imager bins positions into an image. Each is Qt-free and runs on poll() or in a thread of its own.

  python examples/09_coincidence_lowlevel.py
"""
import numpy as np

import common
from analysis.coincidence import CoincidenceFilter
from analysis.filters import SelectionFilter
from analysis.imaging import CartesianBinning, Imager
from daq.data_loader import DetectorSet
from paths import EXAMPLE_DATA

det_a, det_b = common.example_detectors()

ds = DetectorSet(str(EXAMPLE_DATA))
ds.poll()                                                  # discover the files and read what is there
snap = ds.snapshot()
print("run", snap.run, {d: s.n for d, s in snap.detectors.items()}, "rows")
print("columns of detector", det_a, list(snap.detectors[det_a].columns))

flt = CoincidenceFilter(ds, det_a, det_b)
flt.poll()                                                 # one call processes everything available
for _ in range(200):                                       # a long file is searched in blocks
    if flt.poll() == 0:
        break
res = flt.result()
print(res.summary())

sel = SelectionFilter(ds, flt)
sel.set_sum(det_a, 2000, None)                             # keep Sum >= 2000 on the electron detector
sel.poll()
r = sel.result()
print("kept on A:", int(r.mask_a.sum()), "| weight (true pairs after background subtraction):", round(float(r.weight_a.sum()), 1))
centres, counts = r.histogram(det_a, "sum", "true", bins=50)
print("Sum spectrum of true pairs, peak at Sum =", round(float(centres[np.argmax(counts)]), 1))

img = Imager(ds, sel, binning=CartesianBinning(64, 64))
img.poll()
ir = img.result()[det_a]
print(ir.summary())
print(ir.ascii(24))
