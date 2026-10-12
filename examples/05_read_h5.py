"""Read an analysis file with plain h5py: no part of this program is needed.

Prints the layout, the README stored in the file, and picks out a few named datasets. Everything the GUI plots
is under /plots, with the arrays named after what they are.

  python examples/05_read_h5.py examples/output/example_run.h5       (made by 04_save_h5.py)
"""
import sys

import h5py
import numpy as np


def text(ds):
    x = ds[()]
    return x.decode() if isinstance(x, bytes) else str(x)


def tree(group, indent=0, depth=2):
    for name in group:
        obj = group[name]
        if isinstance(obj, h5py.Group):
            print("  " * indent + f"{name}/")
            if indent < depth:
                tree(obj, indent + 1, depth)
        else:
            print("  " * indent + f"{name}  {obj.shape} {obj.dtype}")


def main(path):
    with h5py.File(path, "r") as f:
        print("attributes:", {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()})
        print("\n=== layout (two levels) ===")
        tree(f, depth=1)

        print("\n=== the README inside the file (first lines) ===")
        print("\n".join(text(f["README"]).splitlines()[:28]))

        print("\n=== raw data: detector columns ===")
        for det in f["raw"]:
            g = f["raw"][det]
            print(f"{det}: run {g.attrs['run']}, {g.attrs['n_rows']} rows, columns: {list(g)[:8]} ...")
        first = next(iter(f["raw"]))
        sums = f["raw"][first]["Sum"][()] if "Sum" in f["raw"][first] else None
        if sums is not None:
            print(f"{first}: Sum column mean {np.mean(sums):.1f}, max {np.max(sums):.1f}")

        print("\n=== settings: every stage, with the names of the program ===")
        ab = f["settings/abel/a"]
        print("Abel settings of side A:", {k: (v.item() if hasattr(v, "item") else v) for k, v in ab.attrs.items() if not k.startswith("_")})
        print("cuts:", {k: v for k, v in f["settings/selection/a"].attrs.items() if not k.startswith("_")})

        print("\n=== plots: where is what (the index in the README lists all of them) ===")
        for side in f["plots/analysis"]:
            g = f["plots/analysis"][side]
            print(side, "->", list(g))
            if "abel" in g and "intensity" in g["abel"]:
                I = g["abel/intensity"]
                rho, inten = I["radius_rho"][()], I["intensity"][()]
                betas = {n: g["abel"][n]["beta"][()] for n in g["abel"] if n.startswith("beta_")}
                ok = np.all([np.isfinite(b) for b in betas.values()], axis=0) if betas else np.ones(len(inten), bool)
                i = int(np.argmax(np.where(ok, inten, -np.inf)))      # strongest bin where beta is defined
                print(f"   abel/intensity: {I.attrs['title']!r}, {len(rho)} points, peak {inten[i]:.4g} at rho = {rho[i]:.4g}")
                print("   attributes:", {k: I.attrs[k] for k in ("xlabel", "ylabel", "gui_location")})
                for name, b in betas.items():
                    print(f"   abel/{name}: {b[i]:+.3f} at the peak")
        img = f["plots/analysis"][next(iter(f["plots/analysis"]))]["image_centred_circularised"]
        print("measured polar image:", img["counts"].shape, "= [angle, radius];  axes:", img["angle_deg"].shape, img["radius"].shape)

        print("\n=== results: the stage outputs with the field names of the program ===")
        print("abel side a:", list(f["results/abel/a"]))
        print("   intensity I(rho):", f["results/abel/a/abel/intensity"].shape,
              "| I_n(rho) [order, rho]:", f["results/abel/a/abel/I_n"].shape)

    # optional picture of two plots straight from the file
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with h5py.File(path, "r") as f:
            side = next(iter(f["plots/analysis"]))
            g = f["plots/analysis"][side]
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            c = g["image_centred_circularised"]
            ax[0].imshow(c["counts"][()], origin="lower", aspect="auto",
                         extent=(c["radius"][0], c["radius"][-1], c["angle_deg"][0], c["angle_deg"][-1]))
            ax[0].set(title=str(c.attrs["title"]), xlabel=str(c.attrs["xlabel"]), ylabel=str(c.attrs["ylabel"]))
            if "abel" in g and "intensity" in g["abel"]:
                I = g["abel/intensity"]
                ax[1].plot(I["radius_rho"][()], I["intensity"][()])
                if "lower" in I:
                    ax[1].fill_between(I["radius_rho"][()], I["lower"][()], I["upper"][()], alpha=0.3)
                ax[1].set(title=str(I.attrs["title"]), xlabel=str(I.attrs["xlabel"]))
            t = f["plots/coincidence/timing/dt_zoom"]
            ax[2].step(t["dt_us"][()], t["pairs"][()]); ax[2].set(title=str(t.attrs["title"]), xlabel=str(t.attrs["xlabel"]))
            fig.tight_layout()
            out = path.rsplit(".", 1)[0] + "_plots.png"
            fig.savefig(out, dpi=110)
            print("\npicture:", out)
    except ImportError:
        print("\n(matplotlib not installed: no picture)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
