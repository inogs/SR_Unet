import sys
import numpy as np
import netCDF4 as nc
import os
import matplotlib.pyplot as plt



def load_var(path):
    ds = nc.Dataset(path)
    var_name = list(ds.variables.keys())[-1]  # prende ultima variabile (più robusto: puoi cambiarlo)
    data = ds.variables[var_name][:].filled(np.nan)
    ds.close()
    return data


def main(f1_path, f2_path, tol, out_txt):

    tol = float(tol)

    v0 = load_var(f1_path)
    v1 = load_var(f2_path)

    # -----------------------
    # MASK TERRA (NaN)
    # -----------------------
    valid_mask = ~np.isnan(v0) & ~np.isnan(v1)

    a = v0[valid_mask]
    b = v1[valid_mask]

    # -----------------------
    # EXACT CHECK
    # -----------------------
    exact_equal = np.array_equal(a, b)

    # -----------------------
    # DIFFERENZA
    # -----------------------
    if a.shape != b.shape:
        raise ValueError(f"Shape diverse: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    diff3d = np.abs(v0 - v1)


    dmin = np.min(diff)
    dmax = np.max(diff)

    # dove differiscono di più in verticale
    diff2d = np.nanmax(diff3d, axis=0)

    plt.figure(figsize=(8,6))

    # mask terra
    diff2d_plot = diff2d.copy()
    diff2d_plot[np.isnan(v0[0])] = np.nan

    im = plt.imshow(diff2d_plot, origin="lower")
    plt.colorbar(im, label="|diff|")

    plt.title("Difference map (max diff)")
    plt.savefig(out_txt.replace(".txt", "_diff.png"))
    plt.close()


    # -----------------------
    # TOLERANCE CHECK
    # -----------------------
    diff_mask = diff > tol
    diff_pct = np.sum(diff_mask) / len(diff)

    # -----------------------
    # OUTPUT
    # -----------------------
    lines = []
    lines.append(f"File 1: {f1_path}")
    lines.append(f"File 2: {f2_path}")
    lines.append(f"TOLERANCE: {tol}")
    lines.append("")

    lines.append(f"EXACT EQUALITY (no tolerance): {exact_equal}")
    lines.append(f"MIN DIFF: {dmin}")
    lines.append(f"MAX DIFF: {dmax}")
    lines.append(f"DIFFERENT CELLS PERCENTAGE (>tol): {diff_pct * 100:.6f}%")

    text = "\n".join(lines)

    print(text)

    with open(out_txt, "w") as f:
        f.write(text)

    print(f"\nSaved to: {out_txt}")


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python compare_nc.py file1.nc file2.nc tolerance output.txt")
        sys.exit(1)

    f1 = sys.argv[1]
    f2 = sys.argv[2]
    tol = sys.argv[3]
    out_txt = sys.argv[4]

    main(f1, f2, tol, out_txt)
