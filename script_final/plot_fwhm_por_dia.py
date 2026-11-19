"""Daily FWHM figures using the classification notebook's preprocessing."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


REGIONS = ((405, 435), (710, 750), (1350, 1373), (1926, 1997))
CONDITIONS = (("IRRIG", "Irrigated", "#2166ac"),
              ("NIRRIG", "Non-irrigated", "#d95f02"))


def measure_fwhm(x, y):
    """Measure the largest absolute deviation from the endpoint baseline."""
    baseline = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    residual = y - baseline
    peak = int(np.argmax(np.abs(residual)))
    oriented = residual * (1 if residual[peak] >= 0 else -1)
    half = oriented[peak] / 2
    left = right = peak
    while left > 0 and oriented[left] > half:
        left -= 1
    while right < len(x) - 1 and oriented[right] > half:
        right += 1
    if oriented[peak] <= 1e-12 or left == peak or right == peak:
        limits = np.array([np.nan, np.nan])
    else:
        limits = np.array([
            np.interp(half, oriented[left:left + 2], x[left:left + 2]),
            np.interp(half, [oriented[right], oriented[right - 1]],
                      [x[right], x[right - 1]]),
        ])
    return baseline, peak, residual[peak] / 2, limits


def plot_daily_fwhm(base_normalizada, comprimentos_onda, output_dir, show=False, pool_days=False):
    """Measure morning mean spectra per day or with all days pooled."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_normalizada.loc[base_normalizada["turno"].eq("manha")].copy()
    base["day"] = base["data_coleta"].astype(str).str.extract(r"^(D\d+)")
    if base.empty or base["day"].isna().any():
        raise ValueError("Morning spectra with valid day identifiers are required.")
    x = np.asarray(comprimentos_onda, dtype=float)
    columns = [str(int(w)) for w in x]
    rows = []
    prefix = "pooled_fwhm" if pool_days else "daily_fwhm"
    mean_label = "Pooled mean" if pool_days else "Daily mean"
    groups = [("All days pooled", base)] if pool_days else base.groupby("day", sort=True)
    with PdfPages(output_dir / f"{prefix}.pdf") as pdf:
        for day, group in groups:
            mean = group[columns].mean().to_numpy(dtype=float)
            curves = {key: group.loc[group["condicao"].eq(key), columns].mean().to_numpy(dtype=float)
                      for key, _, _ in CONDITIONS}
            fig = plt.figure(figsize=(17, 10), layout="constrained")
            grid = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])
            full = fig.add_subplot(grid[0, :])
            for key, label, color in CONDITIONS:
                full.plot(x, curves[key], color=color, lw=1.5, label=label)
            full.plot(x, mean, color="black", ls="--", lw=1.2, label=mean_label)
            for region, (start, end) in enumerate(REGIONS, 1):
                mask = (x >= start) & (x <= end)
                xx, yy = x[mask], mean[mask]
                baseline, peak, half_height, limits = measure_fwhm(xx, yy)
                center = xx[peak]
                width = limits[1] - limits[0]
                rows.append(dict(day=day, region=region, start_nm=start, end_nm=end,
                                 center_nm=center, feature="peak" if half_height >= 0 else "trough",
                                 half_height_left_nm=limits[0], half_height_right_nm=limits[1],
                                 fwhm_nm=width, n=len(group)))
                full.axvspan(start, end, color="gray", alpha=0.15)
                full.axvline(center, color="#68459b", ls=":", lw=1.2)
                full.text(center, 0.98, f"R{region}: {center:.0f} nm", rotation=90,
                          transform=full.get_xaxis_transform(), ha="right", va="top", fontsize=9)
                ax = fig.add_subplot(grid[1, region - 1])
                for key, label, color in CONDITIONS:
                    ax.plot(xx, curves[key][mask], color=color, lw=1.3, label=label)
                ax.plot(xx, yy, color="black", lw=1.5, label=mean_label)
                ax.plot(xx, baseline, color="gray", ls=":", label="Baseline")
                ax.plot(limits, np.interp(limits, xx, baseline) + half_height,
                        color="#68459b", ls="--", marker="|", markersize=10,
                        label=f"FWHM: {width:.2f} nm" if np.isfinite(width) else "FWHM: undefined")
                ax.axvline(center, color="#68459b", ls=":", lw=1.2)
                ax.plot(center, yy[peak], "o", color="#68459b", ms=5)
                ax.set(title=f"R{region}: {start}\u2013{end} nm\nCenter wavelength: {center:.0f} nm",
                       xlabel="Wavelength (nm)", ylabel="Normalized response (SNV)")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8, loc="best")
            full.set(xlim=(400, 2450), xlabel="Wavelength (nm)", ylabel="Normalized response (SNV)")
            full.grid(alpha=0.2)
            full.legend(loc="lower right")
            fig.suptitle(f"{day} | Center wavelengths and FWHM of four spectral regions", fontsize=17)
            note = (f"Morning measurements; all {base['day'].nunique()} days and all three genotypes pooled. "
                    "Centers and FWHM calculated from the pooled mean.\n" if pool_days else
                    "Morning measurements; all three genotypes pooled. Centers and FWHM calculated from the daily mean.\n")
            fig.supxlabel(note +
                          "Preprocessing: detector splice correction, Savitzky\u2013Golay smoothing and SNV.", fontsize=10)
            filename = f"{prefix}.png" if pool_days else f"{prefix}_{day}.png"
            fig.savefig(output_dir / filename, dpi=200)
            pdf.savefig(fig)
            if show:
                from IPython.display import display
                display(fig)
            plt.close(fig)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / f"{prefix}.csv", index=False)
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    notebook_path = root / "script_final/Pipeline_Final_Classificacao.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"display": lambda value: None}
    for marker in ("def localizar_raiz(", "def limpar_metadados(", "base_bruta = pd.read_csv",
                   "def jump_correct(", "base_recortada, comprimentos_onda = preprocessar"):
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and marker in "".join(c["source"]))
        # The command-line renderer does not need the notebook display dependency.
        source = source.replace("from IPython.display import display", "")
        exec(compile(source, str(notebook_path), "exec"), namespace)
    output = root / "script_final/outputs/classificacao/fwhm_por_dia"
    table = plot_daily_fwhm(namespace["base_normalizada"], namespace["comprimentos_onda"], output)
    plot_daily_fwhm(namespace["base_normalizada"], namespace["comprimentos_onda"], output, pool_days=True)
    print(f"Saved {table['day'].nunique()} daily figures, PDF and CSV to {output}")
