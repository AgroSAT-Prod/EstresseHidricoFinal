"""Reproduce the reference correlogram using the final prediction notebook data.

Run from the project root. No model fitting or variable selection is performed.
"""
import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "correlogram_final"
LABELS = {
    "CHL total": "Total chlorophyll",
    "Clorofila A": "Chlorophyll a",
    "Clorofila B": "Chlorophyll b",
    "gs": "Stomatal conductance",
    "CRA": "Relative water content",
    "ETR": "Electron transport rate",
    "E": "Transpiration rate",
    "A": "Net CO2 assimilation",
    "Ci": "Intercellular CO2",
}


def main():
    import sys
    if "--cached" in sys.argv:
        cross = pd.read_csv(OUT / "pearson_traits_wavelengths.csv", sep=";", index_col=0)
        within = pd.read_csv(OUT / "pearson_traits.csv", sep=";", index_col=0)
        targets = pd.read_csv(OUT / "paired_physiological_observations.csv", sep=";", index_col=[0, 1, 2, 3])
        wavelengths = cross.columns.astype(int).to_numpy()
    else:
        notebook = json.loads((HERE / "Pipeline_Final_Predicao.ipynb").read_text(encoding="utf-8"))
        ctx = {"__name__": "__correlogram__", "display": lambda *args, **kwargs: None}
        # Reuse the exact data cleaning, preprocessing and plot-level pairing rules.
        with contextlib.redirect_stdout(io.StringIO()):
            for index in (1, 3, 4, 6, 7, 11):
                source = "".join(notebook["cells"][index]["source"]).replace("from IPython.display import display", "")
                exec(compile(source, f"Pipeline_Final_Predicao.ipynb:cell{index}", "exec"), ctx)
                ctx["display"] = lambda *args, **kwargs: None
        OUT.mkdir(parents=True, exist_ok=True)
        wavelengths = ctx["comprimentos_onda"].astype(int)
        columns = [str(w) for w in wavelengths]
        keys = ["data", *ctx["CHAVES_FISIOLOGIA"]]
        spectra = ctx["espectros_fisiologia_norm"].copy()
        reverse_dates = {v: k for k, v in ctx["MAPA_FISIOLOGICO"].items()}
        spectra["data"] = spectra["data_coleta"].map(reverse_dates)
        spectra = spectra.dropna(subset=["data"]).set_index(keys)[columns]
        assert spectra.index.is_unique
        targets = pd.DataFrame(index=spectra.index)
        for target, (_, dates) in ctx["ALVOS_FISIOLOGICOS"].items():
            values = []
            for date in dates:
                matched = ctx["parear_fisiologia"](target, date, ctx["espectros_fisiologia_norm"])
                matched["data"] = date
                values.append(matched.set_index(keys)["observado"])
            targets[LABELS[target]] = pd.concat(values).reindex(targets.index)
        assert targets.shape == (168, 9), targets.shape
        assert np.isfinite(spectra.to_numpy()).all()
        cross = pd.DataFrame({name: spectra.corrwith(targets[name], method="pearson")
                              for name in targets}).T
        within = targets.corr(method="pearson", min_periods=3)
        valid = targets.notna().astype(int)
        counts = valid.T @ valid
        assert np.isfinite(cross.to_numpy()).all()
        assert np.allclose(within, within.T, equal_nan=True)
        assert np.allclose(np.diag(within), 1)
        assert np.nanmax(np.abs(cross.to_numpy())) <= 1 + 1e-12
        cross.to_csv(OUT / "pearson_traits_wavelengths.csv", sep=";", index_label="Trait")
        within.to_csv(OUT / "pearson_traits.csv", sep=";", index_label="Trait")
        counts.to_csv(OUT / "pairwise_sample_counts.csv", sep=";", index_label="Trait")
        targets.to_csv(OUT / "paired_physiological_observations.csv", sep=";")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(17.8, 6.7))
    grid = fig.add_gridspec(1, 4, width_ratios=[2.25, .055, 1.25, .055],
                           left=.135, right=.97, bottom=.29, top=.89, wspace=.30)
    ax_a, cb_a, ax_b, cb_b = [fig.add_subplot(grid[0, i]) for i in range(4)]
    cmap = "RdYlBu"
    image_a = ax_a.imshow(cross.to_numpy(), aspect="auto", interpolation="nearest",
                         cmap=cmap, vmin=-1, vmax=1,
                         extent=[wavelengths[0]-.5, wavelengths[-1]+.5, len(targets.columns)-.5, -.5],
                         rasterized=True)
    ax_a.set_yticks(range(len(targets.columns)), targets.columns)
    ax_a.set_xticks([400, 650, 900, 1150, 1400, 1650, 1900, 2150, 2450])
    ax_a.set_xlabel("Wavelength (nm)", labelpad=30)
    image_b = ax_b.imshow(within.to_numpy(), aspect="auto", interpolation="nearest",
                         cmap=cmap, vmin=-1, vmax=1, rasterized=True)
    short_labels = ["Chl total", "Chl a", "Chl b", "$g_s$", "RWC", "ETR", "E", "A", "$C_i$"]
    ax_b.set_xticks(range(len(targets.columns)), short_labels, rotation=90)
    ax_b.set_yticks(range(len(targets.columns)), short_labels)
    ax_b.tick_params(axis="both", labelsize=8)
    for ax, letter in [(ax_a, "(A)"), (ax_b, "(B)")]:
        ax.text(-.02, 1.065, letter, transform=ax.transAxes, fontsize=15, ha="left")
        ax.tick_params(length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
            spine.set_color("#555555")
    for im, cax in [(image_a, cb_a), (image_b, cb_b)]:
        fig.colorbar(im, cax=cax, ticks=np.linspace(-1, 1, 9), format="%.2f")
        cax.set_title("Pearson r", fontsize=10, pad=12)
        cax.tick_params(labelsize=8, length=2)
        cax.spines[:].set_linewidth(.5)
    # Visible spectrum guide followed by black for the infrared region, as in the reference.
    strip = ax_a.inset_axes([0, -.115, 1, .035])
    colors = np.zeros((1, len(wavelengths), 3))
    visible = wavelengths <= 780
    from matplotlib.colors import hsv_to_rgb
    hue = np.clip((780 - wavelengths[visible]) / 380 * .75, 0, .75)
    colors[0, visible] = hsv_to_rgb(np.column_stack([hue, np.ones(len(hue)), np.ones(len(hue))]))
    strip.imshow(colors, aspect="auto", extent=[400,2450,0,1])
    strip.set_axis_off()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"correlogram_final_en.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = f"""# Final notebook correlogram

Source: script_final/Pipeline_Final_Predicao.ipynb, cells 1, 3, 4, 6, 7 and 11.
Physiological targets: the nine targets defined in ALVOS_FISIOLOGICOS.
Spectra: 400?2450 nm; detector splice correction at 1000 and 1800 nm;
clipping at ?10000; interpolation; Savitzky?Golay window 11, polynomial 2;
SNV per individual spectrum, followed by morning plot-level means.
Matching: block ? genotype ? water treatment ? date, exactly as in the notebook.
All three genotypes and both water treatments are pooled across available dates.
There are {len(targets)} plot-date observations. Missing physiological observations
and dates without a target measurement are excluded pairwise, with no imputation.
Panel A: Pearson correlations between each observed physiological target and each
preprocessed spectral band. Panel B: pairwise Pearson correlations between targets.
Pearson follows the reference image; the notebook uses Spearman for band grouping.
The color scale is fixed at ?1 to +1: red negative, blue positive.
These are descriptive pooled correlations, without p-value or significance claims;
repeated measurements across dates are not independent biological replicates.
FWHM and VIP selection are not applied to this full-spectrum descriptive figure.
The wavelength strip is a schematic visible-to-infrared guide.
No physiological units were added because the notebook does not establish them for all targets.

Valid observations per target:
{targets.notna().sum().to_string()}

Pairwise sample sizes for Panel B: pairwise_sample_counts.csv.
"""
    (OUT / "README.md").write_text(metadata, encoding="utf-8")
    print(targets.notna().sum().to_string())
    print(OUT / "correlogram_final_en.png")


if __name__ == "__main__":
    main()
