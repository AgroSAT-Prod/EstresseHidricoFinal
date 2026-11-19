# Final notebook correlogram

Source: script_final/Pipeline_Final_Predicao.ipynb, cells 1, 3, 4, 6, 7 and 11.
Physiological targets: the nine targets defined in ALVOS_FISIOLOGICOS.
Spectra: 400?2450 nm; detector splice correction at 1000 and 1800 nm;
clipping at ?10000; interpolation; Savitzky?Golay window 11, polynomial 2;
SNV per individual spectrum, followed by morning plot-level means.
Matching: block ? genotype ? water treatment ? date, exactly as in the notebook.
All three genotypes and both water treatments are pooled across available dates.
There are 168 plot-date observations. Missing physiological observations
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
Total chlorophyll          168
Chlorophyll a              168
Chlorophyll b              168
Stomatal conductance       168
Relative water content      71
Electron transport rate     72
Transpiration rate         168
Net CO2 assimilation       168
Intercellular CO2          168

Pairwise sample sizes for Panel B: pairwise_sample_counts.csv.
