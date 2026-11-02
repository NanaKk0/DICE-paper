# DICE paper source data

Frozen numerical inputs for the current manuscript: six main figures, sixteen supplementary figures and twelve supplementary tables. `DATA_REGISTRY.json` is the figure/table-to-file index and records integrity digests. Identical source inputs are stored once; aliases preserve the mapping to the original analysis filenames.

## Contents

- `data/`: numerical observations, paired/repeat summaries, fixed membership definitions and necessary provenance, grouped by scientific purpose.
- `tables/`: the twelve exact English supplementary tables exported as CSV, with definitions in the registry.
- `artwork/`: the twenty-two approved editable SVG figures.
- `references/`: current main-text and supplementary reference bibliographies.
- `DATA_REGISTRY.json`: current figure/table numbers, input mappings and per-file integrity records. Historical numbers embedded in source filenames are provenance only; use this index for current numbering.

The parent paper repository includes reproduce.py, plotting code and numerical
definitions. Run its documented CPU commands using the bundled data.
Trained models and labelled datasets are separate downloads described in
../docs/ASSETS.md. The documented figure/table commands use the bundled source data.

## Statistical and physical conventions

Force-component RMSE pools squared Cartesian component residuals. Error tails use atomic force-vector residual magnitudes. The controlled experiments' five repeats vary configuration selection with training randomness held constant. Random10 and support-guided Low10/High40 use their respective evaluation membership and teacher updates.

The historical cross-phase expansion students are trained for 100k steps. Their evaluation uses 120 liquid–amorphous test structures held out from the 380 training additions. Empirical support measures compatibility with the model's reference distribution.

The common-geometry MIL-120 pathway metric removes the endpoint-linear energy error along the cumulative DFT-path displacement coordinate before averaging the nine interior-image absolute residuals. It is distinct from barriers obtained by model-relaxed CI-NEB. Phonon error statistics use the archived matched modes and DFT frequencies above 0.5 THz; full plotted dispersions retain other modes.

YSZ conductivity uses the oxygen-tracer Nernst–Einstein estimate, compared with digitised experimental DC curves. Transport uncertainty is evaluated over whole independent preparations. Experimental comparison bounds describe the sensitivity of curve digitisation.

Development-sampling cost accounting uses the formal 148 accepted blocks, 75,038 tasks and 26 modes. Full-MD-step timing and inference-only benchmarks have their own datasets and definitions.

## Provenance fields and scope

Some original metadata contain historical local paths or model aliases for provenance. The portable entry point resolves inputs through package-relative paths. Scientific model names and current object numbers are defined by the registry and companion research-asset index.

The authors will finalize licensing and public record addresses before publication. Obtain external tools and fonts under their applicable terms.
