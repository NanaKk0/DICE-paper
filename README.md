# DICE paper data and analysis

Numerical source data and analysis code accompanying the DICE study.

## Figures and statistics

Use Python 3.10 or newer. Figure rendering uses Arial and the system Cairo
library through CairoSVG.

```sh
python -m pip install -r requirements.txt
python reproduce.py figures --output-dir ../dice-results/figures
python reproduce.py summary --output-dir ../dice-results/statistics
python reproduce.py tables --output-dir ../dice-results/tables
python reproduce.py training_sets --output-dir ../dice-results/training-sets
```

Select one figure with `--figure main:3` or `--figure supplement:2` on the
`figures` command. Numerical reconstruction preserves the data and panel assignments;
`reexport` copies the exact approved editable SVG artwork.

- [DATA_REGISTRY.json](source-data/DATA_REGISTRY.json) maps the six main
  figures, thirteen supplementary figures and five supplementary tables to inputs.
- [Numerical definitions](NUMERICAL_DEFINITIONS.md) specifies comparisons,
  statistical calculations, structural analyses and exploration-cost accounting.
- `source-data/data/` contains the numerical observations and repeat-level results.
- `source-data/artwork/` contains editable SVG figures.

## Models and configurations

The [research archive](https://doi.org/10.5281/zenodo.24456706) contains:

| Package | Contents |
| --- | --- |
| DICE-paper.zip | This repository |
| DICE-models.zip | Eleven trained `.pt` checkpoints and their model index |
| DICE-datasets.zip | Labelled reference/training/test configurations and physical-validation inputs |
| DICE-software.zip | The DICE workflow software |

Extract the model and dataset packages into the same directory. Their contents
use the shared `research-assets/` directory. `MODEL_IDENTITY.csv` identifies
checkpoints and `DATASET_SUMMARY.csv` identifies datasets and frame ranges.
[MODEL_USE.md](MODEL_USE.md) gives the model-reading and evaluation entry points.

Physical-validation files are under `research-assets/validation/`;
`validation/INDEX.csv` maps structures and calculation inputs to figures.
Paths in this index are relative to `research-assets/`.

The material name is MIL-120(Al); `mil120`, `MIL120`, `MIL` and
`MOF-MIL-120` remain stable data and model identifiers.

## Licences

Code: MIT. Author-owned data, figures and documentation: CC BY 4.0.
Model terms and attribution are in [MODEL_LICENSE.md](MODEL_LICENSE.md).
See [LICENSING.md](LICENSING.md) for scope.
