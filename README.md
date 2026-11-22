# DICE paper evidence and reproduction

Source data, approved figures and CPU reproduction code for the DICE study.
This snapshot follows the manuscript and Supplementary Information of 22 November 2026.

[DICE software](https://github.com/NanaKk0/DICE) ·
[Figure and table index](FIGURE_TABLE_MAP.csv) ·
[Approved artwork](source-data/artwork/) ·
[Labelled datasets and models](docs/ASSETS.md)

## Reproduce

Use Python 3.10 or newer, install `requirements.txt`, and provide a licensed
Arial font visible to Matplotlib. CairoSVG also uses the system Cairo library.

```sh
python -m pip install -r requirements.txt
python reproduce.py verify --output-dir /path/to/results/verify
python reproduce.py figures --output-dir /path/to/results/figures
python reproduce.py tables --output-dir /path/to/results/tables
python reproduce.py main_table --output-dir /path/to/results/main-table
```

The commands use the numerical inputs in `source-data/`. Choose an output
directory outside the code and data directories. Select one figure with
`--figure main:3` or `--figure supplement:2` on the `figures` command.

## Contents

- Seven main figures and thirteen supplementary figures, including editable SVG artwork.
- Main Table 1 and five supplementary tables.
- Five-repeat 70-frame teacher-update comparisons and QBC/GMM student comparisons.
- Structural, local-environment and learning-curve evidence for Al₂O₃.
- Production-model accuracy and material-property observations for three systems.
- Development-exploration work records and startup-inclusive teacher-time estimates.

`verify` recalculates quantitative summaries and checks the registered files.
`figures` reconstructs numerical panels from their observations; `reexport`
copies the exact approved artwork. See [the figure guide](docs/PAPER_FIGURES.md)
and [numerical definitions](docs/NUMERICAL_DEFINITIONS.md).

```sh
python reproduce.py reexport --output-dir /path/to/results/approved-artwork
python -m unittest discover -s tests
```

## Research assets

The archive comprises `DICE-software.zip`, `DICE-paper.zip`,
`DICE-models.zip` and `DICE-datasets.zip`. The model and dataset packages
share the extracted `research-assets/` directory. Extract the model package
first and the dataset package second. They contain eleven
trained models, labelled reference/training/test configurations, and the
structures and inputs indexed in [VALIDATION.md](docs/VALIDATION.md).

```sh
python restore_assets.py /path/to/research-assets --restore --require-all --verify
```

[MODEL_USE.md](MODEL_USE.md) describes frozen-model evaluation.
The restricted research archive is
[10.5281/zenodo.24433482](https://doi.org/10.5281/zenodo.24433482).

## Licences

Code: MIT. Author-owned data, figures and documentation: CC BY 4.0.
Model weights: the terms and attribution in [MODEL_LICENSE.md](MODEL_LICENSE.md).
See [LICENSING.md](LICENSING.md) for scope.
