# DICE: paper figures, tables and source data

This repository contains the numerical source data and CPU-only reproduction
code for the DICE paper. The commands below run on CPU using the bundled data.

[DICE software and user guide](https://github.com/NanaKk0/DICE) ·
[Main and supplementary figures](source-data/artwork/) ·
[Source tables](source-data/tables/)

## Start

Use Python 3.10 or newer. For `figures`, first install a licensed Arial font visible to Matplotlib and
the system Cairo library. The Python rendering dependencies are included in
`requirements.txt`.

From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify --output-dir /path/to/results/verify
python reproduce.py figures --output-dir /path/to/results/paper-figures
python reproduce.py replot --output-dir /path/to/results/replot
python reproduce.py tables --output-dir /path/to/results/tables
```

Use a separate output directory outside source-data. requirements-tested.txt
records the Python 3.10.20 environment used for local CPU checks.

## Find a figure or table

- [FIGURE_TABLE_MAP.csv](FIGURE_TABLE_MAP.csv): all six main figures, main
  Table 1, sixteen supplementary figures and twelve supplementary tables.
- [MAIN_PANEL_MAP.csv](MAIN_PANEL_MAP.csv): current main-panel letters and their
  exact data files, including old filename aliases such as Fig2f → Fig. 2d.
- source-data/data/: observations and repeat-level numerical evidence.
- source-data/tables/: exact supplementary table CSVs.
- source-data/artwork/: all 22 approved editable SVGs.
- source-data/DATA_REGISTRY.json: precise object mappings and integrity checks.

`figures` rebuilds the quantitative paper panels from their frozen data and
exports the editable schematic. Select one object, for example:

```sh
python reproduce.py figures --figure main:3 --output-dir /path/to/new-fig3
```

It needs a licensed Arial font and CairoSVG/system Cairo for schematic export.
The outputs are SVG/PDF/PNG and a data-use report in your selected directory.
See [the figure guide](docs/PAPER_FIGURES.md) for
panel coverage, statistical definitions and display selections.

`replot` produces 21 source-backed numerical verification figures.
To export the exact approved artwork:

```sh
python reproduce.py reexport --output-dir /path/to/results/artwork
```

The schematic and local-environment illustrations are available as editable SVGs.
Scientific data, repeat identities and statistical masks follow the archived definitions; see
[numerical definitions](docs/NUMERICAL_DEFINITIONS.md).

Export the main-text dataset-count table from its existing numerical source:

```sh
python reproduce.py main_table --output-dir /path/to/results/main-table
```

## Trained models and labelled datasets

DICE-models.zip supplies eleven trained study models. DICE-datasets.zip
supplies labelled training/reference/test configurations, including the historical
cross-phase student datasets and the independent 120-frame liquid–amorphous test.
See [ASSETS.md](docs/ASSETS.md) for their contents and [MODEL_USE.md](MODEL_USE.md)
for frozen-model evaluation with a compatible native environment.

Extract these two archives into the same parent directory and verify their
shared research-assets/ directory:

```sh
python restore_assets.py /path/to/research-assets --restore --require-all --verify
python reproduce.py verify --output-dir /path/to/results/verify
```

The figure/table reconstruction commands use the numerical source data bundled
with this repository.

The research-archive identifier is
[10.5281/zenodo.23885754](https://doi.org/10.5281/zenodo.23906093).

## Licences

Source code is licensed under MIT. Author-owned data, figures and documentation
are licensed under CC BY 4.0. Study model weights carry CC BY-NC 4.0.
See [LICENSING.md](LICENSING.md) for scope and attribution.

## Physical-validation inputs

[Structure and endpoint index](docs/VALIDATION.md) links the paper figures
to the native inputs and saved configurations in DICE-datasets.zip.
