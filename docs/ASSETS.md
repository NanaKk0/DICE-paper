# Research assets

| Use | Package |
| --- | --- |
| DICE software | DICE-software.zip |
| Figures, numerical observations and CPU reproduction | DICE-paper.zip |
| Eleven trained study models | DICE-models.zip |
| Labelled configurations and physical-validation inputs | DICE-datasets.zip |

The model and dataset packages share `research-assets/`. Extract the model
package first and the dataset package second into the same parent directory.
`DATASET_SUMMARY.csv` lists labelled datasets and
counts; `MODEL_IDENTITY.csv` and `MODEL_CARDS.json` identify the models;
`FILE_INDEX.csv` resolves the supplied payloads and exact duplicate aliases.

```sh
python restore_assets.py /path/to/research-assets --restore --require-all --verify
```

The dataset package contains target DFT references, final student training
sets, independent tests, the historical S940/S1320 student sets, and the
120-frame liquid–amorphous test. The 380 analysed additions correspond to
zero-based S1320 frames 940–1319. Structures, key inputs and saved trajectory
endpoints are linked to current figures in `validation/INDEX.csv`.

For figure reconstruction, use the source data bundled with this repository.
For frozen-model evaluation, use [MODEL_USE.md](../MODEL_USE.md).
The restricted archive is
[10.5281/zenodo.24433482](https://doi.org/10.5281/zenodo.24433482).
