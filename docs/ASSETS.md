# Trained models and labelled datasets

| Task | Download |
| --- | --- |
| Read/use the trained study models | DICE-models.zip |
| Read labelled configurations and locked tests | DICE-datasets.zip |
| Reconstruct paper figures, tables and numerical checks | DICE-paper.zip or the public DICE-paper repository |

The Zenodo deposit contains four ZIPs: DICE-software.zip, DICE-paper.zip,
DICE-models.zip and DICE-datasets.zip, accompanied by SHA256SUMS.txt.
Extract the model and dataset archives into the same parent directory; they share
research-assets/. DATASET_SUMMARY.csv lists labelled files and counts.
MODEL_IDENTITY.csv and MODEL_CARDS.json describe model roles, native type maps
and runtime requirements. FILE_INDEX.csv covers the supplied model/dataset files.

```sh
python restore_assets.py /path/to/research-assets --restore --require-all --verify
```

## Dataset contents

The datasets contain all three systems' target DFT references, final teacher-labelled
student geometries and fixed tests, plus the S940/S1320 historical student data and
the independent 120-frame liquid–amorphous test. Labels and frame order are unchanged.

For new training, prepare native training inputs and recipes for your chosen model
and runtime. Obtain pretrained DPA4/DPA4C weights from their official projects.
The software's [own-system guide](https://github.com/NanaKk0/DICE/blob/main/docs/own-system.md)
describes input preparation and its [native-model guide](https://github.com/NanaKk0/DICE/blob/main/docs/native-models.md)
describes model/runtime compatibility.

## Numerical verification

The paper repository bundles figure/table source data, repeat-level statistics,
sampling-work and timing records, and the inputs for its documented CPU numerical
checks. Run reproduce.py verify, figures or tables as described in README.md.
The source tables provide the controlled-experiment metrics and tail curves
for each configuration-selection repeat.

## Model evaluation

MODEL_USE.md describes evaluation of the six production students on the supplied
ordered tests. MODEL_CARDS.json records hardware/build requirements and native
species maps. FIRST_FRAME_REFERENCES.json supplies existing reference predictions
for checking a compatible installation.
