# Optional research assets

| Task | Download |
| --- | --- |
| Read/use the final trained models | DICE-models.zip |
| Read labelled configurations and locked tests | DICE-datasets.zip |
| Replay retained detailed numerical and original-output evidence | All three asset archives |

Extract selected archives into one parent directory; they share research-assets/.
DATASET_SUMMARY.csv gives dataset files and aggregate counts.
Model roles/hashes and static native type maps are in MODEL_IDENTITY.csv
and MODEL_CARDS.json. Byte-identical payloads are stored once; FILE_INDEX.csv and
restore_assets.py recover the logical paths required by deeper analyses:

```sh
python restore_assets.py /path/to/research-assets --restore --verify
```

Use --require-all only when all three asset archives have been extracted.
The helper restores local byte-identical paths and reports conflicting files.

## Dataset contents

Retained datasets include all three systems' target DFT references, final labelled
student geometries and fixed tests, plus S940/S1320 historical student data and
the independent 120-frame liquid-amorphous test. Labels and frame order are unchanged.

For new training, prepare native training inputs and recipes for your selected
model and runtime. Obtain pretrained DPA4/DPA4C weights from their official
projects. The software's [own-system guide](https://github.com/NanaKk0/DICE/blob/main/docs/own-system.md)
explains input preparation and its [native-model guide](https://github.com/NanaKk0/DICE/blob/main/docs/native-models.md)
describes model/runtime compatibility.

Evaluation masks, repeat identifiers and observation-to-original-output mappings
are included for reproducing the published numerical statistics across all
retained observations and repeats.

## Model evaluation and provenance

Use MODEL_CARDS.json for model roles and hardware requirements, and MODEL_USE.md
for frozen-model evaluation. Provenance records describe the available model
and dataset identities. Per-frame teacher attribution for historical Al2O3/YSZ
labels remains incomplete. Numerical definitions describe the exact performance
accounting inputs, including the confirmation measurements used for each mode.
