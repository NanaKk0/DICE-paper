# Optional frozen-potential evaluation

`reproduce.py` uses the archived numerical data. To evaluate the two final
production students for each system, unpack DICE-models.zip and
DICE-datasets.zip into one directory and use `evaluate_models.py` with a
compatible native DeePMD environment. The software repository's
`docs/native-models.md` provides model sources, recorded native versions and
CLI compatibility information.

The supplied `.pt2` exports target Linux x86_64, CUDA compute capability 8.9,
and AVX512 (some also AVX512_VNNI). Use the build and hardware requirements in
MODEL_CARDS.json. For another architecture, use the backend's supported
import/export procedure and compare numerical predictions before production.

First inspect the selected model and ordered test paths:

```bash
python evaluate_models.py --research-root ../research-assets \
  --system al2o3 --role dice_student_500k
```

Once the compatible native environment is qualified, explicitly evaluate the
first labelled test frame:

```bash
python evaluate_models.py --research-root ../research-assets \
  --system al2o3 --role dice_student_500k --evaluate --output ../al-first.json
```

Use `--limit 0` for the full locked test set, change `--system` to `ysz` or
`mil120`, and change `--role` to `direct_dft_student_500k` for the paired control.
YSZ's three source files are concatenated in DATASET_SUMMARY order (240+60+30).
All six full-test force-component RMSE values can then be compared with
`source-data/data/summary/production_force_RMSE.csv`. Use the first-frame check
to inspect predictions and the full test for aggregate RMSE. The saved DFT
frame supplies reference energy and Cartesian forces. Energies retain their
original zero, and species indices follow each model's native element map.

Outputs contain total energy, Cartesian forces, virial, the DFT reference
energy, native species mapping and pooled force-component RMSE. Coordinates
are Å, energies eV, forces eV Å⁻¹, and virials eV. This optional command performs
frozen-model inference and writes results to the selected output file.
Type-mapping and shape handling have CPU fixture tests; the archive checks
validate real-model paths and checksums.

## Existing reference predictions and deployment metadata

MODEL_CARDS.json gives the native type map (including the full foundation map),
active species indices, cutoff, architecture, compiled format and hardware
requirements for each exported model. Adapted-teacher pt2 files are standalone
inference exports with the adaptation already incorporated. Preserve the
model's native 118-element map and use the recorded target-species indices.

FIRST_FRAME_REFERENCES.json contains the six final students' **already archived
model predictions** for the first frame used by the commands above: unshifted
total energy, every Cartesian force, exact ordered input geometry, test checksum,
model checksum and source-parity file identities. These predictions were
extracted from the stored parity records. Compare your single-frame
energy/forces with the corresponding system/role entry, preserving atom order.

Use these reference predictions to check energy and force agreement on your
installation, with numerical tolerances appropriate to its precision and backend.
