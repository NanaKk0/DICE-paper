# Physical-validation structures and inputs

Download DICE-datasets.zip from the [research archive](https://doi.org/10.5281/zenodo.24434161) and
extract it alongside DICE-models.zip. Structures and native inputs are under
research-assets/validation/. [INDEX.csv](../source-data/validation/INDEX.csv)
links each file to the relevant figure or stability table and records its role,
conditions and stored-frame timestep. Figure and table numbers follow the
22 November 2026 manuscript. Paths beginning validation/ are relative
to research-assets/ in the extracted dataset archive.

The collection includes Al2O3 EOS and phase structures, phonon references and
displacements, nine MIL-120 reference and DICE-optimised paths, and saved
initial/final configurations for liquid, coexistence, adsorption and YSZ
transport/heating calculations. The archive README describes native formats,
particle types and units. FILE_PATHS.csv maps the recorded native input paths
to the corresponding archive files. Use MODEL_USE.md for the study models.

The path-profile tables link geometry and prediction files directly to this
collection. EOS rows identify their structures and, for MIL-120, the zero-based
frame in the labelled 42-structure reference file. Numerical energies and
statistical results retain their original values.
