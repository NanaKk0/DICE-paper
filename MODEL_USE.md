# Trained model checkpoints

`DICE-models.zip` provides eleven trained PyTorch checkpoints (`.pt`): three
adapted DPA4 teachers, six final DPA4C students, and the S940/S1320 Al₂O₃
students. Teachers retain the EMA weights used for the paper exports. Every
checkpoint preserves the full native 118-element map and trained parameters.

## Tested environment

DeePMD-kit **3.2.0+g3b2555c46**, source revision
`3b2555c46ecdbd2fcc8ddc029d09195a4e3c0f3a`, Python 3.11,
PyTorch 2.11.0+cu130, CUDA 13.0 and GCC 12.4. The tested LAMMPS build is
22 Jul 2025 Update 4 with Kokkos 4.6.2 CUDA/Serial. The complete version record
is [NATIVE_ENVIRONMENT.json](NATIVE_ENVIRONMENT.json).

Use the `pt` backend for DPA4 and `pt-expt` for DPA4C. Checkpoint architecture,
species indices and file identities are in `MODEL_CARDS.json` and
`MODEL_IDENTITY.csv`.

## Evaluate a student checkpoint

Extract `DICE-models.zip` and `DICE-datasets.zip` into the same parent directory,
then activate the tested native environment. Inspect the model and test paths:

```sh
python evaluate_models.py --research-root ../research-assets \
  --system al2o3 --role dice_student_500k
```

Evaluate the first independent test frame on the selected GPU:

```sh
CUDA_VISIBLE_DEVICES=0 DP_INTERFACE_PREC=high \
python evaluate_models.py --research-root ../research-assets \
  --system al2o3 --role dice_student_500k --evaluate \
  --output ../al-first.json
```

Use `--limit 0` for the full test set, `--system ysz` or `mil120` for the other
systems, and `--role direct_dft_student_500k` for the paired control. YSZ's
ordered test union contains 240+60+30 frames. Outputs contain total energy,
Cartesian forces, virial, reference energy and force-component RMSE, using
Å, eV and eV Å⁻¹. Preserve the native species mapping and atom order.

`FIRST_FRAME_REFERENCES.json` contains predictions from the supplied student
checkpoints in the tested environment, with exact ordered geometries and model
identities. It provides an installation check for energy, forces and virial.

## Export for LAMMPS on your machine

Freeze and compress on the GPU and toolchain used for deployment. Set the
inference policy before export:

```sh
export CUDA_VISIBLE_DEVICES=0
export DP_INTERFACE_PREC=high
export DP_CUDA_INFER=0 DP_TRITON_INFER=0 DP_AMP_INFER=0 DP_TF32_INFER=0
```

For a DPA4 teacher:

```sh
dp --pt freeze -c ../research-assets/models/al2o3/adapted_teacher.pt \
  -o teacher.pt2
```

For a DPA4C student:

```sh
dp --pt-expt freeze -c ../research-assets/models/al2o3/dice_student_500k.pt \
  -o student-frozen.pt2 --lower-kind graph
dp --pt-expt compress -i student-frozen.pt2 -o student.pt2 -s 0.01
```

The resulting local `.pt2` serves `pair_style deepmd` or `pair_style deepmd/kk`.
For Kokkos, use `atom_modify map yes` and a CUDA-enabled Kokkos LAMMPS build.
Map each LAMMPS atom type explicitly with `pair_coeff * *`, for example `Al O`
when the data file uses Al as type 1 and O as type 2. DPA4 teachers use freezing;
DPA4C students support both freezing and compression.

The published model payload consists of checkpoints. Compiled deployment files
are generated locally for the user's hardware. Native export and twenty-step
ordinary/Kokkos LAMMPS checks passed for all eight students in the tested build.
