# Model use

The archive contains three adapted DPA4-Neo teachers, six final DPA4C-Nano
students and two Al₂O₃ learning-stage students. Teachers retain the EMA weights
used in the study. `research-assets/MODEL_IDENTITY.csv` lists files, roles and
training steps.

The fine-tuning starting models were OMat24-pretrained DPA4-Neo
(`DPA4-Neo-OMat24-v20260627.pt`) and DPA4C-Nano
(`DPA4C-Nano-zeng222-model.ckpt-1000000.pt`). The
[DPA4-Neo author release](https://huggingface.co/OutisLi/DPA4-OMat24-v20260627/tree/main)
and [DPA4C documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html)
identify the model families and native recipes.

## Environment and model reading

The checkpoints were checked with DeePMD-kit **3.2.0+g3b2555c46** (revision
`3b2555c46ecdbd2fcc8ddc029d09195a4e3c0f3a`), Python 3.11 and PyTorch
2.11.0+cu130. Native LAMMPS inputs use the 22 Jul 2025 Update 4 interface
and Kokkos 4.6.2. DPA4 uses DeePMD's `pt` backend; DPA4C uses `pt-expt`.

```python
from deepmd.infer import DeepPot
potential = DeepPot("research-assets/models/al2o3/dice_student_500k.pt")
species = potential.get_type_map()
```

Read the checkpoint's native element map and map chemical symbols to its
indices. The checkpoints preserve the full native map. Coordinates use Å,
energies eV and forces eV Å⁻¹. LAMMPS atom types follow each input's
`pair_coeff` element order. Native LAMMPS inputs take the deployed potential
file through the `model` command-line variable.

## Student evaluation

Extract `DICE-models.zip` and `DICE-datasets.zip` together. Run the evaluation
entry point from `DICE-paper.zip`:

```sh
python evaluate_models.py --research-root ../research-assets \
  --system al2o3 --role dice_student_500k --evaluate --limit 0 \
  --output ../dice-results/al2o3.json
```

Use `ysz` or `mil120` for the other systems, and
`direct_dft_student_500k` for the paired direct-fine-tuning control. The ordered
YSZ test union contains 240 + 60 + 30 configurations. Evaluation retains atom
order and reports energies, Cartesian forces, virials and pooled errors.
