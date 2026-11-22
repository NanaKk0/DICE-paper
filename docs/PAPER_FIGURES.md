# Figure reconstruction

`source-data/artwork/` contains the twenty approved editable SVG files.
`reproduce.py reexport` copies those files exactly. `reproduce.py figures`
reconstructs the numerical panels in Python using the bundled observations.
Outputs are SVG, PDF and PNG, with a data-use report.

```sh
python reproduce.py figures --figure main:3 --output-dir /path/to/new-fig3
python reproduce.py figures --figure supplement:4 --output-dir /path/to/new-parity
```

| Object | Evidence |
| --- | --- |
| Main 1 | Conventional distillation and the DICE workflow; approved schematic |
| Main 2 | Matched 70-frame teacher-update comparisons; paired DFT-70 tails and teacher errors |
| Main 3 | QBC/GMM scores and errors in both pools; paired relative student P99 differences |
| Main 4 | Four structural groups, teacher/student accuracy, neighbourhood mixing and aligned AlO₄/₅/₆ examples |
| Main 5 | Production-error distributions, element-resolved high-error fractions, phonons, paths and relative energies |
| Main 6 | Al₂O₃ EOS and liquid correlations, YSZ conductivity and MIL-120 CO₂ adsorption |
| Main 7 | Model execution timings and startup-inclusive exploration-cost estimates |
| Supplement 1–2 | 70-frame teacher/student errors; routing decisions and paired RMSE differences |
| Supplement 3–5 | Al₂O₃, YSZ and MIL-120 energy/force parity |
| Supplement 6–9 | MIL-120 EOS and paths; Al₂O₃ phase comparisons and full phonons |
| Supplement 10–13 | Coexistence, diffusion, thermal response and optimised MIL-120 paths |

Current figure IDs are `main:1`–`main:7` and `supplement:1`–`supplement:13`.
[FIGURE_TABLE_MAP.csv](../FIGURE_TABLE_MAP.csv) resolves each figure to its
inputs and command. [MAIN_PANEL_MAP.csv](../MAIN_PANEL_MAP.csv) records current
main-panel letters. Plot labels use $T_0$, $T_R$ and $T_1$.

The numerical reconstructions preserve observations, repeat identities and
metric definitions. Exact composition and the approved illustrations are
available through `reexport`.
