# Reproducibility scope and data contract

## Paper context and splitting

DICE couples an adapted teacher, empirical-support routing and an expanded
student training set. The archived production comparison uses final 500k-step
students on common independent DFT test geometries, with all force components
pooled.

| System | Target DFT reference records | Final student training geometries | Independent test geometries |
| --- | ---: | ---: | ---: |
| Al2O3 | 140 | 2580 | 200 |
| YSZ | 195 | 4272 | 330 |
| MIL-120–CO2 | 494 | 6041 | 300 |

The archived files define the scientific splits. YSZ's production test is the union of 240 original, 60 additional
1200-K and 30 additional 1600-K geometries. The reference totals above include
30 YSZ dimer records and 24 MIL-120 isolated-reference records (20 CO₂ and four
atoms), respectively, within the totals of 195 and 494.
The labelled files themselves retain the ordered dataset contents.
DATASET_SUMMARY.csv lists the dataset files and aggregate counts.

The 120-geometry liquid–amorphous Al2O3 test and its historical 100k-step
students are a separate experiment. The reported expansion compares 940 with
1320 training geometries (rounds 2 and 10).
Teacher adaptation reduces that test's RMSE by 31.67%; expansion with the
adapted teacher fixed reduces the historical student's RMSE by 15.78%.

The Random10 and Low10/High40 controlled experiments are also distinct. Their
five repeats vary the selected configurations with training randomness held
constant. Random-route evaluation leaves each full R50 selection out:
456 crystal or 373 liquid–amorphous geometries. High40 evaluation uses the
geometry-clean membership: 373/372/373/372/373 geometries for repeats 1–5,
respectively. The score `p` measures empirical support relative to the
reference distribution; high support is `p > 0.1`, low support is `p <= 0.1`.

## Numerical definitions and checks

- Force-component RMSE: `sqrt(sum(force_ss) / sum(force_n))` over the common
  test. The direct-to-DICE reductions round to 47.6%, 21.9% and 18.8%.
- With optional norm caches, the same RMSE is
  `sqrt(mean(force_vector_error_norm**2) / 3)`. Vector P99 uses NumPy's default
  linearly interpolated 0.99 quantile. A top-`q` percent tail averages the
  largest `ceil(N*q/100)` vector errors; paired differences are route minus
  the matched DFT-60 baseline. Negative differences remain negative.
- Optional teacher raw-label checks compare T0/TR forces on forty unqueried
  geometries in each of ten matched cases, using the explicit membership.
  Stored-label results match their reconciliation values. Their small
  differences from the final plotted fresh-batch audit (up to about
  4.14e-7 eV/Angstrom RMSE and 4.15e-5 eV/Angstrom vector P99) are retained in
  `raw_teacher_remaining40_reconciliation.csv`.
  Twenty-two full-source-pool residual arrays independently reproduce the
  Fig. 2 teacher-pool audit, whose evaluation includes the queried configurations.
- The nine MIL-120 paths have eleven images each and use the same DFT
  geometries for both predictions. For `e_i = 1000*(E_model_i - E_DFT_i)`,
  remove `(1-t_i)*e_0 + t_i*e_10` and average absolute residuals over images
  1–9. `t_i` is the archived **DFT-path cumulative Cartesian arc fraction**.
  Units are meV per P40 framework cell.
- Phonon frequency-error statistics use the signed DFT-frequency predicate
  `reference_frequency_THz > 0.5`, retaining 9011 of 9030 grid entries.
  Direct and DICE MAEs are 0.2373848454 and 0.0490118193 THz. Dispersion
  plots retain near-zero modes. The main verification panel spans the full
  Gamma–M–K–Gamma–A–L–H–A path; Supplementary Fig. 13 retains all archived
  segments for alpha, kappa and theta phases.
- Development scope is 148 blocks, 75038 tasks and 26 sampling modes.
  YSZ costs are 6.790833 recorded student hours and 199.901410 estimated
  teacher hours. Teacher costs are estimated for the recorded student
  workload. The current registry includes
  75038 compact task-work records and the 26 native timing receipts. Work is
  reconstructed by task type; rates are the median of three intervals of
  summed client work divided by the slowest client's elapsed time; teacher
  time is work/rate. Student system totals independently sum 148 recorded
  block times. Within-block student mode allocations retain the archived
  native-timing weights. Rates use the final confirmation measurements at
  the selected concurrency. For six of 26 modes, the confirmation rate is
  lower than an earlier scan maximum; the final confirmation rates are the
  inputs to the published cost totals.
- Five-repeat bands show mean ± sample SD. Other intervals, quartiles and
  preparation-level bootstrap intervals are retained from archived source
  data.
- Final YSZ atom-error transitions are reconstructed from the final 500k
  Cartesian parity and saved 1,000-by-42 paired cluster resampling weights.
  The older metadata column `enorm_C100` belongs to a historical model.
  Final-pair verification uses the current Cartesian parity, writes an
  explicitly named final-pair atom table and replays all twelve rows.
- Conductivity verification uses all 100 preparation-level diffusivities,
  oxygen counts and volumes, charge 2e and SI constants to reconstruct 25 NE
  condition means and sample SDs. Sixteen Weller readings and pixel-reading
  bounds are replayed from actual anchors in the original 1000/T–ln(σT)
  coordinates, without extrapolation.

## Figure and table coverage

`plot_manifest.json` lists the actual source files used for every generated
grid and records panel titles. Main Fig. 2 covers signed tails, teacher-pool
errors and support transitions; Fig. 3 covers teachers, historical expansion,
archived embeddings, support and local bond distances; Fig. 4 covers pooled
production errors, YSZ transitions, phonons, path errors and phase ordering;
Fig. 5 covers EOS, pair distributions, adsorption and conductivity; Fig. 6
covers latency and development cost. Supplementary Figs. 1–16 each have a
source-backed verification counterpart, including parity/residual/ECDF
triplets, all nine common paths, relaxed NEB profiles, phase energies,
phonons, interface traces, transport and thermal expansion.

`figures` reconstructs the paper panels from the bundled numerical data;
`replot` produces diagnostic verification plots; `reexport` supplies the
exact approved SVG artwork. Embeddings use archived coordinates. Latency
measurements end at the last successful point, with display-only extensions
to observed OOM sizes. Adsorption displays all three repeats. Conductivity
compares simulated 8/10YSZ with experimental 7.8/10.3YSZ. The SI16 verification
uses TMA coefficients for the same 303.15–1173.15 K interval as the simulation
fit. The source CSV also provides HT-XRD values for their respective intervals.

`tables` exports the twelve final table CSVs exactly, and the verification
commands write numerical summaries separately. `reexport` copies all 22
approved SVGs exactly.

Actual input provenance is also supplied separately from numerical plotting.
The research archive's `sampling/CONDITION_INDEX.csv` resolves all161 accepted
representative conditions to inputs/seeds, including Python-native GCMC/Widom
requests and ASE-relaxation effective contexts. Its
`validation_records/native_physics/` contains the actual six 864-cation heating
preparations and native heating/coexistence input/restart bindings. The local
CPU-only `replay_thermal_observations.py` rebuilds the thermal bins and fits
from74940 preserved native heating observations; the README gives its command.
Original generators and launchers provide the corresponding execution records.
`validation_records/native_protocols/` binds all plotted size-benchmark points
to actual native inputs, includes the four original YSZ protocol files matched
to all100 production receipts, and retains the distinct model-optimized
MIL CI-NEB/FIRE setup and eighteen path summaries. These protocol records
are separate from both the concurrency cost study and the common-DFT-geometry
single-point path comparisons.

## Execution and portability

Only files beneath the supplied archive roots are read; historical absolute
paths inside CSV cells are inert provenance strings. Outputs must be in a
non-overlapping dedicated directory. The registry file controls current
figure numbering and resolves historical source-data filenames.
`reproduce.py` reads local source data and writes derived checks, tables and
figures to the selected output directory. For native model evaluation, use
the separate `evaluate_models.py --evaluate` entry described in MODEL_USE.md.

After installing the analysis requirements and Arial, run the numerical and
figure-contract tests from the paper repository:

```sh
python -m unittest discover -s tests -v
```

The tested analysis environment used Python 3.10, NumPy 1.26, pandas 2.3 and
Matplotlib 3.10; exact versions are in `requirements-tested.txt`.
Font metrics and PNG antialiasing can vary between environments. Numerical
checks compare the source values and statistics. The full paper-panel rendering
environment is recorded in FIGURE_ENVIRONMENT.md.
