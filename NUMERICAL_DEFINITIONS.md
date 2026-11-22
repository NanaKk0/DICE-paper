# Numerical definitions

## Teacher updating: 70 geometries

The initial teacher uses ten γ-Smrčok DFT references. The remaining γ pool
contains 506 configurations; the liquid–amorphous pool contains 423.
For each of five selections, sixty pool configurations augment the initial ten.
Ten of those sixty are selected for additional DFT references; fifty remain
unqueried for teacher training. Held-out evaluation uses 446 γ configurations
or 363 liquid–amorphous configurations, shared by all compared routes.

- $T_0$: initial-teacher labels on all seventy geometries.
- Hybrid: DFT labels on the ten newly queried geometries, $T_0$ labels on the other sixty.
- $T_R$: teacher fine-tuned from the pretrained model on the twenty DFT references, then used to label all seventy geometries.
- DFT-20: student trained on the twenty DFT references.
- DFT-70: DFT labels on all seventy geometries.

Teacher errors are supplied for all seventy geometries, the fifty unqueried
additions, and the held-out pool. Fig. 2 teacher bars separately evaluate the
remaining 506-frame and 423-frame source pools. Atomic force-vector tails use
the highest-error fraction of atom–frame observations, with ceiling selection
counts. Differences are calculated against DFT-70 within each repeat, then averaged.

## Teacher-label routing

Configuration high error is force-component RMSE >0.05 eV Å⁻¹.
QBC accepts σ at or below 0.03, 0.05 or 0.08 eV Å⁻¹; GMM accepts $p>0.1$.
The accepted-set high-error fraction is FN/(FN+TN), the fraction of
accepted teacher-label configurations above the error criterion.

Each expansion targets fifty accepted configurations, using the full accepted
set when smaller. QBC student comparisons use sixty geometries. Initial GMM
comparisons use sixty γ or fifteen liquid–amorphous geometries. Adding ten
selected low-support DFT references updates the teacher to $T_1$; fifty further
accepted geometries expand these student sets to 120 or 75. The updated teacher
labels the expanded student geometries, including the newly queried ten.

All teacher-label/DFT-label pairs share training geometries and evaluation
frames. Relative P99 difference is calculated within each repeat as
100 × (teacher-label student P99 / DFT-label student P99 − 1).
Means and sample standard deviations summarise five paired repeats. The
initial fifteen-frame GMM comparison uses fixed geometries and five training
seeds; the other routed comparisons vary configuration selection.

## Structure and student learning

The four structure groups contain 140 teacher references, 940 earlier student
geometries, 380 additions and 120 independent tests. The additions are
zero-based frames 940–1319 of `crossphase/S1320_training.extxyz`; its first
940 frames form the earlier student training set. In iteration 6,
the teacher is fixed while student cycles r0–r27 are evaluated; the additions
enter during r3–r10. The saved t-SNE coordinates retain every Al and O point.
Neighbourhood mixing is 4u(1−u), where u is the teacher fraction after
source-count normalisation within radius eight in projection units; each
query point is excluded from its own neighbourhood. AlO₄/₅/₆ examples use the
saved atom correspondence and aligned coordinates.

## Production and physical validation

Pooled force-component RMSE is the square root of summed Cartesian residual
squares divided by the number of force components. Force-vector P99 uses
linear quantile interpolation over atomic vector-error norms. YSZ final-model
errors are reconstructed from the ordered final Cartesian parity file; the
element and cluster metadata supply atom identities. Saved cluster weights
replay the existing 1,000 paired bootstrap samples over 42 source clusters.

Phonon comparisons preserve the saved DFT geometries and paths. The main
frequency-error comparison uses DFT frequencies above 0.5 THz; complete
dispersions include all saved modes. Common-DFT-geometry path energies and
independently optimised CI-NEB paths have separate inputs and figure entries.
The transport module recalculates Nernst–Einstein conductivity from the
archived diffusion summaries and reads the existing preparation-level intervals.

## Exploration costs

Recorded student durations cover the 148 retained exploration blocks,
75,038 task records and 26 modes. For every one of the 219 block/mode
allocations, teacher time is work divided by the remeasured native execution
rate, plus measured startup, exit, hybrid subprocess and common preparation /
validation costs. Concurrent startup waves and persistent relaxation pools
follow the archived native execution protocol. `cost_reanalysis.py` verifies
the individual work counters and reconstructs the three system totals.
