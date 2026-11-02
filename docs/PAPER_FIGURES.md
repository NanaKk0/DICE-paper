# Rebuild the paper figures

Install `requirements.txt` and a licensed Arial font visible to Matplotlib.
CairoSVG is used only to export the editable Fig. 1 schematic; it also needs
the system Cairo library. The commands use the bundled numerical source data.
The locally tested rendering versions are listed in [FIGURE_ENVIRONMENT.md](FIGURE_ENVIRONMENT.md).

```sh
python reproduce.py figures --figure main:3 --output-dir /path/to/new-fig3
python reproduce.py figures --figure supplement:6 --output-dir /path/to/new-parity
python reproduce.py figures --output-dir /path/to/new-paper-figures
```

Use an empty output directory outside the source-data tree. Repeat `--figure`
to select several figures. IDs are `main:1` through `main:6` and `supplement:1`
through `supplement:16`. Outputs are editable SVG/PDF and 300 dpi PNG proofs;
`paper-figures.json` records the data used and explicit display selections.

## What is reconstructed

`paper_figures.py` adapts the original paper plotting loops to the bundled
portable data paths, preserving the data, panel content and statistics.
Use `figures` for data-driven reconstruction and `reexport` for the exact
approved artwork.

| Figures | Data and representation |
| --- | --- |
| Main 1 | Approved editable workflow SVG export |
| Main 2 | Cross-pool teacher errors; the two Random10 tails under one b label; representative repeat-1 support/error scatter; separate High40 paired tails |
| Main 3 | Teacher error; all 28 student rounds with the 3–10 addition interval; both element projections; both support distributions; stored AlO4/5/6 rigid overlays |
| Main 4 | All 1,660 configuration/model error observations; saved YSZ transition intervals at 0.30 eV/Å; matched main-path phonons; C→B profile; six fitted minima |
| Main 5 | Five-phase EOS; separate unshifted X-ray and neutron correlations; three-temperature adsorption; 8/10YSZ tracer-NE estimates and experimental DC curves |
| Main 6 | Three-system MD-step latency, saved technical spreads and observed OOM positions; formal development-sampling cost totals |
| Supplement 1–5 | Two allocation matrices, both unqueried-teacher metrics, absolute/paired Random10 tails, both TD partition metrics and High40 absolute tails |
| Supplement 6–8 | Production-student energy and force-component parity; separate models, shared DFT centering, all observations |
| Supplement 9–11 | Six MIL single-point EOS series on common DFT geometries, nine common-DFT-geometry profiles and nine independently optimized DICE CI-NEB profiles, kept distinct |
| Supplement 12–13 | Eleven-phase energy/volume comparison and full α/κ/θ dispersions including low and negative frequencies |
| Supplement 14–16 | Interface velocities plus complete repeat traces; normalized composition-dependent diffusion; measured lattice response and stored expansion coefficients |

## Frozen definitions and display selections

- Tails use the saved top 0.01–1% grid. Means are over the same five
  configuration-selection repeats; supplementary tail panels show every repeat.
- Random10 and Low10/High40 are separate arrays. High40 uses the saved
  geometry-clean evaluation membership.
- Main Fig. 2 support scatter uses all 413 rows of repeat 1 for both teacher
  snapshots. Teacher source-pool evaluation is a different population.
- Main Fig. 3 uses saved projections and histogram bins. Its overlay uses the
  saved aligned coordinates, correspondence and camera.
- Main matched phonons use Γ–M–K–Γ–A–L–H–A; supplementary multi-phase spectra
  retain the full archived paths. Disconnected segments remain separate, and
  the dispersions include all frequency modes.
- Main Fig. 4 error points are the archived per-frame metric. The white point
  marks the median; the violin shows the distribution of configuration errors.
- OOM crosses use observed atom counts and the saved display-only extensions.
  Their y positions are display-only guides; measured timings end at the
  last successful point.
- EOS energies and path zeros use the existing source-data columns.
- Statistical bands and error bars retain their archived definitions; see
  [NUMERICAL_DEFINITIONS.md](NUMERICAL_DEFINITIONS.md) and the paper captions.

## The three different actions

| Action | Result |
| --- | --- |
| `verify` / `replot` | Numerical checks / diagnostic verification figures, unchanged from the previous release |
| `figures` | Reconstructed paper panels from frozen source data, plus explicit schematic export |
| `reexport` | Exact copies of the approved SVG artwork |

Use `reexport` when the exact approved composition is needed, and `figures`
when changing a plotting parameter or inspecting the data-to-panel path.
`FIGURE_TABLE_MAP.csv` and `MAIN_PANEL_MAP.csv` give per-object commands and
source locations. Native inference remains a separate optional task described
in `MODEL_USE.md`.
