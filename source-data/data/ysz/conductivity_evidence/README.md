# Conductivity calculation inputs and experimental digitisation

`preparation_inputs.csv` selects the unchanged final 5 ns values for all 100
preparations. D_star is the oxygen tracer diffusivity in m² s⁻¹; volume_A3 is
the preparation's volume in Å³; O is its oxygen-ion count. Each row permits
independent evaluation of n_O = O/(volume_A3 × 10⁻³⁰) and
σ_NE = n_O(2e)²D_star/(k_B T), dividing S m⁻¹ by 100 for S cm⁻¹.
Condition means and sample standard deviations use four preparations.

The three digitisation files contain the existing manually selected curve-center
anchors, axis calibration, and interpolated experimental readings from Weller
et al., Solid State Ionics 175, 409–413 (2004), Fig. 2a,
https://doi.org/10.1016/j.ssi.2003.12.044.
The original axes are 1000/T and ln(σT/[K S cm⁻¹]); interpolation is piecewise
linear in those coordinates, with no composition interpolation or extrapolation
to 1600 K. The bounds describe pixel-reading sensitivity. The calibration file
identifies the source figure and original pixel dimensions; its image path
records historical provenance. Use the published figure for visual reference.

The companion `dice-paper/transport_reanalysis.py` reconstructs both the NE
values and digitised readings from these portable inputs.
