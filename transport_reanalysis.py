"""Reconstruct archived NE conductivities and original-coordinate digitisation."""
import numpy as np
from reproduce import assert_close


def verify_transport(a, output):
    base = a.registry['conductivity_evidence']['directory']
    p = a.csv(base + '/preparation_inputs.csv')
    if len(p) != 100 or p.groupby(['T', 'comp']).size().ne(4).any():
        raise ValueError('Expected four preparations for each of 25 conditions')
    density = p.O / (p.volume_A3 * 1e-30)
    sigma = density * (2 * 1.602176634e-19)**2 * p.D_star / (1.380649e-23 * p['T']) / 100
    assert_close(density, p.oxygen_density_m3, 'Oxygen-ion number density')
    assert_close(sigma, p.sigma_NE_S_cm, 'Preparation-level NE conductivity')
    p = p.assign(recomputed_sigma_NE_S_cm=sigma)
    means = p.groupby(['T','comp']).agg(mean=('recomputed_sigma_NE_S_cm','mean'),
                                      SD=('recomputed_sigma_NE_S_cm','std'))
    expected = a.csv('data/ysz/NE_summary.csv').set_index(['T','comp']).loc[means.index]
    assert_close(means['mean'], expected.sigma_NE_mean_S_cm, 'NE condition means')
    assert_close(means.SD, expected.sigma_NE_SD_S_cm, 'NE condition sample SD')
    p.to_csv(output/'NE_preparation_reanalysis.csv', index=False)
    config = a.json(base + '/calibration_and_anchors.json')
    ax, bx = np.polyfit(*np.array(config['x_axis_calibration']).T, 1)
    ay, by = np.polyfit(*np.array(config['y_axis_calibration']).T, 1)
    expected = a.csv(base + '/experimental_conductivity_1200_1500K.csv')
    rows = []
    for curve in config['curves']:
        xy = np.array(curve['xy'], dtype=float)
        if not (np.diff(xy[:,0]) > 0).all():
            raise ValueError('Unsorted digitisation anchors')
        for t in config['target_temperatures_K']:
            xp = (1000/t-bx)/ax
            if not xy[:,0].min() < xp < xy[:,0].max():
                raise ValueError('Experimental extrapolation forbidden')
            yp = np.interp(xp, xy[:,0], xy[:,1])
            value = np.exp(ay*yp+by)/t
            e = expected[(expected.T_K == t) & (expected.experimental_mol_percent == curve['mol_percent'])]
            if len(e) != 1:
                raise ValueError('Missing experimental comparison row')
            bound = config['digitization_bound']
            dx = bound['x_px']
            dy = bound['y_px_above_1450K' if t > 1450 else 'y_px'] + bound['extra_y_calibration_px']
            corners = [np.exp(ay*(np.interp(xp+sx*dx,xy[:,0],xy[:,1])+sy*dy)+by)/t
                       for sx in [-1,1] for sy in [-1,1]]
            assert_close(value, e.sigma_S_cm.iloc[0], 'Original-coordinate digitisation')
            assert_close([min(corners), max(corners)],
                         [e.digitization_low_S_cm.iloc[0], e.digitization_high_S_cm.iloc[0]],
                         'Pixel-reading sensitivity')
            rows.append(dict(T_K=t, mol_percent=curve['mol_percent'], sigma_S_cm=float(value)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(output/'Weller_digitisation_reanalysis.csv', index=False)
    return dict(preparations=100, conditions=25, experimental_readings=len(rows),
                carrier='oxygen ions; formal charge 2e',
                experiment_coordinates='piecewise linear in 1000/T and ln(sigma*T)',
                experimental_bounds='pixel-reading sensitivity, not measurement uncertainty')
