"""Replay final 500k YSZ transitions with the archived cluster resamples."""
import numpy as np
import pandas as pd
from reproduce import assert_close


def verify_transitions(a, output):
    metadata = a.csv('data/ysz/validation_atoms.csv.gz')
    force = a.csv('data/accuracy/S_direct_YSZ500_parity_force.csv')
    if not np.array_equal(force.component_index, np.arange(len(force))) or len(force)!=3*len(metadata):
        raise ValueError('Final parity must contain ordered x/y/z components for every atom')
    ref=force.DFT_eV_A.to_numpy().reshape(-1,3)
    direct=np.linalg.norm(force.direct_eV_A.to_numpy().reshape(-1,3)-ref,axis=1)
    dice=np.linalg.norm(force.DICE_eV_A.to_numpy().reshape(-1,3)-ref,axis=1)
    assert_close(dice,metadata.enorm_M,'DICE atom-order reconciliation',atol=1e-6)
    # The old enorm_C100 metadata field belongs to a different historical model.
    # It is never read as the final direct student's error.
    final=metadata.drop(columns=['enorm_C100','enorm_M']).assign(
        direct500k_vector_error_eV_A=direct,DICE500k_vector_error_eV_A=dice)
    final.to_csv(output/'YSZ_final500k_paired_atom_errors.csv',index=False)
    with np.load(a.path('data/ysz/structural_cluster_bootstrap_weights.npz'),allow_pickle=False) as z:
        weights=np.array(z['weights']);clusters=np.array(z['clusters']).astype(str)
    if weights.shape!=(1000,42):raise ValueError('Expected archived 1000-by-42 cluster resamples')
    ids=metadata.cluster.astype(str).to_numpy()
    expected=a.csv('data/provenance/YSZ500k_error_transitions.csv')
    rows=[]
    for row in expected.itertuples(index=False):
        use=metadata.element.to_numpy()==row.element
        t=row.threshold_eV_A
        if row.transition=='removed': event=(direct>t)&(dice<=t)
        elif row.transition=='introduced':event=(direct<=t)&(dice>t)
        else:raise ValueError('Unknown transition')
        totals=np.array([np.count_nonzero(use&(ids==c)) for c in clusters])
        counts=np.array([np.count_nonzero(use&event&(ids==c)) for c in clusters])
        fractions=(weights@counts)/(weights@totals)
        lo,hi=np.quantile(fractions,[.025,.975])
        n=int(use.sum());count=int((use&event).sum())
        assert_close([n,count,count/n,lo,hi],
                     [row.n_atom_observations,row.count,row.fraction,row.CI_low,row.CI_high],
                     'Final 500k transitions and archived paired cluster CI')
        rows.append(dict(element=row.element,threshold_eV_A=t,transition=row.transition,
                         count=count,n_atom_observations=n,fraction=count/n,CI_low=lo,CI_high=hi))
    pd.DataFrame(rows).to_csv(output/'YSZ_final500k_transition_reanalysis.csv',index=False)
    return dict(atom_observations=len(final),transition_rows=len(rows),
                resamples=1000,clusters=42,source='final Cartesian force parity, never enorm_C100',
                bootstrap='replay saved paired cluster weights; no new draws')
