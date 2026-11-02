"""Reconstruct cross-phase teacher errors and transport fits from observations."""
import json,hashlib
import numpy as np
import pandas as pd
from reproduce import assert_close
from raw_reanalysis import read_stored_force_labels,safe_path


def verify_index(root,folder):
    records=json.loads(safe_path(root,folder+'/SOURCE_INDEX.json').read_text())['files']
    for row in records:
        p=safe_path(root,row['file'])
        if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Observation source checksum mismatch')
    return records


def crossphase_teachers(a,root,output):
    folder='crossphase/teacher_predictions';verify_index(root,folder)
    selected=np.array(json.loads(safe_path(root,folder+'/SELECTION.json').read_text())['indices_zero_based'])
    membership=pd.read_csv(safe_path(root,folder+'/FRAME_EVIDENCE.csv'))
    if not np.array_equal(selected,membership.frame) or len(selected)!=120:
        raise ValueError('Expected exact locked120 source-pool membership')
    labels=read_stored_force_labels(safe_path(root,'crossphase/test120_dft.extxyz'))
    if len(labels)!=120:raise ValueError('Expected120 DFT labels')
    expected=a.csv('data/crossphase/PARTITION_ERRORS.csv');rows=[]
    high=membership.p.to_numpy()>.1
    if high.sum()!=112:raise ValueError('Support partition differs from112/8')
    for model in ['Tpre','T140']:
        with np.load(safe_path(root,folder+'/'+model+'_predictions.npz'),allow_pickle=False) as z:
            force,ptr=z['forces'],z['atom_ptr']
            if len(ptr)!=424:raise ValueError('Expected original423-frame pool predictions')
            errors=[force[ptr[j]:ptr[j+1]]-labels[i]['forces'] for i,j in enumerate(selected)]
        for partition,keep in [('all',np.ones(120,dtype=bool)),('high',high),('low',~high)]:
            error=np.concatenate([errors[i] for i in np.flatnonzero(keep)])
            rmse=float(np.sqrt(np.mean(error**2)))
            match=expected[(expected.partition==partition)&(expected.model==model)]
            if len(match)!=1:raise ValueError('Missing historical teacher partition metric')
            assert_close(rmse,match.pooled_component_RMSE.iloc[0],'Locked120 teacher force-component RMSE')
            rows.append(dict(model=model,partition=partition,frames=int(keep.sum()),force_component_RMSE_eV_A=rmse))
    pd.DataFrame(rows).to_csv(output/'crossphase_teacher_predictions_reanalysis.csv',index=False)
    return dict(test_frames=120,high_support=112,low_support=8,teacher_partition_metrics=len(rows),
                source='Original Tpre/T140 Cartesian predictions selected by locked120 membership; project DFT labels, not old423-pool labels')


def msd_fits(a,root,output):
    folder='validation_records/ysz_msd';verify_index(root,folder)
    prep=a.csv('data/ysz/conductivity_evidence/preparation_inputs.csv')
    results=[]
    for row in prep.itertuples(index=False):
        d=pd.read_csv(safe_path(root,folder+'/curves/'+row.id+'.csv.gz'))
        if not np.isfinite(d.to_numpy()).all() or len(d)<5001:
            raise ValueError('Incomplete/nonfinite MSD curve')
        fit=d[(d.tau_ps>=100-1e-8)&(d.tau_ps<=200+1e-8)]
        x=fit.tau_ps.to_numpy();y=fit.MSD_A2.to_numpy();center=x-x.mean()
        slope=float(center@(y-y.mean())/(center@center))
        value=slope/6*1e-8
        assert_close(value,row.D_star,'Full-5ns MSD primary-fit diffusivity',atol=1e-22,rtol=1e-10)
        results.append(dict(id=row.id,T=row.T,comp=row.comp,fit_points=len(x),
                            D_star_m2_s=value,intercept_A2=float(y.mean()-slope*x.mean())))
    pd.DataFrame(results).to_csv(output/'YSZ_all100_MSD_fit_reanalysis.csv',index=False)
    return dict(preparations=len(results),fit_delays_ps=[100,200],free_intercept=True,
                replay='Regression of existing complete full5ns MSD curves, not new trajectories or new MSD calculations')


def reanalyze(a,root,output):
    return dict(crossphase_teachers=crossphase_teachers(a,root,output),
                MSD=msd_fits(a,root,output))
