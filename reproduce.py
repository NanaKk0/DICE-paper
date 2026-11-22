#!/usr/bin/env python3
"""CPU reproduction of the 2026-11-22 DICE paper evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


class Archive:
    def __init__(self, root):
        self.root=Path(root).expanduser().resolve(strict=True)
        self.registry=json.loads(self.path('DATA_REGISTRY.json').read_text())
        if self.registry['schema']!=1: raise ValueError('Unsupported registry schema')

    def path(self, relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Path outside archive')
        return path

    def csv(self, relative): return pd.read_csv(self.path(relative))
    def json(self, relative): return json.loads(self.path(relative).read_text())


def assert_close(actual, expected, description, atol=1e-8, rtol=1e-7):
    if not np.allclose(actual,expected,atol=atol,rtol=rtol,equal_nan=False):
        raise ValueError(description+': numerical mismatch')


def paired_summary(data, groups, value):
    return data.groupby(groups,sort=False)[value].agg(['mean','std','count']).reset_index()


def summary(a, output):
    frames=a.csv('data/prediction_and_mil/frames.csv')
    pooled=frames.groupby(['system','route']).agg(ss=('force_ss','sum'),n=('force_n','sum'))
    pooled['force_component_RMSE_eV_A']=np.sqrt(pooled.ss/pooled.n)
    production=[]
    for system in ['Al2O3','YSZ','MIL-120']:
        direct=pooled.loc[(system,'Direct'),'force_component_RMSE_eV_A']
        dice=pooled.loc[(system,'DICE'),'force_component_RMSE_eV_A']
        production.append(dict(system=system,direct_force_RMSE=direct,DICE_force_RMSE=dice,reduction_percent=100*(1-dice/direct)))
    production=pd.DataFrame(production)
    production.to_csv(output/'production_force_reductions.csv',index=False)
    tails=a.csv('data/controlled/Fig2a_paired_tails.csv')
    assert_close(tails.tail_mean_eV_A-tails.baseline_tail_eV_A,tails.difference_eV_A,'DFT-70 paired tails')
    if set(tails['repeat'])!={1,2,3,4,5}: raise ValueError('Five selection repeats required')
    paired_summary(tails,['pool','route','q_percent'],'difference_eV_A').to_csv(output/'DFT70_paired_tail_summary.csv',index=False)
    students=a.csv('data/controlled/metrics_per_repeat.csv')
    held=students[students.scope=='excluded_training_frames']
    for pool,n in [('gamma506',446),('liquid423',363)]:
        if set(held.loc[held.pool==pool,'frames'])!={n}:raise ValueError('70-frame held-out population')
    held.groupby(['pool','route']).force_component_RMSE_eV_A.agg(['mean','std']).to_csv(output/'controlled70_student_RMSE.csv')
    teachers=a.csv('data/controlled/teacher_metrics_per_repeat.csv')
    if set(teachers.scope)!={'all70','unqueried50','heldout_pool'}:raise ValueError('Teacher comparison partitions')
    teachers.groupby(['pool','scope','teacher']).force_component_RMSE_eV_A.agg(['mean','std']).to_csv(output/'controlled70_teacher_RMSE.csv')
    pairs=a.csv('data/routing/Fig3g_relative_P99_paired_source.csv')
    relative=100*(pairs.teacher_P99_eV_A/pairs.DFT_P99_eV_A-1)
    assert_close(relative,pairs.relative_P99_difference_percent,'Per-repeat relative P99')
    if len(pairs)!=50:raise ValueError('Expected ten matched comparisons, five repeats each')
    raw=a.csv('data/routing/student_metrics_per_repeat.csv')
    for row in pairs.itertuples():
        if row.router=='QBC':routes=['QBC_T0','QBC_DFT']
        elif row.pool=='gamma506':routes=['T0_initial','DFT_initial'] if row.teacher_label_source=='T0' else ['T1_expanded','DFT_expanded']
        else:routes=['T0_15','DFT_15'] if row.teacher_label_source=='T0' else ['T1_75','DFT_75']
        d=raw[(raw.router==row.router)&(raw.pool==row.pool)&(raw.criterion.astype(str)==row.criterion)&
              (raw['repeat']==row.repeat)&(raw.scope==row.scope)]
        for route,value in zip(routes,[row.teacher_P99_eV_A,row.DFT_P99_eV_A]):
            selected=d[d.route==route]
            if len(selected)!=1:raise ValueError('Paired student identity')
            assert_close(selected.force_vector_P99_eV_A.iloc[0],value,'P99 from repeat-level model metrics')
    paired_summary(pairs,['router','pool','criterion','teacher_label_source'],'relative_P99_difference_percent').to_csv(output/'routing_relative_P99.csv',index=False)
    accept=a.csv('data/si_comparisons/routing_acceptance_per_repeat.csv')
    assert_close(100*accept.high_error_accepted/accept.accepted,accept.high_error_fraction_percent,'Accepted-set high-error fraction')
    accept.groupby(['pool','group']).high_error_fraction_percent.agg(['mean','std']).to_csv(output/'accepted_high_error_fraction.csv')
    # Check the representative scatter directly against the acceptance counts.
    scatter=a.csv('data/routing/Fig3_scatter_source.csv')
    for panel,pool in [('a','gamma506'),('b','liquid423')]:
        g=scatter[scatter.panel==panel]
        for threshold in [.03,.05,.08]:
            selected=g[g.x<=threshold]
            row=accept[(accept.pool==pool)&(accept.group==f'QBC_{threshold:.2f}')].iloc[0]
            assert_close(len(selected),row.accepted,'QBC accepted count')
            assert_close((selected.teacher_force_component_RMSE_eV_A>.05).sum(),row.high_error_accepted,'QBC high-error count')
    for panel,pool in [('d','gamma506'),('e','liquid423')]:
        g=scatter[(scatter.panel==panel)&(scatter.teacher=='T0')]
        selected=g[g.x>.1]
        row=accept[(accept.pool==pool)&(accept.group=='GMM_T0')].iloc[0]
        assert_close(len(selected),row.accepted,'GMM initial accepted count')
        assert_close((selected.teacher_force_component_RMSE_eV_A>.05).sum(),row.high_error_accepted,'GMM initial high-error count')
    structure=a.csv('data/structural_analysis/ORDER_METRICS.csv')
    counts=structure.groupby('group').size().to_dict()
    if counts!={'teacher_140':140,'student_before_940':940,'added_380':380,'test_120':120}:raise ValueError('Structural analysis membership')
    mix={}
    for element in ['Al','O']:
        t=a.csv(f'data/crossphase/TSNE_{element}.csv')
        u=t.teacher_density/(t.teacher_density+t.test_density)
        assert_close(4*u*(1-u),t.local_mixing,'Neighbourhood mixing')
        mix[element]=len(t)
    progress=a.csv('data/crossphase/STUDENT_PLOTTED_DATA.csv').set_index('round')
    teacher=a.csv('data/crossphase/TEACHER_PLOTTED_DATA.csv').set_index('model')
    teacher_gain=100*(1-teacher.loc['Adapted teacher','force_component_RMSE_eV_A']/teacher.loc['Pretrained teacher','force_component_RMSE_eV_A'])
    student_gain=100*(1-progress.loc[10,'force_component_RMSE_eV_A']/progress.loc[2,'force_component_RMSE_eV_A'])
    routing=a.csv('data/structural_analysis/ROUTING_COUNTS.csv')
    accepted_percent=100*routing.high_support.sum()/routing.prescreened_candidates.sum()
    from cost_reanalysis import verify_cost_evidence
    from transport_reanalysis import verify_transport
    from transition_reanalysis import verify_transitions
    result=dict(snapshot=a.registry['snapshot'],controlled_training_frames=70,held_out_frames={'gamma':446,'liquid':363},
                routing_pairs=len(pairs),structure_groups=counts,projection_points=mix,
                teacher_reduction_percent=float(teacher_gain),student_reduction_percent=float(student_gain),
                development_high_support_percent=float(accepted_percent),
                production_reductions_percent=production.reduction_percent.tolist(),
                cost_reanalysis=verify_cost_evidence(a,output),
                conductivity_reanalysis=verify_transport(a,output),
                YSZ_transition_reanalysis=verify_transitions(a,output))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify(a,output):
    ids={f['id'] for f in a.registry['figures']}
    if ids!={*[f'main:{i}' for i in range(1,8)],*[f'supplement:{i}' for i in range(1,14)]}:raise ValueError('Current figure inventory')
    if len(a.registry['tables'])!=5:raise ValueError('Current table inventory')
    for f in a.registry['figures']:
        for name in [f['artwork'],*f['sources']]:
            if not a.path(name).is_file():raise ValueError('Missing input '+name)
    result=summary(a,output)
    result['verified_figures']=len(a.registry['figures'])
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def tables(a,output):
    for row in a.registry['tables']:shutil.copyfile(a.path(row['file']),output/Path(row['file']).name)
    return dict(tables=len(a.registry['tables']))


def main_table(a,output):
    d=a.csv('data/provenance/Main_Table1.csv')
    d[['system','target_DFT_records','student_geometries','independent_test']].to_csv(output/'MainTable1.csv',index=False)
    return dict(rows=len(d))


def reexport(a,output):
    for f in a.registry['figures']:shutil.copyfile(a.path(f['artwork']),output/Path(f['artwork']).name)
    return dict(approved_SVG_files=len(a.registry['figures']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','summary','figures','replot','reexport','tables','main_table'])
    p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent/'source-data')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--figure',action='append')
    args=p.parse_args();a=Archive(args.data_root);out=args.output_dir.expanduser().resolve()
    if out.is_relative_to(a.root) or a.root.is_relative_to(out) or out==Path(__file__).resolve().parent:
        p.error('Use a dedicated output directory outside source-data and the code root')
    if args.figure and args.action!='figures':p.error('--figure applies to figures')
    out.mkdir(parents=True,exist_ok=True)
    if args.action in ['figures','replot']:
        from paper_figures import rebuild
        result=rebuild(a,out,args.figure)
    else:result=globals()[args.action](a,out)
    print(json.dumps(dict(action=args.action,output=str(out),result=result),indent=2))


if __name__=='__main__':main()
