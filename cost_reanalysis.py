"""Reconstruct recorded work and startup-inclusive teacher cost."""
from collections import defaultdict
import gzip
import json
import numpy as np
import pandas as pd
from reproduce import assert_close


def verify_cost_evidence(a,output):
    with gzip.open(a.path('data/sampling_costs/task_work_records.json.gz'),'rt') as f:tasks=json.load(f)
    inventory=defaultdict(lambda:dict(work=0,tasks=0));blocks={}
    for t in tasks:
        kind=t['kind']
        work=(t['evaluations'] if kind=='relax' else t['counts']['attempted'] if kind=='widom'
              else t['mc_attempts']+5*t['steps'] if kind=='gcmc' else t.get('actual_steps',t['steps']))
        if work!=t['work']:raise ValueError('Native work counter mismatch')
        inventory[t['group']]['work']+=work;inventory[t['group']]['tasks']+=1
        key=(t['system'],t['iter'],t['round'])
        if key in blocks:assert_close(blocks[key],float(t['stage_seconds']),'Recorded block duration')
        blocks[key]=float(t['stage_seconds'])
    if (len(tasks),len(blocks),len(inventory))!=(75038,148,26):raise ValueError('Sampling scope mismatch')
    b=a.csv('data/sampling_costs/BLOCK_MODE_FULL_ESTIMATES_219.csv')
    cal=a.csv('data/sampling_costs/STARTUP_CALIBRATION_26.csv').set_index('mode')
    if len(b)!=219 or len(cal)!=26:raise ValueError('Cost inventory mismatch')
    for mode,g in b.groupby('mode'):
        assert_close(g.work.sum(),inventory[mode]['work'],'Native mode work')
        assert_close(g.tasks.sum(),inventory[mode]['tasks'],'Native mode task count')
    startup=b.concurrent_startup_waves*b['mode'].map(cal.startup_wave_seconds)
    exit_time=b.concurrent_startup_waves*b['mode'].map(cal.exit_wave_seconds)
    hybrid=b.hybrid_md_parallel_wave_equivalents*b['mode'].map(cal.md_transport_seconds_per_segment)
    assert_close(startup,b.teacher_process_startup_seconds,'Startup waves')
    assert_close(exit_time,b.teacher_process_exit_seconds,'Exit waves')
    assert_close(hybrid,b.teacher_hybrid_transport_seconds,'Hybrid subprocess overhead')
    overhead=startup+exit_time+hybrid+b.common_preparation_validation_seconds
    assert_close(overhead,b.teacher_added_overhead_seconds,'Total task overhead')
    b['recalculated_teacher_seconds']=b.work/b['mode'].map(cal.full_native_body_rate)+overhead
    assert_close(b.recalculated_teacher_seconds,b.teacher_protocol_recheck_seconds,'Native-rate teacher estimates')
    systems=b.groupby('system').recalculated_teacher_seconds.sum()/3600
    reference=a.csv('data/sampling_costs/DEVELOPMENT_SAMPLING_COST.csv').set_index('system')
    rows=[]
    for system,teacher in systems.items():
        student=sum(v for k,v in blocks.items() if k[0]==system)/3600
        assert_close(student,reference.loc[system,'student_recorded_h'],'Recorded student hours')
        assert_close(teacher,reference.loc[system,'teacher_estimated_h'],'Teacher hours')
        rows.append(dict(system=system,student_recorded_h=student,teacher_estimated_h=teacher,
                         student_percent_of_teacher=100*student/teacher,teacher_student_ratio=teacher/student))
    pd.DataFrame(rows).to_csv(output/'development_costs.csv',index=False)
    b.to_csv(output/'block_mode_cost_reanalysis.csv',index=False)
    return dict(task_work_records=len(tasks),recorded_blocks=len(blocks),mode_allocations=len(b),modes=len(cal),
                definition='work/native rate + startup + exit + hybrid transport + preparation and validation',systems=rows)
