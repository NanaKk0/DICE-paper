#!/usr/bin/env python3
"""Evaluate the released student checkpoints on the labelled test sets."""
import argparse
import csv
import json
from pathlib import Path


def contained(root, relative):
    p = (root/relative).resolve(strict=True)
    if not p.is_relative_to(root):
        raise ValueError('Path escapes research archive')
    return p


def evaluate_frame(potential, atoms):
    import numpy as np
    native = list(potential.get_type_map())
    symbols = atoms.get_chemical_symbols()
    if any(s not in native for s in symbols):
        raise ValueError('Model native type map does not include every species')
    types = [native.index(s) for s in symbols]
    if atoms.pbc.any() and not atoms.pbc.all():
        raise ValueError('This example supports fully periodic or isolated frames only')
    cells = atoms.cell.array.reshape(1,9) if atoms.pbc.all() else None
    e, f, v = potential.eval(atoms.positions.reshape(1,-1,3), cells, types)[:3]
    return float(np.asarray(e).ravel()[0]), np.asarray(f).reshape(-1,3), np.asarray(v).reshape(3,3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root', required=True, type=Path)
    p.add_argument('--system', choices=['al2o3','ysz','mil120'], required=True)
    p.add_argument('--role', choices=['dice_student_500k','direct_dft_student_500k'], required=True)
    p.add_argument('--limit', type=int, default=1, help='Frames per ordered test union; 0 means all')
    p.add_argument('--device', default='cuda', help='GPU device for native checkpoint evaluation')
    p.add_argument('--evaluate', action='store_true', help='Explicitly execute native inference; otherwise validate paths only')
    p.add_argument('--output', type=Path, help='New JSON output path outside inputs; mandatory with --evaluate')
    args = p.parse_args()
    root = args.research_root.expanduser().resolve(strict=True)
    models = list(csv.DictReader((root/'MODEL_IDENTITY.csv').open()))
    rows = [r for r in models if r['system']==args.system and r['scientific_role']==args.role]
    if len(rows)!=1: raise ValueError('Expected one exact model')
    row = rows[0]; model = contained(root,row['file'])
    datasets = list(csv.DictReader((root/'DATASET_SUMMARY.csv').open()))
    paths = [contained(root,r['file']) for r in datasets if r['system']==args.system and r['role']=='independent_test']
    if args.limit < 0 or not paths: raise ValueError('Invalid limit or missing test union')
    if not args.evaluate:
        print(json.dumps(dict(model=row['file'],
              tests=[str(x.relative_to(root)) for x in paths],
              action='Model and tests selected. Use --evaluate and --output to run evaluation.'),indent=2))
        return
    if args.output is None: raise ValueError('--output required')
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('Choose a new output outside the input/software roots')
    import numpy as np
    from ase.io import iread
    from deepmd.infer import DeepPot
    potential = DeepPot(str(model), device=args.device, auto_batch_size=False)
    force_ss = force_n = 0; energy_ss = 0.; results=[]
    for source in paths:
        for local_index, atoms in enumerate(iread(source, format='extxyz')):
            if args.limit and len(results)>=args.limit: break
            reference_energy=atoms.get_potential_energy(); reference_forces=atoms.get_forces()
            e,f,v=evaluate_frame(potential,atoms)
            if not (np.isfinite(e) and np.isfinite(f).all() and np.isfinite(v).all()):
                raise ValueError('Non-finite model prediction')
            delta=f-reference_forces; force_ss+=float(np.square(delta).sum()); force_n+=delta.size
            er=(e-reference_energy)/len(atoms);energy_ss+=er**2
            results.append(dict(test=str(source.relative_to(root)),frame=local_index,
                energy_eV=e,forces_eV_A=f.tolist(),virial_eV=v.tolist(),
                reference_energy_eV=reference_energy,energy_error_eV_atom=er))
    if not results: raise ValueError('No frames evaluated')
    payload=dict(system=args.system,scientific_role=args.role,model_file=row['file'],
        native_type_map=list(potential.get_type_map()),frames=len(results),
        force_component_RMSE_eV_A=float(np.sqrt(force_ss/force_n)),
        energy_per_atom_RMSE_eV=float(np.sqrt(energy_ss/len(results))),
        units=dict(positions='angstrom',energy='eV',forces='eV/angstrom',virial='eV'),
        offset='none', results=results)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(payload,f)
    print(json.dumps({k:v for k,v in payload.items() if k!='results'},indent=2))

if __name__=='__main__':main()
