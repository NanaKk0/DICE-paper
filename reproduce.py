#!/usr/bin/env python3
"""Portable, CPU-only reproduction of the archived DICE paper evidence.

No inference, training, simulation, geometry optimization or external file access.
Historical path columns in source CSVs are provenance strings, never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


class Archive:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.registry = json.loads(self.path("DATA_REGISTRY.json").read_text())
        if self.registry.get("schema") != 1:
            raise ValueError("Unsupported DATA_REGISTRY schema")

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Archive path escapes data root: {relative}")
        return path

    def csv(self, relative):
        return pd.read_csv(self.path(relative))

    def json(self, relative):
        return json.loads(self.path(relative).read_text())


def assert_close(actual, expected, description, atol=1e-8, rtol=1e-7):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"{description}: got {actual}; expected {expected}")


def summary(archive, output):
    """Recalculate paper claims from residual sums, timings and path numerics."""
    a = archive
    frames = a.csv("data/prediction_and_mil/frames.csv")
    pooled = frames.groupby(["system", "route"], sort=False).agg(
        frames=("frame_id", "size"), force_ss=("force_ss", "sum"),
        force_n=("force_n", "sum"))
    pooled["RMSE_eV_A"] = np.sqrt(pooled.force_ss / pooled.force_n)
    pooled.to_csv(output / "pooled_force_RMSE.csv")
    rows = []
    for system in ["Al2O3", "YSZ", "MIL-120"]:
        g = pooled.loc[[system]]
        direct = g.loc[(system, "Direct"), "RMSE_eV_A"]
        dice = g.loc[(system, "DICE"), "RMSE_eV_A"]
        rows.append(dict(system=system, direct_force_RMSE=direct,
                         DICE_force_RMSE=dice,
                         reduction_percent=100 * (direct - dice) / direct))
    production = pd.DataFrame(rows)
    expected = a.csv("data/summary/production_force_RMSE.csv")
    assert_close(production.reduction_percent, expected.reduction_percent,
                 "Production reductions")
    production.to_csv(output / "production_force_reductions.csv", index=False)

    profiles = a.csv("data/prediction_and_mil/all_path_profiles.csv")
    path_rows = []
    for (path, route), g in profiles.groupby(["path", "route"], sort=False):
        g = g.sort_values("image")
        if len(g) != 11 or list(g.image) != list(range(11)):
            raise ValueError(f"Expected eleven complete images: {path}/{route}")
        # This t is cumulative DFT-path Cartesian displacement, NOT image / 10.
        t = g.arc_fraction.to_numpy()
        assert_close(t[[0, -1]], [0, 1], "DFT arc endpoints")
        if np.any(np.diff(t) < 0):
            raise ValueError("Non-monotone DFT path coordinate")
        error = 1000 * (g.model_eV.to_numpy() - g.DFT_eV.to_numpy())
        detrended = error - ((1 - t) * error[0] + t * error[-1])
        path_rows.append(dict(path=path, route=route, images=11, interior_images=9,
                              endpoint_detrended_MAE_meV=np.abs(detrended[1:-1]).mean()))
    paths = pd.DataFrame(path_rows)
    reference = a.csv("data/summary/MIL_nine_path_detrended_MAE.csv").set_index("path")
    for row in paths.itertuples():
        assert_close(row.endpoint_detrended_MAE_meV,
                     reference.loc[row.path, row.route + "_interior_MAE_meV"],
                     f"DFT-arc detrended MAE {row.path}/{row.route}")
    paths.to_csv(output / "nine_path_endpoint_detrended_MAE.csv", index=False)

    blocks = a.csv("data/development/BLOCK_LINEAGE_148.csv")
    conditions = a.csv("data/development/SAMPLING_CONDITIONS.csv")
    modes = a.csv("data/sampling_costs/FORMAL_26_SAMPLING_MODES.csv")
    costs = modes.groupby("system", sort=False).agg(
        student_hours=("student_hours", "sum"),
        teacher_estimated_hours=("teacher_estimated_hours", "sum"),
        tasks=("task_count", "sum"), modes=("mode", "size"))
    costs["teacher_to_student_ratio"] = costs.teacher_estimated_hours / costs.student_hours
    costs.to_csv(output / "development_costs.csv")
    expected_cost = a.csv("data/sampling_costs/DEVELOPMENT_SAMPLING_COST.csv").set_index("system")
    for system, row in costs.iterrows():
        assert_close(row.student_hours, expected_cost.loc[system, "student_recorded_h"], "Student hours")
        assert_close(row.teacher_estimated_hours, expected_cost.loc[system, "teacher_estimated_h"], "Teacher estimate")
    if len(blocks) != 148 or conditions.tasks.sum() != 75038 or len(modes) != 26:
        raise ValueError("Development scope differs from 148 blocks / 75038 tasks / 26 modes")

    teacher = a.csv("data/crossphase/TEACHER_PLOTTED_DATA.csv").set_index("model")
    student = a.csv("data/crossphase/STUDENT_PLOTTED_DATA.csv").set_index("round")
    tr = 100 * (1 - teacher.loc["Adapted teacher", "force_component_RMSE_eV_A"] /
                teacher.loc["Pretrained teacher", "force_component_RMSE_eV_A"])
    # Historical expansion is 940 -> 1320 geometries, not final production training.
    sr = 100 * (1 - student.loc[10, "force_component_RMSE_eV_A"] /
                student.loc[2, "force_component_RMSE_eV_A"])
    if round(tr) != 32 or round(sr) != 16:
        raise ValueError("Cross-phase reductions do not round to 32% / 16%")

    ph = a.csv("data/phonons/S26_phonon_dispersion.csv")
    # Relative errors are undefined near acoustic zero. Keep signed DFT > 0.5 THz.
    mask = ph.reference_frequency_THz > 0.5
    ph_rows = []
    for model in ["DFT_only", "DICE"]:
        delta = ph[model + "_frequency_THz"] - ph.reference_frequency_THz
        ph_rows.append(dict(model=model, full_modes=len(ph), retained_DFT_above_0p5_THz=int(mask.sum()),
                            frequency_MAE_THz=np.abs(delta[mask]).mean(),
                            frequency_P95_THz=np.quantile(np.abs(delta[mask]), 0.95),
                            relative_MAE_percent=(100 * np.abs(delta[mask]) /
                                                  ph.reference_frequency_THz[mask]).mean()))
    phonon_source = a.csv("data/provenance/Al500k_phonon_metrics.csv")
    alpha_source = phonon_source[phonon_source.phase == "alpha"].set_index("model_state")
    expected_phonons = {"DFT_only": alpha_source.loc["al_direct500k", "frequency_MAE_THz"],
                        "DICE": alpha_source.loc["al_final4_expanded_500k", "frequency_MAE_THz"]}
    for row in ph_rows:
        assert_close(row["frequency_MAE_THz"], expected_phonons[row["model"]],
                     "DFT > 0.5 THz phonon MAE")
    pd.DataFrame(ph_rows).to_csv(output / "phonon_metrics.csv", index=False)
    tail_audit = []
    for filename, repeat, q in [
        ("Fig2b_random_paired_tail_differences.csv", "replicate", "top_fraction_percent"),
        ("Fig2f_supported_paired_tail_differences.csv", "repeat", "q_percent")]:
        d = a.csv("data/controlled/" + filename)
        assert_close(d.tail_mean_eV_per_A - d.DFT60_tail_eV_A
                     if "tail_mean_eV_per_A" in d else d.tail_mean_eV_A - d.DFT60_tail_eV_A,
                     d.tail_difference_eV_A, "Signed paired tail differences")
        if set(d[repeat]) != {1, 2, 3, 4, 5}:
            raise ValueError("Expected five configuration-selection repeats")
        tail_audit.append(dict(source=filename, rows=len(d), repeats=5,
                               q_min_percent=float(d[q].min()), q_max_percent=float(d[q].max())))
    result = dict(production_reductions_percent=production.reduction_percent.tolist(),
                  teacher_reduction_percent=tr, historical_student_expansion_reduction_percent=sr,
                  historical_student_geometries=[940, 1320], development_blocks=len(blocks),
                  development_tasks=int(conditions.tasks.sum()), sampling_modes=len(modes),
                  nine_common_DFT_paths=profiles.path.nunique(),
                  phonon_relative_threshold="reference_frequency_THz > 0.5", tail_audit=tail_audit)
    if "cost_evidence" in a.registry:
        from cost_reanalysis import verify_cost_evidence
        result["cost_reanalysis"] = verify_cost_evidence(a, output)
    if "conductivity_evidence" in a.registry:
        from transport_reanalysis import verify_transport
        result["conductivity_reanalysis"] = verify_transport(a, output)
    from transition_reanalysis import verify_transitions
    result["YSZ_transition_reanalysis"] = verify_transitions(a, output)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(a, output):
    checks = []
    for entry in a.registry["files"]:
        p = a.path(entry["file"])
        if not p.is_file() or p.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing/size-mismatched archived file: {entry['file']}")
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {entry['file']}")
        checks.append(entry["file"])
    if len(a.registry["figures"]) != 22 or len(a.registry["tables"]) != 12:
        raise ValueError("Expected 22 figures and 12 tables")
    for figure in a.registry["figures"]:
        for path in [figure["artwork"], *figure["sources"]]:
            if not a.path(path).is_file():
                raise ValueError(f"Missing figure input: {path}")
    result = summary(a, output)
    result["verified_files"] = len(checks)
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def tables(a, output):
    for table in a.registry["tables"]:
        shutil.copyfile(a.path(table["file"]), output / Path(table["file"]).name)
    (output / "table_index.json").write_text(json.dumps(a.registry["tables"], indent=2) + "\n")
    return dict(exact_table_CSVs=len(a.registry["tables"]))


def main_table(a, output):
    source = a.csv("data/provenance/Main_Table1.csv")
    columns = ["system", "target_DFT_records", "student_geometries", "independent_test"]
    table = source.set_index("system").loc[["Al2O3", "YSZ", "MIL120"]].reset_index()[columns]
    table.to_csv(output / "MainTable1.csv", index=False)
    return dict(table="Main Table 1", rows=len(table),
                source="data/provenance/Main_Table1.csv", scientific_recalculation=False)


def reexport(a, output):
    for figure in a.registry["figures"]:
        shutil.copyfile(a.path(figure["artwork"]), output / Path(figure["artwork"]).name)
    (output / "README.txt").write_text(
        "22 approved SVG files copied unchanged. This action is artwork export, "
        "NOT numerical reanalysis. Use replot for source-data verification figures.\n")
    return dict(exact_artwork_SVGs=len(a.registry["figures"]), numerical_reanalysis=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "summary", "replot", "figures", "reexport", "tables", "main_table"])
    parser.add_argument("--figure", action="append", help="For figures only: main:3 or supplement:9; repeatable")
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent / "source-data",
                        help="Bundled source-data directory; override for another frozen dataset")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Dedicated output directory, separate from the read-only data archive")
    parser.add_argument("--research-root", type=Path,
                        help="Optional research-assets root; independently rederive controlled metrics from 65 caches")
    args = parser.parse_args()
    if args.figure and args.action != "figures":
        parser.error("--figure is only valid with figures")
    a = Archive(args.data_root)
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(a.root) or a.root.is_relative_to(output):
        parser.error("Output and data directories must not overlap")
    if output == Path(__file__).resolve().parent:
        parser.error("Use a dedicated output directory, not the code directory")
    if args.research_root:
        research = args.research_root.expanduser().resolve(strict=True)
        if output.is_relative_to(research) or research.is_relative_to(output):
            parser.error("Output and research-assets directories must not overlap")
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "figures":
        from paper_figures import rebuild
        if any(output.iterdir()):
            parser.error("figures requires an empty output directory")
        result = rebuild(a, output, args.figure)
    elif args.action == "replot":
        from plots import replot
        summary(a, output)
        result = replot(a, output)
    else:
        result = globals()[args.action](a, output)
    if args.research_root and args.action in ["verify", "summary", "replot"]:
        from raw_reanalysis import reanalyze
        raw_result = reanalyze(a, research, output)
        from physical_reanalysis import reanalyze as reanalyze_physical
        physical_result = reanalyze_physical(a, research, output)
        from remaining_observations import reanalyze as reanalyze_observations
        observations = reanalyze_observations(a, research, output)
        result = dict(summary=result, raw_reanalysis=raw_result,
                      physical_reanalysis=physical_result,
                      observation_reanalysis=observations)
    print(json.dumps(dict(action=args.action, output=str(output), result=result), indent=2))


if __name__ == "__main__":
    main()
