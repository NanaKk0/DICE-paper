"""Independent controlled-experiment metric reconstruction from archived norms.

Only NPZ error arrays and membership CSVs are read. Training input files and
models are indexed provenance; nothing is executed or inferred from them.
"""
import json
import re
import shlex

import numpy as np
import pandas as pd

from reproduce import assert_close


def safe_path(root, relative):
    p = (root / relative).resolve(strict=True)
    if not p.is_relative_to(root):
        raise ValueError("Research path escapes archive root")
    return p


def bool_column(values):
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy()
    if not values.isin(["True", "False"]).all():
        raise ValueError("Expected explicit True/False membership values")
    return (values == "True").to_numpy()


def norm_metrics(norms):
    values = np.asarray(norms, dtype=float).ravel()
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Force-vector error norms must be nonnegative finite values")
    return dict(force_component_RMSE_eV_A=np.sqrt(np.mean(values ** 2) / 3),
                vector_P99_eV_A=np.quantile(values, 0.99), atom_frames=len(values))


def tail_means(norms, q_percent):
    """Mean largest ceil(N q / 100) norms, retaining the paper's tail definition."""
    values = np.sort(np.asarray(norms, dtype=float).ravel())[::-1]
    q = np.asarray(q_percent, dtype=float)
    if not len(values) or not ((q > 0) & (q <= 100)).all():
        raise ValueError("Tail fractions require 0 < q_percent <= 100 and nonempty values")
    n = np.maximum(1, np.ceil(len(values) * q / 100).astype(int))
    return np.cumsum(values)[n - 1] / n


def reanalyze(a, root, output):
    index = pd.read_csv(safe_path(root, "controlled/CACHE_INDEX.csv"))
    if len(index) != 65:
        raise ValueError("Expected 50 Random10 plus 15 High40 caches")
    random_expected = a.csv("data/controlled/random_metrics_per_repeat.csv")
    random_tails = a.csv("data/controlled/Fig2b_random_paired_tail_differences.csv")
    high_tails = a.csv("data/controlled/Fig2f_supported_paired_tail_differences.csv")
    metrics, tails, membership_audit = [], [], []
    for row in index.itertuples(index=False):
        schema = json.loads(row.schema)
        candidates = [key for key in ["force_vector_error_eV_per_A", "force_vector_error"] if key in schema]
        if len(candidates) != 1:
            raise ValueError("Unrecognized force-vector cache schema")
        key = candidates[0]
        with np.load(safe_path(root, row.cache), allow_pickle=False) as cache:
            values = np.array(cache[key], dtype=float)
        if list(values.shape) != schema[key]["shape"] or values.ndim != 2:
            raise ValueError(f"Cache shape mismatch: {row.cache}")
        membership = pd.read_csv(safe_path(root, row.evaluation_membership))
        if row.experiment == "random10":
            membership = membership[(membership.distribution == row.distribution) &
                                    (membership.replicate == row.repeat)].sort_values("pool_row")
            positions = membership.pool_row.to_numpy()
            keep = bool_column(membership.included_in_leave_R50_out)
            expected_frames = 456 if row.distribution == "gamma_smrcok" else 373
            expected_tail = random_tails[(random_tails.distribution == row.distribution) &
                                         (random_tails.replicate == row.repeat) & (random_tails.route == row.route)]
            q = expected_tail.top_fraction_percent.to_numpy()
            scope = "leave_R50_out"
        elif row.experiment == "low10_high40":
            membership = membership[membership["repeat"] == row.repeat].sort_values("frame")
            positions = membership.frame.to_numpy()
            keep = bool_column(membership.included_geometry_clean)
            expected_frames = 373 if row.repeat in [1, 3, 5] else 372
            expected_tail = high_tails[(high_tails["repeat"] == row.repeat) & (high_tails.route == row.route)]
            q = expected_tail.q_percent.to_numpy()
            scope = "excluded_training_geometry_clean"
        else:
            raise ValueError(f"Unknown experiment: {row.experiment}")
        if not np.array_equal(positions, np.arange(len(values))):
            raise ValueError("Membership must identify every cache frame exactly once")
        if keep.sum() != expected_frames:
            raise ValueError(f"Unexpected evaluation membership count: {row.cache}: {keep.sum()}")
        selected = values[positions[keep]]
        metric = dict(experiment=row.experiment, distribution=row.distribution,
                      repeat=row.repeat, route=row.route, scope=scope,
                      frames=int(keep.sum()), **norm_metrics(selected))
        metrics.append(metric)
        if row.experiment == "random10":
            e = random_expected[(random_expected.distribution == row.distribution) &
                                (random_expected.replicate == row.repeat) &
                                (random_expected.route == row.route) &
                                (random_expected.scope == scope)]
            if len(e) != 1:
                raise ValueError("Expected one archived random-route metric row")
            assert_close(metric["force_component_RMSE_eV_A"], e.force_component_rmse_eV_per_A.iloc[0], "Raw random RMSE")
            assert_close(metric["vector_P99_eV_A"], e.force_vector_p99_eV_per_A.iloc[0], "Raw random P99")
        means = tail_means(selected, q)
        field = "tail_mean_eV_per_A" if row.experiment == "random10" else "tail_mean_eV_A"
        assert_close(means, expected_tail[field].to_numpy(), "Raw absolute tail means")
        for fraction, value in zip(q, means):
            tails.append(dict(experiment=row.experiment, distribution=row.distribution,
                              repeat=row.repeat, route=row.route, q_percent=fraction, tail_mean_eV_A=value))
        membership_audit.append(dict(cache=row.cache, experiment=row.experiment,
                                     before_frames=len(values), after_frames=int(keep.sum()),
                                     excluded_frames=int((~keep).sum()),
                                     rule="included_in_leave_R50_out" if row.experiment == "random10"
                                     else "included_geometry_clean"))
    metrics = pd.DataFrame(metrics)
    tails = pd.DataFrame(tails)
    keys = ["experiment", "distribution", "repeat", "q_percent"]
    baseline = tails[tails.route.isin(["A-DFTref", "DFT60"])][keys + ["tail_mean_eV_A"]]
    tails = tails.merge(baseline.rename(columns={"tail_mean_eV_A": "DFT60_tail_eV_A"}), on=keys, validate="many_to_one")
    tails["tail_difference_eV_A"] = tails.tail_mean_eV_A - tails.DFT60_tail_eV_A
    for d, experiment in [(random_tails.rename(columns={"replicate": "repeat", "top_fraction_percent": "q_percent"}), "random10"),
                          (high_tails.assign(distribution="liquid_amorphous"), "low10_high40")]:
        cols = ["distribution", "repeat", "route", "q_percent"]
        merged = tails[tails.experiment == experiment].merge(d[cols + ["tail_difference_eV_A"]], on=cols,
                                                             suffixes=("_raw", "_archived"), validate="one_to_one")
        if len(merged) != len(d):
            raise ValueError("Incomplete signed-tail comparison")
        assert_close(merged.tail_difference_eV_A_raw, merged.tail_difference_eV_A_archived,
                     "Raw paired signed tails")
    metrics.to_csv(output / "raw_controlled_metrics.csv", index=False)
    tails.to_csv(output / "raw_controlled_tail_differences.csv", index=False)
    pd.DataFrame(membership_audit).to_csv(output / "raw_evaluation_membership_audit.csv", index=False)
    result = dict(caches=len(index), metrics=len(metrics), tail_rows=len(tails),
                  independent_unit="five matched configuration-selection repeats, not training seeds",
                  RMSE="sqrt(mean(force_vector_error_norm**2)/3)",
                  P99="numpy.quantile(norms, 0.99), default linear interpolation",
                  tails="mean of largest ceil(N*q_percent/100) vector-error norms",
                  comparisons="All archived absolute and paired signed tail rows match; Random10 RMSE/P99 match")
    if (root / "controlled/teacher_remaining40/CASE_INDEX.csv").is_file():
        result["teacher_remaining40"] = teacher_remaining40(root, output)
    if (root / "controlled/teacher_fullpool/ARRAY_INDEX.csv").is_file():
        result["teacher_fullpool"] = teacher_fullpool(a, root, output)
    (output / "raw_reanalysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def read_stored_force_labels(path):
    """Read explicit extxyz labels without importing any calculator framework.

    This deliberately small reader requires species, Cartesian positions, forces
    and a cell; unsupported/malformed archived records fail closed.
    """
    frames = []
    with path.open() as stream:
        while True:
            first = stream.readline()
            if not first:
                break
            atoms = int(first.strip())
            if atoms <= 0:
                raise ValueError("Expected positive extxyz atom count")
            header = dict(token.split("=", 1) for token in shlex.split(stream.readline()) if "=" in token)
            properties = header["Properties"].split(":")
            if len(properties) % 3:
                raise ValueError("Invalid extxyz Properties schema")
            columns, offset = {}, 0
            for i in range(0, len(properties), 3):
                name, dtype, count = properties[i:i + 3]
                count = int(count)
                if count < 1:
                    raise ValueError("Invalid extxyz property width")
                columns[name] = (offset, offset + count, dtype)
                offset += count
            values = [stream.readline().split() for _ in range(atoms)]
            if any(len(row) != offset for row in values):
                raise ValueError("Truncated/malformed extxyz atom rows")
            result = {}
            for name in ["species", "pos", "forces"]:
                lo, hi, dtype = columns[name]
                expected_width = 1 if name == "species" else 3
                if hi - lo != expected_width or dtype != ("S" if name == "species" else "R"):
                    raise ValueError("Unsupported stored-label property")
                result[name] = [row[lo] for row in values] if name == "species" else np.array(
                    [[float(v) for v in row[lo:hi]] for row in values])
            result["cell"] = np.array([float(x) for x in header["Lattice"].split()]).reshape(3, 3)
            if not np.isfinite(result["pos"]).all() or not np.isfinite(result["forces"]).all():
                raise ValueError("Stored labels must be finite")
            frames.append(result)
    return frames


def teacher_remaining40(root, output):
    index = pd.read_csv(safe_path(root, "controlled/teacher_remaining40/CASE_INDEX.csv"))
    if len(index) != 10:
        raise ValueError("Expected ten matched teacher unqueried-forty cases")
    rows = []
    for case in index.itertuples(index=False):
        membership = pd.read_csv(safe_path(root, case.membership))
        membership = membership[(membership.distribution == case.distribution) &
                                (membership.replicate == case.replicate)].sort_values("frame_0based")
        keep = bool_column(membership.included_unqueried40)
        if len(keep) != 60 or keep.sum() != 40:
            raise ValueError("Expected sixty labelled geometries and forty unqueried geometries")
        dft = read_stored_force_labels(safe_path(root, case.dft60))
        expected = pd.read_csv(safe_path(root, case.expected_metrics))
        for role, field in [("T0", "T0_predictions60"), ("TR", "TR_predictions60")]:
            pred = read_stored_force_labels(safe_path(root, getattr(case, field)))
            if len(dft) != 60 or len(pred) != 60:
                raise ValueError("Expected sixty stored teacher labels")
            errors = []
            for i in np.flatnonzero(keep):
                if dft[i]["species"] != pred[i]["species"]:
                    raise ValueError("Teacher prediction atom ordering mismatch")
                assert_close(dft[i]["pos"], pred[i]["pos"], "Shared stored-label geometry", atol=1e-6)
                assert_close(dft[i]["cell"], pred[i]["cell"], "Shared stored-label cell", atol=1e-6)
                errors.append(pred[i]["forces"] - dft[i]["forces"])
            residual = np.concatenate(errors, axis=0)
            rmse = np.sqrt(np.mean(residual ** 2))
            p99 = np.quantile(np.linalg.norm(residual, axis=1), 0.99)
            e = expected[(expected.distribution == case.distribution) &
                         (expected.replicate == case.replicate) & (expected.role == role)]
            if len(e) != 1:
                raise ValueError("Missing stored-label reconciliation row")
            assert_close(rmse, e.stored_labels_rmse.iloc[0], "Unqueried-forty stored-label RMSE", atol=1e-12, rtol=0)
            assert_close(p99, e.stored_labels_p99.iloc[0], "Unqueried-forty stored-label P99", atol=1e-12, rtol=0)
            rows.append(dict(distribution=case.distribution, repeat=case.replicate, role=role,
                             frames=40, stored_labels_rmse=rmse, stored_labels_p99=p99,
                             original_audit_rmse=e.original_audit_rmse.iloc[0],
                             original_audit_p99=e.original_audit_p99.iloc[0],
                             rmse_difference_from_plotted_audit=rmse - e.original_audit_rmse.iloc[0],
                             p99_difference_from_plotted_audit=p99 - e.original_audit_p99.iloc[0]))
    data = pd.DataFrame(rows)
    data.to_csv(output / "raw_teacher_remaining40_reconciliation.csv", index=False)
    return dict(cases=10, teacher_metric_rows=len(data),
                max_RMSE_difference_from_plotted_audit=float(data.rmse_difference_from_plotted_audit.abs().max()),
                max_P99_difference_from_plotted_audit=float(data.p99_difference_from_plotted_audit.abs().max()),
                scope="Stored-label reconstruction; precision differences from fresh-batch plotted audit are retained, not normalized")


def teacher_fullpool(a, root, output):
    index = pd.read_csv(safe_path(root, "controlled/teacher_fullpool/ARRAY_INDEX.csv"))
    expected = a.csv("data/controlled/Fig2c_teacher_pool_metrics.csv")
    rows = []
    for row in index.itertuples(index=False):
        filename = safe_path(root, row.file)
        match = re.fullmatch(r"(T0|TR_gamma|TR_liquid)_r(\d+)_(gamma506|liquid423)", filename.stem)
        if not match:
            raise ValueError("Unrecognized full-pool teacher cache name")
        teacher, repeat, pool = match.groups()
        with np.load(filename, allow_pickle=False) as cache:
            residual = np.array(cache["force_residual"], dtype=float)
        schema = json.loads(row.schema)
        if list(residual.shape) != schema["force_residual"]["shape"] or residual.shape[-1] != 3:
            raise ValueError("Unexpected full-pool force-residual shape")
        e = expected[(expected.teacher == teacher) & (expected.replicate == int(repeat)) &
                     (expected.evaluation_pool == pool)]
        if len(e) != 1 or not np.isfinite(residual).all():
            raise ValueError("Incomplete full-pool teacher input")
        rmse = np.sqrt(np.mean(residual ** 2))
        assert_close(rmse, e.force_component_RMSE_eV_A.iloc[0], "Full-source-pool teacher RMSE")
        rows.append(dict(teacher=teacher, repeat=int(repeat), pool=pool,
                         frames=residual.shape[0], pooled_force_component_RMSE_eV_A=rmse))
    if len(rows) != 22:
        raise ValueError("Expected twenty-two full-pool teacher residual arrays")
    pd.DataFrame(rows).to_csv(output / "raw_teacher_fullpool_RMSE.csv", index=False)
    return dict(arrays=len(rows), scope="Source-pool audit; query points retained; not independent holdout accuracy")
