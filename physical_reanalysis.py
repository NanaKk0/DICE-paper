"""Cheap replay of archived physical/projection/timing analysis, with no new jobs.

No inference, embedding fitting, MD, DFT, training or trajectory processing.
Scattering transforms, stored projection transforms, interface regressions and
the original seeded bootstrap are numerical postprocessing of archived arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def close(actual, expected, context, atol=1e-8, rtol=1e-7):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(context + " differs from archived evidence")


def path(root, relative):
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("Evidence path escapes its root")
    return target


def scattering(a, root, output):
    folder = root / "scattering"
    curves = pd.read_csv(folder / "total_G_replicas.csv")
    summary = curves.groupby(["kind", "r_A"]).agg(
        G_mean=("G", "mean"), G_sd=("G", "std"), G_experiment=("G_experiment", "first")).reset_index()
    summary["residual"] = summary.G_mean - summary.G_experiment
    expected = a.csv("data/physical_validation/AL_G_figure_source_data.csv")
    check = summary.merge(expected, on=["kind", "r_A"], suffixes=("_raw", "_archived"), validate="one_to_one")
    if len(check) != len(expected):
        raise ValueError("Incomplete scattering curve grid")
    for field in ["G_mean", "G_sd", "G_experiment", "residual"]:
        close(check[field + "_raw"], check[field + "_archived"], "Three-preparation " + field)
    summary.to_csv(output / "scattering_three_preparation_summary.csv", index=False)

    partial = pd.read_csv(folder / "partial_RDF.csv")
    source = json.loads((folder / "PROVENANCE.json").read_text())
    densities = {row["replica"]: row["rho_A3"] for row in source["inputs"]}
    kernel = pd.read_csv(folder / "modified_lorch_delta_digitized.csv")
    with np.load(folder / "scattering_weights.npz", allow_pickle=False) as saved:
        q = saved["Q"]
        weights = {kind: saved[kind] for kind in ["X", "N"]}
    radii = np.sort(curves.r_A.unique())
    delta = np.interp(radii, kernel.r_A, kernel.delta_A)
    z = delta[:, None] * q[None, :]
    # 3*j1(z)/z: analytic small-z series avoids cancellation and SciPy dependency.
    lorch = np.empty_like(z)
    small = np.abs(z) < 0.01
    zz = z[small] ** 2
    lorch[small] = 1 - zz / 10 + zz ** 2 / 280 - zz ** 3 / 15120
    v = z[~small]
    lorch[~small] = 3 * (np.sin(v) - v * np.cos(v)) / v ** 3
    backward_kernel = lorch * np.sin(radii[:, None] * q[None, :]) * q[None, :]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    rows = []
    for replica, rho in densities.items():
        d = partial[partial.replica == replica]
        pairs = [d[d.pair == pair].sort_values("r_A") for pair in ["Al-Al", "Al-O", "O-O"]]
        r = pairs[0].r_A.to_numpy()
        if len(r) != 600:
            raise ValueError("Expected 600 partial-RDF bins to 12 Angstrom")
        x = np.r_[0, r[r < 12], 12]
        forward_kernel = x[:, None] * np.sin(x[:, None] * q[None, :])
        sminus = np.array([4 * np.pi * rho / q * trapz(
            (np.interp(x, r, pair.g, left=0, right=pair.g.iloc[-1]) - 1)[:, None] * forward_kernel,
            x, axis=0) for pair in pairs])
        for kind in ["X", "N"]:
            close(weights[kind].sum(axis=0), 1, "Faber-Ziman weight normalization")
            transformed = 1 + trapz(backward_kernel * np.sum(weights[kind] * sminus, axis=0)[None, :],
                                     q, axis=1) / (2 * np.pi ** 2 * rho * radii)
            expected_g = curves[(curves.replica == replica) & (curves.kind == kind)].sort_values("r_A")
            close(transformed, expected_g.G, "Partial RDF -> total G(r)", atol=1e-8)
            rows.append(dict(replica=replica, kind=kind,
                             max_abs_transform_difference=float(np.max(np.abs(transformed - expected_g.G))),
                             number_density_A3=rho))
    pd.DataFrame(rows).to_csv(output / "scattering_transform_checks.csv", index=False)
    return dict(preparations=3, G_curve_rows=len(curves), transform_curves=len(rows),
                max_abs_transform_difference=max(r["max_abs_transform_difference"] for r in rows),
                normalization="Per-frame shell volume and pair counts, then replica-mean RDF and replica-mean number density; original protocol preserved",
                caveat="Replay starts from archived replica-mean partial RDF, not full trajectories or per-frame histograms; no experimental pointwise covariance")


def projection(a, root, output):
    folder = root / "projection"
    atoms = pd.read_csv(folder / "ATOMS.csv")
    frames = pd.read_csv(folder / "FRAMES.csv")
    support = pd.read_csv(folder / "ATOMIC_SUPPORT.csv")
    hist = a.csv('data/crossphase/SUPPORT_HISTOGRAM.csv')
    for (element, source), group in hist.groupby(['element','source']):
        group = group.sort_values('left')
        values = support[(support.element==element)&(support.source==source)].atomic_p.to_numpy()
        edges = np.r_[group.left.to_numpy(),group.right.iloc[-1]]
        counts,_ = np.histogram(values,edges)
        close(counts,group.n,'Atomic support histogram counts',atol=0,rtol=0)
        close(counts/(len(values)*np.diff(edges)),group.density,'Atomic support histogram density')
    with np.load(folder / "descriptors.npz", allow_pickle=False) as raw:
        atomic, structure, ptr = raw["atomic"], raw["structure"], raw["atom_ptr"]
    if atomic.shape != (31200, 96) or structure.shape != (260, 192):
        raise ValueError("Expected actual 96D atomic/192D structure representations")
    close(np.diff(ptr), frames.natoms, "Frame/atom pointer correspondence", atol=0, rtol=0)
    if len(atoms) != len(atomic) or not np.isfinite(atomic).all():
        raise ValueError("Invalid actual representation rows")
    settings = {}
    transformed = {}
    for element in ["Al", "O"]:
        mask = atoms.element.eq(element).to_numpy()
        meta = atoms[mask].reset_index(drop=True)
        values = atomic[mask].copy()
        with np.load(folder / f"FITTED_{element}.npz", allow_pickle=False) as saved:
            # Apply stored joint preprocessing; do not refit a PCA or a t-SNE.
            values -= saved["scaler_mean"]
            values /= saved["scaler_scale"]
            transformed[element] = (values - saved["pca_mean"]) @ saved["pca_components"].T
            expected = a.csv(f"data/crossphase/TSNE_{element}.csv")
            close(saved["tsne_embedding"], expected[["tsne_1", "tsne_2"]], "Stored t-SNE embedding", atol=1e-5)
            variance = float(saved["pca_variance_ratio"].sum())
        if list(meta.frame_id) != list(expected.frame_id) or list(meta.atom_index) != list(expected.atom_index):
            raise ValueError("Projection atom membership/order differs from published source")
        setting = json.loads((folder / f"SETTINGS_{element}.json").read_text())
        fit = json.loads((folder / f"FIT_{element}.json").read_text())
        close(variance, fit["PCA_variance_retained"], "Stored joint PCA variance")
        if setting["tsne"]["perplexity"] != 30 or setting["tsne"]["random_state"] != 20261024:
            raise ValueError("Unexpected published projection settings")
        settings[element] = dict(atoms=int(mask.sum()), counts=meta.source.value_counts().to_dict(),
                                 original_dimensions=96, PCA_dimensions=50, PCA_variance_retained=variance,
                                 tsne_settings=setting["tsne"])
    np.savez_compressed(output / "stored_joint_PCA_transform.npz", **transformed)
    return dict(frames=len(frames), atomic_rows=len(atoms), elements=settings,
                scientific_boundary="Joint reference+evaluation StandardScaler/PCA for display; distinct from reference-only element-specific operational router PCA. Stored t-SNE coordinates verified; no embedding fitting or inference.")


def hac_fit(time, thickness, lag):
    centered = time - time.mean()
    design = np.column_stack([np.ones(len(time)), centered])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ thickness
    residual = thickness - design @ beta
    weighted = design * residual[:, None]
    meat = weighted.T @ weighted
    for offset in range(1, lag + 1):
        cross = weighted[offset:].T @ weighted[:-offset]
        meat += (1 - offset / (lag + 1)) * (cross + cross.T)
    covariance = bread @ meat @ bread * len(time) / (len(time) - 2)
    return 50 * beta[1], 50 * np.sqrt(max(float(covariance[1, 1]), 0))


def bootstrap_replay(blocks, samples, seed):
    arrays = [np.asarray(values, dtype=float) for values in blocks]
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        draws[i] = np.mean([float(np.mean(rng.choice(arrays[int(j)], size=len(arrays[int(j)]), replace=True)))
                            for j in selected])
    return np.mean([np.mean(v) for v in arrays]), np.percentile(draws, [2.5, 97.5])


def coexistence(a, root, output):
    folder = root / "coexistence"
    rows, temperatures = [], []
    for campaign in ["replicas_1ns_p2_20261023", "replicas_2350_1ns_p3_20261023"]:
        receipt = json.loads((folder / (campaign + "_RESULTS.json")).read_text())
        series = pd.read_csv(folder / (campaign + "_interface_timeseries.csv"))
        for record in receipt["trajectories"]:
            g = series[series.id == record["id"]].sort_values("time_ps")
            if len(g) != 1000:
                raise ValueError("Expected all 1000 one-ps interface observations per velocity replica")
            for lag in [5, 10, 20]:
                for index, expected in enumerate(record["hac_lag_sensitivity"][str(lag)]):
                    start, stop = expected["window_ps"]
                    keep = ((g.time_ps > start) & (g.time_ps <= stop) &
                            (g.solid_fraction_proxy > 0.1) & (g.solid_fraction_proxy < 0.9))
                    window = g[keep]
                    if len(window) < 20:
                        raise ValueError("Cannot silently omit a censored interface block")
                    velocity, stderr = hac_fit(window.time_ps.to_numpy(), window.solid_thickness_A.to_numpy(), lag)
                    close(velocity, expected["velocity_m_per_s"], "Two-interface block velocity", atol=1e-10)
                    close(stderr, expected["hac_stderr_m_per_s"], "Newey-West HAC standard error", atol=1e-10)
                    rows.append(dict(id=record["id"], temperature_K=record["temperature_K"],
                                     lag_ps=lag, window_start_ps=start, window_end_ps=stop,
                                     points=len(window), velocity_m_s=velocity, HAC_standard_error_m_s=stderr))
        for temperature, expected in receipt["temperatures"].items():
            blocks = [r["block_velocities_m_per_s"] for r in receipt["trajectories"] if r["temperature_K"] == int(temperature)]
            mean, interval = bootstrap_replay(blocks, expected["bootstrap_samples"], expected["bootstrap_seed"])
            close(mean, expected["equal_replica_weighted_mean_m_per_s"], "Equal-replica-weighted velocity")
            close(interval, expected["ci95_m_per_s"], "Original-seed hierarchical block bootstrap", atol=1e-10)
            temperatures.append(dict(temperature_K=int(temperature), velocity_m_s=mean,
                                     ci95_low=interval[0], ci95_high=interval[1],
                                     bootstrap_seed=expected["bootstrap_seed"], bootstrap_samples=20000))
    pd.DataFrame(rows).to_csv(output / "coexistence_block_velocity_HAC_replay.csv", index=False)
    pd.DataFrame(temperatures).to_csv(output / "coexistence_seeded_bootstrap_replay.csv", index=False)
    return dict(velocity_replicas=9, nonoverlapping_100ps_blocks=81, HAC_checks=len(rows),
                discarded_first_ps=100, temperatures_K=[2330, 2340, 2350], bootstrap_draws_per_temperature=20000,
                caveat="Original fixed seeds replayed, not new independent evidence. Three velocity replicas per temperature share one structural seed; no independent structural/finite-size/orientation uncertainty.")


def concurrency(a, root, output):
    modes = a.csv("data/sampling_costs/FORMAL_26_SAMPLING_MODES.csv")
    rows, selections = [], []
    for mode in modes.itertuples():
        sweep = json.loads((root / "concurrency/sweeps" / (mode.mode + ".json")).read_text())
        records = sweep["results"]
        for i, result in enumerate(records):
            native = result["client_timings"]
            rates = []
            for j in range(3):
                interval = [client[j] for client in native]
                work = sum(item["work"] if isinstance(item, dict) else item[0] for item in interval)
                seconds = max(item["seconds"] if isinstance(item, dict) else item[1] for item in interval)
                rates.append(work / seconds)
            close(rates, result["rates"], "Concurrency interval aggregate rates")
            close(np.median(rates), result["aggregate_rate"], "Concurrency median throughput")
            rows.append(dict(mode=mode.mode, trial=i, parallel=result["parallel"],
                             thread_percentage=result["percentage"], tag=result["tag"],
                             measured_aggregate_rate=np.median(rates)))
        confirmed = [r for r in records if r["tag"].startswith("confirm")]
        chosen = confirmed[-1]
        close(chosen["aggregate_rate"], mode.teacher_work_per_second, "Selected confirmed formal mode rate")
        close(chosen["parallel"], mode.teacher_parallel_clients, "Selected concurrency")
        close(chosen["percentage"], mode.teacher_thread_percent, "Selected client thread cap")
        maximum = max(records, key=lambda r: r["aggregate_rate"])
        selections.append(dict(mode=mode.mode, candidate_records=len(records),
                               selected_parallel=chosen["parallel"], selected_thread_percentage=chosen["percentage"],
                               selected_confirmation_rate=chosen["aggregate_rate"],
                               maximum_single_candidate_rate=maximum["aggregate_rate"],
                               confirmation_over_maximum_candidate=chosen["aggregate_rate"] / maximum["aggregate_rate"],
                               policy=sweep.get("search_policy", sweep.get("reuse_note", "archived protocol"))))
    pd.DataFrame(rows).to_csv(output / "concurrency_all_measured_candidates.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "concurrency_selected_confirmations.csv", index=False)
    return dict(formal_modes=len(modes), measured_candidate_records=len(rows),
                selected_confirmation_below_maximum_single_trial=sum(r["confirmation_over_maximum_candidate"] < 1 - 1e-10 for r in selections),
                caveat="Empirical early-stop/local search, not exhaustive global optimization. Selected final confirmation can be slower than an earlier scan trial; all candidates and this difference remain visible.")


def reanalyze(a, research_root, output, section="all"):
    root = Path(research_root).resolve() / "analysis_protocols"
    output = Path(output).resolve()
    if output.is_relative_to(Path(research_root).resolve()) or Path(research_root).resolve().is_relative_to(output):
        raise ValueError("Output must not overlap research assets")
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "ANALYSIS_PROTOCOL_INDEX.json").read_text())
    for record in index["files"]:
        p = path(root, record["file"])
        if p.stat().st_size != record["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Analysis-protocol evidence integrity failure")
    results = dict(verified_protocol_files=len(index["files"]), evidence_bytes=index["total_bytes"])
    for name, function in [("scattering", scattering), ("projection", projection),
                           ("coexistence", coexistence), ("concurrency", concurrency)]:
        if section in ["all", name]:
            results[name] = function(a, root, output)
    (output / "physical_reanalysis.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    from reproduce import Archive
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--research-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--section", default="all", choices=["all", "scattering", "projection", "coexistence", "concurrency"])
    args = parser.parse_args()
    a = Archive(args.data_root)
    out = args.output_dir.resolve()
    if out.is_relative_to(a.root) or a.root.is_relative_to(out):
        parser.error("Data and output directories must not overlap")
    print(json.dumps(reanalyze(a, args.research_root, out, args.section), indent=2))
