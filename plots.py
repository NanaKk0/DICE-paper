"""Data-derived verification figures, not facsimiles of approved paper artwork.

All observations remain archived. Curves summarize all five selection repeats with
sample SD (not five training seeds). No new bootstrap or scientific simulation.
"""
from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib import font_manager
import numpy as np
import pandas as pd

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#7A7A7A", "#D55E00"]
MODEL_STYLE = {"DFT": ("#000000", "-"), "DICE": ("#0072B2", "-"),
               "Direct": ("#E69F00", "--")}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.5, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "lines.linewidth": 1,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def same_window_tma_references(frame, compositions, low_K=303.15, high_K=1173.15):
    """Match the current SI comparison by technique and fit endpoints."""
    mask = (frame.method.eq("TMA") & np.isclose(frame.T_low_K, low_K, rtol=0, atol=1e-8)
            & np.isclose(frame.T_high_K, high_K, rtol=0, atol=1e-8))
    selected = frame.loc[mask & frame.composition.isin(compositions)].copy()
    for comp in compositions:
        if len(selected[selected.composition == comp]) != 1:
            raise ValueError("Expected one matching-window TMA coefficient for " + comp)
    return selected


class Plotter:
    def __init__(self, archive, output):
        self.a, self.output, self.records = archive, output, []
        self.sources = set()

    def csv(self, name):
        self.sources.add(name)
        return self.a.csv(name)

    def figure(self, stem, panels, cols=2, width_mm=183):
        """Panel callbacks draw from archived numerics; every used file is logged."""
        self.sources = set()
        rows = math.ceil(len(panels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(width_mm / 25.4, min(170, rows * 47) / 25.4),
                                 squeeze=False, layout="constrained")
        for i, (title, draw) in enumerate(panels):
            ax = axes.flat[i]
            draw(ax)
            ax.set_title(title, loc="left", pad=5)
            ax.text(-0.12, 1.08, chr(97 + i), transform=ax.transAxes,
                    fontsize=8, fontweight="bold", va="top")
        for ax in list(axes.flat)[len(panels):]:
            ax.remove()
        # Arial installations need not contain Unicode super/subscript glyphs.
        # Mathtext keeps those units legible without changing approved artwork.
        substitutions = {"⁻¹": r"$^{-1}$", "⁻⁶": r"$^{-6}$", "³": r"$^3$",
                         "₂": r"$_2$", "₃": r"$_3$"}
        for text in fig.findobj(Text):
            label = text.get_text()
            for old, new in substitutions.items():
                label = label.replace(old, new)
            text.set_text(label)
        fig.savefig(self.output / (stem + ".svg"))
        fig.savefig(self.output / (stem + ".pdf"))
        fig.savefig(self.output / (stem + ".png"), dpi=300)
        plt.close(fig)
        self.records.append(dict(stem=stem, panels=[p[0] for p in panels],
                                 sources=sorted(self.sources), width_mm=width_mm,
                                 height_mm=min(170, rows * 47),
                                 output_role="numerical verification; not approved artwork"))

    def tails(self, ax, file, scope=None, distribution=None, paired=True, zoom=False):
        d = self.csv(file)
        q = "q_percent" if "q_percent" in d else "top_fraction_percent"
        if scope:
            d = d[d.scope == scope]
        if distribution:
            d = d[d.distribution == distribution]
        if zoom:
            d = d[(d[q] >= 0.01) & (d[q] <= 1)]
        value = "tail_difference_eV_A" if paired else (
            "tail_mean_eV_per_A" if "tail_mean_eV_per_A" in d else "tail_mean_eV_A")
        for i, (route, g) in enumerate(d.groupby("route", sort=False)):
            s = g.groupby(q)[value].agg(["mean", "std"])
            if not np.all(s.index > 0):
                raise ValueError("Tail fraction must be positive for log axis")
            x, y, sd = s.index.to_numpy(), s["mean"].to_numpy(), s["std"].to_numpy()
            ax.plot(x, y, color=PALETTE[i % len(PALETTE)], ls=["-", "--", ":"][i % 3], label=route)
            ax.fill_between(x, y - sd, y + sd, color=PALETTE[i % len(PALETTE)], alpha=0.12)
        if paired:
            ax.axhline(0, color="black", lw=0.6)
        ax.set(xscale="log", xlabel="Largest-error tail (%)",
               ylabel="Tail difference (eV Å⁻¹)" if paired else "Tail mean (eV Å⁻¹)")
        ax.legend(ncol=2)

    def pooled(self, ax):
        d = self.csv("data/prediction_and_mil/frames.csv")
        s = d.groupby(["system", "route"]).agg(ss=("force_ss", "sum"), n=("force_n", "sum"))
        systems = list(d.system.unique())
        x = np.arange(len(systems))
        for i, route in enumerate(["Direct", "DICE"]):
            y = [np.sqrt(s.loc[(v, route), "ss"] / s.loc[(v, route), "n"]) for v in systems]
            ax.bar(x + (i - 0.5) * 0.3, y, width=0.3, label=route,
                   color=MODEL_STYLE[route][0], edgecolor="black", linewidth=0.3)
        ax.set_xticks(x, systems)
        ax.set_ylabel("Pooled force-component RMSE\n(eV Å⁻¹)")
        ax.legend()

    def teacher_pool(self, ax):
        d = self.csv("data/controlled/Fig2c_teacher_pool_metrics.csv")
        for i, (teacher, g) in enumerate(d.groupby("teacher", sort=False)):
            s = g.groupby("evaluation_pool", sort=False).force_component_RMSE_eV_A.agg(["mean", "std"])
            x = np.arange(len(s)) + (i - 1) * 0.2
            ax.errorbar(x, s["mean"], yerr=s["std"].fillna(0), fmt=["o", "s", "^"][i % 3],
                        ms=3, color=PALETTE[i], label=teacher)
        ax.set_xticks(np.arange(len(s)), s.index, rotation=15)
        ax.set_ylabel("Teacher force RMSE (eV Å⁻¹)")
        ax.legend()

    def transitions(self, ax):
        d = self.csv("data/controlled/support_transitions_common413.csv")
        s = d.groupby(["replicate", "transition"]).size().unstack(fill_value=0)
        bottom = np.zeros(len(s))
        for i, name in enumerate(s.columns):
            ax.bar(s.index, s[name], bottom=bottom, label=name.replace("_", " "), color=PALETTE[i])
            bottom += s[name].to_numpy()
        ax.set(xlabel="Configuration-selection repeat", ylabel="Common candidate geometries")
        ax.legend(fontsize=5)

    def teachers(self, ax):
        d = self.csv("data/crossphase/TEACHER_PLOTTED_DATA.csv")
        ax.bar(d.model, d.force_component_RMSE_eV_A, color=["#7A7A7A", "#0072B2"])
        ax.set_ylabel("120-test force RMSE (eV Å⁻¹)")
        ax.tick_params(axis="x", labelrotation=12)

    def students(self, ax):
        d = self.csv("data/crossphase/STUDENT_PLOTTED_DATA.csv")
        ax.plot(d.student_training_geometries, d.force_component_RMSE_eV_A, "o-", ms=2.5)
        sel = d[d["round"].isin([2, 10])]
        ax.plot(sel.student_training_geometries, sel.force_component_RMSE_eV_A, "s", ms=4,
                color="#000000", label="940 → 1320 expansion")
        ax.set(xlabel="Historical student training geometries (100k steps)",
               ylabel="120-test force RMSE (eV Å⁻¹)")
        ax.legend()

    def tsne(self, ax, element):
        d = self.csv("data/crossphase/TSNE_" + element + ".csv")
        for i, (source, g) in enumerate(d.groupby("source", sort=False)):
            ax.scatter(g.tsne_1, g.tsne_2, s=0.4, alpha=0.25, rasterized=True,
                       color=PALETTE[i], label=f"{source} (n={len(g)})")
        ax.set(xlabel="t-SNE coordinate 1", ylabel="t-SNE coordinate 2")
        ax.legend(markerscale=4)

    def support_hist(self, ax, element):
        d = self.csv("data/crossphase/SUPPORT_HISTOGRAM.csv")
        for i, (source, g) in enumerate(d[d.element == element].groupby("source", sort=False)):
            ax.stairs(g.density, np.r_[g.left.to_numpy(), g.right.iloc[-1]], label=source, color=PALETTE[i])
        ax.set(xlabel="Empirical support p (not error probability)", ylabel="Density")
        ax.legend()

    def local_distances(self, ax):
        for i, coord in enumerate([4, 5, 6]):
            d = self.csv(f"data/local_environments/AlO{coord}_coordinates_and_correspondence.csv")
            for j, (name, g) in enumerate(d.groupby("set", sort=False)):
                y = g.AlO_distance_A
                if not np.isfinite(y).all():
                    raise ValueError("Local Al–O distances must all be finite")
                ax.plot(np.full(len(y), coord) + (j - 0.5) * 0.12, y, ["o", "s"][j % 2],
                        ms=3, color=PALETTE[j], label=name if i == 0 else None)
        ax.set(xlabel="Al coordination number", ylabel="Archived Al–O distance (Å)", xticks=[4, 5, 6])
        ax.legend()

    def ysz_transitions(self, ax):
        d = self.csv("data/provenance/YSZ500k_error_transitions.csv")
        categories = list(d.element.unique())
        for i, ((transition, threshold), g) in enumerate(d.groupby(["transition", "threshold_eV_A"], sort=False)):
            x = np.array([categories.index(s) for s in g.element]) + (i - 1.5) * 0.12
            ax.errorbar(x, g.fraction, yerr=[g.fraction - g.CI_low, g.CI_high - g.fraction],
                        fmt="o" if threshold == 0.3 else "s", ms=3,
                        label=f"{transition}, {threshold} eV Å⁻¹",
                        color=PALETTE[0] if transition == "removed" else PALETTE[1])
        ax.set_xticks(range(len(categories)), categories)
        ax.set_ylabel("Atom observations\n(transition fraction)")
        ax.legend()

    def phonon(self, ax, phase="alpha", comparison=False):
        if comparison:
            d = self.csv("data/phonons/S26_phonon_dispersion.csv")
            meta = self.a.json("data/phonons/S26_q_path.json")["phases"][phase]
            self.sources.add("data/phonons/S26_q_path.json")
            series = [("reference_frequency_THz", "DFT", "black", "-"),
                      ("DFT_only_frequency_THz", "Direct", "#E69F00", "--"),
                      ("DICE_frequency_THz", "DICE", "#0072B2", "-")]
            # Main path is the complete Γ M K Γ A L H A, not Γ M K Γ only.
            d = d[d.distance <= meta["tick_distances"][7] + 1e-10]
            ticks, labels = meta["tick_distances"][:8], meta["tick_labels"][:8]
        else:
            d = self.csv("data/physical_validation/Al_phonon_band.csv")
            d = d[d.phase == phase]
            meta = self.a.json("data/physical_validation/Al_phonon_path_metadata.json")["phases"][phase]
            self.sources.add("data/physical_validation/Al_phonon_path_metadata.json")
            series = [("reference_frequency_THz", "DFT", "black", "-"),
                      ("model_frequency_THz", "DICE", "#0072B2", "--")]
            ticks, labels = meta["tick_distances"], meta["tick_labels"]
        for column, label, color, ls in series:
            for i, (_, g) in enumerate(d.groupby("reference_mode", sort=False)):
                ax.plot(g.distance, g[column], color=color, ls=ls, lw=0.5, alpha=0.75,
                        label=label if i == 0 else None)
        # Equal-position discontinuity ticks are joined, never silently dropped.
        merged = {}
        for x, lab in zip(ticks, labels):
            lab = "Γ" if lab == "G" else lab
            merged[x] = merged[x] + "|" + lab if x in merged else lab
        ax.set_xticks(list(merged), list(merged.values()), fontsize=5)
        ax.set(xlabel="Reciprocal-space path", ylabel="Frequency (THz)")
        ax.legend(ncol=len(series))

    def path_errors(self, ax):
        d = self.csv("data/prediction_and_mil/all_path_profiles.csv")
        paths = list(d.path.unique())
        for i, route in enumerate(["Direct", "DICE"]):
            y = []
            for path in paths:
                g = d[(d.path == path) & (d.route == route)].sort_values("image")
                e = 1000 * (g.model_eV.to_numpy() - g.DFT_eV.to_numpy())
                t = g.arc_fraction.to_numpy()
                y.append(np.mean(np.abs((e - ((1 - t) * e[0] + t * e[-1]))[1:-1])))
            ax.plot(np.arange(len(paths)), y, ["s--", "o-"][i], label=route, ms=3,
                    color=MODEL_STYLE[route][0])
        ax.set_xticks(range(len(paths)), paths, rotation=45, ha="right")
        ax.set_ylabel("Interior profile MAE\n(meV cell⁻¹; DFT-arc detrended)")
        ax.legend()

    def phase_order(self, ax):
        d = self.csv("data/mil120/mil_phase_order.csv")
        for i, (model, g) in enumerate(d.groupby("model", sort=False)):
            ax.plot(g.phase, g.relative_energy_meV_cell, ["o-", "s--", "^:"][i % 3],
                    label=model, ms=3, color=PALETTE[i])
        ax.set(xlabel="MIL-120 phase", ylabel="Relative energy (meV cell⁻¹)")
        ax.legend()

    def al_eos(self, ax):
        d = self.csv("data/physical_validation/AL_EOS_PLOT.csv")
        for i, (phase, g) in enumerate(d.groupby("phase", sort=False)):
            g = g.sort_values("V_fu")
            ax.plot(g.V_fu, g.DFT_relative_meV_atom, "o", ms=2, color=PALETTE[i], label=phase + " DFT")
            ax.plot(g.V_fu, g.MLIP_relative_meV_atom, "-", color=PALETTE[i], label=phase + " DICE")
        ax.set(xlabel="Volume (Å³ formula unit⁻¹)", ylabel="Relative energy (meV atom⁻¹)")
        ax.legend(ncol=2)

    def al_gr(self, ax):
        d = self.csv("data/physical_validation/AL_G_figure_source_data.csv")
        for i, (kind, g) in enumerate(d.groupby("kind", sort=False)):
            ax.plot(g.r_A, g.G_mean, color=PALETTE[i], label=kind + " DICE")
            ax.fill_between(g.r_A, g.G_mean - g.G_sd, g.G_mean + g.G_sd, color=PALETTE[i], alpha=0.15)
            ax.plot(g.r_A, g.G_experiment, "--", color=PALETTE[i], label=kind + " experiment")
        ax.set(xlabel="r (Å)", ylabel="G(r)")
        ax.legend(ncol=2)

    def adsorption(self, ax):
        d = self.csv("data/physical_validation/MIL_MEAN_SD.csv")
        raw = self.csv("data/physical_validation/MIL_REPLICATES.csv")
        for i, (temperature, g) in enumerate(d.groupby("temperature_K", sort=False)):
            g = g.sort_values("pressure_bar")
            color = PALETTE[i]
            ax.errorbar(g.pressure_bar, g.uptake_mean_mmol_g, yerr=g.uptake_sd_mmol_g,
                        fmt="o-", ms=2.5, color=color, label=f"DICE {temperature} K")
            ax.plot(g.pressure_bar, g.experiment_fit_mmol_g, "--", color=color, label=f"Experiment fit {temperature} K")
            r = raw[raw.temperature_K == temperature]
            ax.scatter(r.pressure_bar, r.uptake_mmol_g, s=3, alpha=0.3, color=color)
        if not (d.pressure_bar > 0).all():
            raise ValueError("Log pressure axis requires positive pressure")
        ax.set(xscale="log", xlabel="Pressure (bar)", ylabel="CO₂ uptake (mmol g⁻¹)")
        ax.legend(ncol=2)

    def conductivity(self, ax):
        d = self.csv("data/physical_validation/YSZ_CONDUCTIVITY.csv")
        for i, (temperature, g) in enumerate(d.groupby("T_K", sort=False)):
            x = g.experiment_mol_percent
            color = PALETTE[i]
            ax.errorbar(x, g.simulation_NE_mean_S_cm, yerr=g.simulation_NE_SD_S_cm,
                        fmt="o-", ms=3, color=color, label=f"NE {temperature} K")
            ax.errorbar(x, g.experiment_S_cm,
                        yerr=[g.experiment_S_cm - g.experiment_digitization_low_S_cm,
                              g.experiment_digitization_high_S_cm - g.experiment_S_cm],
                        fmt="s--", ms=3, color=color, label=f"Experiment {temperature} K")
        ax.set(xlabel="Experimental Y₂O₃ content (mol%; nearby compositions)", ylabel="Conductivity (S cm⁻¹)")
        ax.legend(ncol=2)

    def latency(self, ax, system):
        d = self.csv("data/performance/THREE_SYSTEM_LATENCY.csv")
        for i, (model, g) in enumerate(d[d.system == system].groupby("model", sort=False)):
            g = g.sort_values("atoms")
            if not ((g.atoms > 0) & (g.latency_ms_median > 0)).all():
                raise ValueError("Latency log axes require positive measurements")
            ax.plot(g.atoms, g.latency_ms_median, ["o-", "s--", "^:"][i % 3], ms=2.5,
                    color=PALETTE[i], label=model)
            ax.fill_between(g.atoms, g.latency_ms_q25, g.latency_ms_q75, color=PALETTE[i], alpha=0.12)
        # Do not draw fabricated latency at an OOM location.
        oom = self.csv("data/performance/OOM_OBSERVATIONS.csv")
        for i, row in enumerate(oom[oom.system == system].itertuples()):
            ax.axvline(row.atoms, color="#7A7A7A", lw=0.4, ls=":")
        ax.set(xscale="log", yscale="log", xlabel="Atoms", ylabel="Measured latency (ms step⁻¹)")
        ax.legend()

    def cost(self, ax):
        d = self.csv("data/sampling_costs/FORMAL_26_SAMPLING_MODES.csv")
        s = d.groupby("system", sort=False)[["student_hours", "teacher_estimated_hours"]].sum()
        x = np.arange(len(s))
        ax.bar(x - 0.17, s.student_hours, width=0.34, label="Student recorded", color="#0072B2")
        ax.bar(x + 0.17, s.teacher_estimated_hours, width=0.34, label="Teacher estimated", color="#7A7A7A")
        ax.set_xticks(x, ["Al2O3", "MIL-120", "YSZ"])
        ax.set_ylabel("Development-sampling wall time (h)")
        ax.legend()

    def allocation(self, ax, panel):
        d = self.csv("data/controlled/S2_supervision_allocation.csv")
        d = d[d.panel == panel]
        routes = list(d.route.unique())
        bottoms = np.zeros(len(routes))
        for i, ((geometry, label), g) in enumerate(d.groupby(["geometry_group", "label_source"], sort=False)):
            values = np.array([g.loc[g.route == r, "frames"].sum() for r in routes])
            ax.bar(routes, values, bottom=bottoms, color=PALETTE[i % len(PALETTE)],
                   label=geometry + " / " + label)
            bottoms += values
        ax.set_ylabel("Training / unused geometries")
        ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(1.01, 1))

    def teacher_pairs(self, ax, distribution):
        d = self.csv("data/controlled/S3_unqueried_teacher_pairs.csv")
        d = d[d.distribution == distribution]
        for repeat, g in d.groupby("replicate", sort=False):
            g = g.set_index("role").loc[["T0", "TR"]]
            ax.plot([0, 1], g.force_component_rmse_eV_per_A, "o-", ms=3,
                    color=PALETTE[int(repeat) - 1], label=f"repeat {repeat}")
        ax.set_xticks([0, 1], ["T0", "TR"])
        ax.set_ylabel("Unqueried-forty force RMSE (eV Å⁻¹)")
        ax.legend()

    def partitions(self, ax, metric):
        d = self.csv("data/support/S6_ALL_20_METRIC_VALUES.csv")
        d = d[d.metric == metric]
        for i, (partition, g) in enumerate(d.groupby("partition", sort=False)):
            ax.plot(g.x_categorical, g.value_eV_A, "o", ms=3, label=partition, color=PALETTE[i])
            ax.errorbar(i, g.value_eV_A.mean(), yerr=g.value_eV_A.std(ddof=1), fmt="_", ms=12,
                        color="black", capsize=3)
        ax.set_xticks([0, 1], ["high p > 0.1", "low p ≤ 0.1"])
        ax.set_ylabel("Force RMSE (eV Å⁻¹)" if "component" in metric else "Vector P99 (eV Å⁻¹)")

    def parity_data(self, system, quantity):
        if system == "Al":
            if quantity == "force":
                d = self.csv("data/accuracy/FORCE_COMPONENT_PARITY_74040.csv")
                return [("DICE", d.DFT_eV_A.to_numpy(), d.expanded_eV_A.to_numpy()),
                        ("Direct", d.DFT_eV_A.to_numpy(), d.direct_DFT_eV_A.to_numpy())]
            d = self.csv("data/accuracy/ENERGY_PARITY_200.csv")
            ref = d.DFT_meV_atom - d.shared_DFT_center_meV_atom
            return [("DICE", ref.to_numpy(), (d.expanded_meV_atom - d.shared_DFT_center_meV_atom).to_numpy()),
                    ("Direct", ref.to_numpy(), (d.direct_DFT_meV_atom - d.shared_DFT_center_meV_atom).to_numpy())]
        tag = "MIL" if system == "MIL" else "YSZ500"
        if quantity == "force":
            d = self.csv(f"data/accuracy/S_direct_{tag}_parity_force.csv")
            return [("DICE", d.DFT_eV_A.to_numpy(), d.DICE_eV_A.to_numpy()),
                    ("Direct", d.DFT_eV_A.to_numpy(), d.direct_eV_A.to_numpy())]
        d = self.csv(f"data/accuracy/S_direct_{tag}_parity_energy.csv")
        values = []
        for model, g in d.groupby("model", sort=False):
            label = "Direct" if model in ["mil_direct500", "ysz_C500"] else "DICE"
            values.append((label, (g.DFT_meV_atom - g.shared_DFT_center_meV_atom).to_numpy(),
                           (g.predicted_meV_atom - g.shared_DFT_center_meV_atom).to_numpy()))
        return values

    def parity(self, ax, system, quantity, role):
        series = self.parity_data(system, quantity)
        unit = "eV Å⁻¹" if quantity == "force" else "meV atom⁻¹"
        if role == "parity":
            lo = min(min(x.min(), y.min()) for _, x, y in series)
            hi = max(max(x.max(), y.max()) for _, x, y in series)
            for label, x, y in series:
                ax.scatter(x, y, s=0.5, alpha=0.2, rasterized=True, color=MODEL_STYLE[label][0], label=label)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.6)
            ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel=f"DFT ({unit})", ylabel=f"Prediction ({unit})")
            ax.legend(markerscale=4)
        elif role == "residual":
            for label, x, y in series:
                ax.scatter(x, y - x, s=0.5, alpha=0.2, rasterized=True, color=MODEL_STYLE[label][0], label=label)
            ax.axhline(0, color="black", lw=0.6)
            ax.set(xlabel=f"DFT ({unit})", ylabel=f"Prediction − DFT ({unit})")
            ax.legend(markerscale=4)
        else:
            for label, x, y in series:
                err = np.sort(np.abs(y - x))
                ax.plot(err, np.arange(1, len(err) + 1) / len(err), label=f"{label}, n={len(err)}",
                        color=MODEL_STYLE[label][0], ls=MODEL_STYLE[label][1])
            ax.set(xlabel=f"Absolute error ({unit})", ylabel="Empirical cumulative fraction")
            ax.legend()

    def mil_eos(self, ax, phase):
        d = self.csv("data/prediction_and_mil/all_EOS.csv")
        d = d[d.phase == phase]
        for route, g in d.groupby("route", sort=False):
            g = g.sort_values("volume_A3")
            color, ls = MODEL_STYLE.get(route, ("#7A7A7A", "--"))
            ax.plot(g.volume_A3, g.sp_relative_meV, color=color, ls=ls, label=route + " SP")
            ax.plot(g.volume_A3, g.relaxed_relative_meV, ":", color=color, label=route + " relaxed")
        g = d[d.route == d.route.iloc[0]].sort_values("volume_A3")
        ax.plot(g.volume_A3, g.dft_relative_meV, "ko", ms=2, label="DFT")
        ax.set(xlabel="Volume (Å³ cell⁻¹)", ylabel="Relative energy (meV cell⁻¹)")
        ax.legend(ncol=2)

    def path_profile(self, ax, path, relaxed=False):
        if relaxed:
            d = self.csv("data/physical_validation/MIL_relaxed_NEB_profiles.csv")
            for i, (method, g) in enumerate(d[d.path == path].groupby("method", sort=False)):
                ax.plot(g.normalised_arc, g.relative_energy_meV, ["o-", "s--", "^:"][i % 3],
                        ms=2, color=["black", *PALETTE][i], label=method)
        else:
            d = self.csv("data/prediction_and_mil/all_path_profiles.csv")
            d = d[d.path == path]
            for route, g in d.groupby("route", sort=False):
                color, ls = MODEL_STYLE[route]
                ax.plot(g.arc_fraction, g.model_relative_meV, marker="o", ls=ls, color=color, ms=2, label=route)
            g = d[d.route == "DICE"]
            ax.plot(g.arc_fraction, g.DFT_relative_meV, "ko-", ms=2, lw=0.6, label="DFT")
        ax.set(xlabel="Own relaxed arc fraction" if relaxed else "DFT-path arc fraction",
               ylabel="Initial-zero energy (meV cell⁻¹)")
        ax.legend(fontsize=5)

    def al_phases(self, ax, quantity):
        d = self.csv("data/physical_validation/Al_eleven_phase_comparison.csv")
        if quantity == "energy":
            columns = ["relative_energy_eV_per_fu_DFT", "relative_energy_eV_per_fu_DICE"]
            label = "Relative energy (eV formula unit⁻¹)"
        else:
            columns = ["volume_A3_per_fu_DFT", "volume_A3_per_fu_DICE"]
            label = "Volume (Å³ formula unit⁻¹)"
        for i, col in enumerate(columns):
            ax.plot(d.phase, d[col], ["ko-", "s--"][i], ms=3, label=["DFT", "DICE"][i])
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel(label)
        ax.legend()

    def interface(self, ax, temperature):
        file = "AL_INTERFACE_2350.csv" if temperature == 2350 else "AL_INTERFACE_2330_2340.csv"
        d = self.csv("data/physical_validation/" + file)
        for i, (replica, g) in enumerate(d[d.temperature_K == temperature].groupby("replica", sort=False)):
            ax.plot(g.time_ps, g.solid_thickness_A, color=PALETTE[i], label=f"replica {replica}")
        ax.set(xlabel="Time (ps)", ylabel="Solid-thickness proxy (Å)")
        ax.legend()

    def velocities(self, ax):
        d = self.csv("data/physical_validation/AL_COEXISTENCE.csv")
        ax.errorbar(d.temperature_K, d.velocity_m_s,
                    yerr=[d.velocity_m_s - d.ci95_low, d.ci95_high - d.velocity_m_s], fmt="o", ms=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set(xlabel="Temperature (K)", ylabel="Interface velocity (m s⁻¹), 95% interval")

    def transport(self, ax, metric):
        d = self.csv("data/ysz/transport_normalized_whole_preparation_bootstrap.csv")
        for i, (comp, g) in enumerate(d[d.metric == metric].groupby("comp", sort=False)):
            ax.errorbar(g["T"], g.ratio, yerr=[g.ratio - g.CI_low, g.CI_high - g.ratio],
                        fmt=["o-", "s--", "^:", "v-.", "D-"][i], ms=2.5, color=PALETTE[i], label=comp)
        ax.set(xlabel="Temperature (K)", ylabel=metric + " / 5.8YSZ reference")
        ax.legend(ncol=2)

    def lattice(self, ax):
        d = self.csv("data/physical_validation/YSZ_lattice_bins_common_window.csv")
        for i, (comp, g) in enumerate(d.groupby("composition", sort=False)):
            for j, (_, h) in enumerate(g.groupby("seed", sort=False)):
                ax.plot(h.temperature_K, h.lattice_parameter_A, lw=0.6, alpha=0.7,
                        color=PALETTE[i], label=comp if j == 0 else None)
        ax.set(xlabel="Temperature (K)", ylabel="Lattice parameter (Å)")
        ax.legend(ncol=2)

    def cte(self, ax):
        d = self.csv("data/physical_validation/YSZ_thermal_coefficients_per_preparation.csv")
        exp = self.csv("data/experimental_references/verified_experimental_coefficients.csv")
        categories = list(d.composition.unique())
        exp = same_window_tma_references(exp, categories)
        for i, comp in enumerate(categories):
            g = d[d.composition == comp]
            ax.plot(i + np.linspace(-0.1, 0.1, len(g)), g.alpha_1e6_per_K, "o", ms=3,
                    color="#0072B2", label="Simulation preparations" if i == 0 else None)
            e = exp[exp.composition == comp]
            ax.plot(np.full(len(e), i + 0.2), e.linear_CTE_1e6_per_K, "ks", ms=3,
                    label="TMA, 303.15–1173.15 K" if i == 0 else None)
        ax.set_xticks(range(len(categories)), categories)
        ax.set_ylabel("Linear expansion (10⁻⁶ K⁻¹)")
        ax.legend()


def replot(archive, output):
    p = Plotter(archive, output)
    def tail(ax, **kw):
        p.tails(ax, "data/controlled/Fig2b_random_paired_tail_differences.csv", **kw)
    def supported(ax, **kw):
        p.tails(ax, "data/controlled/Fig2f_supported_paired_tail_differences.csv", **kw)
    p.figure("Fig2_data", [
        ("Random route: crystal; five selection repeats", lambda ax: tail(ax, distribution="gamma_smrcok", zoom=True)),
        ("Random route: liquid/amorphous; signed paired tails", lambda ax: tail(ax, distribution="liquid_amorphous", zoom=True)),
        ("Teacher source-pool errors", p.teacher_pool),
        ("Fixed common-413 support transitions", p.transitions),
        ("High-support route: signed paired tails", lambda ax: supported(ax, zoom=True))])
    p.figure("Fig3_data", [("Teacher adaptation on 120 test geometries", p.teachers),
        ("Historical 100k-step student expansion", p.students),
        ("Archived Al representation (all points)", lambda ax: p.tsne(ax, "Al")),
        ("Archived O representation (all points)", lambda ax: p.tsne(ax, "O")),
        ("O support distribution", lambda ax: p.support_hist(ax, "O")),
        ("Local coordination distance evidence", p.local_distances)])
    p.figure("Fig4_data", [("Final 500k-step production students", p.pooled),
        ("YSZ error transitions; archived cluster intervals", p.ysz_transitions),
        ("α-Al₂O₃: full Γ–M–K–Γ–A–L–H–A main path", lambda ax: p.phonon(ax, comparison=True)),
        ("Nine shared DFT paths; endpoint-detrended errors", p.path_errors),
        ("MIL-120 relative phase energies", p.phase_order)])
    p.figure("Fig5_data", [("Al₂O₃ equations of state", p.al_eos),
        ("Al₂O₃ pair-distribution evidence", p.al_gr),
        ("MIL-120 adsorption: all three repeats", p.adsorption),
        ("YSZ conductivity; nearby experimental compositions", p.conductivity)])
    p.figure("Fig6_data", [(s + ": measured latency; quartile band", lambda ax, s=s: p.latency(ax, s))
                            for s in ["al2o3", "mil", "ysz"]] +
                           [("26-mode development-sampling costs", p.cost)])
    p.figure("SupplementaryFig1_data", [("Random10 route: supervision allocation", lambda ax: p.allocation(ax, "a")),
        ("High40 route: supervision allocation (separate experiment)", lambda ax: p.allocation(ax, "e"))], cols=1)
    p.figure("SupplementaryFig2_data", [(s + ": paired five selection repeats", lambda ax, s=s: p.teacher_pairs(ax, s))
                                         for s in ["gamma_smrcok", "liquid_amorphous"]])
    p.figure("SupplementaryFig3_data", [(s + (": signed paired difference" if paired else ": absolute tail mean"),
        lambda ax, s=s, paired=paired: p.tails(ax, "data/controlled/S4_absolute_and_paired_tails.csv",
                                               distribution=s, paired=paired))
        for s in ["gamma_smrcok", "liquid_amorphous"] for paired in [False, True]])
    p.figure("SupplementaryFig4_data", [("Force-component RMSE: five selection repeats",
        lambda ax: p.partitions(ax, "force_component_rmse_eV_per_A")),
        ("Force-vector P99: five selection repeats", lambda ax: p.partitions(ax, "force_vector_p99_eV_per_A"))])
    p.figure("SupplementaryFig5_data", [("High40: absolute tail mean", lambda ax: supported(ax, paired=False)),
        ("High40: paired difference; all archived fractions", lambda ax: supported(ax)),
        ("High40: signed 0.01–1% rare tails", lambda ax: supported(ax, zoom=True))])
    for n, system in [(6, "Al"), (7, "MIL"), (8, "YSZ")]:
        p.figure(f"SupplementaryFig{n}_data", [
            (system + " " + quantity + ": " + role,
             lambda ax, q=quantity, r=role, s=system: p.parity(ax, s, q, r))
            for quantity in ["energy", "force"] for role in ["parity", "residual", "ECDF"]], cols=3)
    phases = list(archive.csv("data/prediction_and_mil/all_EOS.csv").phase.unique())
    p.figure("SupplementaryFig9_data", [("MIL-120 phase " + s, lambda ax, s=s: p.mil_eos(ax, s)) for s in phases], cols=3)
    paths = list(archive.csv("data/prediction_and_mil/all_path_profiles.csv").path.unique())
    p.figure("SupplementaryFig10_data", [(s + ": common DFT geometries", lambda ax, s=s: p.path_profile(ax, s))
                                          for s in paths], cols=3)
    paths = list(archive.csv("data/physical_validation/MIL_relaxed_NEB_profiles.csv").path.unique())
    p.figure("SupplementaryFig11_data", [(s + ": distinct relaxed paths", lambda ax, s=s: p.path_profile(ax, s, relaxed=True))
                                          for s in paths], cols=3)
    p.figure("SupplementaryFig12_data", [("Eleven-phase relative energies", lambda ax: p.al_phases(ax, "energy")),
        ("Eleven-phase volumes", lambda ax: p.al_phases(ax, "volume"))])
    p.figure("SupplementaryFig13_data", [(s + ": all archived path segments", lambda ax, s=s: p.phonon(ax, phase=s))
                                          for s in ["alpha", "kappa", "theta"]], cols=1)
    p.figure("SupplementaryFig14_data", [(f"{t} K: three interface traces", lambda ax, t=t: p.interface(ax, t))
                                          for t in [2330, 2340, 2350]] + [("Interface velocity; archived intervals", p.velocities)])
    p.figure("SupplementaryFig15_data", [(metric + ": whole-preparation bootstrap intervals",
        lambda ax, m=metric: p.transport(ax, m)) for metric in ["D_star", "DJ", "f_s_ratio_of_means"]], cols=1)
    p.figure("SupplementaryFig16_data", [("All common-window lattice bins / preparations", p.lattice),
        ("Expansion coefficients; matching TMA interval", p.cte)])
    report = dict(backend="Python/matplotlib", task_card="T03 primary; archived downstream-physics evidence",
                  archetype="quantitative verification grids", figures=p.records,
                  available_font=font_manager.findfont(font_manager.FontProperties(family=["sans-serif"])),
                  statistics="Archived estimates/intervals retained; selection repeats are not random training seeds.",
                  exclusions=["Fig1 schematic: export approved SVG only",
                              "SI16 uses matching-window TMA; different-window HT-XRD remains in the source CSV",
                              "Main phonon panel: first eight ticks Γ M K Γ A L H A; all segments retained in SI13",
                              "0.01–1% tail zooms are labelled; complete tails retained in SI3/SI5",
                              "Relative phonon metric: signed reference_frequency_THz > 0.5; all modes retained in dispersion"],
                  caveats=["Verification plots are data-derived, not layout-identical manuscript panels.",
                           "t-SNE coordinates are archived; embedding is not recomputed.",
                           "Local-environment drawing and workflow schematic are approved-artwork-only.",
                           "OOM guide endpoints are not latency measurements and are not plotted as data.",
                           "No new inference, MD, DFT, optimization, fitting or bootstrap draws."])
    (output / "plot_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return dict(numerical_figures=len(p.records), approved_schematic="reexport Fig1.svg", manifest="plot_manifest.json")
