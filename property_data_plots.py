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


