"""Reconstruct work and counterfactual teacher time from archived CPU-readable receipts."""
from collections import defaultdict
import gzip
import json

import numpy as np
import pandas as pd

from reproduce import assert_close


def verify_cost_evidence(a, output):
    spec = a.registry["cost_evidence"]
    with gzip.open(a.path(spec["tasks"]), "rt") as stream:
        tasks = json.load(stream)
    if len(tasks) != 75038:
        raise ValueError("Expected 75038 formal task records")
    inventory = defaultdict(lambda: dict(tasks=0, work=0))
    blocks = {}
    for task in tasks:
        kind = task["kind"]
        if kind == "relax":
            work = task["evaluations"]
        elif kind == "widom":
            work = task["counts"]["attempted"]
        elif kind == "gcmc":
            work = task["mc_attempts"] + 5 * task["steps"]
        else:
            work = task.get("actual_steps", task["steps"])
        if work != task["work"]:
            raise ValueError("Individual task work differs from native work counters")
        entry = inventory[task["group"]]
        entry["tasks"] += 1
        entry["work"] += work
        key = (task["system"], task["iter"], task["round"])
        seconds = float(task["stage_seconds"])
        if key in blocks:
            assert_close(blocks[key], seconds, "One recorded time per block")
        blocks[key] = seconds
    if len(blocks) != 148 or len(inventory) != 26:
        raise ValueError("Expected 148 formal sampling blocks and 26 modes")
    modes = a.csv("data/sampling_costs/FORMAL_26_SAMPLING_MODES.csv")
    allocations = a.csv("data/sampling_costs/FORMAL_BLOCK_MODE_COSTS_219.csv")
    if len(allocations) != 219:
        raise ValueError("Expected 219 formal block-mode cost allocations")
    results = []
    for mode in modes.itertuples():
        native = a.json(spec["native_timing_directory"] + "/" + mode.mode + ".json")
        rates = []
        for j in range(3):
            observations = [client[j] for client in native["client_timings"]]
            work = sum(item["work"] if isinstance(item, dict) else item[0] for item in observations)
            seconds = max(item["seconds"] if isinstance(item, dict) else item[1] for item in observations)
            if seconds <= 0:
                raise ValueError("Native timing interval must be positive")
            rates.append(work / seconds)
        rate = float(np.median(rates))
        assert_close(rate, mode.teacher_work_per_second, "Measured teacher aggregate throughput")
        assert_close(native["parallel"], mode.teacher_parallel_clients, "Parallel clients")
        assert_close(inventory[mode.mode]["tasks"], mode.task_count, "Per-mode task count")
        assert_close(inventory[mode.mode]["work"], mode.total_work, "Per-mode actual work")
        own = allocations[allocations.group == mode.mode]
        teacher_seconds = mode.total_work / rate
        assert_close(own.work.sum(), mode.total_work, "Per-mode allocated work")
        assert_close(teacher_seconds, mode.teacher_estimated_seconds, "Teacher cost work/rate")
        assert_close((own.work / own.neo_rate).sum(), teacher_seconds, "Block-mode teacher cost")
        assert_close(own.stage_seconds.sum(), mode.student_seconds, "Archived student mode allocations")
        results.append(dict(mode=mode.mode, work=inventory[mode.mode]["work"], tasks=mode.task_count,
                            parallel_clients=native["parallel"], rate_from_three_intervals=rate,
                            teacher_estimated_seconds=teacher_seconds))
    block_rows = [dict(system=k[0], iteration=k[1], round=k[2], recorded_student_seconds=v)
                  for k, v in blocks.items()]
    block_data = pd.DataFrame(block_rows)
    for system, g in modes.groupby("system"):
        assert_close(block_data.loc[block_data.system == system, "recorded_student_seconds"].sum(),
                     g.student_seconds.sum(), "Student system total from 148 recorded block times")
    pd.DataFrame(results).to_csv(output / "native_timing_cost_reanalysis.csv", index=False)
    block_data.to_csv(output / "recorded_sampling_block_times.csv", index=False)
    return dict(task_work_records=len(tasks), recorded_blocks=len(blocks), measured_modes=len(results),
                block_mode_records=len(allocations),
                rate_definition="median of three summed-client-work / slowest-client-time intervals",
                student_mode_allocation="Archived native-timing weights retained; system totals independently sum recorded block times")
