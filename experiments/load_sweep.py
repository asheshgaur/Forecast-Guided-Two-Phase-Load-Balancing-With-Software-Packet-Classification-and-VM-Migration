"""Paired offered-load sweep with frozen controller/forecast settings.

Run from the repository root:
  Ensemble-RL/.venv/bin/python experiments/load_sweep.py --workers 2

The full trace and its previously saved LSTM forecasts are multiplied by alpha.
This keeps relative forecast error fixed. Because the LSTM uses training-only
per-flow mean normalization, uniformly scaling both training and test traffic
leaves its normalized inputs unchanged. No model or policy is selected from
these results. Legacy experiment outputs and the bibliography are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gurobipy as gp
import numpy as np
from dctwin.classifier import ClassifierModel
from dctwin.classifier_oracle import load_calibration
from dctwin.topology import FatTree
from dctwin.two_phase import PilotWorkload, Plan, TwoPhaseController, default_path

OUT = ROOT / "results/load_sweep"
BASE = ROOT / "results/revised"
FACTORS = (.25, .5, .75, 1., 1.25, 1.5, 2., 3.)
SEEDS = (7, 11, 19, 23, 31)
MODES = ("ecmp", "migration_only", "rules_only", "two_phase")
EPOCHS = 96


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_forecasts(seed, work):
    with np.load(BASE / f"forecast_mixed_{seed}.npz") as saved:
        demand, predicted = saved["demand"].copy(), saved["predicted"].copy()
    if not np.array_equal(demand, work.demand):
        raise ValueError("Saved trace no longer matches the paired workload generator")
    if not np.isfinite(predicted[320:]).all():
        raise ValueError("Missing held-out LSTM predictions")
    return demand, predicted


def run_seed(seed):
    topo = FatTree(4)
    work = PilotWorkload(topo, "mixed", seed, n_steps=320+EPOCHS)
    demand, predicted = load_forecasts(seed, work)
    clf = load_calibration(ROOT / "results/oracle_calibration.json",
                           ClassifierModel(backend="native", mem_budget_bytes=8192.))
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    summaries, epochs, timings = [], [], []
    try:
        for factor in FACTORS:
            offered, forecasts = factor*demand, factor*predicted
            for mode in MODES:
                c = TwoPhaseController(topo, work, clf, env=env)
                placement = dict(work.placement)
                routing = {f: default_path(topo, placement[a], placement[b])
                           for f, (a,b) in enumerate(work.flows)}
                rules = set()
                episode = []
                for step in range(EPOCHS):
                    truth, forecast = offered[320+step], forecasts[320+step]
                    start = perf_counter()
                    if mode == "ecmp":
                        plan = Plan(placement, routing, rules, [], 0,0,0,2,0,0,0,0,0)
                        solves = []
                    else:
                        plan, solves = c.act(placement, routing, rules, forecast, mode,
                                             coarse=(step % 8 == 0))
                    elapsed = perf_counter()-start
                    measured = c.evaluate(plan, routing, truth)
                    assert c.feasible(plan.placement)
                    for f,(a,b) in enumerate(work.flows):
                        assert plan.routing[f][0] == plan.placement[a]
                        assert plan.routing[f][-1] == plan.placement[b]
                    for switch in {r[0] for r in rules | plan.rules}:
                        assert sum(c.footprint(r) for r in rules | plan.rules
                                   if r[0] == switch) <= clf.mem_budget_bytes+1e-6
                    # Identical identifiers make every sample auditable and pairable.
                    row = dict(seed=seed, regime="mixed", load_factor=factor,
                               controller=mode, step=step, **measured,
                               migrations=len(plan.migrations), rule_changes=len(rules ^ plan.rules),
                               resident_rules=len(plan.rules), phase2_calls=int(len(solves)==2),
                               wall_seconds=elapsed)
                    episode.append(row)
                    for phase,sol in enumerate(solves, 1):
                        timings.append(dict(seed=seed, load_factor=factor, controller=mode,
                                            step=step, phase=phase, status=sol.status,
                                            gap=sol.gap, build_seconds=sol.build_seconds,
                                            solve_seconds=sol.runtime, variables=sol.variables,
                                            constraints=sol.constraints, nodes=sol.nodes))
                    placement, routing, rules = plan.placement, plan.routing, plan.rules
                epochs.extend(episode)
                summary = dict(seed=seed, regime="mixed", load_factor=factor,
                               controller=mode, epochs=EPOCHS,
                               offered_mean_gbps=float(offered[320:].sum(axis=1).mean()/1000),
                               forecast_mae_mbps=float(np.abs(offered[320:]-forecasts[320:]).mean()))
                for key in ("excess_minutes","access","fabric","migrations","migration_gb",
                            "rule_changes","phase2_calls"):
                    summary[key] = sum(r[key] for r in episode)
                summaries.append(summary)
            print(f"Completed seed={seed}, load={factor:g}x, all four controllers", flush=True)
    finally:
        env.dispose()
    return summaries, epochs, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    # Refuse silently mixing a changed controller with the previously published run.
    legacy = json.loads((BASE / "manifest.json").read_text())
    for name, digest in legacy["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f"Base experiment source changed: {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ["experiments/load_sweep.py", "dctwin/two_phase.py",
               "dctwin/topology.py", "dctwin/classifier.py", "dctwin/classifier_oracle.py",
               "results/oracle_calibration.json", "results/revised/manifest.json"]
    sources += [f"results/revised/forecast_mixed_{s}.npz" for s in SEEDS]
    manifest = dict(status="running", factors=FACTORS, seeds=SEEDS, controllers=MODES,
                    epochs=EPOCHS, workers=args.workers, python=sys.version,
                    platform=platform.platform(), gurobi=gp.gurobi.version(), numpy=np.__version__,
                    sources={name:sha(ROOT/name) for name in sources},
                    method="Uniformly scale full saved trace and LSTM forecast; fixed capacity, policy and relative forecast error. No clipping after scaling.")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    groups = [[], [], []]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(run_seed, SEEDS):
            for acc, rows in zip(groups, result):
                acc.extend(rows)
            for name, rows in zip(("summary.csv", "epochs.csv", "timings.csv"), groups):
                write_csv(name, rows)
    manifest["status"] = "complete"
    manifest["outputs"] = {name:sha(OUT/name) for name in ("summary.csv", "epochs.csv", "timings.csv")}
    manifest["runs"] = len(groups[0])
    manifest["epoch_records"] = len(groups[1])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(groups[0])} runs and {len(groups[1])} epoch records to {OUT}")


if __name__ == "__main__":
    main()
