"""Reproducible diagnostic experiments; never overwrites legacy result CSVs.

python experiments/revised_evaluation.py --seeds 7 11 19 23 31
python experiments/revised_evaluation.py --scaling
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gurobipy as gp
import numpy as np
from dctwin.classifier import ClassifierModel
from dctwin.classifier_oracle import load_calibration
from dctwin.topology import FatTree
from dctwin.two_phase import (PilotWorkload, TwoPhaseController, Plan,
                            default_path, lstm_predictions)

OUT = ROOT / "results/revised"
MODES = ("ecmp", "migration_only", "rules_only", "two_phase", "persistence", "perfect")
REGIMES = ("fabric", "access", "mixed", "unpredictable")


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(seed, regime, epochs, env, budget=8192.):
    topo = FatTree(4)
    work = PilotWorkload(topo, regime, seed, n_steps=320+epochs)
    clf = load_calibration(ROOT / "results/oracle_calibration.json",
                           ClassifierModel(backend="native", mem_budget_bytes=budget))
    predicted, meta = lstm_predictions(work.demand, seed=seed)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"forecast_{regime}_{seed}.npz", demand=work.demand,
                        predicted=predicted)
    forecast_rows = []
    truth = work.demand[320:]
    for method, estimate in (("lstm", predicted[320:]),
                              ("persistence", work.demand[319:-1])):
        forecast_rows.append(dict(seed=seed, regime=regime, method=method,
                                  mae_mbps=float(np.mean(np.abs(truth-estimate))),
                                  rmse_mbps=float(np.sqrt(np.mean((truth-estimate)**2))),
                                  **meta))
    summaries, records, timings = [], [], []
    for mode in MODES:
        controller = TwoPhaseController(topo, work, clf, env=env)
        placement = dict(work.placement)
        routing = {f: default_path(topo, placement[a], placement[b])
                   for f, (a, b) in enumerate(work.flows)}
        rules = set()
        epoch_rows = []
        for step in range(epochs):
            t = 320 + step
            true = work.demand[t]
            forecast = (true if mode == "perfect" else work.demand[t-1]
                        if mode == "persistence" else predicted[t])
            start = perf_counter()
            if mode == "ecmp":
                plan = Plan(placement, routing, rules, [], 0, 0, 0, 2, 0, 0, 0, 0, 0)
                solves = []
            else:
                plan, solves = controller.act(placement, routing, rules, forecast,
                    "two_phase" if mode in ("persistence", "perfect") else mode,
                    coarse=(step % 8 == 0))
            wall_seconds = perf_counter() - start
            measured = controller.evaluate(plan, routing, true)
            changes = len(rules ^ plan.rules)
            assert controller.feasible(plan.placement)
            for f, (a, b) in enumerate(work.flows):
                assert plan.routing[f][0] == plan.placement[a]
                assert plan.routing[f][-1] == plan.placement[b]
            for switch in {r[0] for r in rules | plan.rules}:
                assert sum(controller.footprint(r) for r in rules | plan.rules
                           if r[0] == switch) <= budget + 1e-6
            row = dict(seed=seed, regime=regime, controller=mode, step=step,
                       **measured, migrations=len(plan.migrations), rule_changes=changes,
                       resident_rules=len(plan.rules), wall_seconds=wall_seconds,
                       solves=len(solves))
            epoch_rows.append(row)
            for phase, sol in enumerate(solves, 1):
                timings.append(dict(seed=seed, regime=regime, controller=mode, step=step,
                                    phase=phase, solve_seconds=sol.runtime,
                                    build_seconds=sol.build_seconds, variables=sol.variables,
                                    constraints=sol.constraints, gap=sol.gap,
                                    status=sol.status, nodes=sol.nodes))
            placement, routing, rules = plan.placement, plan.routing, plan.rules
        records.extend(epoch_rows)
        summary = dict(seed=seed, regime=regime, controller=mode, epochs=epochs,
                       budget_bytes=budget)
        for key in ("excess_minutes", "access", "fabric", "migration_gb", "migrations", "rule_changes"):
            summary[key] = sum(r[key] for r in epoch_rows)
        summary["p95_control_ms"] = float(np.percentile([r["wall_seconds"]*1000 for r in epoch_rows], 95))
        summaries.append(summary)
        print(f"{regime:14} seed={seed:2} {mode:16} excess={summary['excess_minutes']:8.2f} "
              f"migrations={summary['migrations']:2} changes={summary['rule_changes']:3}", flush=True)
    return summaries, records, forecast_rows, timings


def scaling(env):
    rows = []
    for nflows in (4, 8, 16, 32):
        for seed in (7, 11, 19, 23, 31):
            topo = FatTree(4)
            work = PilotWorkload(topo, "mixed", seed)
            base = list(work.flows[:4])
            work.flows = [base[i % len(base)] for i in range(nflows)]
            work.granularity = ["exact"] * nflows
            # Fixed total offered traffic isolates model size, not load severity.
            demand = np.full(nflows, 24000. / nflows)
            clf = load_calibration(ROOT / "results/oracle_calibration.json", ClassifierModel(backend="native"))
            controller = TwoPhaseController(topo, work, clf, env=env)
            routing = {f: default_path(topo, work.placement[a], work.placement[b])
                       for f, (a, b) in enumerate(work.flows)}
            # Give repetitions distinct deterministic solver inputs.
            demand *= np.random.default_rng(seed).uniform(.95, 1.05, nflows)
            for joint in (False, True):
                sol = controller.solve(work.placement, routing, set(), demand,
                                       allow_migration=joint)
                rows.append(dict(flows=nflows, seed=seed, phase=2 if joint else 1,
                                 build_seconds=sol.build_seconds, solve_seconds=sol.runtime,
                                 variables=sol.variables, constraints=sol.constraints,
                                 status=sol.status, gap=sol.gap, nodes=sol.nodes))
    write_csv("scaling.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 31])
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=REGIMES)
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--scaling", action="store_true")
    args = parser.parse_args()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    if args.scaling:
        scaling(env)
        return
    groups = [[], [], [], []]
    for regime in args.regimes:
        for seed in args.seeds:
            result = evaluate(seed, regime, args.epochs, env)
            for acc, rows in zip(groups, result):
                acc.extend(rows)
            # Save every completed paired block, so interruptions lose no blocks.
            for name, rows in zip(("summary.csv", "epochs.csv", "forecast.csv", "timings.csv"), groups):
                write_csv(name, rows)
    source_files = ["dctwin/two_phase.py", "experiments/revised_evaluation.py", "results/oracle_calibration.json"]
    manifest = dict(arguments=vars(args), python=sys.version, platform=platform.platform(),
                    gurobi=gp.gurobi.version(), numpy=np.__version__,
                    sources={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},
                    notes="Synthetic diagnostic pilot; classifier coefficients reused from existing calibration; no live classifier deployment.")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env.dispose()


if __name__ == "__main__":
    main()
