"""Run the predeclared fixed-network VM-density study without editing old runs."""
from __future__ import annotations

import argparse
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
import torch

from dctwin.classifier import ClassifierModel
from dctwin.classifier_oracle import load_calibration
from dctwin.topology import FatTree
from dctwin.two_phase import Plan, TwoPhaseController, default_path, lstm_predictions
from dctwin.vm_density import DENSITIES, DensityWorkload

SEEDS = (7, 11, 19, 23, 31)
MODES = ("ecmp", "migration_only", "two_phase")
EPOCHS = 96


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def forecast(work, out):
    path = out / "forecasts" / f"n{work.n_vms}_seed{work.seed}.npz"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["demand"], work.demand)
            prediction = saved["predicted"].copy()
        return prediction, json.loads(meta_path.read_text())
    if work.n_vms == 16:
        with np.load(ROOT / f"results/revised/forecast_mixed_{work.seed}.npz") as saved:
            np.testing.assert_array_equal(saved["demand"], work.demand)
            prediction = saved["predicted"].copy()
        with (ROOT / "results/revised/forecast.csv").open() as f:
            row = next(r for r in csv.DictReader(f) if int(r["seed"]) == work.seed
                       and r["regime"] == "mixed" and r["method"] == "lstm")
        meta = {k: float(row[k]) for k in ("training_seconds", "best_epoch", "validation_mse_normalized")}
        fit_seconds = 0.
    else:
        start = perf_counter()
        prediction, meta = lstm_predictions(work.demand, seed=work.seed)
        fit_seconds = perf_counter() - start
    assert np.isfinite(prediction[320:]).all()
    info = dict(vms=work.n_vms, seed=work.seed, classes=len(work.flows),
                reused_baseline=int(work.n_vms == 16), fit_seconds_current_run=fit_seconds,
                training_seconds=meta["training_seconds"], best_epoch=meta["best_epoch"],
                validation_mse_normalized=meta["validation_mse_normalized"],
                forecast_mae_mbps=float(np.abs(work.demand[320:] - prediction[320:]).mean()),
                training_targets=244 * len(work.flows), lookback=12, horizon=1)
    np.savez_compressed(path, demand=work.demand, predicted=prediction)
    info["cache_sha256"] = sha(path)
    meta_path.write_text(json.dumps(info, indent=2) + "\n")
    return prediction, info


def run_case(n, seed, out, env):
    block = out / f"case_n{n}_seed{seed}.json"
    if block.exists():
        cached = json.loads(block.read_text())
        assert cached["complete"]
        return cached
    topo = FatTree(4)
    work = DensityWorkload(topo, n, seed)
    predicted, training = forecast(work, out)
    clf = load_calibration(ROOT / "results/oracle_calibration.json",
                           ClassifierModel(backend="native", mem_budget_bytes=8192.))
    result = dict(complete=False, summary=[], epochs=[], timings=[], training=[training])
    identity = dict(vms=n, seed=seed, classes=len(work.flows), hosts=16)
    for mode in MODES:
        c = TwoPhaseController(topo, work, clf, env=env)
        placement = dict(work.placement)
        routing = {f: default_path(topo, placement[a], placement[b])
                   for f, (a, b) in enumerate(work.flows)}
        rules = set()
        episode, timing_rows = [], []
        for step in range(EPOCHS):
            t = 320 + step
            truth, prediction = work.demand[t], predicted[t]
            start = perf_counter()
            failure = ""
            if mode == "ecmp":
                plan = Plan(placement, routing, rules, [], 0, 0, 0, 2, 0, 0, 0, 0, 0)
                solves = []
            else:
                try:
                    plan, solves = c.act(placement, routing, rules, prediction,
                                         mode, coarse=(step % 8 == 0))
                except RuntimeError as exc:
                    if not str(exc).startswith("No feasible plan: Gurobi status"):
                        raise
                    failure = str(exc)
                    plan = Plan(placement, routing, rules, [], 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    solves = []
            elapsed = perf_counter() - start
            measured = c.evaluate(plan, routing, truth)
            assert c.feasible(plan.placement)
            for f, (a, b) in enumerate(work.flows):
                p = plan.routing[f]
                assert p[0] == plan.placement[a] and p[-1] == plan.placement[b]
                assert all(topo.graph.has_edge(u, v) for u, v in zip(p[:-1], p[1:]))
            for switch in {r[0] for r in rules | plan.rules}:
                assert sum(c.footprint(r) for r in rules | plan.rules if r[0] == switch) <= 8192 + 1e-6
            row = dict(**identity, controller=mode, step=step, **measured,
                       migrations=len(plan.migrations), rule_changes=len(rules ^ plan.rules),
                       resident_rules=len(plan.rules), phase2_calls=int(len(solves) == 2),
                       wall_seconds=elapsed, fallback=int(bool(failure)), failure=failure,
                       placement=json.dumps(plan.placement, sort_keys=True),
                       routing=json.dumps(plan.routing, sort_keys=True),
                       moved_vms=json.dumps([vm for vm, _ in plan.migrations]))
            episode.append(row)
            for phase, sol in enumerate(solves, 1):
                timing_rows.append(dict(**identity, controller=mode, step=step, phase=phase,
                    status=sol.status, gap=sol.gap, build_seconds=sol.build_seconds,
                    solve_seconds=sol.runtime, variables=sol.variables,
                    constraints=sol.constraints, nodes=sol.nodes))
            placement, routing, rules = plan.placement, plan.routing, plan.rules
        summary = dict(**identity, controller=mode, epochs=EPOCHS,
                       offered_mean_gbps=float(work.demand[320:].sum(axis=1).mean()/1000),
                       forecast_mae_mbps=training["forecast_mae_mbps"])
        for field in ("excess_minutes", "access", "fabric", "migrations", "migration_gb",
                      "rule_changes", "phase2_calls", "fallback"):
            summary[field] = sum(r[field] for r in episode)
        times = [r["wall_seconds"] * 1000 for r in episode]
        summary.update(p95_control_ms=float(np.percentile(times, 95)),
                       max_control_ms=max(times),
                       time_limit_solves=sum(r["status"] == 9 for r in timing_rows),
                       max_solver_gap=max((r["gap"] for r in timing_rows), default=0.),
                       migrations_per_100_vms=100*summary["migrations"]/n)
        result["summary"].append(summary)
        result["epochs"].extend(episode)
        result["timings"].extend(timing_rows)
        print(f"N={n:2} seed={seed:2} {mode:14} excess={summary['excess_minutes']:9.2f} "
              f"moves={summary['migrations']:2} p95={summary['p95_control_ms']:8.1f}ms "
              f"timeouts={summary['time_limit_solves']} fallback={summary['fallback']}", flush=True)
    result["complete"] = True
    block.write_text(json.dumps(result) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vms", type=int, nargs="+", choices=DENSITIES, default=list(DENSITIES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--output", type=Path, default=ROOT / "results/vm_density")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "forecasts").mkdir(exist_ok=True)
    legacy = json.loads((ROOT / "results/revised/manifest.json").read_text())
    for name, digest in legacy["sources"].items():
        assert sha(ROOT / name) == digest, f"Original experiment changed: {name}"
    names = list(legacy["sources"]) + ["dctwin/topology.py", "dctwin/classifier.py",
        "dctwin/classifier_oracle.py", "dctwin/vm_density.py", "experiments/vm_density.py",
        "experiments/vm_density_protocol.txt", "results/revised/summary.csv", "results/revised/forecast.csv"]
    names += [f"results/revised/forecast_mixed_{s}.npz" for s in args.seeds]
    sources = {name: sha(ROOT / name) for name in names}
    config = dict(vms=args.vms, seeds=args.seeds, modes=MODES, epochs=EPOCHS,
                  workers=1, time_limit=10., horizon=1, lookback=12)
    # JSON canonicalization avoids list/tuple differences on resume.
    config = json.loads(json.dumps(config))
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert old["config"] == config and old["sources"] == sources, "Use a new output directory after changing inputs"
    manifest = dict(status="running", config=config, sources=sources,
        python=sys.version, platform=platform.platform(), gurobi=gp.gurobi.version(),
        numpy=np.__version__, torch=torch.__version__,
        protocol="experiments/vm_density_protocol.txt; unchanged controller, one worker, nested active-VM populations")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    groups = {k: [] for k in ("summary", "epochs", "timings", "training")}
    try:
        for n in args.vms:
            for seed in args.seeds:
                result = run_case(n, seed, out, env)
                for k in groups:
                    groups[k].extend(result[k])
                write_csv(out / "summary.csv", groups["summary"])
                print(f"Completed {len(groups['summary'])} / {3*len(args.vms)*len(args.seeds)} episodes", flush=True)
    finally:
        env.dispose()
    for name, rows in groups.items():
        write_csv(out / f"{name}.csv", rows)
    manifest.update(status="complete", runs=len(groups["summary"]), epoch_records=len(groups["epochs"]),
                    outputs={p.name: sha(p) for p in sorted(out.glob("*.csv"))},
                    forecasts={str(p.relative_to(out)): sha(p) for p in sorted((out / "forecasts").glob("*"))})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMPLETE: {manifest['runs']} episodes; {manifest['epoch_records']} epoch records", flush=True)


if __name__ == "__main__":
    main()

