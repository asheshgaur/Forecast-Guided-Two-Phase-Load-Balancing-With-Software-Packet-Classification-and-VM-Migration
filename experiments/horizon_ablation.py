"""Paired future-horizon and past-window studies, with separate output/provenance.

Run: Ensemble-RL/.venv/bin/python experiments/horizon_ablation.py --workers 2
Future horizon: L=12, H in {1,2,4,8,16}, frozen recursive LSTM and oracle.
Past window: H=1, L in {4,8,12,24,48}, identical training targets 48..255.
Both: mixed traffic, alpha in {1,2,3}, five seeds, original 96 test epochs.
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
sys.path.insert(0,str(ROOT))
import gurobipy as gp
import numpy as np
import torch
from dctwin.classifier import ClassifierModel
from dctwin.classifier_oracle import load_calibration
from dctwin.topology import FatTree
from dctwin.two_phase import PilotWorkload, default_path
from dctwin.horizon_study import (FixedPlanHorizonController, train_forecaster,
                                  rollout_cache)

SEEDS = (7,11,19,23,31)
HORIZONS = (1,2,4,8,16)
LOOKBACKS = (4,8,12,24,48)
FACTORS = (1.,2.,3.)
MODES = ("migration_only","two_phase")
EPOCHS = 96


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_forecasts(seed, work, out, lookback, first_target, max_horizon, study):
    cache = out/"forecasts"/f"{study}_{seed}_L{lookback}.npz"
    checkpoint = cache.with_suffix(".pt")
    metadata = cache.with_suffix(".json")
    fingerprint = sha(ROOT/"dctwin/horizon_study.py")
    if cache.exists():
        meta = json.loads(metadata.read_text())
        assert meta["source_sha256"] == fingerprint
        assert meta["first_target"] == first_target and meta["horizon"] == max_horizon
        assert meta["cache_sha256"] == sha(cache) and meta["checkpoint_sha256"] == sha(checkpoint)
        with np.load(cache) as stored:
            assert np.array_equal(stored["demand"],work.demand)
            return stored["forecasts"].copy(),meta
    net, scale, one_step, meta = train_forecaster(work.demand,seed,lookback,first_target)
    if study == "horizon":
        with np.load(ROOT/f"results/revised/forecast_mixed_{seed}.npz") as prior:
            assert np.array_equal(prior["demand"],work.demand)
            discrepancy = float(np.max(np.abs(one_step[320:]-prior["predicted"][320:])))
            assert discrepancy < 1e-6, ("Original LSTM not reproduced", seed, discrepancy)
            meta["original_one_step_max_difference"] = discrepancy
    forecasts = rollout_cache(net,scale,work.demand,one_step,lookback,max_horizon)
    np.savez_compressed(cache,demand=work.demand,forecasts=forecasts,scale=scale)
    torch.save(dict(state_dict=net.state_dict(),scale=scale.tolist(),metadata=meta),checkpoint)
    meta.update(horizon=max_horizon,source_sha256=fingerprint,cache_sha256=sha(cache),
                checkpoint_sha256=sha(checkpoint))
    metadata.write_text(json.dumps(meta,indent=2))
    print(f"Trained {study} seed={seed} L={lookback}: checkpoint {meta['best_epoch']}",flush=True)
    return forecasts,meta


def forecast_errors(seed, study, lookback, forecasts, truth):
    rows = []
    # Same forecast origins at every lead avoid changing the evaluated time range.
    common_origins = len(truth)-forecasts.shape[1]+1
    for lead in range(forecasts.shape[1]):
        actual = truth[lead:lead+common_origins]
        error = actual-forecasts[:common_origins,lead]
        rows.append(dict(seed=seed,study=study,lookback=lookback,lead=lead+1,
                         origins=common_origins,mae_mbps=float(np.abs(error).mean()),
                         rmse_mbps=float(np.sqrt((error**2).mean()))))
    return rows


def run_seed(seed, config):
    out = Path(config["output"])
    block = out/f"block_{seed}.json"
    if block.exists():
        prior = json.loads(block.read_text())
        if prior["complete"]:
            return prior
    topo = FatTree(4)
    work = PilotWorkload(topo,"mixed",seed,n_steps=320+EPOCHS)
    clf = load_calibration(ROOT/"results/oracle_calibration.json",
                          ClassifierModel(backend="native",mem_budget_bytes=8192.))
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    result = dict(complete=False,summary=[],epochs=[],timings=[],forecast=[],training=[])
    try:
        for study in config["studies"]:
            lookbacks = [12] if study == "horizon" else config["lookbacks"]
            horizons = config["horizons"] if study == "horizon" else [1]
            types = config["forecast_types"] if study == "horizon" else ["lstm"]
            for lookback in lookbacks:
                forecasts,meta = make_forecasts(seed,work,out,lookback,
                    12 if study == "horizon" else 48,max(horizons),study)
                result["training"].append(dict(study=study,**meta))
                result["forecast"].extend(forecast_errors(seed,study,lookback,forecasts,work.demand[320:]))
                for horizon in horizons:
                    for kind in types:
                        for alpha in config["factors"]:
                            for mode in MODES:
                                controller = FixedPlanHorizonController(topo,work,clf,env=env)
                                placement = dict(work.placement)
                                routing = {f:default_path(topo,placement[a],placement[b])
                                           for f,(a,b) in enumerate(work.flows)}
                                rules = set()
                                identity = dict(seed=seed,study=study,lookback=lookback,horizon=horizon,
                                                forecast_type=kind,load_factor=alpha,controller=mode)
                                episode = []
                                for step in range(EPOCHS):
                                    length = min(horizon,EPOCHS-step)
                                    forecast = alpha*(forecasts[step,:length] if kind == "lstm"
                                                      else work.demand[320+step:320+step+length])
                                    start = perf_counter()
                                    plan,solves = controller.act(placement,routing,rules,forecast,mode,
                                                                  coarse=(step%8 == 0))
                                    wall = perf_counter()-start
                                    measured = controller.evaluate(plan,routing,alpha*work.demand[320+step])
                                    assert controller.feasible(plan.placement)
                                    for f,(a,b) in enumerate(work.flows):
                                        assert plan.routing[f][0] == plan.placement[a]
                                        assert plan.routing[f][-1] == plan.placement[b]
                                    for switch in {r[0] for r in rules | plan.rules}:
                                        assert sum(controller.footprint(r) for r in rules | plan.rules
                                                   if r[0] == switch) <= clf.mem_budget_bytes+1e-6
                                    # Save full actions so later explanations can inspect placement paths.
                                    row = dict(**identity,step=step,effective_horizon=length,**measured,
                                               migrations=len(plan.migrations),rule_changes=len(rules ^ plan.rules),
                                               resident_rules=len(plan.rules),phase2_calls=int(len(solves)==2),
                                               predicted_objective=plan.objective,wall_seconds=wall,
                                               placement=json.dumps(plan.placement,sort_keys=True),
                                               routing=json.dumps(plan.routing,sort_keys=True),
                                               moved_vms=json.dumps([vm for vm,_ in plan.migrations]))
                                    episode.append(row)
                                    for phase,sol in enumerate(solves,1):
                                        result["timings"].append(dict(**identity,step=step,phase=phase,
                                            status=sol.status,gap=sol.gap,build_seconds=sol.build_seconds,
                                            solve_seconds=sol.runtime,variables=sol.variables,
                                            constraints=sol.constraints,nodes=sol.nodes))
                                    placement,routing,rules = plan.placement,plan.routing,plan.rules
                                result["epochs"].extend(episode)
                                summary = dict(**identity,epochs=EPOCHS)
                                for field in ("excess_minutes","access","fabric","migrations","migration_gb",
                                              "rule_changes","phase2_calls"):
                                    summary[field] = sum(r[field] for r in episode)
                                summary["p95_control_ms"] = float(np.percentile([r["wall_seconds"]*1000 for r in episode],95))
                                result["summary"].append(summary)
                            print(f"Done seed={seed} {study} L={lookback} H={horizon} {kind} alpha={alpha:g}",flush=True)
                    block.write_text(json.dumps(result))
        result["complete"] = True
        block.write_text(json.dumps(result))
        return result
    finally:
        env.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    parser.add_argument("--studies",nargs="+",choices=("horizon","history"),default=["horizon","history"])
    parser.add_argument("--horizons",type=int,nargs="+",default=list(HORIZONS))
    parser.add_argument("--lookbacks",type=int,nargs="+",default=list(LOOKBACKS))
    parser.add_argument("--factors",type=float,nargs="+",default=list(FACTORS))
    parser.add_argument("--forecast-types",nargs="+",choices=("lstm","oracle"),default=["lstm","oracle"])
    parser.add_argument("--output",type=Path,default=ROOT/"results/horizon_ablation")
    args = parser.parse_args()
    assert args.workers >= 1 and all(1 <= h <= EPOCHS for h in args.horizons)
    assert all(1 <= l <= 48 for l in args.lookbacks)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"forecasts").mkdir(exist_ok=True)
    config = dict(vars(args),output=str(args.output.resolve()))
    old = json.loads((ROOT/"results/revised/manifest.json").read_text())
    for name,digest in old["sources"].items():
        assert sha(ROOT/name) == digest, f"Original experiment changed: {name}"
    names = ["dctwin/horizon_study.py","experiments/horizon_ablation.py",
             "dctwin/two_phase.py","dctwin/topology.py","dctwin/classifier.py",
             "dctwin/classifier_oracle.py","results/oracle_calibration.json",
             "results/load_sweep/summary.csv","results/revised/manifest.json"]
    names += [f"results/revised/forecast_mixed_{s}.npz" for s in args.seeds]
    sources = {n:sha(ROOT/n) for n in names}
    manifest_path = args.output/"manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous["config"] == config and previous["sources"] == sources, "Use a new output directory after changing study inputs"
    manifest = dict(status="running",config=config,sources=sources,epochs=EPOCHS,
                    controllers=MODES,python=sys.version,platform=platform.platform(),
                    gurobi=gp.gurobi.version(),torch=torch.__version__,numpy=np.__version__,
                    protocol="Preselected grid. Separate factors. Horizon: L=12, original LSTM, recursive forecasts; oracle is noncausal diagnostic only. History: H=1, training targets 48..255 for every L. Validate on 256..319; test 320..415. Fixed current action scored across H; no future-action sequence. Replan every epoch. Truncate H at trace end. Same initial placement, memory, candidates, costs, and migration eligibility. No tuning on test outcomes.")
    manifest_path.write_text(json.dumps(manifest,indent=2))
    groups = {k:[] for k in ("summary","epochs","timings","forecast","training")}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_seed,s,config) for s in args.seeds]
        for future in futures:
            result = future.result()
            for name,rows in groups.items():
                rows.extend(result[name])
                write_csv(args.output/f"{name}.csv",rows)
    manifest.update(status="complete",runs=len(groups["summary"]),epoch_records=len(groups["epochs"]))
    manifest["outputs"] = {p.name:sha(p) for p in args.output.glob("*.csv")}
    manifest["forecast_files"] = {str(p.relative_to(args.output)):sha(p)
                                  for p in sorted((args.output/"forecasts").glob("*"))}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(f"COMPLETE: {manifest['runs']} runs; {manifest['epoch_records']} epochs",flush=True)


if __name__ == "__main__":
    main()
