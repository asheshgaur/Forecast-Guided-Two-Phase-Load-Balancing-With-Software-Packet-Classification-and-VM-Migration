"""Isolated forecast/history ablations; the published one-epoch pilot is unchanged.

The horizon controller scores one current placement/path plan over H forecasts.
It executes only the first epoch and replans. It does NOT optimize a sequence of
future migrations or future routing changes. Candidate VMs still use the first
forecast, preserving the original bounded migration neighborhood.
"""
from __future__ import annotations

from time import perf_counter
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from torch import nn

from .two_phase import TwoPhaseController, Plan, arcs, path_rules, default_path


class ForecastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(1, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, values):
        hidden, _ = self.rnn(values)
        return self.head(hidden[:, -1]).squeeze(-1)


def train_forecaster(data, seed, lookback=12, first_target=None,
                     train_end=256, test_start=320):
    """Same architecture/training as the pilot; optional common training targets."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    first_target = lookback if first_target is None else first_target
    if not lookback <= first_target < train_end < test_start <= len(data):
        raise ValueError("Invalid chronological split or history window")
    scale = np.maximum(np.mean(data[:train_end], axis=0), 1.)
    z = data / scale
    xs, ys, times = [], [], []
    for t in range(first_target, len(data)):
        for f in range(data.shape[1]):
            xs.append(z[t-lookback:t, f, None])
            ys.append(z[t, f])
            times.append(t)
    x = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    times = np.array(times)
    net = ForecastNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=.01)
    train = np.flatnonzero(times < train_end)
    valid = np.flatnonzero((times >= train_end) & (times < test_start))
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch = float("inf"), None, 0
    start = perf_counter()
    for epoch in range(60):
        net.train()
        for idx in np.array_split(rng.permutation(train), max(1, len(train)//256)):
            optimizer.zero_grad()
            loss = ((net(x[idx])-y[idx])**2).mean()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val = float(((net(x[valid])-y[valid])**2).mean())
        if val < best:
            best, best_epoch = val, epoch+1
            best_state = {k:v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        predictions = net(x).numpy().reshape(-1, data.shape[1])*scale
    one_step = np.full_like(data, np.nan)
    one_step[first_target:] = np.maximum(predictions, 0.)
    meta = dict(seed=seed, lookback=lookback, first_target=first_target,
                train_end=train_end, test_start=test_start, best_epoch=best_epoch,
                training_examples=len(train), validation_examples=len(valid),
                validation_mse_normalized=best, training_seconds=perf_counter()-start)
    return net, scale, one_step, meta


def recursive_forecast(net, scale, observed_history, horizon):
    """Forecast from a supplied past window only; predictions feed later inputs."""
    if horizon < 1 or observed_history.ndim != 2:
        raise ValueError("Expected positive horizon and time-by-flow history")
    window = torch.tensor((observed_history/scale).T[:, :, None], dtype=torch.float32)
    result = []
    with torch.no_grad():
        for _ in range(horizon):
            prediction = net(window).clamp(min=0)
            result.append(prediction.numpy()*scale)
            window = torch.cat((window[:, 1:], prediction[:, None, None]), dim=1)
    return np.array(result)


def rollout_cache(net, scale, data, one_step, lookback, horizon=16, test_start=320):
    """Batched independent origins; no realized future observations in a rollout.

    Use the exactly replicated full-batch one-step value as lead 1 to avoid
    batch-size roundoff changing tie-breaking against the original experiment.
    """
    n, nf = len(data)-test_start, data.shape[1]
    windows = np.array([(data[t-lookback:t]/scale).T[:, :, None]
                        for t in range(test_start, len(data))])
    window = torch.tensor(windows.reshape(n*nf, lookback, 1), dtype=torch.float32)
    result = np.empty((n, horizon, nf))
    with torch.no_grad():
        for lead in range(horizon):
            if lead == 0:
                result[:, lead] = one_step[test_start:]
                pred = torch.tensor((result[:, lead]/scale).reshape(-1), dtype=torch.float32)
            else:
                pred = net(window).clamp(min=0)
                result[:, lead] = pred.numpy().reshape(n,nf)*scale
            window = torch.cat((window[:, 1:], pred[:, None, None]), dim=1)
    return result


class FixedPlanHorizonController(TwoPhaseController):
    def solve(self, placement, routing, old_rules, demand, allow_rules=True,
              allow_migration=False):
        forecast = np.asarray(demand, dtype=float)
        if forecast.ndim == 1:
            forecast = forecast[None, :]
        if forecast.ndim != 2 or len(forecast) < 1 or not np.isfinite(forecast).all():
            raise ValueError("Expected a finite horizon-by-flow forecast")
        if len(forecast) == 1:
            return super().solve(placement, routing, old_rules, forecast[0],
                                 allow_rules, allow_migration)
        start = perf_counter()
        configs = self.configurations(placement, forecast[0], allow_migration)
        model = gp.Model("fixed_plan_horizon", env=self.env)
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = .001
        choose = model.addVars(len(configs), vtype=GRB.BINARY, name="placement")
        model.addConstr(choose.sum() == 1)
        variants, route_vars, rule_terms = {}, {}, {}
        future_terms = {link:[] for link in self.links}
        objective = gp.LinExpr()
        old_load = self.routing_load(routing, forecast[0])
        for q, (new, vm, transfer) in enumerate(configs):
            tau = self.copy_seconds if vm is not None else 0.
            mbps = self.work.mem[vm]*8000/tau if vm is not None else 0.
            pre = {link:v+(mbps if link in transfer else 0.) for link,v in old_load.items()}
            objective += (tau/60*self.excess(pre)
                          + (self.migration_cost if vm is not None else 0.))*choose[q]
            first_terms = {link:[] for link in self.links}
            for f, (a,b) in enumerate(self.work.flows):
                src, dst = new[a], new[b]
                paths = self.topo.path_set(src,dst) if src != dst else [[src]]
                if not allow_rules:
                    paths = [default_path(self.topo,src,dst)]
                fvars = []
                old_f = {r for r in old_rules if r[1] == f}
                for pi, path in enumerate(paths):
                    key = q,f,pi
                    var = model.addVar(vtype=GRB.BINARY, name=f"path_{q}_{f}_{pi}")
                    route_vars[key] = var
                    rules = path_rules(self.topo,f,path)
                    variants[key] = path,rules
                    fvars.append(var)
                    objective += self.rule_cost*len(rules ^ old_f)*var
                    for r in rules:
                        rule_terms.setdefault(r,[]).append(var)
                    for link in arcs(path):
                        first_terms[link].append(float(forecast[0,f])*var)
                        future_terms[link].append((f,var))
                model.addConstr(gp.quicksum(fvars) == choose[q])
            for link in self.links:
                overload = model.addVar(lb=0., name=f"first_{q}_{link}")
                model.addConstr(overload >= gp.quicksum(first_terms[link])/self.cap-.85*choose[q])
                objective += (self.epoch_seconds-tau)/60*overload
        # Later slots share the selected plan. Only one q is active, so a single
        # overload per link/lead is exact and avoids a redundant q dimension.
        for lead in range(1,len(forecast)):
            for link, terms in future_terms.items():
                if not terms:
                    continue
                overload = model.addVar(lb=0., name=f"future_{lead}_{link}")
                model.addConstr(overload >= gp.quicksum(float(forecast[lead,f])*v
                                                       for f,v in terms)/self.cap-.85)
                objective += self.epoch_seconds/60*overload
        for switch in {r[0] for r in old_rules | set(rule_terms)}:
            resident = sum(self.footprint(r) for r in old_rules if r[0] == switch)
            additional = gp.quicksum(self.footprint(r)*gp.quicksum(vs)
                                     for r,vs in rule_terms.items()
                                     if r[0] == switch and r not in old_rules)
            model.addConstr(resident+additional <= self.clf.mem_budget_bytes)
        model.setObjective(objective,GRB.MINIMIZE)
        model.update()
        build_seconds = perf_counter()-start
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"No feasible horizon plan: {model.Status}")
        q = next(q for q in range(len(configs)) if choose[q].X > .5)
        new,vm,transfer = configs[q]
        chosen = {f:variants[key][0] for key,v in route_vars.items()
                  for _,f,_ in [key] if v.X > .5}
        rules = set().union(*(path_rules(self.topo,f,p) for f,p in chosen.items()))
        # act() escalates if ANY forecast slot has residual congestion.
        residual = max(self.excess(self.routing_load(chosen,d)) for d in forecast)
        result = Plan(new,chosen,rules,[] if vm is None else [(vm,transfer)],
                      model.ObjVal,residual,model.Runtime,model.Status,model.MIPGap,
                      model.NumVars,model.NumConstrs,model.NodeCount,build_seconds)
        model.dispose()
        return result
