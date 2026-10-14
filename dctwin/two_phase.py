"""Auditable two-phase pilot, separate from the legacy proxy simulator.

Directed full-duplex links, unsplittable paths tied to candidate placements,
local next-hop rules at source ToR/aggregation, CPU/MEM feasibility, and explicit
pre-copy/cutover intervals. Memory coefficients are a cached classifier model,
not measurements of a deployed switch. No synthetic 'rebuild' counter.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .topology import FatTree


def arcs(path):
    return list(zip(path[:-1], path[1:]))


@dataclass
class PilotWorkload:
    topo: FatTree
    regime: str
    seed: int
    n_steps: int = 416

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.placement = {i: s for i, s in enumerate(self.topo.servers)}
        self.cpu = {i: 0.18 for i in self.placement}
        self.mem = {i: 4.0 for i in self.placement}  # GB, also image size
        # Four disjoint elephants collide on a fabric uplink, but their access
        # links have headroom. This is a diagnostic topology, not a trace fit.
        self.flows = [(0, 8), (1, 9), (2, 10), (3, 11)]
        if self.regime in ("access", "mixed"):
            self.placement[1] = self.placement[0]
            self.placement[3] = self.placement[2]
        # Access regime destinations share a rack with their corresponding
        # sources: no equal-cost fabric alternative exists for these flows.
        if self.regime == "access":
            self.placement[8] = self.topo.servers[1]
            self.placement[9] = self.topo.servers[1]
            self.placement[10] = self.topo.servers[3]
            self.placement[11] = self.topo.servers[3]
        # Disjoint low-rate traffic fills the other racks without introducing
        # a second unlabelled elephant bottleneck.
        self.flows += [(4, 12), (5, 13), (6, 14), (7, 15)]
        nf = len(self.flows)
        t = np.arange(self.n_steps)
        phase = rng.uniform(-0.12, 0.12)
        common = 4300 + 2100 * np.sin(2 * np.pi * t / 48 + phase)
        self.demand = np.empty((self.n_steps, nf))
        for f in range(nf):
            if f < 4:
                series = common * rng.uniform(0.95, 1.05)
                noise = rng.normal(0, 100, self.n_steps)
                if self.regime == "unpredictable":
                    noise += rng.normal(0, 1900, self.n_steps)
                self.demand[:, f] = np.clip(series + noise, 100, 9800)
            else:
                self.demand[:, f] = np.clip(300 + rng.normal(0, 30, self.n_steps), 1, None)
        # Per-flow exact matches preserve packet membership on every path.
        self.granularity = ["exact"] * nf


def default_path(topo, src, dst):
    if src == dst:
        return [src]
    return topo.path_set(src, dst)[0]  # controlled ECMP collision, held for all arms


def path_rules(topo, flow, path):
    """Realizable local actions. A rule never changes an upstream link.

    Baseline hash chooses aggregation 0 and core index 0. Destination-side
    downlinks are unique for a fixed destination; no steering rule is needed.
    """
    rules = set()
    if len(path) >= 5:
        tor, agg = path[1:3]
        if agg.startswith("agg_") and not agg.endswith("_0"):
            rules.add((tor, flow, agg))
        if len(path) == 7:
            core = path[3]
            if not core.endswith("_0"):
                rules.add((agg, flow, core))
    return rules


@dataclass
class Plan:
    placement: dict
    routing: dict
    rules: set
    migrations: list
    objective: float
    post_excess: float
    runtime: float
    status: int
    gap: float
    variables: int
    constraints: int
    nodes: float
    build_seconds: float


class TwoPhaseController:
    def __init__(self, topo, work, clf, epoch_seconds=900., copy_seconds=60.,
                 migration_cost=0.5, rule_cost=0.002, time_limit=10., env=None):
        self.topo, self.work, self.clf = topo, work, clf
        self.epoch_seconds, self.copy_seconds = epoch_seconds, copy_seconds
        self.migration_cost, self.rule_cost = migration_cost, rule_cost
        self.time_limit, self.env = time_limit, env
        self.links = [(a, b) for u, v in topo.graph.edges for a, b in ((u, v), (v, u))]
        self.cap = topo.link_capacity

    def footprint(self, rule):
        return self.clf.footprint(self.work.granularity[rule[1]])

    def routing_load(self, routing, demand):
        loads = dict.fromkeys(self.links, 0.)
        for f, path in routing.items():
            for link in arcs(path):
                loads[link] += float(demand[f])
        return loads

    def excess(self, loads):
        return sum(max(0., v / self.cap - .85) for v in loads.values())

    def feasible(self, placement):
        for server in self.topo.servers:
            vms = [v for v, s in placement.items() if s == server]
            if sum(self.work.cpu[v] for v in vms) > .85 + 1e-9:
                return False
            if sum(self.work.mem[v] for v in vms) > 32 * .85 + 1e-9:
                return False
        return True

    def configurations(self, placement, demand, allow_migration):
        configs = [(dict(placement), None, [])]
        if not allow_migration:
            return configs
        traffic = dict.fromkeys(placement, 0.)
        for f, (a, b) in enumerate(self.work.flows):
            traffic[a] += demand[f]
            traffic[b] += demand[f]
        access_out = dict.fromkeys(self.topo.servers, 0.)
        access_in = dict.fromkeys(self.topo.servers, 0.)
        for f, (a, b) in enumerate(self.work.flows):
            if placement[a] != placement[b]:
                access_out[placement[a]] += demand[f]
                access_in[placement[b]] += demand[f]
        overload = {s: max(0., access_out[s]-.85*self.cap)
                    + max(0., access_in[s]-.85*self.cap) for s in self.topo.servers}
        # Bounded candidate neighborhood, selected only from the available
        # demand estimate. Prioritize the overloaded hosts; otherwise repeatedly
        # choosing the largest VM can starve a smaller, unresolved hotspot.
        ranked = sorted(placement, key=lambda v: (-overload[placement[v]], -traffic[v], v))
        for vm in ranked[:2]:
            for dst in self.topo.servers:
                if dst == placement[vm]:
                    continue
                new = {**placement, vm: dst}
                if not self.feasible(new):
                    continue
                transfer = arcs(default_path(self.topo, placement[vm], dst))
                configs.append((new, vm, transfer))
        return configs

    def solve(self, placement, routing, old_rules, demand, allow_rules=True,
              allow_migration=False):
        start = perf_counter()
        configs = self.configurations(placement, demand, allow_migration)
        model = gp.Model("two_phase_pilot", env=self.env)
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = .001
        choose = model.addVars(len(configs), vtype=GRB.BINARY, name="placement")
        model.addConstr(choose.sum() == 1)
        variants, route_vars, rule_terms = {}, {}, {}
        link_terms = {l: [] for l in self.links}
        objective = gp.LinExpr()
        old_load = self.routing_load(routing, demand)
        for q, (new, vm, transfer) in enumerate(configs):
            tau = self.copy_seconds if vm is not None else 0.
            # GB -> megabits, divided by seconds -> Mbps.
            mbps = self.work.mem[vm] * 8000 / tau if vm is not None else 0.
            pre = {l: v + (mbps if l in transfer else 0.) for l, v in old_load.items()}
            pre_cost = tau / 60 * self.excess(pre)
            objective += (pre_cost + (self.migration_cost if vm is not None else 0.)) * choose[q]
            for f, (a, b) in enumerate(self.work.flows):
                src, dst = new[a], new[b]
                paths = self.topo.path_set(src, dst) if src != dst else [[src]]
                if not allow_rules:
                    paths = [default_path(self.topo, src, dst)]
                fvars = []
                old_f = {r for r in old_rules if r[1] == f}
                for pi, path in enumerate(paths):
                    key = q, f, pi
                    var = model.addVar(vtype=GRB.BINARY, name=f"path_{q}_{f}_{pi}")
                    route_vars[key] = var
                    rules = path_rules(self.topo, f, path)
                    variants[key] = path, rules
                    fvars.append(var)
                    objective += self.rule_cost * len(rules ^ old_f) * var
                    for r in rules:
                        rule_terms.setdefault(r, []).append(var)
                    for link in arcs(path):
                        # post-load is conditional on the selected configuration
                        link_terms[link].append((q, float(demand[f]), var))
                model.addConstr(gp.quicksum(fvars) == choose[q])
            for link in self.links:
                overload = model.addVar(lb=0., name=f"post_{q}_{link}")
                load = gp.quicksum(d * v for cq, d, v in link_terms[link] if cq == q)
                model.addConstr(overload >= load / self.cap - .85 * choose[q])
                objective += (self.epoch_seconds - tau) / 60 * overload
        # Exact union of old/new match-action entries for make-before-break.
        switches = {r[0] for r in old_rules | set(rule_terms)}
        for switch in switches:
            resident = sum(self.footprint(r) for r in old_rules if r[0] == switch)
            additional = gp.quicksum(self.footprint(r) * gp.quicksum(vs)
                                    for r, vs in rule_terms.items()
                                    if r[0] == switch and r not in old_rules)
            model.addConstr(resident + additional <= self.clf.mem_budget_bytes)
        model.setObjective(objective, GRB.MINIMIZE)
        model.update()
        build_seconds = perf_counter() - start
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"No feasible plan: Gurobi status {model.Status}")
        q = next(q for q in range(len(configs)) if choose[q].X > .5)
        new, vm, transfer = configs[q]
        chosen = {f: variants[key][0] for key, v in route_vars.items()
                  for cq, f, _ in [key] if cq == q and v.X > .5}
        rules = set().union(*(path_rules(self.topo, f, p) for f, p in chosen.items()))
        result = Plan(new, chosen, rules, [] if vm is None else [(vm, transfer)],
                      model.ObjVal, self.excess(self.routing_load(chosen, demand)),
                      model.Runtime, model.Status, model.MIPGap, model.NumVars,
                      model.NumConstrs, model.NodeCount, build_seconds)
        model.dispose()
        return result

    def act(self, placement, routing, rules, forecast, mode, coarse=True):
        allow_rules = mode != "migration_only"
        first = self.solve(placement, routing, rules, forecast, allow_rules, False)
        plans = [first]
        # Escalate only for residual network congestion; host feasibility is
        # checked for every candidate (pilot CPU/MEM demands are stationary).
        if mode != "rules_only" and coarse and first.post_excess > 1e-8:
            second = self.solve(placement, routing, rules, forecast, allow_rules, True)
            plans.append(second)
            if second.objective < first.objective - 1e-7:
                return second, plans
        return first, plans

    def evaluate(self, plan, old_routing, true_demand):
        tau = self.copy_seconds if plan.migrations else 0.
        pre = self.routing_load(old_routing, true_demand)
        for vm, links in plan.migrations:
            for link in links:
                pre[link] += self.work.mem[vm] * 8000 / self.copy_seconds
        post = self.routing_load(plan.routing, true_demand)
        out = {}
        for name, predicate in (("access", lambda l: any(n.startswith("srv_") for n in l)),
                                ("fabric", lambda l: all(not n.startswith("srv_") for n in l))):
            out[name] = sum((tau * max(0, pre[l] / self.cap - .85)
                            + (self.epoch_seconds - tau) * max(0, post[l] / self.cap - .85))
                           / 60 for l in self.links if predicate(l))
        out["excess_minutes"] = out["access"] + out["fabric"]
        out["migration_gb"] = sum(self.work.mem[v] for v, _ in plan.migrations)
        return out


def lstm_predictions(data, train_end=256, test_start=320, lookback=12, seed=0):
    """Shared LSTM, frozen after training; validation selects the checkpoint.

    Demand observations are normalized using training-only per-flow scales.
    Test windows end at t-1. The model never sees future test values in training.
    """
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    scale = np.maximum(np.mean(data[:train_end], axis=0), 1.)
    z = data / scale
    xs, ys, times = [], [], []
    for t in range(lookback, len(data)):
        for f in range(data.shape[1]):
            xs.append(z[t-lookback:t, f, None])
            ys.append(z[t, f])
            times.append(t)
    x = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    times = np.array(times)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(1, 16, batch_first=True)
            self.head = nn.Linear(16, 1)
        def forward(self, v):
            h, _ = self.rnn(v)
            return self.head(h[:, -1]).squeeze(-1)

    net = Net()
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
            loss = ((net(x[idx]) - y[idx]) ** 2).mean()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val = float(((net(x[valid]) - y[valid]) ** 2).mean())
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        preds = net(x).numpy().reshape(-1, data.shape[1]) * scale
    out = np.full_like(data, np.nan)
    out[lookback:] = np.maximum(preds, 0.)
    return out, {"training_seconds": perf_counter()-start, "best_epoch": best_epoch,
                 "validation_mse_normalized": best, "train_end": train_end,
                 "test_start": test_start, "lookback": lookback}
