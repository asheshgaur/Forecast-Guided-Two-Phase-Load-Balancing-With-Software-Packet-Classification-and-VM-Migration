"""Joint two-stage stochastic MILP with VM migration as a decision variable.

This is Options A + C + Gurobi together, and it is the model the writeup actually
claims: a single optimization that, at one coarse decision epoch, chooses

  * first-stage (here-and-now, before the demand scenario is revealed):
      - VM migration   m[i,d]   (Option C: migration is a variable, not a policy)
      - ToR steering rules  z1[k]  (early commitment)
  * second-stage (wait-and-see, one independent copy per demand scenario omega):
      - aggregation steering rules  z2[k,omega]  (late-binding recourse)
      - flow routing  f[p,omega]  and overload  O[l,omega]

minimizing  first-stage cost  +  E_omega[ recourse cost + realized congestion ].

Because the first stage is shared across all scenarios while the second stage is
scenario-specific, the solver picks early commitments that are good *in
expectation* -- which is exactly what makes forecast reliability (encoded as the
spread of the scenario set) price the choice between committing early at the ToR
and keeping the option to correct at aggregation. The hand-set reliability
thresholds are no longer imposed; threshold structure, if it exists, is emergent.

Classifier costs are real: a per-switch memory budget over measured rule
footprints (`ClassifierModel.footprint`, bytes, calibrated by `ClassifierOracle`)
and per-shape rebuild cost `kappa_new`.

Solved with Gurobi. Requires `gurobipy` + a license.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAVE_GUROBI = True
except Exception:  # pragma: no cover
    _HAVE_GUROBI = False


Link = Tuple[str, str]


# ------------------------------------------------------------------ instance
@dataclass
class Candidate:
    """A candidate steering rule = (flow, path) with a hosting switch, stage,
    classifier granularity, and the links the path uses."""
    cid: int
    flow: int
    stage: int                 # 1 = ToR (first stage), 2 = aggregation (recourse)
    switch: str                # hosting switch (ToR or agg)
    shape: Tuple               # classifier shape class at that switch
    granularity: str           # 'exact' | 'prefix' | 'range'  -> memory footprint
    links: List[Link]          # links this path traverses
    is_default: bool = False   # the do-nothing path (no rule, no shape)


@dataclass
class MigrationOption:
    """A candidate placement for a migratable VM."""
    vm: int
    dest: str                  # destination server id
    is_current: bool           # staying put
    cost: float                # migration cost (0 if current)
    cpu: float
    mem: float
    # migration traffic this option injects: mbps on these links (empty if current)
    mig_links: List[Link] = field(default_factory=list)
    mig_mbps: float = 0.0


@dataclass
class Instance:
    flows: List[int]
    # per flow: candidate rules (list of Candidate), first one should be default
    flow_candidates: Dict[int, List[Candidate]]
    # per flow, per scenario: demand (mbps)
    demand: Dict[Tuple[int, int], float]        # (flow, omega) -> mbps
    scenarios: List[int]
    scenario_prob: Dict[int, float]
    links: List[Link]
    link_capacity: Dict[Link, float]
    # migration
    migratable: List[int]                        # VM ids that may move
    migration_options: Dict[int, List[MigrationOption]]
    servers: List[str]
    server_cpu_cap: Dict[str, float]
    server_mem_cap: Dict[str, float]
    server_base_cpu: Dict[str, float]            # load from non-migratable VMs
    server_base_mem: Dict[str, float]
    # which flows a migration of vm affects: flow -> vm endpoints
    flow_vms: Dict[int, Tuple[int, int]]
    # background link load per scenario from flows not modeled as candidates
    bg_load: Dict[Tuple[Link, int], float] = field(default_factory=dict)
    # load that migration relocates: (link, vm, dest) -> mbps added to `link`
    # (in every scenario) when `vm` is placed on `dest`. This is how migrating a
    # VM moves its flow's load off the old access link onto the new one.
    mig_flow_load: Dict[Tuple[Link, int, str], float] = field(default_factory=dict)


@dataclass
class JointResult:
    migrations: Dict[int, str]          # vm -> chosen destination
    first_stage_rules: List[int]        # candidate ids installed at ToR
    expected_cost: float
    migration_cost: float
    first_stage_rule_cost: float
    expected_congestion: float
    expected_recourse_cost: float
    status: str
    per_scenario_overload: Dict[int, float] = field(default_factory=dict)


class TwoStageMigrationOptimizer:
    def __init__(self, clf, c_cong: float = 1.0, cost_aware: bool = True,
                 lambda_bw: float = 0.85, mip_gap: float = 0.01, time_limit: float = 60.0,
                 late_penalty: float = 0.35, tor_holding_cost: float = 0.0,
                 verbose: bool = False):
        if not _HAVE_GUROBI:
            raise RuntimeError("gurobipy not available; install it with a license.")
        self.clf = clf
        self.c_cong = c_cong
        self.cost_aware = cost_aware
        self.lambda_bw = lambda_bw
        self.mip_gap = mip_gap
        self.time_limit = time_limit
        # late_penalty: the transient-congestion price of binding late at the
        # aggregation layer. A second-stage (recourse) reroute only takes effect
        # after the traffic has begun to materialize, so a fraction `late_penalty`
        # of the rerouted demand is charged as congestion. First-stage ToR rules,
        # installed before the epoch, avoid this. This is what gives early
        # commitment an edge and produces threshold structure in reliability.
        self.late_penalty = late_penalty
        # tor_holding_cost: opportunity cost of holding a first-stage ToR rule in
        # scarce switch memory whether or not the predicted congestion occurs -- the price
        # of committing on a forecast that may be wrong. This is what makes low
        # reliability favor late binding over early commitment.
        self.tor_holding_cost = tor_holding_cost
        self.verbose = verbose

    def _footprint(self, gran: str) -> float:
        # A classifier-blind controller counts rules (every rule costs the nominal
        # per-rule footprint); a classifier-aware one uses the oracle's measured
        # bytes, so on a value/mask datapath a range match consumes its true
        # expanded footprint. This is what stops the aware controller from
        # installing a rule the software switch's memory budget cannot hold.
        return self.clf.assumed_footprint(gran, self.cost_aware)

    def solve(self, inst: Instance) -> JointResult:
        clf = self.clf
        m = gp.Model("two_stage_migration")
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.MIPGap = self.mip_gap
        m.Params.TimeLimit = self.time_limit

        # ---------------- first-stage variables ----------------
        # migration: m[i,d]
        mig = {}
        for i in inst.migratable:
            opts = inst.migration_options[i]
            for o in opts:
                mig[(i, o.dest)] = m.addVar(vtype=GRB.BINARY, name=f"m_{i}_{o.dest}")
            m.addConstr(gp.quicksum(mig[(i, o.dest)] for o in opts) == 1, f"place_{i}")

        # first-stage ToR rules: z1[c] for stage-1 candidates (excluding default)
        z1, y1 = {}, {}
        stage1_shapes = set()
        for f in inst.flows:
            for c in inst.flow_candidates[f]:
                if c.stage == 1 and not c.is_default:
                    z1[c.cid] = m.addVar(vtype=GRB.BINARY, name=f"z1_{c.cid}")
                    stage1_shapes.add((c.switch, c.shape))
        for (w, g) in stage1_shapes:
            y1[(w, g)] = m.addVar(vtype=GRB.BINARY, name=f"y1_{w}_{g}")

        # ---------------- second-stage variables (per scenario) ----------------
        z2, y2, fvar, Ovar = {}, {}, {}, {}
        stage2_shapes = set()
        for w_ in inst.flows:
            for c in inst.flow_candidates[w_]:
                if c.stage == 2 and not c.is_default:
                    stage2_shapes.add((c.switch, c.shape))
        for om in inst.scenarios:
            for f in inst.flows:
                for c in inst.flow_candidates[f]:
                    # a routing selector per (candidate, scenario): use this path
                    fvar[(c.cid, om)] = m.addVar(vtype=GRB.BINARY, name=f"use_{c.cid}_{om}")
                    if c.stage == 2 and not c.is_default:
                        z2[(c.cid, om)] = m.addVar(vtype=GRB.BINARY, name=f"z2_{c.cid}_{om}")
                # exactly one path per flow per scenario
                m.addConstr(
                    gp.quicksum(fvar[(c.cid, om)] for c in inst.flow_candidates[f]) == 1,
                    f"route_{f}_{om}")
            for (w, g) in stage2_shapes:
                y2[(w, g, om)] = m.addVar(vtype=GRB.BINARY, name=f"y2_{w}_{g}_{om}")
            for l in inst.links:
                Ovar[(l, om)] = m.addVar(lb=0.0, name=f"O_{l[0]}_{l[1]}_{om}")

        # ---------------- linking constraints ----------------
        # using a stage-1 path requires its first-stage rule installed
        for om in inst.scenarios:
            for f in inst.flows:
                for c in inst.flow_candidates[f]:
                    if c.is_default:
                        continue
                    if c.stage == 1:
                        m.addConstr(fvar[(c.cid, om)] <= z1[c.cid], f"use1_{c.cid}_{om}")
                    else:
                        m.addConstr(fvar[(c.cid, om)] <= z2[(c.cid, om)], f"use2_{c.cid}_{om}")

        # shape activation
        for c_cid, v in z1.items():
            c = self._cand_by_id(inst, c_cid)
            m.addConstr(v <= y1[(c.switch, c.shape)], f"sa1_{c_cid}")
        for (c_cid, om), v in z2.items():
            c = self._cand_by_id(inst, c_cid)
            m.addConstr(v <= y2[(c.switch, c.shape, om)], f"sa2_{c_cid}_{om}")

        # memory budget per switch (bytes): first stage + each scenario's second
        # stage (hard). First-stage rules are always resident; second-stage add per scenario.
        switches = set(c.switch for f in inst.flows for c in inst.flow_candidates[f]
                       if not c.is_default)
        for w in switches:
            base = gp.quicksum(self._footprint(self._cand_by_id(inst, cid).granularity) * v
                               for cid, v in z1.items()
                               if self._cand_by_id(inst, cid).switch == w)
            for om in inst.scenarios:
                add = gp.quicksum(self._footprint(self._cand_by_id(inst, cid).granularity) * v
                                  for (cid, o), v in z2.items()
                                  if o == om and self._cand_by_id(inst, cid).switch == w)
                m.addConstr(base + add <= clf.mem_budget_bytes, f"mem_{w}_{om}")
            # lookup budget: distinct shapes per switch (first stage + scenario)
            s1 = gp.quicksum(v for (ww, g), v in y1.items() if ww == w)
            for om in inst.scenarios:
                s2 = gp.quicksum(v for (ww, g, o), v in y2.items() if ww == w and o == om)
                m.addConstr(s1 + s2 <= clf.lookup_budget, f"lookup_{w}_{om}")

        # migration affects which flows exist / their demand endpoints: here we
        # model migration's *traffic injection* and *capacity relief*. Path sets
        # are precomputed per candidate; a migrated VM's flows use candidates
        # already reflecting the destination (built by the instance builder).

        # link load & overload per scenario (incl migration traffic)
        for om in inst.scenarios:
            for l in inst.links:
                load = gp.LinExpr()
                for f in inst.flows:
                    d = inst.demand[(f, om)]
                    for c in inst.flow_candidates[f]:
                        if l in c.links:
                            load += d * fvar[(c.cid, om)]
                # migration traffic on this link (the memory-image transfer)
                for i in inst.migratable:
                    for o in inst.migration_options[i]:
                        if l in o.mig_links and not o.is_current:
                            load += o.mig_mbps * mig[(i, o.dest)]
                # load that migration relocates onto/off this link (flow follows VM)
                for (ll, vm, dest), mbps in inst.mig_flow_load.items():
                    if ll == l:
                        load += mbps * mig[(vm, dest)]
                load += inst.bg_load.get((l, om), 0.0)
                cap = inst.link_capacity[l]
                m.addConstr(Ovar[(l, om)] >= load - self.lambda_bw * cap,
                            f"ovl_{l[0]}_{l[1]}_{om}")

        # server capacity (robust): migratable VMs placed by m[i,d]
        for s in inst.servers:
            cpu = inst.server_base_cpu.get(s, 0.0) + gp.quicksum(
                o.cpu * mig[(i, o.dest)]
                for i in inst.migratable for o in inst.migration_options[i]
                if o.dest == s)
            mem = inst.server_base_mem.get(s, 0.0) + gp.quicksum(
                o.mem * mig[(i, o.dest)]
                for i in inst.migratable for o in inst.migration_options[i]
                if o.dest == s)
            m.addConstr(cpu <= inst.server_cpu_cap[s], f"cpu_{s}")
            m.addConstr(mem <= inst.server_mem_cap[s], f"mem_{s}")

        # ---------------- objective ----------------
        mig_cost = gp.quicksum(o.cost * mig[(i, o.dest)]
                               for i in inst.migratable for o in inst.migration_options[i])
        kap_new = clf.kappa_new
        kap_ins = clf.kappa_ins
        # per-shape rebuild cost can depend on the match granularity (measured
        # build time differs by technique). A shape's cost is the min build cost
        # over the granularities that open it.
        kap_by_gran = getattr(clf, "kappa_by_gran", None)
        def shape_kappa(w, g):
            if not kap_by_gran:
                return kap_new
            grans = [self._cand_by_id(inst, cid).granularity
                     for cid in list(z1) if self._cand_by_id(inst, cid).switch == w
                     and self._cand_by_id(inst, cid).shape == g]
            return min((kap_by_gran.get(gr, kap_new) for gr in grans), default=kap_new)
        fs_rule = (kap_ins * gp.quicksum(z1.values())
                   + gp.quicksum(shape_kappa(w, g) * v for (w, g), v in y1.items())
                   + self.tor_holding_cost * gp.quicksum(z1.values()))
        recourse = gp.LinExpr()
        cong = gp.LinExpr()
        late = gp.LinExpr()   # transient congestion from binding late at aggregation
        for om in inst.scenarios:
            p = inst.scenario_prob[om]
            cong += p * self.c_cong * gp.quicksum(Ovar[(l, om)] for l in inst.links)
            recourse += p * (kap_ins * gp.quicksum(v for (cid, o), v in z2.items() if o == om)
                             + kap_new * gp.quicksum(v for (ww, g, o), v in y2.items() if o == om))
            for f in inst.flows:
                d = inst.demand[(f, om)]
                for c in inst.flow_candidates[f]:
                    if c.stage == 2 and not c.is_default:
                        late += p * self.late_penalty * self.c_cong * d * fvar[(c.cid, om)]
        m.setObjective(mig_cost + fs_rule + cong + recourse + late, GRB.MINIMIZE)

        m.optimize()

        if m.SolCount == 0:
            return JointResult({}, [], float("inf"), 0, 0, 0, 0, f"no_solution:{m.Status}")

        migrations = {}
        for i in inst.migratable:
            for o in inst.migration_options[i]:
                if mig[(i, o.dest)].X > 0.5 and not o.is_current:
                    migrations[i] = o.dest
        fs_rules = [cid for cid, v in z1.items() if v.X > 0.5]
        per_scen = {om: sum(Ovar[(l, om)].X for l in inst.links) for om in inst.scenarios}
        return JointResult(
            migrations=migrations,
            first_stage_rules=fs_rules,
            expected_cost=float(m.ObjVal),
            migration_cost=float(mig_cost.getValue()),
            first_stage_rule_cost=float(fs_rule.getValue()),
            expected_congestion=float(cong.getValue()),
            expected_recourse_cost=float(recourse.getValue()),
            status="optimal" if m.Status == GRB.OPTIMAL else f"status_{m.Status}",
            per_scenario_overload=per_scen,
        )

    @staticmethod
    def _cand_by_id(inst: Instance, cid: int) -> Candidate:
        for f in inst.flows:
            for c in inst.flow_candidates[f]:
                if c.cid == cid:
                    return c
        raise KeyError(cid)
