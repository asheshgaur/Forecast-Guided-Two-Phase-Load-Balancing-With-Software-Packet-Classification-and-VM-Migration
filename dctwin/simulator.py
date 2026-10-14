"""Receding-horizon control loop, SLA-violation-seconds accounting, baselines.

One decision epoch:
  1. observe the realized demand of the previous epoch (feed the twin);
  2. forecast next-epoch per-flow demand -> point + conformal interval + event
     reliability R; maintain chronic predictability Q per VM;
  3. route the *predicted* traffic on default (ECMP) paths, find predicted
     congested links, and gather the flows crossing them;
  4. gate each such flow by reliability (the online policy):
        R >= theta_high            -> commit early: proactive ToR steering (MILP)
        theta_low <= R < theta_high-> bind late:   deferred to a reactive recourse
        Q(vm) low & host overloaded-> migrate (coarse epochs only)
  5. apply the chosen routing to the *realized* demand, optionally run a reactive
     recourse pass, and score overload as SLA-violation-seconds.

Each ToR is a software switch whose classifier has a memory budget M_w (bytes).
Controllers plan against the footprint they *believe* a rule has (the oracle's
measured m_k if classifier-aware; a nominal per-rule cost if blind); the switch
admits a new steering rule only if its *real* footprint fits, and a refused rule
leaves the flow on its current path.

Everything -- congestion, migration, rule/classifier churn -- is charged in one
currency: normalized SLA-violation-seconds.

Controllers
-----------
ecmp               : default paths only; no steering, no migration.
reactive           : no forecast; steer + migrate after congestion is observed.
predictive_blind   : forecast + optimizer with cost_aware=False.
predictive_aware   : forecast + optimizer with cost_aware=True + migration on low Q.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .classifier import ClassifierModel
from .forecast import ConformalForecaster
from .optimizer import FlowPlan, RuleOptimizer
from .topology import FatTree, Link
from .workload import Workload


@dataclass
class ControllerConfig:
    name: str = "predictive_aware"
    use_forecast: bool = True
    cost_aware: bool = True
    enable_migration: bool = True
    enable_recourse: bool = True          # late-binding aggregation correction
    theta_high: float = 0.75              # event reliability gate for proactive steering
    theta_low: float = 0.4
    theta_chronic: float = 0.45           # Q below this -> migration-eligible
    lambda_bw: float = 0.85               # congestion threshold (frac of capacity)
    point_forecast: bool = False          # ablation: ignore interval width (R=1 always)
    forecast_alpha: float = 0.1           # conformal miscoverage (nominal coverage 1-alpha)
    migration_period: int = 16            # coarse epochs
    max_alt_paths: int = 3
    epoch_seconds: float = 900.0          # 15-min slots
    # currency conversion into normalized SLA-violation-seconds
    w_cong: float = 1.0
    w_rule: float = 0.02                  # per unit of optimizer rule-churn cost
    w_mig: float = 3.0                    # per migration (downtime + risk)


@dataclass
class EpochRecord:
    t: int
    sla_cong: float
    sla_rule: float
    sla_mig: float
    overload_mbps: float
    n_steered: int
    n_migrations: int
    max_shapes: int
    mean_shapes: float
    new_shapes: int = 0        # shape classes (re)opened this epoch = classifier rebuilds
    rejected: int = 0          # steering installs refused by the switch (memory budget)
    max_mem_util: float = 0.0  # max over ToRs of real steering memory / M_w


class Simulator:
    def __init__(self, topo: FatTree, work: Workload, clf: ClassifierModel,
                 cfg: ControllerConfig, seed: int = 0):
        self.topo = topo
        self.work = work
        self.clf = clf
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.forecaster = ConformalForecaster(period=work.period, alpha=cfg.forecast_alpha)
        # deterministic per-flow match granularity (drives memory-footprint variety)
        grng = np.random.default_rng(1234)
        self._flow_gran = grng.choice(["exact", "prefix", "range"],
                                      size=len(work.flows), p=[0.5, 0.3, 0.2])

    # ---------------------------------------------------------------- helpers
    def _placement(self) -> Dict[int, str]:
        return self.work.placement

    def _flow_servers(self, fi: int) -> Tuple[str, str]:
        a, b = self.work.flows[fi]
        return self.work.placement[a], self.work.placement[b]

    def _flow_paths(self, fi: int) -> List[List[str]]:
        sa, sb = self._flow_servers(fi)
        return self.topo.path_set(sa, sb)

    def _default_path(self, fi: int, paths: List[List[str]]) -> int:
        """Deterministic ECMP choice: hash flow onto one equal-cost path."""
        if not paths:
            return 0
        return (fi * 2654435761) % len(paths)

    def _ecmp_path(self, fi: int) -> List[str]:
        paths = self._flow_paths(fi)
        return paths[self._default_path(fi, paths)] if paths else []

    def _flow_tor(self, fi: int) -> str:
        return self.topo.server_tor[self._flow_servers(fi)[0]]

    def _rule_bytes(self, fi: int, real: bool) -> float:
        """Memory a steering rule for flow fi occupies: its measured footprint
        (real=True, what the switch enforces) or the controller's belief."""
        gran = self._flow_gran[fi]
        if real:
            return self.clf.footprint(gran)
        return self.clf.assumed_footprint(gran, self.cfg.cost_aware)

    def _resident_bytes(self, current_path, real: bool,
                        exclude: frozenset = frozenset()) -> Dict[str, float]:
        """Per-ToR memory held by steered flows (those off their ECMP path)."""
        used: Dict[str, float] = defaultdict(float)
        for fi, path in current_path.items():
            if fi in exclude or not path or path == self._ecmp_path(fi):
                continue
            used[self._flow_tor(fi)] += self._rule_bytes(fi, real)
        return used

    def _open_shapes(self, current_path) -> Dict[str, set]:
        """Shape classes open at each ToR = those of the rules actually installed."""
        shapes: Dict[str, set] = defaultdict(set)
        for fi, path in current_path.items():
            if path and path != self._ecmp_path(fi):
                tor = self._flow_tor(fi)
                shapes[tor].add(self.clf.shape_class(tor, self.topo.agg_on_path(path) or "agg_?"))
        return shapes

    def _plan_and_admit(self, opt, plans, bg, current_path) -> Tuple[object, int]:
        """Solve against the controller's memory accounting, then install on the
        switch, which admits a new rule only if its real footprint fits in M_w."""
        planned = frozenset(pl.fi for pl in plans)
        believed = self._resident_bytes(current_path, real=False, exclude=planned)
        budget = {t: self.clf.mem_budget_bytes - believed.get(t, 0.0)
                  for t in {pl.tor for pl in plans}}
        res = opt.solve(plans, bg, self._open_shapes(current_path), mem_budget=budget)
        used = self._resident_bytes(current_path, real=True, exclude=planned)
        # rules already held (steered -> steered) keep their memory; new ones queue
        rejected = 0
        target = {pl.fi: pl.paths[res.chosen_path.get(pl.fi, 0)] for pl in plans}
        order = sorted(plans, key=lambda pl: (current_path[pl.fi] == self._ecmp_path(pl.fi), pl.fi))
        for pl in order:
            fi, new = pl.fi, target[pl.fi]
            if new == self._ecmp_path(fi):
                current_path[fi] = new
                continue
            need = self._rule_bytes(fi, real=True)
            if used[pl.tor] + need <= self.clf.mem_budget_bytes + 1e-9:
                used[pl.tor] += need
                current_path[fi] = new
            elif current_path[fi] != self._ecmp_path(fi):
                used[pl.tor] += need          # keep the rule it already holds
            else:
                rejected += 1                 # switch refuses; flow stays on ECMP
        return res, rejected

    def _link_loads(self, routing: Dict[int, List[str]],
                    demand: np.ndarray,
                    extra: Optional[Dict[Link, float]] = None) -> Dict[Link, float]:
        loads: Dict[Link, float] = defaultdict(float)
        for fi, path in routing.items():
            d = demand[fi]
            for l in self.topo.path_links(path):
                loads[l] += d
        if extra:
            for l, v in extra.items():
                loads[l] += v
        return loads

    def _overload_norm(self, loads: Dict[Link, float]) -> Tuple[float, float]:
        """Return (sum normalized overload fraction, sum overload Mbps)."""
        frac = 0.0
        mbps = 0.0
        for l in self.topo.all_links():
            cap = self.topo.capacity(l)
            over = loads.get(l, 0.0) - self.cfg.lambda_bw * cap
            if over > 0:
                frac += over / cap
                mbps += over
        return frac, mbps

    # ---------------------------------------------------------------- run
    def run(self, epochs: int, warmup: int = 120) -> Dict:
        cfg = self.cfg
        opt = RuleOptimizer(self.topo, self.clf, cost_aware=cfg.cost_aware,
                            max_alt_paths=cfg.max_alt_paths)

        # persistent controller state
        current_path: Dict[int, List[str]] = {}
        for fi in range(len(self.work.flows)):
            current_path[fi] = self._ecmp_path(fi)

        # migration bookkeeping: fi -> mbps of migration flow, links, epochs left
        mig_active: List[Tuple[float, List[Link], int]] = []
        host_overload_streak: Dict[str, int] = defaultdict(int)
        prev_shapes_snapshot: Dict[str, set] = defaultdict(set)

        records: List[EpochRecord] = []

        # ---- warmup: feed history to the forecaster only ----
        for t in range(warmup):
            demand = self.work.flow_demand(t)
            for fi in range(len(self.work.flows)):
                self.forecaster.observe(f"flow_{fi}", demand[fi])

        for step in range(epochs):
            t = warmup + step
            true_demand = self.work.flow_demand(t)

            # (2) forecast next epoch per flow
            reliab = np.full(len(self.work.flows), 0.3)
            pred = true_demand.copy()
            if cfg.use_forecast:
                for fi in range(len(self.work.flows)):
                    p, lo, hi, r = self.forecaster.forecast(
                        f"flow_{fi}", horizon=1,
                        capacity=self.topo.link_capacity)
                    pred[fi] = p
                    # point-forecast ablation: trust every forecast fully, so the
                    # controller commits proactively regardless of interval width.
                    reliab[fi] = 1.0 if cfg.point_forecast else r
                    # coverage audit against the realized value
                    self.forecaster.record_coverage(f"flow_{fi}", true_demand[fi], lo, hi)

            # background: migration traffic on its links this epoch
            mig_extra: Dict[Link, float] = defaultdict(float)
            still = []
            for mbps, links, left in mig_active:
                for l in links:
                    mig_extra[l] += mbps
                if left - 1 > 0:
                    still.append((mbps, links, left - 1))
            mig_active = still

            # (3) predicted default-path loads -> congested links -> considered flows
            default_routing = {fi: current_path[fi] for fi in current_path}
            pred_loads = self._link_loads(default_routing, pred, mig_extra)
            congested = {l for l in self.topo.all_links()
                         if pred_loads.get(l, 0.0) > cfg.lambda_bw * self.topo.capacity(l)}

            considered = []
            for fi in range(len(self.work.flows)):
                path = current_path[fi]
                if any(l in congested for l in self.topo.path_links(path)):
                    considered.append(fi)

            # (4) reliability gate -> proactive plans
            #   predictive: R>=theta_high commit early (ToR); theta_low<=R<theta_high
            #               bind late (recourse); R<theta_low -> migration candidate.
            #   reactive:   no proactive steering; all considered flows deferred to
            #               the reactive recourse pass (acts on realized demand).
            #   ecmp:       no forecast, no recourse -> nothing steers.
            plans: List[FlowPlan] = []
            deferred: List[int] = []
            for fi in considered:
                if cfg.use_forecast:
                    r = reliab[fi]
                    if r >= cfg.theta_high:
                        plans.append(self._make_plan(fi, pred[fi], current_path[fi]))
                    elif r >= cfg.theta_low:
                        deferred.append(fi)       # bind late (recourse)
                    # else: too unreliable to steer -> left for migration
                else:
                    deferred.append(fi)           # reactive: correct after the fact

            # background load from flows NOT in a plan (fixed during optimize)
            planned = {pl.fi for pl in plans}
            bg = self._link_loads(
                {fi: current_path[fi] for fi in current_path if fi not in planned},
                pred, mig_extra)

            sla_rule = 0.0
            n_rejected = 0
            if plans:
                res, rej = self._plan_and_admit(opt, plans, bg, current_path)
                n_rejected += rej
                # rule cost = objective minus congestion component (already in currency)
                sla_rule += cfg.w_rule * self._rule_churn_cost(res, plans)

            # (5) MEASURE epoch t on realized demand with the routing in effect.
            # Proactive (predictive, high-R) steering above is already applied, so
            # a good forecast helps *this* epoch. Reactive/late-binding recourse
            # below changes only future routing -> it lags by one epoch, which is
            # the price of not forecasting (and the option premium of binding late).
            routing = {fi: current_path[fi] for fi in current_path}
            loads = self._link_loads(routing, true_demand, mig_extra)
            over_frac, over_mbps = self._overload_norm(loads)

            # (4c) migration on coarse epochs (structural; effective onward)
            n_mig = 0
            if cfg.enable_migration and (step % cfg.migration_period == 0) and step > 0:
                n_mig = self._maybe_migrate(t, loads, host_overload_streak,
                                            mig_active, current_path)

            # (4b) late-binding / reactive recourse: detect residual congestion on
            # REALIZED demand and correct it -- but the correction takes effect from
            # t+1 (it persists in current_path), never retroactively fixing epoch t.
            if cfg.enable_recourse and deferred:
                real_congested = {l for l in self.topo.all_links()
                                  if loads.get(l, 0.0) > cfg.lambda_bw * self.topo.capacity(l)}
                rec_plans = []
                for fi in deferred:
                    if any(l in real_congested for l in self.topo.path_links(current_path[fi])):
                        rec_plans.append(self._make_plan(fi, true_demand[fi], current_path[fi]))
                if rec_plans:
                    planned2 = {pl.fi for pl in rec_plans}
                    bg2 = self._link_loads(
                        {fi: current_path[fi] for fi in current_path if fi not in planned2},
                        true_demand, mig_extra)
                    res2, rej = self._plan_and_admit(opt, rec_plans, bg2, current_path)
                    n_rejected += rej
                    sla_rule += cfg.w_rule * self._rule_churn_cost(res2, rec_plans)

            sla_cong = cfg.w_cong * over_frac * cfg.epoch_seconds / 60.0
            sla_mig = cfg.w_mig * n_mig

            open_shapes = self._open_shapes(current_path)
            tele = self.clf.summarize_state(open_shapes)
            mem_real = self._resident_bytes(current_path, real=True)
            max_util = max(mem_real.values(), default=0.0) / self.clf.mem_budget_bytes
            n_steered = sum(len(s) for s in open_shapes.values())
            new_shapes = sum(len(s - prev_shapes_snapshot.get(tor, set()))
                             for tor, s in open_shapes.items())
            prev_shapes_snapshot = {tor: set(s) for tor, s in open_shapes.items()}
            records.append(EpochRecord(
                t=t, sla_cong=sla_cong, sla_rule=sla_rule, sla_mig=sla_mig,
                overload_mbps=over_mbps,
                n_steered=n_steered,
                n_migrations=n_mig,
                max_shapes=tele["max_shapes_per_switch"],
                mean_shapes=tele["mean_shapes_per_switch"],
                new_shapes=new_shapes,
                rejected=n_rejected,
                max_mem_util=max_util,
            ))

            # feed realized demand to the twin for the next round
            for fi in range(len(self.work.flows)):
                self.forecaster.observe(f"flow_{fi}", true_demand[fi])

        return self._aggregate(records)

    # ---------------------------------------------------------------- plan
    def _make_plan(self, fi: int, demand: float, cur_path: List[str]) -> FlowPlan:
        paths = self._flow_paths(fi)
        # index 0 = current path; then the ECMP default (always a rule-free
        # fallback), then up to max_alt distinct alternates
        ecmp = self._ecmp_path(fi)
        ordered = [cur_path] + ([ecmp] if ecmp != cur_path else [])
        ordered += [p for p in paths if p not in ordered]
        ordered = ordered[: 1 + self.cfg.max_alt_paths]
        gran = [self._flow_gran[fi]] * max(0, len(ordered) - 1)
        rule = self._rule_bytes(fi, real=False)
        path_bytes = [0.0 if p == ecmp else rule for p in ordered]
        return FlowPlan(fi=fi, demand=float(demand), tor=self._flow_tor(fi),
                        paths=ordered, granularity=list(gran), prev_path=0,
                        path_bytes=path_bytes)

    def _rule_churn_cost(self, res, plans) -> float:
        """Rule-management cost in cost units: an insert per steered flow. The
        new-shape (rebuild) term is already inside the MILP objective for the
        cost-aware controller; here we surface the insert cost for accounting."""
        n_steer = sum(1 for pl in plans if res.chosen_path.get(pl.fi, 0) >= 1)
        return self.clf.kappa_ins * n_steer

    # ---------------------------------------------------------------- migrate
    def _maybe_migrate(self, t, loads, streak, mig_active, current_path) -> int:
        """Relieve persistent *access-link* (edge-server) overload -- congestion
        no path steering can fix, because every equal-cost path shares the access
        link. Prefer evicting a chronically-unpredictable VM."""
        cfg = self.cfg
        hot_server, worst = None, cfg.lambda_bw
        for sv, tor in self.topo.server_tor.items():
            l = (min(sv, tor), max(sv, tor))
            frac = loads.get(l, 0.0) / self.topo.capacity(l)
            if frac > cfg.lambda_bw:
                streak[sv] += 1
            else:
                streak[sv] = 0
            if streak[sv] >= 2 and frac > worst:
                worst, hot_server = frac, sv
        if hot_server is None:
            return 0

        # candidate VMs on the hot host, ranked: chronically-unpredictable first,
        # then by traffic contribution (evicting a big flow frees the most).
        vms_here = [vm for vm, s in self.work.placement.items() if s == hot_server]
        if not vms_here:
            return 0
        contrib = {vm: 0.0 for vm in vms_here}
        for fi, (a, b) in enumerate(self.work.flows):
            if a in contrib:
                contrib[a] += self.work.flow_demand(t)[fi]
            if b in contrib:
                contrib[b] += self.work.flow_demand(t)[fi]
        def rank(vm):
            q = self._vm_chronic(vm)
            return (0 if q < cfg.theta_chronic else 1, -contrib[vm])
        cand = sorted(vms_here, key=rank)[0]

        dst = self._pick_destination(cand, hot_server, loads)
        if dst is None:
            return 0
        src_srv = self.work.placement[cand]
        mpaths = self.topo.path_set(src_srv, dst)
        if mpaths:
            mlinks = self.topo.path_links(mpaths[0])
            M_i = self.work.vm_mem(cand) * 8000.0     # mem image, spread over D epochs
            D = 2
            mig_active.append((M_i / D, mlinks, D))
        self.work.placement[cand] = dst
        for fi, (a, b) in enumerate(self.work.flows):
            if a == cand or b == cand:
                current_path[fi] = self._ecmp_path(fi)
        streak[hot_server] = 0
        return 1

    def _vm_chronic(self, vm: int) -> float:
        # aggregate chronic predictability over the VM's flows
        qs = []
        for fi, (a, b) in enumerate(self.work.flows):
            if a == vm or b == vm:
                qs.append(self.forecaster.chronic_predictability(f"flow_{fi}"))
        return float(np.mean(qs)) if qs else 0.5

    def _pick_destination(self, vm, src_server, loads) -> Optional[str]:
        best, best_load = None, float("inf")
        src_pod = src_server.split("_")[1]
        vm_count = defaultdict(int)
        for v, s in self.work.placement.items():
            vm_count[s] += 1
        for sv in self.topo.servers:
            if sv == src_server:
                continue
            if vm_count[sv] >= self.work.vms_per_server_cap:
                continue
            tor = self.topo.server_tor[sv]
            up = loads.get((min(sv, tor), max(sv, tor)), 0.0)
            if up < best_load:
                best_load, best = up, sv
        return best

    # ---------------------------------------------------------------- report
    def _aggregate(self, records: List[EpochRecord]) -> Dict:
        arr = lambda k: np.array([getattr(r, k) for r in records])
        total_sla = float(arr("sla_cong").sum() + arr("sla_rule").sum() + arr("sla_mig").sum())
        return {
            "controller": self.cfg.name,
            "epochs": len(records),
            "sla_total": total_sla,
            "sla_cong": float(arr("sla_cong").sum()),
            "sla_rule": float(arr("sla_rule").sum()),
            "sla_mig": float(arr("sla_mig").sum()),
            "mean_overload_mbps": float(arr("overload_mbps").mean()),
            "p95_overload_mbps": float(np.percentile(arr("overload_mbps"), 95)),
            "total_migrations": int(arr("n_migrations").sum()),
            "mean_max_shapes": float(arr("max_shapes").mean()),
            "mean_mean_shapes": float(arr("mean_shapes").mean()),
            "total_new_shapes": int(arr("new_shapes").sum()),
            "total_rejected": int(arr("rejected").sum()),
            "max_mem_util": float(arr("max_mem_util").max()) if records else 0.0,
            "coverage": self.forecaster.empirical_coverage,
            "records": records,
        }
