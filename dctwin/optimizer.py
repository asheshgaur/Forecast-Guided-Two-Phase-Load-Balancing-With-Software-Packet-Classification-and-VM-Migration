"""Classifier-aware rule-placement optimizer (single decision epoch).

Given the twin's forecast for the next horizon, decide which flows to steer off
their default (ECMP) path and onto an alternate equal-cost path, by installing a
steering rule at the source ToR. The objective is denominated in one currency:

    min   sum_l  c_cong * Overload_l            (soft: penalized congestion)
        + kappa_ins * sum  churn                (rule insert/remove)
        + kappa_new * sum  newly-opened shapes  (classifier rebuild)

subject to
    * each flow on exactly one candidate path,
    * memory budget per ToR:  sum m_k * steer <= M_w - resident   (HARD, bytes),
    * shape activation:  steering on path p opens shape g(p) at the ToR,
    * lookup budget:  distinct active shapes per ToR <= Lambda_max (HARD),
    * churn / new-shape linearization against the incumbent state.

Link capacity is intentionally *soft* (penalized overload), because hard link +
hard memory is routinely infeasible and an infeasible controller is undefined in
exactly the moments it matters. The default path uses no rule and no shape, so
the all-default routing is always feasible.

Solved with scipy.optimize.milp (HiGHS) -- no external solver dependency.

Setting cost_aware=False zeroes kappa_ins/kappa_new: the "classifier-blind"
ablation. It still respects a memory budget and the lookup budget, but it counts
rules -- the caller builds its FlowPlan.path_bytes from the nominal per-rule
footprint rather than the oracle's measured m_k -- so its plan can exceed the
switch's real memory and be refused at install time (see Simulator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .classifier import ClassifierModel
from .topology import FatTree, Link, link_key


@dataclass
class FlowPlan:
    """A considered flow and its candidate routing options."""
    fi: int
    demand: float
    tor: str
    paths: List[List[str]]          # index 0 = default; >=1 = steering
    granularity: List[str]          # match granularity per steering path
    prev_path: int                  # incumbent path index (for churn)
    path_bytes: List[float] = None  # memory the flow's rule holds on each path, as
                                    # the controller accounts it (0 = ECMP default)


@dataclass
class OptimizerResult:
    chosen_path: Dict[int, int]          # fi -> path index
    open_shapes: Dict[str, set]          # tor -> set of shape classes
    objective: float
    predicted_overload: float            # sum of O_l at the solution (Mbps)
    n_steered: int
    status: str


class RuleOptimizer:
    def __init__(self, topo: FatTree, clf: ClassifierModel,
                 c_cong: float = 1.0, cost_aware: bool = True,
                 max_alt_paths: int = 3, time_limit: float = 10.0):
        self.topo = topo
        self.clf = clf
        self.c_cong = c_cong
        self.cost_aware = cost_aware
        self.max_alt_paths = max_alt_paths
        self.time_limit = time_limit

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        plans: Sequence[FlowPlan],
        bg_load: Dict[Link, float],
        prev_open_shapes: Dict[str, set],
        mem_budget: Optional[Dict[str, float]] = None,
    ) -> OptimizerResult:
        """mem_budget: tor -> bytes available to the planned flows' rules (M_w
        minus memory held by steered flows outside the plan). Defaults to M_w."""
        if not plans:
            return OptimizerResult({}, {t: set(s) for t, s in prev_open_shapes.items()},
                                   0.0, sum(max(0.0, l) for l in bg_load.values()), 0, "empty")

        topo, clf = self.topo, self.clf

        # ---- enumerate variables -------------------------------------------
        # x[(fi,p)]  binary ; O[l] cont>=0 ; y[(tor,shape)] binary ;
        # a[(fi,p)] cont>=0 (churn, steering only) ; nsh[(tor,shape)] cont>=0
        xvars: List[Tuple[int, int]] = []
        steer_shape: Dict[Tuple[int, int], Tuple[str, str]] = {}
        path_mem: Dict[Tuple[int, int], float] = {}
        for pl in plans:
            for p in range(len(pl.paths)):
                xvars.append((pl.fi, p))
                if pl.path_bytes is not None and pl.path_bytes[p] > 0:
                    path_mem[(pl.fi, p)] = pl.path_bytes[p]
                if p >= 1:
                    egress = topo.agg_on_path(pl.paths[p]) or "agg_?"
                    steer_shape[(pl.fi, p)] = clf.shape_class(pl.tor, egress)

        # links touched by any candidate path
        touched: set = set()
        for pl in plans:
            for path in pl.paths:
                touched.update(topo.path_links(path))
        links = sorted(touched)

        # shape variables: every shape any steering rule could open
        shape_vars: List[Tuple[str, str]] = sorted({s for s in steer_shape.values()})
        tors = sorted({pl.tor for pl in plans})

        # variable index map
        idx: Dict[Tuple, int] = {}

        def add(name):
            idx[name] = len(idx)

        for v in xvars:
            add(("x", *v))
        for l in links:
            add(("O", l))
        for s in shape_vars:
            add(("y", *s))
        for v in xvars:
            if v[1] >= 1:
                add(("a", *v))
        for s in shape_vars:
            add(("nsh", *s))
        N = len(idx)

        c = np.zeros(N)
        integrality = np.zeros(N)
        lb = np.zeros(N)
        ub = np.full(N, np.inf)

        for v in xvars:
            j = idx[("x", *v)]
            integrality[j] = 1
            ub[j] = 1
        for s in shape_vars:
            j = idx[("y", *s)]
            integrality[j] = 1
            ub[j] = 1

        # objective
        for l in links:
            c[idx[("O", l)]] = self.c_cong
        if self.cost_aware:
            for v in xvars:
                if v[1] >= 1:
                    c[idx[("a", *v)]] = clf.kappa_ins
            for s in shape_vars:
                c[idx[("nsh", *s)]] = clf.kappa_new

        constraints: List[LinearConstraint] = []

        def row():
            return np.zeros(N)

        # (1) one path per flow
        for pl in plans:
            r = row()
            for p in range(len(pl.paths)):
                r[idx[("x", pl.fi, p)]] = 1.0
            constraints.append(LinearConstraint(r, 1.0, 1.0))

        # (2) overload:  O_l - sum B*x[in path] >= bg - cap   (and O>=0 via bounds)
        # membership: which (fi,p) traverse link l
        for l in links:
            r = row()
            r[idx[("O", l)]] = 1.0
            for pl in plans:
                for p, path in enumerate(pl.paths):
                    if l in topo.path_links(path):
                        r[idx[("x", pl.fi, p)]] -= pl.demand
            rhs = bg_load.get(l, 0.0) - topo.capacity(l)
            constraints.append(LinearConstraint(r, rhs, np.inf))

        # (3) memory budget per ToR (HARD, bytes): sum m_k x_k <= M_w - resident
        tor_of = {pl.fi: pl.tor for pl in plans}
        for t in tors:
            r = row()
            any_term = False
            for (fi, p), mb in path_mem.items():
                if tor_of[fi] == t:
                    r[idx[("x", fi, p)]] += mb
                    any_term = True
            if any_term:
                cap = clf.mem_budget_bytes if mem_budget is None else mem_budget.get(t, clf.mem_budget_bytes)
                constraints.append(LinearConstraint(r, -np.inf, float(max(0.0, cap))))

        # (4) shape activation: x[fi,p] - y[shape] <= 0
        for (fi, p), s in steer_shape.items():
            r = row()
            r[idx[("x", fi, p)]] = 1.0
            r[idx[("y", *s)]] = -1.0
            constraints.append(LinearConstraint(r, -np.inf, 0.0))

        # (5) lookup budget per ToR (HARD): sum_g y[t,g] <= Lambda_max
        for t in tors:
            r = row()
            any_term = False
            for s in shape_vars:
                if s[0] == t:
                    r[idx[("y", *s)]] = 1.0
                    any_term = True
            if any_term:
                constraints.append(LinearConstraint(r, -np.inf, float(clf.lookup_budget)))

        # (6) churn:  a[fi,p] >= x[fi,p] - xprev
        for pl in plans:
            for p in range(1, len(pl.paths)):
                r = row()
                r[idx[("a", pl.fi, p)]] = 1.0
                r[idx[("x", pl.fi, p)]] = -1.0
                xprev = 1.0 if pl.prev_path == p else 0.0
                constraints.append(LinearConstraint(r, -xprev, np.inf))

        # (7) new shape: nsh[s] >= y[s] - yprev[s]
        for s in shape_vars:
            r = row()
            r[idx[("nsh", *s)]] = 1.0
            r[idx[("y", *s)]] = -1.0
            yprev = 1.0 if s in prev_open_shapes.get(s[0], set()) else 0.0
            constraints.append(LinearConstraint(r, -yprev, np.inf))

        # a time limit keeps the per-epoch loop bounded; on timeout HiGHS returns
        # its best feasible incumbent, which we use
        res = milp(c, constraints=constraints, integrality=integrality,
                   bounds=Bounds(lb, ub), options={"time_limit": self.time_limit})

        if res.x is None:
            # fall back to all-default (always feasible in theory)
            chosen = {pl.fi: 0 for pl in plans}
            return OptimizerResult(chosen, {t: set(s) for t, s in prev_open_shapes.items()},
                                   float("inf"),
                                   sum(max(0.0, l) for l in bg_load.values()),
                                   0, f"fallback:{res.status if hasattr(res,'status') else 'fail'}")

        x = res.x
        chosen: Dict[int, int] = {}
        for pl in plans:
            sel = 0
            for p in range(len(pl.paths)):
                if x[idx[("x", pl.fi, p)]] > 0.5:
                    sel = p
                    break
            chosen[pl.fi] = sel

        open_shapes: Dict[str, set] = {t: set() for t in tors}
        for s in shape_vars:
            if x[idx[("y", *s)]] > 0.5:
                open_shapes.setdefault(s[0], set()).add(s)
        # carry over previously-open shapes at untouched ToRs
        for t, sset in prev_open_shapes.items():
            open_shapes.setdefault(t, set())

        overload = sum(max(0.0, x[idx[("O", l)]]) for l in links)
        n_steered = sum(1 for fi, p in chosen.items() if p >= 1)
        return OptimizerResult(chosen, open_shapes, float(res.fun),
                               float(overload), n_steered,
                               "optimal" if res.success else "time_limit")
