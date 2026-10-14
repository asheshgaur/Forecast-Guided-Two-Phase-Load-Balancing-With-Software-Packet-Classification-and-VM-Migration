"""Packet-classifier cost model -- the load-bearing bridge to prior work.

Switches are *software* switches (Open vSwitch, a P4 software target, a DPDK
pipeline); the classifier underneath each one lives in RAM and is bounded by a
memory budget, not a count of table entries. The controller's rule churn
degrades that classifier. We model three content-dependent costs, all linear so
they drop straight into the MILP:

  * Memory footprint  m_k : bytes steering rule k adds to the switch's
    classifier. The memory constraint is sum(m_k) <= M_w (bytes), not a rule
    count. m_k depends on the datapath backend:
      - "mask"   : value/mask match abstraction (OpenFlow on OVS, P4 ternary
                   tables). A range field is expanded into e_k masked entries,
                   so a range rule costs ~e_k times a prefix rule.
      - "native" : range-native algorithmic classifier (the structures
                   EnsembleClassifier selects among). Ranges are stored as-is;
                   m_k is set by the selected structure, not the match shape.

  * Shape class  g(k) : which fields are wildcarded / the egress group targeted.
    A steering rule's shape here is the *aggregation switch* it steers toward:
    rules sharing an egress agg switch share a tuple / a decision subtree. Lookup
    cost at a switch = number of distinct active shape classes (probes in
    tuple-space search; trees traversed in the EffiCuts family).

  * Update cost : adding a rule to an already-open shape class is an incremental
    insert (kappa_ins); opening a new shape class is a rebuild (kappa_new >>
    kappa_ins).

Because these depend on the *incumbent* rule set, a classifier-aware controller
acquires an incentive nobody has modeled: prefer steering shapes already open at
the target ToR. The observable consequence is *shape reuse*.

A classifier-*blind* controller ignores all three: it prices no rebuilds and
assumes every rule costs the same nominal footprint (it counts rules). The
switch still enforces the real M_w on the real m_k, so a blind plan can be
refused at install time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# Fallback per-rule footprints (bytes) used only when no oracle calibration is
# loaded. These are the values measured by ClassifierOracle at 40 rules (see
# results/oracle_calibration.json); prefer load_calibration() over these.
DEFAULT_FOOTPRINT = {
    "mask": {"exact": 36.6, "prefix": 36.6, "range": 2961.0},
    "native": {"exact": 16.8, "prefix": 16.8, "range": 16.8},
}


@dataclass
class ClassifierModel:
    """Per-switch classifier accounting and cost parameters."""

    mem_budget_bytes: float = 8192.0  # M_w : steering-rule memory budget per ToR (bytes)
    backend: str = "mask"             # datapath match abstraction: "mask" | "native"
    lookup_budget: int = 6            # Lambda_max : max distinct shape classes/switch
    kappa_ins: float = 1.0            # cost of an incremental insert/remove
    kappa_new: float = 20.0           # cost of opening a new shape class (rebuild)
    # backend -> granularity -> measured bytes per rule (set by the oracle)
    footprint_table: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {b: dict(t) for b, t in DEFAULT_FOOTPRINT.items()})
    expansion_table: Optional[Dict[str, int]] = None   # e_k masked entries (report only)

    def footprint(self, granularity: str) -> float:
        """True memory footprint m_k (bytes) of a steering rule on this backend."""
        tbl = self.footprint_table[self.backend]
        return float(tbl.get(granularity, min(tbl.values())))

    def nominal_footprint(self) -> float:
        """What a rule-counting controller assumes every rule costs: the
        footprint of the cheapest match shape."""
        return float(min(self.footprint_table[self.backend].values()))

    def assumed_footprint(self, granularity: str, cost_aware: bool) -> float:
        return self.footprint(granularity) if cost_aware else self.nominal_footprint()

    @staticmethod
    def shape_class(tor: str, egress_agg: str) -> Tuple[str, str]:
        """Shape class of a steering rule = (hosting ToR, egress agg switch)."""
        return (tor, egress_agg)

    # ------------------------------------------------ live-state bookkeeping
    def lookup_cost(self, active_shapes_at_switch: int) -> int:
        return active_shapes_at_switch

    def summarize_state(self, installed: Dict) -> Dict[str, float]:
        """installed: tor -> set of shape classes currently open.

        Returns fabric-wide classifier telemetry.
        """
        shapes = [len(s) for s in installed.values()]
        return {
            "max_shapes_per_switch": max(shapes) if shapes else 0,
            "mean_shapes_per_switch": (sum(shapes) / len(shapes)) if shapes else 0.0,
            "total_open_shapes": sum(shapes),
            "switches_over_lookup_budget": sum(1 for s in shapes if s > self.lookup_budget),
        }
