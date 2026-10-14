"""dctwin: Predictive Hierarchical Control of Routing Rules and VM Migration.

A digital-twin controller for Fat Tree data center networks that chooses, per
predicted congestion event, *where in the switch hierarchy to commit* -- pricing
that choice by calibrated forecast uncertainty -- while modeling the cost its own
rule churn imposes on the packet classifier underneath each switch.

Modules
-------
topology   : k-ary Fat Tree, path enumeration, border/agg/ToR layers.
workload   : VM placement and diurnal + bursty demand generation.
forecast   : split-conformal interval forecaster; event reliability R_l and
             chronic predictability Q_i.
classifier : shape-class cost model (memory footprint m_k vs budget M_w, lookup cost, churn).
optimizer  : classifier-aware rule-placement MILP (scipy HiGHS).
simulator  : receding-horizon control loop + SLA-violation-seconds accounting;
             ECMP and reactive baselines.
"""

from .topology import FatTree
from .workload import Workload
from .forecast import ConformalForecaster
from .classifier import ClassifierModel
from .optimizer import RuleOptimizer
from .simulator import Simulator, ControllerConfig

__all__ = [
    "FatTree",
    "Workload",
    "ConformalForecaster",
    "ClassifierModel",
    "RuleOptimizer",
    "Simulator",
    "ControllerConfig",
]
