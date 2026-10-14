"""VM placement and traffic demand generation.

A Workload places n VMs onto servers (round-robin, respecting a per-server VM
cap), draws a sparse communication graph among VMs, and generates per-flow
bandwidth time series with a diurnal component plus tenant-correlated bursts.

Two *kinds* of predictability are baked in deliberately, because the whole point
of the controller is to tell them apart:

  * event uncertainty -- momentary bursts that widen a single forecast interval;
  * chronic unpredictability -- a subset of VMs (`chaotic_frac`) whose demand is
    driven by high-variance noise no forecaster can track. These are the VMs
    migration should target, and the ones a rule-only controller cannot fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


Flow = Tuple[int, int]  # (src_vm, dst_vm)


@dataclass
class Workload:
    n_vms: int
    servers: List[str]
    seed: int = 0
    vms_per_server_cap: int = 4
    period: int = 96                 # diurnal period in epochs (e.g. 15-min slots/day)
    avg_flows_per_vm: float = 2.0
    base_bw: float = 300.0           # Mbps, mean per-flow demand (mice)
    elephant_frac: float = 0.16      # fraction of flows that are large + steady
    elephant_mult: float = 11.0      # elephant size relative to base
    mouse_mult: float = 0.5
    max_elephants_per_server: int = 2  # keep access links from being the bottleneck
    burst_prob: float = 0.03
    burst_mult: float = 3.0
    chaotic_frac: float = 0.15       # fraction of VMs that are chronically unpredictable
    cpu_mean: float = 0.4            # fraction of a server's CPU per VM
    mem_mean: float = 0.35

    placement: Dict[int, str] = field(default_factory=dict)   # vm -> server
    flows: List[Flow] = field(default_factory=list)
    chaotic_vms: set = field(default_factory=set)
    elephant_flows: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._place()
        self._draw_flows()
        self.chaotic_vms = set(
            self.rng.choice(self.n_vms,
                            size=max(1, int(self.chaotic_frac * self.n_vms)),
                            replace=False).tolist()
        )
        self._designate_elephants()
        # persistent per-VM/per-flow phase offsets so series are individuated
        self._flow_phase = self.rng.uniform(0, 2 * np.pi, size=len(self.flows))
        self._flow_scale = self.rng.uniform(0.6, 1.6, size=len(self.flows))
        self._vm_cpu = np.clip(self.rng.normal(self.cpu_mean, 0.12, self.n_vms), 0.05, 0.95)
        self._vm_mem = np.clip(self.rng.normal(self.mem_mean, 0.10, self.n_vms), 0.05, 0.95)

    def _designate_elephants(self) -> None:
        """Mark a subset of flows as elephants, capping elephants per source
        server so access (edge-server) links never become the bottleneck --
        we want the *fabric* (agg/core) to be where ECMP collisions bite."""
        order = list(range(len(self.flows)))
        self.rng.shuffle(order)
        target = int(self.elephant_frac * len(self.flows))
        per_server = {s: 0 for s in self.servers}
        for fi in order:
            if len(self.elephant_flows) >= target:
                break
            a, _ = self.flows[fi]
            s = self.placement[a]
            if per_server[s] < self.max_elephants_per_server:
                self.elephant_flows.add(fi)
                per_server[s] += 1

    # ------------------------------------------------------------- placement
    def _place(self) -> None:
        counts = {s: 0 for s in self.servers}
        order = list(self.servers)
        si = 0
        for vm in range(self.n_vms):
            # advance to a server with spare capacity
            tries = 0
            while counts[order[si % len(order)]] >= self.vms_per_server_cap:
                si += 1
                tries += 1
                if tries > len(order):
                    raise RuntimeError("Not enough server slots for VMs; raise cap or servers.")
            s = order[si % len(order)]
            self.placement[vm] = s
            counts[s] += 1
            si += 1

    def _draw_flows(self) -> None:
        n_flows = int(self.avg_flows_per_vm * self.n_vms)
        seen = set()
        while len(self.flows) < n_flows:
            a, b = self.rng.integers(0, self.n_vms, size=2)
            if a == b:
                continue
            # skip pairs co-located on the same server (no fabric traffic)
            if self.placement[int(a)] == self.placement[int(b)]:
                continue
            key = (int(a), int(b))
            if key in seen:
                continue
            seen.add(key)
            self.flows.append(key)

    # ------------------------------------------------------------- demand
    def flow_demand(self, t: int) -> np.ndarray:
        """Bandwidth (Mbps) of every flow at epoch t. Deterministic given seed+t."""
        rng = np.random.default_rng(self.seed * 1_000_003 + t)
        out = np.empty(len(self.flows))
        for fi, (a, b) in enumerate(self.flows):
            diurnal = 0.5 * (1 + np.sin(2 * np.pi * t / self.period + self._flow_phase[fi]))
            size = self.elephant_mult if fi in self.elephant_flows else self.mouse_mult
            base = self.base_bw * size * self._flow_scale[fi] * (0.55 + 0.7 * diurnal)
            chaotic = (a in self.chaotic_vms) or (b in self.chaotic_vms)
            if chaotic:
                # heavy multiplicative noise: no forecaster tracks this. Chaotic
                # elephants are the structural problem migration must fix.
                base *= rng.lognormal(mean=0.0, sigma=0.55)
            else:
                # predictable: elephants are steady (high reliability), mice noisier
                sigma = 0.05 if fi in self.elephant_flows else 0.15
                base *= rng.normal(1.0, sigma)
            if rng.random() < self.burst_prob:
                base *= self.burst_mult
            out[fi] = max(0.0, base)
        return out

    def vm_cpu(self, vm: int) -> float:
        return float(self._vm_cpu[vm])

    def vm_mem(self, vm: int) -> float:
        return float(self._vm_mem[vm])

    def flow_endpoints_servers(self, fi: int) -> Tuple[str, str]:
        a, b = self.flows[fi]
        return self.placement[a], self.placement[b]
