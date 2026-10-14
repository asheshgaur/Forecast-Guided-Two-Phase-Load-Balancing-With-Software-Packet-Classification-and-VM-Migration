"""Nested VM populations for a fixed, 16-host Fat Tree density study.

The 16-VM case is exactly the published mixed workload. Additional independent
16-VM cohorts use the same per-flow demand distribution. Host permutations
spread their initial placements without changing CPU or memory requirements.
The final population has four VMs per host, the integer CPU packing limit.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .two_phase import PilotWorkload


DENSITIES = (16, 24, 32, 40, 48, 56, 64)


@dataclass
class DensityWorkload:
    topo: object
    n_vms: int
    seed: int
    n_steps: int = 416

    def __post_init__(self):
        if self.topo.k != 4 or len(self.topo.servers) != 16:
            raise ValueError("This study fixes the physical network at k=4 / 16 hosts")
        if self.n_vms not in DENSITIES:
            raise ValueError(f"n_vms must be one of {DENSITIES}")
        self.regime = "mixed_density"
        base = PilotWorkload(self.topo, "mixed", self.seed, self.n_steps)
        self.placement = dict(base.placement)
        self.flows = list(base.flows)
        traces = [base.demand[:, f] for f in range(8)]
        self.elephant_flags = [True] * 4 + [False] * 4
        self.cohort_seeds = [self.seed]
        server_index = {s: i for i, s in enumerate(self.topo.servers)}
        # Complete successive 8-VM additions contain two elephants and two mice.
        order = (0, 4, 1, 5, 2, 6, 3, 7)
        # XOR 1 swaps sibling hosts; XOR 8 rotates endpoints by two pods.
        # Cohorts 0+1 and 2+3 each contribute exactly two VMs to every host.
        for cohort, permutation in enumerate((0, 1, 8, 9)):
            if cohort == 0 or len(self.placement) == self.n_vms:
                continue
            cohort_seed = int(np.random.SeedSequence([self.seed, cohort, 61002]).generate_state(1)[0])
            self.cohort_seeds.append(cohort_seed)
            extra = PilotWorkload(self.topo, "mixed", cohort_seed, self.n_steps)
            for f in order:
                if len(self.placement) == self.n_vms:
                    break
                a, b = extra.flows[f]
                for vm in (a, b):
                    self.placement[16 * cohort + vm] = self.topo.servers[
                        server_index[extra.placement[vm]] ^ permutation]
                self.flows.append((16 * cohort + a, 16 * cohort + b))
                traces.append(extra.demand[:, f])
                self.elephant_flags.append(f < 4)
        self.demand = np.column_stack(traces)
        self.cpu = {v: .18 for v in self.placement}
        self.mem = {v: 4. for v in self.placement}
        self.granularity = ["exact"] * len(self.flows)
        counts = Counter(self.placement.values())
        assert len(self.placement) == self.n_vms
        assert max(counts.values()) <= 4
        assert len(self.flows) == self.n_vms // 2
        assert sum(self.elephant_flags) == len(self.flows) // 2
        assert {v for pair in self.flows for v in pair} == set(self.placement)

