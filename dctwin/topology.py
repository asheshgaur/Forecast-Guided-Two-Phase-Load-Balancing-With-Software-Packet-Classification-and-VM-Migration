"""k-ary Fat Tree topology and path enumeration.

Standard k-ary Fat Tree (Al-Fares et al., SIGCOMM'08):
  * k pods; each pod has k/2 aggregation switches and k/2 edge (ToR) switches.
  * each edge switch hosts k/2 servers  ->  k^3/4 servers total.
  * (k/2)^2 core switches; each aggregation switch links to k/2 cores.

We map the three-layer rule hierarchy of the formulation onto the three switch
layers of the Fat Tree:
    Border routers  B  <->  core switches   (top of tree, external ingress)
    Aggregation     A  <->  aggregation switches
    ToR             T  <->  edge switches    (source of steering)

Equal-cost paths:
  * inter-pod flows have (k/2)^2 shortest paths (one per core reachable through
    the source pod's aggregation switches);
  * intra-pod, inter-edge flows have k/2 shortest paths (one per aggregation
    switch in the pod);
  * intra-edge flows stay on the ToR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple

import networkx as nx


Node = str
Link = Tuple[Node, Node]  # ordered (u, v); we canonicalize with tuple(sorted(...))


def link_key(u: Node, v: Node) -> Link:
    """Canonical undirected link key."""
    return (u, v) if u <= v else (v, u)


@dataclass
class FatTree:
    """A k-ary Fat Tree with precomputed server-to-server path sets."""

    k: int
    link_capacity: float = 10_000.0  # per-link BW capacity (Mbps), uniform
    graph: nx.Graph = field(default_factory=nx.Graph)

    core: List[Node] = field(default_factory=list)
    agg: List[Node] = field(default_factory=list)
    edge: List[Node] = field(default_factory=list)  # ToR
    servers: List[Node] = field(default_factory=list)

    # server -> its ToR edge switch
    server_tor: Dict[Node, Node] = field(default_factory=dict)
    # (src_server, dst_server) -> list of paths; each path is a list of nodes
    paths: Dict[Tuple[Node, Node], List[List[Node]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k % 2 != 0:
            raise ValueError("Fat Tree arity k must be even.")
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        k = self.k
        half = k // 2
        G = self.graph

        # Core switches: (k/2)^2 of them, indexed by (i, j), i,j in [0, k/2).
        for i, j in product(range(half), range(half)):
            n = f"core_{i}_{j}"
            self.core.append(n)
            G.add_node(n, layer="core")

        for p in range(k):  # pods
            pod_agg, pod_edge = [], []
            for a in range(half):
                n = f"agg_{p}_{a}"
                pod_agg.append(n)
                self.agg.append(n)
                G.add_node(n, layer="agg", pod=p)
            for e in range(half):
                n = f"edge_{p}_{e}"
                pod_edge.append(n)
                self.edge.append(n)
                G.add_node(n, layer="edge", pod=p)

            # edge <-> agg: full mesh within pod
            for e_node in pod_edge:
                for a_node in pod_agg:
                    G.add_edge(e_node, a_node, capacity=self.link_capacity)

            # agg <-> core: aggregation switch a connects to cores core_{a}_{*}
            for a in range(half):
                a_node = f"agg_{p}_{a}"
                for j in range(half):
                    G.add_edge(a_node, f"core_{a}_{j}", capacity=self.link_capacity)

            # servers under each edge switch
            for e in range(half):
                e_node = f"edge_{p}_{e}"
                for s in range(half):
                    sv = f"srv_{p}_{e}_{s}"
                    self.servers.append(sv)
                    self.server_tor[sv] = e_node
                    G.add_node(sv, layer="server", pod=p)
                    G.add_edge(sv, e_node, capacity=self.link_capacity)

        self._enumerate_paths()

    def _enumerate_paths(self) -> None:
        """Precompute all shortest (equal-cost) paths between server pairs."""
        G = self.graph
        for a in self.servers:
            for b in self.servers:
                if a >= b:
                    continue
                try:
                    ps = list(nx.all_shortest_paths(G, a, b))
                except nx.NetworkXNoPath:
                    ps = []
                self.paths[(a, b)] = ps

    # --------------------------------------------------------------- accessors
    def path_set(self, a: Node, b: Node) -> List[List[Node]]:
        """Equal-cost paths between servers a and b (order-independent)."""
        if a > b:
            return [list(reversed(p)) for p in self.paths.get((b, a), [])]
        return self.paths.get((a, b), [])

    def path_links(self, path: List[Node]) -> List[Link]:
        return [link_key(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def all_links(self) -> List[Link]:
        return [link_key(u, v) for u, v in self.graph.edges()]

    def capacity(self, link: Link) -> float:
        u, v = link
        return self.graph[u][v]["capacity"]

    def agg_on_path(self, path: List[Node]) -> Node | None:
        """The aggregation switch a path traverses (defines a steering shape)."""
        for n in path:
            if n.startswith("agg_"):
                return n
        return None

    def summary(self) -> str:
        return (
            f"FatTree(k={self.k}): {len(self.servers)} servers, "
            f"{len(self.edge)} ToR, {len(self.agg)} agg, {len(self.core)} core, "
            f"{self.graph.number_of_edges()} links"
        )
