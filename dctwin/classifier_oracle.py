"""Real packet-classifier cost oracle -- Option B, the bridge made concrete.

This module grounds the optimizer's classifier-cost coefficients in the actual
Ensemble-RL / synthclass packet-classification code, rather than the invented
proxies in `classifier.py`. Four primitives, each grounded in the part of the
real system that is trustworthy:

  * memory footprint m_k -- bytes a steering rule adds to the software switch's
    classifier, MEASURED from the C++ benchmark's Size(bytes) under two
    datapath backends:
      - "mask"   : a value/mask match abstraction (OpenFlow on Open vSwitch, P4
                   ternary tables). A range field has no native encoding, so the
                   rule is expanded into e_k masked entries and stored in
                   OVS-style tuple-space search; m_k = bytes(expanded) / #rules.
      - "native" : a range-native algorithmic classifier -- the structure the
                   EnsembleClassifier selector deploys, fed the unexpanded rule.

  * masked-entry count e_k -- computed EXACTLY from range->prefix decomposition
    of a rule's 5-tuple match. Only the "mask" backend pays it.

  * shape class g(k)    -- the *technique* the trained supervised leaf-selector
    assigns to a rule of that match pattern (PartitionSort / TupleMerge / HyperCuts
    / ...). This IS "supervised structure selection at terminal nodes of the
    composition tree." Rules the classifier would place in the same structure
    share a shape.

  * update / lookup cost -- MEASURED by running the real `./main` benchmark on a
    representative rule subset for the selected technique (construction time,
    classification time, average tables queried, memory). Cached.

Why not use the neural net's raw latency outputs directly? The trained checkpoint
was fit on leaves of ~10^4 rules; a switch steering-table holds <10^2. The net's
*technique choice* generalizes (it is a ranking), but its *absolute* latency
regression does not, so we take the choice from the net and the magnitudes from
the real benchmark.

The oracle's job is to `calibrate()` a `ClassifierModel` -- filling m_k, the shape
map, kappa_ins, and kappa_new with real numbers -- so everything downstream (MILP,
simulator) is unchanged but now runs on measured classifier costs. The measured
report can be saved to JSON and reloaded with `load_calibration`, so experiments
reproduce without re-running the C++ benchmark.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .classifier import ClassifierModel


# 5-tuple field widths (bits): src_ip, dst_ip, src_port, dst_port, protocol
FIELD_BITS = (32, 32, 16, 16, 8)


# --------------------------------------------------------------------- expansion
def range_to_prefix_count(lo: int, hi: int, bits: int) -> int:
    """Exact number of prefixes a range [lo, hi] decomposes into on a `bits`-wide
    field: the multiplicative masked-entry expansion one field contributes on a
    value/mask datapath (OpenFlow, P4 ternary tables)."""
    if lo > hi:
        return 0
    return len(range_to_prefix_blocks(lo, hi, bits))


def range_to_prefix_blocks(lo: int, hi: int, bits: int) -> List[Tuple[int, int]]:
    """The aligned prefix blocks (as (lo, hi) pairs) that exactly cover [lo, hi]."""
    if lo > hi:
        return []
    blocks = []
    top = 1 << bits
    lo &= top - 1
    hi &= top - 1
    while lo <= hi:
        # largest prefix block starting at lo that fits within [lo, hi]
        # size limited by alignment of lo and by remaining span
        max_size_by_align = (lo & -lo) if lo != 0 else top
        span = hi - lo + 1
        size = 1
        while size * 2 <= span and size * 2 <= max_size_by_align:
            size *= 2
        blocks.append((lo, lo + size - 1))
        lo += size
        if lo >= top:  # reached the top of the field
            break
    return blocks


@dataclass
class MatchSpec:
    """A steering rule's 5-tuple match: (lo, hi) per field."""
    ranges: Tuple[Tuple[int, int], ...]  # length 5

    def expansion(self) -> int:
        e = 1
        for f, (lo, hi) in enumerate(self.ranges):
            e *= max(1, range_to_prefix_count(lo, hi, FIELD_BITS[f]))
        return e

    def to_masked(self) -> List["MatchSpec"]:
        """The e_k value/mask entries a mask-only datapath must install for this
        rule: the cross product of each field's prefix blocks."""
        from itertools import product
        per_field = [range_to_prefix_blocks(lo, hi, FIELD_BITS[f])
                     for f, (lo, hi) in enumerate(self.ranges)]
        return [MatchSpec(tuple(combo)) for combo in product(*per_field)]

    def wildcard_pattern(self) -> Tuple[str, ...]:
        """Per-field kind: 'exact' | 'prefix' | 'range' | 'wild'. Determines the
        classifier shape more than the exact values do."""
        out = []
        for f, (lo, hi) in enumerate(self.ranges):
            full = (1 << FIELD_BITS[f]) - 1
            if lo == 0 and hi == full:
                out.append("wild")
            elif lo == hi:
                out.append("exact")
            elif range_to_prefix_count(lo, hi, FIELD_BITS[f]) == 1:
                out.append("prefix")
            else:
                out.append("range")
        return tuple(out)


@dataclass
class ClassifierOracle:
    ensemble_root: Path
    checkpoint_path: Optional[Path] = None
    portfolio: Sequence[str] = ("PartitionSort", "PriorityTuple", "HyperCuts",
                                "HyperSplit", "ByteCuts", "CutSplit", "TabTree", "NPTree")
    packet_count: int = 2000
    accuracy_packets: int = 300
    rep_rules: int = 40           # size of representative leaf used for benchmarking
    mask_technique: str = "PriorityTuple"   # OVS tuple-space search: the mask backend

    _ckpt: Optional[dict] = field(default=None, init=False)
    _synth: Optional[object] = field(default=None, init=False)
    _bench_cache: Dict[Tuple[str, str], dict] = field(default_factory=dict, init=False)
    _tech_cache: Dict[Tuple[str, ...], str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.ensemble_root = Path(self.ensemble_root).resolve()
        if str(self.ensemble_root) not in sys.path:
            sys.path.insert(0, str(self.ensemble_root))
        import synthclass.core as core  # noqa
        import synthclass.supervised as sup  # noqa
        self._core = core
        self._sup = sup
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            try:
                self._ckpt = sup.load_leaf_selector_checkpoint(Path(self.checkpoint_path))
                self.portfolio = self._ckpt["metadata"].portfolio
            except Exception as e:  # torch missing / bad ckpt -> fall back to heuristic
                print(f"[oracle] checkpoint load failed ({e}); using heuristic technique choice")
                self._ckpt = None

    # ----------------------------------------------------- technique selection
    def select_technique(self, matches: Sequence[MatchSpec]) -> str:
        """Which classifier structure the ensemble deploys for these rules.
        Uses the trained supervised selector if available, else a heuristic."""
        pat = self._dominant_pattern(matches)
        if pat in self._tech_cache:
            return self._tech_cache[pat]
        tech = self._select_via_model(matches) if self._ckpt else self._select_heuristic(pat)
        self._tech_cache[pat] = tech
        return tech

    def _dominant_pattern(self, matches: Sequence[MatchSpec]) -> Tuple[str, ...]:
        if not matches:
            return ("wild",) * 5
        # most common per-field kind across the rule set
        from collections import Counter
        cols = list(zip(*[m.wildcard_pattern() for m in matches]))
        return tuple(Counter(c).most_common(1)[0][0] for c in cols)

    def _select_via_model(self, matches: Sequence[MatchSpec]) -> str:
        core, sup = self._core, self._sup
        rules = [self._to_synth_rule(i, m) for i, m in enumerate(matches)]
        packets = [tuple(r.ranges[f][0] for f in range(5)) for r in rules]
        leaf = sup.LeafSubset("sw", "leaf", 0, rules, packets, tuple(range(5)), (), None)
        fv = sup.extract_leaf_feature_vector(leaf, root_rule_count=len(rules),
                                             root_packet_count=max(1, len(packets)))
        meta = self._ckpt["metadata"]
        tech, _ = sup._predict_best_technique(
            self._ckpt["model"], fv, self._ckpt["feature_mean"], self._ckpt["feature_std"],
            meta.portfolio, meta.alpha, meta.beta, meta.gamma)
        return tech

    @staticmethod
    def _select_heuristic(pattern: Tuple[str, ...]) -> str:
        # tuple-space friendly when prefix-heavy; cuts when range-heavy
        if pattern.count("range") >= 2:
            return "HyperCuts"
        if pattern.count("exact") >= 3:
            return "PriorityTuple"
        return "PartitionSort"

    def _to_synth_rule(self, i: int, m: MatchSpec):
        core = self._core
        prefix_lengths = []
        for f, (lo, hi) in enumerate(m.ranges):
            bits = FIELD_BITS[f]
            prefix_lengths.append(bits if lo == hi else 0)
        return core.Rule(i, 1000 - i, tuple(m.ranges), tuple(prefix_lengths), "r")

    # ------------------------------------------------------------- benchmarking
    def benchmark(self, matches: Sequence[MatchSpec], technique: str) -> dict:
        """Measure real (construction_ms, classification_s, avg_queries, bytes)
        for `technique` on this rule set via ./main. Cached by (technique, pattern)."""
        pat = self._dominant_pattern(matches)
        key = (technique, "|".join(pat))
        if key in self._bench_cache:
            return self._bench_cache[key]
        result = self._run_main(matches, technique)
        self._bench_cache[key] = result
        return result

    def _run_main(self, matches: Sequence[MatchSpec], technique: str) -> dict:
        main = self.ensemble_root / "main"
        if not main.exists():
            return {"construction_ms": 1.0, "classification_s": 1e-6,
                    "avg_queries": 1.0, "bytes": 1000.0, "measured": False}
        with tempfile.NamedTemporaryFile("w", suffix=".rules", delete=False,
                                         dir="/tmp") as fh:
            for m in matches:
                fh.write(self._rule_line(m))
            rules_path = fh.name
        out_csv = rules_path + ".csv"
        try:
            subprocess.run(
                [str(main), f"f={rules_path}", f"technique={technique}",
                 "m=Classification", f"PacketCount={self.packet_count}",
                 f"AccuracyPackets={self.accuracy_packets}", f"o={out_csv}"],
                cwd=str(self.ensemble_root), capture_output=True, timeout=120, check=False)
            row = self._parse_csv(out_csv, technique)
        except Exception as e:
            row = {"construction_ms": 1.0, "classification_s": 1e-6,
                   "avg_queries": 1.0, "bytes": 1000.0, "measured": False, "error": str(e)}
        finally:
            for p in (rules_path, out_csv):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        return row

    @staticmethod
    def _parse_csv(path: str, technique: str) -> dict:
        """Tolerant parse: external backends (TabTree, NPTree, ...) omit some
        columns, so default the missing ones and require only the timing fields."""
        def num(row, key, default):
            v = row.get(key, "")
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
        try:
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    ctime = num(row, "ConstructionTime(ms)", None)
                    clat = num(row, "ClassificationTime(s)", None)
                    if ctime is None or clat is None:
                        continue
                    return {
                        "construction_ms": ctime,
                        "classification_s": clat,
                        "avg_queries": num(row, "AvgQueries", 1.0),
                        "bytes": num(row, "Size(bytes)", 0.0),
                        "measured": True,
                    }
        except Exception:
            pass
        return {"construction_ms": 1.0, "classification_s": 1e-6,
                "avg_queries": 1.0, "bytes": 1000.0, "measured": False}

    def _rule_line(self, m: MatchSpec) -> str:
        """Emit a rule in the ClassBench file format the ./main reader expects:
        @sip/len  dip/len  sp0 : sp1  dp0 : dp1  proto/mask"""
        (sl, sh), (dl, dh), (spl, sph), (dpl, dph), (pl, ph) = m.ranges
        def ip(lo, hi):
            plen = 32 if lo == hi else self._prefix_len_for(lo, hi, 32)
            a = (lo >> 24) & 255, (lo >> 16) & 255, (lo >> 8) & 255, lo & 255
            return f"{a[0]}.{a[1]}.{a[2]}.{a[3]}/{plen}"
        proto = "0x11/0xFF" if pl == ph else "0x00/0x00"
        return f"@{ip(sl, sh)}\t{ip(dl, dh)}\t{spl} : {sph}\t{dpl} : {dph}\t{proto}\n"

    @staticmethod
    def _prefix_len_for(lo: int, hi: int, bits: int) -> int:
        span = hi - lo + 1
        plen = bits
        s = 1
        while s < span:
            s <<= 1
            plen -= 1
        return max(0, plen)

    # -------------------------------------------------------------- calibration
    def calibrate(self, base: Optional[ClassifierModel] = None,
                  granularities: Sequence[str] = ("exact", "prefix", "range"),
                  verbose: bool = True) -> Tuple[ClassifierModel, dict]:
        """Measure real classifier costs and return a ClassifierModel whose
        memory footprint / kappa_new / kappa_ins reflect them, plus a report dict
        (JSON-serializable; see `save_calibration` / `load_calibration`)."""
        report: Dict[str, dict] = {}
        for gran in granularities:
            matches = [self._rep_match(gran, i) for i in range(self.rep_rules)]
            exp = int(round(sum(m.expansion() for m in matches) / len(matches)))
            tech = self.select_technique(matches)
            bench = self.benchmark(matches, tech)
            # "mask" backend: expand every rule into its masked entries and store
            # them in OVS-style tuple-space search (the Open vSwitch classifier).
            masked = [e for m in matches for e in m.to_masked()]
            mask_bench = self._run_main(masked, self.mask_technique)
            report[gran] = {
                "expansion": max(1, exp), "technique": tech,
                "construction_ms": bench["construction_ms"],
                "classification_s": bench["classification_s"],
                "avg_queries": bench["avg_queries"],
                "native_bytes": bench["bytes"],
                "native_bytes_per_rule": bench["bytes"] / len(matches),
                "mask_entries": len(masked),
                "mask_bytes": mask_bench["bytes"],
                "mask_bytes_per_rule": mask_bench["bytes"] / len(matches),
                "rules": len(matches),
                "measured": bool(bench.get("measured") and mask_bench.get("measured")),
            }
        model = apply_calibration(report, base, packet_count=self.packet_count)
        if verbose:
            print_calibration(report, model)
        return model, report

    def memory_profile(self, techniques: Sequence[str],
                       granularities: Sequence[str] = ("exact", "prefix", "range"),
                       n_rules: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Measured bytes per rule for each (technique, granularity), on the
        unexpanded rules. Shows where memory heterogeneity actually comes from in
        range-native software classifiers: the structure, not the match shape."""
        n = n_rules or self.rep_rules
        out: Dict[str, Dict[str, float]] = {}
        for tech in techniques:
            out[tech] = {}
            for gran in granularities:
                matches = [self._rep_match(gran, i) for i in range(n)]
                out[tech][gran] = self._run_main(matches, tech)["bytes"] / n
        return out

    def _rep_match(self, granularity: str, i: int) -> MatchSpec:
        """A representative 5-tuple for a granularity class (deterministic)."""
        s = (i * 2654435761) & 0xFFFFFFFF
        d = (i * 40503 + 12345) & 0xFFFFFFFF
        if granularity == "exact":
            return MatchSpec(((s, s), (d, d), (1024 + i, 1024 + i),
                              (80, 80), (6, 6)))
        if granularity == "prefix":
            sp = s & 0xFFFFFF00  # /24
            dp = d & 0xFFFF0000  # /16
            return MatchSpec(((sp, sp | 0xFF), (dp, dp | 0xFFFF),
                              (0, 65535), (0, 65535), (6, 6)))
        # range: non-prefix-aligned port ranges -> real expansion
        lo1, hi1 = 1000 + i, 1000 + i + 700
        lo2, hi2 = 2000, 2000 + 1500
        return MatchSpec(((s & 0xFFFFFF00, (s & 0xFFFFFF00) | 0xFF),
                          (d & 0xFFFF0000, (d & 0xFFFF0000) | 0xFFFF),
                          (lo1, hi1), (lo2, hi2), (0, 255)))


# ------------------------------------------------------------ calibration I/O
def apply_calibration(report: Dict[str, dict], base: Optional[ClassifierModel] = None,
                      packet_count: int = 2000) -> ClassifierModel:
    """Fill a ClassifierModel from a measured oracle report."""
    model = base or ClassifierModel()
    build_by_gran = {g: r["construction_ms"] for g, r in report.items()}
    # normalize measured build cost into kappa_new (per shape open = a rebuild)
    max_build = max(build_by_gran.values()) or 1.0
    mean_build = (sum(build_by_gran.values()) / len(build_by_gran)) or 1.0
    model.kappa_new = 20.0 * mean_build / max(0.01, max_build) + 8.0
    model.kappa_ins = 1.0
    # measured per-rule memory footprint m_k (bytes) under each datapath backend
    model.footprint_table = {
        "mask": {g: r["mask_bytes_per_rule"] for g, r in report.items()},
        "native": {g: r["native_bytes_per_rule"] for g, r in report.items()},
    }
    model.expansion_table = {g: r["expansion"] for g, r in report.items()}
    model.technique_table = {g: r["technique"] for g, r in report.items()}
    model.lookup_ns_table = {g: r["classification_s"] / max(1, packet_count) * 1e9
                             for g, r in report.items()}
    # per-granularity shape-open (rebuild) cost, from measured build time.
    model.kappa_by_gran = {g: 8.0 + 12.0 * (b / mean_build)
                           for g, b in build_by_gran.items()}
    return model


def print_calibration(report: Dict[str, dict], model: ClassifierModel) -> None:
    print("[oracle] calibrated classifier costs from real ./main benchmarks:")
    for gran, r in report.items():
        tag = "measured" if r["measured"] else "FALLBACK"
        print(f"    {gran:7s} e_k={r['expansion']:4d}  tech={r['technique']:13s} "
              f"m_k native={r['native_bytes_per_rule']:7.1f}B  "
              f"mask={r['mask_bytes_per_rule']:8.1f}B  "
              f"build={r['construction_ms']:.3f}ms  [{tag}]")
    print(f"    -> kappa_new={model.kappa_new:.2f}  kappa_ins={model.kappa_ins:.2f}")


def save_calibration(report: Dict[str, dict], path: Path, extra: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"granularities": report, **(extra or {})}, fh, indent=1)


def load_calibration(path: Path, base: Optional[ClassifierModel] = None) -> ClassifierModel:
    """Rebuild a calibrated ClassifierModel from a saved oracle report."""
    with open(path) as fh:
        data = json.load(fh)
    return apply_calibration(data["granularities"], base)
