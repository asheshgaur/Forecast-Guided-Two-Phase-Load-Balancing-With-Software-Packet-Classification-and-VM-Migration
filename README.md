# Two-Phase Data Generator

Core code used to generate the evaluation data for *Forecast-Guided Two-Phase
Load Balancing with Software Packet Classification and VM Migration*.

## Requirements

Python 3.14 and the packages in `requirements.txt` are required. The
experiments also require a valid Gurobi license, which is not included.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate data

Run commands from this directory. Generated files are written under
`results/`.

```bash
python experiments/revised_evaluation.py --seeds 7 11 19 23 31
python experiments/load_sweep.py --workers 2
python experiments/horizon_ablation.py --workers 2
python experiments/vm_density.py
```

The archive intentionally excludes generated data, plotting code, manuscript
sources, and test utilities.
