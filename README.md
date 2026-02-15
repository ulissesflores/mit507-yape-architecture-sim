# 🏦 Yape Architecture Simulation: Monolith vs. Cell-Based

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18645894.svg)](https://doi.org/10.5281/zenodo.18645894)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1taz-NsFko0SNLgnSogwvqj3753Duckyi)

> **Architectural validation for high-frequency FinTech systems.**

## 📋 Overview

This repository contains the scientific artifacts supporting the case study **"Yape: From Monolith to Cell-Based Architecture"**.

Developed by **Carlos Ulisses Flores** for the MIT-507 (Digital Organization) course, this project uses Discrete Event Simulation (DES) to mathematically prove why legacy banking architectures fail under "Black Friday" loads and how **Cell-Based Architectures (Bulkheads)** provide horizontal scalability with bounded user-visible latency.

## 🧪 The Experiment

Two architectural regimes are simulated under identical stress traffic
(2 000 transactions/second, seed = 42, 30-second horizon):

1.  **AS-IS (Monolithic/Legacy):** A single shared `simpy.Resource`
    (capacity = 50) models a centralized database connection pool.
    Transactions block while waiting for a free slot, exposing end-users
    to queueing delay that grows without bound when the traffic
    intensity ρ = λ/μ > 1.

2.  **TO-BE (Cell-Based/Event-Driven):** Transactions are routed to
    10 independent cells (shards) and acknowledged via an asynchronous
    event-bus handoff.  User-visible latency is dominated by the
    ingress ACK (~2 ms baseline), while backend persistence completes
    asynchronously by cell-local workers.

### ACK variability model (cell-based)

In the cell-based regime, the client-observable latency is the **ingress ACK**
(acknowledgement after asynchronous handoff).  In production systems this ACK
is rarely deterministic; it exhibits micro-jitter, load-coupled drift, and
rare tail spikes.  The simulator uses a compact mixture model:

$$
\text{ack}_{ms} = b \cdot \mathrm{LogNormal}\!\bigl(\mu\!=\!-\tfrac{1}{2}\sigma^2,\,\sigma\bigr)
+ \mathcal{N}(0,\,s)
+ q \cdot k
+ \mathbb{I}_{tail}\cdot \mathrm{LogNormal}(\mu_t,\,\sigma_t)
$$

All parameters are recorded in `experiment_provenance.json` for full
reproducibility.

### Key results

| Metric | Monolith | Cell-based |
|--------|----------|------------|
| Transactions completed | 5 311 (8.9%) | 60 458 (100.8%) |
| Median latency (p50) | 13 537 ms | 5.78 ms |
| p99 latency | 26 788 ms | 10.25 ms |
| SLA compliance (≤ 1 s) | 5.4% | 100.0% |
| Traffic intensity (ρ) | 2.0 (unstable) | N/A (async) |

> **Note on transaction counts:** the monolith completes only 8.9% of the
> ~60 000 generated transactions.  The remaining ~55 000 are trapped in the
> unbounded queue when the simulation horizon expires.  This is a correct
> consequence of ρ > 1 (unstable M/G/c queue); the reported metrics refer
> exclusively to **completed** transactions.

## 📂 Repository Structure

```text
mit507-yape-architecture-sim/
├── src/
│   ├── __init__.py
│   └── simulation.py           # SimPy DES engine + ExperimentConfig dataclass
├── notebooks/
│   └── simulation_analysis.ipynb   # Analysis notebook (also on Colab)
├── requirements.txt            # Pinned dependencies
├── CITATION.cff                # Academic citation metadata
└── LICENSE                     # Apache 2.0
```

## 🚀 Quick Start

### Option A: Cloud (Google Colab) — recommended

Click the **"Open in Colab"** badge above.  The notebook clones this
repository, installs only `simpy` (the scientific stack ships with Colab),
and runs both experiments end-to-end.

### Option B: Local execution

```bash
git clone https://github.com/ulissesflores/mit507-yape-architecture-sim.git
cd mit507-yape-architecture-sim
pip install -r requirements.txt
python src/simulation.py --mode MONOLITH --traffic-type STRESS -v
python src/simulation.py --mode CELL_BASED --traffic-type STRESS -v
```

## 📊 Figures

The notebook generates four publication-ready figures (300 dpi, pt-BR
titles):

| Figure | Description |
|--------|-------------|
| **Fig 1** | Latência ao Longo do Tempo (Colapso Monolítico) — log-y scatter |
| **Fig 2** | Distribuição Acumulada de Latência (ECDF) — log-x CDF |
| **Fig 3** | Variância Operacional (Escala Logarítmica) — violin + box |
| **Fig 4** | Vazão Efetiva (Transações Concluídas / Segundo) — zero-filled throughput |

## 📜 License

This project is licensed under the Apache License 2.0 — see the
[LICENSE](LICENSE) file for details.

## ✍️ Citation

If you use this software in your research, please cite it using the
metadata in [CITATION.cff](CITATION.cff) or the Zenodo DOI above.
