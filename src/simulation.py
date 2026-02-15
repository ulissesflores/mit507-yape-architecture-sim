"""
Discrete-event simulation: monolith vs. cell-based write-path architectures.

This module implements a SimPy-based discrete-event simulation (DES) comparing two
write-path architectures for high-throughput financial transaction processing:

Architecture A — Monolith (synchronous, shared-resource)
    A single contended ``simpy.Resource`` models a centralized database connection
    pool.  Transactions block while waiting for a free slot, exposing end-users to
    queueing delay that grows non-linearly under saturation.

Architecture B — Cell-based (asynchronous, sharded bulkheads)
    Transactions are routed to independent cells (shards) and acknowledged via an
    event-bus handoff.  User-visible latency is dominated by the ingress ACK, while
    backend persistence is completed asynchronously by cell-local workers.

Design rationale
----------------
All experiment-tunable parameters are externalized into :class:`ExperimentConfig`,
a frozen dataclass.  The simulation engine (:func:`run_simulation`) accepts this
configuration object, enabling callers — notebooks, scripts, or CI pipelines — to
define and version experimental conditions independently of the simulation code.

Semantic conventions
--------------------
- ``wait_time``: user-observed waiting time (queueing delay in the synchronous path;
  ingress ACK latency in the asynchronous path).
- ``total_time``: wall-clock from arrival to user acknowledgement.  For the
  asynchronous path this does **not** include backend completion unless the model is
  extended to track end-to-end events.
- All time values are in **milliseconds** (ms).

References
----------
.. [1] Law, A. M. (2015). *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill.
.. [2] Simpy documentation — https://simpy.readthedocs.io/

License: Apache 2.0
Author:  Carlos Ulisses Flores
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Final, Generator, List, Optional, Sequence

import numpy as np
import pandas as pd
import simpy

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Valid architecture mode identifiers.
VALID_MODES: Final[tuple[str, ...]] = ("MONOLITH", "CELL_BASED")

#: Valid traffic-profile identifiers.
VALID_TRAFFIC_TYPES: Final[tuple[str, ...]] = ("NORMAL", "STRESS")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable container for all experiment-tunable simulation parameters.

    Frozen to prevent accidental in-flight mutation and to make configuration
    objects safely hashable — useful for keying result caches or provenance
    records.

    Parameters
    ----------
    mode : str
        Architecture under test (``"MONOLITH"`` | ``"CELL_BASED"``).
    traffic_type : str
        Traffic profile selector (``"NORMAL"`` | ``"STRESS"``).
    random_seed : int
        Seed for both ``random`` and ``numpy.random``, ensuring bitwise
        reproducibility across runs.
    simulation_time : int
        Simulation horizon in milliseconds.
    arrival_rate_normal : int
        Poisson arrival rate (transactions / second) under normal traffic.
    arrival_rate_stress : int
        Poisson arrival rate (transactions / second) under stress traffic.
    event_bus_ack_ms : float
        Baseline ingress acknowledgement latency (ms) for the asynchronous
        event-bus handoff in the cell-based architecture.

    event_bus_ack_jitter_lognorm_sigma : float
        Multiplicative jitter on the ACK path, modeled as a log-normal
        factor with mean 1.0. Set to 0.0 to disable.

    event_bus_ack_additive_std_ms : float
        Additive (zero-mean) Gaussian jitter (ms) on top of the
        multiplicative term. Set to 0.0 to disable.

    event_bus_ack_backlog_scale_ms : float
        Linear sensitivity (ms per queued request) that increases the ACK
        latency as the target cell backlog grows (proxying broker/backpressure
        effects). Set to 0.0 to disable.

    event_bus_ack_tail_prob : float
        Probability of a rare tail event on the ACK path (0–1).

    event_bus_ack_tail_lognorm_mu : float
        Location (μ) of the log-normal tail spike (in log-ms). Interpreted as
        the mean of the underlying normal distribution.

    event_bus_ack_tail_lognorm_sigma : float
        Scale (σ) of the log-normal tail spike (in log-ms). Must be > 0 when
        ``event_bus_ack_tail_prob`` > 0.

    event_bus_ack_floor_ms : float
        Strictly positive lower bound applied to the sampled ACK latency to
        prevent non-physical negative/zero values.
    num_cells : int
        Number of independent cell resources (shards) in the cell-based
        architecture.
    db_pool_capacity : int
        Connection-pool size for the monolith database resource.
    cell_capacity : int
        Per-cell worker capacity in the cell-based architecture.
    contention_queue_threshold : int
        Queue depth at which the monolith applies a contention penalty
        multiplier to service time.
    contention_penalty_factor : float
        Multiplicative degradation factor applied to service time when the
        monolith queue exceeds ``contention_queue_threshold``.
    latency_log_mean_ms : float
        Location parameter (mean of the underlying normal) for the
        log-normal service-time generator, in milliseconds.
    latency_log_sigma : float
        Scale parameter (std-dev of the underlying normal) for the
        log-normal service-time generator.
    shard_id_upper_bound : int
        Upper bound (exclusive) for the uniform random shard-id assigned
        to each arriving transaction.

    Examples
    --------
    >>> cfg = ExperimentConfig(mode="CELL_BASED", traffic_type="STRESS")
    >>> cfg.effective_arrival_rate
    500
    """

    # --- Architecture & traffic profile ---
    mode: str = "MONOLITH"
    traffic_type: str = "NORMAL"

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Simulation horizon ---
    simulation_time: int = 1_000  # ms

    # --- Arrival rates (transactions / second) ---
    arrival_rate_normal: int = 50
    arrival_rate_stress: int = 500

    # --- Async path (ACK latency model) ---
    event_bus_ack_ms: float = 2.0

    # Optional jitter/tail model for the *user-visible* ACK in CELL_BASED mode.
    # Defaults preserve the prior behavior (deterministic ACK = event_bus_ack_ms).
    #
    # The model is a lightweight mixture commonly used to approximate empirical
    # latency distributions in production systems:
    #   ack_ms = base * LogNormal(μ=-½σ², σ=event_bus_ack_jitter_lognorm_sigma)
    #          + Normal(0, event_bus_ack_additive_std_ms)
    #          + queue_size * event_bus_ack_backlog_scale_ms
    #          + TailSpike   (with prob = event_bus_ack_tail_prob)
    #
    # Where TailSpike ~ LogNormal(μ=event_bus_ack_tail_lognorm_mu,
    #                            σ=event_bus_ack_tail_lognorm_sigma).
    event_bus_ack_jitter_lognorm_sigma: float = 0.0
    event_bus_ack_additive_std_ms: float = 0.0
    event_bus_ack_backlog_scale_ms: float = 0.0
    event_bus_ack_tail_prob: float = 0.0
    event_bus_ack_tail_lognorm_mu: float = 0.0
    event_bus_ack_tail_lognorm_sigma: float = 0.0
    event_bus_ack_floor_ms: float = 0.1

    # --- Resource sizing ---
    num_cells: int = 10
    db_pool_capacity: int = 50
    cell_capacity: int = 10

    # --- Contention model ---
    contention_queue_threshold: int = 20
    contention_penalty_factor: float = 5.0

    # --- Service-time distribution ---
    latency_log_mean_ms: float = 50.0
    latency_log_sigma: float = 0.5

    # --- Shard space ---
    shard_id_upper_bound: int = 10_000

    def __post_init__(self) -> None:
        """Validate invariants at construction time."""
        if self.mode.upper() not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Expected one of {VALID_MODES}."
            )
        if self.traffic_type.upper() not in VALID_TRAFFIC_TYPES:
            raise ValueError(
                f"Invalid traffic_type '{self.traffic_type}'. "
                f"Expected one of {VALID_TRAFFIC_TYPES}."
            )
        # --- ACK latency model invariants ---
        if self.event_bus_ack_jitter_lognorm_sigma < 0:
            raise ValueError("event_bus_ack_jitter_lognorm_sigma must be >= 0.")
        if self.event_bus_ack_additive_std_ms < 0:
            raise ValueError("event_bus_ack_additive_std_ms must be >= 0.")
        if self.event_bus_ack_backlog_scale_ms < 0:
            raise ValueError("event_bus_ack_backlog_scale_ms must be >= 0.")
        if not (0.0 <= self.event_bus_ack_tail_prob <= 1.0):
            raise ValueError("event_bus_ack_tail_prob must be within [0, 1].")
        if self.event_bus_ack_tail_prob > 0 and self.event_bus_ack_tail_lognorm_sigma <= 0:
            raise ValueError(
                "event_bus_ack_tail_lognorm_sigma must be > 0 when event_bus_ack_tail_prob > 0."
            )
        if self.event_bus_ack_floor_ms <= 0:
            raise ValueError("event_bus_ack_floor_ms must be > 0.")
        # Normalize casing via object.__setattr__ (frozen dataclass).
        object.__setattr__(self, "mode", self.mode.upper())
        object.__setattr__(self, "traffic_type", self.traffic_type.upper())

    @property
    def effective_arrival_rate(self) -> int:
        """Resolve the Poisson arrival rate for the active traffic profile."""
        return (
            self.arrival_rate_stress
            if self.traffic_type == "STRESS"
            else self.arrival_rate_normal
        )

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a plain dictionary (JSON-safe for provenance logs)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transaction:
    """Immutable record for a single transaction entering the system.

    Attributes
    ----------
    id : int
        Monotonically increasing transaction identifier.
    arrival_time : float
        Simulation clock value at which the transaction was generated (ms).
    user_shard_id : int
        Uniformly distributed shard key used for cell routing.
    """

    id: int
    arrival_time: float
    user_shard_id: int


# Type alias for a single log row.
LogRow = Dict[str, float | int | str]


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class BankingArchitecture:
    """Simulation model encapsulating resource topology and logging.

    Parameters
    ----------
    env : simpy.Environment
        SimPy discrete-event environment (simulation clock + scheduler).
    config : ExperimentConfig
        Fully specified experiment configuration.
    """

    def __init__(self, env: simpy.Environment, config: ExperimentConfig) -> None:
        self.env: Final[simpy.Environment] = env
        self.config: Final[ExperimentConfig] = config
        self.logs: List[LogRow] = []

        if config.mode == "MONOLITH":
            self.database: Optional[simpy.Resource] = simpy.Resource(
                env, capacity=config.db_pool_capacity
            )
            self.cells: Optional[List[simpy.Resource]] = None

        elif config.mode == "CELL_BASED":
            self.database = None
            self.cells = [
                simpy.Resource(env, capacity=config.cell_capacity)
                for _ in range(config.num_cells)
            ]
        # Validation already handled by ExperimentConfig.__post_init__.

    # -- Service-time generator ------------------------------------------------

    def _lognormal_latency_ms(self) -> float:
        """Sample a service-time from a log-normal distribution.

        The log-normal is a standard empirical fit for I/O and network
        latency in distributed systems, arising from the multiplicative
        composition of independent delay factors [1]_.

        Returns
        -------
        float
            Sampled latency in milliseconds (strictly positive).

        References
        ----------
        .. [1] Cidon, A., et al. (2015). *Tiered Replication*. USENIX ATC.
        """
        return float(
            np.random.lognormal(
                mean=np.log(self.config.latency_log_mean_ms),
                sigma=self.config.latency_log_sigma,
            )
        )

    # -- ACK latency model ------------------------------------------------------

    def _sample_event_bus_ack_ms(self, *, queue_size: int) -> float:
        """Sample a user-visible event-bus ACK latency (milliseconds).

        Motivation
        ----------
        In the cell-based architecture, end-users observe an ingress ACK rather
        than the full persistence path. In practice, this ACK is *not*
        deterministic: it inherits variability from network jitter, broker
        enqueueing, transient CPU contention, and occasional tail events (e.g.,
        retries, GC pauses, bufferbloat).

        This simulator represents ACK latency via a compact mixture model that
        preserves reproducibility (seeded RNG) while producing distributions
        suitable for manuscript-quality risk/variance analysis.

        Parameters
        ----------
        queue_size : int
            Snapshot of the target cell backlog (proxy for downstream pressure).

        Returns
        -------
        float
            Sampled ACK latency in milliseconds (strictly positive).
        """
        base = float(self.config.event_bus_ack_ms)

        # Multiplicative jitter with mean 1.0 (μ=-½σ² yields E[LogNormal]=1).
        sigma = float(self.config.event_bus_ack_jitter_lognorm_sigma)
        if sigma > 0:
            mu = -0.5 * sigma * sigma
            mult = float(np.random.lognormal(mean=mu, sigma=sigma))
        else:
            mult = 1.0

        # Additive micro-jitter.
        add_std = float(self.config.event_bus_ack_additive_std_ms)
        additive = float(np.random.normal(loc=0.0, scale=add_std)) if add_std > 0 else 0.0

        # Backlog sensitivity (proxying broker/backpressure effects).
        backlog = float(queue_size) * float(self.config.event_bus_ack_backlog_scale_ms)

        ack_ms = base * mult + additive + backlog

        # Rare tail events (heavy-tailed spikes).
        p_tail = float(self.config.event_bus_ack_tail_prob)
        if p_tail > 0 and float(np.random.random()) < p_tail:
            ack_ms += float(
                np.random.lognormal(
                    mean=float(self.config.event_bus_ack_tail_lognorm_mu),
                    sigma=float(self.config.event_bus_ack_tail_lognorm_sigma),
                )
            )

        # Physical lower bound.
        return max(float(self.config.event_bus_ack_floor_ms), ack_ms)


    # -- Transaction processing ------------------------------------------------

    def process_transaction(self, tx: Transaction) -> Generator:
        """SimPy process: route and log a single transaction.

        Parameters
        ----------
        tx : Transaction
            The arriving transaction to process.

        Yields
        ------
        simpy.events.Event
            SimPy scheduling primitives (resource requests, timeouts).
        """
        start_wait: float = self.env.now

        if self.config.mode == "MONOLITH":
            yield from self._process_monolith(tx, start_wait)
        else:
            yield from self._process_cell_based(tx, start_wait)

    def _process_monolith(
        self, tx: Transaction, start_wait: float
    ) -> Generator:
        """Synchronous write-path through a shared database resource."""
        assert self.database is not None  # noqa: S101

        with self.database.request() as request:
            yield request  # Block until a connection-pool slot is available.

            wait_time: float = self.env.now - start_wait
            service_time: float = self._lognormal_latency_ms()

            # Contention penalty: degrade service once queue exceeds threshold.
            if len(self.database.queue) > self.config.contention_queue_threshold:
                service_time *= self.config.contention_penalty_factor

            yield self.env.timeout(service_time)

            self.logs.append(
                {
                    "tx_id": tx.id,
                    "architecture": self.config.mode,
                    "arrival_time": float(tx.arrival_time),
                    "wait_time": float(wait_time),
                    "service_time": float(service_time),
                    "total_time": float(self.env.now - tx.arrival_time),
                    "queue_size": len(self.database.queue),
                    "cell_id": -1,
                }
            )

    def _process_cell_based(
        self, tx: Transaction, start_wait: float
    ) -> Generator:
        """Asynchronous write-path through sharded cell bulkheads."""
        assert self.cells is not None  # noqa: S101

        cell_id: int = tx.user_shard_id % len(self.cells)
        target_cell: simpy.Resource = self.cells[cell_id]

        # User-visible path: event-bus acknowledgement.
        # The backlog snapshot is taken *before* the ACK wait to avoid feedback
        # from the ACK itself.
        backlog_snapshot = len(target_cell.queue)
        ack_ms = self._sample_event_bus_ack_ms(queue_size=backlog_snapshot)

        yield self.env.timeout(ack_ms)
        wait_time: float = self.env.now - start_wait

        # Fork asynchronous backend work (fire-and-forget from user's perspective).
        self.env.process(self._cell_worker(target_cell))

        self.logs.append(
            {
                "tx_id": tx.id,
                "architecture": self.config.mode,
                "arrival_time": float(tx.arrival_time),
                "wait_time": float(wait_time),
                "service_time": float(ack_ms),
                "total_time": float(self.env.now - tx.arrival_time),
                "queue_size": int(backlog_snapshot),
                "cell_id": cell_id,
            }
        )

    def _cell_worker(self, cell_resource: simpy.Resource) -> Generator:
        """Background worker draining a single cell shard.

        Models isolated consumption within each shard.  The user
        acknowledgement is already complete before this coroutine runs.

        Parameters
        ----------
        cell_resource : simpy.Resource
            The cell-local resource to acquire for processing.
        """
        with cell_resource.request() as req:
            yield req
            yield self.env.timeout(self._lognormal_latency_ms())


# ---------------------------------------------------------------------------
# Arrival process
# ---------------------------------------------------------------------------

def _transaction_generator(
    env: simpy.Environment,
    system: BankingArchitecture,
    config: ExperimentConfig,
) -> Generator:
    """Poisson arrival process (inter-arrival times ~ Exp(λ)).

    Parameters
    ----------
    env : simpy.Environment
        Active simulation environment.
    system : BankingArchitecture
        Architecture model receiving generated transactions.
    config : ExperimentConfig
        Experiment parameters (arrival rate, shard space).

    Yields
    ------
    simpy.events.Event
        Inter-arrival timeouts.
    """
    tx_id: int = 0
    rate_tps: int = config.effective_arrival_rate

    while True:
        # Inter-arrival time: Exp(λ) with λ = rate_tps / 1000 (ms⁻¹).
        yield env.timeout(random.expovariate(rate_tps / 1_000.0))

        tx_id += 1
        shard_id: int = random.randint(0, config.shard_id_upper_bound - 1)

        tx = Transaction(id=tx_id, arrival_time=env.now, user_shard_id=shard_id)
        env.process(system.process_transaction(tx))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    config: Optional[ExperimentConfig] = None,
    *,
    verbose: bool = False,
    **overrides: object,
) -> pd.DataFrame:
    """Execute a single simulation run and return row-level transaction logs.

    Parameters
    ----------
    config : ExperimentConfig, optional
        Pre-built configuration object.  If ``None``, a default
        :class:`ExperimentConfig` is constructed, optionally modified by
        ``**overrides``.
    verbose : bool
        If ``True``, print a run-header to stdout.
    **overrides
        Keyword arguments forwarded to :class:`ExperimentConfig` when
        ``config`` is ``None``.  Ignored if ``config`` is provided.

    Returns
    -------
    pd.DataFrame
        Row-level transaction logs with columns: ``tx_id``,
        ``architecture``, ``arrival_time``, ``wait_time``,
        ``service_time``, ``total_time``, ``queue_size``, ``cell_id``.

    Examples
    --------
    >>> # Use defaults
    >>> df = run_simulation()

    >>> # Override via keyword arguments
    >>> df = run_simulation(mode="CELL_BASED", traffic_type="STRESS")

    >>> # Provide a full config object
    >>> cfg = ExperimentConfig(mode="MONOLITH", traffic_type="STRESS", random_seed=7)
    >>> df = run_simulation(config=cfg, verbose=True)
    """
    if config is None:
        config = ExperimentConfig(**overrides)  # type: ignore[arg-type]

    # Seed both RNGs for bitwise reproducibility.
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    env = simpy.Environment()
    system = BankingArchitecture(env, config=config)

    env.process(_transaction_generator(env, system, config))

    if verbose:
        print(
            f"[simulation] mode={config.mode}  traffic={config.traffic_type}  "
            f"rate={config.effective_arrival_rate} tps  "
            f"horizon={config.simulation_time} ms  "
            f"seed={config.random_seed}"
        )

    env.run(until=config.simulation_time)
    return pd.DataFrame(system.logs)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for command-line invocation."""
    parser = argparse.ArgumentParser(
        description="Run a discrete-event simulation comparing monolith vs. cell-based architectures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    defaults = ExperimentConfig()

    parser.add_argument("--mode", choices=["MONOLITH", "CELL_BASED"], default=defaults.mode)
    parser.add_argument("--traffic-type", choices=["NORMAL", "STRESS"], default=defaults.traffic_type)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--simulation-time", type=int, default=defaults.simulation_time)
    parser.add_argument("--arrival-rate-normal", type=int, default=defaults.arrival_rate_normal)
    parser.add_argument("--arrival-rate-stress", type=int, default=defaults.arrival_rate_stress)
    parser.add_argument("--event-bus-ack-ms", type=float, default=defaults.event_bus_ack_ms)
    parser.add_argument("--num-cells", type=int, default=defaults.num_cells)
    parser.add_argument("--db-pool-capacity", type=int, default=defaults.db_pool_capacity)
    parser.add_argument("--cell-capacity", type=int, default=defaults.cell_capacity)
    parser.add_argument("--contention-queue-threshold", type=int, default=defaults.contention_queue_threshold)
    parser.add_argument("--contention-penalty-factor", type=float, default=defaults.contention_penalty_factor)
    parser.add_argument("--latency-log-mean-ms", type=float, default=defaults.latency_log_mean_ms)
    parser.add_argument("--latency-log-sigma", type=float, default=defaults.latency_log_sigma)
    parser.add_argument("--shard-id-upper-bound", type=int, default=defaults.shard_id_upper_bound)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to write CSV output.  Omit to print to stdout.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry-point for standalone or scripted invocation.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument list; defaults to ``sys.argv[1:]``.
    """
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    # Map CLI kebab-case to dataclass snake_case.
    config = ExperimentConfig(
        mode=args.mode,
        traffic_type=args.traffic_type,
        random_seed=args.random_seed,
        simulation_time=args.simulation_time,
        arrival_rate_normal=args.arrival_rate_normal,
        arrival_rate_stress=args.arrival_rate_stress,
        event_bus_ack_ms=args.event_bus_ack_ms,
        num_cells=args.num_cells,
        db_pool_capacity=args.db_pool_capacity,
        cell_capacity=args.cell_capacity,
        contention_queue_threshold=args.contention_queue_threshold,
        contention_penalty_factor=args.contention_penalty_factor,
        latency_log_mean_ms=args.latency_log_mean_ms,
        latency_log_sigma=args.latency_log_sigma,
        shard_id_upper_bound=args.shard_id_upper_bound,
    )

    df = run_simulation(config=config, verbose=args.verbose)

    if args.output:
        df.to_csv(args.output, index=False)
        if args.verbose:
            print(f"[simulation] wrote {len(df)} rows -> {args.output}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
