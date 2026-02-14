"""
Yape/BCP Architecture Simulation Module
---------------------------------------
Author: Carlos Ulisses Flores (ulissesflores.com)
License: Apache 2.0
Date: 2026-02-14

Description:
    This module simulates two banking architectural patterns using Discrete Event Simulation (DES):
    1. Monolithic/Legacy (AS-IS): Single database resource, blocking I/O, susceptible to cascading failure.
    2. Cell-Based/Event-Driven (TO-BE): Sharded resources (bulkheads), asynchronous messaging, high resilience.

    Mathematical Basis:
    - Little's Law (L = λW) is tested by varying the arrival rate (λ) and observing the wait time (W).
    - Queueing Theory (M/M/1 vs M/M/c) is applied to demonstrate throughput limits.

Dependencies:
    - simpy: For process-based discrete-event simulation.
    - numpy: For statistical distributions (latency generation).
    - pandas: For structured data logging.
"""

import simpy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict

# --- CONFIGURATION CONSTANTS (Hyperparameters) ---
RANDOM_SEED = 42
SIMULATION_TIME = 1000  # Time units (e.g., milliseconds or seconds based on scaling)
ARRIVAL_RATE_NORMAL = 50   # Normal traffic (TPS)
ARRIVAL_RATE_STRESS = 500  # Black Friday / Payday traffic (TPS)

@dataclass
class Transaction:
    """Represents a single financial transaction flowing through the system."""
    id: int
    arrival_time: float
    user_shard_id: int  # Used for routing in Cell-Based architecture

class BankingArchitecture:
    """
    The main simulation environment.
    
    Attributes:
        env (simpy.Environment): The simulation clock and scheduler.
        mode (str): 'MONOLITH' or 'CELL_BASED'.
        logs (List): Storage for simulation metrics.
    """
    
    def __init__(self, env: simpy.Environment, mode: str, num_cells: int = 10):
        self.env = env
        self.mode = mode
        self.logs = []
        
        # RESOURCE ALLOCATION
        if mode == 'MONOLITH':
            # AS-IS: One giant database (Mainframe/Single DB)
            # Capacity represents connection pool size.
            self.database = simpy.Resource(env, capacity=50) 
            
        elif mode == 'CELL_BASED':
            # TO-BE: Resources partitioned into independent 'Cells'.
            # Each cell has smaller capacity, but total capacity is higher + isolated.
            # Example: 10 cells, each with capacity 10.
            self.cells = [simpy.Resource(env, capacity=10) for _ in range(num_cells)]
            self.kafka_queue = simpy.Store(env) # Represents the Async Event Bus

    def get_latency(self) -> float:
        """
        Generates processing latency based on Log-Normal distribution.
        This simulates the 'long tail' latency often seen in distributed systems.
        """
        # Mean 50ms, Sigma 0.5 (High variance)
        return np.random.lognormal(mean=np.log(50), sigma=0.5)

    def process_transaction(self, tx: Transaction):
        """
        The core logic for handling a transaction based on architecture type.
        """
        start_wait = self.env.now
        
        if self.mode == 'MONOLITH':
            # --- SCENARIO A: SYNCHRONOUS / BLOCKING ---
            # Request connection to the single DB
            with self.database.request() as request:
                yield request  # Wait in queue (Blocking)
                
                wait_time = self.env.now - start_wait
                
                # Simulate Database Processing (Holding the connection)
                service_time = self.get_latency()
                
                # STRESS FACTOR: If queue is huge, DB degrades (Simulated Lock Contention)
                if len(self.database.queue) > 20:
                    service_time *= 5.0 # Performance degradation multiplier
                    
                yield self.env.timeout(service_time)
                
        elif self.mode == 'CELL_BASED':
            # --- SCENARIO B: ASYNCHRONOUS / SHARDED ---
            # 1. Routing: Determine which Cell owns this user (Hashing)
            cell_id = tx.user_shard_id % len(self.cells)
            target_cell = self.cells[cell_id]
            
            # 2. Async Handoff: Put in Kafka (virtually instant ack to user)
            # In a real app, this 'yield' is very short (network time to broker)
            yield self.env.timeout(2) 
            
            # 3. Worker Processing (The actual work happens in background)
            # We fork a process so the 'User' doesn't wait for the DB write
            self.env.process(self.cell_worker(target_cell, start_wait))
            
            # For the user, the transaction is "Done" (Accepted) immediately after Kafka ack.
            wait_time = self.env.now - start_wait
            service_time = 2.0 # Just the ACK time

        # Log Metrics
        total_time = self.env.now - tx.arrival_time
        self.logs.append({
            "tx_id": tx.id,
            "architecture": self.mode,
            "arrival_time": tx.arrival_time,
            "wait_time": wait_time,   # Critical Metric for Little's Law
            "service_time": service_time,
            "total_time": total_time,
            "queue_size": len(self.database.queue) if self.mode == 'MONOLITH' else 0 # Simplified for cells
        })

    def cell_worker(self, cell_resource, original_start_time):
        """Background worker that drains the queue for a specific cell."""
        with cell_resource.request() as req:
            yield req
            # Processing happens here, isolated from other cells
            proc_time = self.get_latency()
            yield self.env.timeout(proc_time)

def transaction_generator(env, system: BankingArchitecture, arrival_rate):
    """Generates traffic following a Poisson process."""
    tx_id = 0
    while True:
        # Inter-arrival time = 1 / lambda
        yield env.timeout(random.expovariate(arrival_rate / 1000.0)) # scale to ms
        
        tx_id += 1
        # Random shard ID for cell routing
        shard_id = random.randint(0, 9999)
        
        tx = Transaction(tx_id, env.now, shard_id)
        env.process(system.process_transaction(tx))

def run_simulation(mode='MONOLITH', traffic_type='NORMAL'):
    """
    Orchestrates the simulation run.
    Returns a Pandas DataFrame with results.
    """
    # Setup
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    env = simpy.Environment()
    system = BankingArchitecture(env, mode)
    
    # Determine Rate
    rate = ARRIVAL_RATE_STRESS if traffic_type == 'STRESS' else ARRIVAL_RATE_NORMAL
    
    # Start Processes
    env.process(transaction_generator(env, system, rate))
    
    # Run
    print(f"🚀 Starting Simulation: {mode} | Traffic: {traffic_type} | Rate: {rate} TPS")
    env.run(until=SIMULATION_TIME)
    
    # Export Data
    df = pd.DataFrame(system.logs)
    return df

if __name__ == "__main__":
    # Test Run
    df_mono = run_simulation('MONOLITH', 'STRESS')
    print(f"Simulation Complete. Rows generated: {len(df_mono)}")
    print(df_mono.head())