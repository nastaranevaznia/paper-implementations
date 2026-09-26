"""Optimal server selection (paper Sec. 3.3, Eq. 3-6).

Each node i (fog or cloud) has a resource capacity vector
    R_i = {CPU_i, RAM_i, Storage_i}                                   (Eq. 3)
each task j has a demand vector
    D_j = {d_CPU, d_RAM, d_Storage}                                   (Eq. 4)
a node can host the task if every demand fits (Eq. 5), and the best
feasible node minimizes the dual-objective fitness
    Fitness(i) = alpha * (U_CPU(i)/CPU(i)) + beta * (E(i)/E_max)       (Eq. 6)
with alpha + beta = 1.  The node with the LOWEST fitness is selected.
"""

from dataclasses import dataclass


@dataclass
class Node:
    name: str
    cpu: float        # total CPU capacity (Eq. 3)
    ram: float        # total RAM capacity
    storage: float    # total storage capacity
    cpu_used: float   # currently used CPU
    energy: float     # current energy consumption E(i)


@dataclass
class Task:
    name: str
    d_cpu: float      # Eq. 4 demands
    d_ram: float
    d_storage: float


def eq5_feasible(node, task):
    """True if node i can host task j (all demands fit)."""
    return (task.d_cpu <= node.cpu - node.cpu_used
            and task.d_ram <= node.ram
            and task.d_storage <= node.storage)


def eq6_fitness(node, alpha, e_max):
    """Dual-objective fitness (lower is better)."""
    beta = 1.0 - alpha
    load_term = (node.cpu_used / node.cpu) if node.cpu else float("inf")
    energy_term = (node.energy / e_max) if e_max else float("inf")
    return alpha * load_term + beta * energy_term


def select_server(nodes, task, alpha=0.6):
    """Return (best_node, table) among feasible nodes; table lists fitness."""
    e_max = max(n.energy for n in nodes)
    table = []
    for n in nodes:
        feas = eq5_feasible(n, task)
        fit = eq6_fitness(n, alpha, e_max) if feas else None
        table.append({"node": n.name, "feasible": feas, "fitness": fit})
    feasible = [(t, n) for t, n in zip(table, nodes) if t["feasible"]]
    if not feasible:
        return None, table
    best = min(feasible, key=lambda tn: tn[0]["fitness"])[1]
    return best, table
