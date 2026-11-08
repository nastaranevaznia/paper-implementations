"""Single source of truth for every constant used in the paper (Tables 1 and 3, Sec. 3-4).

Values marked ASSUMPTION are not specified in the paper and had to be chosen; each can be
overridden from the command line, e.g.  --set topology.neighbor_min_prob=0.3
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict


@dataclass
class ChannelCfg:                      # Table 1 + Sec. 3.3 (shadowing model, Eq. 2)
    pt: float = 0.28183815             # transmit power [W]
    gt: float = 1.0
    gr: float = 1.0
    sys_loss: float = 1.0              # L
    freq_hz: float = 914e6
    rx_thresh: float = 3.652e-10       # RXThresh [W]
    path_loss_exp: float = 2.7         # beta
    sigma_db: float = 6.0              # shadowing std-dev [dB]
    d0: float = 1.0                    # NS-2 reference distance [m]


@dataclass
class TopologyCfg:
    n_nodes: int = 200                 # paper: 200 and 400
    width: float = 1200.0              # paper: 1200 x 1200
    height: float = 1200.0
    src_xy: list = field(default_factory=lambda: [50.0, 600.0])    # ASSUMPTION
    dst_xy: list = field(default_factory=lambda: [1150.0, 600.0])  # ASSUMPTION
    neighbor_min_prob: float = 0.2     # ASSUMPTION: j is a neighbour of i iff P_link(i,j) >= this
    max_resample: int = 200            # ASSUMPTION: resample until src and dst are connected


@dataclass
class FuzzyCfg:
    profile: str = "text"              # "text": MFs as printed in Sec. 3.3; "figure": read off Fig. 4/5
    rule_file: str = ""                # empty -> frlor/data/rule_base.csv
    n_points: int = 1001               # resolution of the output universe for centroid
    nn_max: float = 20.0               # universe of "number of neighbours"  [0,20]
    dist_max: float = 1200.0           # universe of "distance" [0,1200] m


@dataclass
class CandCfg:
    max_candidates: int = 3            # "Max" in Sec. 3.3 (paper sweeps 2..5)
    require_progress: bool = True      # ASSUMPTION: only neighbours closer to the destination are eligible
    force_destination_first: bool = True  # ASSUMPTION: if destination is a neighbour it has priority 1


@dataclass
class RLCfg:                           # Table 3
    episodes: int = 1000
    episode_len: int = 100
    eta: float = 0.1                   # learning rate
    gamma: float = 0.9                 # discount factor
    eps: float = 0.1
    eps_decay: float = 0.99            # per episode
    eps_min: float = 0.0
    w_dist: float = 0.5                # alpha in Eq. (4)
    w_nbr: float = 0.3                 # beta  in Eq. (4)
    w_etx: float = 0.2                 # omega in Eq. (4)  (renamed: paper also uses omega for the discount)
    goal_bonus: float = 0.0            # ASSUMPTION: paper defines no terminal reward
    dead_end_penalty: float = 0.0      # ASSUMPTION
    q_init: float = 0.0                # paper: Q initialised to 0
    conv_tol: float = 0.01             # Table 3: Q-values stable ...
    conv_window: int = 50              # ... for 50 consecutive episodes
    early_stop_window: int = 100       # Sec. 3.4: stop if no improvement of avg reward over 100 episodes
    use_conv_stop: bool = True
    use_early_stop: bool = True


@dataclass
class EvalCfg:                         # python-level opportunistic forwarding check (not NS-2)
    n_packets: int = 2000
    max_retry: int = 7
    max_hops: int = 64
    t_default: float = 0.05            # Eq. (7): T_i = (i-1) * T_default, T_default = 50 ms


@dataclass
class Config:
    seed: int = 1
    channel: ChannelCfg = field(default_factory=ChannelCfg)
    topology: TopologyCfg = field(default_factory=TopologyCfg)
    fuzzy: FuzzyCfg = field(default_factory=FuzzyCfg)
    cand: CandCfg = field(default_factory=CandCfg)
    rl: RLCfg = field(default_factory=RLCfg)
    evaluation: EvalCfg = field(default_factory=EvalCfg)

    def to_dict(self):
        return asdict(self)


def apply_overrides(cfg: Config, items) -> Config:
    """items like ['rl.eta=0.2', 'cand.max_candidates=4', 'seed=7']"""
    for it in items or []:
        key, val = it.split("=", 1)
        try:
            val = json.loads(val)
        except ValueError:
            pass
        obj, parts = cfg, key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError("unknown config key: " + key)
        setattr(obj, parts[-1], val)
    return cfg
