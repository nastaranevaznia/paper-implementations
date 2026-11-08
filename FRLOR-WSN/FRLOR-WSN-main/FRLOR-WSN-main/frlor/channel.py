"""Log-normal shadowing link model (paper Eq. 2, identical to NS-2 Propagation/Shadowing).

Mean received power follows Friis at the reference distance d0 and a path-loss exponent beta:
    Pr(d) [dB rel. to Pr(d0)] = -10*beta*log10(d/d0) + X,   X ~ N(0, sigma^2)
A packet is received iff Pr(d) >= RXThresh, hence
    P_link(d) = Phi( margin(d) / sigma ),  margin(d) = 10log10(Pr(d0)/RXThresh) - 10*beta*log10(d/d0)
"""
from __future__ import annotations
import math
import numpy as np
from .config import ChannelCfg

_erf = np.vectorize(math.erf, otypes=[float])


def wavelength(c: ChannelCfg) -> float:
    return 3.0e8 / c.freq_hz


def friis_ref_power(c: ChannelCfg) -> float:
    lam = wavelength(c)
    return c.pt * c.gt * c.gr * lam ** 2 / ((4 * math.pi) ** 2 * c.d0 ** 2 * c.sys_loss)


def reference_margin_db(c: ChannelCfg) -> float:
    return 10.0 * math.log10(friis_ref_power(c) / c.rx_thresh)


def link_probability(d, c: ChannelCfg) -> np.ndarray:
    d = np.maximum(np.asarray(d, dtype=float), c.d0)
    margin = reference_margin_db(c) - 10.0 * c.path_loss_exp * np.log10(d / c.d0)
    return 0.5 * (1.0 + _erf(margin / (c.sigma_db * math.sqrt(2.0))))


def median_range(c: ChannelCfg) -> float:
    """Distance at which P_link = 0.5."""
    return c.d0 * 10 ** (reference_margin_db(c) / (10.0 * c.path_loss_exp))
