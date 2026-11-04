"""What applicants pay for their requests.

``cost(r, s)`` returns each applicant's payment given requests ``r`` and
served loads ``s``. A fee that grows with the request is what disciplines
requests under the uniform rule; a flat fee does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Fee(ABC):
    """What applicants pay: subclasses define ``cost(requests, served)``."""
    @abstractmethod
    def cost(self, requests: np.ndarray, served: np.ndarray) -> np.ndarray: ...

    def marginal_request_cost(self) -> float:
        """Cost of one more MW of request that is not served (0 for a flat fee)."""
        return 0.0


@dataclass
class PerMWCommitment(Fee):
    """Commitment ``kappa`` per MW requested, paid whether or not served."""

    kappa: float

    def cost(self, requests, served):
        """``kappa`` per MW requested, served or not."""
        return self.kappa * np.asarray(requests, dtype=float)

    def marginal_request_cost(self):
        """``kappa``."""
        return self.kappa


@dataclass
class FlatFee(Fee):
    """A study fee ``f`` per applicant, independent of the request."""

    fee: float

    def cost(self, requests, served):
        """``fee`` for each applicant with a positive request."""
        r = np.asarray(requests, dtype=float)
        return np.where(r > 0, self.fee, 0.0)


@dataclass
class SecurityDeposit(Fee):
    """A deposit ``per_mw`` per MW requested, returned except for a forfeit.

    The applicant bears the carrying cost ``carrying_rate * per_mw`` per MW
    requested and forfeits ``forfeit_rate * per_mw`` per MW requested but not
    served (as when unused capacity is released), plus an optional flat study fee.
    """

    per_mw: float
    carrying_rate: float = 0.0
    forfeit_rate: float = 0.0
    study_fee: float = 0.0

    def cost(self, requests, served):
        """Carrying cost on the request, forfeit on unserved MW, and the study fee."""
        r = np.asarray(requests, dtype=float)
        s = np.asarray(served, dtype=float)
        flat = np.where(r > 0, self.study_fee, 0.0)
        return flat + self.per_mw * (self.carrying_rate * r + self.forfeit_rate * np.maximum(r - s, 0.0))

    def marginal_request_cost(self):
        """The carrying cost per MW requested."""
        return self.per_mw * self.carrying_rate
