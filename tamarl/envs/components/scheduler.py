"""Decision scheduler: identifies agents that need a routing decision."""

import torch
from tamarl.core.dnl_matsim import TorchDNLMATSim


class DecisionScheduler:
    """Identifies agents that need to choose their next edge.
    
    Decision point: agent is in capacity buffer (status=2), 
    wakeup_time <= current_step, and next_edge == -1 (not yet decided).
    """

    def __init__(self, dnl: TorchDNLMATSim):
        self.dnl = dnl

    def get_deciding_agents(self) -> torch.Tensor:
        """Return tensor of agent indices that need a routing decision now."""
        mask = (
            (self.dnl.status == 2) &                            # In capacity buffer
            (self.dnl.wakeup_time <= self.dnl.current_step) &  # Ready to move
            (self.dnl.next_edge == -1)                         # No decision yet
        )
        return torch.nonzero(mask, as_tuple=True)[0]

    def has_deciding_agents(self) -> bool:
        """Check if any agents need a decision (without allocating index tensor)."""
        mask = (
            (self.dnl.status == 2) &
            (self.dnl.wakeup_time <= self.dnl.current_step) &
            (self.dnl.next_edge == -1)
        )
        return mask.any().item()
