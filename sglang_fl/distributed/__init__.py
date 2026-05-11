"""Distributed communication module for sglang_fl.

Provides CommunicatorFL (FlagCX / torch.distributed) for OOT collective ops.
"""

from sglang_fl.distributed.communicator import CommunicatorFL

__all__ = ["CommunicatorFL"]
