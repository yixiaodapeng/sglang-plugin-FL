"""CommunicatorFL — Full OOT communicator for SGLang plugin.

Wraps FlagCX (when available) or falls back to torch.distributed.
Created per GroupCoordinator via AROUND hook on __init__.

Adapted from vllm-plugin-FL's CommunicatorFL pattern.
"""

import logging
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class CommunicatorFL:
    """OOT communicator that routes collectives through FlagCX or torch.distributed.

    Lifecycle:
      1. Created by AROUND hook on GroupCoordinator.__init__
      2. Stored as gc.fl_communicator
      3. AROUND hooks on all_reduce/reduce_scatter/etc. delegate to this instance
    """

    disabled: bool = False

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        device_group: ProcessGroup,
        world_size: int,
        rank_in_group: int,
        ranks: List[int],
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.device_group = device_group
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.ranks = ranks

        # Determine backend
        from sglang_fl.platform import PlatformFL

        try:
            platform = PlatformFL()
            self._dist_backend = platform._dist_backend
        except Exception:
            self._dist_backend = "nccl"

        # Initialize FlagCX communicator if backend is flagcx
        self._flagcx_comm = None
        if self._dist_backend == "flagcx" and world_size > 1:
            try:
                from sglang_fl.flagcx_communicator import FlagCXCommunicator

                self._flagcx_comm = FlagCXCommunicator(
                    group=cpu_group,
                    device=device,
                )
                if not self._flagcx_comm.available:
                    logger.warning(
                        "FlagCX communicator init failed, falling back to torch.distributed"
                    )
                    self._flagcx_comm = None
            except Exception as e:
                logger.warning(
                    f"FlagCX communicator creation failed: {e}, using torch.distributed"
                )
                self._flagcx_comm = None

        backend_name = "flagcx" if self._flagcx_comm else "torch.distributed"
        logger.info(
            f"CommunicatorFL created: world_size={world_size}, "
            f"rank={rank_in_group}, backend={backend_name}"
        )

    # ─── all_reduce ──────────────────────────────────────────────────────────

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """In-place all-reduce. Returns the input tensor (modified in-place)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            out = self._flagcx_comm.all_reduce(input_)
            if out is not None:
                # FlagCX all_reduce returns a new tensor; copy back for in-place semantics
                input_.copy_(out)
                return input_
        # Fallback: torch.distributed
        dist.all_reduce(input_, group=self.device_group)
        return input_

    # ─── reduce_scatter ──────────────────────────────────────────────────────

    def reduce_scatter(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        """Reduce-scatter tensor (in-place into output)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.reduce_scatter(output, input_)
            return
        dist.reduce_scatter_tensor(output, input_, group=self.device_group)

    # ─── reduce_scatterv ─────────────────────────────────────────────────────

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Reduce-scatter with variable sizes per rank."""
        world_size = self.world_size

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_.shape[0] % world_size == 0
            chunk_size = input_.shape[0] // world_size

        output_shape = (chunk_size,) + input_.shape[1:]
        if output is None:
            output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
        else:
            assert output.shape == output_shape

        if self._flagcx_comm and not self._flagcx_comm.disabled:
            if sizes is not None and hasattr(self._flagcx_comm, "reduce_scatterv"):
                self._flagcx_comm.reduce_scatterv(output, input_, sizes=sizes)
            else:
                self._flagcx_comm.reduce_scatter(output, input_)
            return output

        # Fallback: torch.distributed (only supports equal sizes)
        dist.reduce_scatter_tensor(output, input_, group=self.device_group)
        return output

    # ─── all_gather ──────────────────────────────────────────────────────────

    def all_gather(self, output: torch.Tensor, input_: torch.Tensor) -> None:
        """All-gather into tensor (in-place into output)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.all_gather(output, input_)
            return
        dist.all_gather_into_tensor(output, input_, group=self.device_group)

    # ─── all_gatherv ─────────────────────────────────────────────────────────

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, List[torch.Tensor]],
        sizes: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """All-gather with variable sizes per rank."""
        world_size = self.world_size

        def _all_gather_single(inp: torch.Tensor, sizes: Optional[List[int]]):
            input_size = inp.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert inp.shape[0] == sizes[self.rank_in_group]
                output_size = (sum(sizes),) + input_size[1:]
                # If all sizes equal, treat as uniform
                if all(s == sizes[0] for s in sizes):
                    sizes = None
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]

            output_tensor = torch.empty(output_size, dtype=inp.dtype, device=inp.device)

            if self._flagcx_comm and not self._flagcx_comm.disabled:
                if sizes is not None and hasattr(self._flagcx_comm, "all_gatherv"):
                    self._flagcx_comm.all_gatherv(output_tensor, inp, sizes=sizes)
                else:
                    self._flagcx_comm.all_gather(output_tensor, inp)
            else:
                dist.all_gather_into_tensor(output_tensor, inp, group=self.device_group)

            return output_tensor

        if isinstance(input_, torch.Tensor):
            input_ = [input_]

        if self._flagcx_comm and not self._flagcx_comm.disabled:
            output_list = []
            self._flagcx_comm.flagcx.flagcxGroupStart() if hasattr(
                self._flagcx_comm, "flagcx"
            ) else None
            for inp in input_:
                output_list.append(_all_gather_single(inp, sizes=sizes))
            self._flagcx_comm.flagcx.flagcxGroupEnd() if hasattr(
                self._flagcx_comm, "flagcx"
            ) else None
            return output_list
        else:
            output_list = []
            for inp in input_:
                output_list.append(_all_gather_single(inp, sizes=sizes))
            return output_list

    # ─── send ────────────────────────────────────────────────────────────────

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        """Send tensor to destination rank (rank_in_group)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.send(tensor, dst)
            return
        dist.send(tensor, self.ranks[dst], self.device_group)

    # ─── recv ────────────────────────────────────────────────────────────────

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        """Receive tensor from source rank (rank_in_group)."""
        if self._flagcx_comm and not self._flagcx_comm.disabled:
            self._flagcx_comm.recv(tensor, src)
            return
        dist.recv(tensor, self.ranks[src], self.device_group)
