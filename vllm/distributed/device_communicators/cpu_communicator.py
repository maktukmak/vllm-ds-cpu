import os
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform


class CpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not current_platform.is_cpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        self._init_cpu_shm()

    def _init_cpu_shm(self):
        instance_name = os.environ["VLLM_INSTANCE_ID"]
        group_name = f"{instance_name}-cpushm"
        torch.ops._C.init_shm_manager(
            group_name,
            self.world_size,
            self.rank,
        )
        torch.distributed.barrier(self.group)
        torch.ops._C.join_shm_manager(group_name, )
        torch.distributed.barrier(self.group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        torch.ops._C.shm_allreduce(input_)
        return input_

    def gather(self,
               input_: torch.Tensor,
               rank_in_group: int,
               ranks: List[int],
               dst: int = 0,
               dim: int = -1):
        # Allocate output tensor.
        if rank_in_group == dst:
            gather_list = [
                torch.empty_like(input_) for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        # Gather.
        torch.ops._C.shm_gather(input_, gather_list, ranks[dst])

        if rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None

        return output_tensor
