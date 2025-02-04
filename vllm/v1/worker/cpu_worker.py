import os
from typing import Optional, Tuple

import torch

from vllm import envs
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import (Worker,
                                       init_worker_distributed_environment)

logger = init_logger(__name__)


class CPUWorker(Worker):

    def __init__(self, vllm_config: VllmConfig, local_rank: int, rank: int,
                 distributed_init_method: str):
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method)

        self.parallel_config.disable_custom_all_reduce = True

    def initialize(self):
        assert self.device_config.device_type == "cpu"
        #
        # Environment variables for CPU worker
        #

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Intel OpenMP setting
        ld_prealod_str = os.getenv("LD_PRELOAD", "")
        if "libiomp5.so" in ld_prealod_str:
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            os.environ['KMP_BLOCKTIME'] = "1"
            # Prevents the CPU to run into low performance state
            os.environ['KMP_TPAUSE'] = "0"
            # Provides fine granularity parallelism
            os.environ['KMP_FORKJOIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_PLAIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_REDUCTION_BARRIER_PATTERN'] = "dist,dist"

        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "all":
            self.local_omp_cpuid = "all"
        else:
            self.local_omp_cpuid = omp_cpuids.split("|")[self.rank]
            ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
            if ret:
                logger.info(ret)

        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank, "gloo")
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu"))

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This determines how many KV blocks can fit into the configured CPU
        KV cache space.

        Note that since vLLM assumes a block resides on GPU if it can be
        modified, we return num_gpu_blocks=num_cpu_blocks and num_cpu_blocks=0.
        This allows us to reuse the scheduler of vLLM without generalizing it
        to different devices.
        """
        # For CPU device, the block number will be calculated based on the
        # cpu_kvcache_space.
        cache_block_size = _get_cache_block_size_bytes(
            self.cache_config,
            self.model_config,
            self.parallel_config,
        )
        cpu_space = self.cache_config.cpu_kvcache_space_bytes  # type: ignore
        num_cpu_blocks = int(cpu_space // cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_gpu_blocks = num_cpu_blocks
        num_cpu_blocks = 0
        return num_gpu_blocks, num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.rank == 0 else None


def _get_cache_block_size_bytes(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    """Return the size in bytes of a single KV cache block.
    """
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_layers = model_config.get_num_layers(parallel_config)
    block_size = cache_config.block_size
    cache_dtype = cache_config.cache_dtype

    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_layers * (key_cache_block + value_cache_block)
    if cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    return dtype_size * total
