from contextlib import contextmanager

import numpy as np
import torch

from vllm.attention.backends.torch_sdpa import (TorchSDPABackend,
                                                TorchSDPAMetadata)
from vllm.config import CompilationLevel, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.utils import bind_kv_cache
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.worker.cpu_input_batch import CPUInputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert device == torch.device("cpu")
        super().__init__(vllm_config, device)

        self.input_batch: CPUInputBatch = CPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )
        self.use_cuda_graph = False

        self.input_ids = self.input_ids_cpu
        self.positions = self.positions_cpu

        self.seq_lens_tensor_cpu = torch.zeros(self.max_num_reqs,
                                               dtype=torch.int32,
                                               device="cpu")
        self.seq_lens_tensor_np = self.seq_lens_tensor_cpu.numpy()

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_states(scheduler_output)
        self.input_batch.reorder()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

    def warming_up_model(self) -> None:
        compilation_config = self.vllm_config.compilation_config
        if compilation_config.level in [
                CompilationLevel.NO_COMPILATION,
                CompilationLevel.DYNAMO_AS_IS,
        ]:
            return
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings():
            self._dummy_run(self.model, self.max_num_tokens, self.kv_caches)
        logger.info("Warming up done.")

    def _prepare_inputs(self, scheduler_output: SchedulerOutput):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens])

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens,
                  out=self.query_start_loc_np[1:num_reqs + 1])

        seq_lens_np = self.seq_lens_tensor_np[:num_reqs]
        np.add(self.input_batch.num_computed_tokens_cpu[:num_reqs],
               num_scheduled_tokens,
               out=seq_lens_np)
        max_seq_len = seq_lens_np.max().item()
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens_np, out=self.seq_start_loc_np[1:num_reqs + 1])

        num_prompt_reqs = self.input_batch.num_prompt_req
        num_prefill_tokens = self.query_start_loc_np[num_prompt_reqs].item()
        num_decode_tokens = self.query_start_loc_np[num_reqs].item(
        ) - num_prefill_tokens
        slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].long(
        )
        max_query_len = num_scheduled_tokens.max().item()  # type: ignore

        attn_metadata = TorchSDPAMetadata(
            num_prefills=num_prompt_reqs,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=self.
            seq_lens_tensor_cpu[num_prompt_reqs:num_reqs],  # decode
            max_decode_seq_len=max_seq_len,  # decode
            block_tables=self.input_batch.
            block_table.get_device_tensor()[num_prompt_reqs:num_reqs],  # decode
            chunked_prefill=True,
            max_query_len=max_query_len,
            max_kv_len=max_seq_len,
            query_start_loc=self.query_start_loc_cpu[:num_prompt_reqs +
                                                     1],  # prefill
            kv_start_loc=self.seq_start_loc_cpu[:num_prompt_reqs +
                                                1],  # prefill
            prefill_block_tables=self.input_batch.
            block_table.get_device_tensor()[:num_prompt_reqs],  # prefill
            multi_modal_placeholder_index_maps=None,
        )

        query_start_loc = self.query_start_loc_cpu[:num_reqs + 1]

        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return attn_metadata, logits_indices

    def initialize_kv_cache(self, num_blocks: int) -> None:
        """Allocates KV cache on CPU."""
        assert len(self.kv_caches) == 0
        kv_cache_shape = TorchSDPABackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))
        bind_kv_cache(
            self.vllm_config.compilation_config.static_forward_context,
            [self.kv_caches])


@contextmanager
def _set_global_compilation_settings():
    import torch._inductor.config

    # Note: The CPPGEMM backend requires freezing parameters.
    freezing_value = torch._inductor.config.freezing
    torch._inductor.config.freezing = True
    yield
    torch._inductor.config.freezing = freezing_value
