MODEL="opensourcerelease/DeepSeek-R1-bf16"
WARMUP=2
ITERS=5
GPU_MEM=0.9


LATENCY_INPUT_LEN=32
LATENCY_OUTPUT_LEN=128
LATENCY_BATCH_SIZE=8

THROUGHPUT_INPUT_LEN=1024
THROUGHPUT_OUTPUT_LEN=1024

# Environment variables
export VLLM_CONTIGUOUS_PA=false
export VLLM_SKIP_WARMUP=false
export VLLM_CPU_KVCACHE_SPACE=50

# For more detailed HPU logs [optional]
# export VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL=true
# export VLLM_HPU_LOG_STEP_CPU_FALLBACKS=true

python3 /workspaces/vllm/benchmarks/benchmark_latency.py --model $MODEL --num-iters-warmup $WARMUP \
    --num-iters $ITERS \
    --gpu-memory-utilization $GPU_MEM \
    --input-len $LATENCY_INPUT_LEN \
    --output-len $LATENCY_OUTPUT_LEN \
    --batch-size $LATENCY_BATCH_SIZE \
    --trust-remote-code \
    --max-model-len=2048 \
    --load-format 'dummy' \
    --output-json "results/latency.json" > "results/latency.log"


python3 /workspaces/vllm/benchmarks/benchmark_throughput.py --model $MODEL \
    --gpu-memory-utilization $GPU_MEM \
    --input-len $THROUGHPUT_INPUT_LEN \
    --output-len $THROUGHPUT_OUTPUT_LEN \
    --trust-remote-code \
    --max-model-len=2048 \
    --load-format 'dummy' \
    --output-json "results/throughput.json" > "results/throughput.log"