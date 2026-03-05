#!/bin/bash

set -ex

MEGATRON_PATH=`pwd`
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

TOKENIZER_PATH="/pah/to/your/hf_model/Qwen/Qwen3-1.7B-Base"
TRAIN_DATA_PATH="/pah/to/your/preprocessed_dataset/fineweb_edu_dedup_text_document"
DATA_CACHE="/pah/to/your/cache_dir"

OUTPUT_ROOT="/pah/to/your/output_dir"
filename=$(basename "$0")
SCRIPT_NAME=$(basename "$filename" .sh)
EXP_NAME="${SCRIPT_NAME}"
SAVE_FILE_DIR="$OUTPUT_ROOT/$EXP_NAME"
mkdir -p $SAVE_FILE_DIR

CHECKPOINT_DIR="$SAVE_FILE_DIR/checkpoint"
TENSORBOARD_DIR="$SAVE_FILE_DIR/tensorboard"
LOG_PATH="$SAVE_FILE_DIR/log_node${RANK}.txt"

# meki model spec ===============================
NUM_LAYERS=28
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=6144
NUM_ATT_HEADS=16
NUM_QUERY_GROUPS=8

TP=1
PP=1

LR=4e-4
MIN_LR=2e-4
DTYPE=bf16

# training 50B tokens = 4096 * 256 * 50000
MICRO_BATCH=4
GLOBAL_BATCH=256
SEQ_LENGTH=4096

TRAIN_ITERS=50000
LR_WARMUP_ITERS=500

GPT_ARGS="
    --kv-channels 128 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-query-groups $NUM_QUERY_GROUPS \
    --num-attention-heads $NUM_ATT_HEADS \
    --sequence-parallel \
    --group-query-attention \
    --transformer-impl transformer_engine \
    --use-mcore-models \
    --data-cache-path $DATA_CACHE \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-percent 1.0 \
    --rotary-base 500000 \
    --init-method-std 0.014 \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --$DTYPE \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --disable-bias-linear \
    --swiglu \
    --no-bias-gelu-fusion \
    --normalization RMSNorm \
    --num-workers 8 \
    --norm-epsilon 1e-6 \
    --lr $LR \
    --min-lr $MIN_LR \
    --train-iters $TRAIN_ITERS \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --lr-decay-style cosine \
    --log-interval 1 \
    --log-throughput \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --save-interval 10000 \
    --use-flash-attn \
    --seed 42 \
    --distributed-timeout-minutes 60 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-checkpoint-opt_param-scheduler \
    --ckpt-format torch \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss \
    --recompute-granularity selective \
    --recompute-modules mlp \
"

MEKI_ARGS=" --meki-dim 256 "

DISTRIBUTED_ARGS=(
    --nproc_per_node $NPROC_PER_NODE
    --nnodes $WORLD_SIZE
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $RANK
)

export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun ${DISTRIBUTED_ARGS[@]} \
       pretrain_gpt.py \
       $GPT_ARGS \
       $MEKI_ARGS \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --save $CHECKPOINT_DIR \
       --load $CHECKPOINT_DIR \
       --data-path $TRAIN_DATA_PATH --split 98,2,0 \
       --eval-iters 100 --eval-interval 1000 \
       --tensorboard-dir $TENSORBOARD_DIR 2>&1 | tee $LOG_PATH
