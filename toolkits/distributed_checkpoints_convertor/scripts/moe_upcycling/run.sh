#!/bin/bash
set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
# export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250908:${CONVERTOR_DIR}/impl:$PYTHONPATH
export PYTHONPATH=$( dirname ${MEGATRON_PATCH_PATH})/YuLan-Pretrain:${CONVERTOR_DIR}/impl:/mnt/yulan_pretrain/gaoyanzipeng/models/moe/L56-D1920-E64A8-SE1-ED512-MLP0-Gfalse-S4096-mix10b-iter100/iter_100-hf:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6


NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

MODEL_SIZE=$1 # NOTE: not used
LOAD_DIR=$2
SAVE_DIR=$3
MG2HF=$4
USE_CUDA=$5
PR=$6
HF_DIR=$7

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
echo "MODEL SIZE: ${MODEL_SIZE}"
echo "LOAD DIR: ${LOAD_DIR}"
echo "SAVE DIR: ${SAVE_DIR}"
echo "HF_DIR: ${HF_DIR}"

OTHER_ARGS=()
if [ ${MG2HF} = true ]; then
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${HF_DIR}
        --hf-dir ${HF_DIR}
        --mcore2hf
        --load "${LOAD_DIR}"
        --use-checkpoint-args
    )
    printf "${YELLOW}Warning: Using checkpoint: ${LOAD_DIR} args. Will override args passed by GPT_ARGS ${NC}\n"
    mkdir -p ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${HF_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
else
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${LOAD_DIR}
    )
    mkdir -p ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVE_DIR}
fi

if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(
        --use-gpu
    )
fi

if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(
        --fp16
    )
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(
        --bf16
    )
fi

if [ -z ${NUM_NODES} ]; then
    echo "Please Provide WORLD_SIZE"
    exit
fi

if [ -z ${NODE_RANK} ]; then
    echo "Please Provide RANK"
    exit
fi

if [ -z ${MASTER_ADDR} ]; then
    echo "Please Provide MASTER_ADDR"
    exit
fi

if [ -z ${MASTER_PORT} ]; then
    echo "Please Provide MASTER_PORT"
    exit
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
GPT_MODEL_ARGS=(
    --attention-backend auto
)

if [ $MODEL_SIZE = gyzp ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 56
        --hidden-size 1920
        --num-attention-heads 30
        --kv-channels 64                # 对应字典: 'kv_channels': 64
        --group-query-attention
        --num-query-groups 6            # 对应字典: 'num_query_groups': 6
        --ffn-hidden-size 512           # 对应字典: 'ffn_hidden_size': 512 (注意这里非常小，与MoE expert size一致)
        --swiglu
        
        # 位置编码参数
        --position-embedding-type rope
        --rotary-base 490000            # 对应字典: 'rotary_base': 490000
        --max-position-embeddings 4096  # 对应字典: 'max_position_embeddings': 4096
        --seq-length 4096               # 需与 max-position-embeddings 保持一致
        
        # 词表参数
        --vocab-size 99072              # 对应字典: 'padded_vocab_size': 99072
        --padded-vocab-size 99072       # 强制指定
        
        # 其他结构参数
        --untie-embeddings-and-output-weights
        --disable-bias-linear           # 对应字典: 'add_bias_linear': False
        --add-qkv-bias
        --normalization RMSNorm
        --norm-epsilon 1e-05            # 对应字典: 'norm_epsilon': 1e-05

        # === MoE 关键参数 ===
        --num-experts 64                # 对应字典: 'num_experts': 64
        --moe-router-topk 2             # 对应字典: 'moe_router_topk': 2 (原脚本注释说8，但这里依据提供的配置改为2)
        --moe-ffn-hidden-size 512       # 对应字典: 'moe_ffn_hidden_size': 512
        --moe-router-load-balancing-type global_aux_loss # 对应字典: 'moe_router_load_balancing_type': 'global_aux_loss'
        --moe-grouped-gemm              # 对应字典: 'moe_grouped_gemm': True
        --moe-expert-capacity-factor 0  # 对应字典: 'moe_expert_capacity_factor': None (通常设为0或None表示无限制)
    )
    if [ -z  "$MODEL_PARALLEL_ARGS" ]; then
        MODEL_PARALLEL_ARGS=(
            --tensor-model-parallel-size 1
            --pipeline-model-parallel-size 1
        )
    fi
fi

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
)

CONVERT_ARGS=(
    --model-type GPT 
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    
    --padded-vocab-size 99072
    --no-load-optim
    --no-load-rng
    --logging-level 1

    --debug
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} impl/convert.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo $cmd
eval $cmd