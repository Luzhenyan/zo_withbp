#!/bin/bash

# 两个实验脚本
# 实验1：BP和ZO使用不同的数据子集（不重合）
#   - BP: batch_size=2，从数据集的前50%获取
#   - ZO: 64个方向，每个方向batch_size=2，从数据集的后50%获取，总共128条数据
# 实验2：ZO的64个方向共享128条数据
#   - BP: batch_size=2
#   - ZO: 64个方向，共享128条数据（batch_size=128）

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT="core/reproduce_zo_paper_1106.py"

# 通用参数
BATCH_SIZE=1  # 主训练循环的batch size（实际不会直接用于计算）
BP_BATCH_SIZE=2  # BP梯度计算用2条数据
ZO_QUERY_BATCH_SIZE=2  # 每个ZO方向用2条数据
Q=64  # 64个方向
OPTIMIZER="mudamw"
MODEL_SIZE="200M"
EPOCHS=2
EVAL_INTERVAL=100
SCOPE="full"
LEARNING_RATE=0.001
BP_INTERVAL=1

# GPU分配：2个实验使用GPU 3和6
GPUS=(3 6)

# 创建日志目录
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# 激活conda环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate MeZO

echo "=========================================="
echo "开始运行两个实验（并行执行）"
echo "使用GPU: ${GPUS[@]}"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  - Mode: Instruct"
echo "  - BP batch size: $BP_BATCH_SIZE"
echo "  - ZO query batch size: $ZO_QUERY_BATCH_SIZE"
echo "  - BP interval: $BP_INTERVAL (每个step都计算BP梯度)"
echo "  - Query budget q: $Q"
echo "  - Dataset: dclm-pubmedqa-merged (合并数据集)"
echo "=========================================="

# 存储所有后台任务的PID
PIDS=()

# 实验1：BP和ZO使用不同的数据子集（不重合）
echo ""
echo "实验1: BP和ZO使用不同的数据子集（不重合）"
echo "  - BP: batch_size=$BP_BATCH_SIZE，从数据集的前50%获取"
echo "  - ZO: $Q个方向，每个方向batch_size=$ZO_QUERY_BATCH_SIZE，从数据集的后50%获取"
echo "  - 总共ZO使用: $Q × $ZO_QUERY_BATCH_SIZE = $((Q * ZO_QUERY_BATCH_SIZE)) 条数据"
RUN_NAME1="Instruct_merged_bp${BP_BATCH_SIZE}_zo${ZO_QUERY_BATCH_SIZE}_q${Q}_split"
LOG_FILE1="$LOG_DIR/exp1_split_bp_zo.log"

CUDA_VISIBLE_DEVICES=${GPUS[0]} python $SCRIPT \
    --mode Instruct \
    --scope $SCOPE \
    --query_budget_q $Q \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --bp_batch_size $BP_BATCH_SIZE \
    --zo_query_batch_size $ZO_QUERY_BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset dclm-pubmedqa-merged \
    --bp_interval $BP_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --queries_use_different_data \
    --split_dataset_for_bp_zo \
    --run_name "$RUN_NAME1" \
    > $LOG_FILE1 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[0]}, 日志: $LOG_FILE1"

# 实验2：ZO的64个方向共享128条数据
echo ""
echo "实验2: ZO的64个方向共享128条数据"
echo "  - BP: batch_size=$BP_BATCH_SIZE"
echo "  - ZO: $Q个方向，共享 $((Q * ZO_QUERY_BATCH_SIZE)) 条数据"
RUN_NAME2="Instruct_merged_bp${BP_BATCH_SIZE}_zo${ZO_QUERY_BATCH_SIZE}_q${Q}_shared"
LOG_FILE2="$LOG_DIR/exp2_shared_zo_data.log"

CUDA_VISIBLE_DEVICES=${GPUS[1]} python $SCRIPT \
    --mode Instruct \
    --scope $SCOPE \
    --query_budget_q $Q \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --bp_batch_size $BP_BATCH_SIZE \
    --zo_query_batch_size $ZO_QUERY_BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --model_size $MODEL_SIZE \
    --dataset dclm-pubmedqa-merged \
    --bp_interval $BP_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --log_interval 10 \
    --queries_use_different_data \
    --zo_share_data_across_queries \
    --run_name "$RUN_NAME2" \
    > $LOG_FILE2 2>&1 &
PIDS+=($!)
echo "  启动在后台，PID: $!, GPU: ${GPUS[1]}, 日志: $LOG_FILE2"

echo ""
echo "所有实验已启动。等待它们完成..."
wait "${PIDS[@]}"
echo "所有实验完成！"

