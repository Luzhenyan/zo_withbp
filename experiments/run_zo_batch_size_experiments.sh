#!/bin/bash

# ZO Query Batch Size 实验脚本
# 测试不同的 zo_query_batch_size 值：2, 4, 8, 16, 32, 64
# 所有实验使用: mode=Instruct, bp_interval=1, bp_batch_size=1, q=64

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT="core/reproduce_zo_paper_1106.py"

# 通用参数
BATCH_SIZE=1  # 主训练循环的batch size（实际不会直接用于计算）
BP_BATCH_SIZE=1  # BP梯度计算用1条数据
Q=64
OPTIMIZER="mudamw"
MODEL_SIZE="200M"
EPOCHS=2
EVAL_INTERVAL=100
SCOPE="full"
LEARNING_RATE=0.001
BP_INTERVAL=1

# ZO Query Batch Size 值列表
ZO_BATCH_SIZES=(2 4 8 16 32 64)

# GPU分配：6个实验使用GPU 2-7
GPUS=(2 3 4 5 6 7)

# 创建日志目录
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# 激活conda环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate MeZO

echo "=========================================="
echo "开始运行 ZO Query Batch Size 实验（并行执行）"
echo "测试 zo_query_batch_size: ${ZO_BATCH_SIZES[@]}"
echo "使用GPU: ${GPUS[@]}"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  - Mode: Instruct"
echo "  - BP batch size: $BP_BATCH_SIZE (每个step用1条数据计算BP梯度)"
echo "  - ZO query batch size: 2, 4, 8, 16, 32, 64 (每个方向用不同数量的数据)"
echo "  - BP interval: $BP_INTERVAL (每个step都计算BP梯度)"
echo "  - Query budget q: $Q"
echo "  - Dataset: dclm-pubmedqa-merged (合并数据集)"
echo "=========================================="

# 存储所有后台任务的PID
PIDS=()

# 启动6个实验
for i in "${!ZO_BATCH_SIZES[@]}"; do
    ZO_BS=${ZO_BATCH_SIZES[$i]}
    GPU_ID=${GPUS[$i]}
    EXP_NUM=$((i+1))
    
    echo ""
    echo "实验 $EXP_NUM: ZO Query Batch Size = $ZO_BS (GPU $GPU_ID)"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
        --mode Instruct \
        --scope $SCOPE \
        --query_budget_q $Q \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --bp_batch_size $BP_BATCH_SIZE \
        --zo_query_batch_size $ZO_BS \
        --optimizer $OPTIMIZER \
        --model_size $MODEL_SIZE \
        --dataset dclm-pubmedqa-merged \
        --bp_interval $BP_INTERVAL \
        --queries_use_different_data \
        --eval_interval $EVAL_INTERVAL \
        --log_interval 10 \
        --run_name "Instruct_zo_bs${ZO_BS}_bp_bs${BP_BATCH_SIZE}" \
        > $LOG_DIR/exp_zo_bs${ZO_BS}.log 2>&1 &
    
    PIDS+=($!)
    echo "  启动在后台，PID: $!, GPU: $GPU_ID, 日志: $LOG_DIR/exp_zo_bs${ZO_BS}.log"
done

echo ""
echo "=========================================="
echo "所有实验已启动！"
echo "总共 ${#PIDS[@]} 个实验在后台运行"
echo "PID列表: ${PIDS[@]}"
echo ""
echo "监控命令:"
echo "  - 查看所有进程: ps aux | grep reproduce_zo_paper_1106"
echo "  - 查看GPU使用: watch -n 1 nvidia-smi"
echo "  - 查看日志: tail -f $LOG_DIR/exp_zo_bs*.log"
echo ""
echo "等待所有实验完成..."
echo "=========================================="

# 等待所有后台任务完成
for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "PID $PID 完成 (退出码: $EXIT_CODE)"
    else
        echo "PID $PID 失败 (退出码: $EXIT_CODE)"
    fi
done

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="


