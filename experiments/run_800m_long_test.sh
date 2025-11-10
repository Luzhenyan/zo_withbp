#!/bin/bash

# 800M模型长时间测试 - BP only 和 ZO only
# GPU 6和7，训练25000步

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$BASE_DIR"

SCRIPT="core/reproduce_zo_paper_1106.py"
DATASET="dclm-pubmedqa-merged"  # 混合数据集

# 测试参数
MODEL_SIZE="800M"  # 800M模型（872M参数）
EPOCHS=1  # 1个epoch，但会限制最大steps
EVAL_INTERVAL=500  # 每500步评估一次
SCOPE="full"
LEARNING_RATE=0.001
OPTIMIZER="mudamw"
BATCH_SIZE=2  # 增大batch_size以加快训练

# GPU分配：使用GPU 6和7
GPUS=(6 7)

# 创建日志目录
LOG_DIR="experiment_logs/800m_long_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "800M模型长时间测试 (25000步)"
echo "=========================================="
echo "模型大小: $MODEL_SIZE (872M参数)"
echo "数据集: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "日志目录: $LOG_DIR"
echo ""

# 测试1: BP only (FO模式)
echo "测试1: BP only (First-Order)"
echo "  - GPU: ${GPUS[0]}"
echo "  - batch_size: $BATCH_SIZE"
echo ""

TEST1_NAME="test_800m_bp_only_25ksteps"
TEST1_LOG="$LOG_DIR/${TEST1_NAME}.log"
TEST1_CSV="logs/${TEST1_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python $SCRIPT \
        --mode FO \
        --scope $SCOPE \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --dataset $DATASET \
        --model_size $MODEL_SIZE \
        --csv_file $TEST1_CSV \
        --run_name $TEST1_NAME \
        --disable_checkpoint \
        > $TEST1_LOG 2>&1
" &

PID1=$!
echo "  测试1 PID: $PID1, GPU: ${GPUS[0]}, 日志: $TEST1_LOG"
echo ""

# 测试2: ZO only
echo "测试2: ZO only"
echo "  - GPU: ${GPUS[1]}"
echo "  - query_budget_q: 8"
echo "  - batch_size: $BATCH_SIZE"
echo ""

TEST2_NAME="test_800m_zo_only_25ksteps"
TEST2_LOG="$LOG_DIR/${TEST2_NAME}.log"
TEST2_CSV="logs/${TEST2_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[1]} python $SCRIPT \
        --mode ZO \
        --scope $SCOPE \
        --query_budget_q 8 \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --dataset $DATASET \
        --model_size $MODEL_SIZE \
        --csv_file $TEST2_CSV \
        --run_name $TEST2_NAME \
        --disable_checkpoint \
        > $TEST2_LOG 2>&1
" &

PID2=$!
echo "  测试2 PID: $PID2, GPU: ${GPUS[1]}, 日志: $TEST2_LOG"
echo ""

echo "=========================================="
echo "测试已启动"
echo "=========================================="
echo "测试1 (BP only): PID $PID1, GPU ${GPUS[0]}"
echo "测试2 (ZO only): PID $PID2, GPU ${GPUS[1]}"
echo ""
echo "查看日志:"
echo "  tail -f $TEST1_LOG"
echo "  tail -f $TEST2_LOG"
echo ""
echo "查看进程:"
echo "  ps aux | grep $SCRIPT"
echo ""
echo "监控GPU:"
echo "  watch -n 1 'nvidia-smi | grep -E \"GPU|python\"'"
echo ""
echo "提示: 实验会运行较长时间（预计数小时），使用 --disable_checkpoint 避免磁盘空间问题"
echo ""
echo "等待实验完成..."
wait

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo "CSV日志:"
echo "  $TEST1_CSV"
echo "  $TEST2_CSV"
echo ""

