#!/bin/bash

# 测试1B模型是否能正常运行
# 测试三种模式：BP only, ZO only, BP+ZO (Instruct)

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$BASE_DIR"

SCRIPT="core/reproduce_zo_paper_1106.py"
DATASET="dclm-pubmedqa-merged"  # 混合数据集

# 测试参数（使用小数据集和少步数快速测试）
MODEL_SIZE="800M"  # 使用800M模型（872M参数），比1B更节省显存
EPOCHS=1  # 只测试1个epoch
MAX_SAMPLES=20  # 只使用20个样本，快速验证
EVAL_INTERVAL=0  # 不进行评估，加快速度
SCOPE="full"
LEARNING_RATE=0.001
BP_INTERVAL=1
OPTIMIZER="mudamw"

# GPU分配：3个测试使用GPU 3, 4, 5
GPUS=(3 4 5)

# 创建日志目录
LOG_DIR="test_logs/1b_model_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "800M模型测试 - 验证能否正常运行"
echo "=========================================="
echo "模型大小: $MODEL_SIZE (实际约872M参数)"
echo "数据集: $DATASET"
echo "最大样本数: $MAX_SAMPLES (约10-20步)"
echo "日志目录: $LOG_DIR"
echo ""

# 测试1: BP only (FO模式)
echo "测试1: BP only (First-Order)"
echo "  - 模式: FO"
echo "  - batch_size: 1 (小batch节省显存)"
echo ""

TEST1_NAME="test1_1b_bp_only"
TEST1_LOG="$LOG_DIR/${TEST1_NAME}.log"
TEST1_CSV="logs/${TEST1_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python $SCRIPT \
        --mode FO \
        --scope $SCOPE \
        --learning_rate $LEARNING_RATE \
        --batch_size 1 \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --dataset $DATASET \
        --model_size $MODEL_SIZE \
        --max_samples $MAX_SAMPLES \
        --csv_file $TEST1_CSV \
        --run_name $TEST1_NAME \
        > $TEST1_LOG 2>&1
" &

PID1=$!
echo "  测试1 PID: $PID1, GPU: ${GPUS[0]}, 日志: $TEST1_LOG"
echo ""

# 测试2: ZO only
echo "测试2: ZO only"
echo "  - 模式: ZO"
echo "  - query_budget_q: 8 (测试时用较少方向)"
echo "  - batch_size: 1"
echo ""

TEST2_NAME="test2_1b_zo_only"
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
        --batch_size 1 \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --dataset $DATASET \
        --model_size $MODEL_SIZE \
        --max_samples $MAX_SAMPLES \
        --csv_file $TEST2_CSV \
        --run_name $TEST2_NAME \
        > $TEST2_LOG 2>&1
" &

PID2=$!
echo "  测试2 PID: $PID2, GPU: ${GPUS[1]}, 日志: $TEST2_LOG"
echo ""

# 测试3: BP+ZO (Instruct模式)
echo "测试3: BP+ZO (Instruct模式)"
echo "  - 模式: Instruct"
echo "  - BP batch_size: 1"
echo "  - ZO query_budget_q: 8"
echo "  - ZO query_batch_size: 1"
echo ""

TEST3_NAME="test3_1b_bp_zo"
TEST3_LOG="$LOG_DIR/${TEST3_NAME}.log"
TEST3_CSV="logs/${TEST3_NAME}_$(date +%Y%m%d_%H%M%S)/training_log.csv"

nohup bash -c "
    source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zo_withbp
    CUDA_VISIBLE_DEVICES=${GPUS[2]} python $SCRIPT \
        --mode Instruct \
        --scope $SCOPE \
        --query_budget_q 8 \
        --learning_rate $LEARNING_RATE \
        --batch_size 1 \
        --bp_batch_size 1 \
        --zo_query_batch_size 1 \
        --epochs $EPOCHS \
        --eval_interval $EVAL_INTERVAL \
        --optimizer $OPTIMIZER \
        --bp_interval $BP_INTERVAL \
        --dataset $DATASET \
        --model_size $MODEL_SIZE \
        --max_samples $MAX_SAMPLES \
        --csv_file $TEST3_CSV \
        --run_name $TEST3_NAME \
        > $TEST3_LOG 2>&1
" &

PID3=$!
echo "  测试3 PID: $PID3, GPU: ${GPUS[2]}, 日志: $TEST3_LOG"
echo ""

echo "=========================================="
echo "所有测试已启动"
echo "=========================================="
echo "测试1 (BP only):    PID $PID1, GPU ${GPUS[0]}"
echo "测试2 (ZO only):    PID $PID2, GPU ${GPUS[1]}"
echo "测试3 (BP+ZO):      PID $PID3, GPU ${GPUS[2]}"
echo ""
echo "查看日志:"
echo "  tail -f $TEST1_LOG"
echo "  tail -f $TEST2_LOG"
echo "  tail -f $TEST3_LOG"
echo ""
echo "查看进程:"
echo "  ps aux | grep $SCRIPT"
echo ""
echo "监控GPU使用:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "等待所有测试完成..."
echo ""

# 等待所有后台进程完成
wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?
wait $PID3
STATUS3=$?

echo ""
echo "=========================================="
echo "测试完成状态"
echo "=========================================="
if [ $STATUS1 -eq 0 ]; then
    echo "✅ 测试1 (BP only): 成功 - 1B模型可以运行BP模式"
else
    echo "❌ 测试1 (BP only): 失败 (退出码: $STATUS1)"
    echo "   查看日志: $TEST1_LOG"
fi

if [ $STATUS2 -eq 0 ]; then
    echo "✅ 测试2 (ZO only): 成功 - 1B模型可以运行ZO模式"
else
    echo "❌ 测试2 (ZO only): 失败 (退出码: $STATUS2)"
    echo "   查看日志: $TEST2_LOG"
fi

if [ $STATUS3 -eq 0 ]; then
    echo "✅ 测试3 (BP+ZO): 成功 - 1B模型可以运行Instruct模式"
else
    echo "❌ 测试3 (BP+ZO): 失败 (退出码: $STATUS3)"
    echo "   查看日志: $TEST3_LOG"
fi

echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
SUCCESS_COUNT=0
if [ $STATUS1 -eq 0 ]; then SUCCESS_COUNT=$((SUCCESS_COUNT + 1)); fi
if [ $STATUS2 -eq 0 ]; then SUCCESS_COUNT=$((SUCCESS_COUNT + 1)); fi
if [ $STATUS3 -eq 0 ]; then SUCCESS_COUNT=$((SUCCESS_COUNT + 1)); fi

echo "成功: $SUCCESS_COUNT/3"
echo ""

if [ $SUCCESS_COUNT -eq 3 ]; then
    echo "🎉 所有测试通过！1B模型可以在所有模式下运行。"
    echo ""
    echo "可以开始正式训练，建议参数："
    echo "  - batch_size: 根据GPU显存调整（建议1-2）"
    echo "  - query_budget_q: 根据显存调整（建议32-64）"
    echo "  - 使用梯度累积或混合精度训练以节省显存"
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️  部分测试通过，请检查失败的测试日志。"
    echo "   可能需要："
    echo "   1. 减小batch_size"
    echo "   2. 减小query_budget_q"
    echo "   3. 使用gradient checkpointing"
    echo "   4. 使用混合精度训练"
else
    echo "❌ 所有测试失败，1B模型可能超出当前GPU显存。"
    echo "   建议："
    echo "   1. 检查GPU显存是否足够（1B模型需要约4-6GB显存）"
    echo "   2. 尝试使用更小的batch_size (如1)"
    echo "   3. 使用gradient checkpointing"
    echo "   4. 考虑使用模型并行或数据并行"
fi

echo ""
echo "CSV日志文件:"
echo "  $TEST1_CSV"
echo "  $TEST2_CSV"
echo "  $TEST3_CSV"
echo ""

