#!/bin/bash

# ALT vs ZO vs FO 对比实验脚本
# 运行不同 alt_zo_steps 的 ALT 模式，以及纯 ZO 和纯 FO 模式
# 固定参数：300 steps, q=8, log_interval=1

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# GPU 配置（固定使用 GPU 5）
GPU_ID=5

# 固定参数
BASE_PARAMS="--scope full --batch_size 2 --query_budget_q 8 --learning_rate 1e-3 --optimizer mudamw --log_interval 1"
MAX_STEPS=300
EPOCHS=1
ALT_ZO_STEPS_LIST=(1 10 20 40)

# 创建结果目录（参考 parallel_sweep.sh 的命名规范）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_ALT_ZO_FO_altzo1-10-20-100_q8_maxsteps300_${TIMESTAMP}"
CSV_DIR="csv_logs_ALT_ZO_FO_altzo1-10-20-100_q8_maxsteps300_${TIMESTAMP}"
JOB_LOG_DIR="job_logs_${TIMESTAMP}"
CACHE_DIR="cache"
TEMP_DIR="temp"
mkdir -p "$RESULTS_DIR" "$CSV_DIR" "$JOB_LOG_DIR" "$CACHE_DIR" "$TEMP_DIR"

# 日志文件
LOG_FILE="alt_sweep_${TIMESTAMP}.log"
SUMMARY_FILE="alt_sweep_summary_${TIMESTAMP}.txt"

# 运行单个实验
run_single_experiment() {
    local exp_name="$1"
    local mode="$2"
    local alt_zo_steps="$3"
    local csv_file="$4"
    local plot_file="$5"
    local job_log="${JOB_LOG_DIR}/${exp_name}.log"
    
    echo -e "${YELLOW}📊 Starting experiment: $exp_name (GPU: $GPU_ID)${NC}" | tee -a "$job_log"
    
    # 构建命令
    local cmd="python /data/lzy/zo-test-cdq/reproduce_zo_paper_withbp.py"
    cmd="$cmd --mode $mode"
    cmd="$cmd --scope full"
    cmd="$cmd --batch_size 2"
    cmd="$cmd --learning_rate 1e-3"
    cmd="$cmd --optimizer mudamw"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --max_steps $MAX_STEPS"
    cmd="$cmd --log_interval 1"
    cmd="$cmd --csv_file $csv_file"
    
    if [ "$mode" = "ALT" ]; then
        cmd="$cmd --alt_zo_steps $alt_zo_steps"
        cmd="$cmd --query_budget_q 8"
    elif [ "$mode" = "ZO" ]; then
        cmd="$cmd --query_budget_q 8"
    fi
    
    echo "Command: $cmd" >> "$job_log"
    echo "GPU: $GPU_ID" >> "$job_log"
    echo "Start time: $(date)" >> "$job_log"
    echo "----------------------------------------" >> "$job_log"
    
    # 运行实验
    if CUDA_VISIBLE_DEVICES=$GPU_ID eval $cmd >> "$job_log" 2>&1; then
        echo -e "${GREEN}✅ Experiment $exp_name completed successfully${NC}" | tee -a "$job_log"
        echo "End time: $(date)" >> "$job_log"
        echo "SUCCESS" >> "$job_log"
        
        # 移动图片到结果目录
        local plot_pattern="results/${mode}_full_*.png"
        if ls $plot_pattern 2>/dev/null | head -1 | read latest_plot; then
            mv "$latest_plot" "$plot_file" 2>/dev/null || true
        fi
        
        return 0
    else
        echo -e "${RED}❌ Experiment $exp_name failed${NC}" | tee -a "$job_log"
        echo "End time: $(date)" >> "$job_log"
        echo "FAILED" >> "$job_log"
        return 1
    fi
}

# 运行所有实验
run_all_experiments() {
    local successful=0
    local failed=0
    
    echo -e "${GREEN}📊 Running ALT mode experiments...${NC}"
    for alt_zo_steps in "${ALT_ZO_STEPS_LIST[@]}"; do
        exp_name="ALT_full_q8_altzo${alt_zo_steps}_optmudamw_lr1e-3_bs2"
        csv_file="${CSV_DIR}/${exp_name}.csv"
        plot_file="${RESULTS_DIR}/${exp_name}.png"
        
        if run_single_experiment "$exp_name" "ALT" "$alt_zo_steps" "$csv_file" "$plot_file"; then
            successful=$((successful + 1))
        else
            failed=$((failed + 1))
        fi
        echo ""
    done
    
    echo -e "${GREEN}📊 Running ZO mode experiment...${NC}"
    exp_name="ZO_full_q8_optmudamw_lr1e-3_bs2"
    csv_file="${CSV_DIR}/${exp_name}.csv"
    plot_file="${RESULTS_DIR}/${exp_name}.png"
    
    if run_single_experiment "$exp_name" "ZO" "" "$csv_file" "$plot_file"; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
    fi
    echo ""
    
    echo -e "${GREEN}📊 Running FO mode experiment...${NC}"
    exp_name="FO_full_optmudamw_lr1e-3_bs2"
    csv_file="${CSV_DIR}/${exp_name}.csv"
    plot_file="${RESULTS_DIR}/${exp_name}.png"
    
    if run_single_experiment "$exp_name" "FO" "" "$csv_file" "$plot_file"; then
        successful=$((successful + 1))
    else
        failed=$((failed + 1))
    fi
    echo ""
    
    # 返回统计信息（通过全局变量）
    export SWEEP_SUCCESSFUL=$successful
    export SWEEP_FAILED=$failed
    export SWEEP_TOTAL=$((successful + failed))
}

# 生成最终报告
generate_final_report() {
    local end_time=$(date +%s)
    local duration=$((end_time - SWEEP_START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "${BLUE}📋 ALT/ZO/FO SWEEP SUMMARY REPORT${NC}"
    echo -e "${BLUE}==================================${NC}"
    echo "GPU ID used: $GPU_ID"
    echo "Max steps per experiment: $MAX_STEPS"
    echo "Total experiments: $SWEEP_TOTAL"
    echo -e "Successful: ${GREEN}$SWEEP_SUCCESSFUL${NC}"
    echo -e "Failed: ${RED}$SWEEP_FAILED${NC}"
    if [ $SWEEP_TOTAL -gt 0 ]; then
        echo "Success rate: $(( SWEEP_SUCCESSFUL * 100 / SWEEP_TOTAL ))%"
    fi
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo "CSV logs directory: $CSV_DIR"
    echo "Job logs directory: $JOB_LOG_DIR"
    echo "Log file: $LOG_FILE"
    echo "Summary file: $SUMMARY_FILE"
    echo ""
    
    # 列出所有结果文件
    echo -e "${BLUE}📁 Generated Files:${NC}"
    echo "PNG plots:"
    ls -la "$RESULTS_DIR"/*.png 2>/dev/null | head -10 || echo "  No PNG files found"
    local png_count=$(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l)
    if [ $png_count -gt 10 ]; then
        echo "  ... and $((png_count - 10)) more files"
    fi
    echo ""
    echo "CSV logs:"
    ls -la "$CSV_DIR"/*.csv 2>/dev/null | head -10 || echo "  No CSV files found"
    local csv_count=$(ls -1 "$CSV_DIR"/*.csv 2>/dev/null | wc -l)
    if [ $csv_count -gt 10 ]; then
        echo "  ... and $((csv_count - 10)) more files"
    fi
    echo ""
    
    # 实验详情
    echo -e "${BLUE}📊 Experiment Details:${NC}"
    echo "  ALT mode experiments with alt_zo_steps: ${ALT_ZO_STEPS_LIST[@]}"
    echo "  ZO mode experiment (pure zeroth-order)"
    echo "  FO mode experiment (pure first-order)"
    echo ""
}

# 主程序
main() {
    SWEEP_START_TIME=$(date +%s)
    
    echo -e "${BLUE}🚀 Starting ALT/ZO/FO Comparison Experiments${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo "GPU ID: $GPU_ID"
    echo "Max steps: $MAX_STEPS"
    echo "Results will be saved to: $RESULTS_DIR"
    echo "CSV logs will be saved to: $CSV_DIR"
    echo "Job logs will be saved to: $JOB_LOG_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # 设置 GPU 环境变量
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo -e "${BLUE}Using GPU: $GPU_ID${NC}"
    echo ""
    
    # 记录配置
    echo "Configuration:" >> "$LOG_FILE"
    echo "GPU_ID: $GPU_ID" >> "$LOG_FILE"
    echo "MAX_STEPS: $MAX_STEPS" >> "$LOG_FILE"
    echo "EPOCHS: $EPOCHS" >> "$LOG_FILE"
    echo "ALT_ZO_STEPS_LIST: ${ALT_ZO_STEPS_LIST[@]}" >> "$LOG_FILE"
    echo "BASE_PARAMS: $BASE_PARAMS" >> "$LOG_FILE"
    echo "RESULTS_DIR: $RESULTS_DIR" >> "$LOG_FILE"
    echo "CSV_DIR: $CSV_DIR" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    
    # 运行所有实验
    run_all_experiments >> "$LOG_FILE" 2>&1
    
    # 生成报告
    generate_final_report | tee -a "$LOG_FILE" | tee "$SUMMARY_FILE"
    
    echo -e "${GREEN}🎉 ALT/ZO/FO sweep completed!${NC}"
    echo "Check the results in the $RESULTS_DIR and $CSV_DIR directories."
    echo "Detailed logs available in: $LOG_FILE"
    echo "Summary available in: $SUMMARY_FILE"
}

# 运行主程序
main "$@"
