# Scripts Documentation

This document provides detailed descriptions of all scripts in this repository.

## Table of Contents

1. [Main Training Scripts](#main-training-scripts)
2. [Data Management Scripts](#data-management-scripts)
3. [Model Configuration Scripts](#model-configuration-scripts)
4. [Visualization Scripts](#visualization-scripts)
5. [Testing Scripts](#testing-scripts)
6. [Experiment Runner Scripts](#experiment-runner-scripts)
7. [Other Scripts](#other-scripts)

---

## Main Training Scripts

### `reproduce_zo_paper_1106.py`
**Primary training script with full features**

Main script for training GPT-2 models using Zeroth-Order (ZO) optimization methods. This is the most up-to-date version with comprehensive features.

**Key Features:**
- Multiple training modes: `FO` (First-Order/Backprop), `ZO` (Zeroth-Order), `Instruct` (BP-guided ZO)
- Flexible data usage: supports separate datasets for BP and ZO, dataset splitting, and shared data across ZO queries
- Custom optimizers: `CustomAdamOptimizer` and `MuDaMWOptimizer`
- Evaluation support with configurable intervals
- Comprehensive logging to CSV files
- Checkpoint saving

**Main Arguments:**
- `--mode`: Training mode (`FO`, `ZO`, `Instruct`)
- `--scope`: Parameter scope (`full`, `reduced`)
- `--query_budget_q`: Number of ZO query directions (default: 64)
- `--learning_rate`: Learning rate
- `--batch_size`: Main training batch size
- `--bp_batch_size`: Batch size for BP gradient computation
- `--zo_query_batch_size`: Batch size for ZO query data
- `--split_dataset_for_bp_zo`: Split dataset into non-overlapping subsets for BP and ZO
- `--zo_share_data_across_queries`: Share same data batch across all ZO query directions
- `--eval_dataset_name`: Dataset name for evaluation
- `--eval_interval`: Steps between evaluations

**Usage Example:**
```bash
python reproduce_zo_paper_1106.py \
    --mode Instruct \
    --scope full \
    --query_budget_q 64 \
    --learning_rate 0.001 \
    --batch_size 1 \
    --bp_batch_size 2 \
    --zo_query_batch_size 2 \
    --dataset dclm-pubmedqa-merged \
    --split_dataset_for_bp_zo \
    --eval_interval 100
```

---

### `reproduce_zo_paper_1105.py`
**Previous version with evaluation functionality**

Earlier version of the training script that includes evaluation functionality. This script was used as a reference when merging evaluation features into `reproduce_zo_paper_1106.py`.

**Key Differences from 1106:**
- Simpler data handling (no dataset splitting or shared ZO data)
- Evaluation functionality included
- Tokenizer loading prioritizes local path

---

### `reproduce_zo_paper_new.py`
**Alternative training script variant**

Another variant of the training script with different implementation details. Used for experimental purposes.

---

### `reproduce_zo_paper_withbp.py`
**Training script with BP gradient support**

Training script that includes backpropagation gradient computation alongside ZO optimization.

---

### `reproduce_zo_paper.py`
**Original training script**

The original training script for reproducing ZO paper results. This is the base version from which other variants were developed.

---

## Data Management Scripts

### `data.py`
**Dataset configuration and loading**

Centralized dataset configuration and data loading utilities.

**Key Features:**
- Dataset configuration dictionary (`DATASET_CONFIGS`) for multiple datasets:
  - `cosmopedia-100k`, `cosmopedia`: High-quality synthetic educational data
  - `wikitext-103`: Wikipedia text
  - `dclm-local`, `pubmedqa-local`: Local medical datasets
  - `dclm-pubmedqa-merged`: Merged dataset (20,000 samples)
- `get_dataloader()`: Main function to load and prepare datasets
- Caching support for faster subsequent loads
- Tokenization and DataLoader creation

**Main Functions:**
- `get_dataloader(dataset_name, tokenizer, batch_size, ...)`: Load dataset and create DataLoader
- `get_dataset_info(dataset_name)`: Get information about a dataset
- `list_available_datasets()`: List all available datasets

---

### `merge_datasets.py`
**Merge multiple datasets into one**

Script to combine `dclm-local` and `pubmedqa-local` datasets into a unified dataset.

**Features:**
- Processes DCLM and PubMedQA datasets
- Unifies text field format
- Shuffles merged data
- Saves to disk for reuse

**Usage:**
```bash
python merge_datasets.py
```

**Output:** `./datasets_subset/dclm_pubmedqa_merged/`

---

### `check_dataset_size.py`
**Check dataset size**

Simple utility to check the number of samples in datasets.

**Usage:**
```bash
python check_dataset_size.py
```

---

### `check_data_distribution.py`
**Verify dataset distribution**

Checks if merged datasets are uniformly mixed by analyzing data distribution across chunks.

**Usage:**
```bash
python check_data_distribution.py
```

---

### `download_datasets.py`
**Download datasets from Hugging Face**

Utility script to download datasets from Hugging Face Hub.

---

### `test_merged_dataset.py`
**Test merged dataset loading**

Test script to verify that merged datasets can be loaded correctly.

---

## Model Configuration Scripts

### `model.py`
**Model configuration and creation**

Defines GPT-2 model configurations and creation utilities.

**Key Features:**
- Model size configurations: `20M`, `200M`, `500M`, `1B`
- `create_model()`: Create GPT-2 model with specified size
- `get_model_info()`: Get information about a model configuration
- `list_available_models()`: List all available model configurations

**Model Sizes:**
- `20M`: ~20M parameters, 6 layers, 4 heads
- `200M`: ~200M parameters, 12 layers, 12 heads (GPT-2 Small)
- `500M`: ~500M parameters, 24 layers, 16 heads
- `1B`: ~1B parameters, 36 layers, 20 heads

**Usage:**
```python
from model import create_model
model = create_model(model_size='200M', vocab_size=50257)
```

---

## Visualization Scripts

### `plot_two_experiments.py`
**Plot loss curves for two experiments**

Creates comparison plots for two specific experiments (split vs shared data usage).

**Features:**
- Automatically finds latest CSV files for split and shared experiments
- Plots training and evaluation loss
- Samples data every 100 steps for training loss
- English labels only

**Usage:**
```bash
python plot_two_experiments.py
```

**Output:** `plots/loss_curves_two_experiments.png`

---

### `plot_loss_curves.py`
**Plot loss curves for multiple experiments**

Creates loss curve plots for experiments 1-3 and 4-6.

**Features:**
- Groups experiments by dataset combinations
- Plots training and evaluation loss
- Supports multiple experiment configurations

**Usage:**
```bash
python plot_loss_curves.py
```

**Output:** 
- `plots/loss_curves_exp1-3.png`
- `plots/loss_curves_exp4-6.png`

---

### `plot_all_results.py`
**Comprehensive plotting and analysis**

Advanced visualization script with multiple analysis types.

**Features:**
- Loss curve plotting
- ZO-specific analysis
- FO vs ZO comparison
- Summary report generation
- ALT experiment analysis

**Usage:**
```bash
python plot_all_results.py
```

**Output:** Multiple plots in `plots/` directory

---

### `quick_plot.py`
**Quick plotting utility**

Simple script for quick visualization of training results.

---

## Testing Scripts

### `test_training.py`
**Test training functionality**

Test script to verify training pipeline works correctly.

---

### `test_setup.py`
**Test environment setup**

Verifies that all dependencies and configurations are set up correctly.

---

### `test_quick.py`
**Quick test script**

Fast test script for basic functionality verification.

---

### `test_zo_vs_fo.py`
**Compare ZO vs FO methods**

Test script to compare Zeroth-Order and First-Order optimization methods.

---

## Experiment Runner Scripts

### `run_two_experiments.sh`
**Run two specific experiments**

Shell script to launch two experiments comparing different data usage patterns:
- Experiment 1: BP and ZO use different data subsets (non-overlapping)
- Experiment 2: ZO's 64 directions share 128 samples

**Features:**
- Configurable GPU assignment
- Automatic log file management
- Background execution

**Usage:**
```bash
chmod +x run_two_experiments.sh
./run_two_experiments.sh
```

**Configuration:**
- Edit GPU assignment: `GPUS=(3 6)`
- Modify experiment parameters in the script

---

### `run_zo_batch_size_experiments.sh`
**Run experiments with different ZO batch sizes**

Launches multiple experiments testing different `zo_query_batch_size` values (2, 4, 8, 16, 32, 64).

**Usage:**
```bash
chmod +x run_zo_batch_size_experiments.sh
./run_zo_batch_size_experiments.sh
```

---

### `run_experiments.sh`
**Run standard experiments**

Main script for running standard experiment configurations.

---

### `run_instruct_25000steps.sh`
**Run Instruct mode for 25000 steps**

Specific script for long-running Instruct mode experiments.

---

### `run_alt_zo_steps_sweep.sh`
**Sweep ALT ZO steps parameter**

Parameter sweep script for ALT ZO steps.

---

### `run_and_plot.sh`
**Run experiments and plot results**

Combined script that runs experiments and generates plots.

---

### `parallel_sweep.sh`
**Parallel parameter sweep**

Script for running parallel parameter sweeps across multiple GPUs.

---

### `quick_parallel_test.sh`
**Quick parallel testing**

Fast parallel test script.

---

### `quick_test.sh`
**Quick test runner**

Simple test runner script.

---

## Other Scripts

### `flwr_server.py`
**Flower federated learning server**

Federated learning server implementation using Flower framework with ZO optimization support.

**Features:**
- ZOCloudMuonStrategy for federated ZO optimization
- MuDaMW optimizer integration
- Server-side aggregation

---

### `zo_sst_finetune.py`
**SST-2 fine-tuning with ZO**

Fine-tuning script for SST-2 sentiment classification using ZO optimization.

**Features:**
- Classification task support
- Encoder freezing option
- ZO gradient estimation for classification

---

## Common Patterns

### CSV Logging Format

All training scripts log to CSV files with the following columns:
- `timestamp`: Timestamp of the log entry
- `epoch`: Current epoch
- `step`: Current training step
- `mode`: Training mode (FO/ZO/Instruct)
- `scope`: Parameter scope (full/reduced)
- `q`: Number of ZO query directions
- `lr`: Learning rate
- `batch_size`: Batch size
- `optimizer`: Optimizer type
- `bp_interval`: BP gradient computation interval
- `loss`: Training loss
- `grad_norm`: Gradient norm
- `eval_loss`: Evaluation loss (if available)

### Log File Locations

- Training logs: `logs/{run_name}_{timestamp}/`
- Experiment logs: `experiment_logs/`
- CSV files: `logs/{run_name}_{timestamp}/*.csv`
- Plots: `plots/`

### Environment Setup

Most scripts require:
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- Matplotlib (for plotting)
- Conda environment: `MeZO` or `speechbrain`

Activate environment:
```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate MeZO
```

---

## Quick Start

1. **Setup environment:**
   ```bash
   source /opt/anaconda3/etc/profile.d/conda.sh
   conda activate MeZO
   ```

2. **Run a simple training experiment:**
   ```bash
   python reproduce_zo_paper_1106.py \
       --mode ZO \
       --scope full \
       --query_budget_q 64 \
       --learning_rate 0.001 \
       --batch_size 2 \
       --dataset cosmopedia-100k
   ```

3. **Run two experiments with different data usage:**
   ```bash
   ./run_two_experiments.sh
   ```

4. **Plot results:**
   ```bash
   python plot_two_experiments.py
   ```

---

## Notes

- All scripts use relative paths for data and logs
- CSV logs are automatically created in `logs/` directory
- Checkpoints are saved periodically during training
- Evaluation is optional and can be disabled by setting `eval_interval=0`
- GPU assignment can be controlled via `CUDA_VISIBLE_DEVICES` environment variable

---

## Version History

- **1106**: Latest version with full features (dataset splitting, shared ZO data, evaluation)
- **1105**: Previous version with evaluation functionality
- **new**: Alternative implementation variant
- **withbp**: Version with BP gradient support
- **original**: Base version for reproducing ZO paper

---

Last Updated: 2025-11-09

