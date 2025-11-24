#!/bin/bash

# ============================================================
# NeFL Heterogeneity Experiments (Simplified)
# ============================================================
# This script runs experiments with random_fast communication
# scenario for different data heterogeneity levels.
# ============================================================

echo "========================================================"
echo "NeFL Experiments - random_fast scenario only"
echo "========================================================"
echo "Start time: $(date)"
echo ""

# Fixed parameters
FIXED_PARAMS="--noniid noniid --epochs 300 --lr 1e-2 --rs 2 --train_ratio 16-1 --num_experiment 1 --model_name resnet18 --device_id 0 --learnable_step True --pretrained false"

# Device ratios to test
DEVICE_RATIOS=("S9-W1" "S8-W2" "S7-W3" "S6-W4" "S5-W5" "S4-W6" "S3-W7" "S2-W8" "S1-W9")

# Communication scenario (fixed to random_fast)
COMM="random_fast"

# Counter for tracking progress
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

# Calculate total number of experiments
# 3 class settings × 9 device ratios = 27 experiments
TOTAL_EXPERIMENTS=$((3 * 9))
echo "Total experiments to run: $TOTAL_EXPERIMENTS"
echo "Communication scenario: $COMM (fixed)"
echo ""

# Create log directory
LOG_DIR="./experiment_logs"
mkdir -p $LOG_DIR

# ============================================================
# Experiment Set 1: random_fast with class=2
# ============================================================
echo "========================================================"
echo "Experiment Set 1: class=2 (High data heterogeneity)"
echo "========================================================"
echo ""

CLASS_PER_CLIENT=2

echo "----------------------------------------"
echo "Communication Scenario: $COMM"
echo "Class per client: $CLASS_PER_CLIENT"
echo "----------------------------------------"

for DEVICE_RATIO in "${DEVICE_RATIOS[@]}"; do
    COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
    
    echo ""
    echo "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] Running experiment:"
    echo "  Device Ratio: $DEVICE_RATIO"
    echo "  Comm Scenario: $COMM"
    echo "  Class per client: $CLASS_PER_CLIENT"
    echo "  Start time: $(date +%H:%M:%S)"
    
    # Log file name
    LOG_FILE="$LOG_DIR/exp_comm-${COMM}_device-${DEVICE_RATIO}_class-${CLASS_PER_CLIENT}.log"
    
    # Run experiment
    python NeFL-toy.py \
        $FIXED_PARAMS \
        --class_per_each_client $CLASS_PER_CLIENT \
        --device_ratio $DEVICE_RATIO \
        --comm_scenario $COMM \
        2>&1 | tee $LOG_FILE
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $EXIT_CODE"
        echo "  Check log: $LOG_FILE"
    fi
    
    echo "  End time: $(date +%H:%M:%S)"
    echo ""
done

# ============================================================
# Experiment Set 2: random_fast with class=4
# ============================================================
echo "========================================================"
echo "Experiment Set 2: random_fast Communication (class=4)"
echo "========================================================"
echo ""

COMM="random_fast"
CLASS_PER_CLIENT=4

echo "----------------------------------------"
echo "Communication Scenario: $COMM"
echo "Class per client: $CLASS_PER_CLIENT"
echo "----------------------------------------"

for DEVICE_RATIO in "${DEVICE_RATIOS[@]}"; do
    COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
    
    echo ""
    echo "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] Running experiment:"
    echo "  Device Ratio: $DEVICE_RATIO"
    echo "  Comm Scenario: $COMM"
    echo "  Class per client: $CLASS_PER_CLIENT"
    echo "  Start time: $(date +%H:%M:%S)"
    
    # Log file name
    LOG_FILE="$LOG_DIR/exp_comm-${COMM}_device-${DEVICE_RATIO}_class-${CLASS_PER_CLIENT}.log"
    
    # Run experiment
    python NeFL-toy.py \
        $FIXED_PARAMS \
        --class_per_each_client $CLASS_PER_CLIENT \
        --device_ratio $DEVICE_RATIO \
        --comm_scenario $COMM \
        2>&1 | tee $LOG_FILE
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $EXIT_CODE"
        echo "  Check log: $LOG_FILE"
    fi
    
    echo "  End time: $(date +%H:%M:%S)"
    echo ""
done

# ============================================================
# Experiment Set 3: random_fast with class=6
# ============================================================
echo "========================================================"
echo "Experiment Set 3: random_fast Communication (class=6)"
echo "========================================================"
echo ""

COMM="random_fast"
CLASS_PER_CLIENT=6

echo "----------------------------------------"
echo "Communication Scenario: $COMM"
echo "Class per client: $CLASS_PER_CLIENT"
echo "----------------------------------------"

for DEVICE_RATIO in "${DEVICE_RATIOS[@]}"; do
    COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
    
    echo ""
    echo "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] Running experiment:"
    echo "  Device Ratio: $DEVICE_RATIO"
    echo "  Comm Scenario: $COMM"
    echo "  Class per client: $CLASS_PER_CLIENT"
    echo "  Start time: $(date +%H:%M:%S)"
    
    # Log file name
    LOG_FILE="$LOG_DIR/exp_comm-${COMM}_device-${DEVICE_RATIO}_class-${CLASS_PER_CLIENT}.log"
    
    # Run experiment
    python NeFL-toy.py \
        $FIXED_PARAMS \
        --class_per_each_client $CLASS_PER_CLIENT \
        --device_ratio $DEVICE_RATIO \
        --comm_scenario $COMM \
        2>&1 | tee $LOG_FILE
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $EXIT_CODE"
        echo "  Check log: $LOG_FILE"
    fi
    
    echo "  End time: $(date +%H:%M:%S)"
    echo ""
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================================"
echo "All Experiments Completed!"
echo "========================================================"
echo "End time: $(date)"
echo "Total experiments run: $COMPLETED_EXPERIMENTS"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Experiment breakdown:"
echo "  Set 1 (random_fast × 9 ratios × class=2): 9 experiments"
echo "  Set 2 (random_fast × 9 ratios × class=4): 9 experiments"
echo "  Set 3 (random_fast × 9 ratios × class=6): 9 experiments"
echo "  Total: 27 experiments"
echo ""
echo "Check wandb dashboard for results visualization!"
echo "========================================================"
