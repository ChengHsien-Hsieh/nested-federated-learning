#!/bin/bash
# Script to compare different frac values with num_users=1000
# Date: 2025-12-08

echo "=============================================="
echo "Experiment: Comparing different frac values"
echo "num_users=1000, noniid, class_per_each_client=2"
echo "=============================================="

# Common parameters
NUM_USERS=1000
NONIID="noniid"
CLASS_PER_CLIENT=2
EPOCHS=300
LR=1e-2
RS=2
NUM_EXP=1
MODEL="resnet18"

# Experiment 1: frac=0.01 (10 clients per round)
echo ""
echo "[Experiment 1/2] frac=0.01 (${NUM_USERS} * 0.01 = 10 clients/round)"
echo "Started at: $(date)"
echo "----------------------------------------------"

python NeFL-toy.py \
    --num_users $NUM_USERS \
    --noniid $NONIID \
    --class_per_each_client $CLASS_PER_CLIENT \
    --frac 0.01 \
    --epochs $EPOCHS \
    --lr $LR \
    --rs $RS \
    --num_experiment $NUM_EXP \
    --model_name $MODEL

echo "Experiment 1 finished at: $(date)"
echo ""

# Experiment 2: frac=0.1 (100 clients per round)
echo "[Experiment 2/2] frac=0.1 (${NUM_USERS} * 0.1 = 100 clients/round)"
echo "Started at: $(date)"
echo "----------------------------------------------"

python NeFL-toy.py \
    --num_users $NUM_USERS \
    --noniid $NONIID \
    --class_per_each_client $CLASS_PER_CLIENT \
    --frac 0.1 \
    --epochs $EPOCHS \
    --lr $LR \
    --rs $RS \
    --num_experiment $NUM_EXP \
    --model_name $MODEL

echo "Experiment 2 finished at: $(date)"
echo ""

echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
