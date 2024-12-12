#!/bin/bash
#PBS -l walltime=100:00:00


cd $PBS_O_WORKDIR


module list

pwd

echo $CUDA_VISIBLE_DEVICES

PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# Load local python environment
source activate earthquakeNPP


python train_stpp.py --data $data --model $model --tpp neural --l2_attn --tol $tol --seed $seed