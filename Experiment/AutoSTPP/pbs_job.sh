#!/bin/bash
#PBS -N pascalq
#PBS -q pascalq
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -l walltime=12:00:00

cd $PBS_O_WORKDIR

module list

pwd

echo $CUDA_VISIBLE_DEVICES

source activate earthquakeNPP

make run_stpp_earthquakeNPP config=$CONFIG

