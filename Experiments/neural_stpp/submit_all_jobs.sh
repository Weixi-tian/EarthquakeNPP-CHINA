#!/bin/bash


qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o ETAS_25_output_seed_0.txt -e ETAS_25_error_seed_0.txt -v data=ETAS_25,tol=1e-3,model=attncnf,seed=0 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o Japan_25_output_seed_0.txt -e Japan_25_error_seed_0.txt -v data=Japan_25,tol=1e-3,model=attncnf,seed=0 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o ETAS_incomplete_25_output_seed_0.txt -e ETAS_incomplete_25_error_seed_0.txt -v data=ETAS_incomplete_25,tol=1e-3,model=attncnf,seed=0 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SaltonSea_10_output_seed_0.txt -e SaltonSea_10_error_seed_0.txt -v data=SaltonSea_10,tol=1e-3,model=attncnf,seed=0 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SanJac_10_output_seed_0.txt -e SanJac_10_error_seed_0.txt -v data=SanJac_10,tol=1e-3,model=attncnf,seed=0 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o WHITE_06_output_seed_0.txt -e WHITE_06_error_seed_0.txt -v data=WHITE_06,tol=1e-3,model=attncnf,seed=0 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_20_output_seed_0.txt -e SCEDC_20_error_seed_0.txt -v data=SCEDC_20,tol=1e-3,model=attncnf,seed=0 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_25_output_seed_0.txt -e SCEDC_25_error_seed_0.txt -v data=SCEDC_25,tol=1e-3,model=attncnf,seed=0 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_30_output_seed_0.txt -e SCEDC_30_error_seed_0.txt -v data=SCEDC_30,tol=1e-3,model=attncnf,seed=0 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o ComCat_25_output_seed_0.txt -e ComCat_25_error_seed_0.txt -v data=ComCat_25,tol=1e-3,model=attncnf,seed=0 job.sh;



qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o ETAS_25_output_seed_1.txt -e ETAS_25_error_seed_1.txt -v data=ETAS_25,tol=1e-3,model=attncnf,seed=1 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o Japan_25_output_seed_1.txt -e Japan_25_error_seed_1.txt -v data=Japan_25,tol=1e-3,model=attncnf,seed=1 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=1 -o ETAS_incomplete_25_output_seed_1.txt -e ETAS_incomplete_25_error_seed_1.txt -v data=ETAS_incomplete_25,tol=1e-3,model=attncnf,seed=1 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SaltonSea_10_output_seed_1.txt -e SaltonSea_10_error_seed_1.txt -v data=SaltonSea_10,tol=1e-3,model=attncnf,seed=1 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SanJac_10_output_seed_1.txt -e SanJac_10_error_seed_1.txt -v data=SanJac_10,tol=1e-3,model=attncnf,seed=1 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o WHITE_06_output_seed_1.txt -e WHITE_06_error_seed_1.txt -v data=WHITE_06,tol=1e-3,model=attncnf,seed=1 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_20_output_seed_1.txt -e SCEDC_20_error_seed_1.txt -v data=SCEDC_20,tol=1e-3,model=attncnf,seed=1 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_25_output_seed_1.txt -e SCEDC_25_error_seed_1.txt -v data=SCEDC_25,tol=1e-3,model=attncnf,seed=1 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_30_output_seed_1.txt -e SCEDC_30_error_seed_1.txt -v data=SCEDC_30,tol=1e-3,model=attncnf,seed=1 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o ComCat_25_output_seed_1.txt -e ComCat_25_error_seed_1.txt -v data=ComCat_25,tol=1e-3,model=attncnf,seed=1 job.sh;

qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=1 -o ETAS_25_output_seed_2.txt -e ETAS_25_error_seed_2.txt -v data=ETAS_25,tol=1e-3,model=attncnf,seed=2 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o Japan_25_output_seed_2.txt -e Japan_25_error_seed_2.txt -v data=Japan_25,tol=1e-3,model=attncnf,seed=2 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=1 -o ETAS_incomplete_25_output_seed_2.txt -e ETAS_incomplete_25_error_seed_2.txt -v data=ETAS_incomplete_25,tol=1e-3,model=attncnf,seed=2 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SaltonSea_10_output_seed_2.txt -e SaltonSea_10_error_seed_2.txt -v data=SaltonSea_10,tol=1e-3,model=attncnf,seed=2 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SanJac_10_output_seed_2.txt -e SanJac_10_error_seed_2.txt -v data=SanJac_10,tol=1e-3,model=attncnf,seed=2 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o WHITE_06_output_seed_2.txt -e WHITE_06_error_seed_2.txt -v data=WHITE_06,tol=1e-3,model=attncnf,seed=2 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_20_output_seed_2.txt -e SCEDC_20_error_seed_2.txt -v data=SCEDC_20,tol=1e-3,model=attncnf,seed=2 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_25_output_seed_2.txt -e SCEDC_25_error_seed_2.txt -v data=SCEDC_25,tol=1e-3,model=attncnf,seed=2 job.sh;
# qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o SCEDC_30_output_seed_2.txt -e SCEDC_30_error_seed_2.txt -v data=SCEDC_30,tol=1e-3,model=attncnf,seed=2 job.sh;
qsub -N pascalq -q pascalq -l select=1:ncpus=8:ngpus=2 -o ComCat_25_output_seed_2.txt -e ComCat_25_error_seed_2.txt -v data=ComCat_25,tol=1e-3,model=attncnf,seed=2 job.sh;
