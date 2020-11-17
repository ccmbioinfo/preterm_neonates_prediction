#!/bin/bash
#PBS -q gpu  -l nodes=1:ppn=1:gpus=1,vmem=60g,mem=60g,walltime=12:00:00
#module load python/3.5.6_torch
run_name=${run}
view=${view}

module load anaconda
#if [ ${view} == "axial" ]
#then
#    module load glibc/2.14 # honestly not sure why this needs module laod glibc if view = axial but okay
#fi
#module load glibc/2.14
source activate nnet

#echo 'Run name:${run_name}' 
python3 /home/delvinso/neuro/analysis/net/train_and_eval.py \
     --root_path='/home/delvinso/neuro' \
     --outcome='multitask' \
     --view=${view}  \
     --num_epochs=300 \
     --manifest_path='/home/delvinso/neuro/output/ubc_npy_outcomes_v3_ss_PD.csv' \
     --model_out='/home/delvinso/neuro/output/models'   \
     --metrics_every_iter=20 \
     --task=multitask \
     --run_name="${run_name}"

