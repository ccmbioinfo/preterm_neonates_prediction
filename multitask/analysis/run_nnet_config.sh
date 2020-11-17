#!/bin/bash
#PBS -q gpu  -l nodes=1:ppn=1:gpus=1,vmem=60g,mem=60g,walltime=12:00:00

# Usage:
#   qsub run_nnet_config.sh -F <JSON config file>

module load anaconda
source activate /hpf/largeprojects/ccmbio/sufkes/conda/envs/neuro
echo "Reading configuration from: $1" 
python3 /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/analysis/net/train_and_eval.py "$1"
