#!/bin/bash
#PBS -l nodes=1:ppn=2,vmem=40g,mem=40g,gres=localhd:100,walltime=05:00:00

module load python/3.7.0

cd /home/delvinso/neuro/analysis/eda

python3 parse_dicoms.py --root_path='/home/delvinso/neuro/data' --pattern='.*' --outname='dcm_info' --save_to_disk --verbose



