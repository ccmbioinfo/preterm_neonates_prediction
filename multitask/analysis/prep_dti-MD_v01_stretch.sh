#!/bin/bash
#PBS -l nodes=1:ppn=1,vmem=50g,mem=50g,walltime=48:00:00
#PBS -joe .

module load anaconda
source activate /hpf/largeprojects/ccmbio/sufkes/conda/envs/neuro
dirname=dti-MD_v01_wholebrain_norm_percentilerescale

python3 /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/analysis/prepro_pipeline2.py \
  --root_path=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/data/ubc_misc \
  --image_dir=ubc_ss_dti-MD_v01 \
  --mask_dir=ubc_masks_dti_v01 \
  --seg_masks_dir=Prem-UBC_WMI/WMI_v01_renamed \
  --prefix=birth \
  --out_dir=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/output/${dirname}


