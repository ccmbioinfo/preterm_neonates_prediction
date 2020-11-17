#!/bin/bash
#PBS -l nodes=1:ppn=1,vmem=50g,mem=50g,walltime=48:00:00
#PBS -joe .

module load anaconda
#source activate intensity_normalization
source activate /hpf/largeprojects/ccmbio/sufkes/conda/envs/neuro
dirname=t2_v01_wholebrain_norm_percentilerescale

#python3 /home/delvinso/neuro/analysis/prepro_pipeline2.py \
#  --root_path=/home/delvinso/neuro/data/ubc_misc \
#  --image_dir=ubc_ss_v01 \
#  --mask_dir=ubc_masks_v01 \
#  --seg_masks_dir=Prem-UBC_WMI/WMI_v01_renamed \
#  --prefix=birth \
#  --out_dir=/home/delvinso/neuro/output/${dirname}
python3 /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/analysis/prepro_pipeline2.py \
  --root_path=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/data/ubc_misc \
  --image_dir=ubc_ss_t2_v01 \
  --mask_dir=ubc_masks_t2_v01 \
  --seg_masks_dir=Prem-UBC_WMI/WMI_v01_renamed \
  --prefix=birth \
  --out_dir=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/output/${dirname}

for dir in ${dirname}
    do
#    python3 /home/delvinso/neuro/analysis/dump_npy_to_tif.py \
#        --in_dir=/home/delvinso/neuro/output/$dir/axial_seg_masks \
#        --out_dir=/home/delvinso/neuro/output/$dir/tifs/masks --file_suffix='masks' --folder_suffix='V01'

    python3 /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/analysis/dump_npy_to_tif.py \
        --in_dir=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/output/$dir/axial_seg_masks \
        --out_dir=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/output/$dir/tifs/masks \
	--file_suffix='masks' \
	--folder_suffix='V01'

#    python3 /home/delvinso/neuro/analysis/dump_npy_to_tif.py \
#        --in_dir=/home/delvinso/neuro/output/$dir/axial \
#        --out_dir=/home/delvinso/neuro/output/$dir/tifs/imgs --file_suffix='imgs' --folder_suffix='V01'
    python3 /hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/analysis/dump_npy_to_tif.py \
        --in_dir=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/output/$dir/axial \
        --out_dir=/hpf/largeprojects/ccmbio/sufkes/preterm_neonates_prediction/multitask/output/$dir/tifs/imgs \
	--file_suffix='imgs' \
	--folder_suffix='V01'
    done


