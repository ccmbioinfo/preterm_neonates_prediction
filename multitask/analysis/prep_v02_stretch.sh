#!/bin/bash
#PBS -l nodes=1:ppn=1,vmem=50g,mem=50g,walltime=12:00:00
#PBS -joe .
module load anaconda
source activate intensity_normalization
dirname=hardstretch_v02
python3 /home/delvinso/neuro/analysis/prepro_pipeline2.py \
  --root_path=/home/delvinso/neuro/data/ubc_misc \
  --image_dir=ubc_ss_v02 \
  --mask_dir=ubc_masks_v02 \
  --seg_masks_dir=Prem-UBC_WMI/WMI_v02_renamed \
  --prefix=term \
  --out_dir=/home/delvinso/neuro/output/${dirname}


for dir in ${dirname}
    do
    python3 /home/delvinso/neuro/analysis/dump_npy_to_tif.py \
        --in_dir=/home/delvinso/neuro/output/$dir/axial_seg_masks \
        --out_dir=/home/delvinso/neuro/output/$dir/tifs/masks --file_suffix='masks' --folder_suffix='V02'

    python3 /home/delvinso/neuro/analysis/dump_npy_to_tif.py \
            --in_dir=/home/delvinso/neuro/output/$dir/axial \
                      --out_dir=/home/delvinso/neuro/output/$dir/tifs/imgs --file_suffix='imgs' --folder_suffix='V02'
    done

