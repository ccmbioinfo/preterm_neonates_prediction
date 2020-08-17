
from glob import glob
import os
from PIL import Image
import numpy as np
import argparse

def to_uint8(data):
    data = data.astype(np.float)
    data[data < 0] = 0
    return ((data-data.min())*255.0/data.max()).astype(np.uint8)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True, help='name of input directory containing numpy arrays')
    parser.add_argument('--out_dir', type=str, required=True,  help='name of output directory')
    parser.add_argument('--file_suffix', type=str, required=False,  help='name of suffix to append to file (eg. input, mask')
    parser.add_argument('--folder_suffix', type=str, required=False,  help='name of suffix to append to file (eg. V01, V02')
    return parser

def main(in_dir,out_dir, folder_suffix = None, file_suffix = None):
    ""


    glob_input = glob(in_dir + '/*')
    if not os.path.exists(out_dir): os.makedirs(out_dir)


    for f in glob_input:
        print(f)
        arr = np.load(f)
        pid = f.split('_')[-2]
        print(pid)
        if folder_suffix is not None:
            pid_out = os.path.join(out_dir, pid + '_' + folder_suffix)
        else:
            pid_out = os.path.join(out_dir, pid)
        if not os.path.exists(pid_out): os.makedirs(pid_out)
        for s in range(arr.shape[0]):
            img = Image.fromarray(to_uint8(arr[s]), "P")
            s = '0' + str(s) if len(str(s)) < 2 else str(s) # account for single digit formatting
            if file_suffix is not None:
                out = os.path.join(pid_out, '{}_{}_{}.tif'.format(pid, s, file_suffix))
            else:
                out = os.path.join(pid_out, '{}_{}.tif'.format(pid, s))
            img.save(out)

if __name__ == '__main__':
    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))
    #
    # root_dir = '/Users/delvin/Downloads/test_pipeline/test_pipeline_out/'
    # image_dir = os.path.join(root_dir, 'axial')
    # mask_dir = os.path.join(root_dir, 'axial_seg_masks')
    # out_dir = '/Users/delvin/Downloads/test_pipeline/imgs'

    main(in_dir = args.in_dir, out_dir = args.out_dir, folder_suffix = args.folder_suffix, file_suffix = args.file_suffix)

    print('Done!')
    # main(root_dir = root_dir, image_dir = image_dir, masks_dir= mask_dir, out_dir = out_dir, seg_masks_dir = seg_masks_dir)
    # python3 dump_npy_to_tif.py \
    #     --in_dir=/Users/delvin/Downloads/test_pipeline/test_pipeline_out/axial \
    #     --out_dir=/Users/delvin/Downloads/test_pipeline/test_pipeline/imgs --folder_suffix='V01' --file_suffix='imgs'

    # python3 dump_npy_to_tif.py \
    #     --in_dir=/Users/delvin/Downloads/test_pipeline/test_pipeline_out/axial_seg_masks \
    #     --out_dir=/Users/delvin/Downloads/test_pipeline/test_pipeline/masks --folder_suffix='V01' --file_suffix='masks'

#!/bin/bash
# for dir in histo_norm_v01 kde_norm_v01

# for dir in kde_norm_v01
#     do
#     python3 dump_npy_to_tif.py \
#         --in_dir=/home/delvinso/neuro/output/$dir/axial_seg_masks \
#         --out_dir=/home/delvinso/neuro/output/$dir/tifs/masks --file_suffix='masks' --folder_suffix='V01'
#
#     python3 dump_npy_to_tif.py \
#             --in_dir=/home/delvinso/neuro/output/$dir/axial \
#                       --out_dir=/home/delvinso/neuro/output/$dir/tifs/imgs --file_suffix='imgs' --folder_suffix='V01'
#     done
#
# for dir in histo_norm_v02 kde_norm_v02
# for dir in kde_norm_v02
#     do
#     python3 dump_npy_to_tif.py \
#         --in_dir=/home/delvinso/neuro/output/$dir/axial_seg_masks \
#         --out_dir=/home/delvinso/neuro/output/$dir/tifs/masks --file_suffix='masks' --folder_suffix='V02'
#
#     python3 dump_npy_to_tif.py \
#             --in_dir=/home/delvinso/neuro/output/$dir/axial \
#                       --out_dir=/home/delvinso/neuro/output/$dir/tifs/imgs --file_suffix='imgs' --folder_suffix='V02'
#     done