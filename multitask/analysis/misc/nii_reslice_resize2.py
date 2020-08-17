import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
import img_helpers as ih
import seaborn as sns
import argparse
#
# nii_dir = valid_nii[7]
# nii_dir = os.path.join(ss_dir, nii_dir)
# # load in
# nii = nib.load(nii_dir)
# # retrieve the array
# pxl_arr = np.asarray(nii.dataobj)
# print(pxl_arr.shape)
#
#
# # let's take a look visually to assess where these series of images 'starts' and 'ends'
# # reformat so it is s x 256 x 256
# reslice_arr = ih.reslice(start = 0, end = pxl_arr.shape[2], pxl = pxl_arr)
# reslice_arr.shape
# # plt.imshow(reslice_arr[150, :, :].T, cmap = 'gray')
# ih.sample_stack(reslice_arr, cols = 13, rows = 13)
# # looks liek the brain begins to appear on slice 95 and ends on slice 166
#
#
# # find the first and last slices
# f, l = ih.find_start_end(pxl_arr)
# f, l
# # reslice axially given the first and last slices
# new_arr = ih.reslice(pxl = pxl_arr, start = f,  end = l)
# print(new_arr.shape)
# # visualize the new set of slices, there should be no more black slices remaining
# ih.sample_stack(new_arr, cols = 5, rows = 5)
#
# # reslice
# for f in list(valid_nii):
#     f_abs = ss_dir + f
#     # print(f_abs)
#     # read in the nifti
#     try:
#         nii = nib.load(f_abs)
#     except:
#         print('Error for {}'.format(f_abs))
#         next
#     pxl_arr = np.asarray(nii.dataobj)
#     first, last = ih.find_start_end(pxl_arr)
#     new_arr = ih.reslice(pxl = pxl_arr, start = first,  end = last)
#     new_fn = f.split('.')[0] + '_reslice.npy'
#     new_fn_abs = os.path.join(ss_dir, new_fn)
#     np.save(arr = new_arr, file = new_fn_abs)
#
# t = os.listdir(ss_dir)
# test = np.load(ss_dir + t[-1])
# test.shape
#
# ih.sample_stack(test, cols = 5, rows = 5)
# # ---- for each skull stripped image, get the mean pixel intensities for each slice ----
# ss_int_d = {}
# for f in list(valid_nii):
#     # try:
#         # print(f)
#     f_abs = ss_dir + f
#     # print(f_abs)
#     # read in the nifti
#     nii = nib.load(f_abs)
#     # except:
#     #     print('Error for: {}'.format(f_abs))
#     #     next
#     # extract the array
#     pxl_arr = np.asarray(nii.dataobj)
#     # compute the mean intensity for each slice.
#     # series_mean_intensity() returns an array where each element is the
#     # mean intensity of the ith slice
#     intensity = ih.series_mean_intensity(pxl_arr)
#     # remove the black images
#     intensity = intensity[intensity > 0]
#     ss_int_d[f] = intensity
#
# intens_df = pd.DataFrame.from_dict(ss_int_d, orient = 'index').reset_index()\
#     .melt(id_vars = 'index',var_name = 'slice',value_name = 'mean_intensity').dropna()
# intens_df['id'] = intens_df.iloc[:,0].map(lambda x: x.split('_')[0])
# intens_df['fn'] = intens_df.iloc[:, 0]
#
# intens_df.groupby(['id'])['fn'].size()
# # iterate through the IDs
#
#
#
# for fn in intens_df.fn.drop_duplicates():
#     print(fn)
#     subset = intens_df[intens_df['fn'] == fn]
#     # subset['log_mean_intensity'] = np.log10(subset.mean_intensity)
#
#     # Draw the density plot
#     sns.distplot(subset.mean_intensity, hist=False, kde=True,
#                  kde_kws={'linewidth': 1.5})
#                  # label=airline)
#
#
# # for id in intens_df.id.drop_duplicates():
# #     print(id)
# #     subset = intens_df[intens_df['id'] == id]
# #     # subset['log_mean_intensity'] = np.log10(subset.mean_intensity)
# #
# #     # Draw the density plot
# #     sns.distplot(subset.mean_intensity, hist=False, kde=True,
# #                  kde_kws={'linewidth': 1.5})
# #                  # label=airline)
#
# # Plot formatting
# plt.legend(prop={'size': 16}, title='Patient ID')
# plt.title('Distribution of Mean Pixel Intensities.')
# plt.xlabel('Mean Pixel Intensity')
# plt.ylabel('Density')
# plt.show()
# # plt.savefig('distribution_mean_pxl_intensities.png')
#
#
# # ---- debugging ----
# px_list = []
# for s in range(pxl_arr.shape[2]):
#     this_slice = pxl_arr[:, :, s]
#     flat = np.ndarray.flatten(this_slice)
#     px_list.append(np.mean(flat))
#
# intensity_arr = np.array(px_list)
# # mean of slices that are not fully black. could also do something like proportion of image that is black?
# mean_intensity = np.mean(intensity_arr[intensity_arr > 0])
# intensity_arr[intensity_arr > 0].shape[0] / intensity_arr.shape[0]  # proportion of slices that are NOT black
#
# mean_intensity
# f_idx = np.argmax(intensity_arr >= mean_intensity)
# f_idx
# # reverse the array. the first slice greater than or less than the mean pixel intensity is the last slice.
# l_idx = intensity_arr.shape[0] - np.argmax(intensity_arr[::-1] >= mean_intensity)
#
#
# new_arr = ih.reslice(pxl = pxl_arr, start = f_idx, end = l_idx)
# new_arr.shape[0]
# ih.sample_stack(new_arr, rows = 6, cols = 6 )

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='name of input image directory, abs path')
    parser.add_argument('--manifest_path', type=str, required=True, help='name of manifest directory, abs path')
    parser.add_argument('--out_dir', type=str, required=True, help='name of output directory')
    # TODO
    parser.add_argument('--axis', type=int, required=True, help='0 = sagittal, 1 = coronal, 2 = axial, -1 = keep the same')
    return parser

def main(out_dir, mn_path, image_dir, axis = 2):

    if not os.path.exists(out_dir):
        print("{} doesn't exist yet. Creating...".format(out_dir))
        os.makedirs(out_dir)

    mn = pd.read_csv(mn_path)
    # ---- the skull stripped niftis ----

    valid_nii = mn.ss_nii.dropna()
    print('# of Valid niftis: {}'.format(valid_nii.shape[0]))

    # ---- reslice ----
    view_dict = {0:'sagittal', 1:'coronal', 2:'axial', -1:'keep'}
    print('Reslicing scans {}..'.format(view_dict[axis]))
    f_list = []
    for f in list(valid_nii):

        f_abs = os.path.join(image_dir, f)
        # print(f_abs)
        # read in the nifti
        try:
            print('Reading in and processing {}..'.format(f_abs))
            nii = nib.load(f_abs)
        except KeyboardInterrupt:
            print('Error for {}'.format(f_abs))
            next
        # convert to array
        pxl_arr = np.asarray(nii.dataobj)
        print(pxl_arr.shape)
        # find first and last slice which exceeds the mean intensity
        first, last = ih.find_start_end(pxl_arr, verbose = True, axis = axis)

        # new_arr = ih.reslice(pxl = pxl_arr, start = first,  end = last, verbose = False, axis = axis)

        if axis == 0:
            new_arr = ih.fit_resize_image(pxl_arr[first:last, :, :], axis = axis)
        elif axis == 1:
            new_arr = ih.fit_resize_image(pxl_arr[:, first:last, :], axis = axis)
        elif axis == 2:
            new_arr = ih.fit_resize_image(pxl_arr[:, :, first:last], axis = axis)
        new_arr = new_arr[:, :, :, 0]

        # create new file name
        new_fn = 'birth_' + f.split('_')[0] + '_ss.npy'
        new_fn_abs = os.path.join(out_dir, new_fn)
        print(new_fn_abs)

        np.save(arr = new_arr, file = new_fn_abs)
        f_list.append(new_fn_abs)

    dir_name = '{}_dir_ss'.format(view_dict[axis])
    df = pd.DataFrame({dir_name : f_list})
    print(f_list)
    df['study_id'] = df[dir_name].apply(lambda x: x.split('_')[4])
    print(df)
    if all(pd.Series(dir_name).isin(mn.columns)): mn.drop(columns = dir_name, inplace = True)
    mn_jn = mn.set_index('study_id').join(df.set_index('study_id')).reset_index()

    assert(mn_jn.shape[0] == mn.shape[0])
    mn_jn.to_csv(mn_path, index = False)

    print('Manifest saved to {}'.format(mn_path))
    print('Numpy arrays saved to {}'.format(out_dir))

if __name__ == '__main__':
    # with lsq normalized

    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))


    # out_dir = '/home/dso/Documents/Projects/neuro/output/ubc_3d/axial_ss_lsq/'
    # mn_path = '/Users/delvin/Documents/Projects/neuro/output/ubc_npy_outcomes_v3_ss.csv'
    # ss_dir = '/home/delvinso/neuro/data/ubc_misc/normalized/all/lsq'


    # out_dir = '/Users/delvin/Documents/Projects/neuro/output/ubc_3d/axial_ss/'
    # mn_path = '/Users/delvin/Documents/Projects/neuro/output/ubc_npy_outcomes_v3_ss.csv'
    # ss_dir = '/Users/delvin/Documents/Projects/neuro/output/ubc_misc/ubc_ss/'

    # out_dir = '/home/delvinso/neuro/output/ubc_3d/axial_ss/'
    # mn_path = '/home/delvinso/neuro/output/ubc_npy_outcomes_v3_ss.csv'
    # ss_dir = '/home/delvinso/neuro/data/ubc_misc/ubc_ss/'

    main(out_dir = args.out_dir, mn_path = args.manifest_path, image_dir = args.image_dir, axis = args.axis)





