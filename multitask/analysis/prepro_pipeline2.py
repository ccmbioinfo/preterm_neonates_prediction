#TODO logging
import os
from glob import glob
import nibabel as nib
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
# utils
from skimage.exposure import equalize_hist, rescale_intensity
import img_helpers as ih


def sample_stack(stack, rows=7, cols=7, start_with=1, show_every=1, transpose = True, cmap = "gray"):
    # adapted from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python
    fig, ax = plt.subplots(rows, cols, figsize=[20, 20])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        if transpose:
            s = stack[ind, :, :].T
        else:
            s = stack[ind, :, :]
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(s, cmap=cmap)
        ax[int(i / rows), int(i % rows)].axis('off')
        fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type = str, required = True, help = 'root directory, eg. neuro/data/ubc_misc')
    parser.add_argument('--image_dir', type=str, required=True, help='name of input image directory, assumed to be under root_path')
    parser.add_argument('--mask_dir', type=str, required=True, help='name of input mask directory, assumed to be under root_path')
    parser.add_argument('--out_dir', type=str, required=True,  help='name of output direcotry')
    parser.add_argument('--seg_masks_dir', type=str, required=False, help='name of segmentation masks to be resized')
    parser.add_argument('--prefix', type=str, default = 'birth', help = 'prefix to append to final numpy files (defaults to \'birth\' for UBC cohort')
    return parser

def main(root_dir, image_dir, masks_dir, out_dir, prefix, seg_masks_dir = None):

    out_dir = os.path.join(root_dir, out_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    out_n4b = os.path.join(out_dir, 'n4bias_corrected')
    if not os.path.exists(out_n4b): os.mkdir(out_n4b)

    out_norm = os.path.join(out_dir, 'normalized')
    if not os.path.exists(out_norm): os.mkdir(out_norm)

    out_figs= os.path.join(out_dir, 'figures')
    if not os.path.exists(out_figs): os.mkdir(out_figs)

    for view in ['axial', 'sagittal', 'coronal']:
        out_view = os.path.join(out_dir, view)
        if not os.path.exists(out_view): os.mkdir(out_view)

    for view in ['axial', 'sagittal', 'coronal']:
        out_view = os.path.join(out_dir, '{}_seg_masks'.format(view))
        if not os.path.exists(out_view): os.mkdir(out_view)

    # glob the niftis
    imgs = glob(os.path.join(root_dir, image_dir ,  '*.nii*'))
    imgs = [f for f in imgs if 'TSE' not in f]

    # masks = glob(os.path.join(root_dir, masks_dir, '*.nii*'))
    ############################################## COMMENT OUT #############################################
    # imgs = [f for f in imgs if 'V01' in f or 'V11' in f]
    #######################################################################################################


    print('There are {} images in the directory.'.format(len(imgs)))

    # names between the images and masks are consistent up to the last underscore so we can use this to our advantage
    fns = pd.Series(imgs).str.extract(r'(BC.*_)').values.tolist()

    for f in fns:
        f = f[0]

        # if not all(pd.Series(f).str.contains('V01|V11')): next
        print('Current Image: {}'.format(f))

        # get the image and mask
        img = os.path.join(root_dir, image_dir, f + 'stripped.nii')
        mask = os.path.join(root_dir, masks_dir, f + 'labels.nii')
        seg_mask = os.path.join(root_dir, seg_masks_dir, f.split('_')[0] + '.nii' )


        # TODO: make code chunk below into a function
        # input_image = sitk.ReadImage(img)
        # input_image_np = sitk.GetArrayFromImage(input_image)
        # print('Shape of Original Image: {}'.format(input_image_np.shape))
        #
        # cast_image = sitk.Cast(input_image, sitk.sitkFloat32)
        # mask_image = sitk.ReadImage(mask, sitk.sitkUInt8)
        # corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # corrector.SetMaximumNumberOfIterations([200] * 4) # iters * fitting levels
        # output_image = corrector.Execute(cast_image, mask_image)
        #
        # out_f = os.path.join(out_n4b, f + 'BC' + '.nii.gz')
        # print('Saving N4 Bias Field Corrected Image to: {}'.format(out_f))
        # sitk.WriteImage(output_image, out_f)
        # n4_corrected_np = sitk.GetArrayFromImage(output_image)
        # print('Shape after N4 Bias: {}'.format(n4_corrected_np.shape))


        # read in nifti and convert to np array
        input_image = nib.load(img)
        input_image_np = np.asarray(input_image.dataobj)


        # TODO: save the output file
        # print(input_image_np.mean())
        pxl_arr = ih.normalize_volume(input_image_np)
        # print(pxl_arr.mean())


        # plot the original and normalized images, separately
        og_stack = sample_stack(np.transpose(input_image_np, (2, 1, 0)), transpose = False,
                                start_with = 70, show_every = 2, cmap = 'gray')
        og_stack.savefig(os.path.join(out_norm, f.split('_')[0] + '_og.png'))
        og_stack.clear()
        stretched_stack = sample_stack(np.transpose(pxl_arr, (2, 1, 0)), transpose = False,
                                       start_with = 70, show_every = 2, cmap = 'gray')
        # stretched_stack.show()
        stretched_stack.savefig(os.path.join(out_norm, f.split('_')[0] + '_stretched.png'))
        stretched_stack.clear()

        print('Shape after Normalization: {}'.format(pxl_arr.shape)) # should not have changed!

        view_dict = {0:'sagittal', 1:'coronal', 2:'axial', -1:'keep'}

        # load segmentation mask
        try:
            seg_mask_image = nib.load(seg_mask)
            print('Segmentation Mask found for {}'.format(f))
            mask_arr = np.asarray(seg_mask_image.dataobj)
        except FileNotFoundError:
            print('No Segmentation Mask for {}'.format(f))
            mask_arr = np.zeros(pxl_arr.shape)

        pad = True # passes to fit image_mask to pad instead of resizing and interpolating the images
        for axis in range(3):
            # find first and last slice which exceeds the mean intensity
            # TODO: if mask is null then mask = (256 x 256 x 256)
            print('Reslicing scans {}..'.format(view_dict[axis]))
            if axis == 0:
                first, last = ih.find_start_end(pxl_arr, verbose = True, axis = axis)
                # new_arr = ih.fit_resize_image(pxl_arr[first:last, :, :], axis = axis)
                new_arr, new_mask = ih.fit_image_mask(pxl_arr[first:last, :, :],
                                                      axis = axis,
                                                      mask = mask_arr[first:last, :, :],
                                                      pad = pad)
                new_mask[new_mask != 0] = 1
                # sag_np = new_arr[:, :, :, 0]
                # sag_mask = new_mask[:, :, :, 0]
                # sag_first, sag_last = first, last
            elif axis == 1:
                first, last = ih.find_start_end(pxl_arr, verbose = True, axis = axis)
                # new_arr = ih.fit_resize_image(pxl_arr[:, first:last, :], axis = axis)
                new_arr, new_mask = ih.fit_image_mask(pxl_arr[:, first:last, :], axis = axis, mask = mask_arr[:, first:last, :],
                                                      pad = pad )
                new_mask[new_mask != 0] = 1
                # cor_np = new_arr[:, :, :, 0]
                # cor_mask = new_mask[:, :, :, 0]
                # cor_first, cor_last = first, last
            elif axis == 2:
                first, last = ih.find_start_end(pxl_arr, verbose = True, axis = axis)
                # new_arr = ih.fit_resize_image(pxl_arr[:, :, first:last], axis = axis)
                new_arr, new_mask = ih.fit_image_mask(pxl_arr[:, :, first:last], axis = axis, mask = mask_arr[:, :, first:last],
                                                      pad = pad)
                new_mask[new_mask != 0] = 1
                axi_np = new_arr[:, :, :, 0]
                axi_mask = new_mask[:, :, :, 0]
                axi_first, axi_last = first, last
            new_arr = new_arr[:, :, :, 0]
            new_mask = new_mask[:, :, :,0]

            # print('Shape of ')
            # saving down image
            new_fn = f'{prefix}_' + f.split('_')[0] + '_ss.npy'
            new_fn_abs = os.path.join(out_dir,view_dict[axis] , new_fn)
            print(new_fn_abs)

            np.save(arr = new_arr, file = new_fn_abs)

            # saving down mask
            new_mask_fn = f'{prefix}_' + f.split('_')[0] + '_mask.npy'
            new_mask_fn_abs = os.path.join(out_dir, '{}_seg_masks'.format(view_dict[axis]), new_mask_fn)
            print(new_mask_fn_abs)

            np.save(arr = new_mask, file = new_mask_fn_abs)

        #
        # axi_first, axi_last
        # sample_stack(input_image_np, rows = 4, cols =4, start_with = 89, show_every=2, transpose = False)
        # sample_stack(axi_np, rows = 4, cols =4, start_with = 0, show_every=2, transpose = False)
        # sample_stack(cor_np, rows = 4, cols =4, start_with = 10, show_every=2, transpose = False)
        # sample_stack(sag_np, rows = 4, cols =4, start_with = 10, show_every=2, transpose = False)

        # normalizing may have made some values < 0 so we 'reset' it to black
        axi_np[axi_np < 0] = 0
        # black out (as opposed to white out b/c of similar intensity) masked areas
        masked_array = np.where(axi_mask==0, axi_np , 0)
        og_masked_arr = np.where(mask_arr == 0, input_image_np, 0)
        # take a middle slice for plotting

        slice = int(axi_np.shape[0]/2)

        # list of images for plotting
        # TODO: make this into a dictionary?
        imgs = [input_image_np[:, :, axi_first + slice ].T, # sitk
                #n4_corrected_np[axi_first + slice ], #sitk
                pxl_arr[:, :, axi_first + slice ].T, # nib, x, y z
                og_masked_arr.transpose(2, 1, 0)[axi_first + slice], # compatability b/w sitk and nib
                masked_array[slice], # preprocessed, s x 256 x 256
                mask_arr[:, :, axi_first + slice].T, # originally read in  nib mask
                new_mask[slice], # preprocessed, s x 256 x 256
                axi_np[slice],# preprocessed, s x 256 x 256
                np.mean(mask_arr, axis = 2).T, # original mask average
                np.mean(new_mask, axis = 0),# new mask average
                ]

        titles = ['original', #'n4 bias corrected',
                  'normalized', 'original masked',
                  'cropped, resized, masked', 'original mask', 'cropped, resized mask',
                  'cropped, resized', 'og mask average', 'new mask average']
        fig, axs = plt.subplots(nrows=1, ncols=9, figsize=(40,10),
                                gridspec_kw = {'wspace':0, 'hspace':0})
        for i, ax in enumerate(axs.flatten()):
            plt.sca(ax)
            plt.title("{}: \n{}".format(titles[i], imgs[i].shape))
            plt.imshow(imgs[i], cmap = 'gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.tight_layout()
        plt.suptitle(f)
        # plt.show()
        plt.savefig(os.path.join(out_figs,'{}_figure_check.png'.format(f.split('_')[0])))



if __name__ == '__main__':
    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))

    main(root_dir = args.root_path, image_dir = args.image_dir, masks_dir= args.mask_dir,
         out_dir = args.out_dir, seg_masks_dir = args.seg_masks_dir, prefix = args.prefix)

