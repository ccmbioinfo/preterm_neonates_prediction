import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.exposure import equalize_hist, rescale_intensity

def normalize_volume(volume):
    # contrast stretch
    # this gives a window level of 600-200 = 400 and a window width of 200
    # p_btm = np.percentile(volume.flatten()[volume.flatten() != 0], 1)
    # p_top = np.percentile(volume.flatten()[volume.flatten() != 0], 100)
    p_btm = 200
    p_top = 600
    print(p_btm)
    print(p_top)
    volume = rescale_intensity(volume, in_range=(p_btm, p_top))
    # mean normalization
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume


def reslice(pxl, start = 102, end = 162, axis =  2, verbose = True, resize = True):
    """
    :param pxl: input 3D array, retrieved from the nifti
    :param start: index which will become the 0 index of the resliced array
    :param end: index which will become the nth index of the resliced array
    :param view: TBD
    :return:

    returns an s x 256 x 256 array
    """
    # creating a new, resized array of s x 256 x 256 and removing some beginning and end slices

    start_idx = int(start)
    end_idx = int(end)
    print(start_idx)
    print(end_idx)
    # first we create a s x 256 x 256 volume where s is the number of slices after
    # trimming the start and end
    if resize:
        new_arr = np.zeros([end_idx - start_idx, 256, 256])  # s x 256 x 256
    # else:
    #
    #     new_arr = np.zeros([end_idx - start_idx], pxl.shape[1], pxl.shape[2])
    print(new_arr.shape)

    # the chosen start index is now our 0-index
    for s in range(start_idx, end_idx):
        if verbose: print('Slice: {}'.format(s)) # debug
        new_index = s - start_idx
        if verbose: print('New Index: {}'.format(new_index))

        if axis == 2:
            temp_arr = pxl[:, :, s].T # axial
        elif axis == 1:
            temp_arr = np.rot90(pxl[:, s, :], 1) # coronal
        elif axis == 0:
            temp_arr = np.rot90(pxl[s, :, :], 1) # sagittal

        if resize:
            new_arr[new_index, :, :] = cv2.resize(temp_arr,
                                                  dsize = (256, 256),
                                                  interpolation = cv2.INTER_CUBIC)
        # else:
        #     # TODO: this doesn't work because the temp array is size s x 256 x 256
        #     new_arr[new_index, :, :] = temp_arr
    return(new_arr)


def series_mean_intensity(volume, axis = 2):
    """
    Computes the mean pixel intensity for each slice of an input volume, returned as a list.

    :param volume:
    :return: an array of size volume.shape[2]
    """
    if axis not in [0, 1, 2]: raise ValueError

    x = volume
    px_list = []
    for s in range(x.shape[axis]):
        if axis == 0:
            this_slice = x[s, :, :]
        elif axis == 1:
            this_slice = x[:, s, :]
        elif axis == 2:
            this_slice = x[:, :, s]
        flat = np.ndarray.flatten(this_slice)
        px_list.append(np.mean(flat))


    return(np.asarray(px_list))

def find_start_end(volume, verbose = True, axis = 2):
    """
    :param volume:
    :return: first (int) and last (int) , corresponding to the first and last slices with mean pixel intensity
             larger than the mean of all slices excluding black slices
    """
    # x = volume
    # px_list = []
    # for s in range(x.shape[2]):
    #     this_slice = x[:, :, s]
    #     flat = np.ndarray.flatten(this_slice)
    #     px_list.append(np.mean(flat))
    #
    # # what is the mean pixel intensity removing all black images? could have some other threshold
    # px_arr = np.array(px_list)

    px_arr = series_mean_intensity(volume, axis = axis)
    mean_pxl_inten = np.mean(px_arr[px_arr > 0 ])
    print(mean_pxl_inten)
    # find the first slice (because we want it continuously) where the pixel intensity is greater than the average across slices
    first_idx = np.argmax(px_arr >= mean_pxl_inten)
    # last slice before being less than 10
    # reverse the array.
    # the first slice greater than or less than the mean pixel intensity is the last slice.
    # substract it from the length of the array
    last_idx = px_arr.shape[0] - np.argmax(px_arr[::-1] >= mean_pxl_inten)  # )
    if verbose: print('First Slice @ {}, Last Slice @ {}'.format(first_idx, last_idx))

    return(first_idx, last_idx)


def sample_stack(stack, rows=7, cols=7, start_with=1, show_every=1, transpose = True):
    # adapted from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python
    fig, ax = plt.subplots(rows, cols, figsize=[20, 20])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        if transpose:
            s = stack[ind, :, :].T
        else:
            s = stack[ind, :, :]
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(s, cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
        fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def pad_image(data):
    """

    pouria fit utils
    :param data:
    :return:
    """
    max_size = max(data.shape[0], data.shape[1])
    smaller_size = min(data.shape[0], data.shape[1])

    pad_size_l = (max_size - smaller_size) // 2
    pad_size_r = smaller_size + (max_size - smaller_size) // 2

    uniform_data = np.zeros((max_size, max_size))
    if data.shape[0] >= data.shape[1]:
        uniform_data[:, pad_size_l:pad_size_r] = data
    else:
        uniform_data[pad_size_l:pad_size_r, :] = data
    return uniform_data

def fit_image_mask(data, axis, mask = None, pad = True):
    """
    adapted code of
    - crops out brain (thanks to Pouria)
    - if pad, pad else resize to 128 x 128 #TODO make this an argument
    :param data:
    :param axis:
    :param mask:
    :param pad:
    :return:
    """
    print(axis)
    data_resized = []
    mask_resized = []

    if mask is None:
        mask = np.zeros((256, 256, 256))
    for sample in range(data.shape[axis]):
        if axis == 2:
            c_data = np.rot90(data[:, :, sample], 1) # axial
            m_data = np.rot90(mask[:, :, sample], 1)
            # c_data = data[:, :, sample].T # axial
            # m_data = mask[:, :, sample].T
        elif axis == 1:
            c_data = np.rot90(data[:, sample, :], 1)
            m_data = np.rot90(mask[:, sample, :], 1)
        elif axis == 0:
            # c_data = data[sample, :, :]
            c_data = np.rot90(data[sample, :, :], 1) # sagittal
            m_data = np.rot90(mask[sample, :, :], 1)
        bg_mask = c_data != 0
        # print(sum(mask.flatten()))
        if sum(bg_mask.flatten()) == 0: # pure black image
            next
        else:
            top = np.delete(np.where(bg_mask.any(axis=0), bg_mask.argmax(axis=0), -1), -1)
            left = np.delete(np.where(bg_mask.any(axis=1), bg_mask.argmax(axis=1), -1), -1)
            bottom = np.delete(np.where(bg_mask.any(axis=0), c_data.shape[0] - np.flipud(bg_mask).argmax(axis=0) - 1, -1), -1)
            right = np.delete(np.where(bg_mask.any(axis=1), c_data.shape[1] - np.fliplr(bg_mask).argmax(axis=1) - 1, -1), -1)

            left = np.min(left[left >= 0])
            top = np.min(top[top >= 0])
            right = np.max(right[right >= 0])
            bottom = np.max(bottom[bottom >= 0])

            c_data = c_data[top:bottom, left:right]
            c_data = pad_image(c_data)

            m_data = m_data[top:bottom, left:right]
            m_data = pad_image(m_data)
            # c_data = cv2.resize(c_data, dsize=( int(data.shape[1]/ 2), int(data.shape[2]/ 2)), interpolation=cv2.INTER_CUBIC)
            if pad:
                # print(c_data.shape)
                # print(m_data.shape)
                c_data = pad_to_size(c_data)
                m_data = pad_to_size(m_data)
                # print(c_data.shape)
                # print(m_data.shape)
            else:
                c_data = cv2.resize(c_data, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                m_data = cv2.resize(m_data, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            data_resized.append(c_data)
            mask_resized.append(m_data)

    data_resized = np.asarray(data_resized)
    data_resized = np.expand_dims(data_resized, axis=-1)

    mask_resized = np.asarray(mask_resized)
    mask_resized = np.expand_dims(mask_resized, axis=-1)

    if mask is None:
        assert(mask_resized.flatten().sum() == 0)
    return data_resized, mask_resized

def pad_to_size(original, new_size = (128, 128)):
    """
    pads a square 2D image to a new size, keeping it in the center
    :param original:
    :param new_size:
    :return:
    """
    # print(original.shape)
    w, h = original.shape
    # eg. 69 x 69 after cropping.
    to_pad_v = int((new_size[0] - w)/2) # divide by 2 since we want to add to top and bottom to center the image
    to_pad_h = int((new_size[1] - h)/2)

    # eg. 69 needs to be offset. 256 - 69 = 187. int(187/2) = 93
    # 93 * 2 + 69 = 255. so we need to add an additional row vertically
    #
    new_w = to_pad_v*2 + w
    new_h = to_pad_v*2 + h
    if (to_pad_v*2 + h) != new_size[1] :
        offset_h =  new_size[0] - new_h
    else:
        offset_h = 0
    if (to_pad_h*2 + w) != new_size[0]:
        offset_w =  new_size[1] - new_w
    else:
        offset_w = 0

    try:
        padded  = np.pad(original, pad_width = ((to_pad_v,to_pad_v+offset_h), (to_pad_h,to_pad_h+offset_w)),
                         mode = 'constant', constant_values = 0)
    except ValueError:
        # for when there is motion blur such as in BC055
        # https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
        crop_w = new_size[0]
        crop_h = new_size[1]
        start_w = w//2-(crop_w//2)
        start_h = w//2-(crop_h//2)
        padded = original[start_h:start_h+crop_h,start_w:start_w+crop_w]

    w2, h2 = padded.shape
    assert(w2 == new_size[0] & h2 == new_size[1])

    return padded


def fit_image_mask(data, axis, mask = None, mask2 = None, # TODO; make more robust to additional images
                   pad = True):
    """

    - crops out brain (thanks to Pouria)
    - if pad, pad else resize to 256 x 256 #TODO make this an argument
    :param data:
    :param axis:
    :param mask:
    :param pad:
    :return:
    """
    print(axis)
    data_resized = []
    mask_resized = []
    mask2_resized = []

    # TODO: make more robust to addition of new images
    # currently shoehorned in because of time constraints
    if mask is None:
        mask = np.zeros((256, 256, 256))
    if mask2 is None:
        mask2 = np.zeros((256, 256, 256))
    for sample in range(data.shape[axis]):
        if axis == 2:
            c_data = np.rot90(data[:, :, sample], 1) # axial
            m_data = np.rot90(mask[:, :, sample], 1)
            m2_data = np.rot90(mask2[:, :, sample], 1)
            # c_data = data[:, :, sample].T # axial
            # m_data = mask[:, :, sample].T
        elif axis == 1:
            c_data = np.rot90(data[:, sample, :], 1)
            m_data = np.rot90(mask[:, sample, :], 1)
            m2_data = np.rot90(mask2[:, sample, :], 1)
        elif axis == 0:
            # c_data = data[sample, :, :]
            c_data = np.rot90(data[sample, :, :], 1) # sagittal
            m_data = np.rot90(mask[sample, :, :], 1)
            m2_data = np.rot90(mask2[sample, :, :], 1)

        bg_mask = c_data != 0
        # print(sum(mask.flatten()))
        if sum(bg_mask.flatten()) == 0: # pure black image
            next
        else:
            top = np.delete(np.where(bg_mask.any(axis=0), bg_mask.argmax(axis=0), -1), -1)
            left = np.delete(np.where(bg_mask.any(axis=1), bg_mask.argmax(axis=1), -1), -1)
            bottom = np.delete(np.where(bg_mask.any(axis=0), c_data.shape[0] - np.flipud(bg_mask).argmax(axis=0) - 1, -1), -1)
            right = np.delete(np.where(bg_mask.any(axis=1), c_data.shape[1] - np.fliplr(bg_mask).argmax(axis=1) - 1, -1), -1)

            left = np.min(left[left >= 0])
            top = np.min(top[top >= 0])
            right = np.max(right[right >= 0])
            bottom = np.max(bottom[bottom >= 0])

            c_data = c_data[top:bottom, left:right]
            c_data = pad_image(c_data)

            m_data = m_data[top:bottom, left:right]
            m_data = pad_image(m_data)

            m2_data = m2_data[top:bottom, left:right]
            m2_data = pad_image(m2_data)
            # c_data = cv2.resize(c_data, dsize=( int(data.shape[1]/ 2), int(data.shape[2]/ 2)), interpolation=cv2.INTER_CUBIC)
            if pad:
                # print(c_data.shape)
                # print(m_data.shape)
                c_data = pad_to_size(c_data)
                m_data = pad_to_size(m_data)
                m2_data = pad_to_size(m2_data)
                # print(c_data.shape)
                # print(m_data.shape)
            else:
                c_data = cv2.resize(c_data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                m_data = cv2.resize(m_data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                m2_data = cv2.resize(m2_data, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            data_resized.append(c_data)
            mask_resized.append(m_data)
            mask2_resized.append(m2_data)

    data_resized = np.asarray(data_resized)
    data_resized = np.expand_dims(data_resized, axis=-1)

    mask_resized = np.asarray(mask_resized)
    mask_resized = np.expand_dims(mask_resized, axis=-1)

    mask2_resized = np.asarray(mask2_resized)
    mask2_resized = np.expand_dims(mask2_resized, axis=-1)

    # if mask is None:
    #     assert(mask_resized.flatten().sum() == 0)
    return data_resized, mask_resized, mask2_resized

# OLD reslice function using start and end based off of percentages
# def reslice(pxl, start = 0.45, end = 0.75, view = 'axial' ):
#     """
#
#     :param arr:
#     :param start:
#     :param end:
#     :param view:
#     :return:
#     """
#     # creating a new, resized array of s x 256 x 256 and removing some beginning and end slices
#
#     start_idx = int(pxl.shape[2] * start)
#     end_idx = int(pxl.shape[2] * end)
#     print(start_idx)
#     print(end_idx)
#     # first we create a s x 256 x 256 volume where s is the number of slices after
#     # trimming the start and end
#     new_arr = np.zeros([end_idx - start_idx, 256, 256])  # s x 256 x 256
#     print(new_arr.shape)
#
#     # the chosen start index is now our 0-index
#
#     for s in range(start_idx, end_idx):  # axial, Z axis
#         #     print('Slice: {}'.format(s)) # debug
#         new_index = s - start_idx
#         #     print('New Index: {}'.format(new_index))
#         new_arr[new_index, :, :] = cv2.resize(pxl[:, :, s],  # don't forget to TRANSPOSE
#                                               dsize = (256, 256),
#                                               interpolation = cv2.INTER_CUBIC)
#     return(new_arr)
# python3 /home/delvinso/neuro/analysis/net/train_and_eval.py \
# --root_path='/home/delvinso/neuro' \
#             --outcome='multitask' \
#                       --view=axial \
#                               --num_epochs=300 \
#                                            --manifest_path='/home/delvinso/neuro/output/ubc_npy_outcomes_v3_ss_PD.csv' \
#                                                            --model_out='/home/delvinso/neuro/output/models' \
#                                                                        --metrics_every_iter=20 \
#                                                                                             --task=multitask \
#                                                                                                    --run_name=condatest