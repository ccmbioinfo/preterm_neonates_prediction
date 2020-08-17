import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
import img_helpers as ih

# ---- the skull stripped niftis ----
ss_dir = '/home/delvinso/neuro/data/ubc_misc/ubc_ss/'

# for each skull stripped image, get the mean pixel intensities for each slice
ss_int_d = {}
for f in os.listdir(ss_dir):
    f_abs = os.path.join(ss_dir, f)
    print(f_abs)
    # read in the nifti
    try:
        nii = nib.load(f_abs)
    except KeyError:
        print('Error with {}'.format(f_abs))
        next
    # extract the array
    pxl_arr = np.asarray(nii.dataobj)
    # compute the mean intensity for each slice.
    # series_mean_intensity() returns an array where each element is the
    # mean intensity of the ith slice
    intensity = ih.series_mean_intensity(pxl_arr)
    # remove the black images
    intensity = intensity[intensity > 0]
    ss_int_d[f] = intensity

intens_df = pd.DataFrame.from_dict(ss_int_d, orient = 'index').reset_index()\
    .melt(id_vars = 'index',var_name = 'slice',value_name = 'mean_intensity').dropna()
intens_df['id'] = intens_df.iloc[:,0].map(lambda x: x.split('_')[0])
intens_df['fn'] = intens_df.iloc[:, 0]

intens_df.groupby(['id'])['fn'].size()
# iterate through the IDs

import seaborn as sns

# for fn in intens_df.fn.drop_duplicates():
#     print(fn)
#     subset = intens_df[intens_df['fn'] == fn]
#     # subset['log_mean_intensity'] = np.log10(subset.mean_intensity)
#
#     # Draw the density plot
#     sns.distplot(subset.mean_intensity, hist=False, kde=True,
#                  kde_kws={'linewidth': 1.5})
#                  # label=airline)


for id in intens_df.id.drop_duplicates():
    print(id)
    subset = intens_df[intens_df['id'] == id]
    # subset['log_mean_intensity'] = np.log10(subset.mean_intensity)

    # Draw the density plot
    sns.distplot(subset.mean_intensity, hist=False, kde=True,
                 kde_kws={'linewidth': 1.5})
                 # label=airline)

# Plot formatting
plt.legend(prop={'size': 16}, title='Patient ID')
plt.title('Distribution of Mean Pixel Intensities.')
plt.xlabel('Mean Pixel Intensity')
plt.ylabel('Density')
plt.show()
plt.savefig('distribution_mean_pxl_intensities.png')

