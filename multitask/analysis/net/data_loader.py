import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import time


class Dataset(Dataset):
    # define all members and class variables
    def __init__(self, view,  set,root_dir,   manifest_path, outcomes, task,
                 return_pid, preprocessed_img_dirs=None, transform=None):
        """
        A pytorch DataSet for reading in one planar view of a dicom at a time (three planes total)

        :param view: (str) - one of: sagittal, coronal, axial
        :param set: (str) - one of: train, valid (and maybe set in the future)
        :param root_dir:  (str) - root directory, should be something along the lines of 'MRNet-v1.0
        :param transform: (torchvision transforms) - a list of transforms to apply to the DataSet when being fetched
        :param preprocessed_img_dirs (list of str): - list containing the names of three directories in which the preprocessed images are stored for each of the three color channels.
        """
        self.view = view
        self.transform = transform
        self.set = set
        self.root_dir = root_dir
        self.manifest_path = manifest_path
        self.outcomes = outcomes
        #self.task = task
        self.return_pid = return_pid
        self.preprocessed_img_dirs = preprocessed_img_dirs

        # read in the manifest
        fn = os.path.join(self.manifest_path)

        self.task_manifest = pd.read_csv(fn).reset_index()

        #if self.outcome == 'multitask':
        if task == 'multitask': # Steven Ufkes: Change logic in previous line to allow different numbers of outcomes.
            # don't drop the missing values, hardcode the outcomes for now
            #self.task_manifest = self.task_manifest[['study_id', 'set', 'axial_dir_ss', 'bay_cog_comp_sb_18m', 'bay_language_comp_sb_18m', 'bay_motor_comp_sb_18m', 'bay_language_comp_sb_33m', 'bay_motor_comp_sb_33m', 'birthGA_2019.10.30']]

            #####
            # Steven Ufkes: Replace line above with following 3 lines.
            task_manifest_fields = ['study_id', 'set', 'axial_dir_ss', 'birthGA_2019.10.30']
            task_manifest_fields.extend(self.outcomes)
            self.task_manifest = self.task_manifest[task_manifest_fields]
            #####

            # keep only the observations where there are images
            self.task_manifest = self.task_manifest[self.task_manifest.axial_dir_ss.notnull()]
            self.task_manifest['num_miss'] = self.task_manifest.groupby('study_id').apply(lambda x: x.isnull()).sum(axis = 1)
            # self.task_manifest.num_miss.sort_values().value_counts()
            # keep any row that has at least 1 of the outcomes
            # num_outcomes = self.task_manifest.columns.shape[0] - 4 # number of outcomes = number of columns minus d, set, dir and num_miss
            # Steven Ufkes: Should probably replace next line with num_outcomes = len(outcomes)
            num_outcomes = self.task_manifest.columns.shape[0] - 5 # number of outcomes = number of columns minus d, set, dir and num_miss, and birthga
            # print(num_outcomes)

            print("num_outcomes, num_missing:", num_outcomes)
            print(self.task_manifest['num_miss'])

            self.task_manifest = self.task_manifest[self.task_manifest.num_miss < num_outcomes]
            # print(self.task_manifest.shape)

        else: # if not multitask, just use the requested outcome
            self.task_manifest = self.task_manifest[['study_id', 'set', 'axial_dir_ss',
                                                     'birthGA_2019.10.30',
                                                     self.outcomes[0]]].dropna()
        # print('Outcome: {} retains {} participants.'.format(str(self.outcome), str(self.task_manifest.shape())))

        if self.set == "train":
            self.task_manifest = self.task_manifest[self.task_manifest.set == 'train']
        elif self.set == "valid":
            self.task_manifest = self.task_manifest[self.task_manifest.set == 'val']
        else:
            print('Set needs to be one of train, valid')

        # self.covariates = ['days_intubated', 'rop', 'necstage', 'birthGA_2019.10.30', 'study_id']
        # self.task_manifest = self.task_manifest.set_index('study_id').join(copy_mn[[self.covariates]].set_index('study_id'))
        # print(self.task_manifest)

        # print error if outcome is not found in the manifest
        # if outcome.isin(task_manifest.columns):

    def __len__(self):
        return((self.task_manifest.shape[0]))

    def get_label(self, pid):
#        if self.outcome == 'multitask':
#            outcome = ['bay_cog_comp_sb_18m', 'bay_language_comp_sb_18m', 'bay_motor_comp_sb_18m', 'bay_cog_comp_sb_33m', 'bay_language_comp_sb_33m', 'bay_motor_comp_sb_33m']
#            label = np.asarray(self.task_manifest[self.task_manifest.study_id == pid][outcome])
#        else:
#            label = np.asarray(self.task_manifest[self.task_manifest.study_id == pid][[self.outcome]])

        #####
        # Steven Ufkes: Changed logic to handle any number of outcomes. Replaced 5 lines above with next line.
        label = np.asarray(self.task_manifest[self.task_manifest.study_id == pid][self.outcomes])
        #####

        return torch.tensor([label])

    def get_covariates(self, pid):
        # TODO: make it more robust to the list of covariates
        covars = np.asarray(self.task_manifest[self.task_manifest.study_id == pid]['birthGA_2019.10.30'])

        return(torch.tensor(covars[0])) # gestational age


    def __getitem__(self, index):

        pid = self.task_manifest.iloc[index].study_id

        y = self.get_label(pid)
        covars = self.get_covariates(pid)

        # img_dir = os.path.join(self.root_dir, 'output', 'ubc_3d', self.view, 'birth_' + pid + '_ss.npy')

        # change this to the path of the .npys
        # we want to use these ones, which are 128x128. can pad them or change architecture

        # Steven Ufkes: This used to be hardcoded to look for a single image modality in a single directory. I changed it to read three (possibly redundant) modalities from three directories, taken as an argument. By default, this will read all three color channels from the same old directory.
        if self.preprocessed_img_dirs: # if a list of 3 image directories from which color channels will be taken has been specified.
            img_dir1 = os.path.join(self.root_dir, 'output', self.preprocessed_img_dirs[0], self.view, 'birth_' + pid + '_ss.npy' )
            img_dir2 = os.path.join(self.root_dir, 'output', self.preprocessed_img_dirs[1], self.view, 'birth_' + pid + '_ss.npy' )
            img_dir3 = os.path.join(self.root_dir, 'output', self.preprocessed_img_dirs[2], self.view, 'birth_' + pid + '_ss.npy' )
        else:
            img_dir1 = os.path.join(self.root_dir, 'output', 'hardstretch_v01', self.view, 'birth_' + pid + '_ss.npy' )
            img_dir2 = os.path.join(self.root_dir, 'output', 'hardstretch_v01', self.view, 'birth_' + pid + '_ss.npy' )
            img_dir3 = os.path.join(self.root_dir, 'output', 'hardstretch_v01', self.view, 'birth_' + pid + '_ss.npy' )

        # the old preprocessed files
        #img_dir = os.path.join(self.root_dir, 'output', 'archive', 'ubc_3d', self.view, 'birth_' + pid + '_ss.npy' )

        # Steven Ufkes: I removed the zeroing of negative pixels. I don't konw why we would do that after normalization.
        img1 = np.load(img_dir1).astype(np.float32)
        #img1[img1 < 0] = 0 # set black pixels to 0
        img2 = np.load(img_dir2).astype(np.float32)
        #img2[img2 < 0] = 0 # set black pixels to 0
        img3 = np.load(img_dir3).astype(np.float32)
        #img3[img3 < 0] = 0 # set black pixels to 0

        min_slices = min(img1.shape[0], img2.shape[0], img3.shape[0])
        img1 = img1[:min_slices, :, :]
        img2 = img2[:min_slices, :, :]
        img3 = img3[:min_slices, :, :]

        # stack so 3 channels (RGB)
        img = torch.tensor(np.stack((img1, img2, img3), axis=1))
        # print('Max: {}]\tMin: {}\tShape: {}'.format(img.max(), img.min(), img.shape))

        if self.transform is not None:
            # print(img.split(1).shape)
            for i, slice in enumerate(img.split(1)):
                img[i] = self.transform(slice.squeeze())
        if self.return_pid:
            return img, y, pid, covars
        else:
            return img, y, covars


def get_dataloader(sets, data_dir, view, manifest_path, outcomes, task, return_pid=False, batch_size=1, preprocessed_img_dirs=None):
    """
    Takes a list of training sets (ie. any of 'train', 'valid', or 'test) and returns a dictionary of dataloaders.
    TODO: can make the transform more modular (ie. taken as input from command line)
    :param sets: (list) - list of any of  'train', 'valid', or 'test
    :param data_dir: (str) - where the root directory is
    :param view: (str) - one of: sagittal, coronal, axial
    :param batch_size: (int) - hardcoded to 1
    :return: (dict) of dataloaders
    """
    data_loaders = {}

    for set in ['train', 'valid', 'test']:  # test doesn't apply to MRNet but will keep in
        if set in sets:
            dir = os.path.join(data_dir, set)
            if set == 'train':
                ds = Dataset(set='train', view=view,
                             # task=task,
                             return_pid = return_pid,
                             outcomes=outcomes,
                             task=task,
                             root_dir=data_dir, manifest_path = manifest_path,
                             preprocessed_img_dirs = preprocessed_img_dirs,
                             # transform=transforms.Compose([transforms.ToPILImage(),
                             #                               # transforms.RandomHorizontalFlip(),  # default is 50%
                             #                               # transforms.RandomAffine(25,  # rotation
                             #                               #                         translate=(0.1, 0.1),
                             #                               #                         shear = (-10, 10)),
                             #                               transforms.ToTensor()])

                             )
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            else:
                ds = Dataset(set='valid', view=view,
                             return_pid = return_pid,
                             outcomes=outcomes,
                             task=task,
                             root_dir=data_dir, manifest_path = manifest_path,
                             preprocessed_img_dirs = preprocessed_img_dirs,
                             # transform = transforms.Compose([transforms.ToPILImage(),
                             #            transforms.ToTensor()])
                             )
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            data_loaders[set] = loader
    return (data_loaders)
