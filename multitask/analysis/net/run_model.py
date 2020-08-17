import logging
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
# custom
from net import Net
from data_loader import Dataset, get_dataloader
import pandas as pd
import numpy as np
import helpers

def run_model(model, loader, optimizer, criterion, task, tensorboard, metrics_every_iter = False, train = True):
    """
    Executes a full pass over the training or validation data, ie. an epoch, calculating the metrics of interest

    :param model: (torch.nn.module) - network architecture to train
    :param loader: (DataLoader) - torch DataLoader
    :param optimizer: (optimizer) - an optimizer to be passed in
    :param metrics_every_iter: (int) - evaluate metrics every ith iteration
    :param epoch (int) - the current epoch number, needed for tensorboard calculations
    :param train: (bool) - if True then training mode, else evaluation mode (no gradient evaluation, backprop, weight updates, etc)
    :param criterion (torch.nn.modules.loss) - loss function
    :param writer (tensorboard.SummaryWriter)
    :returns: epoch loss, auc, list of predictions and labels
    """

    # use cpu or cuda depending on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if train:
        model.train().to(device)
    else:
        model.eval().to(device)

    # initialize lists for keeping track of predictions and labels
    preds_list = []
    labels_list = []
    temp_list = []
    pid_list = []
    # total_epoch_loss = 0
    ith_loss = 0            # keeps track of loss every ith iterations for metrics
                            # resets to 0 every ith iteration
    total_running_loss = 0.0 # keeps track of total loss throughout the epoch
                           # divided by the # of batches at the end of an epoch to obtain an epoch's avg loss
    num_batches = 1        # total n will be 1130 in MRNet's case, must start at 1 for tensorboard logging else the first step will repeat 2x


    for image, label, pid, covars in loader:

        # send X,Y and weights (if applicable) to device - only really used for gpu
        image = image.float().to(device)
        labels = label.float().to(device).flatten().unsqueeze(1)
        covars = covars.float().to(device)

        # print(labels)
        # wts = wts.float().to(device)

        print(image.shape)

        #  ----- a single iteration of forward and backward pass ----

        if train:
            optimizer.zero_grad()                       # zero gradients

        outputs = model.forward(image)                  # forward pass, squeeze to fix size
        # outputs = model.forward(image, covars)                  # forward pass, squeeze to fix size
        # print(outputs.size)
        # print('label: {}\noutput: {}'.format(outputs, labels))
        if task == 'multitask':
            miss_mask, outputs, labels, loss = helpers.mask_multitask_loss(output = outputs, ground_truth = labels, criterion = criterion )
            # print(outputs)
            # print(labels)
        else:
            loss = criterion(outputs, labels)               # compute loss

        total_running_loss += loss.item()
        ith_loss += loss.item()

        if train:
            loss.backward()                             # backprop
            optimizer.step()                            # update parameters

        # extract data from torch Variable, move to cpu, converting to numpy arrays
        # https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/train.py
        if task == 'classification':
            outputs = torch.sigmoid(outputs)

        try:
            preds_numpy = outputs.detach().cpu().numpy().flatten()#[0]#[0]
        except AttributeError:
            preds_numpy = np.concatenate([x.detach().cpu().numpy().flatten() for x in outputs])

        labels_numpy = labels.detach().cpu().numpy().flatten()#[0]#[0]
        preds_list.append(preds_numpy)
        labels_list.append(labels_numpy)
        # print(preds_list); print(labels_list)

        # recreate the original vector with corresponding missing values, if applicable for predictions
        df = []
        if task == "multitask":
            miss_mask = miss_mask.detach().cpu().numpy().flatten()
            miss_mask = miss_mask.astype(bool)
            # print(miss_mask)
            # print(miss_mask.shape)
            temp = np.zeros([6]) # whatever the number of outcomes is
            temp[temp == 0] = np.NaN # set to NaN
            # print(temp)
            # print(temp.shape)
            # print(preds_numpy)
            # print(preds_numpy.shape)
            temp[~miss_mask] = preds_numpy # the indices corresponding to the non-missing outcomes are filled in with the predictions
            temp_list.append(temp)
            pid_list.append(pid[0])
            col_names = ['bay_cog_comp_sb_18m', 'bay_language_comp_sb_18m','bay_motor_comp_sb_18m',
                         'bay_cog_comp_sb_33m', 'bay_language_comp_sb_33m', 'bay_motor_comp_sb_33m']
            pd.options.display.width = None
            df = pd.DataFrame(temp_list, columns = col_names)
            df['pid'] = pid_list
            # print(df)

        num_batches += 1
        # if indicated, will print and (eventually) output running auc and average batch loss
        if metrics_every_iter:
            if (num_batches % metrics_every_iter == 0) & (num_batches > 0):
                if task in ['regression', 'multitask']:
                    # mse = metrics.mean_squared_error(labels_list, preds_list)
                    logging.info('\t{} Batch(es)\t{} Average Batch Loss:{:.3f}'. \
                                 # print('{} Batch(es)\n\tAUC: {:.3f}\n\tRunning Average Loss:{:.3f}\n\tAccuracy:{:.3f}'.\
                                 format(str(num_batches),(metrics_every_iter), (ith_loss / metrics_every_iter)))

                elif task == 'classification':

                    fpr, tpr, threshold = metrics.roc_curve(np.concatenate(labels_list), np.concatenate(preds_list))
                    auc = metrics.auc(fpr, tpr)
                    logging.info('\t{} Batch(es)\tAUC: {:.3f}\t{} Average Batch Loss:{:.3f}'.\
                          format(str(num_batches), (auc), (metrics_every_iter),  (ith_loss/metrics_every_iter)))

                # tensorflow
                # if tensorboard:
                #     import tensorflow as tf
                #     with writer.as_default():
                #         tf.summary.scalar('Iter_Average_Loss/train_iter', ith_loss/metrics_every_iter, num_batches * epoch)
                #         tf.summary.scalar('Running_AUC/train_iter', auc, num_batches * epoch)
                ith_loss = 0 # reset loss every ith iterations since this keeps track of the ith iteration
                print(preds_numpy); print(labels_numpy)

    # epoch-level metrics
    epoch_mean_loss = total_running_loss/len(loader)
    # print(pd.DataFrame({'labels': labels_list, 'preds': preds_list}))

    labels_list = np.concatenate(labels_list)
    preds_list = np.concatenate(preds_list)

    if task in ['regression', 'multitask']:
        epoch_metric = metrics.mean_squared_error(labels_list, preds_list)
    elif task == 'classification':
        fpr, tpr, threshold = metrics.roc_curve(labels_list, preds_list)
        epoch_metric = metrics.auc(fpr, tpr)

    return epoch_mean_loss, epoch_metric, preds_list, labels_list, df

