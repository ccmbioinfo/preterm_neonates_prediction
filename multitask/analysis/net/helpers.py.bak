import logging
import shutil
import os
import torch
from datetime import datetime
import re
import numpy as np
# utilities for main, such as logging and checkpointing

# some utilities adapted from https://github.com/cs230-stanford/cs230-code-examples/blob/96ac6fd7dd72831a42f534e65182471230007bee/pytorch/vision/utils.py#L63

def save_checkpoint(state, is_best, checkpoint_dir):

    """
    Saves model parameters at checkpoint directory (the model_outdir).
    If is_best == True, then this will also save as 'best.pth.tar'

    :param state:  (dict)
    :param is_best:  (bool)
    :param checkpoint_dir: (str)
    :return:
    """
    # if len(str(epoch)) < 2:
    #     epoch = '0' + str(epoch)
    # fn = epoch + '_last.pth.tar'
    filepath = os.path.join(checkpoint_dir, 'last.pth.tar')

    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist, Making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    # else:
        # print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        # fn = epoch + '_best.path.tar'
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer = None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    print('States for Epoch {} loaded...'.format(checkpoint['epoch']))

    # return checkpoint


# did not edit from CS230
def set_logger(log_path):
    """
    Initialize logger so it will log in both console and `log_path`. Will be output into `model_dir/train.log`
    :param log_path: where to log
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path) # overwrite existing
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def create_tb_log_dir(model_outdir):
    """
    Creates a log directory given model_outdir, which is comprised of the specified output directory, the run name, the outcome and the view used
    eg. model_outdir = neuro/output/models/${run_name}/${outcome}-${view}
    the corresponding log directory would be neuro/output/models/logs/${run_name}/${outcome}-${view}/$(%Y_%m_%d-%H_%M)
    :param model_outdir: (str) model directory from 'train_and_eval.py'
    :return log_fn: (str) path of log directory
    """
    recover_root_dir = os.path.dirname(os.path.dirname(model_outdir)) # removes the view and outcome from the directory
    log_dir = os.path.join(recover_root_dir, "logs") #
    outcome_view = re.split(r'/|\\', model_outdir)[-1]
    run_name = re.split(r'/|\\', model_outdir)[-2]
    log_fn = os.path.join(log_dir, run_name, outcome_view)

    dtnow = datetime.now()
    log_fn = os.path.join(log_fn,  dtnow.strftime("%Y_%m_%d-%H_%M_%S"))

    # make directory if it doesn't exist. this shouldn't exist though so no need for a check
    # print('{} does not exist, creating..!'.format(log_fn))
    logging.info('------------------TENSORBOARD LOG DIRECTORY: {}------------------'.format(log_fn))
    os.makedirs(log_fn)

    return log_fn
    # TEST:
    # test = /home/delvinso/neuro/output/models/run_name/outcome-view/
    # create_log_dir(test)
    # expected: /home/delvinso/neuro/output/models/logs/run_name/outcome-view/2020_04_16-10_37

def check_make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('{} does not exist, creating..!'.format(dir_name))
    else:
        print('{} already exists!'.format(dir_name))
def mask_multitask_loss(output, ground_truth, criterion = torch.nn.MSELoss()):
    """
    :param output:
    :param ground_truth:
    :param criterion:
    :return:
    """
    output = list(output)
    ground_truth = ground_truth.flatten()

    miss_mask = torch.isnan(ground_truth)
    
    ground_truth = ground_truth[~ miss_mask]  # retain non-NAs
    output = [output[i] for i in range(len(output)) if miss_mask[i] == 0] # tensors have to be in a list, tuple or array

    # assert(output.shape[0] == )
    # print(output)
    # print(output[0].nelement())
    # # print(ground_truth)
    # print(ground_truth[0].nelement())

    # compute loss for each output and sum
    losses = []
    for i in range(len(output)):
        losses.append(criterion(output[i], ground_truth[i]))
    # mt_loss = ([criterion(output[i], ground_truth[i]) for i in range(len(output))])
    print("# of Valid Outputs used in Loss: {}".format(len(losses)))
    # print(losses)
    mt_loss = sum(losses)
    assert(mt_loss.nelement() == 1)
    return miss_mask, output, ground_truth, mt_loss

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
