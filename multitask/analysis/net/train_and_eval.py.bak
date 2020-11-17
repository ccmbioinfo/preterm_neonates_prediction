import logging
import argparse
import torch
import helpers
import os
import re
from datetime import datetime
from run_model import run_model                    # the main training and evaluation loop
from data_loader import Dataset, get_dataloader    # grabbing the dataloaders

from net import Net

#TODO: implement json
# # trains and evaluates the neural net by calling run_model.py
def train_and_eval(model, train_loader, valid_loader, learning_rate, epochs,
                   model_outdir, #pos_wt,
                   metrics_every_iter, task, tensorboard = False,
                   restore_chkpt = None, run_suffix = None):
    """
    Contains the powerhouse of the network, ie. the training and validation iterations called through run_model().
    All parameters from the command line/json are parsed and then passed into run_model().
    Performs checkpointing each epoch, saved as 'last.pth.tar' and the best model thus far (based on validation AUC), saved as 'best.pth.tar'

    :param model: (nn.Module) -
    :param train_loader: (torch DataLoader)
    :param valid_loader: (torch DataLoader)
    :param learning_rate: (float) - the learning rate, defaults to 1e-05
    :param epochs: (int) - the number of epochs
    :param model_outdir: (str) - the output directory for checkpointing, checkpoints will be saved as output_dir/task/view/*.tar
    :param restore_chkpt: (str) - the directory to reload the checkpoint, if specified
    :param run_suffix: (str) - suffix to be appended to the event file. removed for now.
    :return:
    """

    log_fn = helpers.create_tb_log_dir(model_outdir)

    log_fn = log_fn.strip("/") # remove leading forward slash which messes up tf log

    if tensorboard:
        import tensorflow as tf
        writer = tf.summary.create_file_writer(log_fn) # tf 2.0+
        writer = tf.compat.v1.summary.FileWriter(log_fn) # tf v1.15

    current_best_val_loss = float('Inf')
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.01)
    # taken directly from MRNet code
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = 5, # how many epochs to wait for before acting
                                                           factor = 0.3, # factor to reduce LR by, LR = factor * LR
                                                           threshold = 1e-4)  # threshold to measure new optimum
    #
    losses = {'regression': torch.nn.MSELoss(),
              'classification': torch.nn.BCEWithLogitsLoss(),
              'multitask': torch.nn.MSELoss()}

    #
    criterion = losses[task]
    print(criterion)

    metric = {'regression':'MSE',
              'classification':'AUC',
              'multitask': 'MSE'}

    # # TODO: reloading checkpoint
    # if restore_chkpt:
    #     logging.info("Restoring Checkpoint from {}".format(restore_chkpt))
    #     helpers.load_checkpoint(checkpoint = restore_chkpt,
    #                             model = model,
    #                             optimizer = optimizer,
    #                             # scheduler = scheduler,
    #                             epochs = epochs)
    #     # print(loaded_epoch)
    #     # so epochs - loaded_epoch is where we would need to start, right?
    #     logging.info("Starting again at Epoch {}....".format(epochs))
    #     logging.info("Finished Restoring Checkpoint...")


    for epoch in range(epochs):
        logging.info('[Epoch {}]'.format(epoch + 1))

        # main training loop
        epoch_loss, epoch_metric, epoch_preds, epoch_labels, train_df = run_model(
                                               model = model,
                                               loader = train_loader,
                                               optimizer = optimizer,
                                               criterion = criterion,
                                               metrics_every_iter  = metrics_every_iter,
                                               task = task,
                                               tensorboard = tensorboard,
                                               train = True)

        logging.info('[Epoch {}]\t\tTraining {}: {:.3f}\t Training Average Loss: {:.5f}'\
                     .format(epoch + 1, metric[task], epoch_metric, epoch_loss))

        # main validation loop
        epoch_val_loss, epoch_val_metric, epoch_val_preds, epoch_val_labels, val_df = run_model(model = model,
                                                             loader = valid_loader,
                                                             optimizer = optimizer,
                                                             criterion = criterion,
                                                             task = task,
                                                             tensorboard = tensorboard,
                                                             metrics_every_iter = False, # default, just show the epoch validation metrics..
                                                             train = False)

        logging.info('[Epoch {}]\t\tValidation {}: {:.3f}\t Validation Average Loss: {:.5f}'.format(epoch + 1, metric[task], epoch_val_metric, epoch_val_loss))
        scheduler.step(epoch_val_loss) # check per epoch, how does the threshold work?!?!?
        logging.info('[Epoch {}]\t\tOptimizer Learning Rate: {}'.format(epoch + 1, {optimizer.param_groups[0]['lr']}))

        # with writer:#.as_default():
        # temp = torch.tensor([epoch + 1]) # needs to be a tesor in tf v1.5?
        # writer.add_summary(tf.compat.v1.summary.scalar('Loss/train', epoch_loss), temp).eval()
        # writer.add_summary(tf.compat.v1.summary.scalar('Loss/val', epoch_val_loss), temp).eval()
        # writer.add_summary(tf.compat.v1.summary.scalar('{}/train'.format(metric[task]), epoch_metric), temp).eval()
        # writer.add_summary(tf.compat.v1.summary.scalar('{}/val'.format(metric[task]), epoch_val_metric), temp).eval()
        # writer_flush = writer.flush()
        # with writer.as_default():
        #     tf.summary.scalar('Loss/train', epoch_loss, epoch + 1)
        #     tf.summary.scalar('Loss/val', epoch_val_loss, epoch + 1)
        #     tf.summary.scalar('{}/train'.format(metric[task]), epoch_metric, epoch + 1)
        #     tf.summary.scalar('{}/val'.format(metric[task]), epoch_val_metric, epoch + 1)


        print('Loss/train: {} for epoch: {}'.format(str(epoch_loss), str(epoch + 1)))
        print('Loss/val: {} for epoch: {}'.format(str(epoch_val_loss), str(epoch + 1)))
        print('{}/train: {} for epoch: {}'.format(metric[task], str(epoch_metric), str(epoch + 1)))
        print('{}/val: {} for epoch: {}'.format(metric[task], str(epoch_val_metric), str(epoch + 1)))


        # check whether the most recent epoch loss is better than previous best
        is_best_val_loss = epoch_val_loss < current_best_val_loss

        # save state in a dictionary
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'validation_metric': epoch_val_metric,
                 'metric': metric[task],
                 'best_validation_loss': epoch_val_loss,
                 # 'metrics': metrics # read more into this
                 'scheduler_dict': scheduler.state_dict(),
                 'optim_dict': optimizer.state_dict()}

        # save as last epoch
        helpers.save_checkpoint(state,
                                is_best = is_best_val_loss,
                                checkpoint_dir = model_outdir)

        if is_best_val_loss:
            current_best_val_loss = epoch_val_loss
            logging.info('[Epoch {}]\t\t******New Best Validation Loss: {:.3f}******'.format(epoch + 1, epoch_val_loss))
            helpers.save_checkpoint(state,
                                  is_best = is_best_val_loss,
                                  checkpoint_dir = model_outdir)
            if task == 'multitask':
                train_df.to_csv(os.path.join(model_outdir, 'best_epoch_training_results.csv'))
                val_df.to_csv(os.path.join(model_outdir, 'best_epoch_validation_results.csv'))
# the arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type = str, required = True, help = 'root directory of MRNet')
    parser.add_argument('--manifest_path', type=str, required=True, help = 'absolute path of the manifest file')
    parser.add_argument('--outcome', type = str, required = True, help = 'any column in the outcome sheet, or multitask')
    parser.add_argument('--view', type = str, required = True, help = 'one of sagittal, axial, coronal')
    parser.add_argument('--model_out', type=str, required = True, help = 'output directory')
    parser.add_argument('--num_epochs', type = int, required = True, help = 'int, the number of epochs')
    parser.add_argument('--learning_rate', type = float, default = 1e-5, help = 'the learning rate')
    parser.add_argument('--run_suffix', type=str, default=False, help='suffix to append to end of tensorflow log file')
    parser.add_argument('--metrics_every_iter', type = int, default = 20, help = 'calculates metrics every i batches')
    parser.add_argument('--tensorboard', type=bool, default=False, help = 'tensorboard logging, false by default')
    parser.add_argument('--run_name', type=str, default='my_run',
                        help='name of output directory where checkpoints and log is saved')
    parser.add_argument('--task', type = str, default = 'regression',
                        help = 'regression, classification or multitask. current support is only for binary classification.')
    return parser

if __name__ == '__main__':
    # get arguments
    args = get_parser().parse_args()

    torch.manual_seed(1)

    # create model outdir if it doesn't already exist, corresponding to task and view
    model_outdir = os.path.join(args.model_out, args.outcome + '-' + args.view, args.run_name)

    helpers.check_make_dir(model_outdir)

    # set up the logger
    helpers.set_logger(os.path.join(model_outdir, 'train.log'))
    # print arguments to log
    logging.info('-' * 20 + 'ARGUMENTS' + '-' * 20)
    for arg in vars(args):
        logging.info(arg + ": " + str(getattr(args, arg)))
    logging.info('-' * 49)

    # check device..
    logging.info('Device: {}'.format(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    logging.info('Model and Log Directory: {}'.format(model_outdir))

    # get dataloaders for task and view
    logging.info('Retrieving Dataloaders..')
    dataloaders = get_dataloader(sets = ['train', 'valid'],
                                 data_dir = args.root_path, # this must be ..../neuro/!!!!
                                 view = args.view,
                                 outcome = args.outcome,
                                 manifest_path = args.manifest_path,
                                 batch_size=1,
                                 return_pid = True)

    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']

    logging.info('Finished retrieving dataloaders..')

    # the main body
    if args.task == 'multitask':
        nnet = Net(mod = 'multitask')
    else:
        nnet = Net()

    logging.info("Training for {} epoch(s)..".format(args.num_epochs))
    train_and_eval(model = nnet,
                   train_loader = train_loader,
                   valid_loader = valid_loader,
                   learning_rate = args.learning_rate,
                   epochs = args.num_epochs,
                   metrics_every_iter = args.metrics_every_iter,
                   model_outdir = model_outdir,
                   task = args.task)
    logging.info("Done!!..\n Please check {} for the results!".format(model_outdir))

