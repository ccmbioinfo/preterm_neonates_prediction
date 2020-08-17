

# given a checkpoint model, generates predictions and fitted values for the validation and training set, respectively
# and saves to a csv

import pandas as pd
import torch
from data_loader import Dataset, get_dataloader    # grabbing the dataloaders
import os
import helpers
from net import Net
from sklearn import metrics

model = Net()
# checkpoint_path = '/home/delvinso/neuro/output/bay_cog_comp_sb_18m/axial/_best.path.tar'
# checkpont_path = '/home/delvinso/neuro/output/archive/bay_cog_comp_sb_18m/axial/_best.path.tar'
# checkpoint_path = '/home/delvinso/neuro/output/tensor_check_norm/bay_cog_comp_sb_18m-axial/_best.path.tar'
# checkpoint_path = '/home/delvinso/neuro/output/no_norm/bay_cog_comp_sb_18m-axial/_best.path.tar'
# checkpoint_path = '/home/delvinso/neuro/output/models/notensor_dropout/bay_cog_comp_sb_18m-axial/_best.path.tar'
# checkpoint_path = '/home/delvinso/neuro/output/models/notensor/bay_cog_comp_sb_18m-axial/_best.path.tar'
# checkpoint_path = '/home/delvinso/neuro/output/models/bay_cog_comp_sb_18m-axial/_best.path.tar'

checkpoint_path = '/home/delvinso/neuro/output/models/test_ss/bay_cog_comp_sb_18m-axial/_best.path.tar'
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval().to(device)

# optimizer doesn't really matter for making predictions ...
helpers.load_checkpoint(checkpoint = checkpoint_path,
                model = model)

# set the arguments
sets = ['train', 'valid']
view = 'axial'
outcome = 'bay_cog_comp_sb_18m'
data_dir = '/home/delvinso/neuro'
manifest_path = '/home/delvinso/neuro/output/ubc_npy_outcomes_v3_ss.csv'
# initialize the dataloaders
dls = get_dataloader(sets = sets,
                     view = view,
                     outcome = outcome,
                     data_dir = data_dir,
                     return_pid=True,
                     manifest_path = manifest_path)


list_holder = []
for set in sets:
    print(set)
    loader = dls[set]
    preds_list = []
    labels_list = []
    pids_list = []
    for i, (image, label, pid) in enumerate(loader):
        with torch.no_grad():
        # for i, (image, label) in enumerate(loader):
            print(i)
            image = image.float().to(device)
            labels = label.float().to(device)#.squeeze(1)
            outputs = model.forward(image)                        # forward pass
            pid = pid[0]
            print("pid: {}, Label: {} \tOutput: {}".format(pid, labels, outputs))
            preds_numpy = outputs.detach().cpu().numpy()[0][0].squeeze(0)
            labels_numpy = labels.detach().cpu().numpy().flatten()[0]
            preds_list.append(preds_numpy)
            labels_list.append(labels_numpy)
            pids_list.append(pid)
    list_holder.append({'pid' :pids_list, 'labels':labels_list, 'preds':preds_list, 'set': set})
metrics.mean_squared_error(labels_list, preds_list)
# create the dataframe
df_res = pd.concat([pd.DataFrame(list_holder[0]),
                    pd.DataFrame(list_holder[1])])

df_res['outcome'] = outcome

df_res.to_csv(os.path.join('/home/delvinso/neuro/output', outcome +'_ss.csv'))
