import torch
import torch.nn as nn
from torchvision import models
from helpers import set_parameter_requires_grad

class Net(nn.Module):
    #TODO: make robust to number of units before FC layer
    def __init__(self, mod = 'single', num_heads = 1, model_name = 'resnet18'):
        # Steven Ufkes: I think that num_heads should be set to the number of variables in the multitask prediction; it is ignored if task is "regression" or "classification".

        super().__init__()
        #TODO: take input as 128 x 128 tensor
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained = True)
        elif model_name == 'vg11_bn':
            self.model = models.vgg11_bn(pretrained = True)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2]) # layer before the average pool, 2048

            # Steven Ufkes: Try to freeze the weights as Delvin did for resnet18.
            set_parameter_requires_grad(self.model, True) # added by Steven Ufkes 2020-10-29.
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])  # layer before the average pool, 2048

            # Steven Ufkes: Try to freeze the weights as Delvin did for resnet18. I'm not sure if this is correct. Give it a try to see if it works.
            set_parameter_requires_grad(self.model, True) # added by Steven Ufkes 2020-10-29.
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2]) # layer before the average pool, 512
            set_parameter_requires_grad(self.model, True) # freeze weights

        #self.model = nn.Sequential(*list(self.model.children())[:-2])
        # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707

        # Steven:
        model_num_features_last_layer_dict = {'alexnet':256,
                               'vg11_bn':512,
                               'resnet50':2048,
                               'resnet101':2048,
                               'resnet18':512}
        num_features_last_layer = model_num_features_last_layer_dict[model_name]

        # adaptive pool
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.lin = nn.Linear(512, 1) # TODO: need to change this depending on which resnet. use a dictionary
        # Steven: Not sure if this is correct; not sure if the regression task was ever fully set up.
        self.lin = nn.Linear(num_features_last_layer, 1) # Added by Steven; based on line above which was commented out.
        self.mod = mod
        self.dropout = nn.Dropout(p=0.5)
        self.heads = nn.ModuleList([])

        # Steven Ufkes 2020-10-30 : I modified the section below to try to get other CNNs to work.
        # Original:
#        for n in range(num_heads):
#            self.heads.append(nn.Sequential(
#                nn.Linear(512, 1) # 256 for alexnet, 512 for vgg and for resnet18, 2048 for resnet50,101, 152
#                # TODO: make robust to covariates
#            ))

        for n in range(num_heads):
            self.heads.append(nn.Sequential(
                nn.Linear(num_features_last_layer, 1) # 256 for alexnet, 512 for vgg and for resnet18, 2048 for resnet50,101, 152
                # TODO: make robust to covariates
            ))


    def forward(self, x, covars = None):
        # print(x.shape)
        x = torch.squeeze(x, dim = 0) # only batch size 1 supported
        # print(x.shape)
        # x = self.model.features(x) # if vgg or alexnet
        x = self.model(x)
        # print(x.shape)
        x = self.gap(x).view(x.size(0), -1)
        # print(x.shape)
        x = torch.max(x, 0, keepdim = True)[0]
        # print(x.shape)
        # x = self.classifier(self.dropout(x))
        x = self.dropout(x)
        if covars:
            x = torch.flatten(x)
            # print(x)
            # print(x.shape)
            # print(covars)
            # print(covars.shape)
            x = torch.cat((x, covars), dim = 0)
            # print(x)
            # print(x.shape)
        if self.mod == 'multitask':
            # TODO: https://stackoverflow.com/questions/59763775/how-to-use-pytorch-to-construct-multi-task-dnn-e-g-for-more-than-100-tasks
            outputs = []
            for head in self.heads:
                outputs.append(head(x))
            # [print(x) for x in outputs]
            return(outputs)
        else:
            x = self.lin(x)
            return x
