

import torch
import torch.nn as nn
from torchvision import models
from helpers import set_parameter_requires_grad

class Net(nn.Module):
    #TODO: make robust to number of units before FC layer
    def __init__(self, mod = 'single', num_heads = 6, model_name = 'resnet18'):
        super().__init__()
        #TODO: take input as 128 x 128 tensor
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained = True)
        elif model_name == 'vg11_bn':
            self.model = models.vgg11_bn(pretrained = True)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2]) # layer before the average pool, 2048
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretraiend = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])  # layer before the average pool, 2048
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2]) # layer before the average pool, 512
            set_parameter_requires_grad(self.model, True) # freeze weights

        #self.model = nn.Sequential(*list(self.model.children())[:-2])
        # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707

        # adaptive pool
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.lin = nn.Linear(512, 1) # TODO: need to change this depending on which resnet. use a dictionary
        self.mod = mod
        self.dropout = nn.Dropout(p=0.5)
        self.heads = nn.ModuleList([])

        for n in range(num_heads):
            self.heads.append(nn.Sequential(
                nn.Linear(512, 1) # 256 for alexnet, 512 for vgg and for resnet18, 2048 for resnet50,101, 152
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


