from torch import nn
import torch
import numpy as np
class Classifier(nn.Module):
    def __init__(self,input_nodes):
        super(Classifier, self).__init__()
        self.input_nodes = input_nodes

        self.sharedlayer = nn.Sequential(
            nn.Linear(input_nodes, 300),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(),
        )


        self.att1 = nn.Sequential(
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(20, 1)
        )
        self.att2 = nn.Sequential(
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(20, 1)
        )
        self.att3 = nn.Sequential(
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(20, 1)
        )

    def forward(self, x):

        h_shared = self.sharedlayer(x)
        out1 = self.att1(h_shared)
        out2 = self.att2(h_shared)
        out3 = self.att3(h_shared)

        return out1, out2, out3

criterion = nn.MSELoss()
model = Classifier(6000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 1
train_x = torch.rand(150, 6000) # 150 x 6000
test_x = torch.rand(25, 6000) # 25 x 6000

# ys are the column-wise sums 
y1 = train_x[:, :1000].sum(axis = 1)
y2 = train_x[:, :3000].sum(axis = 1)
y3 = train_x[:, :6000].sum(axis = 1)


# criterion(train_x[, :])

for epoch in range(n_epochs):
    running_loss = 0
    i = 0
    model.train()
    for idx in range(train_x.shape[0]):
        i = i + 1
        print(idx)
        x = train_x[idx, :]
        label1 = y1[idx]
        label2 = y2[idx]
        label3 = y3[idx]
        out1, out2, out3 = model(x)
        
        l1 = criterion(out1, label1)
        l2 = criterion(out2, label2)
        l3 = criterion(out3, label3)

        loss = (l1 + l2 + l3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        truth = np.asarray([label1, label2, label3])
        preds = np.asarray([out1, out2, out3])
        print('Truth: {}, Labels: {}'.format(truth, preds))
        if i == 10: break
