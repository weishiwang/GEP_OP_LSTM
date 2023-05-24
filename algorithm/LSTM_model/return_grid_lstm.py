import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset  # 抽象类不能实例化
from torch.utils.data import DataLoader, TensorDataset  # dataloader可实例化
import log
# 导入数据
def load_data():
    data = np.load("../data/data_for_model.npz", allow_pickle=True)
    x = data["x"]
    y = data["y"]
    return x, y


def split_tr_te(x, y, train_size=0.8):
    train_num = int(train_size * len(x))
    return x, y, x[train_num:], y[train_num:]


class mydataset(Dataset):
    def __init__(self, mode):
        super(mydataset, self).__init__()

        self._data, self._labels = load_data()

        # self._data = normalize(self._data)
        self.data = None
        self.labels = None

        if mode == "train":
            self.data, self.labels, _, _ = split_tr_te(self._data, self._labels)
        elif mode == "test":
            _, _, self.data, self.labels = split_tr_te(self._data, self._labels)

    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


# 搭建模型

class LSTM(nn.Module):
    def __init__(self, n_neurouse, n_layers, drop_rate, hidden_size, Active_fun):
        super().__init__()
        self.input_size = 5

        self.lstm = nn.LSTM(self.input_size, n_neurouse, num_layers=n_layers, batch_first=True)

        self.a = nn.Sequential(
            nn.Linear(n_neurouse, hidden_size),
            # nn.ReLU(True),
            nn.Dropout(drop_rate),
        )
        self.action = self.get_active_fun(Active_fun)

        self.pred = nn.Linear(hidden_size, 1)

    def get_active_fun(self, n):
        if n == 1:
            return nn.ReLU6(True)
        elif n == 2:
            return nn.Tanh()
        elif n == 3:
            return nn.Sigmoid()
        else:
            return nn.ReLU(True)

    def forward(self, input_seq):
        # LSTM input=(seq_len, batch_size,n_feature)
        input_seq = input_seq.permute([1, 0, 2])

        out, _ = self.lstm(input_seq)  # output(5, 30, 64)
        # output=output.resize(batch_size,-1)
        # print(output.shape)
        out = out[-1, :, :]
        out = self.a(out)
        out = self.action(out)
        pred = self.pred(out)
        return pred

def invent_mean_std(data,mean,std):
    return data*std+mean



def get_optimizer(model, opt, Learning_rate):
    if opt == 0:
        return torch.optim.RMSprop(model.parameters(), lr=Learning_rate)
    elif opt == 1:
        return torch.optim.SGD(model.parameters(), lr=Learning_rate)
    elif opt == 2:
        return torch.optim.Adagrad(model.parameters(), lr=Learning_rate)
    else:
        return torch.optim.Adam(model.parameters(), lr=Learning_rate)


def conversion_Mini_batch(Mini_batch):
    Mini_batch = int(Mini_batch)
    if Mini_batch == 0:
        return 8
    elif Mini_batch == 1:
        return 16
    elif Mini_batch == 2:
        return 32
    elif Mini_batch == 3:
        return 64
    else:
        return 128



def train_net(data):
    n_neurouse = int(data["n_neurouse"])
    drop_rate = round(data["drop_rate"],2)
    hidden_size = int(data["hidden_size"])
    Mini_batch = conversion_Mini_batch(data["Mini_batch"])
    Active_function = int(data["Active_function"])
    Optimizer = int(data["Optimizer"])
    n_layers = 2
    Learning_rate = 0.0001
    model = LSTM(n_neurouse, n_layers, drop_rate, hidden_size, Active_function)

    loss_function = nn.MSELoss(reduction='mean')
    # loss_function=My_loss()

    optimizer = get_optimizer(model, Optimizer, Learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    trainSet = mydataset(mode="train")
    testSet = mydataset(mode="test")

    trainLoader = DataLoader(trainSet, batch_size=Mini_batch, shuffle=False, num_workers=0)
    testLoader = DataLoader(testSet, batch_size=Mini_batch, shuffle=False, num_workers=0)


    TEMP_LOSSING = float(999)
    loss_epoch = []
    for epoch in range(1, 50):
        model.train()
        # print(f"lr = {optimizer.state_dict()['param_groups'][0]['lr']}")

        running_loss = 0
        loss = 0

        for train_step, train_data in enumerate(trainLoader):
            # count+=1
            inputs, labels = train_data
            inputs, labels = inputs.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = loss_function(outputs, labels)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().numpy()
        scheduler.step()
        loss_epoch.append(running_loss)
        # print("epoch:{:d},running_loss:{:.2f}".format(epoch, running_loss))

        if running_loss < TEMP_LOSSING:
            TEMP_LOSSING = running_loss
    # print('------------',-TEMP_LOSSING)
    # l = log.Logger('grid_result.log', level='debug')
    # # log.logger.debug('debug')
    # l.logger.info('pastamer: {},loss: {}'.format(data,TEMP_LOSSING))
    return TEMP_LOSSING


