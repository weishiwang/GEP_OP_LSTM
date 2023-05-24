import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset  # 抽象类不能实例化
from torch.utils.data import DataLoader, TensorDataset  # dataloader可实例化
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

        # 定义一个输入的特征维度，这里input_size设为5（根据实验数据）
        self.input_size = 5

        # 定义一个LSTM层，其中的参数含义依次为：输入特征维度、神经元数、LSTM层数、batch_first参数表示输入数据的第一维是batch的维度
        self.lstm = nn.LSTM(self.input_size, n_neurouse, num_layers=n_layers, batch_first=True)

        # 定义一个神经网络的后半部分，包括一个全连接层，一个dropout层，以及一个自定义的激活函数。
        self.a = nn.Sequential(
            nn.Linear(n_neurouse, hidden_size),
            nn.Dropout(drop_rate),
        )
        self.action = self.get_active_fun(Active_fun)

        # 定义最后的输出层，输出一个标量。
        self.pred = nn.Linear(hidden_size, 1)

    # 定义一个根据用户输入设置不同激活函数的方法
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
        # 将输入的序列transpose到LSTM默认的格式，即(seq_len, batch_size,n_feature)
        input_seq = input_seq.permute([1, 0, 2])

        # 将transpose后的序列输入到LSTM中，得到输出out。
        out, _ = self.lstm(input_seq)

        # 取得最后一个时间步的输出
        out = out[-1, :, :]

        # 将LSTM的输出输入全连接层、dropout层、激活层。
        out = self.a(out)
        out = self.action(out)

        # 输出最后的回归结果
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



def train_net(n_neurouse, drop_rate, hidden_size, Mini_batch, Active_function, Optimizer, n_layers=2, Learning_rate=0.0001):

    # 定义一个LSTM模型，参数通过输入的超参数（n_neurouse, drop_rate, hidden_size, Active_function）进行设置。
    model = LSTM(n_neurouse, n_layers, drop_rate, hidden_size, Active_function)

    # 定义一个均方误差函数作为损失函数
    loss_function = nn.MSELoss(reduction='mean')

    # 定义一个实现梯度下降的优化器，同时学习率初始值也从输入的超参数中定义。
    optimizer = get_optimizer(model, Optimizer, Learning_rate)

    # 定义一个学习率调度程序，使用step方式按照一定规则（这里是每100个epoch降低0.9倍）调节学习率。
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # 定义训练集和测试集的数据集类
    trainSet = mydataset(mode="train")
    testSet = mydataset(mode="test")

    # 定义训练集和测试集的DataLoader，用于加载数据。
    trainLoader = DataLoader(trainSet, batch_size=Mini_batch, shuffle=False, num_workers=0)
    testLoader = DataLoader(testSet, batch_size=Mini_batch, shuffle=False, num_workers=0)

    # 初始化一个最小的loss值（用于后续比较）
    TEMP_LOSSING = float(999)

    # 初始化一个list，用于记录每个epoch的平均训练损失（也就是running_loss）
    loss_epoch = []

    # 开始循环训练模型
    for epoch in range(1, 50):

        # 进入训练模式
        model.train()

        # 初始化一个running_loss为0，记录每个batch的loss情况。
        running_loss = 0

        # 循环遍历训练集中所有的batch。
        for train_step, train_data in enumerate(trainLoader):

            # 将输入和标签数据转换成torch.FloatTensor类型
            inputs, labels = train_data
            inputs, labels = inputs.float(), labels.float()

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播，计算输出
            outputs = model(inputs)

            # 计算损失
            loss = loss_function(outputs.squeeze(), labels.squeeze())

            # 反向传播，计算每个参数的梯度
            loss.backward()

            # 更新权重
            optimizer.step()

            # 累加running_loss
            running_loss += loss.detach().numpy()

        # 根据训练损失和step_size、gamma修改当前的学习率
        scheduler.step()

        # 记录当前epoch的平均训练损失
        loss_epoch.append(running_loss)

        # 如果当前running_loss比之前所有的都小，就更新最小的loss值
        if running_loss < TEMP_LOSSING:
            TEMP_LOSSING = running_loss

    # 返回最小的训练损失
    return TEMP_LOSSING

