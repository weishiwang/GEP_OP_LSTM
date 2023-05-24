import matplotlib.pyplot as plt
from pylab import xticks,yticks,np
from pylab import *
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset  # 抽象类不能实例化
from torch.utils.data import DataLoader, TensorDataset  # dataloader可实例化


df = pd.read_csv("sangebijiao.csv")
plt.rcParams['font.family'] = 'SimHei'
     # data = np.concatenate([test_y, pred_y], axis=1)
    # df=pd.DataFrame(data,columns=["true","predict"])
    # df.to_csv("..//result//true_pred.csv",index=False)
x = ["2021/9/20","2021/11/30","2022/2/10","2022/4/24","2022/7/5","2022/9/12","2022/11/16"]
plt.figure(figsize=(12, 6))
plt.xlabel("时间日期",fontsize="20")  # x轴名称
plt.ylabel("收盘价",fontsize="20")  # y 轴名称
# plt.xticks(x)
plt.xticks(np.linspace(0, 300, 7, endpoint=True), x,size = 18)
plt.yticks(size = 18)
plt.title("GEP优化LSTM预测结果与其他方法的比较",size=20)
plt.plot(df['GEP'], label="GEP-OP-LSTM",linewidth =1.0,color='b')
plt.plot(df['bayes'], label="bayes－OP－LSTM",linewidth =1.0,color='g')
plt.plot(df['wangge'], label="RandomSearch－OP－LSTM",linewidth =1.0,color='purple')
plt.plot(df['true'], label="true",linewidth =1.0,color='r')
plt.legend(fontsize="20")
plt.show()