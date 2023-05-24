
from LSTM_model.return_grid_lstm import *
from sklearn.model_selection import cross_val_score
import pickle
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

# 定义参数的搜索空间

space = {
         #    "max_depth":hp.randint("max_depth",15),
         # "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         # "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         # "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         # "min_child_weight":hp.randint("min_child_weight",5), #

            "n_neurouse": hp.randint("n_neurouse",0,201),
             "drop_rate": hp.uniform("drop_rate", 0, 1),
            "hidden_size": hp.randint("hidden_size",0,201),
            "Mini_batch": hp.randint("Mini_batch",0,6),
            "Active_function": hp.randint("Active_function",0,5),
             "Optimizer": hp.randint("Optimizer",0,5)
        }
# algo = partial(tpe.suggest,n_startup_jobs=1)  # 定义随机搜索算法。搜索算法本身也有内置的参数决定如何去优化目标函数
best = fmin(train_net,space,algo=rand.suggest,max_evals=500)  # 对定义的参数范围，调用搜索算法，对模型进行搜索

print(best)
# print(train_net(best))

