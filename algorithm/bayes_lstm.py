from LSTM_model.return_bayes_lstm import *
import LSTM_model.model_LSTM as lastrun
from bayes_opt import BayesianOptimization


positive_integer = []
for i in range(1,201):
    positive_integer.append(i)
optimizer = BayesianOptimization(
    f=train_net,
    pbounds={
            'n_neurouse': (1,201),
             'drop_rate': (0,1),
            'hidden_size': (1,201),
            'Mini_batch': (0,5),
            'Active_function': (0,4),
             'Optimizer': (0,4)
    },
    random_state=0,
    verbose=2
)

optimizer.maximize(init_points=20, n_iter=180)
print("Final result:", optimizer.max)