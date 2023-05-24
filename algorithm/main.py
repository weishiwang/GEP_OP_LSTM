from LSTM_model.model_LSTM import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 主函数
    # n_layers = 2  # LSTM网络层数 正整数，不要太大，默认2 不变了
    n_neurouse = 152  # 神经元个数，正整数即可
    Active_function = 2  # 取值范围：0,1，2，3
    drop_rate = 0.01  # 0--1之间的小数
    hidden_size = 182  # 神经层元个数，正整数即可
    Optimizer = 0  # 取值：0 1 2 3三个
    # Learning_rate = 0.0001  # 一般在0.00001--0.01之间 不变了
    Mini_batch = 64  # 正整数即可一般32  64  128

    mes = train_net(n_neurouse, drop_rate,
                        hidden_size, Mini_batch, Active_function, Optimizer)

    print(mes)