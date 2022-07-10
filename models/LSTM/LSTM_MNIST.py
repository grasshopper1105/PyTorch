import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.rnn = nn.Sequential(
            nn.LSTM(
                input_size=32,
                hidden_size=256,  # rnn hidden unit
                num_layers=3,  # 有几层 RNN layers
                batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            ))

        self.dnn = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)

        rnn1 = self.rnn(x.squeeze(1))
        # print(cnn1.shape)
        r_out, (h_n, h_c) = rnn1
        # print(r_out[:, -1, :].shape)
        out = self.dnn(r_out[:, -1, :])

        return (out)


def LSTM_MNIST():
    return NN()

# my = NN()
# input = torch.randn((64, 600, 16))
# logits = my(input)
# print(logits.detach().cpu().shape)
