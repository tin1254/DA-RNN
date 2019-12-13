import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size)).to(device)


# def roll(x, n):
#     """
#     use with 2D tensor, usage see numpy.roll
#     """
#     return torch.cat((x[:, -n], x[:, :-n]), dim=1)


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int, linear_dropout=0):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + self.T, out_features=1)  # out_features=T if we want to have v_e & tanh
        self.attn_dropout = nn.Dropout(linear_dropout)

    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)

        input_weighted = Variable(torch.zeros(input_data.size(0), self.T , self.input_size)).to(device)  # batch, T, n
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T , self.hidden_size)).to(device)  # batch, T, n
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size).to(device)  # 1, batch_size, hidden_size
        cell = init_hidden(input_data, self.hidden_size).to(device)    # 1, batch_size, hidden_size

        for t in range(self.T ):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)
            # batch_size * input_size * (2*hidden_size + T - 1)

            # row: feature, column: h_(t-1) & cell & input_data

            # hidden     batch_size, input_size, hidden_size
            # cell       batch_size, input_size, hidden_size

            # input_data batch_size, input_size, T
            # row: feature, column: time steps

            # attn_linear: W_e & U_e

            # Eqn. 8: Get attention weights, doesn't use v_e and tanh
            # e_t
            x = self.attn_dropout(x)
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T ))  # (batch_size * input_size) * 1

            # Eqn. 9: Softmax the attention weights
            # alpha_t
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)

            # Eqn. 10: LSTM
            # ~x
            # multiple each feature element-wise
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))

            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, linear_dropout=0, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(),
                                        nn.Linear(encoder_hidden_size, 1))  # out_features=hidden_size if we want to have v_d & tanh

        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc_dropout = nn.Dropout(linear_dropout)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final_dropout = nn.Dropout(linear_dropout)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size)).to(device)

        y_tilde = Variable(torch.zeros(input_encoded.size(0), 1)).to(device)

        for t in range(self.T):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.T),
                dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            # if on_train:
            #     x = torch.cat((context, y_history[:, t]), dim=1)  # (batch_size, out_size)
            # else:
            #     # use the previous predictions as input
            x = torch.cat((context, y_tilde), dim=1)  # (batch_size, out_size)

            y_tilde = self.fc(x)

            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        x = self.fc_final_dropout(torch.cat((hidden[0], context), dim=1))
        return self.fc_final(x)
