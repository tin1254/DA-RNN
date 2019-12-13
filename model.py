from modules import init_hidden, Encoder, Decoder
import numpy as np
import torch
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def toTorch(data):
    return torch.from_numpy(data).float().to(device)


class DA_RNN:
    def __init__(self, X_dim, Y_dim, encoder_hidden_size=64, decoder_hidden_size=64,
                 linear_dropout=0, T=10, learning_rate=1e-5, batch_size=128, decay_rate=0.9):
        self.T = T
        # self.decay_rate = decay_rate
        self.batch_size = batch_size
        # self.learning_rate = learning_rate
        self.X_dim = X_dim
        self.Y_dim = Y_dim

        self.encoder = Encoder(X_dim, encoder_hidden_size, T, linear_dropout).to(device)
        self.decoder = Decoder(encoder_hidden_size, decoder_hidden_size, T, linear_dropout, Y_dim).to(device)

        self.encoder_optim = torch.optim.Adam(params=self.encoder.parameters(), lr=learning_rate)
        self.decoder_optim = torch.optim.Adam(params=self.decoder.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()

    def ToTrainingBatches(self, X, Y, perm_idx=None):
        X_batches = []
        Y_batches = []

        N = X.shape[0]
        batch_num = math.ceil((N-self.T)/self.batch_size)
        i = self.T-1

        if perm_idx is not None:
            X = X[perm_idx]
            Y = Y[perm_idx] if len(Y.shape) > 1 else Y[perm_idx, np.newaxis]

        for b in range(batch_num):
            # number of output = N - T + 1
            # N is length, i is an index
            _batch_size = self.batch_size if N-i >= self.batch_size else N-i
            X_batch = np.empty((_batch_size, self.T, self.X_dim))
            Y_batch = np.empty((_batch_size, self.Y_dim))

            for b_idx in range(_batch_size):
                # print(N, i, i-self.T+1, i+1)
                # print(X[i-self.T+1:i+1].shape)
                X_batch[b_idx, :, :] = X[i-self.T+1:i+1]
                Y_batch[b_idx, :] = Y[i]
                i += 1

            X_batches.append(X_batch)
            Y_batches.append(Y_batch)

        # print(X.shape[0], np.sum([_.shape[0] for _ in X_batches]))
        return X_batches, Y_batches

    def ToTestingBatch(self, X):
        N = X.shape[0]
        i = self.T-1

        X_batch = np.empty((N-self.T+1, self.T, self.X_dim))
        b_idx = 0
        while i < N:
            X_batch[b_idx, :, :] = X[i-self.T+1:i+1]
            i += 1
            b_idx += 1
        return X_batch

    def train(self, X_train, Y_train, X_val, Y_val, epochs):
        if len(Y_train.shape) == 1:
            Y_train = Y_train[:, np.newaxis]
        if len(Y_val.shape) == 1:
            Y_val = Y_val[:, np.newaxis]

        assert len(X_train) == len(Y_train)
        assert len(X_val) == len(Y_val)

        self.encoder.train()
        self.decoder.train()

        epoch_loss_hist = []
        iter_loss_hist = []

        N = X_train.shape[0]
        iter_num = (N-self.T)//self.batch_size

        for _e in range(epochs):

            # TODO: use slice instead
            # perm_idx = np.random.permutation(N - self.T)

            X_train_batches, Y_train_batches = self.ToTrainingBatches(X_train, Y_train)
            for X_train_batch, Y_train_batch in zip(X_train_batches, Y_train_batches):
                X_train_loss = self.train_iter(X_train_batch, Y_train_batch)
                iter_loss_hist.append(np.mean(X_train_loss))

            epoch_loss_hist.append(np.mean(iter_loss_hist[-iter_num:]))

            if _e % 2 == 0:
                print("Epoch: {}\t".format(_e), end="")
                X_val_batch = self.ToTestingBatch(X_val)
                Y_val_pred = self._predict(X_val_batch)
                Y_val_loss = self.loss_func(Y_val_pred, toTorch(Y_val[-(N-self.T+1):]))
                print("train_loss: {} val_loss: {:.4f}".format(X_train_loss, Y_val_loss))

        return epoch_loss_hist, iter_loss_hist

    def train_iter(self, X, Y):
        self.encoder.train(), self.decoder.train()
        self.encoder_optim.zero_grad(), self.decoder_optim.zero_grad()

        _, X_encoded = self.encoder(toTorch(X))
        Y_pred = self.decoder(X_encoded)

        loss = self.loss_func(Y_pred, toTorch(Y))
        loss.backward()

        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.item()

    # TODO
    
    # def predict(self,X):
    #     # used for inference
    #     self.encoder.eval(), self.decoder.eval()
    #     X_batches = self.dataToBatches(X)

    #     _, X_encoded = self.encoder(toTorch(X_batches))
    #     Y_pred = self.decoder(X_encoded)

    #     Y_pred.view(-1, self.Y_dim)
    #     Y_pred = Y_pred.cpu()

    #     return Y_pred

    def _predict(self, X):
        # used during training

        # switch to evaluation mode if predicting
        # required if the model is using dropout
        self.encoder.eval(), self.decoder.eval()

        _, X_encoded = self.encoder(toTorch(X))
        Y_pred = self.decoder(X_encoded)

        return Y_pred
