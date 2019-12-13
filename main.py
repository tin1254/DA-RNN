from model import DA_RNN

from sklearn.utils import shuffle
import numpy as np

data = np.load("./data.npy")
data_size = len(data)
# (506, 15)
val_percent = 0.2
val_size = int(data_size*val_percent)
train_size = data_size-val_size

X_dim = 14
Y_dim = 1

X_train = data[:train_size, :-1]
Y_train = data[:train_size, -1]
X_val = data[val_size:, :-1]
Y_val = data[val_size:, -1]

model = DA_RNN(X_dim=X_dim, Y_dim=Y_dim, batch_size=64,
               learning_rate=5e-4, linear_dropout=0.3, T=20)
model.train(X_train, Y_train, X_val, Y_val, 200)
# print(model.predict(X_val))