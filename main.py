import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import helpers
from helpers.data_processing import build_sequence_data
from helpers.models import convLSTM_model

# Set the random seed for reproducible results
np.random.seed(123)
tf.random.set_seed(123)

# Import training, validation, and test data
train_file = 'MilanoTrain40.csv'
test_file = 'MilanoTest5060.csv'
val_file = 'MilanoValidation4050.csv'

train_path = os.path.join(os.getcwd(), train_file)
test_path = os.path.join(os.getcwd(), test_file)
val_path = os.path.join(os.getcwd(), val_file)

dataset_train = pd.read_csv(train_path, header=None).values
dataset_test = pd.read_csv(test_path, header=None).values
dataset_val = pd.read_csv(val_path, header=None).values

# Parameter initialization
r = 10 # size of spatial data
recurrent_length = 12 # number of 10 minutes intervals
rec_out=3 # number of predicted time slots in future
train_length = 6*24*40 # length of training data
test_length = 6*24*10 # length of test data
val_length = 6*24*10 # length of validation data
area_length = 100 # Length of the area of the collected data
starting_row = 6050 # start row in the original dataset
start_row, end_row = 60, 70 # start and end rows in the processed input data
start_col, end_col = 50, 60 # start and end columns in the processed input data
area_borders = [start_row, end_row, start_col, end_col]

X_train, Y_train, min_data, max_data = build_sequence_data(dataset_train, recurrent_length, r, area_borders, train_length, rec_out, area_length)
X_val, Y_val, _, _ = build_sequence_data(dataset_val, recurrent_length, r, area_borders, val_length, rec_out, area_length)
X_test, Y_test, _, _ = build_sequence_data(dataset_test, recurrent_length, r, area_borders, test_length, rec_out, area_length)

# Normalize the data
X_train_new = (X_train-min_data)/(max_data-min_data)
Y_train_new = (Y_train-min_data)/(max_data-min_data)
X_val_new = (X_val-min_data)/(max_data-min_data)
Y_val_new = (Y_val-min_data)/(max_data-min_data)
X_test_new = (X_test-min_data)/(max_data-min_data)
Y_test_new = (Y_test-min_data)/(max_data-min_data)


# Reshape input data to be copatible to the ConvLSTM input format
x_train=np.reshape(X_train_new,(X_train_new.shape[0],recurrent_length,10,10,1))
print('X train shape: {}'.format(x_train.shape))

y_train=np.reshape(Y_train_new,(Y_train_new.shape[0],rec_out,10,10))
print('Y train shape: {}'.format(y_train.shape))

x_val=np.reshape(X_val_new,(X_val_new.shape[0],recurrent_length,10,10,1))
print('X validation shape: {}'.format(x_val.shape))

y_val=np.reshape(Y_val_new,(Y_val_new.shape[0],rec_out,10,10))
print('Y validation shape: {}'.format(y_val.shape))

x_test=np.reshape(X_test_new,(X_test_new.shape[0],recurrent_length,10,10,1))
print('X test shape: {}'.format(x_test.shape))

y_test=np.reshape(Y_test_new,(Y_test_new.shape[0],rec_out,10,10))
print('Y test shape: {}'.format(y_test.shape))

# Concatenate train and validation data for K-fold cross-validation
X = np.concatenate((x_train,x_val), axis = 0)
y = np.concatenate((y_train,y_val), axis = 0)
print('X final shape: {}'.format(X.shape))
print('Y final shape: {}'.format(y.shape))


# List of number of filters for different layers in our model
filters = [32, 1, 6, rec_out]


# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))
# Define network output
out = convLSTM_model(inp, filters, recurrent_length, r)

# Next, we will build the complete model and compile it.
model = tf.keras.Model(inp, out)
model.compile(
    loss='mse', optimizer=keras.optimizers.Adam(),
)
model.save_weights('model.h5')
model.summary()

# Training
# Fit the model to the training data.
# number of implementation
K=5
# We will change the batch size gradually to optimize the training process
batch_size_initial = 64
batch_size_final = 512
num_epochs = 200

# Kfold cross validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, random_state=None, shuffle=True)
mse_total = []
mse_test = []

for counter, (train, test) in enumerate(kfold.split(X, y), 1):
    X_train = X[train, :, :]
    Y_train = y[train, :, :]
    X_test = X[test, :, :]
    Y_test = y[test, :, :]
    
    # reset model to random initial weights
    model.load_weights('model.h5')
    for epoch in range(num_epochs):
        batch_size = int(batch_size_initial + (batch_size_final - batch_size_initial) * epoch / num_epochs)
        history = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=(X_test, Y_test)
        )
        print(f"Epoch {epoch+1}/{num_epochs}, Batch Size: {batch_size}")

    model.save_weights(f"model{counter}.h5")

    mse_total.append(np.mean((model.predict(X_test) - Y_test) ** 2))
    mse_test.append(np.mean((model.predict(x_test) - y_test) ** 2))
    print(mse_total[-1])
    print(mse_test[-1])

print("The number of trainings:", counter)
mse_total_mean = np.mean(mse_total)
print(mse_total_mean)
mse_test_mean = np.mean(mse_test)
print(mse_test_mean)
