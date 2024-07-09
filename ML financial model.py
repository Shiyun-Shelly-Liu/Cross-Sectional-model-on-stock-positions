# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 08:48:02 2024

@author: Shiyun Liu
"""
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import pandas as pd
from keras.callbacks import Callback, ModelCheckpoint
import pickle
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Conv1D, Flatten
from keras.models import Sequential
from keras.layers import SimpleRNN
from tensorflow.keras.layers import Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
import csv
from io import StringIO
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Input, concatenate, Dropout, Masking
from keras.models import Model
import tensorflow as tf
import ast
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from scikeras.wrappers import KerasClassifier
import openpyxl
# %% input&output normalisation

# # Step 1: Read the Excel file into pandas DataFrame
# file_path = 'AAPL_input.xlsx'
# df = pd.read_excel(file_path)

# # Step 2: Convert 'Date' column to numerical values
# # Assuming 'Date' column is already in datetime format, if not, convert it first
# df['Numerical Date'] = df['Date'].apply(lambda x: x.timestamp())

# # Step 3: Normalize columns to values within 0 and 1 using MinMaxScaler
# scaler = MinMaxScaler()
# # Selecting columns to normalize (excluding 'Date' and any other non-numeric columns)
# cols_to_normalize = df.columns.difference(['Date', 'Non-Numeric Column'])
# df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# # Step 4: Optionally, save the modified DataFrame back to Excel
# df.to_excel('AAPL_input.xlsx', index=False)


# # normalize outpot
# # Load the output data
# df_output = pd.read_excel('AAPL_output.xlsx')

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Assuming the output column is named 'stock_price'
# # Fit and transform the data
# df_output['Close'] = scaler.fit_transform(df_output[['Close']])

# # Save the normalized output data back to an Excel file (optional)
# df_output.to_excel('AAPL_output.xlsx', index=False)


# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

# Load data
df_input = pd.read_excel('AAPL_input.xlsx')
df_output = pd.read_excel('AAPL_output.xlsx')

# Prepare input and output data
# Assuming df_input has your features and df_output has your output column (e.g., 'stock_price')

# Initialize MinMaxScaler for input data
scaler_input = MinMaxScaler()
cols_to_normalize = df_input.columns.difference(['Date'])
scaled_input = scaler_input.fit_transform(df_input[cols_to_normalize])

# # Initialize MinMaxScaler for output data
# scaler_output = MinMaxScaler()
# scaled_output = scaler_output.fit_transform(df_output[['Close']])

# Define your sequence length
sequence_length = 5

X = []
y = []

# Assuming df_output has a column 'Close' which is your output
for i in range(sequence_length, len(scaled_input)):
    # Adjust if you have multiple features
    X.append(scaled_input[i-sequence_length:i, :])
    # Assuming the output column is the first column
    y.append(df_output.iloc[i, 0])

X, y = np.array(X), np.array(y)

# Split data into training and validation sets
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %%
# Define the create_model function


def create_model():
    model = Sequential()

    model.add(Bidirectional(LSTM(units=50, return_sequences=True),
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dropout(0.2))
    # Dense output layer
    model.add(Dense(units=1))

    # Compile the model with Mean Squared Error loss function
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Create the model
model = create_model()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=50,
                    validation_data=(X_test, y_test), shuffle=False)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Make predictions on the test set
predicted_probabilities = model.predict(X_test)
predicted_classes = (predicted_probabilities > 0.5).astype('int32')

# Calculate accuracy manually (if needed)
test_accuracy = accuracy_score(y_test, predicted_classes)
print(f'Manual test accuracy: {test_accuracy}')

# %% optimise epoch number


def create_model():
    input_fixed = Input(shape=(5,))
    input_variable = Input(shape=(2, 69))

    masking_layer = Masking(mask_value=0.0)(input_variable)

    lstm_layer_forward = LSTM(
        units=50, activation='tanh', return_sequences=False)(masking_layer)
    lstm_layer_backward = LSTM(units=50, activation='tanh',
                               return_sequences=False, go_backwards=True)(masking_layer)

    lstm_sum = concatenate([lstm_layer_forward, lstm_layer_backward])

    dense_layer1 = Dense(units=500, activation='relu')(lstm_sum)
    dropout_layer1 = Dropout(0.)(dense_layer1)

    dense_layer2 = Dense(units=500, activation='relu')(dropout_layer1)
    dropout_layer2 = Dropout(0.)(dense_layer2)

    merged_layer = concatenate([dropout_layer2, input_fixed])

    dense_layer3 = Dense(units=500, activation='relu')(merged_layer)
    dropout_layer3 = Dropout(0.)(dense_layer3)

    dense_layer4 = Dense(units=500, activation='relu')(dropout_layer3)
    dropout_layer4 = Dropout(0.)(dense_layer4)

    output_layer = Dense(units=136, activation='softmax')(dropout_layer4)

    model = Model(inputs=[input_fixed, input_variable], outputs=output_layer)
    return model


# Compile the model with a lower learning rate, binary crossentropy loss, and sigmoid activation for binary classification


# Define number of folds
n_splits = 3
# Define K-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True)


# Define the TestAccuracyCallback class

class TestAccuracyCallback(Callback):
    def __init__(self, X_fixed_test, X_variable_test, y_test, test_accuracies):
        self.X_fixed_test = X_fixed_test
        self.X_variable_test = X_variable_test
        self.y_test = y_test
        self.test_accuracies = test_accuracies

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(
            [self.X_fixed_test, self.X_variable_test], self.y_test, verbose=0)
        print(f'Test accuracy at epoch {epoch + 1}: {test_accuracy}')
        self.test_accuracies[epoch].append(test_accuracy)
        if logs is not None:
            logs['test_accuracy'] = test_accuracy


# Define lists to store histories
all_histories_25 = []
all_histories_50 = []
all_histories_100 = []
all_histories_200 = []

fold_index = 1
for train_index, val_index in kf.split(X_fixed_train):
    print(f"Fold {fold_index}/{n_splits}")

    X_fixed_train_fold, X_fixed_val_fold = X_fixed_train[train_index], X_fixed_train[val_index]
    X_variable_train_fold, X_variable_val_fold = X_variable_train[
        train_index], X_variable_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    print(len(X_fixed_train_fold), len(X_fixed_val_fold))

    # Create and compile the model
    model = create_model()
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(
    ), metrics=['accuracy'])

    # Define checkpoint callbacks
    checkpoint_25 = ModelCheckpoint(
        f'model_epoch_25_fold_{fold_index}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint_50 = ModelCheckpoint(
        f'model_epoch_50_fold_{fold_index}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # checkpoint_100 = ModelCheckpoint(
    #     f'model_epoch_100_fold_{fold_index}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # checkpoint_200 = ModelCheckpoint(
    #     f'model_epoch_200_fold_{fold_index}.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Custom callback to save test accuracy at the end of each epoch
    test_accuracies_per_fold = [[]
                                for _ in range(50)]  # Assuming maximum 100 epochs
    test_accuracy_callback = TestAccuracyCallback(
        X_fixed_test, X_variable_test, y_test, test_accuracies_per_fold)

    # Train the model for 25 epochs
    history_25 = model.fit([X_fixed_train_fold, X_variable_train_fold], y_train_fold, epochs=25, batch_size=300,
                           validation_data=(
                               [X_fixed_val_fold, X_variable_val_fold], y_val_fold),
                           callbacks=[checkpoint_25, test_accuracy_callback])
    all_histories_25.append(history_25.history)

    # Load the best weights and continue training for another 25 epochs (total 50)
    model.load_weights(f'model_epoch_25_fold_{fold_index}.keras')
    history_50 = model.fit([X_fixed_train_fold, X_variable_train_fold], y_train_fold, epochs=25, batch_size=300,
                           validation_data=(
                               [X_fixed_val_fold, X_variable_val_fold], y_val_fold),
                           callbacks=[checkpoint_50, test_accuracy_callback])
    all_histories_50.append(history_50.history)

    # Train for another 50 epochs starting from epoch 51 with a checkpoint at epoch 100
    # Reload weights from the best model after 50 epochs
    # model.load_weights(f'model_epoch_50_fold_{fold_index}.keras')
    # history_100 = model.fit([X_fixed_train_fold, X_variable_train_fold], y_train_fold, epochs=50, batch_size=300,
    #                         validation_data=(
    #                             [X_fixed_val_fold, X_variable_val_fold], y_val_fold),
    #                         callbacks=[checkpoint_100, test_accuracy_callback])
    # all_histories_100.append(history_100.history)

    # Train for another 100 epochs starting from epoch 101 with a checkpoint at epoch 100
    # Reload weights from the best model after 100 epochs
    # model.load_weights(f'model_epoch_100_fold_{fold_index}.keras')
    # history_200 = model.fit([X_fixed_train_fold, X_variable_train_fold], y_train_fold, epochs=100, batch_size=300,
    #                         validation_data=(
    #     [X_fixed_val_fold, X_variable_val_fold], y_val_fold),
    #     callbacks=[checkpoint_200, test_accuracy_callback])
    # all_histories_200.append(history_200.history)
    fold_index += 1

# Save histories
with open('histories_25_test.pkl', 'wb') as f:
    pickle.dump(all_histories_25, f)

with open('histories_50_test.pkl', 'wb') as f:
    pickle.dump(all_histories_50, f)

# with open('histories_100_2nd.pkl', 'wb') as f:
#     pickle.dump(all_histories_100, f)

# with open('histories_200_2nd.pkl', 'wb') as f:
#     pickle.dump(all_histories_200, f)
# %% Build and Train the MLP

# Create the MLP model

# Create the MLP model
model = Sequential()
model.add(Dense(100, input_dim=time_step, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

# %% Evaluate the Model

# Predict on the test data
predictions = model.predict(X_test)

# Inverse transform to get actual prices
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual_prices = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Evaluate the model

plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
