import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import pickle
import time

subject_id = 7

np.random.seed(33)

# Load and preprocess dataset
dataset_dir = "../../training dataset for CRNN/3D_CNN/"
with open(dataset_dir + str(subject_id) + "_shuffle_dataset_3D_win_10.pkl", "rb") as fp:
    datasets = pickle.load(fp)
with open(dataset_dir + str(subject_id) + "_shuffle_labels_3D_win_10.pkl", "rb") as fp:
    labels = pickle.load(fp)

print("\n reading: " + dataset_dir + str(subject_id) + "_shuffle_dataset_3D_win_10.pkl \n")

window_size = 10
datasets = datasets.reshape(len(datasets), window_size, 10, 11, 1)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

split = np.random.rand(len(datasets)) < 0.75
train_x, test_x = datasets[split], datasets[~split]
train_y, test_y = labels[split], labels[~split]

# Parameters
input_height, input_width, input_channel_num = 10, 11, 1
n_labels = 6 # 5 -> 6
fc_size = 1024
n_lstm_layers = 2
dropout_prob = 0.5
learning_rate = 1e-4
batch_size = 300
training_epochs = 10 # 300 -> 10

# Build the model
inputs = tf.keras.Input(shape=(window_size, input_height, input_width, input_channel_num))

# Convolutional layers
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(32, (3, 3), activation='elu', padding='same'))(inputs)
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='same'))(x)
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(128, (3, 3), activation='elu', padding='same'))(x)

# Flatten and Fully Connected
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(fc_size, activation='elu'))(x)
x = tf.keras.layers.Dropout(dropout_prob)(x)

# LSTM layers
for _ in range(n_lstm_layers - 1):
    x = tf.keras.layers.LSTM(fc_size, return_sequences=True)(x)
x = tf.keras.layers.LSTM(fc_size, return_sequences=False)(x)

# Fully connected and output layer
x = tf.keras.layers.Dense(fc_size, activation='elu')(x)
x = tf.keras.layers.Dropout(dropout_prob)(x)
outputs = tf.keras.layers.Dense(n_labels, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_x, train_y, batch_size=batch_size, epochs=training_epochs, validation_data=(test_x, test_y))

# Evaluation
test_loss, test_accuracy = model.evaluate(test_x, test_y)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predictions and metrics
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(test_y, axis=1)

recall = recall_score(y_true_classes, y_pred_classes, average=None)
precision = precision_score(y_true_classes, y_pred_classes, average=None)
f1 = f1_score(y_true_classes, y_pred_classes, average=None)
auc_values = roc_auc_score(test_y, y_pred, average=None)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("AUC:", auc_values)
print("Confusion Matrix:\n", conf_matrix)
