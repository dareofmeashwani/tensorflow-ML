import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import os
import sys
sys.path.insert(0,'../')
import utils
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]
DATASET_PATH = "/media/batman/ent/datasets/uep/UCI HAR Dataset/UCI HAR Dataset/"
TRAIN = "train/"
TEST = "test/"


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1
X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

x_train = load_X(X_train_signals_paths)
x_test = load_X(X_test_signals_paths)

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)
train={}
train['x']=x_train
train['y']=utils.convert_to_onehot(y_train,6)
val={}
val['x']=x_test
val['y']=utils.convert_to_onehot(y_test,6)
print(train['x'].shape,train['y'].shape,val['x'].shape,val['y'].shape)

import rnn_classifier
text_labels=LABELS
model=rnn_classifier.model()
model.batch_size=128
model.epochs=10
model.learning_rate=0.0001
model.sequence_dimensions=9
model.sequence_length=128
model.no_of_cell=2
model.cell_size=32
model.no_of_classes=6
model.model_restore=True
model.hidden_layers=[50]
model.working_dir='rnn_test'
model.activation_list=['leaky_relu']
model.setup()
model.train(train,val)
model.clear()
