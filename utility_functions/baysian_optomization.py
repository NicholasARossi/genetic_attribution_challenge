
from tensorflow.python.keras import backend

import numpy as np
from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import Conv1D,Conv2D, Activation, MaxPool2D, MaxPool1D, Flatten, Dense, Dropout,BatchNormalization

import time


global train_pi_labels_onehot
global train_dna_seqs
global cl_weight
global val_pi_labels_onehot
global val_dna_seqs_onehot
global dna_bp_length

import pandas as pd

from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization


def target(total_epoch, filter_num, filter_len, num_dense_nodes):
    """Take number of epochs, number of filters, filter length, and number of fully connected nodes as inputs.
    Compute the maximum validation accuracy for that network divided by wall clock time.
    Return the result for Bayesian optimization.
    """

    start = time.time()
    total_epoch = int(round(total_epoch))
    filter_num = int(round(filter_num))
    filter_len = int(round(filter_len))
    num_dense_nodes = int(round(num_dense_nodes))
    print("Epochs =", total_epoch, "| # Conv filters =", filter_num, "| Filter length =", filter_len,
          "| # Dense nodes =", num_dense_nodes)

    # model specification


    # model specification

    model = Sequential()
    model.add(Conv2D(filters=filter_num,kernel_size=(4,filter_len), input_shape=dna_dims[1:],
                            activation="relu",padding='same'))
    model.add(MaxPool2D(pool_size=(1,dna_dims[2])))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(num_dense_nodes))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    print(model.summary())

    max_val_acc = 0.0
    max_acc_pair = 0.0
    num_chunks = 6

    epoch_train_acc = np.zeros((total_epoch, num_chunks))
    epoch_val_acc = np.zeros((total_epoch, 1))

    # train the model
    for e in range(total_epoch):
        print("Epoch =", e + 1, "out of", total_epoch)
        for f in range(num_chunks - 1):
            X_train = np.load("/mnt/data" + str(f) + ".npy")
            y_train = np.load("/mnt/labels" + str(f) + ".npy")
            history = model.fit(X_train, y_train, batch_size=8, \
                                validation_split=0.0, nb_epoch=1, verbose=1, class_weight=cl_weight)
            epoch_train_acc[e, f] = history.history['acc'][0]

        # train final chunk and do validation
        X_train = np.load("/mnt/data" + str(num_chunks - 1) + ".npy")
        y_train = np.load("/mnt/labels" + str(num_chunks - 1) + ".npy")
        history = model.fit(X_train, y_train, batch_size=8, \
                            validation_data=(val_dna_seqs_onehot, val_pi_labels_onehot), nb_epoch=1, verbose=1,
                            class_weight=cl_weight)
        epoch_train_acc[e, num_chunks - 1] = history.history['acc'][0]
        epoch_val_acc[e, 0] = history.history['val_acc'][0]

        # record max validation accuracy
        if history.history['val_acc'][0] > max_val_acc:
            max_val_acc = history.history['val_acc'][0]
            max_acc_pair = history.history['acc'][0]

    # save network stats
    print("Epoch training accuracy")
    print(epoch_train_acc)
    print("Mean epoch training accuracy")
    print(np.transpose(np.mean(epoch_train_acc, axis=1)))
    end = time.time()
    np.save(str(int(end)) + 'conv' + str(filter_num) + 'x' + str(filter_len) + 'dense' + str(
        num_dense_nodes) + 'time' + str(int(end - start)) + '_mean_train_acc.out',
            np.transpose(np.mean(epoch_train_acc, axis=1)))
    print("Epoch validation accuracy")
    print(epoch_val_acc)
    np.save(str(int(end)) + 'conv' + str(filter_num) + 'x' + str(filter_len) + 'dense' + str(
        num_dense_nodes) + 'time' + str(int(end - start)) + '_epoch_val_acc.out', epoch_val_acc, end - start)

    return max_val_acc / (end - start)

if __name__ == '__main__':
    from DNA_CNN_utils import convert_DNA_onehot2D

    upper_dir='../data/'
    test_values=pd.read_csv(f'{upper_dir}test_values.csv').head(100)
    train_labels=pd.read_csv(f'{upper_dir}train_labels.csv').head(100)
    train_values=pd.read_csv(f'{upper_dir}train_values.csv').head(100)
    submission_format=pd.read_csv(f'{upper_dir}submission_format.csv')

    X = convert_DNA_onehot2D(list(train_values.sequence))[..., np.newaxis]
    y =train_labels[train_labels.columns[1:]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=88)




    # load data
    train_pi_labels_onehot = y_train
    train_dna_seqs = X_train

    val_pi_labels_onehot = y_test
    val_dna_seqs = X_test

    global num_classes
    num_classes = val_pi_labels_onehot.shape[1]
    global dna_dims
    dna_dims = X.shape

    # perform bayesian optimization within hyperparameter ranges, with initial guesses
    print("Start Bayesian optimization")
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    bo = BayesianOptimization(target, {'total_epoch': (5, 5), 'filter_num': (1, 512), 'filter_len': (1, 48),
                                       'num_dense_nodes': (1, 256)})

    bo.maximize(init_points=0, n_iter=20, acq="ucb", kappa=5, **gp_params)

    # print output values from bayesian optimization
    print(bo.res['max'])
    print(bo.res['all'])