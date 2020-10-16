
from tensorflow.python.keras import backend

import numpy as np
from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import Conv1D,Conv2D, Activation, MaxPool2D, MaxPool1D, Flatten, Dense, Dropout,BatchNormalization

import time
import gzip
from glob import glob
import os


import pandas as pd

from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization


class SequenceCNNBaysianOptimzier():


    def __init__(self,search_params,optimization_params,train_dir):
        self.search_params=search_params
        self.optimization_params=optimization_params
        self.train_dir=train_dir
        self.feature_chunks=glob(os.path.join(train_dir,'*.gz'))
        self.label_chunks = glob(os.path.join(train_dir, '*.npy'))

        # extract dimensions from our data

        with gzip.GzipFile(self.feature_chunks[-1], "r") as handle:
            val_data=np.load(handle)
            self.val_seqs=val_data
            self.dna_dims=np.shape(val_data)

        val_labels=np.load(self.label_chunks[-1])
        self.val_labels=val_labels
        self.num_classes=val_labels.shape[1]
        self.num_chunks=len(self.feature_chunks)-1
        self.class_weights=np.load('../data/class_weights.npy')

        # lets set the validation data to just be the last one of the dataset





    def run_optimization(self):

        bo = BayesianOptimization(self.optimization_round, self.search_params)

        bo.maximize(init_points=0, n_iter=20, acq="ucb", kappa=5, **self.optimization_params)
        print(bo.res['max'])
        print(bo.res['all'])

    @classmethod
    def set_data_directories(cls):
        pass


    def optimization_round(self,total_epoch, filter_num, filter_len, num_dense_nodes):
        """Take number of epochs, number of filters, filter length, and number of fully connected nodes as inputs.
        Compute the maximum validation accuracy for that network divided by wall clock time.
        Return the result for Bayesian optimization.
        """

        start = time.time()
        ## load test set to get general information




        total_epoch = int(round(total_epoch))
        filter_num = int(round(filter_num))
        filter_len = int(round(filter_len))
        num_dense_nodes = int(round(num_dense_nodes))
        print("Epochs =", total_epoch, "| # Conv filters =", filter_num, "| Filter length =", filter_len,
              "| # Dense nodes =", num_dense_nodes)

        # model specification


        # model specification

        model = Sequential()
        model.add(Conv2D(filters=filter_num,kernel_size=(4,filter_len), input_shape=self.dna_dims[1:],
                                activation="relu",padding='same'))
        model.add(MaxPool2D(pool_size=(1,self.dna_dims[2])))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(num_dense_nodes))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        print(model.summary())

        max_val_acc = 0.0
        max_acc_pair = 0.0
        num_chunks = self.num_chunks

        epoch_train_acc = np.zeros((total_epoch, num_chunks))
        epoch_val_acc = np.zeros((total_epoch, 1))

        # train the model
        for e in range(total_epoch):
            print("Epoch =", e + 1, "out of", total_epoch)
            for f in range(num_chunks - 1):
                with gzip.GzipFile(self.feature_chunks[f], "r") as xhandle:

                    X_train = np.load(xhandle)

                y_train = np.load(self.label_chunks[f])

                history = model.fit(X_train, y_train, batch_size=8, \
                                    validation_split=0.0, epochs=1, verbose=1, class_weight=self.class_weights)
                epoch_train_acc[e, f] = history.history['accuracy'][0]

            # train final chunk and do validation
            with gzip.GzipFile(self.feature_chunks[num_chunks], "r") as xhandle:

                X_train = np.load(xhandle)

            y_train = np.load(self.label_chunks[num_chunks])


            history = model.fit(X_train, y_train, batch_size=8, \
                                validation_data=(self.val_seqs, self.val_labels), epochs=1, verbose=1,
                                class_weight=self.class_weights)
            epoch_train_acc[e, num_chunks - 1] = history.history['accuracy'][0]
            epoch_val_acc[e, 0] = history.history['val_acc'][0]

            # record max validation accuracy
            if history.history['val_acc'][0] > max_val_acc:
                max_val_acc = history.history['val_acc'][0]
                max_acc_pair = history.history['accuracy'][0]

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
    SequenceCNNBaysianOptimzier.set_data_directories()


    search_params={'total_epoch': (5, 5), 'filter_num': (1, 512), 'filter_len': (1, 48),
                                       'num_dense_nodes': (1, 256)}
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
    train_dir='../data/training_data'
    search=SequenceCNNBaysianOptimzier(search_params,gp_params,train_dir)
    # perform bayesian optimization within hyperparameter ranges, with initial guesses
    search.run_optimization()