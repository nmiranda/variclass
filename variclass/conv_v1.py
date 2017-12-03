# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import data

def scale_dataset(X, mag_range=(0,1)):
    """
    Function to scale the whole magnitude time series dataset to a given range
    """

    new_X = np.copy(X)

    data_min = np.amin(new_X, axis=None)
    data_max = np.amax(new_X, axis=None)

    data_range = data_max - data_min
    this_scale = (mag_range[1] - mag_range[0]) / data_range
    this_min = mag_range[0] - data_min * this_scale

    new_X *= this_scale
    new_X += this_min
    
    return new_X

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir")
    parser.add_argument('-t', '--top', type=float)
    parser.add_argument("-i", "--inner", action='store_true')
    parser.add_argument("-a", '--augment', type=int)
    parser.add_argument("-e", '--epochs', type=int, default=25)
    parser.add_argument("-s", '--simulate', type=int)
    args = parser.parse_args()

    # Model parameters
    input_dim = 2 # <----  dt, q(t-dt)
    input_dtype = 'float64'
    conv_filters = 32
    dense_activation = "sigmoid"
    batchsize = 32
    dense_dim = 64
    num_epochs = args.epochs
    window_size = 5
    dropout_rate = 0.25
    validation_ratio = 0.33
    scale = False
    stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')

    # Load light curve data
    if args.augment:
        jd_list, q_list, q_err_list, type_list = data.load(directory=args.dir, with_errors=True, sel_longest=args.top)
    elif args.simulate:
        jd_list, q_list, type_list = data.simulate(args.simulate)
    else:
        jd_list, q_list, type_list = data.load(directory=args.dir, with_errors=False, sel_longest=args.top)

    # Input data dimensions for matrix
    max_jd = max([x.shape[0] for x in jd_list])
    median_jd = int(np.median([x.shape[0] for x in jd_list]))
    num_samples = len(jd_list)

    # Initializing matrixes
    jd_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    for i, jd_array in enumerate(jd_list):
        jd_matrix[i,:jd_array.shape[0]] = jd_array

    q_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    for i, q_array in enumerate(q_list):
        q_matrix[i,:q_array.shape[0]] = q_array

    if args.augment:
        q_err_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
        for i, q_err_array in enumerate(q_err_list):
            q_err_matrix[i,:q_err_array.shape[0]] = q_err_array

    #class_matrix = np.asarray(type_list)[..., np.newaxis]
    class_matrix = type_list[..., np.newaxis]

    delta_jd_matrix = jd_matrix[:,1:] - jd_matrix[:,:-1]
    delta_jd_matrix = delta_jd_matrix.clip(min=0)

    q_matrix = q_matrix[:,:-1]
    if augment_data_factor:
        q_err_matrix = q_err_matrix[:,:-1]

    ### DEFINING MODEL
    _input = Input(shape=(max_jd-1, input_dim))
    conv_1 = Conv1D(conv_filters, window_size)(_input)
    max_pool_1 = MaxPooling1D(window_size)(conv_1)
    conv_2 = Conv1D(conv_filters, window_size)(max_pool_1)
    max_pool_2 = MaxPooling1D(window_size)(conv_2)
    conv_3 = Conv1D(conv_filters, window_size)(max_pool_2)
    max_pool_3 = MaxPooling1D(window_size)(conv_3)
    dropout_1 = Dropout(dropout_rate)(max_pool_3)
    flatten = Flatten()(dropout_1)
    dense_1 = Dense(dense_dim)(flatten)
    dropout_2 = Dropout(dropout_rate)(dense_1)
    dense_2 = Dense(1, activation=dense_activation)(dropout_2)

    model = Model(inputs=[_input], outputs=[dense_2])

    model_optimizer = Adam()
    #model_optimizer = SGD(lr=0.05)

    model.compile(optimizer=model_optimizer, loss="binary_crossentropy", metrics=['accuracy'])

    #train_delta_jd = delta_jd_matrix
    #train_q = q_matrix[:,:-1]
    #train_class = class_matrix

    #train_X = np.stack((train_delta_jd, train_q), axis=2)
    #train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2])

    """
    histories = list()

    for i in xrange(3):

        train_X, test_X, train_class, test_class = train_test_split(train_X, train_class, test_size=validation_ratio)

        start_time = time.time()
        history = model.fit(x=[train_X], y=[train_class], batch_size=batchsize, epochs=num_epochs, validation_data=(test_X, test_class))
        total_time = time.time() - start_time

        # Evaluating model
        test_dataset = test_X
        test_predict = model.predict(test_dataset)

        test_predict = np.reshape(test_predict, (test_predict.shape[0]))
        test_predict = 1.0*(test_predict > 0.5)

        rec_score = metrics.recall_score(np.reshape(test_class.astype('int'), test_class.shape[0]), test_predict.astype('int'))
        f1_score = metrics.f1_score(np.reshape(test_class.astype('int'), test_class.shape[0]), test_predict.astype('int'))
        acc_score = metrics.accuracy_score(np.reshape(test_class.astype('int'), test_class.shape[0]), test_predict.astype('int'))

        histories.append({
            'start_time': start_time,
            'end_time': end_time,
            'history': history.history
            'rec_score': rec_score
            'f1_score': f1_score,
            'acc_score': acc_score
            })

        print("RECALL_SCORE:", rec_score)

    """

    if args.augment:

        augment_data_factor = args.augment

        train_delta_jd_matrix, test_delta_jd_matrix, train_q_matrix, test_q_matrix, train_q_err_matrix, test_q_err_matrix, train_class_matrix, test_class_matrix = train_test_split(delta_jd_matrix, q_matrix, q_err_matrix, class_matrix, test_size=validation_ratio)

        augmented_train_q_matrix = np.full((train_q_matrix.shape[0] * augment_data_factor, max_jd-1), 0., dtype=input_dtype)
        augmented_train_delta_jd_matrix = np.full((train_q_matrix.shape[0] * augment_data_factor, max_jd-1), 0., dtype=input_dtype)
        augmented_train_class_matrix = np.full((train_q_matrix.shape[0] * augment_data_factor, 1), 0)
        for num_augment_cycle in range(augment_data_factor):
            for i in xrange(train_q_matrix.shape[0]):
                for j in xrange(train_q_matrix.shape[1]):
                    augmented_train_q_matrix[(num_augment_cycle * train_q_matrix.shape[0] + i), j] = np.random.normal(train_q_matrix[i,j], train_q_err_matrix[i,j])
                    augmented_train_delta_jd_matrix[(num_augment_cycle * train_q_matrix.shape[0] + i), j] = train_delta_jd_matrix[i,j]
                augmented_train_class_matrix[(num_augment_cycle * train_q_matrix.shape[0] + i), 0] = train_class_matrix[i,0]

        train_X = np.stack((augmented_train_delta_jd_matrix, augmented_train_q_matrix), axis=2)
        train_class = augmented_train_class_matrix
        test_X = np.stack((test_delta_jd_matrix, test_q_matrix), axis=2)
        test_class = test_class_matrix


    #train_X, test_X, train_class, test_class = train_test_split(train_X, train_class, test_size=validation_ratio)

    #import ipdb;ipdb.set_trace()

    start_time = time.time()
    #history = model.fit(x=[train_X], y=[train_class], batch_size=batchsize, epochs=num_epochs, validation_data=(test_X, test_class))
    history = model.fit(x=[train_X], y=[train_class], batch_size=batchsize, epochs=num_epochs)
    total_time = time.time() - start_time

    # Evaluating model
    test_dataset = test_X
    test_predict = model.predict(test_dataset)

    test_predict = np.reshape(test_predict, (test_predict.shape[0]))
    test_predict = 1.0*(test_predict > 0.5)

    rec_score = metrics.recall_score(np.reshape(test_class.astype('int'), test_class.shape[0]), test_predict.astype('int'))
    f1_score = metrics.f1_score(np.reshape(test_class.astype('int'), test_class.shape[0]), test_predict.astype('int'))
    acc_score = metrics.accuracy_score(np.reshape(test_class.astype('int'), test_class.shape[0]), test_predict.astype('int'))

    print("RECALL_SCORE:", rec_score)

    with open(os.path.join(stats_dir, 'conv_' + str(int(start_time)) + '.txt'), 'w') as stats_file:

        print type(history.history)
        stats_file.write("Number of samples: " + str(num_samples) + "\n")
        stats_file.write("Validation ratio: " + str(validation_ratio) + "\n")
        stats_file.write("Scale dataset: " + str(scale) + "\n")
        stats_file.write("Conv filters: " + str(conv_filters) + "\n")
        stats_file.write("Dense activation: " + str(dense_activation) + "\n")
        stats_file.write("Batch size: " + str(batchsize) + "\n")
        stats_file.write("Num epochs: " + str(num_epochs) + "\n")
        stats_file.write("Optimizer: " + str(model_optimizer.__class__()) + "\n")
        stats_file.write("Optimizer config: " + str(model_optimizer.get_config()) + "\n")
        stats_file.write("Recall score: " + str(rec_score) + "\n")
        stats_file.write("F1 score: " + str(f1_score) + "\n")
        stats_file.write("Accuracy score: " + str(acc_score) + "\n")
        stats_file.write("Execution time: " + str(total_time) + " seconds (" + str(total_time/60.0) + " minutes)\n")
        stats_file.write(str(history.history) + "\n")
        stats_file.write("Model config: " + str(model.get_config()) + "\n")

    plot_model(model, os.path.join(stats_dir, 'conv_model_' + str(int(start_time)) + '.png'))

    for loss_vals in history.history.values():
        plt.plot(loss_vals)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(history.history.keys(), loc='best')
    plt.savefig(os.path.join(stats_dir, 'conv_' + str(int(start_time)) + '.png'))

if __name__ == '__main__':
    main()
