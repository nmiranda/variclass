    # -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Dropout, Flatten
from keras.losses import mean_squared_error, binary_crossentropy
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import itertools

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir")
    parser.add_argument("-i", "--inner", action='store_true')
    args = parser.parse_args()

    # Model parameters
    input_dim = 2 # <----  dt, q(t-dt)
    input_dtype = 'float64'
    lstm_memory = 64
    lstm_activation = 'tanh'
    dense_activation = "sigmoid"
    lambda_loss=0.05
    batchsize = 32
    num_epochs = 25
    stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')
    validation_ratio = 0.3
    inner_validation = args.inner

    if args.dir:
        # Reading from a directory of .fits files
        type_list = list()
        jd_list = list()
        q_list = list()
        jd_delta_list = list()
        q_pred_list = list()

        fits_files = glob.glob(os.path.join(args.dir, '*.fits'))
        for fits_file in fits_files:
            this_fits = pyfits.open(fits_file, memmap=False)
            try:
                this_data = this_fits[1].data
            except TypeError:
                continue
            this_header = this_fits[0].header

            this_jd = this_data['JD']
            this_q = this_data['Q']

            if this_header['TYPE_SPEC'].strip() == 'QSO':
                type_list.append(1)
            else:
                type_list.append(0)

            jd_list.append(this_jd)
            q_list.append(this_q)

            this_fits.close()

        np.savez('jd_list', *jd_list)
        np.savez('q_list', *q_list)
        np.savez('type_list', type_list)
        exit()
    
    else:
        # Reading from saved dataset files
        jd_list = []
        with np.load('jd_list.npz') as jd_npzfile:
            for _, jd_array in jd_npzfile.iteritems():
                jd_list.append(jd_array)

        q_list = []
        with np.load('q_list.npz') as q_npzfile:
            for _, q_array in q_npzfile.iteritems():
                q_list.append(q_array)

        with np.load('type_list.npz') as type_npzfile:
            type_list = type_npzfile.items()[0][1]

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

    #class_matrix = type_list[..., np.newaxis, np.newaxis]
    class_matrix = type_list[..., np.newaxis]

    delta_jd_matrix = jd_matrix[:,1:] - jd_matrix[:,:-1]
    delta_jd_matrix = delta_jd_matrix.clip(min=0)

    ### DEFINING MODEL
    #LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda shape: (shape[0], 1, shape[2]))
    LastOutput = Lambda(lambda x: x[:, median_jd-1:median_jd, :], output_shape=lambda shape: (shape[0], 1, shape[2]))

    SelectPredict = Lambda(lambda x: x[:, :, :lstm_memory/2], output_shape=lambda shape: (shape[0], shape[1], lstm_memory/2))
    SelectClassify = Lambda(lambda x: x[:, :, lstm_memory/2:], output_shape=lambda shape: (shape[0], shape[1], lstm_memory/2))

    _input = Input(shape=(max_jd-1, input_dim))
    lstm_1 = LSTM(lstm_memory, return_sequences=True, activation=lstm_activation)(_input)
    select_predict = SelectPredict(lstm_1)
    time_distributed = TimeDistributed(Dense(1), input_shape=(max_jd, lstm_memory))(select_predict)
    select_classify = SelectClassify(lstm_1)
    flatten = Flatten()(select_classify)
    dense = Dense(1, activation=dense_activation)(flatten)

    model = Model(inputs=[_input], outputs=[dense, time_distributed])

    #layer_name = 'lstm_1'
    #intermediate_layer_model = Model(inputs=[_input], outputs=lstm_1)

    model_optimizer = Adam()
    #model_optimizer = SGD(lr=0.05)

    model.compile(optimizer=model_optimizer, loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, lambda_loss], metrics=['accuracy'])

    if inner_validation:

        train_delta_jd = delta_jd_matrix
        train_q = q_matrix
        train_prev_q = train_q[:,:-1]
        train_next_q = train_q[:,1:][..., np.newaxis]
        train_class = class_matrix

        train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

        # model_optimizer = Adam()
        # intermediate_layer_model.compile(optimizer=model_optimizer, loss=["mean_squared_error"]*3)
        # import ipdb;ipdb.set_trace()
        # history = intermediate_layer_model.fit(x=[train_X], y=[train_next_q], batch_size=batchsize, epochs=1)
        # intermediate_layer_output = intermediate_layer_model.predict(train_X)
        # import ipdb;ipdb.set_trace()

        start_time = time.time()
        start_time_str = time.strftime('%Y%m%dT%H%M%S', time.gmtime(start_time))
        history = model.fit(x=[train_X], y=[train_class, train_next_q], batch_size=batchsize, epochs=num_epochs, validation_split=validation_ratio)
        total_time = time.time() - start_time

        # Evaluating model in the same training dataset (just for testing purposes)
        test_dataset = train_X

        #intermediate_output = intermediate_layer_model.predict(test_dataset)
        #import ipdb; ipdb.set_trace()

        test_predict = model.predict(test_dataset)[0]

        test_predict = np.reshape(test_predict, (test_predict.shape[0]))
        test_predict = 1.0*(test_predict > 0.5)
        rec_score = recall_score(np.reshape(class_matrix.astype('int'), class_matrix.shape[0]), test_predict.astype('int'))

        print("RECALL_SCORE:", rec_score)

        for loss_vals in history.history.values():
            plt.plot(loss_vals)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(history.history.keys(), loc='best')
        plt.savefig(os.path.join(stats_dir, 'lstm_' + start_time_str + '.png'))

    else:
        train_delta_jd, test_delta_jd, train_q_matrix, test_q_matrix, train_class_matrix, test_class_matrix = train_test_split(delta_jd_matrix, q_matrix, class_matrix, test_size=validation_ratio)

        train_prev_q = train_q_matrix[:,:-1]
        train_next_q = train_q_matrix[:,1:][..., np.newaxis]
        train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

        start_time = time.time()
        start_time_str = time.strftime('%Y%m%dT%H%M%S', time.gmtime(start_time))
        history = model.fit(x=[train_X], y=[train_class_matrix, train_next_q], batch_size=batchsize, epochs=num_epochs)
        total_time = time.time() - start_time
        
        test_prev_q = test_q_matrix[:,:-1]
        test_next_q = test_q_matrix[:,1:][..., np.newaxis]
        test_dataset = np.stack((test_delta_jd, test_prev_q), axis=2)

        test_predict = model.predict(test_dataset)[0]

        test_predict = np.reshape(test_predict, (test_predict.shape[0]))
        test_predict = 1.0*(test_predict > 0.5)
        rec_score = recall_score(np.reshape(test_class_matrix.astype('int'), test_class_matrix.shape[0]), test_predict.astype('int'))

        print("RECALL_SCORE:", rec_score)

        cnf_matrix = confusion_matrix(test_class_matrix, test_predict)

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=('NON-QSO', 'QSO'), normalize=True, title='Normalized confusion matrix')
        plt.savefig(os.path.join(stats_dir, 'lstm_conf_matrix_' + start_time_str + '.png'))


    with open(os.path.join(stats_dir, 'lstm_' + start_time_str + '.txt'), 'w') as stats_file:

        stats_file.write("Inner validation: " + str(inner_validation) + "\n")
        stats_file.write("Lstm memory: " + str(lstm_memory) + "\n")
        stats_file.write("Lstm activation: " + str(lstm_activation) + "\n")
        stats_file.write("Dense activation: " + str(dense_activation) + "\n")
        stats_file.write("Lambda loss: " + str(lambda_loss) + "\n")
        stats_file.write("Batch size: " + str(batchsize) + "\n")
        stats_file.write("Num epochs: " + str(num_epochs) + "\n")
        stats_file.write("Optimizer: " + str(model_optimizer.__class__()) + "\n")
        stats_file.write("Optimizer config: " + str(model_optimizer.get_config()) + "\n")
        stats_file.write("Recall score: " + str(rec_score) + "\n")
        stats_file.write("Execution time: " + str(total_time) + " seconds (" + str(total_time/60.0) + " minutes)\n")
        stats_file.write(str(history.history) + "\n")
        stats_file.write("Model config: " + str(model.get_config()) + "\n")

    plot_model(model, os.path.join(stats_dir, 'lstm_model_' + start_time_str + '.png'))

if __name__ == '__main__':
    main()
