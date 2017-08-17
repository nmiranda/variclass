# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Lambda, Input, Dropout
from keras.losses import mean_squared_error, binary_crossentropy
from keras.layers.wrappers import TimeDistributed
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

def scale_dataset(X, mag_range=(0,1)):

    new_X = np.copy(X)

    """
    for i in range(new_X.shape[2]):
    
        this_dataset = new_X[:,:,i]

        data_min = np.amin(this_dataset, axis=None)
        data_max = np.amax(this_dataset, axis=None)

        data_range = data_max - data_min
        this_scale = (mag_range[1] - mag_range[0]) / data_range
        this_min = mag_range[0] - data_min * this_scale

        this_dataset *= this_scale
        this_dataset += this_min
    
    return new_X
    """

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
    args = parser.parse_args()

    #input_dim = 3 # <----  t, dt, q(t-dt)
    input_dim = 2 # <----  dt, q(t-dt)
    input_dtype = 'float64'
    #lstm_memory = 32
    lstm_memory = 64
    #dropout_rate = 0.5
    dropout_rate = 0.1
    lambda_loss=5
    #lambda_loss=0.0
    batchsize = 32
    num_epochs = 1

    if args.dir:
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
            #jd_list.append(this_jd[1:])
            #q_list.append(this_q[:-1])
            #jd_delta_list.append(this_jd[1:] - this_jd[:-1])
            #q_pred_list.append(this_q[1:])

            jd_list.append(this_jd)
            q_list.append(this_q)
            #jd_delta_list.append(this_jd[1:] - this_jd[:-1])
            #_pred_list.append(this_q[1:])

            this_fits.close()

        np.savez('jd_list', *jd_list)
        np.savez('q_list', *q_list)
        np.savez('type_list', type_list)
        exit()
    
    else:

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
    num_samples = len(jd_list)

    """
    data_X = np.full((num_samples, max_jd, input_dim), 0., dtype=input_dtype)
    for i in xrange(num_samples):
        #data_X[i,:jd_list[i].shape[0],0] = jd_list[i]
        #data_X[i,:jd_delta_list[i].shape[0],1] = jd_delta_list[i]
        #data_X[i,:q_list[i].shape[0],2] = q_list[i]
        data_X[i,:jd_list[i].shape[0],0] = jd_list
        data_X[i,:jd_list[i].shape[0],0] = jd_list
    """

    jd_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    for i, jd_array in enumerate(jd_list):
        jd_matrix[i,:jd_array.shape[0]] = jd_array

    q_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    for i, q_array in enumerate(q_list):
        q_matrix[i,:q_array.shape[0]] = q_array

    """
    class_real = np.asarray(type_list, dtype=input_dtype)[..., np.newaxis, np.newaxis]
    next_real = np.full((num_samples, max_jd, 1), 0., dtype=input_dtype)
    for i in xrange(num_samples):
        next_real[i,:q_pred_list[i].shape[0],0] = q_pred_list[i]

    """

    #class_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    #class_matrix[:,-1] = type_list

    class_matrix = type_list[..., np.newaxis, np.newaxis]

    delta_jd_matrix = jd_matrix[:,1:] - jd_matrix[:,:-1]
    delta_jd_matrix = delta_jd_matrix.clip(min=0)

    #prev_q_matrix = q_matrix[:,:-1]

    #next_q_matrix = q_matrix[:,1:]




    stratified_k_fold = StratifiedKFold(n_splits=3, shuffle=True)
    cv_scores = list()

    for train_index, test_index in stratified_k_fold.split(jd_list, type_list):

        ### DEFINING MODEL
        LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda shape: (shape[0], 1, shape[2]))
        #ClassIndicator = Lambda(lambda x: 1.0*(x > 0.5), output_shape=lambda shape: (shape[0], 1, 1))

        #_input = Input(shape=(max_jd, input_dim))
        _input = Input(shape=(max_jd-1, input_dim))
        #lstm = LSTM(lstm_memory, return_sequences=True, activation='relu')(_input)
        lstm = LSTM(lstm_memory, return_sequences=True, activation='tanh')(_input)
        #dropout = Dropout(dropout_rate)(lstm)
        #time_distributed = TimeDistributed(Dense(1), input_shape=(max_jd, lstm_memory))(dropout)
        #last_output = LastOutput(dropout)
        time_distributed = TimeDistributed(Dense(1), input_shape=(max_jd, lstm_memory))(lstm)
        last_output = LastOutput(lstm)
        dense = Dense(1, activation="sigmoid")(last_output)
        #class_indicator = ClassIndicator(dense)

        model = Model(inputs=[_input], outputs=[dense, time_distributed])
        #model = Model(inputs=[_input], outputs=[class_indicator, time_distributed])
        #plot_model(model, "model.png")

        model.compile(optimizer='adam', loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, lambda_loss])

        train_delta_jd = scale_dataset(delta_jd_matrix[train_index])
        train_q = scale_dataset(q_matrix[train_index])
        train_prev_q = train_q[:,:-1]
        train_next_q = train_q[:,1:][..., np.newaxis]
        train_class = class_matrix[train_index]

        train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

        #import ipdb;ipdb.set_trace()

        #train_X = scale_dataset(data_X[train_index])

        #train_class = np.copy(class_real[train_index])
        #train_next = scale_dataset(np.copy(next_real[train_index]))
        
        #model.fit(x=[train_X], y=[train_class, train_next], batch_size=batchsize, epochs=num_epochs)
        model.fit(x=[train_X], y=[train_class, train_next_q], batch_size=batchsize, epochs=num_epochs)

        #test_predict = model.predict(scale_dataset(data_X[test_index]))[0]
        test_predict = model.predict(scale_dataset(train_X))[0]
        test_predict = np.reshape(test_predict, (test_predict.shape[0]))

        test_predict = 1.0*(test_predict > 0.5)

        #rec_score = recall_score(np.reshape(class_real[test_index].astype('int'), class_real[test_index].shape[0]), test_predict.astype('int'))
        rec_score = recall_score(np.reshape(class_matrix[train_index].astype('int'), class_matrix[train_index].shape[0]), test_predict.astype('int'))

        print("RECALL_SCORE:", rec_score)

        cv_scores.append(rec_score)

    print(cv_scores)


if __name__ == '__main__':
    main()
