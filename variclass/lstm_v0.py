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
    args = parser.parse_args()

    # Model parameters
    input_dim = 2 # <----  dt, q(t-dt)
    input_dtype = 'float64'
    lstm_memory = 64
    lambda_loss=5
    batchsize = 32
    num_epochs = 10

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
    num_samples = len(jd_list)

    # Initializing matrixes
    jd_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    for i, jd_array in enumerate(jd_list):
        jd_matrix[i,:jd_array.shape[0]] = jd_array

    q_matrix = np.full((num_samples, max_jd), 0., dtype=input_dtype)
    for i, q_array in enumerate(q_list):
        q_matrix[i,:q_array.shape[0]] = q_array

    class_matrix = type_list[..., np.newaxis, np.newaxis]

    delta_jd_matrix = jd_matrix[:,1:] - jd_matrix[:,:-1]
    delta_jd_matrix = delta_jd_matrix.clip(min=0)

    ### DEFINING MODEL
    LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda shape: (shape[0], 1, shape[2]))

    _input = Input(shape=(max_jd-1, input_dim))
    lstm = LSTM(lstm_memory, return_sequences=True, activation='tanh')(_input)
    time_distributed = TimeDistributed(Dense(1), input_shape=(max_jd, lstm_memory))(lstm)
    last_output = LastOutput(lstm)
    dense = Dense(1, activation="sigmoid")(last_output)

    model = Model(inputs=[_input], outputs=[dense, time_distributed])

    model.compile(optimizer='adam', loss=["binary_crossentropy", "mean_squared_error"], loss_weights=[1.0, lambda_loss])

    # Scaling dataset
    train_delta_jd = scale_dataset(delta_jd_matrix)
    train_q = scale_dataset(q_matrix)
    train_prev_q = train_q[:,:-1]
    train_next_q = train_q[:,1:][..., np.newaxis]
    train_class = class_matrix

    train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

    model.fit(x=[train_X], y=[train_class, train_next_q], batch_size=batchsize, epochs=num_epochs)

    # Evaluating model in the same training dataset (just for testing purposes)
    test_predict = model.predict(scale_dataset(train_X))[0]

    test_predict = np.reshape(test_predict, (test_predict.shape[0]))
    test_predict = 1.0*(test_predict > 0.5)
    rec_score = recall_score(np.reshape(class_matrix.astype('int'), class_matrix.shape[0]), test_predict.astype('int'))

    print("RECALL_SCORE:", rec_score)


if __name__ == '__main__':
    main()
