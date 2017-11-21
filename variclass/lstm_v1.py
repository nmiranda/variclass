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
import data
import plot



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
    lstm_memory = 64
    lstm_activation = 'tanh'
    dense_activation = "sigmoid"
    lambda_loss=0.05
    batchsize = 32
    num_epochs = args.epochs
    stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')
    validation_ratio = 0.3
    inner_validation = args.inner

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

    #class_matrix = type_list[..., np.newaxis, np.newaxis]
    class_matrix = type_list[..., np.newaxis]

    delta_jd_matrix = jd_matrix[:,1:] - jd_matrix[:,:-1]
    delta_jd_matrix = delta_jd_matrix.clip(min=0)

    ### DEFINING MODEL
    #LastOutput = Lambda(lambda x: x[:, -1:, :], output_shape=lambda shape: (shape[0], 1, shape[2]))
    #LastOutput = Lambda(lambda x: x[:, median_jd-1:median_jd, :], output_shape=lambda shape: (shape[0], 1, shape[2]))

    SelectPredict = Lambda(lambda x: x[:, :, :lstm_memory/2], output_shape=lambda shape: (shape[0], shape[1], lstm_memory/2))
    SelectClassify = Lambda(lambda x: x[:, :, lstm_memory/2:], output_shape=lambda shape: (shape[0], shape[1], lstm_memory/2))

    _input = Input(shape=(max_jd-1, input_dim))
    lstm_1 = LSTM(lstm_memory, return_sequences=True, activation=lstm_activation)(_input)
    lstm_2 = LSTM(lstm_memory, return_sequences=True, activation=lstm_activation)(lstm_1)
    select_predict = SelectPredict(lstm_2)
    time_distributed = TimeDistributed(Dense(1), input_shape=(max_jd, lstm_memory))(select_predict)
    select_classify = SelectClassify(lstm_2)
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

        plot.learning_process(history, start_time_str)

    else:

        if args.augment:

            augment_data_factor = args.augment

            train_jd_matrix, test_jd_matrix, train_q_matrix, test_q_matrix, train_q_err_matrix, test_q_err_matrix, train_class_matrix, test_class_matrix = train_test_split(jd_matrix, q_matrix, q_err_matrix, class_matrix, test_size=validation_ratio)

            augmented_train_q_matrix = np.full((train_q_matrix.shape[0] * augment_data_factor, train_q_matrix.shape[1]), 0., dtype=input_dtype)
            augmented_train_jd_matrix = np.full((train_jd_matrix.shape[0] * augment_data_factor, train_jd_matrix.shape[1]), 0., dtype=input_dtype)
            augmented_train_class_matrix = np.full((train_q_matrix.shape[0] * augment_data_factor, 1), 0)
            for num_augment_cycle in range(augment_data_factor):
                for i in xrange(train_q_matrix.shape[0]):
                    for j in xrange(train_q_matrix.shape[1]):
                        augmented_train_q_matrix[(num_augment_cycle * train_q_matrix.shape[0] + i), j] = np.random.normal(train_q_matrix[i,j], train_q_err_matrix[i,j])
                        augmented_train_jd_matrix[(num_augment_cycle * train_q_matrix.shape[0] + i), j] = train_jd_matrix[i,j]
                    augmented_train_class_matrix[(num_augment_cycle * train_q_matrix.shape[0] + i), 0] = train_class_matrix[i,0]

            train_delta_jd = augmented_train_jd_matrix[:,1:] - augmented_train_jd_matrix[:,:-1]
            train_delta_jd = train_delta_jd.clip(min=0)

            test_delta_jd = test_jd_matrix[:,1:] - test_jd_matrix[:,:-1]
            test_delta_jd = test_delta_jd.clip(min=0)

            # train_X = np.stack((augmented_train_delta_jd, augmented_train_q_matrix), axis=2)
            # train_class = augmented_train_class_matrix
            # test_X = np.stack((test_delta_jd, test_q_matrix), axis=2)
            # test_class = test_class_matrix

            train_q_matrix = augmented_train_q_matrix
            train_class_matrix = augmented_train_class_matrix

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
        plot.confusion_matrix(cnf_matrix, classes=('NON-QSO', 'QSO'), normalize=False, title='Normalized confusion matrix', time_str=start_time_str)

    with open(os.path.join(stats_dir, 'lstm_' + start_time_str + '.txt'), 'w') as stats_file:

        stats_file.write("Inner validation: " + str(inner_validation) + "\n")
        stats_file.write("Augment factor: " + str(args.augment) + "\n")
        stats_file.write("Simulate samples: " + str(args.simulate) + "\n")
        stats_file.write("Lstm memory: " + str(lstm_memory) + "\n")
        stats_file.write("Lstm activation: " + str(lstm_activation) + "\n")
        stats_file.write("Dense activation: " + str(dense_activation) + "\n")
        stats_file.write("Lambda loss: " + str(lambda_loss) + "\n")
        stats_file.write("Number of samples: " + str(num_samples) + "\n")
        stats_file.write("Batch size: " + str(batchsize) + "\n")
        stats_file.write("Num epochs: " + str(num_epochs) + "\n")
        stats_file.write("Optimizer: " + str(model_optimizer.__class__()) + "\n")
        stats_file.write("Optimizer config: " + str(model_optimizer.get_config()) + "\n")
        stats_file.write("Recall score: " + str(rec_score) + "\n")
        stats_file.write("Execution time: " + str(total_time) + " seconds (" + str(total_time/60.0) + " minutes)\n")
        stats_file.write("Select longest series: " + str(args.top) + "\n")
        stats_file.write("Confusion matrix: " + str(cnf_matrix) + "\n")
        stats_file.write(str(history.history) + "\n")
        stats_file.write("Model config: " + str(model.get_config()) + "\n")

    plot_model(model, os.path.join(stats_dir, 'lstm_model_' + start_time_str + '.png'))

if __name__ == '__main__':
    main()
