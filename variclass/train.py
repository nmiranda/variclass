# -*- coding: utf-8 -*-
import argparse
import os
import data
import numpy as np
from recurrent import Recurrent
from convolutional import Convolutional
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import recall_score, confusion_matrix
import plot
from keras.utils import plot_model
import pprint

import sys
sys.setrecursionlimit(10000)

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
    batchsize = 32
    num_epochs = args.epochs
    stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')
    validation_ratio = 0.3
    inner_validation = args.inner

    # Loading data
    if args.augment:
        jd_list, q_list, q_err_list, type_list = data.load(directory=args.dir, with_errors=True, sel_longest=args.top)
    elif args.simulate:
        jd_list, q_list, type_list = data.simulate(args.simulate, single_jd='clean_morechip_150.245140_-0.023871_COSMOS.fits')
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

    class_matrix = type_list[..., np.newaxis]

    delta_jd_matrix = jd_matrix[:,1:] - jd_matrix[:,:-1]
    delta_jd_matrix = delta_jd_matrix.clip(min=0)    

    # Defining model
    #model = Recurrent(input_dim=input_dim, max_jd=max_jd)
    model = Convolutional(max_jd=max_jd)

    if inner_validation:

        train_delta_jd = delta_jd_matrix
        train_q = q_matrix
        train_prev_q = train_q[:,:-1]
        train_next_q = train_q[:,1:][..., np.newaxis]
        train_class = class_matrix

        train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

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

            train_q_matrix = augmented_train_q_matrix
            train_class_matrix = augmented_train_class_matrix

        else:

            train_delta_jd, test_delta_jd, train_q_matrix, test_q_matrix, train_class_matrix, test_class_matrix = train_test_split(delta_jd_matrix, q_matrix, class_matrix, test_size=validation_ratio, stratify=class_matrix)

        if len(model.outputs) == 2:

            train_prev_q = train_q_matrix[:,:-1]
            train_next_q = train_q_matrix[:,1:][..., np.newaxis]
            train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

            start_time = time.time()
            start_time_str = time.strftime('%Y%m%dT%H%M%S', time.gmtime(start_time))
        
            history = model.fit(x=[train_X], y=[train_class_matrix, train_next_q], batch_size=batchsize, epochs=num_epochs)

            total_time = time.time() - start_time
        
            test_prev_q = test_q_matrix[:,:-1]
            test_next_q = test_q_matrix[:,1:][..., np.newaxis]
            test_X = np.stack((test_delta_jd, test_prev_q), axis=2)

            test_predict = model.predict(test_X)[0]

        elif len(model.outputs) == 1:

            train_prev_q = train_q_matrix[:,:-1]
            train_X = np.stack((train_delta_jd, train_prev_q), axis=2)

            start_time = time.time()
            start_time_str = time.strftime('%Y%m%dT%H%M%S', time.gmtime(start_time))

            history = model.fit(x=[train_X], y=[train_class_matrix], batch_size=batchsize, epochs=num_epochs)

            total_time = time.time() - start_time

            test_prev_q = test_q_matrix[:,:-1]
            test_X = np.stack((test_delta_jd, test_prev_q), axis=2)
            test_predict = model.predict(test_X)

        #import ipdb;ipdb.set_trace()

        test_predict = np.reshape(test_predict, (test_predict.shape[0]))
        test_predict = 1.0*(test_predict > 0.5)
        rec_score = recall_score(np.reshape(test_class_matrix.astype('int'), test_class_matrix.shape[0]), test_predict.astype('int'))

        print("RECALL_SCORE:", rec_score)

        cnf_matrix = confusion_matrix(test_class_matrix, test_predict)
        plot.confusion_matrix(cnf_matrix, classes=('NON-QSO', 'QSO'), normalize=False, title='Normalized confusion matrix', time_str=start_time_str, model_name=str(model.__class__.__name__))

    model_name = str(model.__class__.__name__)

    with open(os.path.join(stats_dir, model_name + '_' + start_time_str + '.txt'), 'w') as stats_file:

        stats_file.write("Execution time: " + str(total_time) + " seconds (" + str(total_time/60.0) + " minutes)\n")
        stats_file.write("Inner validation: " + str(inner_validation) + "\n")
        stats_file.write("Augment factor: " + str(args.augment) + "\n")
        stats_file.write("Simulate samples: " + str(args.simulate) + "\n")
        stats_file.write("Number of samples: " + str(num_samples) + "\n")
        stats_file.write("Batch size: " + str(batchsize) + "\n")
        stats_file.write("Num epochs: " + str(num_epochs) + "\n")
        stats_file.write("Recall score: " + str(rec_score) + "\n")
        stats_file.write("Select longest series: " + str(args.top) + "\n")
        stats_file.write(model.config_str())
        #stats_file.write("Model config: " + str(model.get_config()) + "\n")
        stats_file.write("Model config: " + pprint.pformat(model.get_config()) + "\n")
        stats_file.write("Confusion matrix: " + repr(cnf_matrix) + "\n")
        stats_file.write(str(history.history) + "\n")
        

    plot_model(model, os.path.join(stats_dir, model_name + '_model_' + start_time_str + '.png'))

if __name__ == '__main__':
    main()