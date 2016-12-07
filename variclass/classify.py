import argparse
from features import FeatureData
from sklearn import preprocessing, svm, model_selection
import numpy as np
import pandas as pd
import time
import sys
import pickle

#plt.style.use('ggplot')

feature_list = [
    #'ZSPEC',
    'Mean',
    'Std',
    'Meanvariance',
    'MedianBRP',
    'Rcs',
    'PeriodLS',
    'Period_fit',
    #'Color',
    'Autocor_length',
    #'SlottedA_length',
    'StetsonK',
    #'StetsonK_AC',
    'Eta_e',
    'Amplitude',
    'PercentAmplitude',
    #'Con',
    'LinearTrend',
    'Beyond1Std',
    #'FluxPercentileRatioMid20',
    #'FluxPercentileRatioMid35',
    #'FluxPercentileRatioMid50',
    #'FluxPercentileRatioMid65',
    #'FluxPercentileRatioMid80',
    #'PercentDifferenceFluxPercentile',
    'Q31',
    #'CAR_sigma',
    #'CAR_mean',
    #'CAR_tau',
    'A_mcmc',
    #'A_mcmc_err_inf',
    #'A_mcmc_err_sup',
    'gamma_mcmc',
    #'gamma_mcmc_err_inf',
    #'gamma_mcmc_err_sup',
    'p_var',
    'ex_var',
    #'ex_verr',
    #'wmcc_bestperiod',
    #'wmcc_bestfreq',
    'pg_best_period',
    #'pg_peak',
    #'pg_sig5',
    #'pg_sig1',
    'tau_mc',
    #'tau_mc_inf_err',
    #'tau_mc_sup_err',
    'sigma_mc',
    #'sigma_mc_inf_err',
    #'sigma_mc_sup_err',
    'wave_coef',
    'wave_tau',
    'u',
    #'ERR_u',
    'g',
    #'ERR_g',
    'r',
    #'ERR_r',
    'i',
    #'ERR_i',
    'z',
    #'ERR_z',
    #'zspec',
    #'zspec_err',
    #'class',
    #'subClass',
    'u_g',
    'g_r',
    'r_i',
    'i_z',
    'diff_exvar',
    ]

def tag_qso(label):
    if label != 'QSO':
        return 'NON-QSO'
    return label

def tag_qso_bin(label):
    if label == 'QSO':
        return 1
    return 0

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training', help='CSV file with training data vectors as rows and features as columns')
    parser.add_argument('-c', '--classtag', default='class', help='Tag or name of column that defines the classes of the respective vectors in training data file (default: "class")')
    parser.add_argument('-s', '--test', help='CSV file with data to classify. Vectors as rows and features as columns.')
    parser.add_argument('-o', '--output', help='Output CSV file to write the result of the classification.')
    parser.add_argument('-n', '--ncores', type=int, default=1, help='Number of cores or jobs in parallel to run')
    parser.add_argument('-d', '--dump', help='File to which save or dump the trained model')
    parser.add_argument('-l', '--load', help='File from which load a trained model')
    parser.add_argument('-a', '--stats', help='File to write classification performance statistics')
    args = parser.parse_args()


    if not args.load:
        if not args.training:
            parser.error("option '-t'/'--training' is required when not loading a trained model.")

        print "Reading training data in \"%s\"" % args.training
        training_data = pd.read_csv(args.training)

        print "Pre-processing training data"
        training_X = training_data[feature_list].astype('float64')

        label_encoder = preprocessing.LabelEncoder()
        training_Y = training_data[args.classtag].apply(tag_qso)
        label_encoder.fit(training_Y)
        training_Y = label_encoder.transform(training_Y)

        try:
            scaler = preprocessing.StandardScaler().fit(training_X)
        except ValueError:
            column_is_invalid = training_X.applymap(lambda x: x==np.inf).any()
            invalid_columns = column_is_invalid[column_is_invalid].index.tolist()
            raise ValueError("Column(s) %s has(have) invalid values. Please exclude from feature list or remove respective rows." % invalid_columns)
        training_X = scaler.transform(training_X)

        param_grid = {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

        model = svm.SVC(class_weight='balanced', probability=True)
        print "Training classifier \"%s\"" % model

        inner_cv = model_selection.KFold(shuffle=True)
        outer_cv = model_selection.KFold(shuffle=True)

        modeselektor = model_selection.GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, n_jobs=args.ncores)
        modeselektor.fit(training_X, training_Y)
        f1_score = model_selection.cross_val_score(modeselektor, X=training_X, y=training_Y, cv=outer_cv, n_jobs=args.ncores, scoring='f1').mean()
        accuracy_score = model_selection.cross_val_score(modeselektor, X=training_X, y=training_Y, cv=outer_cv, n_jobs=args.ncores, scoring='accuracy').mean()
        neg_log_loss_score = model_selection.cross_val_score(modeselektor, X=training_X, y=training_Y, cv=outer_cv, n_jobs=args.ncores, scoring='neg_log_loss').mean()
        precision_score = model_selection.cross_val_score(modeselektor, X=training_X, y=training_Y, cv=outer_cv, n_jobs=args.ncores, scoring='precision').mean()
        recall_score = model_selection.cross_val_score(modeselektor, X=training_X, y=training_Y, cv=outer_cv, n_jobs=args.ncores, scoring='recall').mean()
        roc_auc_score = model_selection.cross_val_score(modeselektor, X=training_X, y=training_Y, cv=outer_cv, n_jobs=args.ncores, scoring='roc_auc').mean()
        scores = {
            'f1': f1_score,
            'accuracy': accuracy_score,
            'neg_log_loss': neg_log_loss_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'roc_auc_score': roc_auc_score,
            }

        if args.dump:
            print "Dumping trained model in file \"%s\"" % args.dump
            with open(args.dump, 'wb') as pickle_file:
                model_dump = {
                    'modeselektor': modeselektor,
                    'scaler': scaler,
                    'scores': scores,
                    'label_encoder': label_encoder,
                    }
                pickle.dump(model_dump, pickle_file)
        
    else:
        print "Loading trained model in file \"%s\"." % args.load
        with open(args.load, 'rb') as pickle_file:
            model_dump = pickle.load(pickle_file)
            modeselektor = model_dump['modeselektor']
            scaler = model_dump['scaler']
            scores = model_dump['scores']
            label_encoder = model_dump['label_encoder']


    if args.stats:
        with open(args.stats, 'wb') as stats_file:
            for key, value in scores.items():
                stats_file.write("Classification score \"%s\": %f\n" % (key, value))
    else:
        for key, value in scores.items():
            print "Classification score \"%s\": %f" % (key, value)


    if args.test:
        if not args.output:
            parser.error("option '-o'/'--output' is required when using test data.")
        
        print "Reading test data in \"%s\"" % args.test
        test_data = pd.read_csv(args.test)

        print "Pre-processing test data"
        test_X = test_data[feature_list].astype('float64')
        test_X = scaler.transform(test_X)

        print "Classifying test data"
        test_Y = label_encoder.inverse_transform(modeselektor.predict(test_X))
        test_Y_proba = np.amax(modeselektor.predict_proba(test_X), axis=1)

        print "Writing results to \"%s\"" % args.output
        test_data['predicted_class'] = test_Y
        test_data['predicted_class_proba'] = test_Y_proba
        test_data.to_csv(args.output)
        
if __name__=="__main__":
    main()
