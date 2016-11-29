import argparse
from features import FeatureData
#from sklearn import preprocessing, svm, metrics
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import time
import sys

#plt.style.use('ggplot')

feature_list = [
    'Mean',
    'Std',
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
    parser.add_argument('-t', '--training', required=True)
    #parser.add_argument('-s', '--test', required=True)
    args = parser.parse_args()

    training_data_store = FeatureData(args.training)
    

    #training_data_features = training_data_store.get_features()
    #test_data_store = FeatureData(args.test)
    #test_data_features = test_data_store.get_features()

    #X_training = training_data_features.drop(['TYPE', 'ZSPEC', 'Mean', 'u', 'g', 'r', 'i', 'z', 'Eta_e'], axis=1)
    #y_training = training_data_features['TYPE']
    #y_training = y_training.apply(tag_qso)
    
#    import ipdb;ipdb.set_trace()

    #scaler = preprocessing.StandardScaler().fit(X_training)
    #X_training_norm = scaler.transform(X_training)
    
    #classifier = svm.SVC(kernel='linear')
    #classifier = svm.LinearSVC()
    classifier.fit(X_training_norm, y_training)

    coef = pd.Series(classifier.coef_[0], index=X_training.columns)
    scores = coef.abs().sort_values(ascending=False)
    print scores
    
    features_training = pd.DataFrame(X_training_norm, index=X_training.index,  columns=X_training.columns)

    def plot(labels, plane_limits, filename):

        plt.figure()
        
        #features_training.plot(x='Amplitude', y='i', style=['x', 'o'])
        plt.scatter(features_training[y_training == 'QSO'][labels[0]], features_training[y_training == 'QSO'][labels[1]], c='b', label='QSO')
        plt.scatter(features_training[y_training != 'QSO'][labels[0]], features_training[y_training != 'QSO'][labels[1]], c='r', label='NON-QSO')

        #import ipdb;ipdb.set_trace()

        ww = coef[[labels[0], labels[1]]]
        y0 = classifier.intercept_[0]
        aa = -ww[0]/ww[1]
        #xx = np.linspace(features_training['Amplitude'].min(),  features_training['Amplitude'].max())
        xx = np.linspace(plane_limits[0], plane_limits[1])
        yy = aa * xx - y0 / ww[1]

        plt.plot(xx, yy, 'k-', label='SVM hyperplane')

        plt.legend(scatterpoints=1)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        plt.savefig(filename)

        #import ipdb;ipdb.set_trace()

    plot(('Amplitude', 'Std'), (-1,2), 'amplitude_std')
    plot(('Amplitude', 'CAR_tau'), (0,2), 'amplitude_cartau')
    plot(('Amplitude', 'A_mcmc'), (-1,1), 'amplitude_amcmc') 
    plot(('CAR_tau', 'Meanvariance'), (-1,4), 'cartau_meanvariance')
    plot(('PercentDifferenceFluxPercentile', 'Autocor_length'), (1,8), 'percent_carmean')


if __name__=="__main__":
    main()
