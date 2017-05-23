# -*- coding: utf-8 -*-
"""
Implements feature selection for an input csv file with features per object as rows.

For more information in the methods and procedures see: http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html and http://scikit-learn.org/stable/modules/feature_selection.html
"""


import matplotlib
matplotlib.use('Agg')

import argparse
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

#feature_list = [u'RA', u'DEC', u'Mean', u'Std', u'Meanvariance', u'MedianBRP', u'Rcs', u'PeriodLS', u'Period_fit', u'Autocor_length', u'StetsonK', u'Eta_e', u'Amplitude', u'PercentAmplitude', u'LinearTrend', u'Beyond1Std', u'Q31', u'A_mcmc', u'gamma_mcmc', u'p_var', u'ex_var', u'pg_best_period', u'tau_mc', u'sigma_mc', u'wave_coef', u'wave_tau', u'u', u'ERR_u', u'g', u'ERR_g', u'r', u'ERR_r', u'i', u'ERR_i', u'z', u'ERR_z', u'zspec', u'zspec_err', u'g_r', u'u_g', u'r_i', u'i_z', u'diff_exvar']

#feature_list = [u'RA',u'DEC',u'Mean',u'Std',u'Meanvariance',u'MedianBRP',u'Rcs',u'PeriodLS',u'Period_fit',u'Autocor_length',u'StetsonK',u'Eta_e',u'Amplitude',u'PercentAmplitude',u'LinearTrend',u'Beyond1Std',u'Q31',u'A_mcmc',u'gamma_mcmc',u'p_var',u'ex_var',u'pg_best_period',u'pg_peak',u'pg_sig5',u'pg_sig1',u'tau_mc',u'sigma_mc',u'wave_coef',u'wave_tau',u'Q',u'ERR_Q',u'u',u'ERR_u',u'g',u'ERR_g',u'r',u'ERR_r',u'i',u'ERR_i',u'z',u'ERR_z',u'u_g',u'g_r',u'r_i',u'i_z',u'diff_exvar']

#feature_list = [u'RA',u'DEC',u'Mean',u'Std',u'Meanvariance',u'MedianBRP',u'Rcs',u'PeriodLS',u'Period_fit',u'Autocor_length',u'StetsonK',u'Eta_e',u'Amplitude',u'PercentAmplitude',u'LinearTrend',u'Beyond1Std',u'Q31',u'CAR_sigma',u'CAR_mean',u'CAR_tau',u'A_mcmc',u'gamma_mcmc',u'p_var',u'ex_var',u'pg_best_period',u'pg_peak',u'pg_sig5',u'pg_sig1',u'tau_mc',u'sigma_mc',u'wave_coef',u'wave_tau',u'u',u'ERR_u',u'g',u'ERR_g',u'r',u'ERR_r',u'i',u'ERR_i',u'z',u'ERR_z',u'zspec',u'zspec_err',u'g_r',u'u_g',u'r_i',u'i_z',u'diff_exvar']

feature_list = [u'RA', u'DEC', u'Mean', u'Std', u'Meanvariance', u'MedianBRP', u'Rcs', u'PeriodLS', u'Period_fit', u'Autocor_length', u'StetsonK', u'Eta_e', u'Amplitude', u'PercentAmplitude', u'LinearTrend', u'Beyond1Std', u'Q31', u'A_mcmc', u'gamma_mcmc', u'p_var', u'ex_var', u'pg_best_period', u'tau_mc', u'sigma_mc', u'wave_coef', u'wave_tau', u'u', u'ERR_u', u'g', u'ERR_g', u'r', u'ERR_r', u'i', u'ERR_i', u'z', u'ERR_z', u'zspec', u'zspec_err', u'g_r', u'u_g', u'r_i', u'i_z']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('FEATURES', help='CSV file containing a list of features for objects as rows.')
    parser.add_argument('-p', '--plot', help='Plot the results of the crossvalidation process for selection of optimal number of features, in file <PLOT>.png.')
    parser.add_argument('-n', '--num_feat', type=int, help='Print the best <NUM_FEAT> features selected from the input file.')
    args = parser.parse_args()

    input_array = pd.read_csv(args.FEATURES)

    features = input_array[feature_list].astype('float64').dropna(axis=1)
    n_features = features.columns.size
    
    pca = PCA(svd_solver='full')
    #fa = FactorAnalysis()

    pca_scores = list()
    #fa_scores = list()

    #import ipdb;ipdb.set_trace()
    
    n_components = range(0, n_features)
    for n in n_components:
        pca.n_components = n
        #fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, features)))
        #fa_scores.append(np.mean(cross_val_score(fa, features)))

    n_components_pca = n_components[np.argmax(pca_scores)]

    if args.plot:

        n_components_fa = n_components[np.argmax(fa_scores)]

        pca = PCA(svd_solver='full', n_components='mle')
        pca.fit(features)
        n_components_pca_mle = pca.n_components_

        print("Best number of components by PCA CV = %d" % n_components_pca)
        print("Best number of components by FactorAnalysis CV = %d" % n_components_fa)
        print("Best number of components by PCA MLE = %d" % n_components_pca_mle)

        shrinkages = np.logspace(-2, 0, 30)
        cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
        shrunk_cov_score = np.mean(cross_val_score(cv.fit(features).best_estimator_, features))

        lw_score = np.mean(cross_val_score(LedoitWolf(), features))

        plt.figure()
        plt.plot(n_components, pca_scores, 'b', label='PCA scores')
        plt.plot(n_components, fa_scores, 'r', label='FA scores')
        plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
        plt.axvline(n_components_fa, color='r', label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
        plt.axvline(n_components_pca_mle, color = 'k', label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

        plt.axhline(shrunk_cov_score, color='violet', label='Shrunk Covariance MLE', linestyle='-.')
        plt.axhline(lw_score, color='orange', label='LedoitWolf MLE', linestyle='-.')

        plt.xlabel('Number of components')
        plt.ylabel('CV score')
        plt.legend(loc='best')
        plt.title('Component selection results')
        #plt.show()
        plt.savefig(args.plot + ".png", bbox_inches='tight')

    if args.num_feat:

        best_pca = PCA(svd_solver='full', n_components=n_components_pca)
        pca.fit(features)
        components_sorted = sorted(zip(features.columns, np.sum(np.abs(pca.components_), axis=0)), key=lambda x:x[1], reverse=True)
        print "PCA:"
        print components_sorted[:args.num_feat]

        variances = VarianceThreshold().fit(features).variances_
        print "Variance Threshold:"
        print sorted(zip(features.columns, variances), key=lambda x:x[1])

        label_encoder = LabelEncoder().fit(input_array['class'])
        target = label_encoder.transform(input_array['class'])

        select_chi2 = SelectKBest(chi2, k=args.num_feat).fit(np.abs(features), target)
        print "chi2:"
        print features.columns[select_chi2.get_support()]

        select_f_classif = SelectKBest(f_classif, k=args.num_feat).fit(features, target)
        print "f_classif:"
        print features.columns[select_f_classif.get_support()]

        select_mutual_info_classif = SelectKBest(mutual_info_classif, k=args.num_feat).fit(features, target)
        print "mutual_info_classif:"
        print features.columns[select_mutual_info_classif.get_support()]
    

if __name__=="__main__":
    main()

    
