# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import argparse
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt

#feature_list = [u'RA', u'DEC', u'Mean', u'Std', u'Meanvariance', u'MedianBRP', u'Rcs', u'PeriodLS', u'Period_fit', u'Autocor_length', u'StetsonK', u'Eta_e', u'Amplitude', u'PercentAmplitude', u'LinearTrend', u'Beyond1Std', u'Q31', u'A_mcmc', u'gamma_mcmc', u'p_var', u'ex_var', u'pg_best_period', u'tau_mc', u'sigma_mc', u'wave_coef', u'wave_tau', u'u', u'ERR_u', u'g', u'ERR_g', u'r', u'ERR_r', u'i', u'ERR_i', u'z', u'ERR_z', u'zspec', u'zspec_err', u'g_r', u'u_g', u'r_i', u'i_z', u'diff_exvar']

#feature_list = [u'RA',u'DEC',u'Mean',u'Std',u'Meanvariance',u'MedianBRP',u'Rcs',u'PeriodLS',u'Period_fit',u'Autocor_length',u'StetsonK',u'Eta_e',u'Amplitude',u'PercentAmplitude',u'LinearTrend',u'Beyond1Std',u'Q31',u'A_mcmc',u'gamma_mcmc',u'p_var',u'ex_var',u'pg_best_period',u'pg_peak',u'pg_sig5',u'pg_sig1',u'tau_mc',u'sigma_mc',u'wave_coef',u'wave_tau',u'Q',u'ERR_Q',u'u',u'ERR_u',u'g',u'ERR_g',u'r',u'ERR_r',u'i',u'ERR_i',u'z',u'ERR_z',u'u_g',u'g_r',u'r_i',u'i_z',u'diff_exvar']

#feature_list = [u'RA',u'DEC',u'Mean',u'Std',u'Meanvariance',u'MedianBRP',u'Rcs',u'PeriodLS',u'Period_fit',u'Autocor_length',u'StetsonK',u'Eta_e',u'Amplitude',u'PercentAmplitude',u'LinearTrend',u'Beyond1Std',u'Q31',u'CAR_sigma',u'CAR_mean',u'CAR_tau',u'A_mcmc',u'gamma_mcmc',u'p_var',u'ex_var',u'pg_best_period',u'pg_peak',u'pg_sig5',u'pg_sig1',u'tau_mc',u'sigma_mc',u'wave_coef',u'wave_tau',u'u',u'ERR_u',u'g',u'ERR_g',u'r',u'ERR_r',u'i',u'ERR_i',u'z',u'ERR_z',u'zspec',u'zspec_err',u'g_r',u'u_g',u'r_i',u'i_z',u'diff_exvar']
feature_list = [u'RA', u'DEC', u'Mean', u'Std', u'Meanvariance', u'MedianBRP', u'Rcs', u'PeriodLS', u'Period_fit', u'Autocor_length', u'StetsonK', u'Eta_e', u'Amplitude', u'PercentAmplitude', u'LinearTrend', u'Beyond1Std', u'Q31', u'A_mcmc', u'gamma_mcmc', u'p_var', u'ex_var', u'pg_best_period', u'tau_mc', u'sigma_mc', u'wave_coef', u'wave_tau', u'u', u'ERR_u', u'g', u'ERR_g', u'r', u'ERR_r', u'i', u'ERR_i', u'z', u'ERR_z', u'zspec', u'zspec_err', u'g_r', u'u_g', u'r_i', u'i_z']

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('FEATURES')
    args = parser.parse_args()

    features = pd.read_csv(args.FEATURES)[feature_list].astype('float64').dropna(axis=1)
    n_features = features.columns.size
    
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores = list()
    fa_scores = list()

    #import ipdb;ipdb.set_trace()
    
    n_components = range(0, n_features)
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, features)))
        fa_scores.append(np.mean(cross_val_score(fa, features)))

    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(features)
    n_components_pca_mle = pca.n_components_

    print("Best number of components by PCA CV = %d" % n_components_pca)
    print("Best number of components by FactorAnalysis CV = %d" % n_components_fa)
    print("Best number of components by PCA MLE = %d" % n_components_pca_mle)
    
    variances = VarianceThreshold().fit(features).variances_
    print("Best components: %s" % sorted(zip(variances, features), key=lambda x:x[0])[-n_components_fa:])

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
    plt.savefig("PCA_FA.png", bbox_inches='tight')
    

if __name__=="__main__":
    main()

    
