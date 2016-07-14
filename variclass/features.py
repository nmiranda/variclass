# -*- coding: utf-8 -*-

import os
import pyfits
#from SF import fitSF_mcmc, Pvar
#import FATS
#import numpy as np
import pandas as pd
import glob

# Lista de features a calcular
feature_list = [
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
    'Con',
    'LinearTrend',
    'Beyond1Std',
    'FluxPercentileRatioMid20',
    'FluxPercentileRatioMid35',
    'FluxPercentileRatioMid50',
    'FluxPercentileRatioMid65',
    'FluxPercentileRatioMid80',
    'PercentDifferenceFluxPercentile',
    'Q31',
    'CAR_sigma',
    'CAR_mean',
    'CAR_tau',
]

class Features(object):
    
    def __init__(self):
        
        self.u = None
        self.g = None
        self.r = None
        self.i = None
        self.z = None

class LightCurve(object):

    def __init__(self, date, mag, mag_err):
        
        mag_series = pd.Series(mag, index=date)
        mag_err_series = pd.Series(mag_err, index=date)
        data_dict = {'mag': mag_series, 'mag_err': mag_err_series}
        
        self.series = pd.DataFrame(data_dict)
        self.features = Features()
        self.ra = None
        self.dec = None
        self.obj_type = None
        self.zspec = None

def load_from_fits(fits_file):

    this_fits = pyfits.open(fits_file)
    fits_data = this_fits[1].data
    fits_header = this_fits[0].header

    this_lc = LightCurve(fits_data['JD'], fits_data['Q'], fits_data['errQ'])
    this_lc.ra = fits_header['ALPHA']
    this_lc.dec = fits_header['DELTA']
    this_lc.features.u = fits_header['U']
    this_lc.features.g = fits_header['G']
    this_lc.features.r = fits_header['R']
    this_lc.features.i = fits_header['I']
    this_lc.features.z = fits_header['Z']
    this_lc.obj_type = fits_header['TYPE']
    this_lc.zspec = fits_header['ZSPEC']

def curves_from_dir(folder):
    
    for fits_file in glob.glob(os.path.join(folder, '*.fits')):
        yield load_from_fits(fits_file)

def run_fats(dir_path, filename, x_var, y_var, y_var_err):

    # Variables para indexar y contener features
    features_keys = feature_list
    num_features = len(features_keys)
    key_index = {key: idx for idx, key in enumerate(features_keys)}
    fits_features = np.zeros(num_features, dtype=np.float64)
    
    # Función auxiliar para setear valor del arreglo de features
    def set_array_val(arr, key, val):
        arr[key_index[key]] = val

    # Abrir archivo FITS
    this_fits = pyfits.open(os.path.join(dir_path, filename))

    # Extraer datos de curva de luz
    fits_data = this_fits[1].data
    data_array = np.array([fits_data[y_var], fits_data[x_var], fits_data[y_var_err]])
    data_ids = ['magnitude', 'time', 'error']

    # Correr algoritmos MCMC del script SF.py
    mcmc_vals = fitSF_mcmc(fits_data['JD'], fits_data['Q'], fits_data['errQ'], 2, 250, 500)
    A_mcmc = mcmc_vals[0][0]
    gamma_mcmc = mcmc_vals[1][0]
    set_array_val(fits_features, 'A', A_mcmc)
    set_array_val(fits_features, 'gamma', gamma_mcmc)

    this_pvar = Pvar(fits_data[x_var], fits_data[y_var], fits_data[y_var_err])
    set_array_val(fits_features, 'P', this_pvar)

    # Calcular y guardar features de FATS
    feat_space = FATS.FeatureSpace(featureList=feature_list, Data=data_ids)
    feat_vals = feat_space.calculateFeature(data_array)
    feat_dict = {feat_key: feat_val for feat_key, feat_val in zip(feat_vals.featureList, feat_vals.result())}
        
    for feat_key, feat_val in feat_dict.iteritems():
            set_array_val(fits_features, feat_key, feat_val)

    return fits_features

