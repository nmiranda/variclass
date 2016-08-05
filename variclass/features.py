# -*- coding: utf-8 -*-

import os
import pyfits
from SF import fitSF_mcmc, Pvar
import FATS
import numpy as np
import pandas as pd
import glob
import argparse
import inspect
import P4J

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

class FeatureMethod(object):
    def __init__(self):
        pass

class FATSMethod(FeatureMethod):

    def __init__(self, selected_features=[], data_ids=['magnitude', 'time', 'error']):
        self.supported_features = list()
        for name, _ in inspect.getmembers(FATS.FeatureFunctionLib, inspect.isclass):
            self.supported_features.append(name)
        self.features = list()
        for feature in selected_features:
            if feature in self.supported_features:
                self.features.append(feature)
        self.feat_space = FATS.FeatureSpace(featureList=self.features, Data=data_ids)

    def calculate_features(self, light_curve):
        return self.feat_space.calculateFeature(light_curve.as_array())

class MCMCMethod(FeatureMethod):

    supported_features = [
            'A_mcmc',
            'gamma_mcmc',
            'pvar',
        ]
    
    def __init__(self, selected_features=[]):
        self.features = list()
        for feature in selected_features:
            if feature in supported_features:
                self.features.append(feature)

    def calculate_features(self, light_curve):
        return_vals = dict()
        mcmc_vals = fitSF_mcmc(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err(), 2, 24, 50)
        if 'A_mcmc' in self.features:
            return_vals['A_mcmc'] = mcmc_vals[0][0]
        if 'gamma_mcmc' in self.features:
            return_vals['gamma_mcmc'] = mcmc_vals[1][0]
        if 'pvar' in self.features:
            this_pvar = Pvar(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err())
            return_vals['pvar'] = this_pvar
        return return_vals

class P4JMethod(FeatureMethod):

    supported_features = ['wmcc_bestperiod', 'wmcc_bestfreq']
    
    def __init__(self, selected_features=[]):
        self.features = list()
        for feature in selected_features:
            if feature in supported_features:
                self.features.append()

    def calculate_features(self, light_curve):
        return_vals = dict()
        my_per = P4J.periodogram(M=1,  method="WMCC")
        my_per.fit(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err())
        my_per.grid_search(fmin=0.0, fmax=10.0, fres_coarse=1.0, fres_fine=0.1, n_local_max=10)
        fbest = my_per.get_best_frequency()
        if 'wmcc_bestperiod' in selected_features:
            result_vals['wmcc_bestperiod'] = fbest[1]
        if 'wmcc_bestfreq' in selected_features:
            result_vals['wmcc_bestfreq'] = fbest[0]
        return result_vals
    
            
class LightCurve(object):

    def __init__(self, date, mag, mag_err):
        
        mag_series = pd.Series(mag, index=date)
        mag_err_series = pd.Series(mag_err, index=date)
        data_dict = {'mag': mag_series, 'mag_err': mag_err_series}
        
        self.series = pd.DataFrame(data_dict)
        self.features = dict()
        self.ra = None
        self.dec = None
        self.obj_type = None
        self.zspec = None

    def get_dates(self):
        return self.series.axes[0].values

    def get_mag(self):
        return self.series.mag.values

    def get_mag_err(self):
        return self.series.mag_err.values

    def as_array(self):
        return np.array([
            self.get_mag(),
            self.get_dates(),
            self.get_mag_err()
        ])

    def set_features(self, feature_names=None, feature_values=None, feature_dict=None):
        if feature_dict:
            self.features.update(feature_dict)
            return
        for this_name, this_value in zip(feature_names, feature_values):
            self.features[this_name] = this_value


class FeatureData(object):

    def __init__(self, filename):
        self.store = pd.HDFStore(filename, format='table')
        self.features = dict()

    def add_features(self, lightcurve):
        lc_index = (lightcurve.ra, lightcurve.dec)
        feature_names = lightcurve.features.keys()
        this_features = dict()
        for feature_name in feature_names:
            this_features[feature_name] = lightcurve.features[feature_name]
        this_features['ZSPEC'] = lightcurve.zspec  
        this_features['TYPE'] = lightcurve.obj_type 
        self.features[lc_index] = this_features

    def save_to_store(self):
        this_frame = pd.DataFrame(self.features).T
        this_frame = this_frame.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        #import ipdb;ipdb.set_trace()
        self.store.append('features', this_frame, format='table', min_itemsize={'TYPE':8})
        self.features = dict()

    def get_features(self, exclude=None):
        
        this_features = self.store.features
        return this_features

def load_from_fits(fits_file):

    this_fits = pyfits.open(fits_file)
    fits_data = this_fits[1].data
    fits_header = this_fits[0].header

    try:
        this_lc = LightCurve(fits_data['JD'], fits_data['Q'], fits_data['errQ'])
    except KeyError:
       raise ValueError("FITS file \"{}\" does not contain light curve data.".format(fits_file)) 
    this_lc.ra = fits_header['ALPHA']
    this_lc.dec = fits_header['DELTA']
    this_lc.features['u'] = fits_header['U']
    this_lc.features['g'] = fits_header['G']
    this_lc.features['r'] = fits_header['R']
    this_lc.features['i'] = fits_header['I']
    this_lc.features['z'] = fits_header['Z']
    try:
        this_lc.obj_type = fits_header['TYPE']
        this_lc.zspec = fits_header['ZSPEC']
    except KeyError:
        pass
    
    return this_lc

def curves_from_dir(folder):
    
    #fits_list = list()
    files = glob.glob(os.path.join(folder, '*.fits'))
    print "Reading..."
    for index, fits_file in enumerate(files):
    #    fits_list.append(load_from_fits(fits_file))
        print "File [%d/%d] \"%s\"" % (index, len(files), fits_file)
        try:
            yield index, load_from_fits(fits_file)
        except ValueError as e:
            print e
    print "Done"
    #return fits_list

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


def main():

    data_ids = ['magnitude', 'time', 'error']

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-s', '--store', required=True)
    args = parser.parse_args()

    fats_method = FATSMethod(feature_list, data_ids)
    mcmc_method = MCMCMethod(feature_list)
    p4j_method = P4JMethod(feature_list)
    
    feature_data = FeatureData(args.store)
    
    for index, light_curve in curves_from_dir(args.directory):
        
        #mcmc_vals = fitSF_mcmc(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err(), 2, 250, 500)
        mcmc_vals = mcmc_method.calculate_features(light_curve)
        light_curve.set_features(feature_dict=mcmc_vals)
        
        fats_vals = fats_method.calculate_features(light_curve)
        light_curve.set_features(fats_vals.featureList, fats_vals.result())

        p4j_vals = p4j_method.calculate_features(light_curve)
        light_curve.set_features(feature_dict=p4j_vals)

        feature_data.add_features(light_curve)
        
        if index % 100 == 0 and index  > 0:
            #import ipdb;ipdb.set_trace()
            feature_data.save_to_store()
    
    #import ipdb;ipdb.set_trace()
    feature_data.save_to_store()
    #import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    main()
