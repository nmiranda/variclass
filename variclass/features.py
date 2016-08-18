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
from multiprocessing import Process, Manager

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
        return self.feat_space.calculateFeature(light_curve.as_array()).result(method='dict')

class MCMCMethod(FeatureMethod):

    supported_features = [
            'A_mcmc',
            'gamma_mcmc',
            'pvar',
        ]
    
    def __init__(self, selected_features=[]):
        self.features = list()
        for feature in selected_features:
            if feature in self.supported_features:
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
            if feature in self.supported_features:
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

    def set_features(self, feature_dict=None):
        if feature_dict:
            self.features.update(feature_dict)

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

def multi_calc_features(shared_dict, light_curve, method):

    this_method_vals = method.calculate_features(light_curve)
    shared_dict.update(this_method_vals)

def main():
    
    method_classes = [
            FATSMethod,
            MCMCMethod,
            P4JMethod,
            ]

    NUMBER_OF_PROCESSES = len(method_classes)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-s', '--store', required=True)
    args = parser.parse_args()

    manager = Manager()
    processes = list()

    methods = [method_class(feature_list) for method_class in method_classes]
    
    feature_data = FeatureData(args.store)
    
    for index, light_curve in curves_from_dir(args.directory):

        shared_dict = manager.dict()
        for method in methods:

            this_proc = Process(target=multi_calc_features, args=(shared_dict, light_curve, method))
            this_proc.start()
            processes.append(this_proc)
       
        for process in processes:
            process.join()
        
        light_curve.set_features(shared_dict)
        feature_data.add_features(light_curve)
        
        if index % 100 == 0 and index  > 0:
            feature_data.save_to_store()
    
    feature_data.save_to_store()

if __name__ == "__main__":
    main()
