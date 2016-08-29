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
from pgram_func2 import get_period_sigf
from pathos.multiprocessing import ProcessingPool as Pool
import carmcmc as cm
import itertools

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
    'A_mcmc',
    'gamma_mcmc',
    'wmcc_bestperiod', 
    'wmcc_bestfreq',
    'pg_best_period', 
    'pg_peak', 
    'pg_sig5', 
    'pg_sig1',
    'tau_mc', 
    'sigma_mc',
    'pvar',
    'ex_var',
    'ex_var_err',
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
    
    def __init__(self, selected_features=supported_features):
        self.features = list()
        for feature in selected_features:
            if feature in self.supported_features:
                self.features.append(feature)

    def calculate_features(self, light_curve):
        return_vals = dict()
        my_per = P4J.periodogram(M=1,  method="WMCC")
        my_per.fit(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err())
        my_per.grid_search(fmin=0.0, fmax=10.0, fres_coarse=1.0, fres_fine=0.1, n_local_max=10)
        fbest = my_per.get_best_frequency()
        if 'wmcc_bestperiod' in self.features:
            return_vals['wmcc_bestperiod'] = fbest[1]
        if 'wmcc_bestfreq' in self.features:
            return_vals['wmcc_bestfreq'] = fbest[0]
        return return_vals

class PeriodgramMethod(FeatureMethod):

    supported_features = ['pg_best_period', 'pg_peak', 'pg_sig5', 'pg_sig1']

    def __init__(self, selected_features=supported_features):
        self.features = list()
        for feature in selected_features:
            if feature in self.supported_features:
                self.features.append(feature)

    def calculate_features(self, light_curve):
        return_vals = dict()
        best_period, peak, sig5, sig1 = get_period_sigf(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err())
        return_vals[self.supported_features[0]] = best_period
        return_vals[self.supported_features[1]] = peak
        return_vals[self.supported_features[2]] = sig5
        return_vals[self.supported_features[3]] = sig1
        return return_vals

class CARMCMCMethod(FeatureMethod):

    supported_features = ['tau_mc', 'sigma_mc']

    def __init__(self, selected_features=supported_features):
        self.features = list()
        for feature in selected_features:
            if feature in self.supported_features:
                self.features.append(feature)

    def calculate_features(self, light_curve):
        jd = light_curve.get_dates()
        mag = light_curve.get_mag()
        errmag = light_curve.get_mag_err()
        z = light_curve.zspec

        import ipdb;ipdb.set_trace()
        
        model = cm.CarmaModel(jd/(1+z), mag, errmag, p=1, q=0)
        sample = model.run_mcmc(20000)
        log_omega=sample.get_samples('log_omega')
        tau=np.exp(-1.0*log_omega)
        sigma=sample.get_samples('sigma')
        tau_mc=(np.percentile(tau, 50),np.percentile(tau, 50)-np.percentile(tau, 15.865),np.percentile(tau, 84.135)-np.percentile(tau, 50))
        sigma_mc=(np.percentile(sigma, 50),np.percentile(tau, 50)-np.percentile(sigma, 15.865),np.percentile(sigma, 84.135)-np.percentile(sigma, 50))
        return_vals = {self.supported_features[0]: tau_mc, self.supported_features[1]: sigma_mc}
        
        return return_vals


class PaulaMethod(FeatureMethod):

    supported_features = ['pvar', 'ex_var', 'ex_var_err']

    def __init__(self, selected_features=supported_features):
        self.features = list()
        for feature in selected_features:
            if feature in self.supported_features:
                self.features.append(feature)


    def calculate_features(self, light_curve):
        vals = var_parameters(light_curve.get_dates(), light_curve.get_mag(), light_curve.get_mag_err())
        return {
            supported_features[0]: vals[0],
            supported_features[1]: vals[1],
            supported_features[2]: vals[2],
            }
            
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

    def __repr__(self):
        return "LightCurve(ra=%s, dec=%s, len=%s)" % (self.ra, self.dec, self.series.size)

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

    def add_light_curves(self, lightcurves):
        for lightcurve in lightcurves:
            self.add_features(lightcurve)

    def add_features(self, lightcurve):
        lc_index = (str(lightcurve.ra), str(lightcurve.dec))
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

def load_from_fits(fits_file, args):

    this_fits = pyfits.open(fits_file)
    fits_data = this_fits[1].data
    fits_header = this_fits[0].header

    try:
#        this_lc = LightCurve(fits_data['JD'], fits_data['Q'], fits_data['errQ'])
        this_lc = LightCurve(fits_data[args.dates], fits_data[args.mag], fits_data[args.mag_err])
    except KeyError:
       raise ValueError("FITS file \"{}\" data does not contain specified values. Please check.".format(fits_file)) 
    this_lc.ra = fits_header['ALPHA']
    this_lc.dec = fits_header['DELTA']
#    this_lc.features['u'] = fits_header['U']
#    this_lc.features['g'] = fits_header['G']
#    this_lc.features['r'] = fits_header['R']
#    this_lc.features['i'] = fits_header['I']
#    this_lc.features['z'] = fits_header['Z']
    try:
        this_lc.obj_type = fits_header['TYPE']
    except KeyError:
        pass
    try:
        this_lc.zspec = fits_header['ZSPEC']
    except KeyError:
        pass
    return this_lc

def curves_from_dir(args):
    
    fits_list = list()
    files = glob.glob(os.path.join(args.folder, 'agn_*.fits'))
    print "Reading..."
    for index, fits_file in enumerate(files):
        print "File [%d/%d] \"%s\"" % (index, len(files), fits_file)
        try:
            fits_list.append(load_from_fits(fits_file, args))
        except ValueError as e:
            print e
    print "Done"
    return fits_list

def calc_features(light_curve, methods):
    for method in methods:
        light_curve.set_features(method.calculate_features(light_curve))
    return light_curve

def main():
    
    method_classes = [
            #FATSMethod,
            MCMCMethod,
            P4JMethod,
            PeriodgramMethod,
            CARMCMCMethod,
            ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-s', '--store', required=True)
    parser.add_argument('-n', '--num_cores', nargs='?', type=int, default=1)
    parser.add_argument('-d', '--dates', required=True)
    parser.add_argument('-m', '--mag', required=True)
    parser.add_argument('-e', '--mag_err', required=True)

    args = parser.parse_args()

    methods = [method_class(feature_list) for method_class in method_classes]
    
    feature_data = FeatureData(args.store)
    
    light_curves = curves_from_dir(args)

    if args.num_cores > 1:
        proc_pool = Pool(processes=int(args.num_cores))
        res_light_curves = proc_pool.map(calc_features, light_curves, itertools.repeat(methods))
        proc_pool.close()
        proc_pool.join()

    elif args.num_cores == 1:
        res_light_curves = map(calc_features, light_curves, itertools.repeat(methods))

    feature_data.add_light_curves(res_light_curves)
             
    feature_data.save_to_store()

if __name__ == "__main__":
    main()
