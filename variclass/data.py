# -*- coding: utf-8 -*-
import numpy as np
import pyfits
import glob
import os
import random
from scipy import stats
import lc_simulation

TIMES_ARR_FILE = 'jd_list'
MAGS_ARR_FILE = 'q_list'
MAGS_ERR_ARR_FILE = 'q_err_list'
TYPES_ARR_FILE = 'type_list'

def load_from_file(filename, single_list=False):
    with np.load(filename) as npzfile:
        if single_list:    
            return npzfile.items()[0][1]
        else:
            return [array for _, array in npzfile.iteritems()]

def save_to_file(filename, dataset, single_list=False):
    if single_list:
        np.savez(filename, dataset)
    else:
        np.savez(filename, *dataset)

def load(directory=None, subset_num=None, with_errors=False, sel_longest=None):

    if directory:

        jd_list = list()
        q_list = list()
        type_list = list()
        q_err_list =list() 

        fits_files = glob.glob(os.path.join(directory, '*.fits'))
        for fits_file in fits_files:
            this_fits = pyfits.open(fits_file, memmap=False)
            try:
                this_data = this_fits[1].data
            except TypeError:
                continue
            this_header = this_fits[0].header

            this_jd = this_data['JD']
            this_q = this_data['Q']
            this_err_q = this_data['errQ']

            if this_header['TYPE_SPEC'].strip() == 'QSO':
                type_list.append(1)
            else:
                type_list.append(0)

            jd_list.append(this_jd)
            q_list.append(this_q)
            if with_errors:
                q_err_list.append(this_err_q)

            this_fits.close()

        save_to_file(TIMES_ARR_FILE, jd_list)
        save_to_file(MAGS_ARR_FILE, q_list)
        if with_errors:
            save_to_file(MAGS_ERR_ARR_FILE, q_err_list)
        save_to_file(TYPES_ARR_FILE, type_list, single_list=True)
        
    else:
        jd_list = load_from_file(TIMES_ARR_FILE + '.npz')
        q_list = load_from_file(MAGS_ARR_FILE + '.npz')
        if with_errors:
            q_err_list = load_from_file(MAGS_ERR_ARR_FILE + '.npz')
        type_list = load_from_file(TYPES_ARR_FILE + '.npz', single_list=True)


    # Option for selecting a random subset
    # if subset_num:
    #     try:
    #         subset_indexes = random.sample(xrange(len(jd_list)), subset_num)

    #         jd_list = [jd_list[index] for index in subset_indexes]
    #         q_list = [q_list[index] for index in subset_indexes]
    #         type_list = [type_list[index] for index in subset_indexes]
    #     except ValueError:
    #         pass

    if sel_longest:
        """
        sort_index = [i[0] for i in sorted(enumerate(jd_list), key=lambda x:len(x[1]), reverse=True)]
        sort_index = sort_index[:int(sel_longest * len(sort_index))]
        jd_list = [jd_list[i] for i in sort_index]
        q_list = [q_list[i] for i in sort_index]
        if with_errors:
            q_err_list = [q_err_list[i] for i in sort_index]
        type_list = np.asarray([type_list[i] for i in sort_index])
        """

        """
        idx_jd_type = [(idx, jd_type[0], jd_type[1]) for idx, jd_type in enumerate(zip(jd_list, type_list))]
        idx_jd_positive = [(idx, jd) for idx, jd, _type in idx_jd_type if _type == 1]
        idx_jd_negative = [(idx, jd) for idx, jd, _type in idx_jd_type if _type == 0]
        import ipdb;ipdb.set_trace()

        num_samples = len(idx_jd_positive)
        """

        # Calculate linear regression values for number of epochs and time span
        num_samples = len(jd_list)
        num_epochs = [jd.shape[0] for jd in jd_list]
        length = [jd[-1] - jd[0] for jd in jd_list]
        slope, intercept, _, _, _ = stats.linregress(num_epochs, length)

        x_epochs = np.linspace(0, 1500, num=500)
        y_lengths = x_epochs*slope + intercept
        for x, y in zip(x_epochs, y_lengths):
            selected_indexes = [idx_jdlist[0] for idx_jdlist in enumerate(jd_list) if idx_jdlist[1].shape[0] > x and (idx_jdlist[1][-1] - idx_jdlist[1][0]) > y]
            this_ratio = float(len(selected_indexes)) / num_samples
            if this_ratio < sel_longest:
                break
        jd_list = [jd_list[i] for i in selected_indexes]
        q_list = [q_list[i] for i in selected_indexes]
        type_list = np.asarray([type_list[i] for i in selected_indexes])

    if with_errors:
        return (jd_list, q_list, q_err_list, type_list)

    return (jd_list, q_list, type_list)

def simulate(num_samples):

    jd_list, q_list, q_err_list, type_list = load(with_errors=True)

    #num_samples = len(jd_list)
    pos_type_idx = [x[0] for x in enumerate(type_list) if x[1] == 1]
    neg_type_idx = [x[0] for x in enumerate(type_list) if x[1] == 1]

    curr_sample_num = 0
    pos_idx = 0
    neg_idx = 0

    synth_q_list = list()
    synth_jd_list = list()
    synth_type_list = list()

    while True:
        # QSO
        this_jd = jd_list[pos_type_idx[pos_idx]] - jd_list[pos_type_idx[pos_idx]][0]
        this_q = q_list[pos_type_idx[pos_idx]]
        this_q_err = q_err_list[pos_type_idx[pos_idx]]
        mean_mag = np.mean(this_q)
        err_mag = np.mean(this_q_err)
        err_nu = np.random.normal(10.0**-6, 10.0**-7)
        synth_q = lc_simulation.gen_lc_long(0, 0, 0, mean_mag, err_mag, 'bending-pl', 2.7, True, this_jd, err_nu)
        synth_q_list.append(synth_q[3])
        synth_jd_list.append(this_jd)
        synth_type_list.append(1)
        curr_sample_num += 1
        pos_idx += 1

        # NON-QSO
        this_jd = jd_list[neg_type_idx[neg_idx]] - jd_list[neg_type_idx[neg_idx]][0]
        this_q = q_list[neg_type_idx[neg_idx]]
        this_q_err = q_err_list[neg_type_idx[neg_idx]]
        mean_mag = np.mean(this_q)
        err_mag = np.mean(this_q_err)
        synth_q = lc_simulation.gen_gaussian_noise(this_jd, mean_mag, err_mag)
        synth_q_list.append(synth_q)
        synth_jd_list.append(this_jd)
        synth_type_list.append(0)
        curr_sample_num += 1
        neg_idx += 1

        if curr_sample_num >= num_samples:
            break

        if pos_idx == len(pos_type_idx):
            pos_idx = 0
        if neg_idx == len(neg_type_idx):
            neg_idx = 0

    return synth_jd_list, synth_q_list, np.asarray(synth_type_list)


    """
    for index in xrange(num_samples):
        this_jd = jd_list[index] - jd_list[index][0]
        this_q = q_list[index]
        this_q_err = q_err_list[index]
        mean_mag = np.mean(this_q)
        err_mag = np.mean(this_q_err)
        if type_list[index] == 1:
            err_nu = np.random.normal(10.0**-6, 10.0**-7)
            synth_q = lc_simulation.gen_lc_long(0, 0, 0, mean_mag, err_mag, 'bending-pl', 2, True, this_jd, err_nu)
        elif type_list[index] == 0:
    """