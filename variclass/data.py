# -*- coding: utf-8 -*-
import numpy as np
import pyfits
import glob
import os
import random
from scipy import stats
import lc_simulation
import pickle

FILENAME_ARR_FILE = 'filename_list'
TIMES_ARR_FILE = 'jd_list'
MAGS_ARR_FILE = 'q_list'
MAGS_ERR_ARR_FILE = 'q_err_list'
TYPES_ARR_FILE = 'type_list'

def load_as_pickle(filename):
    with open(filename, 'rb') as _input:
        return pickle.load(_input)

def save_as_pickle(data, filename):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

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

def load(directory=None, subset_num=None, with_filenames=True, with_errors=True, sel_longest=None, prefix='clean', sel_timespan=None, sel_epochs=None):

    if directory:

        filename_list = list()
        jd_list = list()
        q_list = list()
        type_list = list()
        q_err_list =list() 

        fits_files = glob.glob(os.path.join(directory, prefix + '*.fits'))
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

            if with_filenames:
                filename = os.path.basename(fits_file)
                #filename = '.'.join(os.path.basename(fits_file).split('.')[:-1])
                filename_list.append(filename)

            if 'TYPE_SPEC' in this_header:
                if this_header['TYPE_SPEC'].strip() == 'QSO':
                    type_list.append(1)
                else:
                    type_list.append(0)
            elif 'SDSS_NAME' in this_header:
                type_list.append(1)
            else:
                raise TypeError('No type')

            jd_list.append(this_jd)
            q_list.append(this_q)
            if with_errors:
                q_err_list.append(this_err_q)

            this_fits.close()

        if with_filenames:
            save_to_file(FILENAME_ARR_FILE, filename_list)
        save_to_file(TIMES_ARR_FILE, jd_list)
        save_to_file(MAGS_ARR_FILE, q_list)
        if with_errors:
            save_to_file(MAGS_ERR_ARR_FILE, q_err_list)
        save_to_file(TYPES_ARR_FILE, type_list, single_list=True)
        
    else:
        if with_filenames:
            filename_list = load_from_file(FILENAME_ARR_FILE + '.npz')
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

    if sel_timespan:
        num_to_select = int(sel_timespan * len(jd_list))
        timespans = [jd[-1] - jd[0] for jd in jd_list]
        sorted_idx = np.argsort(timespans)[::-1]
        sorted_idx = sorted_idx[:num_to_select]
        filename_list = [filename_list[idx] for idx in sorted_idx]
        jd_list = [jd_list[idx] for idx in sorted_idx]
        q_list = [q_list[idx] for idx in sorted_idx]
        q_err_list = [q_err_list[idx] for idx in sorted_idx]
        type_list = [type_list[idx] for idx in sorted_idx]

    if sel_epochs:
        num_to_select = int(sel_timespan * len(jd_list))
        epochs = [jd.shape[0] for jd in jd_list]
        sorted_idx = np.argsort(epochs)[::-1]
        sorted_idx = sorted_idx[:num_to_select]
        filename_list = [filename_list[idx] for idx in sorted_idx]
        jd_list = [jd_list[idx] for idx in sorted_idx]
        q_list = [q_list[idx] for idx in sorted_idx]
        q_err_list = [q_err_list[idx] for idx in sorted_idx]
        type_list = [type_list[idx] for idx in sorted_idx]

    return_list = list()
    if with_filenames:
        return_list.append(filename_list)
    return_list.append(jd_list)
    return_list.append(q_list)
    if with_errors:
        return_list.append(q_err_list)
    return_list.append(type_list)

    return tuple(return_list)

def simulate(num_samples, single_jd=None, sel_timespan=None, sel_epochs=None, load_cache=True, save_cache=False):

    if load_cache:
        sim_jd_list = load_as_pickle('sim_jd_list.pkl')
        sim_q_list = load_as_pickle('sim_q_list.pkl')
        sim_type_list = load_as_pickle('sim_type_list.pkl')

        subset = num_samples

        sim_jd_list_pos = [sim_jd_list[i] for i, _type in enumerate(sim_type_list) if _type == 1]
        sim_jd_list_neg = [sim_jd_list[i] for i, _type in enumerate(sim_type_list) if _type == 0]
        sim_q_list_pos = [sim_q_list[i] for i, _type in enumerate(sim_type_list) if _type == 1]
        sim_q_list_neg = [sim_q_list[i] for i, _type in enumerate(sim_type_list) if _type == 0]
        sim_type_list_pos = [_type for _type in sim_type_list if _type == 1]
        sim_type_list_neg = [_type for _type in sim_type_list if _type == 0]

        half_subset = subset/2
        sim_jd_list = sim_jd_list_pos[:half_subset] + sim_jd_list_neg[:half_subset]
        sim_q_list = sim_q_list_pos[:half_subset] + sim_q_list_neg[:half_subset]
        sim_type_list = sim_type_list_pos[:half_subset] + sim_type_list_neg[:half_subset]

        print "Loaded {} samples.".format(len(sim_jd_list))

        return sim_jd_list, sim_q_list, np.asarray(sim_type_list)

    print "Simulating {} samples...".format(num_samples)

    filename_list, jd_list, q_list, q_err_list, type_list = load(with_filenames=True, with_errors=True, sel_timespan=None, sel_epochs=None)

    if single_jd:
        # Searching for selected jd
        i = 0
        for filename in filename_list:
            if filename == single_jd:
                break
            i += 1
        selected_jd = jd_list[i]

    curr_sample_num = 0

    half_num_samples = num_samples/2

    synth_q_list_pos = list()
    synth_jd_list_pos = list()
    synth_type_list_pos = list()
    for i in xrange(half_num_samples):
        if not single_jd:
            selected_jd = random.choice(jd_list)
        this_jd_pos = selected_jd - selected_jd[0]
        this_q_pos = lc_simulation.gen_DRW_long(i, sampling=True, timestamp=this_jd_pos)[3]
        synth_jd_list_pos.append(this_jd_pos)
        synth_q_list_pos.append(this_q_pos)
        synth_type_list_pos.append(1)

        curr_sample_num += 1
        if curr_sample_num % 200 == 0:
            print "{} samples generated...".format(curr_sample_num)

    synth_q_list_neg = list()
    synth_jd_list_neg = list()
    synth_type_list_neg = list()
    for i in xrange(len(synth_q_list_pos)):
        if not single_jd:
            selected_jd = random.choice(jd_list)
        this_jd_neg = selected_jd - selected_jd[0]
        mean_mag = np.mean(synth_q_list_pos[i])
        err_mag = np.std(synth_q_list_pos[i])
        this_q_neg = lc_simulation.gen_gaussian_noise(this_jd_neg, mean_mag, err_mag)
        synth_jd_list_neg.append(this_jd_neg)
        synth_q_list_neg.append(this_q_neg)
        synth_type_list_neg.append(0)

        curr_sample_num += 1
        if curr_sample_num % 200 == 0:
            print "{} samples generated...".format(curr_sample_num)

    print "Finished."
    
    synth_jd_list = synth_jd_list_pos + synth_jd_list_neg
    synth_q_list = synth_q_list_pos + synth_q_list_neg
    synth_type_list = synth_type_list_pos + synth_type_list_neg

    if save_cache:
        save_as_pickle(synth_jd_list, 'sim_jd_list.pkl')
        save_as_pickle(synth_q_list, 'sim_q_list.pkl')
        save_as_pickle(synth_type_list, 'sim_type_list.pkl')

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