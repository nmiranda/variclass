# -*- coding: utf-8 -*-
import numpy as np
import pyfits
import glob
import os
import random

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

        sort_index = [i[0] for i in sorted(enumerate(jd_list), key=lambda x:len(x[1]), reverse=True)]
        sort_index = sort_index[:int(sel_longest * len(sort_index))]
        jd_list = [jd_list[i] for i in sort_index]
        q_list = [q_list[i] for i in sort_index]
        if with_errors:
            q_err_list = [q_err_list[i] for i in sort_index]
        type_list = np.asarray([type_list[i] for i in sort_index])

    if with_errors:
        return (jd_list, q_list, q_err_list, type_list)

    return (jd_list, q_list, type_list)