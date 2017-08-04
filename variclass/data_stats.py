# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
from collections import Counter
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	args = parser.parse_args()

	type_list = list()
	num_obs_list = list()
	fits_files = glob.glob(os.path.join(args.dir, '*.fits'))

	#print "Fits files: %s" % len(fits_files)
	for fits_file in fits_files:
		this_fits = pyfits.open(fits_file)
		this_header = this_fits[0].header
		this_data = this_fits[1].data
		this_num_obs = this_data['JD'].shape[0]
		num_obs_list.append(this_num_obs)
		type_list.append(this_header['TYPE_SPEC'])

	print "Fits files: %s" % len(type_list)
	counter = Counter(type_list)

	print counter

	plt.figure(figsize=(16,8))
	plt.hist(num_obs_list, bins=100, cumulative=False, histtype='stepfilled')
	plt.xticks(range(0,1800,100))
	plt.xlabel('Number of observations')
	plt.ylabel('Number of lightcurves')

	plt.show()




if __name__ == '__main__':
    main()