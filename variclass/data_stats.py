# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pyfits
from collections import Counter

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	args = parser.parse_args()

	type_list = list()
	fits_files = glob.glob(os.path.join(args.dir, '*.fits'))

	#print "Fits files: %s" % len(fits_files)
	for fits_file in fits_files:
		this_fits = pyfits.open(fits_file)
		this_header = this_fits[0].header
		type_list.append(this_header['TYPE_SPEC'])

	print "Fits files: %s" % len(type_list)
	counter = Counter(type_list)

	print counter




if __name__ == '__main__':
    main()