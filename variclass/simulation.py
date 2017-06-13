# -*- coding: utf-8 -*-
import argparse
from lc_simulation import gen_DRW_long
import pandas as pd
import time
import os

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('num_curves', type=int)
	parser.add_argument('-d', '--dir')
	args = parser.parse_args()

	for i in xrange(args.num_curves):
		cur_time_millis = int(time.time() * 1000)
		this_curve = gen_DRW_long(seed=cur_time_millis % (10**9))
		this_series = pd.DataFrame(this_curve[3], index=this_curve[0])
		this_series.to_csv(os.path.join(args.dir, str(cur_time_millis) + '.csv'))

if __name__ == "__main__":
    main()