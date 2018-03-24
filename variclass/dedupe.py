from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from pyfits import getval
import glob
import os
import shutil

def is_in_coords(new_coord, old_coords):
    for old_file, old_coord in old_coords:
        sep = old_coord.separation(new_coord).arcsecond
        if sep <= 1.0:
            return True
    return False

new_files = glob.glob(os.path.join('/home/nmiranda/workspace/thesis/data/new_curves/*fits'))
old_files = glob.glob(os.path.join('/home/nmiranda/workspace/thesis/data/train_all/clean_*fits'))

#import ipdb;ipdb.set_trace()

new_coords = [(new_file, SkyCoord(ra=getval(new_file, 'ALPHA', 0)*u.degree, dec=getval(new_file, 'DELTA', 0)*u.degree)) for new_file in new_files]
old_coords = [(old_file, SkyCoord(ra=getval(old_file, 'ALPHA', 0)*u.degree, dec=getval(old_file, 'DELTA', 0)*u.degree)) for old_file in old_files if getval(old_file, 'TYPE_SPEC', 0).strip() == 'QSO']

#a, b, c = search_around_sky(new_coords, old_coords, 1*u.arcsec)

#import ipdb;ipdb.set_trace()

num_duplicates = 0
checked_files = 0

for new_file, new_coord in new_coords:
    if is_in_coords(new_coord, old_coords):
        num_duplicates += 1
    else:
        shutil.copy2(new_file, '/home/nmiranda/workspace/thesis/data/train_all_new/')
    checked_files += 1
    #if checked_files % 100 == 0:
    print checked_files

print "Duplicates: ", num_duplicates

