
import argparse
import sklearn
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import pyfits
import os

def get_class(val):
    if val == 1:
        return 'QSO'
    elif val == -1:
        return 'STAR'
    else:
        raise ValueError()

parser = argparse.ArgumentParser(description='Star/QSO classifier')
parser.add_argument('-tr', '--train', required=True, help='Input training features file')
parser.add_argument('-te', '--test', help='Input test features file')
parser.add_argument('-c', '--coef', help='Output coefficients file')
parser.add_argument('-o', '--out', required=True, help='Output classified objects file')
parser.add_argument('-d', '--dir', required=True, help='Testing objects directory')
args = parser.parse_args()

#feats = np.genfromtxt(args.in_file, dtype=None)
loaded_arrays = np.load(args.train)

ones = np.ones(len(loaded_arrays['y']))
minus_ones = ones * -1

print "Loading arrays... "
X = preprocessing.scale(loaded_arrays['X'])
#X = preprocessing.scale(np.delete(loaded_arrays['X'], 15, 1))
X_names = loaded_arrays['X_names']
X_keys = loaded_arrays['X_keys']
y = np.where(loaded_arrays['y'] == 'QSO', ones, minus_ones)

if args.test:

    loaded_test_args = np.load(args.test)
    loaded_X = loaded_test_args['X']
    loaded_names = loaded_test_args['X_names']
    loaded_keys = loaded_test_args['X_keys']

    # load ra dec 

    dir_testing = args.dir

    meta_data = np.zeros((loaded_names.shape[0], 4), dtype=np.float64)

    for idx, file_name in enumerate(loaded_names):
        print "Reading file [%d/%d]" % (idx+1, loaded_names.shape[0])
        fits_file = pyfits.open(os.path.join(dir_testing, file_name))
        meta_data[idx,0] = fits_file[0].header['ALPHA']
        meta_data[idx,1] = fits_file[0].header['DELTA']
        meta_data[idx,2] = fits_file[1].data.shape[0]
        meta_data[idx,3] = fits_file[1].data[-1][0] - fits_file[1].data[0][0]

    miss_index = None
    for idx, key in enumerate(X_keys):
        if key not in loaded_keys:
            miss_index = idx
            break
    if miss_index is not None:
        X = np.delete(X, miss_index, 1)
        X_keys = np.delete(X_keys, miss_index, 0)

    miss_index = None
    for idx, key in enumerate(loaded_keys):
        if key not in X_keys:
            miss_index = idx
    if miss_index is not None:
        loaded_X = np.delete(loaded_X, miss_index, 1)
        loaded_keys = np.delete(loaded_keys, miss_index, 0)

    try:
        nan_index = np.where(np.isnan(loaded_X[0]))[0][0]
        loaded_X = np.delete(loaded_X, nan_index, 1)
        loaded_keys = np.delete(loaded_keys, nan_index, 0)
        X = np.delete(X, nan_index, 1)
        X_keys = np.delete(X_keys, nan_index, 0)
    except IndexError:
        pass


print "loaded."

print "Splitting training set... ",
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print "ready."

estimator = SVC(kernel='linear')

cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

gammas = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
print "Fitting..."
classifier.fit(X_train, y_train)
print "Done"

print "Classifier score: ", classifier.score(X_test, y_test)



if args.test:

    X_testing = preprocessing.scale(loaded_X)
    result = classifier.predict(X_testing)

    result_classes = [get_class(val) for val in result]

    this_header = '\t'.join(['file', 'ra', 'dec', 'num_epochs', 't_range'] + [val for val in loaded_keys] + ['pred_class', 'conf_qso'])
    
    result_conf = classifier.decision_function(X_testing)
    output_matrix = np.column_stack((loaded_names, meta_data, loaded_X, result_classes, result_conf))

    np.savetxt(args.out, output_matrix, fmt="%s", header=this_header, delimiter="\t")
else:
    result = classifier.predict(X)
    result_classes = [get_class(val) for val in result]
    output_matrix = np.column_stack((X_names, loaded_arrays['y'], result_classes))
    np.savetxt(args.out, output_matrix, fmt="%s", header="name\treal_class\tpred_class", delimiter="\t")


if args.coef:
    coef_matrix = np.column_stack((X_keys, np.absolute(classifier.best_estimator_.coef_[0])))
    coef_matrix = coef_matrix[coef_matrix[:,1].argsort()]
    np.savetxt(args.coef, coef_matrix, fmt="%s %s")
