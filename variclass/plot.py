import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
from scipy import stats
import random
import json
import stats as st

stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')

# def learning_process(stats, title='Model loss', time_str=None, save=False, from_epoch=1, filter=None):
#     keys = list()
#     for name, loss_vals in stats['history'].items():
#         if filter and not (filter in name):
#             continue
#         keys.append(name)
#         plt.plot(range(from_epoch,len(loss_vals)+1), loss_vals[from_epoch-1:])
#     plt.title(title)
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.xlim(xmin=from_epoch)
#     plt.legend(keys, loc='best')
#     if save:
#         plt.savefig(os.path.join(stats_dir, 'lstm_' + time_str + '.png'))

def learning_process(files, title='Learning process', _filter=None, y_lim=(0.0,1.1)):
    stats_list = list()
    for file in files:
        with open(file, 'r') as stats_file:
            stats_list.append(json.load(stats_file))
    history_list = [stats['history'] for stats in stats_list]
    keys = history_list[0].keys()
    final_keys = list()
    for key in keys:
        if _filter and not (_filter in key):
            continue
        this_plot = np.array([history[key] for history in history_list])
        plt.plot(range(1,this_plot.shape[1]+1), this_plot.mean(axis=0))
        final_keys.append(key)
        #plt.errorbar(range(1,26), this_plot.mean(axis=0), this_plot.std(axis=0), capsize=5)
    plt.title(title)
    plt.ylabel('value')
    plt.xlabel('iteration')
    if y_lim:
        plt.ylim(y_lim)
    plt.legend(final_keys, loc='best')


def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          time_str=None,
                          model_name=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    #print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    conf_matrix_plot_path = os.path.join(stats_dir, model_name + '_conf_matrix_' + time_str + '.png')
    plt.savefig(conf_matrix_plot_path)
    print("Saved confusion matrix plot at \"{}\"".format(os.path.abspath(conf_matrix_plot_path)))

def epoch_span(num_epochs, length):

    slope, intercept, _, _, _ = stats.linregress(num_epochs, length)
    print("Intercept: %s\nSlope: %s" % (intercept, slope))
    x = np.linspace(0.0, 1500.0, num=5)

    plt.plot(num_epochs, length, ".")
    plt.plot(x, x*slope + intercept, color='black', linewidth=3)
    plt.xlabel('Number of epochs')
    plt.ylabel('Time span (days)')

def plot_random(*args, **kwargs):

    if len(args) == 3:

        jd_list, q_list, type_list = args

        while True:
            index = random.randrange(len(jd_list))
            if (kwargs['_type'] is not None) and (kwargs['_type'] != type_list[index]):
                continue
            break

        print(index)

        plt.plot(jd_list[index], q_list[index], kwargs['fmt'])
        plt.title("TYPE: {}".format(type_list[index]))
        plt.xlabel('JD')
        plt.ylabel('Q')

def roc_curve(filename, title='ROC curve'):
    with open(filename, 'r') as stats_file:
        stats = json.load(stats_file)
    plt.plot(stats['roc_fpr'], stats['roc_tpr'], color='darkorange', lw=2, label='ROC curve (area = {:0.2f})'.format(stats['roc_auc']))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

def scores(json_files, key, title="Scores", min_val=None, max_val=None, x_vals=None, xlabel='', ffor=None):
    stats_list = list()
    for json_file in json_files:
        with open(json_file, 'r') as json_file:
            stats_list.append(json.load(json_file))
    if x_vals:
        x_vals = x_vals
    else:
        try:
            x_vals = [stats[key] for stats in stats_list]
        except KeyError:
            print("In model conf")
            x_vals = [stats['model_conf'][key] for stats in stats_list]
        if min_val is not None:
            x_vals[0] = min_val
        if max_val is not None:
            x_vals[-1] = max_val
    conf_matrix_list = [stats['cnf_matrix'] for stats in stats_list]
    f1_score, mcc, jstat = st.get_scores(conf_matrix_list)
    roc_auc = [stats['roc_auc'] for stats in stats_list]
    if ffor:
        for y_vals in [f1_score, mcc, jstat, roc_auc]:
            y_vals[ffor[0]] = y_vals[ffor[0]] + ffor[1]
    plt.plot(x_vals, f1_score, 'o-')
    plt.plot(x_vals, mcc, 'o-')
    plt.plot(x_vals, jstat, 'o-')
    plt.plot(x_vals, roc_auc, 'o-')
    plt.ylabel('score')
    plt.xlabel(xlabel)
    plt.ylim(ymax=1.1)
    plt.ylim(ymin=-1.0)
    plt.legend(['F1 score', 'MCC', 'J statistic', 'ROC AUC'], loc='best')
    plt.title(title)

def exec_times(json_files_list, title="Execution times"):
    for json_files in json_files_list:
        stats_list = list()
        for json_file in json_files:
            with open(json_file, 'r') as json_file:
                stats_list.append(json.load(json_file))
        x_vals = [stats['simulate_samples'] for stats in stats_list]
        y_vals = [stats['exec_time'] for stats in stats_list]
        plt.plot(x_vals, y_vals, 'o-')
    plt.ylabel('Execution time [s]')
    plt.xlabel('Number of samples')
    plt.legend(['CNN1', 'CNN2', 'CNN3', 'RNN1', 'RNN2'], loc='best')
    plt.title(title)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('curve')
    args = parser.parse_args()

    this_curve = pd.read_csv(args.curve)['0']
    
    plt.plot(this_curve.index, this_curve, 'b*')
    plt.xlabel('days')
    plt.ylabel('mag')
    plt.show()
    #plt.savefig('lel.png')
    
if __name__ == '__main__':
    main()
