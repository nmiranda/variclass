import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
from scipy import stats
import random

stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')

def learning_process(stats, title='Model loss', time_str=None, save=False, from_epoch=1):
    for loss_vals in stats['history'].values():
        plt.plot(range(from_epoch,len(loss_vals)+1), loss_vals[from_epoch-1:])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(xmin=from_epoch)
    plt.legend(stats['history'].keys(), loc='best')
    if save:
        plt.savefig(os.path.join(stats_dir, 'lstm_' + time_str + '.png'))

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
    print "Saved confusion matrix plot at \"{}\"".format(os.path.abspath(conf_matrix_plot_path))

def epoch_span(num_epochs, length):

    slope, intercept, _, _, _ = stats.linregress(num_epochs, length)
    print "Intercept: %s\nSlope: %s" % (intercept, slope)
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

        print index

        plt.plot(jd_list[index], q_list[index], "bs--")
        plt.title("TYPE: {}".format(type_list[index]))
        plt.xlabel('JD')
        plt.ylabel('Q')

def roc_curve(stats, title='ROC curve'):
    plt.plot(stats['roc_fpr'], stats['roc_tpr'], color='darkorange', lw=2, label='ROC curve (area = {:0.2f})'.format(stats['roc_auc']))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
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
