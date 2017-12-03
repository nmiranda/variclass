import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os

stats_dir = os.path.join(os.pardir, os.pardir, 'data', 'results')

def learning_process(stats, time_str):
    for loss_vals in stats.history.values():
        plt.plot(loss_vals)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(history.history.keys(), loc='best')
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    plt.savefig(os.path.join(stats_dir, model_name + '_conf_matrix_' + time_str + '.png'))

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
