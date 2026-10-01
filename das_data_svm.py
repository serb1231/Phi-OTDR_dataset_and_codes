#coding = UTF-8
import sys


class Logger(object):
    # Duplicates everything printed to stdout into a log file as well.
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger('svm_result.log', sys.stdout)
import datetime
import os
from sklearn import svm, preprocessing
from get_das_data import get_das_data
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLASS_NAMES = ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking']


def safe_div(a, b):
    # Division that returns 0 instead of raising / producing NaN when b == 0.
    return a / b if b != 0 else 0.0


# By default the dataset is expected in a "das_data" folder next to this script.
rootpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'das_data')
train_rootpath = rootpath+'/train'
train_labelpath = rootpath+'/train/label.txt'
test_rootpath = rootpath+'/test'
test_labelpath = rootpath+'/test/label.txt'

# Extract the 384 hand-crafted features (12 channels x 32 features) for every sample.
start_train = datetime.datetime.now()
X_train, y_train = get_das_data(train_rootpath, train_labelpath)
X_test, y_test = get_das_data(test_rootpath, test_labelpath)

pre_y_test = y_test[:, np.newaxis]   # labels as a column vector, (n_test, 1)

# Scale every feature to [0, 1]. The scaler is fitted on the training set only and then
# applied unchanged to the test set, so no information from the test set leaks into training.
minMaxScaler = preprocessing.MinMaxScaler()
trainingData = minMaxScaler.fit_transform(X_train)
testData = minMaxScaler.transform(X_test)

# Save the scaled test features with the label as the last column (used by feature_visualization.py).
feature_data = np.concatenate((testData, pre_y_test), axis=1)
np.savetxt('5km_10km_svm_feature_data.csv', feature_data, delimiter=',')

# Support vector machine with an RBF kernel.
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(trainingData, y_train)
end_train = datetime.datetime.now()

train_result = clf.predict(trainingData)

start_test = datetime.datetime.now()
test_result = clf.predict(testData)
end_test = datetime.datetime.now()

# C[i][j] = number of samples whose true class is i and predicted class is j.
labels = list(range(len(CLASS_NAMES)))
train_matrix = confusion_matrix(y_train, train_result, labels=labels)
test_matrix = confusion_matrix(y_test, test_result, labels=labels)
print('train_matrix:\n', train_matrix)
print('test_matrix:\n', test_matrix)
print('train time is ', end_train - start_train)
print('test time is ', end_test - start_test)
C = test_matrix
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
df = pd.DataFrame(C)
sns.heatmap(df, fmt='g', annot=True, robust=True,
                annot_kws={'size': 10},
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues', ax=ax)
ax.set_xlabel('Predicted label', fontsize=15)  # x axis
ax.set_ylabel('True label', fontsize=15)  # y axis
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.savefig('./SVM_confusion_matrix.jpg')
plt.show()


# Accuracy: correctly classified samples (the diagonal) / all samples
Acc = safe_div(np.trace(C), np.sum(C))
print('acc: %.4f' % Acc)
# NAR (nuisance alarm rate): background samples predicted as an event (false alarms)
# divided by all samples predicted as an event (all alarms). See README.
NAR = safe_div(np.sum(C[0, 1:]), np.sum(C[:, 1:]))
print('NAR: %.4f' % NAR)
# FNR (false negative rate): real events predicted as background (missed events)
# divided by all real events.
FNR = safe_div(np.sum(C[1:, 0]), np.sum(C[1:, :]))
print('FNR: %.4f' % FNR)
column_sum = np.sum(C, axis=0)   # number of predictions per class
print(column_sum)
row_sum = np.sum(C, axis=1)      # number of true samples per class
print(row_sum)

# Per-class precision, recall and F1-score (classes are numbered 1-6 in the output).
for i in range(1, len(C) + 1):
    Precision = safe_div(C[i - 1][i - 1], column_sum[i - 1])
    Recall = safe_div(C[i - 1][i - 1], row_sum[i - 1])
    F1 = safe_div(2*Precision*Recall, Precision+Recall)
    print('precision_%d: %.3f' % (i, Precision))
    print('Recall_%d: %.3f' % (i, Recall))
    print('F1_%d: %.3f' % (i, F1))
