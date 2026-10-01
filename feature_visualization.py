# Visualise how separable the 6 event classes are in feature space.
# The high-dimensional features (384 for the SVM, 400 for the CNN) are projected to 3-D
# with Linear Discriminant Analysis (LDA) and plotted as a 3-D scatter plot.
#
# Usage:
#   python feature_visualization.py                      # SVM features (default)
#   python feature_visualization.py --model cnn          # CNN features
#   python feature_visualization.py --datapath my.csv    # any CSV with the label in the last column
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

CLASS_NAMES = ['background', 'digging', 'knocking', 'watering', 'shaking', 'walking']
CLASS_COLORS = ['red', 'blue', 'darkorange', 'green', 'm', 'black']   # 'm' = magenta/purple

# CSV files written by das_data_svm.py and das_data_cnn.py respectively.
DEFAULT_FILES = {'svm': '5km_10km_svm_feature_data.csv', 'cnn': 'feature_data.csv'}

parser = argparse.ArgumentParser(description="3-D LDA visualisation of the extracted features")
parser.add_argument("--model", choices=['svm', 'cnn'], default='svm',
                    help="which model's features to plot")
parser.add_argument("--datapath", type=str, default=None,
                    help="CSV file to plot (defaults to the file written by the chosen model)")
args = parser.parse_args()
datapath = args.datapath or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         DEFAULT_FILES[args.model])

dataset = pd.read_csv(datapath, header=None)
dataset = np.array(dataset)  # (n_samples, n_features + 1): SVM test set -> (3084, 385), CNN -> (3084, 401)
print(dataset.shape)
X = dataset[:, :-1]   # features: every column except the last
y = dataset[:, -1]    # labels: the last column
lda = LinearDiscriminantAnalysis(n_components=3)
X = lda.fit_transform(X, y)  # fit LDA and reduce to 3 dimensions

ax = plt.subplot(projection="3d")
for label, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    mask = y == label
    if not np.any(mask):   # skip classes that are absent from the file
        continue
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], s=20, color=color, marker=".", label=name)

ax.view_init(elev=30, azim=45)
ax.legend(fontsize='small', edgecolor='black',
          bbox_to_anchor=(1, 0.4), loc='lower left')
plt.savefig('./%s_feature_data.jpg' % args.model)
plt.show()
