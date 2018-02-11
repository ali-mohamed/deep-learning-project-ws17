from __future__ import print_function
import argparse
import os
import numpy as np
import random
import sys
import time

seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)

num_classes = 101
data_per_class = 750
num_features = 8192

X = np.load("data.dat.npy")
y = np.load("labels.dat.npy")

print(X.shape)
print(y.shape)

from sklearn.svm import SVC
from sklearn.externals import joblib
test_per_class = 250

X_test = np.load("test.dat.npy")
y_test = np.load("labels_test.dat.npy")

sys.stdout.flush()

for c in range(1, 11):
  start_time = time.time()

  print("Number of classes: " + str(c*10))
  clf = SVC(C=0.1, kernel='linear', cache_size=40000, random_state=seed)
  clf.fit(X[0:c*10*data_per_class, :],y[0:c*10*data_per_class])

  print("X Shape: " + str(X[0:c*10*data_per_class, :].shape))
  print("y shape: " + str(y[0:c*10*data_per_class].shape))

  print("Training Done.")

  print("Test X Shape: " + str(X_test[0:c*10*test_per_class, :].shape))
  print("Test y shape: " + str(y_test[0:c*10*test_per_class].shape))

  print("Score is: " + str(clf.score(X_test[0:c*10*test_per_class, :], y_test[0:c*10*test_per_class])))

  print("--- %s seconds ---" % (time.time() - start_time))

  sys.stdout.flush()


print("Done")
sys.stdout = orig_stdout
f.close()
