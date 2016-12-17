import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy

# To check the version of python
# print "Python : {}".format(sys.version)

# Keep track of values for F1 score in buckets
buckets = [0] * 10
num_exact_matches = 0
num_exact_zeros = 0

n_x = 22  # Columns which are considered features
n_y = 22  # the column for label

# Load dataset
data_file_path = "../data/featuredata_br/"
data_files = [f for f in listdir(data_file_path) if isfile(join(data_file_path, f)) and f.endswith('.csv')]
for i in data_files:
    url = data_file_path + i
    # names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url)  # , names=names)

    # Calculate histogram of the F1 score
    f1_scores = (dataset.values)[:, n_y]
    curr_hist = numpy.histogram(f1_scores, 10)
    buckets = numpy.add(curr_hist[0], buckets)
    num_exact_matches += (f1_scores == 1).sum()
    num_exact_zeros += (f1_scores == 0).sum()

print buckets
print num_exact_matches
print num_exact_zeros
