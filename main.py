import torch

from dataset_reader import *

labels_directory = "data/labels"
labels_arr, features_arr = process_mats(labels_directory)
print labels_arr, features_arr
