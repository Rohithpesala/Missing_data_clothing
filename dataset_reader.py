
from utils import *

def process_mats(directory):
	label_arr, num_files = load_labels(directory)
	num_instances = label_arr.shape[0]
	features_arr = load_features(directory, num_files, num_instances)
	process_features_multinomial(features_arr)
	return label_arr.shape, features_arr.shape



def process_images(directory):
	pass
