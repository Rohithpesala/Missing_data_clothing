
from utils import *

def preprocess_data(labels_directory):
	labels_arr, features_arr = process_mats(labels_directory)
	split_nan_images("data",labels_arr)

def process_mats(directory):
	label_arr, num_files = load_labels(directory)
	num_instances = label_arr.shape[0]
	features_arr = load_features(directory, num_files, num_instances)
	# process_features_multinomial(features_arr)
	process_features_constant(features_arr)
	return label_arr, features_arr

def process_images(directory):
	pass

def split_data(train_ratio, labels_arr):
	mask_train = np.array([False for i in range(len(labels_arr))])
	mask_test = np.array([False for i in range(len(labels_arr))])
	train_val = 0
	test_val = 0
	for i in range(len(labels_arr)):
		if np.isnan(labels_arr[i]):
			pass
		else:
			rand_num = np.random.random()
			if rand_num < train_ratio:
				mask_train[i] = True
				train_val += 1
			else:
				mask_test[i] = True
				test_val += 1
	mask = {"train":mask_train, "test": mask_test}
	print train_val, test_val, train_val+test_val
	return mask