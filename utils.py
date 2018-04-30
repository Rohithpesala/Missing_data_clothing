import scipy.io as sio
import numpy as np
import os
import shutil

def load_labels(directory):
	"""
	Loads the category mat file to obtain labels for the classification task
	"""
	label_arr = None
	total_files = 0
	for filename in os.listdir(directory):
	    if filename.endswith(".mat"): 
	        if filename.startswith("category"):
	            #use this as labels vector
	            fname = os.path.join(os.getcwd(),directory,filename)
	            m = sio.loadmat(fname)
	            label_arr = np.array(m['GT'])
	            label_arr = np.squeeze(label_arr)
	        else:
	            total_files += 1
	return label_arr, total_files

def load_features(directory, num_files, num_instances):
	"""
	Loads all the mat files other than category.mat as features for the particular product
	"""
	f_arr = np.zeros([num_files,num_instances])
	count = 0
	for filename in os.listdir(directory):
	    if filename.endswith(".mat"): 
	        if filename.startswith("category"):
	            continue
	        else:
	            #add this to feature vector
	            fname = os.path.join(os.getcwd(),directory,filename)
	            m = sio.loadmat(fname)
	            arr = np.array(m['GT'])
	            f_arr[count] = np.squeeze(arr)
	            count+=1
	return np.transpose(f_arr)

def process_features_multinomial(arr):
	"""
	Replaces nans by sampling from a multinomial distribution
	"""
	arr = np.transpose(arr)
	for i in range(arr.shape[0]):
		np.place(arr[i], np.isnan(arr[i]), 0)
		freq = np.bincount(arr[i].astype(int))
		np.place(arr[i], arr[i]==0, random_multinomial(freq[1:],freq[0]))

def process_features_constant(arr):
	"""
	Replaces nans by sampling from a multinomial distribution
	"""
	for i in range(arr.shape[0]):
		np.place(arr[i], np.isnan(arr[i]), 1)

def random_multinomial(freq_arr,out_len):
	"""
	Returns a random integer sampled from the MLE of the multinomial distribution
	"""
	freq_arr = freq_arr.astype(float)/np.sum(freq_arr)
	out_arr = [0 for i in range(out_len)]
	for j in range(out_len):
	    rand_num = np.random.random()
	    cdf_sum = 0.0
	    for i in range(len(freq_arr)):
	        cdf_sum += freq_arr[i]
	        if rand_num < cdf_sum:
	            out_arr[j] = i+1
	            break
	return out_arr

def split_nan_images(directory, label_arr):
	"""
	Splits images that don't have labels into different directory as they are not useful
	"""
	labeled_dir = os.path.join(os.getcwd(),directory,"labeled")
	unlabeled_dir = os.path.join(os.getcwd(),directory,"unlabeled")
	images_dir = os.path.join(os.getcwd(),directory,"images")
	if not os.path.exists(labeled_dir):
	    os.makedirs(labeled_dir)
	if not os.path.exists(unlabeled_dir):
	    os.makedirs(unlabeled_dir)
	for filename in os.listdir(images_dir):
		f_id = int(filename[0:6])	
		filepath = os.path.join(images_dir,filename)
		if np.isnan(label_arr[f_id-1]):
			shutil.copy(filepath,unlabeled_dir)
		else:
			shutil.copy(filepath,labeled_dir)
		print f_id,label_arr[f_id-1]
	for i in range(len(label_arr)):
		if np.isnan(label_arr[i]):
			pass
		else:
			pass
	pass