import scipy.io as sio
import numpy as np
import os

def load_labels(directory):
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

def random_multinomial(freq_arr):
	freq_arr = freq_arr.astype(float)/np.sum(freq_arr)
	rand_num = np.random.random()
	cdf_sum = 0.0
	print rand_num
	for i in range(len(freq_arr)):
	    cdf_sum += freq_arr[i]
	    print cdf_sum
	    if rand_num < cdf_sum:
	        return i+1