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

def process_features_multinomial(arr):
	for i in range(arr.shape[0]):
		np.place(arr[i], np.isnan(arr[i]), 0)
		freq = np.bincount(arr[i].astype(int))
		np.place(arr[i], arr[i]==0, random_multinomial(freq[1:],freq[0]))

def random_multinomial(freq_arr,out_len):
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