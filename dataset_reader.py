import torchvision
from torch.utils import data
from PIL import Image
import torch

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
	mask_nan  = np.array([False for i in range(len(labels_arr))])
	train_val = 0
	test_val = 0
	nan_val = 0
	for i in range(len(labels_arr)):
		if np.isnan(labels_arr[i]):
			mask_nan[i] = True
			nan_val += 1
		else:
			rand_num = np.random.random()
			if rand_num < train_ratio:
				mask_train[i] = True
				train_val += 1
			else:
				mask_test[i] = True
				test_val += 1
	mask = {"train":mask_train, "test": mask_test,"nan":mask_nan}
	print train_val, test_val, train_val+test_val,nan_val,train_val+test_val+nan_val
	return mask

def split_image_data(labels_arr,image_arr, mask,features_arr):
	image_arr = np.array(image_arr)
	labels_arr = np.array(labels_arr)
	X_train = image_arr[mask["train"]]
	y_train = labels_arr[mask["train"]]
	X_test = image_arr[mask["test"]]
	y_test = labels_arr[mask["test"]]
	X_nan = image_arr[mask["nan"]]
	y_nan = labels_arr[mask["nan"]]
	f_train = features_arr[mask["train"]]
	f_test = features_arr[mask["test"]]
	f_nan = features_arr[mask["nan"]]
	return y_train,X_train,y_test,X_test,y_nan,X_nan, f_train, f_test, f_nan

class imageLoader(data.Dataset):
	def __init__(self,labelIds,imageIds,f_Ids,img_size=(512,1024)):
		self.labelIds = labelIds
		self.imageIds = imageIds
		self.img_size = img_size
		self.f_Ids = f_Ids
		print("Found %d  images" % (len(self.labelIds)))
	

	def __len__(self):
		return len(self.labelIds)

	def __getitem__(self, index):
		img_path = self.imageIds[index].rstrip()
		lbl = self.labelIds[index]
		if np.isnan(lbl):
			lbl = 0
		f_present = self.f_Ids[index]
		img = Image.open(img_path)
		img = img.resize(self.img_size, Image.NEAREST)
		img = np.transpose(np.array(img,dtype=np.uint8),[2,1,0])
		return torch.from_numpy(img), torch.LongTensor([int(lbl)]), torch.from_numpy(f_present).float()
 
def dataReader(args,labels_arr,label_mask,features_arr):
	path=args.data_dir
	img_size = (args.image_size,args.image_size)
	trainLoaderDict = {}


	images_base = os.path.join(path,'images')
	imageIds = getImagesList(images_base)
	imageIds.sort()

	img_data = split_image_data(labels_arr,imageIds,label_mask,features_arr)
	train_labels, train_images, test_labels, test_images, nan_labels, nan_images, f_train, f_test, f_nan = img_data
	train_set = imageLoader(train_labels, train_images, f_train, img_size=img_size)
	test_set = imageLoader(test_labels, test_images, f_test, img_size=img_size)
	nan_set = imageLoader(nan_labels, nan_images, f_nan, img_size=img_size)


	trainLoaderDict['train'] = data.DataLoader(train_set,batch_size=args.batch_size,shuffle=args.shuffle)
	trainLoaderDict['test'] = data.DataLoader(test_set,batch_size=args.batch_size,shuffle=args.shuffle)
	trainLoaderDict['nan'] = data.DataLoader(nan_set,batch_size=args.batch_size,shuffle=args.shuffle)
	return trainLoaderDict