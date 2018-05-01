import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models import *
import argparse
from torch.autograd import Variable
import torch.autograd as ag
from tqdm import tqdm
import time

from dataset_reader import *

HAVE_CUDA = torch.cuda.is_available()

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    data_type = "gta5"
    mode = "train"
    output_dir = "outputs/"
    num_classes = 2
    parser.add_argument('-d', "--data_type", default=data_type)
    parser.add_argument('-m', "--mode", default=mode)
    parser.add_argument('-o', "--output_dir", default=output_dir)
    parser.add_argument('-n', "--num_classes", default=num_classes, type=int)
    parser.add_argument('-p', "--pool_size", default=2, type=int)
    parser.add_argument('-k', "--kernel_size", default=3, type=int)
    parser.add_argument('-t', "--pad_type", default="reflect")
    parser.add_argument('-e', "--num_epochs", default=10, type=int)
    parser.add_argument('-b', "--batch_size", default=4, type=int)
    parser.add_argument('-l', "--model_type", default="unet")
    parser.add_argument('-r', "--train_ratio", default=0.8, type=float)
    parser.add_argument('-s', "--save_every", default=100, type=int)
    parser.add_argument('-i', "--run_id", default="00")
    parser.add_argument('-a', "--data_dir", default="data")
    parser.add_argument('-w', "--image_size", default=64, type=int)
    parser.add_argument('-f', "--shuffle", default=True, type=bool)
    parser.add_argument('-z', "--use_unlabeled", default=False, type=bool)

    return parser.parse_args()




def train(args):
	
	# Dataset loading and preparing env
	labels_directory = "data/labels"
	# preprocess_data(labels_directory)
	labels_arr, features_arr = process_mats(labels_directory)
	mask = split_data(args.train_ratio, labels_arr)
	all_datasets = dataReader(args,labels_arr,mask,features_arr)
	train_data = all_datasets['train']
	test_data = all_datasets['test']
	nan_data = all_datasets['nan']
	logger = ""
	best_train_loss = float("Inf")
	best_test_loss = float("Inf")
	best_epoch = -1

	# instantiate model and optimizer
	model = UNet(feature_size = 25,image_size = args.image_size)
	model.train()
	if HAVE_CUDA:
		model = model.cuda()
	optimizer = optim.Adam(model.parameters(),lr=0.01)
	
	# log info
	prepare_env(args)
	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		
		# run model
		print "==========================================================================================="
		print "Epoch(",i,"/",args.num_epochs,")"
		present_train_loss,train_accuracy = train_epoch(args,train_data,nan_data,model,optimizer)
		present_test_loss, test_accuracy = get_validation_loss(args,test_data,model)
		save_data(args,model,optimizer)
		if present_test_loss<best_test_loss:
			save_data(args,model,optimizer,True)
			best_epoch = i
			best_test_loss = present_test_loss
		##########################################Logging info#############################################
		end_time = time.time()
		epoch_duration = end_time-epoch_start_time
		epoch_start_time = end_time
		print "Duration: ", epoch_duration
		print ""
		print "Loss:"
		print "Training  ", present_train_loss
		print "test", present_test_loss
		print ""
		print "Accuracy:"
		print "Training  ", train_accuracy
		print "Test", test_accuracy
		print ""
		print "Best Epoch", best_epoch
		###################################################################################################

	total_training_time = time.time()-train_start_time
	print "Total training time: ", total_training_time

def get_validation_loss(args,iterator,model):
	model.eval()
	total_validation_loss = 0.0
	num_batches = 0
	accuracy = 0
	for batch in tqdm(iterator):
		if HAVE_CUDA:
			batch[0] = batch[0].cuda()
			batch[1] = batch[1].cuda()
			batch[2] = batch[2].cuda()
		# Forward pass
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(np.squeeze(batch[1]-1).long())
		batch_features = ag.Variable(batch[2])
		loss,pred_labels = model(batch_data,batch_labels,batch_features,labeled=True)
		total_validation_loss += loss.data.cpu().numpy()

		temp_accuracy = get_metrics(pred_labels,batch_labels)
		accuracy += temp_accuracy
		num_batches += 1

	total_validation_loss /= num_batches
	accuracy /= num_batches

	return total_validation_loss, accuracy

def train_epoch(args,labeled_iterator, unlabeled_iterator, model, optimizer):
	model.train()
	total_training_loss = 0.0
	num_batches = 0
	accuracy = 0
	for batch in tqdm(labeled_iterator):
		if HAVE_CUDA:
			batch[0] = batch[0].cuda()
			batch[1] = batch[1].cuda()
			batch[2] = batch[2].cuda()
		optimizer.zero_grad()
		# Forward pass
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(np.squeeze(batch[1]-1).long())
		batch_features = ag.Variable(batch[2])
		loss,pred_labels = model(batch_data,batch_labels,batch_features,labeled=True)
		loss.backward()
		total_training_loss += loss.data.cpu().numpy()

		temp_accuracy = get_metrics(pred_labels,batch_labels)
		accuracy += temp_accuracy

		#Optimize
		optimizer.step()
		num_batches += 1

		#Log info
		if (num_batches)%args.save_every == 0:
			save_data(args,model,optimizer)
		print "Loss:", loss.data.cpu().numpy(), "Accuracy", temp_accuracy

	accuracy /= num_batches
	if args.use_unlabeled:
		for batch in tqdm(unlabeled_iterator):
			if HAVE_CUDA:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
			optimizer.zero_grad()
			# Forward pass
			batch_data = ag.Variable(batch[0].float())
			batch_labels = ag.Variable(np.squeeze(batch[1]-1).long())
			batch_features = ag.Variable(batch[2])
			loss,pred_labels = model(batch_data,batch_labels,batch_features,labeled=False)
			loss.backward()
			total_training_loss += loss.data.cpu().numpy()

			#Optimize
			optimizer.step()
			num_batches += 1

			#Log info
			if (num_batches)%args.save_every == 0:
				save_data(args,model,optimizer)
			print "Loss:", loss.data.cpu().numpy(), "Accuracy", temp_accuracy
		total_training_loss /= num_batches

	return total_training_loss, accuracy

def main():
	args = get_args()
	
	np.random.seed(0)
	# Uncomment the two lines below if this is the first time you are running
	# labels_directory = "data/labels"
	# # preprocess_data(labels_directory)
	# train_ratio = 0.8
	# num_epochs = 10
	# labels_arr, features_arr = process_mats(labels_directory)
	# mask = split_data(train_ratio, labels_arr)
	# all_datasets = dataReader(args,labels_arr,mask,features_arr)
	# # print type(all_datasets['train'])
	# # for ele in tqdm(all_datasets['train']):
	# # 	# print ele
	# # 	print "yo"
	# # 	# break
	# # return
	# train_data = all_datasets['train']
	# nan_data = all_datasets['nan']
	# model = UNet(feature_size = 25,image_size = args.image_size)
	# optimizer = optim.Adam(model.parameters(),lr=0.01)
	# loss = train_epoch(train_data,nan_data,model,optimizer)

	# return


	# criterion1 = nn.CrossEntropyLoss()
	# criterion2 = nn.MSELoss()
	# rand_input = Variable(torch.randn(32,3,args.image_size,args.image_size))
	# rand_features = Variable(torch.randn(32,3))
	# rand_labels = Variable(torch.ones(32).long())
	# for i in range(100):
	# 	optimizer.zero_grad()
	# 	out1, out2 = model(rand_input,rand_features)
	# 	loss1 = criterion1(out1,rand_labels)
	# 	loss2 = criterion2(out2,rand_input)
	# 	loss = loss1+loss2
	# 	loss.backward()
	# 	print "loss", loss1,loss2
	# 	optimizer.step()

	# # print labels_arr, features_arr
	# X_train = features_arr[mask["train"]]
	# y_train = labels_arr[mask["train"]]
	# X_test = features_arr[mask["test"]]
	# y_test = labels_arr[mask["test"]]

	# print X_train.shape, X_test.shape

	# print(confusion_matrix(y_test, y_pred))  
	# print(classification_report(y_test, y_pred))
	# print(accuracy_score(y_test, y_pred))  

	if args.mode == "validation":
		validation(args)
	elif args.mode == "test":
		test(args)
	elif args.mode == "train":
		train(args)
	else:
		dtrain2(args)

if __name__ == "__main__":
	main()
