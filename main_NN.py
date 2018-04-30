import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models import *
from torch.autograd import Variable

from dataset_reader import *

np.random.seed(0)
# Uncomment the two lines below if this is the first time you are running
labels_directory = "data/labels"
# preprocess_data(labels_directory)

train_ratio = 0.8
num_epochs = 10
labels_arr, features_arr = process_mats(labels_directory)
mask = split_data(train_ratio, labels_arr)

model = UNet(feature_size = 3)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
rand_input = Variable(torch.randn(32,3,64,64))
rand_features = Variable(torch.randn(32,3))
rand_labels = Variable(torch.ones(32).long())
for i in range(100):
	optimizer.zero_grad()
	out1, out2 = model(rand_input,rand_features)
	loss1 = criterion1(out1,rand_labels)
	loss2 = criterion2(out2,rand_input)
	loss = loss1+loss2
	loss.backward()
	print "loss", loss1,loss2
	optimizer.step()

# # print labels_arr, features_arr
# X_train = features_arr[mask["train"]]
# y_train = labels_arr[mask["train"]]
# X_test = features_arr[mask["test"]]
# y_test = labels_arr[mask["test"]]

# print X_train.shape, X_test.shape

# print(confusion_matrix(y_test, y_pred))  
# print(classification_report(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))  

def train(args):
	
	# Dataset loading and preparing env
	all_datasets = dataReader(args)
	train_data = all_datasets['train']
	validation_data = all_datasets['validation']
	logger = ""
	best_train_loss = float("Inf")
	best_validation_loss = float("Inf")
	best_epoch = -1

	# instantiate model and optimizer
	model = load_model(args)
	model.train()
	if HAVE_CUDA:
		model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = load_optimizer(args,model)
	
	# log info
	prepare_env(args)
	train_start_time = time.time()
	epoch_start_time = train_start_time
	for i in range(args.num_epochs):
		
		# run model
		print "==========================================================================================="
		print "Epoch(",i,"/",args.num_epochs,")"
		model, optimizer, present_train_loss, train_accuracy = train_epoch(args,train_data,model,optimizer,criterion)
		present_validation_loss, validation_accuracy = get_validation_loss(args,validation_data,model,criterion)
		save_data(args,model,optimizer)
		if present_validation_loss<best_validation_loss:
			save_data(args,model,optimizer,True)
			best_epoch = i
			best_validation_loss = present_validation_loss
		##########################################Logging info#############################################
		end_time = time.time()
		epoch_duration = end_time-epoch_start_time
		epoch_start_time = end_time
		print "Duration: ", epoch_duration
		print ""
		print "Loss:"
		print "Training  ", present_train_loss
		print "Validation", present_validation_loss
		print ""
		print "Accuracy:"
		print "Training  ", train_accuracy
		print "Validation", validation_accuracy
		print ""
		print "Best Epoch", best_epoch
		###################################################################################################
		# break

	total_training_time = time.time()-train_start_time
	print "Total training time: ", total_training_time

def train_epoch(iterator, model, optimizer, criterion, save_every=100):
	model.train()
	total_training_loss = 0.0
	num_batches = 0
	accuracy = 0
	for batch in tqdm(iterator):
		if HAVE_CUDA:
			batch = batch.cuda()
		optimizer.zero_grad()

		# Forward pass
		batch_data = ag.Variable(batch[0].float())
		batch_labels = ag.Variable(batch[1].long())
		pred_labels = model(batch_data)
		# print torch.min(batch_labels.view(-1))
		#Backward pass
		loss = criterion(pred_labels,batch_labels)
		loss.backward()
		total_training_loss += loss.data.cpu().numpy()
		# print total_training_loss
		temp_accuracy = get_metrics(pred_labels,batch_labels)
		accuracy += temp_accuracy

		#Optimize
		optimizer.step()
		num_batches += 1

		#Log info
		if (num_batches)%args.save_every == 0:
			save_data(args,model,optimizer)
		print "Loss:", loss.data[0], " Accuracy:", temp_accuracy

	total_training_loss /= num_batches
	accuracy /= num_batches

	return model, optimizer, total_training_loss, accuracy