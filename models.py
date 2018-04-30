import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class UNet(nn.Module):
	"""docstring for JointNet"""
	def __init__(self, k_size = 3, p_size = 2, num_classes=7, pad_type="zero",encoder_only = False, feature_size = None):
		"""
		the inputs should be 128x128
		"""
		super(UNet, self).__init__()
		print("mine")
		self.kernel_size = k_size
		self.pool_size = p_size
		self.num_classes = num_classes
		self.encoder_only = encoder_only
		self.feature_size = feature_size
		if pad_type == "reflect":
			pad_layer = nn.ReflectionPad2d((k_size-1)/2)
		else:
			pad_layer = nn.ZeroPad2d((k_size-1)/2)
		
		# Parts of encoder
		self.seq1 = nn.Sequential(
		nn.Conv2d(in_channels=3, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),

		nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),
		)

		self.seq2 = nn.Sequential(
		nn.Conv2d(in_channels=64, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),

		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),
		)

		self.seq3 = nn.Sequential(
		nn.Conv2d(in_channels=128, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),

		nn.Conv2d(in_channels=256, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		)

		self.seq4 = nn.Sequential(
		nn.Conv2d(in_channels=256, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),

		nn.Conv2d(in_channels=512, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),
		)

		self.bot = nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(1024),
		nn.ReLU(True),

		nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(1024),
		nn.ReLU(True),
		)

		self.upseq1 = nn.Sequential(
		nn.Conv2d(in_channels=128, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),

		nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),
		)

		self.upseq2 = nn.Sequential(
		nn.Conv2d(in_channels=256, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),

		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(128),
		nn.ReLU(True),
		)

		self.upseq3 = nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),

		nn.Conv2d(in_channels=256, out_channels=256, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(256),
		nn.ReLU(True),
		)

		self.upseq4 = nn.Sequential(
		nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),

		nn.Conv2d(in_channels=512, out_channels=512, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(512),
		nn.ReLU(True),
		)

		self.out_classify = nn.Sequential(
		nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(64),
		nn.ReLU(True),

		nn.Conv2d(in_channels=64, out_channels=1, kernel_size=k_size, stride=1),
		pad_layer,
		nn.BatchNorm2d(1),
		nn.ReLU(True),
		)

		if feature_size:
			self.out_classify_linear = nn.Sequential(
			nn.Linear(16+feature_size,16),
			nn.ReLU(True),
			nn.Linear(16,num_classes)
			)
		else:
			self.out_classify_linear = nn.Sequential(
			nn.Linear(16,16),
			nn.ReLU(True),
			nn.Linear(16,num_classes)

			# nn.Softmax()
			)

		self.up_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=k_size-1, stride=2)
		self.up_conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=k_size-1, stride=2)
		self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=k_size-1, stride=2)
		self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=k_size-1, stride=2)

		self.max_pool = nn.MaxPool2d(kernel_size=p_size, stride=p_size)

		self.final_layer = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)


		

	def forward(self, images,features = None):
		N = images.size()[0]
		out1 = self.seq1(images)
		# print out1
		out2 = self.max_pool(out1)
		out2 = self.seq2(out2)
		out3 = self.max_pool(out2)
		out3 = self.seq3(out3)
		out4 = self.max_pool(out3)
		out4 = self.seq4(out4)
		out_bot = self.max_pool(out4)
		out_bot = self.bot(out_bot)
		out_class = self.out_classify(out_bot)
		out_class = out_class.view(N,-1)
		if self.feature_size:
			out_class = torch.cat((out_class,features),1)
		out_class = self.out_classify_linear(out_class)
		if not self.encoder_only:
			in_down_4 = self.up_conv4(out_bot)
			# print(out4,out_bot,in_down_4)
			in_4 = torch.cat((out4,in_down_4),1)
			in_4 = self.upseq4(in_4)
			in_down_3 = self.up_conv3(out4)
			in_3 = torch.cat((out3,in_down_3),1)
			in_3 = self.upseq3(in_3)
			in_down_2 = self.up_conv2(out3)
			in_2 = torch.cat((out2,in_down_2),1)
			in_2 = self.upseq2(in_2)
			in_down_1 = self.up_conv1(out2)
			in_1 = torch.cat((out1,in_down_1),1)																																																																													
			in_1 = self.upseq1(in_1)
			final_out = self.final_layer(in_1)
		
			return out_class, final_out
		return out_class