from __future__ import print_function
from __future__ import (absolute_import, unicode_literals, print_function)
from __future__ import division, print_function, absolute_import
import math
# import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))
# from sys import exit
# from sys.path import path.insert
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import numpy as np
from sklearn import tree
# import pandas as pd
from m_open_extension import *
from m_det_features import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import RobustScaler  
from sklearn.preprocessing import minmax_scale

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.neural_network import BernoulliRBM
from pykalman import UnscentedKalmanFilter
from pykalman import KalmanFilter
from pykalman import AdditiveUnscentedKalmanFilter

from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import Pipeline
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.5, 'n_pre':0.2, 'm_post':0.1, 'alpha_pro':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers_pro':[10], 'solver_pro':'adam', 'rs_pro':1, 'activation_pro':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature2':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature_array':['RMS'], 'layers_fus':[10], 'rs_fus':1, 'alpha_fus':1, 'solver_fus':'lbfgs', 'activation_fus':'identity', 'source_file':'OFF'}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	if config['mode'] == 'mode1_depe':

		print('Select MASTER Features xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		length_train = int(config['train']*len(rownames))
		length_test = length_data - length_train
		
		mydict = mydict.to_dict(orient='list')		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])
		
		Features_Train = []
		for i in range(length_train):
			example = []
			for feature in config['feature_array']:
				example.append(newdict[feature][i])
			Features_Train.append(example)		
		Features_Train = np.array(Features_Train)
		
		Features_Test = []
		for i in range(length_test):
			example = []
			for feature in config['feature_array']:
				example.append(newdict[feature][i+length_train])
			Features_Test.append(example)		
		Features_Test = np.array(Features_Test)
		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:
				example.append(newdict[feature][i])
			Features.append(example)		
		Features = np.array(Features)
		
		print(len(Features))
		print(len(Features_Test))
		print(len(Features_Train))
		

		
		# plt.plot(Features_Train)
		# plt.show()
		# sys.exit()
		
		
		scaler_model = StandardScaler()
		scaler_model.fit(Features_Train)
		Features_Train = scaler_model.transform(Features_Train)
		Features_Test = scaler_model.transform(Features_Test)
		
		scaler_model_full = StandardScaler()
		scaler_model_full.fit(Features)
		Features = scaler_model_full.transform(Features)
		
		
		# plt.plot(Features)
		# plt.show()
		

		nn_fus = MLPRegressor(hidden_layer_sizes=config['layers_fus'], activation=config['activation_fus'], solver=config['solver_fus'], alpha=config['alpha_fus'], random_state=config['rs_fus'])
		
		nn_fus.fit(X=Features_Train, y=np.linspace(0, 1, length_train))		
		fused_train = nn_fus.predict(Features_Train)
		fused_test = nn_fus.predict(Features_Test)
		
		
		nn_fus_full = MLPRegressor(hidden_layer_sizes=config['layers_fus'], activation=config['activation_fus'], solver=config['solver_fus'], alpha=config['alpha_fus'], random_state=config['rs_fus'])
		nn_fus_full.fit(X=Features, y=np.linspace(0, 1, length_data))				
		fused = nn_fus_full.predict(Features)
		
		
		
		
		nn_pro = MLPRegressor(hidden_layer_sizes=config['layers_pro'], activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=config['rs_pro'])
		
		
		
		n_pre = int(config['n_pre']*length_train)
		m_post = int(config['m_post']*length_train)
		n_ex = length_train + 1 - n_pre - m_post
		
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex):
			T_Inputs.append(fused_train[k : k + n_pre])
			T_Outputs.append(fused_train[k + n_pre : k + n_pre + m_post])
			
		nn_pro.fit(T_Inputs, T_Outputs)
		print(T_Inputs)
		
		
		
		
		
		fused_predict = []
		it_fused = list(fused_train)

		for k in range(length_test + m_post - 1):
			P_Input = it_fused[n_ex + k + 1 : n_ex + n_pre + k + 1]
			# print(P_Input)
			# sys.exit()
			P_Output = nn_pro.predict([P_Input])
			P_Output = P_Output[0]			
			
			fused_predict.append(P_Output[-1])
			it_fused.append(P_Output[-1])

		fused_predict = np.array(fused_predict[:-(m_post-1)])
		
		# plt.plot(fused_predict, 'r', fused_test, 'b')
		
		x_full = np.arange((len(fused)))
		x_train = np.arange((len(fused_train)))	
		x_predict = np.linspace(len(fused_train), len(fused), num=len(fused_test), endpoint=False)
		
		
		plt.plot(x_full, fused, 'b', x_predict, fused_predict, 'r', x_train, fused_train, 'k')
		plt.show()
		
		error = 0
		for i in range(len(fused_predict)):
			error += (fused_predict[i] - fused[length_train + i])**2.0
		error_final = np.absolute(fused_predict[length_test - 1] - fused[length_data - 1])
		print('error= ', error)
		print('error_final= ', error_final)
	
	elif config['mode'] == 'mode2':
		
		if config['source_file'] == 'OFF':
			print('Select MASTER Features xls')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = 'C:\\Felix\\29_THESIS\\Analysis\\LAST_MASTER_AE_Features.xlsx'
		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		length_train = int(config['train']*len(rownames))
		length_test = length_data - length_train
		
		mydict = mydict.to_dict(orient='list')		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])
		
		
		
		Features_Train = []
		for i in range(length_train):
			example = newdict[config['feature']][i]
			Features_Train.append(example)		
		Features_Train = np.array(Features_Train)
		
		Features_Test = []
		for i in range(length_test):
			example = newdict[config['feature']][i+length_train]
			Features_Test.append(example)		
		Features_Test = np.array(Features_Test)
		
		Features = []
		for i in range(length_data):
			example = newdict[config['feature']][i]
			Features.append(example)		
		Features = np.array(Features)
		
		print(len(Features))
		print(len(Features_Test))
		print(len(Features_Train))

		
		
		# scaler_model = StandardScaler()
		# scaler_model.fit(Features_Train)
		# Features_Train = scaler_model.transform(Features_Train)
		# Features_Test = scaler_model.transform(Features_Test)
		
		# scaler_model_full = StandardScaler()
		# scaler_model_full.fit(Features)
		# Features = scaler_model_full.transform(Features)
		
		
		# plt.plot(Features)
		# plt.show()
		

		# nn_fus = MLPRegressor(hidden_layer_sizes=config['layers_fus'], activation=config['activation_fus'], solver=config['solver_fus'], alpha=config['alpha_fus'], random_state=config['rs_fus'])
		
		# nn_fus.fit(X=Features_Train, y=np.linspace(0, 1, length_train))		
		# fused_train = nn_fus.predict(Features_Train)
		# fused_test = nn_fus.predict(Features_Test)
		
		
		# nn_fus_full = MLPRegressor(hidden_layer_sizes=config['layers_fus'], activation=config['activation_fus'], solver=config['solver_fus'], alpha=config['alpha_fus'], random_state=config['rs_fus'])
		# nn_fus_full.fit(X=Features, y=np.linspace(0, 1, length_data))				
		# fused = nn_fus_full.predict(Features)
		
		
		fused_train = np.ravel(Features_Train)
		fused_test = np.ravel(Features_Test)
		fused = np.ravel(Features)


		n_pre = int(config['n_pre']*length_train)
		m_post = int(config['m_post']*length_train)
		n_ex = length_train + 1 - n_pre - m_post
		
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		# a = input('enter to continue...')
		
		if config['layers_pro'][0] == 0:
			print('Auto config of layers')
			List_Layers = list(np.arange(m_post, n_pre, int((n_pre-m_post)*0.1)))
			# List_Layers = [50, 100]
		else:
			print('Auto config of layers IS not optional')
			sys.exit()
			
		
		ERROR = []
		ERROR_FINAL = []
		# print(List_Layers)
		# print(int((n_pre-m_post)*0.1))
		# print(np.arange(n_pre, m_post, int((n_pre-m_post)*0.1)))
		# sys.exit()
		for layers in List_Layers:
			
			nn_pro = MLPRegressor(hidden_layer_sizes=layers, activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=config['rs_pro'])
			
			T_Inputs = []
			T_Outputs = []
			for k in range(n_ex):
				T_Inputs.append(fused_train[k : k + n_pre])
				T_Outputs.append(fused_train[k + n_pre : k + n_pre + m_post])
				
			nn_pro.fit(T_Inputs, T_Outputs)
			print(T_Inputs)
			

			
			fused_predict = []
			it_fused = list(fused_train)

			for k in range(length_test + m_post - 1):
				P_Input = it_fused[n_ex + k + 1 : n_ex + n_pre + k + 1]
				# print(P_Input)
				# sys.exit()
				P_Output = nn_pro.predict([P_Input])
				P_Output = P_Output[0]			
				
				fused_predict.append(P_Output[-1])
				it_fused.append(P_Output[-1])

			fused_predict = np.array(fused_predict[:-(m_post-1)])
			
			# plt.plot(fused_predict, 'r', fused_test, 'b')
			
			x_full = np.arange((len(fused)))
			x_train = np.arange((len(fused_train)))	
			x_predict = np.linspace(len(fused_train), len(fused), num=len(fused_test), endpoint=False)
			
			
			# plt.plot(x_full, fused, 'b', x_predict, fused_predict, 'r', x_train, fused_train, 'k')
			# plt.show()
			
			error = 0
			for i in range(len(fused_predict)):
				error += (fused_predict[i] - fused[length_train + i])**2.0
			error_final = fused_predict[length_test - 1] - fused[length_data - 1]
			print('error= ', error)
			print('error_final= ', error_final)
			ERROR.append(error)
			ERROR_FINAL.append(error_final)
		
		
		mydict = {'Error_Final':ERROR_FINAL, 'Error':ERROR}
		writer = pd.ExcelWriter(config['name'] + '.xlsx')			
		DataFr = pd.DataFrame(data=mydict, index=List_Layers)
		DataFr.to_excel(writer, sheet_name='Result')
		writer.close()
		
		
		mydict = {'alpha':config['alpha_pro'], 'solver':config['solver_pro'], 'activation':config['activation_pro'], 'rs':config['rs_pro'], 'n_pre':config['n_pre'], 'm_post':config['m_post'], 'n_mov_avg':config['n_mov_avg'], 'train':config['train'], 'feature':config['feature']}
		writer = pd.ExcelWriter('config_' + config['name'] + '.xlsx')			
		DataFr = pd.DataFrame(data=mydict, index=['value'])
		DataFr.to_excel(writer, sheet_name='Result')
		writer.close()
	
	elif config['mode'] == 'mode3':
		

		__all__ = ['AutoEncoder', 'Layer']

		import time
		import logging
		import itertools

		log = logging.getLogger('sknn')


		import sklearn

		from sknn import nn, backend


		class Layer(nn.Layer):
			"""
			Specification for a layer to be passed to the auto-encoder during construction.  This
			includes a variety of parameters to configure each layer based on its activation type.
			Parameters
			----------
			activation: str
				Select which activation function this layer should use, as a string.  Specifically,
				options are ``Sigmoid`` and ``Tanh`` only for such auto-encoders.
			type: str, optional
				The type of encoding and decoding layer to use, specifically ``denoising`` for randomly
				corrupting data, and a more traditional ``autoencoder`` which is used by default.
			name: str, optional
				You optionally can specify a name for this layer, and its parameters
				will then be accessible to scikit-learn via a nested sub-object.  For example,
				if name is set to ``layer1``, then the parameter ``layer1__units`` from the network
				is bound to this layer's ``units`` variable.
				The name defaults to ``hiddenN`` where N is the integer index of that layer, and the
				final layer is always ``output`` without an index.
			units: int
				The number of units (also known as neurons) in this layer.  This applies to all
				layer types except for convolution.
			cost: string, optional
				What type of cost function to use during the layerwise pre-training.  This can be either
				``msre`` for mean-squared reconstruction error (default), and ``mbce`` for mean binary
				cross entropy.
			tied_weights: bool, optional
				Whether to use the same weights for the encoding and decoding phases of the simulation
				and training.  Default is ``True``.
			corruption_level: float, optional
				The ratio of inputs to corrupt in this layer; ``0.25`` means that 25% of the inputs will be
				corrupted during the training.  The default is ``0.5``.
			warning: None
				You should use keyword arguments after `type` when initializing this object. If not,
				the code will raise an AssertionError.
			"""

			def __init__(self,
						 activation,
						 warning=None,
						 type='autoencoder',
						 name=None,
						 units=None,
						 cost='msre',
						 tied_weights=True,
						 corruption_level=0.5):

				assert warning is None, \
					"Specify layer parameters as keyword arguments, not positional arguments."

				if type not in ['denoising', 'autoencoder']:
					raise NotImplementedError("AutoEncoder layer type `%s` is not implemented." % type)
				if cost not in ['msre', 'mbce']:
					raise NotImplementedError("Error type '%s' is not implemented." % cost)
				if activation not in ['Sigmoid', 'Tanh']:
					raise NotImplementedError("Activation type '%s' is not implemented." % activation)

				self.activation = activation
				self.type = type
				self.name = name
				self.units = units
				self.cost = cost
				self.tied_weights = tied_weights
				self.corruption_level = corruption_level


		class AutoEncoder(nn.NeuralNetwork, sklearn.base.TransformerMixin):

			def _setup(self):
				assert not self.is_initialized,\
					"This auto-encoder has already been initialized."

				backend.setup()
				self._backend = backend.AutoEncoderBackend(self)

			def fit(self, X):
				"""Fit the auto-encoder to the given data using layerwise training.
				Parameters
				----------
				X : array-like, shape (n_samples, n_inputs)
					Training vectors as real numbers, where ``n_samples`` is the number of
					samples and ``n_inputs`` is the number of input features.
				Returns
				-------
				self : object
					Returns this instance.
				"""
				num_samples, data_size = X.shape[0], X.size

				log.info("Training on dataset of {:,} samples with {:,} total size.".format(num_samples, data_size))
				if self.n_iter:
					log.debug("  - Terminating loop after {} total iterations.".format(self.n_iter))
				if self.n_stable:
					log.debug("  - Early termination after {} stable iterations.".format(self.n_stable))

				if self.verbose:
					log.debug("\nEpoch    Validation Error        Time"
							  "\n-------------------------------------")
				
				self._backend._fit_impl(X)
				return self

			def transform(self, X):
				"""Encode the data via the neural network, as an upward pass simulation to
				generate high-level features from the low-level inputs.
				Parameters
				----------
				X : array-like, shape (n_samples, n_inputs)
					Input data to be passed through the auto-encoder upward.
				Returns
				-------
				y : numpy array, shape (n_samples, n_features)
					Transformed output array from the auto-encoder.
				"""
				return self._backend._transform_impl(X)

			def transfer(self, nn):
				assert not nn.is_initialized,\
					"Target multi-layer perceptron has already been initialized."

				for a, l in zip(self.layers, nn.layers):
					assert a.activation == l.type,\
						"Mismatch in activation types in target MLP; expected `%s` but found `%s`."\
						% (a.activation, l.type)
					assert a.units == l.units,\
						"Different number of units in target MLP; expected `%i` but found `%i`."\
						% (a.units, l.units)
			   
				self._backend._transfer_impl(nn)
		
		print('hola')
		
		from sknn.ae import AutoEncoder , Layer
		ae = AutoEncoder(layers=[Layer("Sigmoid", units=8)])
		X =[[1,1], [1,0], [0,0]]
		ae.fit(X)
	
	elif config['mode'] == 'mode4':
		import torch.nn as nn
		import torch 
		import torchvision.datasets as dsets
		import torchvision.transforms as transforms
		import torchvision
		from torch.autograd import Variable

		from time import time

		# from AE import *


		num_epochs = 2
		batch_size = 100
		hidden_size = 30


		# MNIST dataset
		dataset = dsets.MNIST(root='../data',
									train=True,
									transform=transforms.ToTensor(),
									download=False)

		# Data loader
		data_loader = torch.utils.data.DataLoader(dataset=dataset,
													batch_size=batch_size,
		shuffle=True)
		
		
		
		
		def to_var(x):
			if torch.cuda.is_available():
				x = x.cuda()
			return Variable(x)
		
		
		class Autoencoder(nn.Module):
		
			def __init__(self, in_dim=784, h_dim=400):
				super(Autoencoder, self).__init__()

				self.encoder = nn.Sequential(
					nn.Linear(in_dim, h_dim),
					nn.ReLU()
					)

				self.decoder = nn.Sequential(
					nn.Linear(h_dim, in_dim),
					nn.Sigmoid()
					)


			def forward(self, x):
				"""
				Note: image dimension conversion will be handled by external methods
				"""
				out = self.encoder(x)
				out = self.decoder(out)
				return out
		
		
		ae = Autoencoder(in_dim=784, h_dim=hidden_size)

		if torch.cuda.is_available():
			ae.cuda()

		criterion = nn.BCELoss()
		optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
		iter_per_epoch = len(data_loader)
		data_iter = iter(data_loader)

		# save fixed inputs for debugging
		fixed_x, _ = next(data_iter)
		torchvision.utils.save_image(Variable(fixed_x).data.cpu(), './data/real_images.png')
		fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))
		
		print(data_loader)
		print(enumerate(data_loader))
		sys.exit()
		
		for epoch in range(num_epochs):
			t0 = time()
			for i, (images, _) in enumerate(data_loader):
				print(i)
				# flatten the image
				images = to_var(images.view(images.size(0), -1))
				out = ae(images)
				loss = criterion(out, images)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i+1) % 100 == 0:
					print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs' 
						%(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.data[0], time()-t0))

			# save the reconstructed images
			reconst_images = ae(fixed_x)
			reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
		torchvision.utils.save_image(reconst_images.data.cpu(), './data/reconst_images_%d.png' % (epoch+1))
	
	
	elif config['mode'] == 'mode5':
		import torch.nn as nn
		import torch 
		import torchvision.datasets as dsets
		import torchvision.transforms as transforms
		import torchvision
		from torch.autograd import Variable

		from time import time



		
		
		def to_var(x):

			return Variable(x)
		
		
		class Autoencoder(nn.Module):
		
			def __init__(self, in_dim=6, h_dim=2):
				super(Autoencoder, self).__init__()

				self.encoder = nn.Sequential(
					nn.Linear(in_dim, h_dim),
					nn.ReLU()
					)

				self.decoder = nn.Sequential(
					nn.Linear(h_dim, in_dim),
					nn.Sigmoid()
					)


			def forward(self, x):
				print('caca')
				"""
				Note: image dimension conversion will be handled by external methods
				"""
				out = x.view(x.size(0), -1)
				out = self.encoder(out)
				print(out)
				out = self.decoder(out)
				return out
		
		num_epochs = 2
		batch_size = 3
		hidden_size = 2

		# felix = torch.tensor([5, 6, 8, 8, 8, 2], dtype=torch.float)
		felix = torch.tensor([[1, 1.1, 1, 1, 1.1, 1], [1, 0.9, 1, 1.1, 1, 1.1], [1.1, 0.99, 1, 1.1, 1.1, 1.1], [1.2, 0.9, 1, 1.1, 1.3, 0.9]], dtype=torch.float)
		# MNIST dataset
		# dataset = dsets.MNIST(root='../data',
									# train=True,
									# transform=transforms.ToTensor(),
									# download=False)

		# # Data loader
		# # print(type(dataset))
		# # print(dataset)
		# # sys.exit()
		# data_loader = torch.utils.data.DataLoader(dataset=dataset,
													# batch_size=batch_size,
		# shuffle=True)
		
		# print(type(data_loader))
		# sys.exit()
		ae = Autoencoder(in_dim=6, h_dim=hidden_size)

		# if torch.cuda.is_available():
			# ae.cuda()
			# print('!+++++++++++++++++++++!!')
			# sys.exit()

		criterion = nn.BCELoss()
		optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
		iter_per_epoch = len(felix)
		# data_iter = iter(data_loader)

		# # save fixed inputs for debugging
		# fixed_x, _ = next(data_iter)
		# torchvision.utils.save_image(Variable(fixed_x).data.cpu(), './data/real_images.png')
		# fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))
		
		
		# images = to_var(felix.view(felix.size(0), -1))
		out = ae(felix)
		print(out)
		loss = criterion(out, felix)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		
		# print(felix.size())
		# sys.exit()
		data_loader = torch.utils.data.TensorDataset(felix)
		for epoch in range(num_epochs):
			t0 = time()
			for i, (images, _) in enumerate(data_loader):
				# print(i)
				print(type(images))
				# print(type(images[0]))
				# print(type(images[0][0]))
				
				print(images.size())
				# print(images[0].size())
				# print(images[0][0].size())
				# print(_)
				images = felix
				# sys.exit()
				
				
				
				# flatten the image
				images = to_var(images.view(images.size(0), -1))
				out = ae(images)
				loss = criterion(out, images)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i+1) % 100 == 0:
					print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs' 
						%(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.data[0], time()-t0))

			# save the reconstructed images
			reconst_images = ae(fixed_x)
			reconst_images = reconst_images.view(reconst_images.size(0), 1, 3, 3)
		torchvision.utils.save_image(reconst_images.data.cpu(), './data/reconst_images_%d.png' % (epoch+1))
		
		
	else:
		print('unknown mode')
		sys.exit()

		
	return






def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'layers_pro' or element == 'feature_array':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float
	
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# config['n_batches'] = int(config['n_batches'])
	# config['db'] = int(config['db'])
	# config['divisions'] = int(config['divisions'])
	config['n_mov_avg'] = int(config['n_mov_avg'])
	config['train'] = float(config['train'])
	
	config['n_pre'] = float(config['n_pre'])
	config['m_post'] = float(config['m_post'])
	
	config['alpha_pro'] = float(config['alpha_pro'])
	config['tol'] = float(config['tol'])	
	config['learning_rate_init'] = float(config['learning_rate_init'])	
	#Type conversion to int	
	config['max_iter'] = int(config['max_iter'])
	config['rs_pro'] = int(config['rs_pro'])

	# Variable conversion	
	correct_layers = tuple([int(element) for element in (config['layers_pro'])])
	config['layers_pro'] = correct_layers
	
	config['ylabel'] = config['ylabel'].replace('_', ' ')
	config['zlabel'] = config['zlabel'].replace('_', ' ')
	config['title'] = config['title'].replace('_', ' ')

	
	# Variable conversion
	
	# Variable conversion
	if config['sheet'] == 'OFF':
		config['sheet'] = 0
	
	return config


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def add_Brbm(Visible, components, rs, learning_rate, verbose=None, n_iter=None):
	
	rbm = BernoulliRBM(n_components=components, random_state=rs, learning_rate=learning_rate, verbose=False, n_iter=50)
	rbm.fit(Visible)
	rbm_data = {'coefs':np.transpose(np.array(rbm.components_)), 'bias':np.array(rbm.intercept_hidden_), 'hidden':rbm.transform(Visible)}
	return rbm_data

# def sigmoid_array(x):
	# return 1 / (1 + math.exp(-x))
	
	
if __name__ == '__main__':
	main(sys.argv)
