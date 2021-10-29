from __future__ import print_function
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



from sklearn.pipeline import Pipeline
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.7, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature2':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature_array':['RMS']}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	

	if config['mode'] == 'pca':

		print('Select MASTER Features xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		
		
		mydict = mydict.to_dict(orient='list')
		

		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:

				example.append(mydict[feature][i])
			Features.append(example)
		
		Features = np.array(Features)
		# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
		# print(X)
		# print(np.size(X))
		# print(np.array(Features))
		# print(np.size(np.array(Features)))
		# sys.exit()
		
		# plt.plot(Features)
		# plt.show()
		
		scaler_model = StandardScaler()
		scaler_model.fit(Features)
		Features = scaler_model.transform(Features)
		
		# plt.plot(Features)
		# plt.show()
		
		
		
		pca_model = PCA(n_components=4)
		# pca_model = KernelPCA(n_components=4, kernel='cosine')
		pca_model.fit(Features)
		
		# print(pca_model)
		print(pca_model.explained_variance_)
		new = pca_model.transform(Features)
		
		plt.plot(new)
		plt.show()
	
	
	elif config['mode'] == 'dbn_fuse':

		print('Select MASTER Features xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		length_train = int(length_data*config['train'])		
		
		mydict = mydict.to_dict(orient='list')		

		Features_Train = []
		Features_Test = []
		Targets_Train = []
		Targets_Test = []
		for i in range(length_data):			
			example = [1]
			for feature in config['feature_array']:
				# example.append(mydict[feature][i])
				example.append(sigmoid(mydict[feature][i]))				
				
			if i <= length_train:
				Features_Train.append(example)
				# Targets_Train.append([mydict['Target'][i]])
				Targets_Train.append([sigmoid(mydict['Target'][i])])
			else:
				Features_Test.append(example)
				# Targets_Test.append([mydict['Target'][i]])
				Targets_Test.append([sigmoid(mydict['Target'][i])])
		
		Features_Train = np.array(Features_Train)
		Features_Test = np.array(Features_Test)
		
		Targets_Train = np.array(Targets_Train)
		Targets_Test = np.array(Targets_Test)
		
		
		# scaler_model = StandardScaler()
		# scaler_model.fit(Features_Train)
		# Features_Train = scaler_model.transform(Features_Train)
		# Features_Test = scaler_model.transform(Features_Test)
		
		# scaler_target = StandardScaler()
		# scaler_target.fit(Targets_Train)
		# Targets_Train = scaler_target.transform(Targets_Train)
		# Targets_Test = scaler_target.transform(Targets_Test)
		
		# Features_Train = sigmoid(Features_Train)
		# Features_Test = sigmoid(Features_Test)
		# Targets_Train = sigmoid(Targets_Train)
		# Targets_Test = sigmoid(Targets_Test)
		
		
		
		
		rbm = BernoulliRBM(n_components=8, random_state=0, verbose=True, learning_rate=0.6, n_iter=20)
		regressor = MLPRegressor(hidden_layer_sizes=[8] ,random_state=11, verbose=True, activation='identity', alpha=0.01)	

		
		DBN = Pipeline(steps=[('rbm', rbm), ('regressor', regressor)])
		
		DBN.fit(Features_Train, Targets_Train)
		
		Targets_Predict = DBN.predict(Features_Test)
		
		fig, ax = plt.subplots()
		ax.plot(Targets_Test, 'bo')
		ax.plot(Targets_Predict, 'ro')
		plt.show()
	
	
	elif config['mode'] == 'dbn_fuse_2':

		print('Select MASTER Features xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		length_train = int(length_data*config['train'])		
		
		mydict = mydict.to_dict(orient='list')		

		Features_Train = []
		Features_Test = []
		Targets_Train = []
		Targets_Test = []
		for i in range(length_data):			
			example = []
			for feature in config['feature_array']:
				example.append(mydict[feature][i])	
				
			if i <= length_train:
				Features_Train.append(example)
				Targets_Train.append([mydict['Target'][i]])
			else:
				Features_Test.append(example)
				Targets_Test.append([mydict['Target'][i]])
		
		Features_Train = np.array(Features_Train)
		Features_Test = np.array(Features_Test)
		
		Targets_Train = np.array(Targets_Train)
		Targets_Test = np.array(Targets_Test)
		
		
		# scaler_model = StandardScaler()
		# scaler_model.fit(Features_Train)
		# Features_Train = scaler_model.transform(Features_Train)
		# Features_Test = scaler_model.transform(Features_Test)
		
		Features_Train = minmax_scale(Features_Train, feature_range=(0,1))
		Features_Test = minmax_scale(Features_Test, feature_range=(0,1))
		
		
		
		
		# scaler_target = StandardScaler()
		# scaler_target.fit(Targets_Train)
		# Targets_Train = scaler_target.transform(Targets_Train)
		# Targets_Test = scaler_target.transform(Targets_Test)
		
		Targets_Train = Targets_Train/100.
		Targets_Test = Targets_Test/100.
		
		
		# plt.plot(Features_Train)
		# plt.show()
		
		
		
		
		# rbm = BernoulliRBM(n_components=2, random_state=1, verbose=False, learning_rate=0.06, n_iter=50)
		
		# regressor = MLPRegressor(hidden_layer_sizes=[2] ,random_state=11, verbose=False, activation='identity', alpha=0.01)	
		# print(rbm.intercept_visible_)
		# print(rbm.intercept_hidden_)
		# print(rbm.components_)

		# rbm.fit(Features_Train)
		# regressor.fit(Features_Train, Targets_Train)
		# print('RBM++++++++++++')
		# print(rbm.intercept_visible_)
		# print(rbm.intercept_hidden_)
		# print(rbm.components_)
		# caca = np.array(rbm.components_)		
		# caca2 = np.transpose(caca)		
		# print(caca)
		# print(caca2)
		# print('MLP++++++++++++')
		# print(regressor.intercepts_)
		# print(regressor.coefs_)
		
		# pelo = [caca2, regressor.coefs_[1]]
		# print('aaaaaa')
		# # print(pelo)
		
		# regressor.coefs_ = pelo
		# print(regressor.coefs_)
		print('RBM FUNCTION++++++++++++')
		_STOCHASTIC_SOLVERS = ['sgd', 'adam']
		class MLPRegressorOverride(MLPRegressor):
			# Overriding _init_coef method
			# def _init_coef(self, fan_in, fan_out):
				# if self.activation == 'logistic':
					# init_bound = np.sqrt(2. / (fan_in + fan_out))
				# elif self.activation in ('identity', 'tanh', 'relu'):
					# init_bound = np.sqrt(6. / (fan_in + fan_out))
				# else:
					# raise ValueError("Unknown activation function %s" %
									 # self.activation)
				
				# # first = add_Brbm(Visible=Features_Train, components=2, rs=1, learning_rate=0.06, verbose=None, n_iter=None)		
				# # second = add_Brbm(Visible=first['hidden'], components=1, rs=5, learning_rate=0.06, verbose=None, n_iter=None)	
				# # Coefs = [first['coefs'], second['coefs']]
				# # Bias = [first['bias'], second['bias']]

				# regressor = MLPRegressor(hidden_layer_sizes=[2] ,random_state=10, verbose=False, activation='identity', alpha=0.01)	
				
				# Tra = np.array([[1., 2., 3., 4.], [4., 2.5, 3., 3.], [5., 2., 3.8, 2.], [4., 5.5, 2., 3.]])
				# Te = np.array([[1.], [1.1], [0.7], [1.9]])
				# regressor.fit(Tra, Te)
				
				# coef_init = regressor.coefs_
				# intercept_init = regressor.intercepts_

				# return coef_init, intercept_init
				
			def _initialize(self, y, layer_units):
				# set all attributes, allocate weights etc for first call
				# Initialize parameters
				self.n_iter_ = 0
				self.t_ = 0
				self.n_outputs_ = y.shape[1]

				# Compute the number of layers
				self.n_layers_ = len(layer_units)

				# Output for regression
				# if not is_classifier(self):
					# self.out_activation_ = 'identity'
				# # Output for multi class
				# elif self._label_binarizer.y_type_ == 'multiclass':
					# self.out_activation_ = 'softmax'
				# # Output for binary class and multi-label
				# else:
					# self.out_activation_ = 'logistic'
				self.out_activation_ = 'identity'
				# Initialize coefficient and intercept layers
				# self.coefs_ = []
				# self.intercepts_ = []

				# for i in range(self.n_layers_ - 1):
					# coef_init, intercept_init = self._init_coef(layer_units[i],
																# layer_units[i + 1])
					# self.coefs_.append(coef_init)
					# self.intercepts_.append(intercept_init)
				
				
				# regressor = MLPRegressor(hidden_layer_sizes=[2] ,random_state=10, verbose=False, activation='identity', alpha=0.01)					
				# Tra = np.array([[1., 2., 3., 4.], [4., 2.5, 3., 3.], [5., 2., 3.8, 2.], [4., 5.5, 2., 3.]])
				# Te = np.array([[1.], [1.1], [0.7], [1.9]])
				# regressor.fit(Tra, Te)				
				# self.coefs_ = regressor.coefs_
				# self.intercepts_ = regressor.intercepts_
				
				
				first = add_Brbm(Visible=Features_Train, components=60, rs=1, learning_rate=0.006, verbose=None, n_iter=None)	
				second = add_Brbm(Visible=first['hidden'], components=20, rs=5, learning_rate=0.006, verbose=None, n_iter=None)
				third = add_Brbm(Visible=second['hidden'], components=1, rs=7, learning_rate=0.006, verbose=None, n_iter=None)					
				Coefs = [first['coefs'], second['coefs'], third['coefs']]
				Bias = [first['bias'], second['bias'], third['bias']]
				self.coefs_ = Coefs
				self.intercepts_ = Bias
				
				
				

				if self.solver in _STOCHASTIC_SOLVERS:
					self.loss_curve_ = []
					self._no_improvement_count = 0
					if self.early_stopping:
						self.validation_scores_ = []
						self.best_validation_score_ = -np.inf
					else:
						self.best_loss_ = np.inf
					
		
		# first = add_Brbm(Visible=Features_Train, components=2, rs=1, learning_rate=0.06, verbose=None, n_iter=None)		
		# second = add_Brbm(Visible=first['hidden'], components=1, rs=5, learning_rate=0.06, verbose=None, n_iter=None)		
		# Coefs = [first['coefs'], second['coefs']]
		# Bias = [first['bias'], second['bias']]
		

		# regressor = MLPRegressor(hidden_layer_sizes=[2] ,random_state=11, verbose=False, activation='identity', alpha=0.01)
		regressor = MLPRegressorOverride(hidden_layer_sizes=[60, 20] ,random_state=11, verbose=False, activation='identity', alpha=0.001)	
		
		
		# regressor.coefs_ = Coefs
		# regressor.intercepts_ = Bias
		
		regressor.partial_fit(Features_Train, Targets_Train)
		
		# print(regressor.coefs_)
		# print(regressor.intercepts_)
		
		# print(Coefs)
		# print(Bias)
		
		
		Targets_Predict = regressor.predict(Features_Test)
		
		
		fig, ax = plt.subplots()
		ax.plot(Targets_Test, 'bo')
		ax.plot(Targets_Predict, 'ro')
		plt.show()
		
		
		sys.exit()
		
		Targets_Predict = rbm.predict(Features_Test)
		
		
	
	elif config['mode'] == 'dbn_fuse_3':
		

		class RBM:
		  
			def __init__(self, num_visible, num_hidden):
				self.num_hidden = num_hidden
				self.num_visible = num_visible
				self.debug_print = True

				# Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
				# a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
				# and sqrt(6. / (num_hidden + num_visible)). One could vary the 
				# standard deviation by multiplying the interval with appropriate value.
				# Here we initialize the weights with mean 0 and standard deviation 0.1. 
				# Reference: Understanding the difficulty of training deep feedforward 
				# neural networks by Xavier Glorot and Yoshua Bengio
				np_rng = np.random.RandomState(1234)

				self.weights = np.asarray(np_rng.uniform(
						low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
									high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
									size=(num_visible, num_hidden)))


				# Insert weights for the bias units into the first row and first column.
				self.weights = np.insert(self.weights, 0, 0, axis = 0)
				self.weights = np.insert(self.weights, 0, 0, axis = 1)

			def train(self, data, max_epochs = 1000, learning_rate = 0.1):
				"""
				Train the machine.
				Parameters
				----------
				data: A matrix where each row is a training example consisting of the states of visible units.    
				"""

				num_examples = data.shape[0]

				# Insert bias units of 1 into the first column.
				data = np.insert(data, 0, 1, axis = 1)

				for epoch in range(max_epochs):      
					# Clamp to the data and sample from the hidden units. 
					# (This is the "positive CD phase", aka the reality phase.)
					pos_hidden_activations = np.dot(data, self.weights)      
					pos_hidden_probs = self._logistic(pos_hidden_activations)
					pos_hidden_probs[:,0] = 1 # Fix the bias unit.
					pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
					# Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
					# themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
					# "A Practical Guide to Training Restricted Boltzmann Machines" for more.
					pos_associations = np.dot(data.T, pos_hidden_probs)

					# Reconstruct the visible units and sample again from the hidden units.
					# (This is the "negative CD phase", aka the daydreaming phase.)
					neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
					neg_visible_probs = self._logistic(neg_visible_activations)
					neg_visible_probs[:,0] = 1 # Fix the bias unit.
					neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
					neg_hidden_probs = self._logistic(neg_hidden_activations)
					# Note, again, that we're using the activation *probabilities* when computing associations, not the states 
					# themselves.
					neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

					# Update weights.
					self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

					error = np.sum((data - neg_visible_probs) ** 2)
					if self.debug_print:
						print("Epoch %s: error is %s" % (epoch, error))

			def run_visible(self, data):
				"""
				Assuming the RBM has been trained (so that weights for the network have been learned),
				run the network on a set of visible units, to get a sample of the hidden units.
				
				Parameters
				----------
				data: A matrix where each row consists of the states of the visible units.
				
				Returns
				-------
				hidden_states: A matrix where each row consists of the hidden units activated from the visible
				units in the data matrix passed in.
				"""
				
				num_examples = data.shape[0]
				
				# Create a matrix, where each row is to be the hidden units (plus a bias unit)
				# sampled from a training example.
				hidden_states = np.ones((num_examples, self.num_hidden + 1))
				
				# Insert bias units of 1 into the first column of data.
				data = np.insert(data, 0, 1, axis = 1)

				# Calculate the activations of the hidden units.
				hidden_activations = np.dot(data, self.weights)
				# Calculate the probabilities of turning the hidden units on.
				hidden_probs = self._logistic(hidden_activations)
				# Turn the hidden units on with their specified probabilities.
				hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
				# Always fix the bias unit to 1.
				# hidden_states[:,0] = 1
			  
				# Ignore the bias units.
				hidden_states = hidden_states[:,1:]
				return hidden_states
			
		  # TODO: Remove the code duplication between this method and `run_visible`?
			def run_hidden(self, data):
				"""
				Assuming the RBM has been trained (so that weights for the network have been learned),
				run the network on a set of hidden units, to get a sample of the visible units.
				Parameters
				----------
				data: A matrix where each row consists of the states of the hidden units.
				Returns
				-------
				visible_states: A matrix where each row consists of the visible units activated from the hidden
				units in the data matrix passed in.
				"""

				num_examples = data.shape[0]

				# Create a matrix, where each row is to be the visible units (plus a bias unit)
				# sampled from a training example.
				visible_states = np.ones((num_examples, self.num_visible + 1))

				# Insert bias units of 1 into the first column of data.
				data = np.insert(data, 0, 1, axis = 1)

				# Calculate the activations of the visible units.
				visible_activations = np.dot(data, self.weights.T)
				# Calculate the probabilities of turning the visible units on.
				visible_probs = self._logistic(visible_activations)
				# Turn the visible units on with their specified probabilities.
				visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
				# Always fix the bias unit to 1.
				# visible_states[:,0] = 1

				# Ignore the bias units.
				visible_states = visible_states[:,1:]
				return visible_states
			
			def daydream(self, num_samples):
				"""
				Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
				(where each step consists of updating all the hidden units, and then updating all of the visible units),
				taking a sample of the visible units at each step.
				Note that we only initialize the network *once*, so these samples are correlated.
				Returns
				-------
				samples: A matrix, where each row is a sample of the visible units produced while the network was
				daydreaming.
				"""

				# Create a matrix, where each row is to be a sample of of the visible units 
				# (with an extra bias unit), initialized to all ones.
				samples = np.ones((num_samples, self.num_visible + 1))

				# Take the first sample from a uniform distribution.
				samples[0,1:] = np.random.rand(self.num_visible)

				# Start the alternating Gibbs sampling.
				# Note that we keep the hidden units binary states, but leave the
				# visible units as real probabilities. See section 3 of Hinton's
				# "A Practical Guide to Training Restricted Boltzmann Machines"
				# for more on why.
				for i in range(1, num_samples):
					visible = samples[i-1,:]

					# Calculate the activations of the hidden units.
					hidden_activations = np.dot(visible, self.weights)      
					# Calculate the probabilities of turning the hidden units on.
					hidden_probs = self._logistic(hidden_activations)
					# Turn the hidden units on with their specified probabilities.
					hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
					# Always fix the bias unit to 1.
					hidden_states[0] = 1

					# Recalculate the probabilities that the visible units are on.
					visible_activations = np.dot(hidden_states, self.weights.T)
					visible_probs = self._logistic(visible_activations)
					visible_states = visible_probs > np.random.rand(self.num_visible + 1)
					samples[i,:] = visible_states

				# Ignore the bias units (the first column), since they're always set to 1.
				return samples[:,1:]        
			  
			def _logistic(self, x):
				return 1.0 / (1 + np.exp(-x))

		r = RBM(num_visible = 6, num_hidden = 6)
		# training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
		training_data = np.array([[1.1,1,1,0,0,0],[1.3,0,1,0,0,0],[10.,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
		# training_data = np.array([[10.,11.,13.,0.9,0.,0.4],[21.,10.,31.,0.,0.9,0.],[1.,16.,1.,0.4,0.6,0.6],[0.3,0.,16.,1.,11.,0.], [0.,0.8,1.,1.,0.6,0.7],[10.,16.,15.,1.,1.,0.6]])
		r.train(training_data, max_epochs = 5000)
		print(r.weights)
		user = np.array([[0.,1.,0.,1.,0.1,0.8]])
		print(r.run_visible(user))
		
	
	
	elif config['mode'] == 'dbn_fuse_4':
		print('Select MASTER Features xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		length_train = int(length_data*config['train'])		
		
		mydict = mydict.to_dict(orient='list')		

		Features_Train = []
		Features_Test = []
		Targets_Train = []
		Targets_Test = []
		Features = []
		for i in range(length_data):			
			example = [0]
			for feature in config['feature_array']:
				example.append(mydict[feature][i])
			Features.append(example)
			if i <= length_train:
				Features_Train.append(example)
				Targets_Train.append([mydict['Target'][i]])
			else:
				Features_Test.append(example)
				Targets_Test.append([mydict['Target'][i]])
		
		
		Features_Train = np.array(Features_Train)
		Features_Test = np.array(Features_Test)
		Features = np.array(Features)
		
		Targets_Train = np.array(Targets_Train)
		Targets_Test = np.array(Targets_Test)
		
		
		scaler_model = StandardScaler()
		scaler_model.fit(Features_Train)
		Features_Train = scaler_model.transform(Features_Train)
		Features_Test = scaler_model.transform(Features_Test)
		
		scaler_target = StandardScaler()
		scaler_target.fit(Targets_Train)
		Targets_Train = scaler_target.transform(Targets_Train)
		Targets_Test = scaler_target.transform(Targets_Test)
		
		
		Targets = np.array(list(Targets_Train) + list(Targets_Test))
		
		
		pca_model = PCA(n_components=6)
		pca_model.fit(Features_Train)
		

		Features_Train = pca_model.transform(Features_Train)
		Features_Test = pca_model.transform(Features_Test)
		Features = pca_model.transform(Features)
		
		# print(np.ravel(Targets_Train))
		# plt.plot(Targets_Train)
		# plt.show()
		corr = []
		TFeatures_Train = np.transpose(Features_Train)
		TFeatures = np.transpose(Features)
		for feature_pca in TFeatures_Train:

			corr.append(np.corrcoef(np.ravel(feature_pca), np.ravel(Targets_Train))[0][1])		
		print(corr)
		
		plt.plot(Features_Train)
		plt.show()
		
		plt.plot(TFeatures_Train[np.argmax(np.absolute(corr))])
		plt.show()
		
		
		

		# regressor1 = MLPRegressor(hidden_layer_sizes=[3] ,random_state=11, verbose=False, activation='identity', alpha=0.01)
		# # regressor1 = NuSVR(kernel='linear', nu=0.001)
		# # regressor1 = tree.DecisionTreeRegressor()
		
		# # regressor1 = GaussianNB()		
		# regressor1.fit(Features_Train, Targets_Train)		
		# Targets_Predict = regressor1.predict(Features_Test)		
		
		
		# fig, ax = plt.subplots()
		# ax.plot(Targets_Test, 'bo')
		# ax.plot(Targets_Predict, 'ro')
		# plt.show()
		
		
		Feature = TFeatures[np.argmax(np.absolute(corr))]
		plt.plot(Feature, 'm')
		plt.show()
		
		Feature = np.array(Feature)
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))				
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	

		clf = MLPRegressor(solver='lbfgs', alpha=1.e-1, hidden_layer_sizes=[700, 500], random_state=2, activation='identity', verbose=False)

		
		n_pre = int(0.2*len(Train))
		m_post = int(0.1*len(Train))
		n_ex = len(Train) - n_pre - m_post
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex + 1):

			T_Inputs.append(Train[k : k + n_pre])
			# print(Train[k : k + n_pre])
			# sys.exit()
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
		clf.fit(T_Inputs, T_Outputs)
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		Predict = []
		It_Train = list(Train)

		for k in range(len(x_Predict) + m_post - 1):
			P_Input = It_Train[n_ex + k + 1 : n_ex + n_pre + k + 1]
			# print(P_Input)
			# sys.exit()
			P_Output = clf.predict([P_Input])
			P_Output = P_Output[0]
			
			
			Predict.append(P_Output[-1])
			It_Train.append(P_Output[-1])

		Predict = Predict[:-(m_post-1)]
	
		plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		plt.show()
	
	
	elif config['mode'] == 'dbn_fuse_5':
	
		class MLPRegressorOverride(MLPRegressor):
		# Overriding _init_coef method
			count = 0
			def _init_coef(self, fan_in, fan_out):
				if self.activation == 'logistic':
					init_bound = np.sqrt(2. / (fan_in + fan_out))
				elif self.activation in ('identity', 'tanh', 'relu'):
					init_bound = np.sqrt(6. / (fan_in + fan_out))
				else:
					raise ValueError("Unknown activation function %s" %
									 self.activation)
				coef_init = caca
				print(caca)
				print(count)
				print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
				intercept_init = caca
				count += 1

				return coef_init, intercept_init
	
	
		print('Select MASTER Features xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		

		mydict = pd.read_excel(filepath)
		rownames = list(mydict.index.values)
		length_data = len(rownames)
		length_train = int(length_data*config['train'])		
		
		mydict = mydict.to_dict(orient='list')		

		Features_Train = []
		Features_Test = []
		Targets_Train = []
		Targets_Test = []
		Features = []
		for i in range(length_data):			
			example = [0]
			for feature in config['feature_array']:
				example.append(mydict[feature][i])
			Features.append(example)
			if i <= length_train:
				Features_Train.append(example)
				Targets_Train.append([mydict['Target'][i]])
			else:
				Features_Test.append(example)
				Targets_Test.append([mydict['Target'][i]])
		
		
		Features_Train = np.array(Features_Train)
		Features_Test = np.array(Features_Test)
		Features = np.array(Features)
		
		Targets_Train = np.array(Targets_Train)
		Targets_Test = np.array(Targets_Test)
		
		
		scaler_model = StandardScaler()
		scaler_model.fit(Features_Train)
		Features_Train = scaler_model.transform(Features_Train)
		Features_Test = scaler_model.transform(Features_Test)
		
		scaler_target = StandardScaler()
		scaler_target.fit(Targets_Train)
		Targets_Train = scaler_target.transform(Targets_Train)
		Targets_Test = scaler_target.transform(Targets_Test)
		
		
		Targets = np.array(list(Targets_Train) + list(Targets_Test))
		
		
		pca_model = PCA(n_components=6)
		pca_model.fit(Features_Train)
		

		Features_Train = pca_model.transform(Features_Train)
		Features_Test = pca_model.transform(Features_Test)
		Features = pca_model.transform(Features)
		
		# print(np.ravel(Targets_Train))
		# plt.plot(Targets_Train)
		# plt.show()
		corr = []
		TFeatures_Train = np.transpose(Features_Train)
		TFeatures = np.transpose(Features)
		for feature_pca in TFeatures_Train:

			corr.append(np.corrcoef(np.ravel(feature_pca), np.ravel(Targets_Train))[0][1])		
		print(corr)
		
		plt.plot(Features_Train)
		plt.show()
		
		plt.plot(TFeatures_Train[np.argmax(np.absolute(corr))])
		plt.show()
		
		
		

		# regressor1 = MLPRegressor(hidden_layer_sizes=[3] ,random_state=11, verbose=False, activation='identity', alpha=0.01)
		# # regressor1 = NuSVR(kernel='linear', nu=0.001)
		# # regressor1 = tree.DecisionTreeRegressor()
		
		# # regressor1 = GaussianNB()		
		# regressor1.fit(Features_Train, Targets_Train)		
		# Targets_Predict = regressor1.predict(Features_Test)		
		
		
		# fig, ax = plt.subplots()
		# ax.plot(Targets_Test, 'bo')
		# ax.plot(Targets_Predict, 'ro')
		# plt.show()
		caca = 45
		
		Feature = TFeatures[np.argmax(np.absolute(corr))]
		plt.plot(Feature, 'm')
		plt.show()
		
		Feature = np.array(Feature)
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))				
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	

		clf = MLPRegressorOverride(solver='lbfgs', alpha=1.e-1, hidden_layer_sizes=[700, 500], random_state=2, activation='identity', verbose=False)

		
		n_pre = int(0.2*len(Train))
		m_post = int(0.1*len(Train))
		n_ex = len(Train) - n_pre - m_post
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex + 1):

			T_Inputs.append(Train[k : k + n_pre])
			# print(Train[k : k + n_pre])
			# sys.exit()
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
		clf.fit(T_Inputs, T_Outputs)
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		Predict = []
		It_Train = list(Train)

		for k in range(len(x_Predict) + m_post - 1):
			P_Input = It_Train[n_ex + k + 1 : n_ex + n_pre + k + 1]
			# print(P_Input)
			# sys.exit()
			P_Output = clf.predict([P_Input])
			P_Output = P_Output[0]
			
			
			Predict.append(P_Output[-1])
			It_Train.append(P_Output[-1])

		Predict = Predict[:-(m_post-1)]
	
		plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		plt.show()
		
		
		
		
	
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
		if element == 'layers' or element == 'feature_array':
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
	
	config['alpha'] = float(config['alpha'])
	config['tol'] = float(config['tol'])	
	config['learning_rate_init'] = float(config['learning_rate_init'])	
	#Type conversion to int	
	config['max_iter'] = int(config['max_iter'])
	config['rs'] = int(config['rs'])

	# Variable conversion	
	correct_layers = tuple([int(element) for element in (config['layers'])])
	config['layers'] = correct_layers
	
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
