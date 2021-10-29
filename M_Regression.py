# Reco_Signal_Training.py
# Last updated: 23.09.2017 by Felix Leaman
# Description:
# 

#++++++++++++++++++++++ IMPORT MODULES +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cm
# from tkinter import filedialog
# from skimage import img_as_uint
from tkinter import Tk
from tkinter import Button
# import skimage.filters
from tkinter import filedialog
from tkinter import Tk
import os.path
import sys
sys.path.insert(0, './lib') #to open user-defined functions
from scipy import stats
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_pattern import *
from m_denois import *
from m_det_features import *
from m_processing import *
from os.path import isfile, join, basename
import pickle
import argparse
from os import chdir, listdir
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import datetime
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import RFE
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
from sklearn.decomposition import PCA
from sklearn import tree
import graphviz 
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import FactorAnalysis
from random import randint
Inputs = ['mode']


InputsOpt_Defaults = {'plot':'OFF', 'rs':0, 'save':'ON', 'scaler':None, 'pca_comp':0, 'mypath':None, 'name':'auto', 'feature':'RMS', 'train':0.60, 'n_mov_avg':1, 'predict_node':'last', 'mini_batches':10, 'activation':'identity', 'alpha':0.001, 'epochs':10, 'hidden_layers':[50], 'valid':0.2, 'n_pre':0.3, 'm_post':0.05, 'auto_layers':None, 'save_plot':'ON', 'n_bests':3, 'weight':0.05, 'n_children':7, 'save_model':'OFF'}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	


	if config['mode'] == 'mode1':
		print('Select xls with features')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		Features = {}
		
		
		# for feature in config['features']:
			# Features[feature] = []		
		# for filepath in Filepaths:
			# mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			# mydict = mydict.to_dict(orient='list')
			
			# for feature in config['features']:
				# if feature in mydict.keys():
					# Features[feature] += mydict[feature]
		# n_samples = len(Features[config['features'][0]])
		# n_features = len(config['features'])
		DF_data = pd.read_excel(filepath)
		mydict = DF_data.to_dict(orient='list')
		
		if config['features'][0] == 'all':
			config['features'] = []
			for element in mydict.keys():
				if element.find('target') == -1:
					config['features'].append(element)
		print(config['features'])
		
		for feature in config['features']:
			
			if feature == 'KURT':
				Features[feature] = 1./np.array(mydict[feature])
			else:
				Features[feature] = np.array(mydict[feature])
			# plt.plot(Features[feature])
			# plt.title(feature)
			# plt.show()
		n_samples = len(Features[config['features'][0]])
		n_features = len(config['features'])
		
		X = np.zeros((n_samples, n_features))
		count = 0
		for feature in config['features']:
			X[:,count] = Features[feature]
			count += 1
			
		# n = X.shape[0]
		# ones = np.ones((n,1))
		# ones[:,0] = ones[:,0]*np.arange(n)		
		# X = np.append(ones, X, axis=1)

		
		y = mydict['target1']
		y = 0.01*np.random.rand(len(y)) + np.array(y)
		
		
		# Stages = {'A':0, 'GN':0, 'Q':0}
		# for date in DF_data.index:
			# if date.find('20180227') != -1:
				# Stages['A'] += 1
			# elif date.find('20180000') != -1:
				# Stages['A'] += 1
			# elif date.find('20181102') != -1:
				# Stages['Q'] += 1
			# else:
				# Stages['GN'] += 1		
		# y = np.zeros(n_samples)
		# y[0 : Stages['A']] = np.ones(len(Stages['A']))
		# y[Stages['A'] : Stages['A'] + Stages['GN']] = np.linspace(3, 6, Stages['GN'])
		# y[Stages['A'] + Stages['GN'] :] = np.linspace(9, 10, Stages['Q'])
		
		
		
		X_train = X[0:int(config['train']*X.shape[0]), :]
		X_test = X[int(config['train']*X.shape[0]) :, :]
		
		y_train = y[0:int(config['train']*len(y))]
		y_test = y[int(config['train']*len(y)) :]
		
		# t_test = np.linspace(X_train.shape[0], X.shape[0], num=X.shape[0] - X_train.shape[0], endpoint=False)
		t = np.arange(X.shape[0])
		t_train = np.arange(X_train.shape[0])
		t_test = np.linspace(X_train.shape[0], X.shape[0], num=X_test.shape[0], endpoint=False)
		
		# plt.plot(t, y, 'k'), plt.plot(t_train, y_train, 'ob'), plt.plot(t_test, y_test, 'r')
		# plt.show()
		
		
		
		# scaler = StandardScaler()
		scaler = MinMaxScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		

		
		
		# pca = PCA(n_components=7)
		# pca.fit(X_train)
		# print('PCA comps: ', pca.explained_variance_ratio_*100.)
		# X_train = pca.transform(X_train)
		# X_test = pca.transform(X_test)		
		
		# regressor = SVR(kernel='rbf', gamma='auto')
		# regressor = tree.DecisionTreeRegressor(min_samples_split=10, min_samples_leaf=3, max_leaf_nodes=None, max_features=None, random_state=0)		
		# regressor.fit(X_train, y_train)	

		
		
		regressor = MLPRegressor(solver='adam', alpha=0.1 , hidden_layer_sizes=[7, 6, 5, 4, 3, 2], random_state=0, activation='identity', verbose=False)
		regressor.fit(X_train, y_train)
		kf = KFold(n_splits=5, shuffle=True)
		count = 0
		epochs = 100
		for i in range(epochs):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			numb = 0
			for train_index, test_index in kf.split(X_train):
				print('+++++++++++Batch', numb)
				X_train_train, X_train_test = X_train[train_index], X_train[test_index]
				y_train_train, y_train_test = y_train[train_index], y_train[test_index]
				regressor.partial_fit(X_train_test, y_train_test)
				numb += 1
			count += 1
		
		
		cv = cross_val_score(regressor, X_train, y_train, cv=10)
		print(np.mean(cv))
		
		y_predict = regressor.predict(X_test)
		
		plt.plot(t, y, 'k'), plt.plot(t_train, y_train, 'ob'), plt.plot(t_test, y_predict, 'r')
		plt.show()
		
		sys.exit()
	
	elif config['mode'] == 'simple_ann_regression':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		Feature = np.array(Feature)
		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				# print(Feature[i])
				count += 1
				Feature[i] = Feature[i + count]				
			# print(Feature[i])
		# Feature = np.nan_to_num(Feature)
		Feature = movil_avg(Feature, config['n_mov_avg'])		
		
		# Feature = np.arange(500)
		
		# Feature = median_filter(data=Feature, points=5, same_length=True)

		x_Feature = np.arange(len(Feature))

		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		# print(config['hidden_layers'])
		# a = input('blaaa')
		clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=0, activation=config['activation'], verbose=False)
		

		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train))
		
		# n_pre = int(0.3*len(Train))
		# m_post = int(0.05*len(Train))
		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			# print(T_Inputs)
			# print(T_Outputs)
			# b = input('...')
			# aa = np.arange(len(Train[k : k + n_pre]))
			# plt.plot(aa, Train[k : k + n_pre], 'b')
			# bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# plt.show()
		# sys.exit()
			
		
		from sklearn.model_selection import KFold, GroupKFold
		kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=0)
	

		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		epochs = config['epochs']
		for i in range(epochs):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			numb = 0
			for train_index, test_index in kf.split(T_Inputs):
				print('+++++++++++Batch', numb)
				T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				clf.partial_fit(T_Inputs_test, T_Outputs_test)
				numb += 1
			count += 1


		
		
		
		# clf.fit(T_Inputs, T_Outputs)
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		Predict = []
		It_Train = list(Train)

		for k in range(len(x_Predict) + m_post - 1):
		
			if config['predict_node'] == 'last':
				P_Input = It_Train[n_ex + k : n_ex + n_pre + k]
				P_Output = clf.predict([P_Input])
				P_Output = P_Output[0]			
				Predict.append(P_Output[-1])
				It_Train.append(P_Output[-1])
			elif config['predict_node'] == 'first':				
				P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
				P_Output = clf.predict([P_Input])
				P_Output = P_Output[0]			
				Predict.append(P_Output[0])
				It_Train.append(P_Output[0])
			
			# print(P_Input)
			# b = input('...')
			
			
			

		Predict = Predict[:-(m_post-1)]
		
		
		fig, ax = plt.subplots()
		ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		ax.set_ylabel('Health Index', fontsize=13)
		ax.set_title('Linear Regression', fontsize=13)
		fact = 1.
		ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		ax.plot(x_Predict*fact, Predict, 'r', label='Prediction')
		ax.plot(x_Train*fact, Train, 'k', label='Training')
		ax.legend()
		plt.show()
	
	elif config['mode'] == 'test_model_ann':		
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			# Filepath = config['mypath']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		print('Select MODEL')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_model = filedialog.askopenfilename()
		root.destroy()
		
		clf = read_pickle(filepath_model)
		
		Feature = []		
		for filepath in Filepaths:
			# print(os.path.basename(filepath))
			mydict = pd.read_excel(filepath)
			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]		
		Feature = np.array(Feature)
		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				count += 1
				Feature[i] = Feature[i + count]				
		# Feature = np.nan_to_num(Feature)
		
		Feature = movil_avg(Feature, config['n_mov_avg'])
		# plt.plot(Feature, 'g')
		# plt.show()
		# print(len(Feature))
		# print(type(Feature))
		# print(Feature.shape)
		# 
		# Feature = np.arange(1000)
		# Feature = Full_Feature[0:int((config['train'] + config['valid'])*len(Feature))]	
		
		x_Feature = np.arange(len(Feature))

		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		
		
		Test = Feature[int(config['train']*len(Feature)) : ]
		x_Test = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		

		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train))		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		# if config['hidden_layers'][0] == 0:
			# print('Auto config of layers')
			# if config['auto_layers'] == '1x50':			
				# config['hidden_layers'] = (int(0.5*(n_pre - m_post)) + m_post)
			# elif config['auto_layers'] == '1x70':			
				# config['hidden_layers'] = (int(0.7*(n_pre - m_post)) + m_post)
			# elif config['auto_layers'] == '1x40':			
				# config['hidden_layers'] = (int(0.4*(n_pre - m_post)) + m_post)
			# elif config['auto_layers'] == '2x75x25':			
				# config['hidden_layers'] = (int(0.75*(n_pre - m_post)) + m_post, int(0.25*(n_pre - m_post)) + m_post)
			# elif config['auto_layers'] == '3x80x50x20':			
				# config['hidden_layers'] = (int(0.80*(n_pre - m_post)) + m_post, int(0.50*(n_pre - m_post)) + m_post , int(0.25*(n_pre - m_post)) + m_post)			
			# print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
			# # List_Layers = [50, 100]
		# else:
			# if config['auto_layers'] == 'coded':
				# lay = uncode_tuple_layers(config['hidden_layers'][0])
				# if len(lay) == 2:
					# config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				# elif len(lay) == 1:
					# config['hidden_layers'] = (int(lay[0]),)
				# else:
					# print('fatal error 557575')
					# sys.exit()
			# else:
				# print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
		# # print('---------------------....', config['hidden_layers'])
		# # a = input('pause-------....')
		# clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=0, activation=config['activation'], verbose=False)
		
		# # a = input('enter to continue...')
		# T_Inputs = []
		# T_Outputs = []

		# for k in range(n_ex):

			# T_Inputs.append(Train[k : k + n_pre])
			# T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			# # print(T_Inputs)
			# # print(T_Outputs)
			# # b = input('...')
			# # aa = np.arange(len(Train[k : k + n_pre]))
			# # plt.plot(aa, Train[k : k + n_pre], 'b')
			# # bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# # plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# # plt.show()
		# # sys.exit()
			
		
		# from sklearn.model_selection import KFold, GroupKFold
		# kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=0)
	

		# T_Inputs = np.array(T_Inputs)
		# T_Outputs = np.array(T_Outputs)
		count = 0
		epochs = config['epochs']
		MSE_Test = []
		MSE_Train = []
		
		CRC_Train = []
		CRC_Test = []
		
		DME_Train = []
		DME_Test = []
		
		DST_Train = []
		DST_Test = []
		
		DKU_Train = []
		DKU_Test = []
		

		
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

		
		
		Predict_Test = []
		It_Train = list(Train)
		for k in range(len(x_Test) + m_post - 1):		
			if config['predict_node'] == 'last':
				P_Input = It_Train[n_ex + k: n_ex + n_pre + k]
				# print(P_Input)
				# b = input('...')		
				P_Output = clf.predict([P_Input])
				P_Output = P_Output[0]			
				Predict_Test.append(P_Output[-1])
				It_Train.append(P_Output[-1])
			elif config['predict_node'] == 'first':				
				P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
				P_Output = clf.predict([P_Input])
				P_Output = P_Output[0]			
				Predict_Test.append(P_Output[0])
				It_Train.append(P_Output[0])
		Predict_Test = np.array(Predict_Test[:-(m_post-1)])
		
		# Predict_Train = list(Train)[0:n_pre+m_post-1]
		# It_TrainFirst = list(Train)[0:n_pre]
		# for k in range(len(x_Train) + m_post - 1 - (n_pre+m_post-1)):		
			# if config['predict_node'] == 'last':
				# P_Input = It_TrainFirst[k : n_pre + k]
				# # print(P_Input)
				# # b = input('...')		
				# P_Output = clf.predict([P_Input])
				# P_Output = P_Output[0]			
				# Predict_Train.append(P_Output[-1])
				# It_TrainFirst.append(P_Output[-1])
			# elif config['predict_node'] == 'first':				
				# P_Input = It_TrainFirst[k : n_pre + k]			
				# P_Output = clf.predict([P_Input])
				# P_Output = P_Output[0]			
				# Predict_Train.append(P_Output[0])
				# It_TrainFirst.append(P_Output[0])
		# Predict_Train = np.array(Predict_Train[:-(m_post-1)])
		
		
		MSE_Test.append(np.sum((Predict_Test - Test)**2.0)/len(Test))
		# MSE_Train.append(np.sum((Predict_Train[n_pre+m_post-1:] - Train[n_pre+m_post-1:])**2.0)/len(Train))
		
		CRC_Test.append(np.corrcoef(Predict_Test, Test)[0][1])
		# CRC_Train.append(np.corrcoef(Predict_Train[n_pre+m_post-1:], Train[n_pre+m_post-1:])[0][1])
		# print('MSE Test = ', MSE_Test)
		DME_Test.append(np.absolute(np.mean(Predict_Test) - np.mean(Test)))
		# DME_Train.append(np.absolute(np.mean(Predict_Train[n_pre+m_post-1:]) - np.mean(Train[n_pre+m_post-1:])))
		
		DST_Test.append(np.absolute(np.std(Predict_Test) - np.std(Test)))
		# DST_Train.append(np.absolute(np.std(Predict_Train[n_pre+m_post-1:]) - np.std(Train[n_pre+m_post-1:])))
		
		DKU_Test.append(np.absolute(stats.kurtosis(Predict_Test, fisher=False) - stats.kurtosis(Test, fisher=False)))
		# DKU_Train.append(np.absolute(stats.kurtosis(Predict_Train[n_pre+m_post-1:], fisher=False) - stats.kurtosis(Train[n_pre+m_post-1:], fisher=False)))
			
		# save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Test':MSE_Test, 'MSE_Train':MSE_Train, 'CRC_Train':CRC_Train, 'CRC_Test':CRC_Test, 'DME_Train':DME_Train, 'DME_Test':DME_Test, 'DST_Train':DST_Train, 'DST_Test':DST_Test, 'DKU_Train':DKU_Train, 'DKU_Test':DKU_Test})
		# save_pickle('config_' + config['name'] + '.pkl', config)
		
		# plt.plot(Feature, 'g')
		# plt.show()
		# if config['save_model'] == 'ON':
			# save_pickle('model_' + config['name'] + '.pkl', clf)
			
		fig, ax = plt.subplots()
		ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		ax.set_ylabel('Health Index', fontsize=13)
		ax.set_title('Linear Regression', fontsize=13)
		fact = 1.
		ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		ax.plot(x_Test*fact, Predict_Test, 'r', label='Test')
		ax.plot(x_Train*fact, Train, 'm', label='Training')
		ax.legend()
		plt.show()
		# if config['plot'] == 'ON':
			# plt.show()
	
	elif config['mode'] == 'auto_ann_regression_novalid':		
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			# Filepath = config['mypath']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		Feature = []		
		for filepath in Filepaths:
			# print(os.path.basename(filepath))
			mydict = pd.read_excel(filepath)
			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]		
		Feature = np.array(Feature)
		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				count += 1
				Feature[i] = Feature[i + count]				
		# Feature = np.nan_to_num(Feature)
		
		Feature = movil_avg(Feature, config['n_mov_avg'])
		# plt.plot(Feature, 'g')
		# plt.show()
		# print(len(Feature))
		# print(type(Feature))
		# print(Feature.shape)
		# 
		# Feature = np.arange(1000)
		# Feature = Full_Feature[0:int((config['train'] + config['valid'])*len(Feature))]	
		
		x_Feature = np.arange(len(Feature))

		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		
		
		Test = Feature[int(config['train']*len(Feature)) : ]
		x_Test = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		

		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train))		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		if config['hidden_layers'][0] == 0:
			print('Auto config of layers')
			if config['auto_layers'] == '1x50':			
				config['hidden_layers'] = (int(0.5*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '1x70':			
				config['hidden_layers'] = (int(0.7*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '1x40':			
				config['hidden_layers'] = (int(0.4*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '2x75x25':			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post)) + m_post, int(0.25*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '3x80x50x20':			
				config['hidden_layers'] = (int(0.80*(n_pre - m_post)) + m_post, int(0.50*(n_pre - m_post)) + m_post , int(0.25*(n_pre - m_post)) + m_post)			
			print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
			# List_Layers = [50, 100]
		else:
			if config['auto_layers'] == 'coded':
				lay = uncode_tuple_layers(config['hidden_layers'][0])
				if len(lay) == 2:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				elif len(lay) == 1:
					config['hidden_layers'] = (int(lay[0]),)
				else:
					print('fatal error 557575')
					sys.exit()
			else:
				print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
		# print('---------------------....', config['hidden_layers'])
		# a = input('pause-------....')
		clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=0, activation=config['activation'], verbose=False)
		
		# a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			# print(T_Inputs)
			# print(T_Outputs)
			# b = input('...')
			# aa = np.arange(len(Train[k : k + n_pre]))
			# plt.plot(aa, Train[k : k + n_pre], 'b')
			# bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# plt.show()
		# sys.exit()
			
		
		from sklearn.model_selection import KFold, GroupKFold
		kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=0)
	

		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		epochs = config['epochs']
		MSE_Test = []
		MSE_Train = []
		
		CRC_Train = []
		CRC_Test = []
		
		DME_Train = []
		DME_Test = []
		
		DST_Train = []
		DST_Test = []
		
		DKU_Train = []
		DKU_Test = []
		for i in range(epochs):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			numb = 0
			for train_index, test_index in kf.split(T_Inputs):
				print('+++++++++++Batch', numb)
				T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				clf.partial_fit(T_Inputs_test, T_Outputs_test)
				numb += 1
			count += 1


		
		
		
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

			
			
			Predict_Test = []
			It_Train = list(Train)
			for k in range(len(x_Test) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_Train[n_ex + k: n_ex + n_pre + k]
					# print(P_Input)
					# b = input('...')		
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[-1])
					It_Train.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[0])
					It_Train.append(P_Output[0])
			Predict_Test = np.array(Predict_Test[:-(m_post-1)])
			
			# Predict_Train = list(Train)[0:n_pre+m_post-1]
			# It_TrainFirst = list(Train)[0:n_pre]
			# for k in range(len(x_Train) + m_post - 1 - (n_pre+m_post-1)):		
				# if config['predict_node'] == 'last':
					# P_Input = It_TrainFirst[k : n_pre + k]
					# # print(P_Input)
					# # b = input('...')		
					# P_Output = clf.predict([P_Input])
					# P_Output = P_Output[0]			
					# Predict_Train.append(P_Output[-1])
					# It_TrainFirst.append(P_Output[-1])
				# elif config['predict_node'] == 'first':				
					# P_Input = It_TrainFirst[k : n_pre + k]			
					# P_Output = clf.predict([P_Input])
					# P_Output = P_Output[0]			
					# Predict_Train.append(P_Output[0])
					# It_TrainFirst.append(P_Output[0])
			# Predict_Train = np.array(Predict_Train[:-(m_post-1)])
			
			
			MSE_Test.append(np.sum((Predict_Test - Test)**2.0)/len(Test))
			# MSE_Train.append(np.sum((Predict_Train[n_pre+m_post-1:] - Train[n_pre+m_post-1:])**2.0)/len(Train))
			
			CRC_Test.append(np.corrcoef(Predict_Test, Test)[0][1])
			# CRC_Train.append(np.corrcoef(Predict_Train[n_pre+m_post-1:], Train[n_pre+m_post-1:])[0][1])
			# print('MSE Test = ', MSE_Test)
			DME_Test.append(np.absolute(np.mean(Predict_Test) - np.mean(Test)))
			# DME_Train.append(np.absolute(np.mean(Predict_Train[n_pre+m_post-1:]) - np.mean(Train[n_pre+m_post-1:])))
			
			DST_Test.append(np.absolute(np.std(Predict_Test) - np.std(Test)))
			# DST_Train.append(np.absolute(np.std(Predict_Train[n_pre+m_post-1:]) - np.std(Train[n_pre+m_post-1:])))
			
			DKU_Test.append(np.absolute(stats.kurtosis(Predict_Test, fisher=False) - stats.kurtosis(Test, fisher=False)))
			# DKU_Train.append(np.absolute(stats.kurtosis(Predict_Train[n_pre+m_post-1:], fisher=False) - stats.kurtosis(Train[n_pre+m_post-1:], fisher=False)))
			
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Test':MSE_Test, 'MSE_Train':MSE_Train, 'CRC_Train':CRC_Train, 'CRC_Test':CRC_Test, 'DME_Train':DME_Train, 'DME_Test':DME_Test, 'DST_Train':DST_Train, 'DST_Test':DST_Test, 'DKU_Train':DKU_Train, 'DKU_Test':DKU_Test})
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		# plt.plot(Feature, 'g')
		# plt.show()
		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		if config['save_plot'] == 'ON':
			fig, ax = plt.subplots()
			ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
			ax.set_ylabel('Health Index', fontsize=13)
			ax.set_title('Linear Regression', fontsize=13)
			fact = 1.
			ax.plot(x_Feature*fact, Feature, 'b', label='Real')
			ax.plot(x_Test*fact, Predict_Test, 'r', label='Test')
			ax.plot(x_Train*fact, Train, 'm', label='Training')
			ax.legend()
			plt.savefig('fig_' + config['name'] + '.png')
		# if config['plot'] == 'ON':
			# plt.show()
	
	elif config['mode'] == 'auto_ann_regression':		
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			# Filepath = config['mypath']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		Feature = []		
		for filepath in Filepaths:
			# print(os.path.basename(filepath))
			mydict = pd.read_excel(filepath)
			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]		
		Feature = np.array(Feature)
		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				count += 1
				Feature[i] = Feature[i + count]				
		# Feature = np.nan_to_num(Feature)
		
		Feature = movil_avg(Feature, config['n_mov_avg'])
		# plt.plot(Feature, 'g')
		# plt.show()
		# print(len(Feature))
		# print(type(Feature))
		# print(Feature.shape)
		# 
		# Feature = np.arange(1000)
		# Feature = Full_Feature[0:int((config['train'] + config['valid'])*len(Feature))]	
		
		x_Feature = np.arange(len(Feature))

		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Valid = Feature[int(config['train']*len(Feature)) : int(config['train']*len(Feature)) + int(config['valid']*len(Feature))]
		x_Valid = np.linspace(len(Train), len(Train) + len(Valid), num=len(Valid), endpoint=False)
		
		
		Test = Feature[int(config['train']*len(Feature)) + int(config['valid']*len(Feature)) : ]
		x_Test = np.linspace(len(Train) + len(Valid), len(Feature), num=len(Feature) - len(Train) - len(Valid), endpoint=False)
		

		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train))		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		if config['hidden_layers'][0] == 0:
			print('Auto config of layers')
			if config['auto_layers'] == '1x50':			
				config['hidden_layers'] = (int(0.5*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '1x70':			
				config['hidden_layers'] = (int(0.7*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '1x40':			
				config['hidden_layers'] = (int(0.4*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '2x75x25':			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post)) + m_post, int(0.25*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '3x80x50x20':			
				config['hidden_layers'] = (int(0.80*(n_pre - m_post)) + m_post, int(0.50*(n_pre - m_post)) + m_post , int(0.25*(n_pre - m_post)) + m_post)			
			print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
			# List_Layers = [50, 100]
		else:
			if config['auto_layers'] == 'coded':
				lay = uncode_tuple_layers(config['hidden_layers'][0])
				if len(lay) == 2:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				elif len(lay) == 1:
					config['hidden_layers'] = (int(lay[0]),)
				elif len(lay) == 3:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]), int(lay[2]))
				else:
					print('fatal error 557575')
					sys.exit()
			else:
				print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
		# print('---------------------....', config['hidden_layers'])
		# a = input('pause-------....')
		clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=0, activation=config['activation'], verbose=False)
		
		# a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			# print(T_Inputs)
			# print(T_Outputs)
			# b = input('...')
			# aa = np.arange(len(Train[k : k + n_pre]))
			# plt.plot(aa, Train[k : k + n_pre], 'b')
			# bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# plt.show()
		# sys.exit()
			
		
		from sklearn.model_selection import KFold, GroupKFold
		kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=0)
	

		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		epochs = config['epochs']
		MSE_Valid = []
		
		for i in range(epochs):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			numb = 0
			for train_index, test_index in kf.split(T_Inputs):
				print('+++++++++++Batch', numb)
				T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				clf.partial_fit(T_Inputs_test, T_Outputs_test)
				numb += 1
			count += 1


		
		
		
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
			Predict_Valid = []
			It_Train = list(Train)
			for k in range(len(x_Valid) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_Train[n_ex + k : n_ex + n_pre + k]
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Valid.append(P_Output[-1])
					It_Train.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Valid.append(P_Output[0])
					It_Train.append(P_Output[0])
				
				# print(P_Input)
				# b = input('...')		
			Predict_Valid = np.array(Predict_Valid[:-(m_post-1)])
			
			
			Predict_Test = []
			It_TrainValid = list(Train) + list(Valid)
			for k in range(len(x_Test) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_TrainValid[n_ex + k + len(Valid): n_ex + n_pre + len(Valid) + k]
					# print(P_Input)
					# b = input('...')		
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[-1])
					It_TrainValid.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_TrainValid[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[0])
					It_TrainValid.append(P_Output[0])
			Predict_Test = np.array(Predict_Test[:-(m_post-1)])
			
			Predict_Train = list(Train)[0:n_pre+m_post-1]
			It_TrainFirst = list(Train)[0:n_pre]
			for k in range(len(x_Train) + m_post - 1 - (n_pre+m_post-1)):		
				if config['predict_node'] == 'last':
					P_Input = It_TrainFirst[k : n_pre + k]
					# print(P_Input)
					# b = input('...')		
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Train.append(P_Output[-1])
					It_TrainFirst.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_TrainFirst[k : n_pre + k]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Train.append(P_Output[0])
					It_TrainFirst.append(P_Output[0])
			Predict_Train = np.array(Predict_Train[:-(m_post-1)])
			
			
			MSE_Valid.append(np.sum((Predict_Valid - Valid)**2.0)/len(Valid))
			MSE_Test.append(np.sum((Predict_Test - Test)**2.0)/len(Test))
			MSE_Train.append(np.sum((Predict_Train[n_pre+m_post-1:] - Train[n_pre+m_post-1:])**2.0)/len(Train))
			
			CRC_Valid.append(np.corrcoef(Predict_Valid, Valid)[0][1])
			CRC_Test.append(np.corrcoef(Predict_Test, Test)[0][1])
			CRC_Train.append(np.corrcoef(Predict_Train[n_pre+m_post-1:], Train[n_pre+m_post-1:])[0][1])
			# print('MSE Test = ', MSE_Test)
			DME_Valid.append(np.absolute(np.mean(Predict_Valid) - np.mean(Valid)))
			DME_Test.append(np.absolute(np.mean(Predict_Test) - np.mean(Test)))
			DME_Train.append(np.absolute(np.mean(Predict_Train[n_pre+m_post-1:]) - np.mean(Train[n_pre+m_post-1:])))
			
			DST_Valid.append(np.absolute(np.std(Predict_Valid) - np.std(Valid)))
			DST_Test.append(np.absolute(np.std(Predict_Test) - np.std(Test)))
			DST_Train.append(np.absolute(np.std(Predict_Train[n_pre+m_post-1:]) - np.std(Train[n_pre+m_post-1:])))
			
			DKU_Valid.append(np.absolute(stats.kurtosis(Predict_Valid, fisher=False) - stats.kurtosis(Valid, fisher=False)))
			DKU_Test.append(np.absolute(stats.kurtosis(Predict_Test, fisher=False) - stats.kurtosis(Test, fisher=False)))
			DKU_Train.append(np.absolute(stats.kurtosis(Predict_Train[n_pre+m_post-1:], fisher=False) - stats.kurtosis(Train[n_pre+m_post-1:], fisher=False)))
			
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Test':MSE_Test, 'MSE_Train':MSE_Train, 'CRC_Train':CRC_Train, 'CRC_Test':CRC_Test, 'DME_Train':DME_Train, 'DME_Test':DME_Test, 'DST_Train':DST_Train, 'DST_Test':DST_Test, 'DKU_Train':DKU_Train, 'DKU_Test':DKU_Test})
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		# plt.plot(Feature, 'g')
		# plt.show()
		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		if config['save_plot'] == 'ON':
			fig, ax = plt.subplots()
			ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
			ax.set_ylabel('Health Index', fontsize=13)
			ax.set_title('Linear Regression', fontsize=13)
			fact = 1.
			ax.plot(x_Feature*fact, Feature, 'b', label='Real')
			ax.plot(x_Valid*fact, Predict_Valid, 'g', label='Validation')
			ax.plot(x_Test*fact, Predict_Test, 'r', label='Test')
			ax.legend()
			plt.savefig('fig_' + config['name'] + '.png')
		# if config['plot'] == 'ON':
			# plt.show()
	

	
	elif config['mode'] == 'auto_ann_regression_msevalid':		
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			# Filepath = config['mypath']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		Feature = []		
		for filepath in Filepaths:
			# print(os.path.basename(filepath))
			mydict = pd.read_excel(filepath)
			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]		
		Feature = np.array(Feature)
		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				count += 1
				Feature[i] = Feature[i + count]				
		# Feature = np.nan_to_num(Feature)
		
		Feature = movil_avg(Feature, config['n_mov_avg'])
		# plt.plot(Feature, 'g')
		# plt.show()
		# print(len(Feature))
		# print(type(Feature))
		# print(Feature.shape)
		# 
		# Feature = np.arange(1000)
		# Feature = Full_Feature[0:int((config['train'] + config['valid'])*len(Feature))]	
		
		x_Feature = np.arange(len(Feature))

		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Valid = Feature[int(config['train']*len(Feature)) : int(config['train']*len(Feature)) + int(config['valid']*len(Feature))]
		x_Valid = np.linspace(len(Train), len(Train) + len(Valid), num=len(Valid), endpoint=False)
		
		
		Test = Feature[int(config['train']*len(Feature)) + int(config['valid']*len(Feature)) : ]
		x_Test = np.linspace(len(Train) + len(Valid), len(Feature), num=len(Feature) - len(Train) - len(Valid), endpoint=False)
		

		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train))		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		if config['hidden_layers'][0] == 0:
			print('Auto config of layers')
			if config['auto_layers'] == '1x50':			
				config['hidden_layers'] = (int(0.5*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '1x70':			
				config['hidden_layers'] = (int(0.7*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '1x40':			
				config['hidden_layers'] = (int(0.4*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '2x75x25':			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post)) + m_post, int(0.25*(n_pre - m_post)) + m_post)
			elif config['auto_layers'] == '3x80x50x20':			
				config['hidden_layers'] = (int(0.80*(n_pre - m_post)) + m_post, int(0.50*(n_pre - m_post)) + m_post , int(0.25*(n_pre - m_post)) + m_post)			
			print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
			# List_Layers = [50, 100]
		else:
			if config['auto_layers'] == 'coded':
				print(config['hidden_layers'][0])
				lay = uncode_tuple_layers(config['hidden_layers'][0])
				print(lay)
				if len(lay) == 2:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				elif len(lay) == 1:
					config['hidden_layers'] = (int(lay[0]),)
				elif len(lay) == 3:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]), int(lay[2]))
				else:
					print('fatal error 557575b')
					sys.exit()
			else:
				print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
		# print('---------------------....', config['hidden_layers'])
		# a = input('pause-------....')
		clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=0, activation=config['activation'], verbose=False)
		
		# a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			# print(T_Inputs)
			# print(T_Outputs)
			# b = input('...')
			# aa = np.arange(len(Train[k : k + n_pre]))
			# plt.plot(aa, Train[k : k + n_pre], 'b')
			# bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# plt.show()
		# sys.exit()
			
		
		from sklearn.model_selection import KFold, GroupKFold
		kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=0)
	

		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		epochs = config['epochs']
		MSE_Valid = []

		for i in range(epochs):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			numb = 0
			for train_index, test_index in kf.split(T_Inputs):
				print('+++++++++++Batch', numb)
				T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				clf.partial_fit(T_Inputs_test, T_Outputs_test)
				numb += 1
			count += 1


		
		
		
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
			Predict_Valid = []
			It_Train = list(Train)
			for k in range(len(x_Valid) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_Train[n_ex + k : n_ex + n_pre + k]
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Valid.append(P_Output[-1])
					It_Train.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Valid.append(P_Output[0])
					It_Train.append(P_Output[0])
					
			Predict_Valid = np.array(Predict_Valid[:-(m_post-1)])
			
			
			Predict_Test = []
			It_TrainValid = list(Train) + list(Valid)
			for k in range(len(x_Test) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_TrainValid[n_ex + k + len(Valid): n_ex + n_pre + len(Valid) + k]
					# print(P_Input)
					# b = input('...')		
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[-1])
					It_TrainValid.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_TrainValid[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[0])
					It_TrainValid.append(P_Output[0])
			Predict_Test = np.array(Predict_Test[:-(m_post-1)])
			

			
			MSE_Valid.append(np.sum((Predict_Valid - Valid)**2.0)/len(Valid))

			
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Valid':MSE_Valid})
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		# plt.plot(Feature, 'g')
		# plt.show()
		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		if config['save_plot'] == 'ON':
			fig, ax = plt.subplots()
			ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
			ax.set_ylabel('Health Index', fontsize=13)
			ax.set_title('Linear Regression', fontsize=13)
			fact = 1.
			ax.plot(x_Feature*fact, Feature, 'b', label='Real')
			ax.plot(x_Valid*fact, Predict_Valid, 'g', label='Validation')
			ax.plot(x_Test*fact, Predict_Test, 'r', label='Test')
			ax.legend()
			plt.savefig('fig_' + config['name'] + '.png')
		# if config['plot'] == 'ON':
			# plt.show()
		
	
	# elif config['mode'] == 'eval_mse':		
		# print('Select PKL Files with MSE')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		
		
		# MSE_Valid = []
		# MSE_Train = []
		# CRC_Valid = []
		# CRC_Train = []
		# Best_Epochs_CRC_Valid = []
		# Best_Epochs_MSE_Valid = []
		# Combi = []
		# Epoch_Combi = []
		# for filepath in Filepaths:
			# data = read_pickle(filepath)
			# # plt.plot(data['MSE_Valid'])
			# # plt.show()
			# CRC_Valid.append(np.max(np.array(data['CRC_Valid'])))
			# CRC_Train.append(np.max(np.array(data['CRC_Train'])))
			
			# MSE_Valid.append(np.min(np.array(data['MSE_Valid'])))
			# MSE_Train.append(np.min(np.array(data['MSE_Train'])))

			# Best_Epochs_CRC_Valid.append(np.argmax(np.array(data['CRC_Valid'])))
			# Best_Epochs_MSE_Valid.append(np.argmin(np.array(data['MSE_Valid'])))
			
			# mini_combi = []
			# for i, j in zip(data['CRC_Valid'], data['MSE_Valid']):
				# mini_combi.append(i/j)
			# Combi.append(np.max(np.array(mini_combi)))
			# Epoch_Combi.append(np.argmax(np.array(mini_combi)))
		
		# plt.figure(0)
		# plt.plot(CRC_Valid, '-o', color='red', label='CRC Valid')
		# plt.legend()
		# # plt.figure(1)
		# # plt.plot(Best_Epochs_CRC_Valid, '-o', color='black', label='CRC Valid Best Epoch')
		# # plt.legend()
		# # plt.figure(2)
		# # plt.plot(CRC_Train, '-o', label='CRC Train')
		# # plt.legend()
		# # plt.figure(3)
		# # plt.plot(np.array(MSE_Train)*np.array(MSE_Valid), '-s', color='m', label='')
		
		# # plt.figure(4)
		# # plt.plot(np.array(CRC_Valid)/np.array(MSE_Valid), '-o', color='cyan', label='CRC_Valid/MSE Valid')
		# # plt.legend()
		
		# plt.figure(5)
		# plt.plot(MSE_Valid, '-o', label='MSE Valid')
		# plt.legend()
		
		# # plt.figure(6)
		# # plt.plot(Best_Epochs_MSE_Valid, '-o', color='gray', label='MSE Valid Best Epoch')
		# # plt.legend()
		
		# plt.figure(7)
		# plt.plot(Combi, '-s', label='Combi')
		# plt.legend()
		
		# # plt.figure(8)
		# # plt.plot(Epoch_Combi, '-s', color='blue', label='Combi Epoch')
		# # plt.legend()
		
		# plt.show()
	
	# elif config['mode'] == 'eval_mse_2':		
		# print('Select PKL Files with MSE')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()		
		
		# tipo = 'Train'
		
		# count = 0
		# for filepath in Filepaths:
			# data = read_pickle(filepath)

			# if count == 0:
				# CRC_amax = np.max(np.absolute(np.array(data['CRC_' + tipo])))
				# MSE_amax = np.max(np.absolute(np.array(data['MSE_' + tipo])))				
				# DME_amax = np.max(np.absolute(np.array(data['DME_' + tipo])))
				# DST_amax = np.max(np.absolute(np.array(data['DST_' + tipo])))
				# DKU_amax = np.max(np.absolute(np.array(data['DKU_' + tipo])))
			# else:
				# if np.max(np.absolute(np.array(data['CRC_' + tipo]))) > CRC_amax:
					# CRC_amax = np.max(np.absolute(np.array(data['CRC_' + tipo])))
					
				# if np.max(np.absolute(np.array(data['MSE_' + tipo]))) > MSE_amax:
					# MSE_amax = np.max(np.absolute(np.array(data['MSE_' + tipo])))
				
				# if np.max(np.absolute(np.array(data['DME_' + tipo]))) > DME_amax:
					# DME_amax = np.max(np.absolute(np.array(data['DME_' + tipo])))
					
				# if np.max(np.absolute(np.array(data['DST_' + tipo]))) > DST_amax:
					# DST_amax = np.max(np.absolute(np.array(data['DST_' + tipo])))
					
				# if np.max(np.absolute(np.array(data['DKU_' + tipo]))) > DKU_amax:
					# DKU_amax = np.max(np.absolute(np.array(data['DKU_' + tipo])))

		# Combi = []
		# Epoch_Combi = []
		# for filepath in Filepaths:
			# data = read_pickle(filepath)

			# mini_combi = []
			# for crc, mse, dme, dst, dku in zip(data['CRC_' + tipo], data['MSE_' + tipo], data['DME_' + tipo], data['DST_' + tipo], data['DKU_' + tipo]):
				# mini_combi.append(crc/CRC_amax - mse/MSE_amax - dme/DME_amax - dst/DST_amax - dku/DKU_amax)
			# Combi.append(np.max(np.array(mini_combi)))
			# Epoch_Combi.append(np.argmax(np.array(mini_combi)))	
		
		# plt.figure(0)
		# plt.plot(Combi, '-o', color='red', label='Combi')
		# plt.legend()
		
		# plt.figure(1)
		# plt.plot(Epoch_Combi, '-s', color='black', label='Combi')
		# plt.legend()
		
		
		# plt.show()

	elif config['mode'] == 'eval_mse_3':		
		print('Select PKL Files with MSE')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()		
		
		# Max_Ind_Scores =  calculate_amax_scores(Filepaths)
		
		Combi, Epoch_Combi = calculate_combi_scoreB_msevalid(Filepaths)		
		
		plt.figure(0)
		plt.plot(Combi, '-o', color='red', label='Combi')
		plt.legend()
		
		plt.figure(1)
		plt.plot(Epoch_Combi, '-s', color='black', label='Combi')
		plt.legend()
		
		
		plt.show()
	
	elif config['mode'] == 'generate_genetic':

		print('Select PKL Files with SCORES')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print(Filepaths[3][-25:])
		# sys.exit()
		
		print('Select PKL Files with CONFIG')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Configs = filedialog.askopenfilenames()
		root.destroy()
		
		for perro, gato in zip(Filepaths, Filepaths_Configs):
			if perro[-25:] !=  gato[-25:]:
				print('fatal error 32154')
				sys.exit()
		
		# Max_Ind_Scores =  calculate_amax_scores(Filepaths)		
		Combi, Epoch_Combi = calculate_combi_scoreB_msevalid(Filepaths)
		Idx_Combi = list(np.arange(len(Combi)))
		
		Combi_sorted = sorted(Combi)
		Epoch_Combi_sorted = sort_X_based_on_Y(Epoch_Combi, Combi)
		Idx_Combi_sorted = sort_X_based_on_Y(Idx_Combi, Combi)
		
		n_bests = config['n_bests']
		best_combis = Idx_Combi_sorted[-n_bests:]
		
		# print(best_combis)
		# sys.exit()
		best_epoch_combis = Epoch_Combi_sorted[-n_bests:]		
		
		Filepaths_Configs_bests = [Filepaths_Configs[i] for i in best_combis]
		Filepaths_bests = [Filepaths[i] for i in best_combis]
		
		# for dudu, dada in zip(Filepaths_Configs_bests, Filepaths_bests):
			# pupu = read_pickle(dudu)
			# save_pickle(basename(dudu), pupu)
			
			# papa = read_pickle(dada)
			# save_pickle(basename(dada), papa)
		
		# # print(Filepaths_Configs_bests)
		# sys.exit()
		
		Hidden_Layers = []
		Alphas = []
		N_Pres = []
		M_Posts = []
		MiniBatches = []
		Epochs = best_epoch_combis
		for data in Filepaths_Configs_bests:
			configLOC = read_pickle(data)
			# print(configLOC['hidden_layers'])
			if isinstance(configLOC['hidden_layers'], tuple) == False:
				configLOC['hidden_layers'] = (configLOC['hidden_layers'],)
				# print(configLOC['hidden_layers'])
			codlayers = code_tuple_layers(configLOC['hidden_layers'])
			
			# print(codlayers)
			# Hidden_Layers.append(configLOC['hidden_layers'])
			Hidden_Layers.append(codlayers)
			Alphas.append(configLOC['alpha'])
			N_Pres.append(configLOC['n_pre'])
			M_Posts.append(configLOC['m_post'])
			MiniBatches.append(configLOC['mini_batches'])
		# sys.exit()
		# print(Hidden_Layers[10])
		# print(uncode_tuple_layers(Hidden_Layers[10]))
		Best_Hyper = np.empty((n_bests, 6))
		for i in range(n_bests):
			nn_hyper = {'hidden_layers':Hidden_Layers[i], 'alpha':Alphas[i], 'n_pre':N_Pres[i], 'm_post':M_Posts[i], 'mini_batches':MiniBatches[i], 'epochs':Epochs[i]+1}
			# Best_Hyper.append(nn_hyper)
			# print(Hidden_Layers[i])
			Best_Hyper[i, :] = np.array([Hidden_Layers[i], Alphas[i], N_Pres[i], M_Posts[i], MiniBatches[i], Epochs[i]+1])
			
		parents = Best_Hyper
		
		n_children = config['n_children']
		offspring = crossover_ann(parents, n_children)
		
		weight = config['weight']
		
		# print(offspring[0][0])
		# print(uncode_tuple_layers(offspring[0][0]))
		offspring = mutation_ann(offspring, weight)
		# print(offspring[0][0])
		# print(uncode_tuple_layers(offspring[0][0]))
		# sys.exit()
		
		# new_population = offspring
		new_population = np.empty((n_bests+n_children, 6))	
		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring
		
		print(new_population)
		
		opt = input('Generate new generation? By Yes: Enter "number" / By No: Enter "no" ... ')
		# Layers = new_population[:, 0]
		# Alphas = new_population[:, 1]
		# N_Pres = new_population[:, 2]
		# M_Posts = new_population[:, 3]
		# MiniBatches = new_population[:, 4]
		# Epochs = new_population[:, 5]
		
		if opt != 'no':

			Layers = [str(int(element)) for element in list(new_population[:, 0])]
			Alphas = [str(element) for element in list(new_population[:, 1])]
			N_Pres = [str(element) for element in list(new_population[:, 2])]
			M_Posts = [str(element) for element in list(new_population[:, 3])]
			MiniBatches = [str(int(element)) for element in list(new_population[:, 4])]
			Epochs = [str(int(element)) for element in list(new_population[:, 5])]

			# print(Layers)
			# sys.exit()
			mypath = 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Tri_Analysis\\Idx14'
			count = 0
			for layer, alpha, n_pre, m_post, minibatches, epochs in zip(Layers, Alphas, N_Pres, M_Posts, MiniBatches, Epochs):
				if count >= 0:
				# print('++++++++++++++++...', layer)
					os.system('python M_Regression.py --plot OFF --rs 0 --hidden_layers ' + layer + ' --mode auto_ann_regression_msevalid --name Gen' + opt + '_CORR_Idx14_ANN_Idx_' + str(count) + ' --activation identity --auto_layers coded --alpha ' + alpha + ' --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg 12 --train 0.6 --feature CORR' + ' --predict_node last --valid 0.2 --mini_batches ' + minibatches + ' --epochs ' + epochs + ' --mypath ' + mypath)
				count += 1
	
	elif config['mode'] == 'generate_genetic_msevalid_5gen':

		print('Select PKL Files with SCORES')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print(Filepaths[3][-25:])
		# sys.exit()
		
		print('Select PKL Files with CONFIG')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Configs = filedialog.askopenfilenames()
		root.destroy()
		
		for perro, gato in zip(Filepaths, Filepaths_Configs):
			if perro[-25:] !=  gato[-25:]:
				print('fatal error 32154')
				sys.exit()
		
		# Max_Ind_Scores =  calculate_amax_scores(Filepaths)		
		Combi, Epoch_Combi = calculate_combi_scoreB_msevalid(Filepaths)
		Idx_Combi = list(np.arange(len(Combi)))
		
		Combi_sorted = sorted(Combi)
		Epoch_Combi_sorted = sort_X_based_on_Y(Epoch_Combi, Combi)
		Idx_Combi_sorted = sort_X_based_on_Y(Idx_Combi, Combi)
		
		n_bests = config['n_bests']
		best_combis = Idx_Combi_sorted[-n_bests:]
		
		# print(best_combis)
		# sys.exit()
		best_epoch_combis = Epoch_Combi_sorted[-n_bests:]		
		
		Filepaths_Configs_bests = [Filepaths_Configs[i] for i in best_combis]
		Filepaths_bests = [Filepaths[i] for i in best_combis]
		
		# for dudu, dada in zip(Filepaths_Configs_bests, Filepaths_bests):
			# pupu = read_pickle(dudu)
			# save_pickle(basename(dudu), pupu)
			
			# papa = read_pickle(dada)
			# save_pickle(basename(dada), papa)
		
		# # print(Filepaths_Configs_bests)
		# sys.exit()
		
		Hidden_Layers = []
		Alphas = []
		N_Pres = []
		M_Posts = []
		MiniBatches = []
		Epochs = best_epoch_combis
		for data in Filepaths_Configs_bests:
			configLOC = read_pickle(data)
			# print(configLOC['hidden_layers'])
			if isinstance(configLOC['hidden_layers'], tuple) == False:
				configLOC['hidden_layers'] = (configLOC['hidden_layers'],)
				# print(configLOC['hidden_layers'])
			# print(configLOC['hidden_layers'])
			codlayers = code_tuple_layers(configLOC['hidden_layers'])
			# print(codlayers)
			# print(codlayers)
			# Hidden_Layers.append(configLOC['hidden_layers'])
			Hidden_Layers.append(codlayers)
			Alphas.append(configLOC['alpha'])
			N_Pres.append(configLOC['n_pre'])
			M_Posts.append(configLOC['m_post'])
			MiniBatches.append(configLOC['mini_batches'])
		# sys.exit()
		# print(Hidden_Layers[10])
		# print(uncode_tuple_layers(Hidden_Layers[10]))
		Best_Hyper = np.empty((n_bests, 5))
		for i in range(n_bests):
			# nn_hyper = {'hidden_layers':Hidden_Layers[i], 'alpha':Alphas[i], 'n_pre':N_Pres[i], 'm_post':M_Posts[i], 'mini_batches':MiniBatches[i], 'epochs':Epochs[i]+1}
			# Best_Hyper.append(nn_hyper)
			# print(Hidden_Layers[i])
			Best_Hyper[i, :] = np.array([Hidden_Layers[i], Alphas[i], N_Pres[i], M_Posts[i], MiniBatches[i]])
			
		parents = Best_Hyper
		
		n_children = config['n_children']
		offspring = crossover_ann_5gen(parents, n_children)
		
		weight = config['weight']
		
		# print(offspring[0][0])
		# print(uncode_tuple_layers(offspring[0][0]))
		offspring = mutation_ann_5gen(offspring, weight)
		# print(offspring[0][0])
		# print(uncode_tuple_layers(offspring[0][0]))
		# sys.exit()
		
		# new_population = offspring
		new_population = np.empty((n_bests+n_children, 5))	
		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring
		
		print(new_population)
		
		opt = input('Generate new generation? By Yes: Enter "number" / By No: Enter "no" ... ')

		
		if opt != 'no':
			Layers = [str(int(element)) for element in list(new_population[:, 0])]
			Alphas = [str(element) for element in list(new_population[:, 1])]
			N_Pres = [str(element) for element in list(new_population[:, 2])]
			M_Posts = [str(element) for element in list(new_population[:, 3])]
			MiniBatches = [str(int(element)) for element in list(new_population[:, 4])]
			Epochs = [str(int(config['epochs'])) for i in range(len(new_population[:, 4]))]

			# print(Layers)
			# sys.exit()
			mypath = 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Tri_Analysis\\Idx14'
			count = 0
			for layer, alpha, n_pre, m_post, minibatches, epochs in zip(Layers, Alphas, N_Pres, M_Posts, MiniBatches, Epochs):
				if count >= 0:
				# print('++++++++++++++++...', layer)
					os.system('python M_Regression.py --plot OFF --rs 0 --hidden_layers ' + layer + ' --mode auto_ann_regression_msevalid_5gen --name Gen' + opt + '_CORR_Idx14_ANN_Idx_' + str(count) + ' --activation identity --auto_layers coded --alpha ' + alpha + ' --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg 12 --train 0.6 --feature CORR' + ' --predict_node last --valid 0.2 --mini_batches ' + minibatches + ' --epochs ' + epochs + ' --mypath ' + mypath)
				count += 1
	
		
	else:
		print('unknown mode')

	
	return

def narray_features(filepath, names_features):				
	Dict_Features = {}
	for feature in names_features:
		Dict_Features[feature] = []
	mydict = pd.read_excel(filepath)			
	mydict = mydict.to_dict(orient='list')			
	for element in names_features:
		Dict_Features[element] += mydict[element]				
	n_samples = len(Dict_Features[names_features[0]])
	n_features = len(names_features)		
	Features = np.zeros((n_samples, n_features))
	count = 0
	for feature in names_features:
		Features[:, count] = Dict_Features[feature]
		count += 1
	return Features	


def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'files' or element == 'hidden_layers':
				parser.add_argument('--' + element, nargs='+')
			else:
				parser.add_argument('--' + element, nargs='?')
		print(parser.parse_args())
		args = parser.parse_args()
		
	except:
		# args = argv
		arguments = [element for element in argv if element[0:2] == '--']
		values = [element for element in argv if element[0:2] != '--']

		# from argparse import ArgumentParser
		# from ArgumentParser import Namespace
		parser = ArgumentParser()
		for element in arguments:
			parser.add_argument(element)

		args = parser.parse_args(argv)

		# print(test)
		# sys.exit()
		
	config = {}	
		
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float

	config['rs'] = int(config['rs'])
	config['pca_comp'] = int(config['pca_comp'])
	# config['clusters'] = int(config['clusters'])
	config['n_mov_avg'] = int(config['n_mov_avg'])
	config['mini_batches'] = int(config['mini_batches'])
	
	config['epochs'] = int(config['epochs'])
	
	config['alpha'] = float(config['alpha'])
	
	correct_layers = tuple([int(element) for element in (config['hidden_layers'])])
	config['hidden_layers'] = correct_layers
	
	config['n_pre'] = float(config['n_pre'])
	config['m_post'] = float(config['m_post'])
	
	config['train'] = float(config['train'])
	config['valid'] = float(config['valid'])
	
	config['n_bests'] = int(config['n_bests'])
	config['weight'] = float(config['weight'])
	config['n_children'] = int(config['n_children'])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config
	


	
def calculate_amax_scores(Filepaths):
	count = 0
	for filepath in Filepaths:
		data = read_pickle(filepath)			
		
		if count == 0:
			CRC_amax = np.max(np.absolute(np.array(data['CRC_Valid'] + data['CRC_Train'])))
			MSE_amax = np.max(np.absolute(np.array(data['MSE_Valid'] + data['MSE_Train'])))				
			DME_amax = np.max(np.absolute(np.array(data['DME_Valid'] + data['DME_Train'])))
			DST_amax = np.max(np.absolute(np.array(data['DST_Valid'] + data['DST_Train'])))
			DKU_amax = np.max(np.absolute(np.array(data['DKU_Valid'] + data['DKU_Train'])))
		else:
			if np.max(np.absolute(np.array(data['CRC_Valid'] + data['CRC_Train']))) > CRC_amax:
				CRC_amax = np.max(np.absolute(np.array(data['CRC_Valid'] + data['CRC_Train'])))
				
			if np.max(np.absolute(np.array(data['MSE_Valid'] + data['MSE_Train']))) > MSE_amax:
				MSE_amax = np.max(np.absolute(np.array(data['MSE_Valid'] + data['MSE_Train'])))
			
			if np.max(np.absolute(np.array(data['DME_Valid'] + data['DME_Train']))) > DME_amax:
				DME_amax = np.max(np.absolute(np.array(data['DME_Valid'] + data['DME_Train'])))
				
			if np.max(np.absolute(np.array(data['DST_Valid'] + data['DST_Train']))) > DST_amax:
				DST_amax = np.max(np.absolute(np.array(data['DST_Valid'] + data['DST_Train'])))
				
			if np.max(np.absolute(np.array(data['DKU_Valid'] + data['DKU_Train']))) > DKU_amax:
				DKU_amax = np.max(np.absolute(np.array(data['DKU_Valid'] + data['DKU_Train'])))
	return {'CRC_amax':CRC_amax, 'MSE_amax':MSE_amax, 'DME_amax':DME_amax, 'DST_amax':DST_amax, 'DKU_amax':DKU_amax}
	
	

def calculate_combi_score(Filepaths, Max_Ind_Scores):
	CRC_amax = Max_Ind_Scores['CRC_amax']
	MSE_amax = Max_Ind_Scores['MSE_amax']
	DME_amax = Max_Ind_Scores['DME_amax']
	DST_amax = Max_Ind_Scores['DST_amax']
	DKU_amax = Max_Ind_Scores['DKU_amax']
	Combi = []
	Epoch_Combi = []
	for filepath in Filepaths:
		data = read_pickle(filepath)

		mini_combi = []
		for crc_v, mse_v, dme_v, dst_v, dku_v, crc_t, mse_t, dme_t, dst_t, dku_t in zip(data['CRC_Valid'], data['MSE_Valid'], data['DME_Valid'], data['DST_Valid'], data['DKU_Valid'], data['CRC_Train'], data['MSE_Train'], data['DME_Train'], data['DST_Train'], data['DKU_Train']):
			valid = crc_v/CRC_amax - mse_v/MSE_amax - dme_v/DME_amax - dst_v/DST_amax - dku_v/DKU_amax
			train = crc_t/CRC_amax - mse_t/MSE_amax - dme_t/DME_amax - dst_t/DST_amax - dku_t/DKU_amax
			mini_combi.append((valid + train)/2.)
		Combi.append(np.max(np.array(mini_combi)))
		Epoch_Combi.append(np.argmax(np.array(mini_combi)))
	return Combi, Epoch_Combi

def calculate_combi_scoreB(Filepaths):
	
	Combi = []
	Epoch_Combi = []
	for filepath in Filepaths:
		data = read_pickle(filepath)

		mini_combi = []
		for crc_v, mse_v, dme_v, dst_v, dku_v, crc_t, mse_t, dme_t, dst_t, dku_t in zip(data['CRC_Valid'], data['MSE_Valid'], data['DME_Valid'], data['DST_Valid'], data['DKU_Valid'], data['CRC_Train'], data['MSE_Train'], data['DME_Train'], data['DST_Train'], data['DKU_Train']):
			# valid = crc_v / (mse_v + dme_v + dst_v + dku_v)
			# train = crc_t/ (mse_t + dme_t + dst_t + dku_t)
			
			# valid = crc_v - dme_v - dst_v - dku_v
			# train = crc_t - dme_t - dst_t - dku_t
			
			# valid = crc_v / mse_v - dme_v - dst_v
			# train = crc_t / mse_t - dme_t - dst_t
			
			# valid = crc_v / mse_v
			# train = crc_t / mse_t
			
			valid = 1/(mse_v)
			# train = 1/(mse_t)
			
			# valid = crc_v
			# train = crc_t
			
			
			mini_combi.append(valid)
		Combi.append(np.max(np.array(mini_combi)))
		Epoch_Combi.append(np.argmax(np.array(mini_combi)))
	return Combi, Epoch_Combi

def calculate_combi_scoreB_msevalid(Filepaths):
	
	Combi = []
	Epoch_Combi = []
	for filepath in Filepaths:
		data = read_pickle(filepath)

		mini_combi = []
		for mse_v in data['MSE_Valid']:
			# valid = crc_v / (mse_v + dme_v + dst_v + dku_v)
			# train = crc_t/ (mse_t + dme_t + dst_t + dku_t)
			
			# valid = crc_v - dme_v - dst_v - dku_v
			# train = crc_t - dme_t - dst_t - dku_t
			
			# valid = crc_v / mse_v - dme_v - dst_v
			# train = crc_t / mse_t - dme_t - dst_t
			
			# valid = crc_v / mse_v
			# train = crc_t / mse_t
			
			valid = 1/(mse_v)
			# train = 1/(mse_t)
			
			# valid = crc_v
			# train = crc_t
			
			
			mini_combi.append(valid)
		Combi.append(np.max(np.array(mini_combi)))
		Epoch_Combi.append(np.argmax(np.array(mini_combi)))
	return Combi, Epoch_Combi

# def calculate_combi_scoreC(Filepaths):
	
	# Combi = []
	# Epoch_Combi = []
	# for filepath in Filepaths:
		# data = read_pickle(filepath)

		# mini_combi = []
		# for crc_v, mse_v, dme_v, dst_v, dku_v, crc_t, mse_t, dme_t, dst_t, dku_t in zip(data['CRC_Valid'], data['MSE_Valid'], data['DME_Valid'], data['DST_Valid'], data['DKU_Valid'], data['CRC_Train'], data['MSE_Train'], data['DME_Train'], data['DST_Train'], data['DKU_Train']):
			# # valid = crc_v / (mse_v + dme_v + dst_v + dku_v)
			# # train = crc_t/ (mse_t + dme_t + dst_t + dku_t)
			
			# # valid = crc_v - dme_v - dst_v - dku_v
			# # train = crc_t - dme_t - dst_t - dku_t
			
			# # valid = crc_v / mse_v - dme_v - dst_v
			# # train = crc_t / mse_t - dme_t - dst_t
			
			# # valid = crc_v / mse_v
			# # train = crc_t / mse_t
			
			# valid = 1/(mse_v)
			# # train = 1/(mse_t)
			
			# # valid = crc_v
			# # train = crc_t
			
			
			# mini_combi.append(valid)
		# Combi.append(np.max(np.array(mini_combi)))
		# Epoch_Combi.append(np.argmax(np.array(mini_combi)))
	# return Combi, Epoch_Combi

# def calculate_combi_scoreC(Filepaths):
	
	# Combi = []
	# Epoch_Combi = []
	# for filepath in Filepaths:
		# data = read_pickle(filepath)

		# mini_combi = []
		# for crc_v, mse_v, dme_v, dst_v, dku_v, crc_t, mse_t, dme_t, dst_t, dku_t in zip(data['CRC_Valid'], data['MSE_Valid'], data['DME_Valid'], data['DST_Valid'], data['DKU_Valid'], data['CRC_Train'], data['MSE_Train'], data['DME_Train'], data['DST_Train'], data['DKU_Train']):
			# # valid = crc_v / (mse_v + dme_v + dst_v + dku_v)
			# # train = crc_t/ (mse_t + dme_t + dst_t + dku_t)
			
			# # valid = crc_v - dme_v - dst_v - dku_v
			# # train = crc_t - dme_t - dst_t - dku_t
			
			# valid = mse_v
			# train = mse_t
			
			
			# mini_combi.append((valid + train)/2.)
		# Combi.append(np.max(np.array(mini_combi)))
		# Epoch_Combi.append(np.argmax(np.array(mini_combi)))
	# return Combi, Epoch_Combi

def mutation_pos(value, weight):

	random_value = np.random.uniform(-weight, weight, None)
	value = value * (1. + random_value)
	while value <= 0:
		random_value = np.random.uniform(-weight, weight, None)
		value = value * (1. + random_value)

	return value

def mutation_pos_codedlayer(value, weight):
	tuple_layers = uncode_tuple_layers(value)
	
	random_value = np.random.uniform(-weight, weight, None)
	
	new_value0 = tuple_layers[0] * (1. + random_value)
	while new_value0 <= 0:
		random_value = np.random.uniform(-weight, weight, None)
		new_value0 = tuple_layers[0] * (1. + random_value)
	
	if len(tuple_layers) == 2:
		new_value1 = tuple_layers[1] * (1. + random_value)
		while new_value1 <= 0:
			random_value = np.random.uniform(-weight, weight, None)
			new_value1 = tuple_layers[1] * (1. + random_value)
		
		new_tuple_layers = (int(new_value0), int(new_value1))
		value = code_tuple_layers(new_tuple_layers)
	
	
	elif len(tuple_layers) == 3:
		new_value1 = tuple_layers[1] * (1. + random_value)
		while new_value1 <= 0:
			random_value = np.random.uniform(-weight, weight, None)
			new_value1 = tuple_layers[1] * (1. + random_value)
		
		new_value2 = tuple_layers[2] * (1. + random_value)
		while new_value2 <= 0:
			random_value = np.random.uniform(-weight, weight, None)
			new_value2 = tuple_layers[2] * (1. + random_value)
		
		
		
		new_tuple_layers = (int(new_value0), int(new_value1), int(new_value2))
		value = code_tuple_layers(new_tuple_layers)
	
	
	else: 
		new_tuple_layers = (int(new_value0),)
		value = code_tuple_layers(new_tuple_layers)
	
	return int(value)

def mutation_ann(offspring, weight):
	for idx in range(offspring.shape[0]):
		offspring[idx, 0] = int(round(mutation_pos_codedlayer(offspring[idx, 0], weight))) #layer 
		offspring[idx, 1] = mutation_pos(offspring[idx, 1], weight) #alpha
		offspring[idx, 2] = mutation_pos(offspring[idx, 2], weight) #npre
		offspring[idx, 3] = mutation_pos(offspring[idx, 3], weight) #mpost
		offspring[idx, 4] = int(round(mutation_pos(offspring[idx, 4], weight))) #minibatches
		offspring[idx, 5] = int(round(mutation_pos(offspring[idx, 5], weight))) #epochs
	return offspring

def mutation_ann_5gen(offspring, weight):
	for idx in range(offspring.shape[0]):
		offspring[idx, 0] = int(round(mutation_pos_codedlayer(offspring[idx, 0], weight))) #layer 
		offspring[idx, 1] = mutation_pos(offspring[idx, 1], weight) #alpha
		offspring[idx, 2] = mutation_pos(offspring[idx, 2], weight) #npre
		offspring[idx, 3] = mutation_pos(offspring[idx, 3], weight) #mpost
		offspring[idx, 4] = int(round(mutation_pos(offspring[idx, 4], weight))) #minibatches
	return offspring	

def code_tuple_layers(tuple_layers):
	if len(tuple_layers) == 1:
		codlayers = str(1) + str(len(str(tuple_layers[0]))) + '0' + str(tuple_layers[0])
	elif len(tuple_layers) == 2:
		codlayers = str(2) + str(len(str(tuple_layers[0]))) + str(len(str(tuple_layers[1]))) + str(tuple_layers[0]) + str(tuple_layers[1])
	elif len(tuple_layers) == 3:
		codlayers = str(3) + str(len(str(tuple_layers[0]))) + str(len(str(tuple_layers[1]))) + str(len(str(tuple_layers[2]))) + str(tuple_layers[0]) + str(tuple_layers[1]) + str(tuple_layers[2])
	return codlayers
	
def uncode_tuple_layers(code_layers):
	code = str(code_layers)
	if code[0] == '1':
		uncode_layers = (float(code[3:]),)
	elif code[0] == '2':
		uncode_layers = (float(code[3:3+int(code[1])]), float(code[3+int(code[1]):]))
	elif code[0] == '3':
		uncode_layers = (float(code[4:4+int(code[1])]), float(code[4+int(code[1]):4+int(code[1])+int(code[2])]), float(code[4+int(code[1])+int(code[2]):]))
	else:
		print('fatal error 515485')
		sys.exit()
	return uncode_layers

def crossover_ann(parents, n_children):
	offspring_size = (n_children, 6) 		
	offspring = np.empty(offspring_size)		 
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]
		rint = randint(1, 5)
		if rint == 1:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent2_idx, 1] #alpha
			offspring[k, 2] = parents[parent2_idx, 2] #npre
			offspring[k, 3] = parents[parent2_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
			offspring[k, 5] = parents[parent2_idx, 5] #epochs
		elif rint == 2:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent2_idx, 2] #npre
			offspring[k, 3] = parents[parent2_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
			offspring[k, 5] = parents[parent2_idx, 5] #epochs
		elif rint == 3:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent1_idx, 2] #npre
			offspring[k, 3] = parents[parent2_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
			offspring[k, 5] = parents[parent2_idx, 5] #epochs
		elif rint == 4:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent1_idx, 2] #npre
			offspring[k, 3] = parents[parent1_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
			offspring[k, 5] = parents[parent2_idx, 5] #epochs
		elif rint == 5:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent1_idx, 2] #npre
			offspring[k, 3] = parents[parent1_idx, 3] #mpost
			offspring[k, 4] = parents[parent1_idx, 4] #minibatches
			offspring[k, 5] = parents[parent2_idx, 5] #epochs
	return offspring

def crossover_ann_5gen(parents, n_children):
	offspring_size = (n_children, 5) 		
	offspring = np.empty(offspring_size)		 
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]
		rint = randint(1, 4)
		if rint == 1:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent2_idx, 1] #alpha
			offspring[k, 2] = parents[parent2_idx, 2] #npre
			offspring[k, 3] = parents[parent2_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
		elif rint == 2:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent2_idx, 2] #npre
			offspring[k, 3] = parents[parent2_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
		elif rint == 3:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent1_idx, 2] #npre
			offspring[k, 3] = parents[parent2_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
		elif rint == 4:
			offspring[k, 0] = parents[parent1_idx, 0] #layer
			offspring[k, 1] = parents[parent1_idx, 1] #alpha
			offspring[k, 2] = parents[parent1_idx, 2] #npre
			offspring[k, 3] = parents[parent1_idx, 3] #mpost
			offspring[k, 4] = parents[parent2_idx, 4] #minibatches
	return offspring
			

if __name__ == '__main__':
	main(sys.argv)


