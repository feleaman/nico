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
from pykalman import UnscentedKalmanFilter
from pykalman import KalmanFilter
from pykalman import AdditiveUnscentedKalmanFilter

from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import Pipeline
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':1, 'sheet':0, 'train':0.7, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature2':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature_array':['RMS']}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	

	if config['mode'] == 'fuse_kalman':

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
		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])

		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:

				example.append(newdict[feature][i])
			Features.append(example)
		
		Features = np.array(Features)

		
		plt.plot(Features)
		plt.show()
		scaler_model = StandardScaler()
		scaler_model.fit(Features)
		Features = scaler_model.transform(Features)

		plt.plot(Features)
		plt.show()


		kf = UnscentedKalmanFilter(n_dim_state=1, n_dim_obs=len(config['feature_array']), random_state=1)
		# kf = AdditiveUnscentedKalmanFilter(n_dim_state=1, n_dim_obs=6, random_state=1)
		# kf = KalmanFilter(n_dim_state=1, n_dim_obs=len(config['feature_array']), random_state=1)
		measurements = [i for i in Features]

		
		
		# kf.em(measurements)	

		
		z = kf.smooth(measurements)
		z = z[0]
		
		plt.plot(z)
		plt.show()
	
	elif config['mode'] == 'testtt':
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
		
		plt.plot(Feature)
		plt.show()
	
	elif config['mode'] == 'fuse_factor':

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
		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])

		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:

				example.append(newdict[feature][i])
			Features.append(example)
		
		Features = np.array(Features)

		
		plt.plot(Features)
		plt.show()
		scaler_model = StandardScaler()
		scaler_model.fit(Features)
		Features = scaler_model.transform(Features)
		plt.plot(Features)
		plt.show()
		


		fa = FactorAnalysis(n_components=1, random_state=1)
		fa.fit(Features)

		z = fa.transform(Features)
		
		
		plt.plot(z)
		plt.show()
	
	elif config['mode'] == 'fuse_pca':

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
		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])

		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:

				example.append(newdict[feature][i])
			Features.append(example)
		
		Features = np.array(Features)

		
		plt.plot(Features)
		plt.show()
		
		
		scaler_model = StandardScaler()
		scaler_model.fit(Features)
		Features = scaler_model.transform(Features)
		
		
		plt.plot(Features)
		plt.show()
		

		
		from sklearn.cluster import FeatureAgglomeration
		from sklearn.decomposition import FastICA
		from sklearn.decomposition import KernelPCA
		
		FeatureAgglomeration
		
		# model = PCA(n_components=len(config['feature_array']))
		model = PCA(n_components=2)
		# model = FeatureAgglomeration(n_clusters=1)
		# model = FastICA(n_components=1)
		# model = KernelPCA(n_components=1, kernel='rbf')
		# model = KernelPCA(n_components=len(config['feature_array']), kernel='rbf')
		
		model.fit(Features)
		Features = model.transform(Features)
		
		print(model.explained_variance_)
		print(model.explained_variance_ratio_)


		plt.plot(Features)
		plt.show()
		
		
		
		
		# TFeatures = np.transpose(Features)
		# plt.plot(TFeatures[0])
		# plt.show()		
		
		# mydict_out = {}
		# mydict_out['FuFFT_ma1'] = TFeatures[0]


		# writer = pd.ExcelWriter('MASTER_Features.xlsx')			
		# DataFr = pd.DataFrame(data=mydict_out, index=rownames)
		# DataFr.to_excel(writer, sheet_name='AE_Features')
		# writer.close()
		
		
	
	
	elif config['mode'] == 'fuse_corr':
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
		
		Features = {}
		for key, values in mydict.items():
			Features[key] = movil_avg(mydict[key], config['n_mov_avg'])

		

		
		CorrCoefs = {}
		for feature1 in config['feature_array']:
			mylist = []
			for feature2 in config['feature_array']:
				value = np.corrcoef(Features[feature1], Features[feature2])[0][1]
				mylist.append(value)
			CorrCoefs[feature1] = mylist
				



		


		writer = pd.ExcelWriter('Corr_Features.xlsx')			
		DataFr = pd.DataFrame(data=CorrCoefs, index=config['feature_array'])
		DataFr.to_excel(writer, sheet_name='Corr_Features')
		writer.close()
	
	elif config['mode'] == 'test_cluster':
		
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
		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])

		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:

				example.append(newdict[feature][i])
			Features.append(example)
		
		Features = np.array(Features)
		
		
		from sklearn.cluster import DBSCAN, AffinityPropagation
		
		model = AffinityPropagation()
		print(model.fit_predict(Features))
	
	
	elif config['mode'] == 'fuse_mlp':

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
		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])

		
		Features = []
		for i in range(length_data):
			example = []
			for feature in config['feature_array']:

				example.append(newdict[feature][i])
			Features.append(example)
		
		Features = np.array(Features)

		
		plt.plot(Features)
		plt.show()
		
		
		scaler_model = StandardScaler()
		scaler_model.fit(Features)
		Features = scaler_model.transform(Features)
		
		
		plt.plot(Features)
		plt.show()
		

		nn = MLPRegressor(hidden_layer_sizes=(5), activation='tanh', solver='lbfgs0', alpha=1.e1)
		nn.fit(X=Features, y=np.linspace(0, 1, length_data))
		
		z = nn.predict(Features)
		

		# Features = pca.transform(Features)


		# corr = []
		# TFeatures = np.transpose(Features)
		# for feature_pca in TFeatures:
			# corr.append(np.corrcoef(np.ravel(feature_pca), np.arange(len(feature_pca)))[0][1])		
		
		# z = TFeatures[np.argmax(np.absolute(corr))]
		# print(corr)
		plt.plot(z)
		plt.show()
	
	
	elif config['mode'] == 'predict_from_xls_mlp':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			# Feature += mydict[config['feature']][:-2]
			Feature += mydict[config['feature']]
		Feature = list(np.nan_to_num(Feature))
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		

		
		
		
		
		Feature = np.array(Feature)
		# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
		# ax[0].plot(Feature, 'r')
		Feature = median_filter(data=Feature, points=5, same_length=True)
		# ax[1].plot(Feature, 'b')
		# plt.show()
		x_Feature = np.arange(len(Feature))
				
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))				
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	

		clf = MLPRegressor(solver=config['solver'], alpha=config['alpha'], hidden_layer_sizes=config['layers'], random_state=config['rs'], activation=config['activation'], tol=config['tol'], verbose=True, max_iter=config['max_iter'])
		
		# from sklearn.tree import DecisionTreeRegressor
		# clf = DecisionTreeRegressor()
		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train))
		n_ex = len(Train) - n_pre - m_post
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []
		# plt.plot(Train, 'k')
		# plt.show()
		for k in range(n_ex + 1):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			
			# aa = np.arange(len(Train[k : k + n_pre]))
			# plt.plot(aa, Train[k : k + n_pre], 'b')
			# bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# plt.show()
		# sys.exit()
			
		
		from sklearn.model_selection import KFold, GroupKFold
		kf = KFold(n_splits=100, shuffle=True)
	
		# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 19]])
		# y = np.array([[3], [5], [7], [9], [11], [13], [15], [17], [19]])
		# # X = T_Inputs
		# # y = T_Outputs
		# for i in range(3):
			# for train_index, test_index in kf.split(X):
				# print("TRAIN:", train_index, "TEST:", test_index)
				# # X_train, X_test = X[train_index], X[test_index]
				# # y_train, y_test = y[train_index], y[test_index]
		# sys.exit()
		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		epochs = 10
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
			P_Input = It_Train[n_ex + k + 1 : n_ex + n_pre + k + 1]

			P_Output = clf.predict([P_Input])
			P_Output = P_Output[0]
			
			
			Predict.append(P_Output[-1])
			It_Train.append(P_Output[-1])

		Predict = Predict[:-(m_post-1)]
		
		
		fig, ax = plt.subplots()
		ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		ax.set_ylabel('Health Index', fontsize=13)
		ax.set_title('Linear Regression', fontsize=13)
		fact = 5./3600.
		ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		ax.plot(x_Predict*fact, Predict, 'r', label='Prediction')
		ax.plot(x_Train*fact, Train, 'k', label='Training')
		ax.legend()
		plt.show()
		
		
		plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		plt.show()
	
	# elif config['mode'] == 'predict_from_xls_mlp':
		# print('Select xls')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		# Feature = []		
		# for filepath in Filepaths:
			# mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			# mydict = mydict.to_dict(orient='list')
			# # Feature += mydict[config['feature']][:-2]
			# Feature += mydict[config['feature']]
		# Feature = list(np.nan_to_num(Feature))
		# Feature = movil_avg(Feature, config['n_mov_avg'])
		
		

		
		
		
		
		# Feature = np.array(Feature)
		# # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
		# # ax[0].plot(Feature, 'r')
		# Feature = median_filter(data=Feature, points=5, same_length=True)
		# # ax[1].plot(Feature, 'b')
		# # plt.show()
		# x_Feature = np.arange(len(Feature))
				
		# Train = Feature[0:int(config['train']*len(Feature))]
		# x_Train = np.arange(float(len(Train)))				
		
		# x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# # scaler = StandardScaler()
		# # scaler = RobustScaler()
		# # scaler.fit(Train)
		# # Train = scaler.transform(Train)	

		# clf = MLPRegressor(solver=config['solver'], alpha=config['alpha'], hidden_layer_sizes=config['layers'], random_state=config['rs'], activation=config['activation'], tol=config['tol'], verbose=True, max_iter=config['max_iter'])
		
		# # from sklearn.tree import DecisionTreeRegressor
		# # clf = DecisionTreeRegressor()
		
		# n_pre = int(config['n_pre']*len(Train))
		# m_post = int(config['m_post']*len(Train))
		# n_ex = len(Train) - n_pre - m_post
		# print('+++++++++++++Info: Input points n = ', n_pre)
		# print('+++++++++++++Info: Output points m = ', m_post)
		# print('+++++++++++++Info: Training examples = ', n_ex)
		# a = input('enter to continue...')
		# T_Inputs = []
		# T_Outputs = []
		# # plt.plot(Train, 'k')
		# # plt.show()
		# for k in range(n_ex + 1):

			# T_Inputs.append(Train[k : k + n_pre])
			# T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
			
			# # aa = np.arange(len(Train[k : k + n_pre]))
			# # plt.plot(aa, Train[k : k + n_pre], 'b')
			# # bb = np.max(aa) + np.arange(len(Train[k + n_pre : k + n_pre + m_post]))
			# # plt.plot(bb, Train[k + n_pre : k + n_pre + m_post], 'r')
			# # plt.show()
		# # sys.exit()
			
		
		# from sklearn.model_selection import KFold, GroupKFold
		# kf = KFold(n_splits=100, shuffle=True)
	
		# # X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 19]])
		# # y = np.array([[3], [5], [7], [9], [11], [13], [15], [17], [19]])
		# # # X = T_Inputs
		# # # y = T_Outputs
		# # for i in range(3):
			# # for train_index, test_index in kf.split(X):
				# # print("TRAIN:", train_index, "TEST:", test_index)
				# # # X_train, X_test = X[train_index], X[test_index]
				# # # y_train, y_test = y[train_index], y[test_index]
		# # sys.exit()
		# T_Inputs = np.array(T_Inputs)
		# T_Outputs = np.array(T_Outputs)
		# count = 0
		# epochs = 10
		# for i in range(epochs):
			# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			# numb = 0
			# for train_index, test_index in kf.split(T_Inputs):
				# print('+++++++++++Batch', numb)
				# T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				# T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				# clf.partial_fit(T_Inputs_test, T_Outputs_test)
				# numb += 1
			# count += 1
				
			
			

		
		
		
		# # clf.fit(T_Inputs, T_Outputs)
		# print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		# Predict = []
		# It_Train = list(Train)

		# for k in range(len(x_Predict) + m_post - 1):
			# P_Input = It_Train[n_ex + k + 1 : n_ex + n_pre + k + 1]

			# P_Output = clf.predict([P_Input])
			# P_Output = P_Output[0]
			
			
			# Predict.append(P_Output[-1])
			# It_Train.append(P_Output[-1])

		# Predict = Predict[:-(m_post-1)]
		
		
		# fig, ax = plt.subplots()
		# ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		# ax.set_ylabel('Health Index', fontsize=13)
		# ax.set_title('Linear Regression', fontsize=13)
		# fact = 5./3600.
		# ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		# ax.plot(x_Predict*fact, Predict, 'r', label='Prediction')
		# ax.plot(x_Train*fact, Train, 'k', label='Training')
		# ax.legend()
		# plt.show()
		
		
		# plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		# plt.show()
	
	elif config['mode'] == 'predict_from_xls_svr':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		Feature = list(np.nan_to_num(Feature))
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		Feature = np.array(Feature)
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))				
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	

		# clf = NuSVR(nu=0.5, C=1.0, kernel='poly')
		clf = SVR(kernel='poly', verbose=True, degree=6, C=0.5)
		
		
		n_pre = int(config['n_pre']*len(Train))
		m_post = 1
		# m_post = int(config['m_post']*len(Train))
		n_ex = len(Train) - n_pre - m_post
		

		
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex + 1):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
		
		clf.fit(T_Inputs, T_Outputs)
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		Predict = []
		It_Train = list(Train)

		for k in range(len(x_Predict) + m_post - 1):
			P_Input = It_Train[n_ex + k + 1 : n_ex + n_pre + k + 1]

			P_Output = clf.predict([P_Input])
			P_Output = P_Output[0]
			
			
			Predict.append(P_Output)
			It_Train.append(P_Output)

		# Predict = Predict[:-(m_post-1)]
		
		fig, ax = plt.subplots()
		ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		ax.set_ylabel('Health Index', fontsize=13)
		ax.set_title('Linear Regression', fontsize=13)
		fact = 5./3600.
		ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		ax.plot(x_Predict*fact, Predict, 'r', label='Prediction')
		ax.plot(x_Train*fact, Train, 'k', label='Training')
		ax.legend()
		plt.show()
		
		
		# plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		# plt.show()
	
	
	elif config['mode'] == 'predict_from_xls_lin':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		Feature = list(np.nan_to_num(Feature))
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		
		# Feature = median_filter(data=Feature, points=5, same_length=True)
		
		Feature = np.array(Feature)
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))		
		
		slope, intercept, r_value, p_value, std_err = stats.linregress(x_Train, Train)
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)

		Predict = slope*x_Predict + intercept
		
		fig, ax = plt.subplots()
		ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		ax.set_ylabel('Health Index', fontsize=13)
		# ax.set_title('Linear Regression', fontsize=13)
		ax.set_title('Neural Network', fontsize=13)
		fact = 5./3600.
		ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		ax.plot(x_Predict*fact, Predict, 'r', label='Prediction')
		ax.plot(x_Train*fact, Train, 'k', label='Training')
		ax.legend()
		plt.show()
	
		print(len(Feature))
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
