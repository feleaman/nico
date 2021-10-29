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
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.6, 'valid':0.2, 'n_pre':0.2, 'm_post':0.1, 'alpha_pro':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers_pro':[10], 'solver_pro':'adam', 'rs_pro':1, 'activation_pro':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature2':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature_array':['RMS'], 'layers_fus':[10], 'rs_fus':1, 'alpha_fus':1, 'solver_fus':'lbfgs', 'activation_fus':'identity', 'source_file':'OFF'}

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
		List_RSs = [1]
		
		for layers in List_Layers:
			rs_error = 0
			rs_error_final = 0
			for rs_pro in List_RSs:
				nn_pro = MLPRegressor(hidden_layer_sizes=layers, activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=rs_pro)
				
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
				error_final = np.absolute(fused_predict[length_test - 1] - fused[length_data - 1])
				print('error= ', error)
				print('error_final= ', error_final)
				
				rs_error += error
				rs_error_final += error_final
				
				
			ERROR.append(rs_error/len(List_RSs))
			ERROR_FINAL.append(rs_error_final/len(List_RSs))
		
		
		mydict = {'Error_Final':ERROR_FINAL, 'Error':ERROR}
		writer = pd.ExcelWriter('regre_' + config['name'] + '.xlsx')			
		DataFr = pd.DataFrame(data=mydict, index=List_Layers)
		DataFr.to_excel(writer, sheet_name='Result')
		writer.close()
		
		
		mydict = {'alpha':config['alpha_pro'], 'solver':config['solver_pro'], 'activation':config['activation_pro'], 'rs':config['rs_pro'], 'n_pre':config['n_pre'], 'm_post':config['m_post'], 'n_mov_avg':config['n_mov_avg'], 'train':config['train'], 'feature':config['feature']}
		writer = pd.ExcelWriter('config_' + config['name'] + '.xlsx')			
		DataFr = pd.DataFrame(data=mydict, index=['value'])
		DataFr.to_excel(writer, sheet_name='Result')
		writer.close()
	
	
	elif config['mode'] == 'mode2b':
		
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
		
		# if config['layers_pro'][0] == 0:
			# print('Auto config of layers')
			# List_Layers = list(np.arange(m_post, n_pre, int((n_pre-m_post)*0.1)))
			# # List_Layers = [50, 100]
		# else:
			# print('Auto config of layers IS not optional')
			# sys.exit()
			
		
		ERROR = []
		ERROR_FINAL = []
		# print(List_Layers)
		# print(int((n_pre-m_post)*0.1))
		# print(np.arange(n_pre, m_post, int((n_pre-m_post)*0.1)))
		# sys.exit()
		# List_RSs = [1]
		
		rs_error = 0
		rs_error_final = 0
		nn_pro = MLPRegressor(hidden_layer_sizes=config['layers_pro'], activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=config['rs_pro'])
		
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
		
		
		
		
		error = 0
		for i in range(len(fused_predict)):
			error += (fused_predict[i] - fused[length_train + i])**2.0
		error_final = np.absolute(fused_predict[length_test - 1] - fused[length_data - 1])
		print('error= ', error)
		print('error_final= ', error_final)
		
		plt.plot(x_full, fused, 'b', x_predict, fused_predict, 'r', x_train, fused_train, 'k')
		plt.show()
		
		# rs_error += error
		# rs_error_final += error_final
			
			
		# ERROR.append(rs_error/len(List_RSs))
		# ERROR_FINAL.append(rs_error_final/len(List_RSs))
		
		
		# mydict = {'Error_Final':ERROR_FINAL, 'Error':ERROR}
		# writer = pd.ExcelWriter('regre_' + config['name'] + '.xlsx')			
		# DataFr = pd.DataFrame(data=mydict, index=List_Layers)
		# DataFr.to_excel(writer, sheet_name='Result')
		# writer.close()
		
		
		# mydict = {'alpha':config['alpha_pro'], 'solver':config['solver_pro'], 'activation':config['activation_pro'], 'rs':config['rs_pro'], 'n_pre':config['n_pre'], 'm_post':config['m_post'], 'n_mov_avg':config['n_mov_avg'], 'train':config['train'], 'feature':config['feature']}
		# writer = pd.ExcelWriter('config_' + config['name'] + '.xlsx')			
		# DataFr = pd.DataFrame(data=mydict, index=['value'])
		# DataFr.to_excel(writer, sheet_name='Result')
		# writer.close()
	
	
	elif config['mode'] == 'mode2c': #para run automatico, con train, valid... el test no se usa
		
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
		
		
		Full_Feature = newdict[config['feature']]
		# Full_Feature = np.arange(100)
		# TV_Feature = Full_Feature[0:int(config['valid']*len(Full_Feature))]
		
		Features = np.array(Full_Feature)
		Features_Train = np.array(Full_Feature[0:int(config['train']*len(Full_Feature))])
		Features_Valid = np.array(Full_Feature[int(config['train']*len(Full_Feature)) : int(config['train']*len(Full_Feature)) + int(config['valid']*len(Full_Feature))])
		
		Features_Test = np.array(Full_Feature[int(config['train']*len(Full_Feature)) + int(config['valid']*len(Full_Feature)) :])
		
		
		# Features_Train = []
		# for i in range(length_train):
			# example = TV_Feature[i]
			# Features_Train.append(example)		
		# Features_Train = np.array(Features_Train)
		
		# Features_Test = []
		# for i in range(length_test):
			# example = TV_Feature[i+length_train]
			# Features_Test.append(example)		
		# Features_Test = np.array(Features_Test)
		
		# Features = []
		# for i in range(length_data):
			# example = TV_Feature[i]
			# Features.append(example)		
		# Features = np.array(Features)
		
		# print(len(Features))
		# print(len(Features_Test))
		# print(len(Features_Train))

		
		
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
		fused_valid = np.ravel(Features_Valid)
		
		fused_test = np.ravel(Features_Test)
		fused = np.ravel(Features)
		# print(fused_train)
		# print(fused_valid)
		# print(fused_test)
		# sys.exit()

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
		List_RSs = [1, 7, 12]
		
		for layers in List_Layers:
			rs_error = 0
			rs_error_final = 0
			for rs_pro in List_RSs:
				nn_pro = MLPRegressor(hidden_layer_sizes=layers, activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=rs_pro, max_iter=100000)
				
				T_Inputs = []
				T_Outputs = []
				for k in range(n_ex):
					T_Inputs.append(fused_train[k : k + n_pre])
					T_Outputs.append(fused_train[k + n_pre : k + n_pre + m_post])
					
				nn_pro.fit(T_Inputs, T_Outputs)
				print(T_Inputs)
				

				
				fused_predict = []
				it_fused = list(fused_train)

				for k in range(len(fused_valid) + m_post - 1):
					P_Input = it_fused[n_ex + k + 1 : n_ex + n_pre + k + 1]
					# print(P_Input)
					# sys.exit()
					P_Output = nn_pro.predict([P_Input])
					P_Output = P_Output[0]			
					
					fused_predict.append(P_Output[-1])
					it_fused.append(P_Output[-1])

				fused_predict = np.array(fused_predict[:-(m_post-1)])
				
				# plt.plot(fused_predict, 'r', fused_test, 'b')
				
				# x_full = np.arange((len(fused)))
				# x_train = np.arange((len(fused_train)))	
				# x_predict = np.linspace(len(fused_train), len(fused), num=len(fused_test), endpoint=False)
				
				
				# plt.plot(x_full, fused, 'b', x_predict, fused_predict, 'r', x_train, fused_train, 'k')
				# plt.show()
				
				error = 0
				for i in range(len(fused_predict)):
					error += (fused_predict[i] - fused_valid[i])**2.0
				# print(fused_predict)
				# print(fused_valid)
				error_final = np.absolute(fused_predict[-1] - fused_valid[-1])
				print('error= ', error)
				print('error_final= ', error_final)
				
				rs_error += error
				rs_error_final += error_final
				
				
			ERROR.append(rs_error/len(List_RSs))
			ERROR_FINAL.append(rs_error_final/len(List_RSs))
		
		
		mydict = {'Error_Final':ERROR_FINAL, 'Error':ERROR}
		writer = pd.ExcelWriter('train_' +  str(config['train']) + '_regre_' + config['name'] + '.xlsx')			
		DataFr = pd.DataFrame(data=mydict, index=List_Layers)
		DataFr.to_excel(writer, sheet_name='Result')
		writer.close()
		
		
		mydict = {'alpha':config['alpha_pro'], 'solver':config['solver_pro'], 'activation':config['activation_pro'], 'rs':config['rs_pro'], 'n_pre':config['n_pre'], 'm_post':config['m_post'], 'n_mov_avg':config['n_mov_avg'], 'train':config['train'], 'feature':config['feature'], 'valid':config['valid']}
		writer = pd.ExcelWriter('train_' + str(config['train']) + '_config_' + config['name'] + '.xlsx')			
		DataFr = pd.DataFrame(data=mydict, index=['value'])
		DataFr.to_excel(writer, sheet_name='Result')
		writer.close()
	
	
	
	elif config['mode'] == 'mode2d': #para run manual, con train, valid y test
		
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
		
		
		Full_Feature = newdict[config['feature']]
		# Full_Feature = np.arange(100)
		# TV_Feature = Full_Feature[0:int(config['valid']*len(Full_Feature))]
		
		Features = np.array(Full_Feature)
		Features_Train = np.array(Full_Feature[0:int(config['train']*len(Full_Feature))])
		Features_Valid = np.array(Full_Feature[int(config['train']*len(Full_Feature)) : int(config['train']*len(Full_Feature)) + int(config['valid']*len(Full_Feature))])
		
		Features_Test = np.array(Full_Feature[int(config['train']*len(Full_Feature)) + int(config['valid']*len(Full_Feature)) :])
		
		
		# Features_Train = []
		# for i in range(length_train):
			# example = TV_Feature[i]
			# Features_Train.append(example)		
		# Features_Train = np.array(Features_Train)
		
		# Features_Test = []
		# for i in range(length_test):
			# example = TV_Feature[i+length_train]
			# Features_Test.append(example)		
		# Features_Test = np.array(Features_Test)
		
		# Features = []
		# for i in range(length_data):
			# example = TV_Feature[i]
			# Features.append(example)		
		# Features = np.array(Features)
		
		# print(len(Features))
		# print(len(Features_Test))
		# print(len(Features_Train))

		
		
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
		fused_valid = np.ravel(Features_Valid)
		
		fused_test = np.ravel(Features_Test)
		fused = np.ravel(Features)
		# print(fused_train)
		# print(fused_valid)
		# print(fused_test)
		# sys.exit()

		n_pre = int(config['n_pre']*length_train)
		m_post = int(config['m_post']*length_train)
		n_ex = length_train + 1 - n_pre - m_post
		
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		# a = input('enter to continue...')
		
		# if config['layers_pro'][0] == 0:
			# print('Auto config of layers')
			# List_Layers = list(np.arange(m_post, n_pre, int((n_pre-m_post)*0.1)))
			# # List_Layers = [50, 100]
		# else:
			# print('Auto config of layers IS not optional')
			# sys.exit()
			

		# print(List_Layers)
		# print(int((n_pre-m_post)*0.1))
		# print(np.arange(n_pre, m_post, int((n_pre-m_post)*0.1)))
		# sys.exit()

		nn_pro = MLPRegressor(hidden_layer_sizes=config['layers_pro'], activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=config['rs_pro'], max_iter=100000)
		
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex):
			T_Inputs.append(fused_train[k : k + n_pre])
			T_Outputs.append(fused_train[k + n_pre : k + n_pre + m_post])
			
		nn_pro.fit(T_Inputs, T_Outputs)
		print(T_Inputs)
		

		
		fused_predict_valid = []
		it_fused = list(fused_train)

		for k in range(len(fused_valid) + m_post - 1):
			P_Input = it_fused[n_ex + k + 1 : n_ex + n_pre + k + 1]
			# print(P_Input)
			# sys.exit()
			P_Output = nn_pro.predict([P_Input])
			P_Output = P_Output[0]			
			
			fused_predict_valid.append(P_Output[-1])
			it_fused.append(P_Output[-1])

		fused_predict_valid = np.array(fused_predict_valid[:-(m_post-1)])
		
		# plt.plot(fused_predict, 'r', fused_test, 'b')
		
		# x_full = np.arange((len(fused)))
		# x_train = np.arange((len(fused_train)))	
		# x_predict = np.linspace(len(fused_train), len(fused), num=len(fused_test), endpoint=False)
		
		
		# plt.plot(x_full, fused, 'b', x_predict, fused_predict, 'r', x_train, fused_train, 'k')
		# plt.show()
		
		error = 0
		for i in range(len(fused_predict_valid)):
			error += (fused_predict_valid[i] - fused_valid[i])**2.0
		# print(fused_predict)
		# print(fused_valid)
		error_final = np.absolute(fused_predict_valid[-1] - fused_valid[-1])
		print('error validacion= ', error)
		print('error_final validacion= ', error_final)
		
		
		
		
		
		fused_predict_test = []
		it_fused = list(fused_train) + list(fused_valid)

		for k in range(len(fused_test) + m_post - 1):
			P_Input = it_fused[len(fused_valid) + n_ex + k + 1 : len(fused_valid) + n_ex + n_pre + k + 1]
			# print(P_Input)
			# sys.exit()
			P_Output = nn_pro.predict([P_Input])
			P_Output = P_Output[0]			
			
			fused_predict_test.append(P_Output[-1])
			it_fused.append(P_Output[-1])

		fused_predict_test = np.array(fused_predict_test[:-(m_post-1)])
		
		
		error = 0
		for i in range(len(fused_predict_test)):
			error += (fused_predict_test[i] - fused_test[i])**2.0
		# print(fused_predict)
		# print(fused_valid)
		error_final = np.absolute(fused_predict_test[-1] - fused_test[-1])
		print('error test= ', error)
		print('error_final test= ', error_final)
		
		x_full = np.arange((len(fused)))
		x_train = np.arange((len(fused_train)))	
		x_valid = np.linspace(len(fused_train), len(fused_train)+len(fused_valid), num=len(fused_valid), endpoint=False)
		x_test = np.linspace(len(fused_train)+len(fused_valid), len(fused_train)+len(fused_valid)+len(fused_test), num=len(fused_test), endpoint=False)
		
		
		plt.plot(x_full, fused, 'b', x_train, fused_train, 'k', x_valid, fused_predict_valid, 'r', x_test, fused_predict_test, 'm')
		plt.show()


		
		
		# mydict = {'Error_Final':ERROR_FINAL, 'Error':ERROR}
		# writer = pd.ExcelWriter('train_' +  str(config['train']) + '_regre_' + config['name'] + '.xlsx')			
		# DataFr = pd.DataFrame(data=mydict, index=List_Layers)
		# DataFr.to_excel(writer, sheet_name='Result')
		# writer.close()
		
		
		# mydict = {'alpha':config['alpha_pro'], 'solver':config['solver_pro'], 'activation':config['activation_pro'], 'rs':config['rs_pro'], 'n_pre':config['n_pre'], 'm_post':config['m_post'], 'n_mov_avg':config['n_mov_avg'], 'train':config['train'], 'feature':config['feature'], 'valid':config['valid']}
		# writer = pd.ExcelWriter('train_' + str(config['train']) + '_config_' + config['name'] + '.xlsx')			
		# DataFr = pd.DataFrame(data=mydict, index=['value'])
		# DataFr.to_excel(writer, sheet_name='Result')
		# writer.close()
	
	
	elif config['mode'] == 'mode2e': #para run manual, con train y test-- el valid es para entrenar
		
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
		length_train = int((config['train']+config['valid'])*len(rownames))
		length_test = length_data - length_train
		
		mydict = mydict.to_dict(orient='list')		
		newdict = {}
		for key, values in mydict.items():
			newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])
		
		
		Full_Feature = newdict[config['feature']]
		# Full_Feature = np.arange(100)
		# TV_Feature = Full_Feature[0:int(config['valid']*len(Full_Feature))]
		
		Features = np.array(Full_Feature)
		Features_Train = np.array(Full_Feature[0:int((config['train']+config['valid'])*len(Full_Feature))])
		
		Features_Test = np.array(Full_Feature[int(config['train']*len(Full_Feature)) + int(config['valid']*len(Full_Feature)) :])
		
		
		# Features_Train = []
		# for i in range(length_train):
			# example = TV_Feature[i]
			# Features_Train.append(example)		
		# Features_Train = np.array(Features_Train)
		
		# Features_Test = []
		# for i in range(length_test):
			# example = TV_Feature[i+length_train]
			# Features_Test.append(example)		
		# Features_Test = np.array(Features_Test)
		
		# Features = []
		# for i in range(length_data):
			# example = TV_Feature[i]
			# Features.append(example)		
		# Features = np.array(Features)
		
		# print(len(Features))
		# print(len(Features_Test))
		# print(len(Features_Train))

		
		
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
		print(fused_train)
		fused_test = np.ravel(Features_Test)
		fused = np.ravel(Features)
		
		# fused = np.arange(100)
		# fused_train = np.arange(60)
		# fused_test = np.arange(60, 100, 1)
		
		# length_data = 100
		# length_train = 60
		# length_test = 40
		
		
		print(fused_train)
		# print(fused_test)
		# sys.exit()

		n_pre = int(config['n_pre']*length_train)
		m_post = int(config['m_post']*length_train)
		n_ex = length_train + 1 - n_pre - m_post
		
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		# a = input('enter to continue...')
		
		# if config['layers_pro'][0] == 0:
			# print('Auto config of layers')
			# List_Layers = list(np.arange(m_post, n_pre, int((n_pre-m_post)*0.1)))
			# # List_Layers = [50, 100]
		# else:
			# print('Auto config of layers IS not optional')
			# sys.exit()
			

		# print(List_Layers)
		# print(int((n_pre-m_post)*0.1))
		# print(np.arange(n_pre, m_post, int((n_pre-m_post)*0.1)))
		# sys.exit()

		nn_pro = MLPRegressor(hidden_layer_sizes=config['layers_pro'], activation=config['activation_pro'], solver=config['solver_pro'], alpha=config['alpha_pro'], random_state=config['rs_pro'], max_iter=100000)
		
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex):
			T_Inputs.append(fused_train[k : k + n_pre])
			T_Outputs.append(fused_train[k + n_pre : k + n_pre + m_post])
		
		
		
		nn_pro.fit(T_Inputs, T_Outputs)
				

		
		
		fused_predict_test = []
		it_fused = list(fused_train)

		for k in range(len(fused_test) + m_post - 1):
			P_Input = it_fused[n_ex + k + 1 :n_ex + n_pre + k + 1]
			# print(P_Input)
			# sys.exit()
			# print(P_Input)
			P_Output = nn_pro.predict([P_Input])
			P_Output = P_Output[0]			
			
			fused_predict_test.append(P_Output[-1])
			it_fused.append(P_Output[-1])

		fused_predict_test = np.array(fused_predict_test[:-(m_post-1)])
		
		
		error = 0
		for i in range(len(fused_predict_test)):
			error += (fused_predict_test[i] - fused_test[i])**2.0
		# print(fused_predict)
		# print(fused_valid)
		error_final = np.absolute(fused_predict_test[-1] - fused_test[-1])
		print('error test= ', error)
		print('error_final test= ', error_final)
		
		x_full = np.arange((len(fused)))
		x_train = np.arange((len(fused_train)))	
		x_predict = np.linspace(len(fused_train), len(fused), num=len(fused_test), endpoint=False)
		
		
		plt.plot(x_full, fused, 'b', x_predict, fused_predict_test, 'r', x_train, fused_train, 'k')
		plt.show()


		
		
		# mydict = {'Error_Final':ERROR_FINAL, 'Error':ERROR}
		# writer = pd.ExcelWriter('train_' +  str(config['train']) + '_regre_' + config['name'] + '.xlsx')			
		# DataFr = pd.DataFrame(data=mydict, index=List_Layers)
		# DataFr.to_excel(writer, sheet_name='Result')
		# writer.close()
		
		
		# mydict = {'alpha':config['alpha_pro'], 'solver':config['solver_pro'], 'activation':config['activation_pro'], 'rs':config['rs_pro'], 'n_pre':config['n_pre'], 'm_post':config['m_post'], 'n_mov_avg':config['n_mov_avg'], 'train':config['train'], 'feature':config['feature'], 'valid':config['valid']}
		# writer = pd.ExcelWriter('train_' + str(config['train']) + '_config_' + config['name'] + '.xlsx')			
		# DataFr = pd.DataFrame(data=mydict, index=['value'])
		# DataFr.to_excel(writer, sheet_name='Result')
		# writer.close()
	

		
		
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
	config['valid'] = float(config['valid'])
	
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
