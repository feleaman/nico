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
from m_plots import *

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import RFE
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
from sklearn.decomposition import PCA
from sklearn import tree
# import graphviz 
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import KFold, GroupKFold
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold, GroupKFold
from random import randint
Inputs = ['mode']


InputsOpt_Defaults = {'plot':'OFF', 'rs':0, 'save':'ON', 'scaler':None, 'pca_comp':0, 'mypath':None, 'name':'auto', 'feature':'RMS', 'train':0.60, 'n_mov_avg':1, 'predict_node':'last', 'mini_batches':10, 'activation':'identity', 'alpha':0.001, 'epochs':10, 'hidden_layers':[50], 'valid':0.2, 'n_pre':0.3, 'm_post':0.05, 'auto_layers':None, 'save_plot':'ON', 'n_bests':3, 'weight':0.05, 'n_children':7, 'save_model':'OFF', 'train_test_ref':'OFF', 'early_stop':'OFF', 'stop_epochs':10, 'kernel':'linear', 'degree':3, 'C':1., 'score':'mse_valid'}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	

	
	if config['mode'] == 'ann_test':		
		#++++++++++++++++++++++++++++++LOAD DATA
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		#++++++++++++++++++++++++++++++APPEND DATA TO ARRAY
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
				
		#++++++++++++++++++++++++++++++MOVING AVERAGE
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		# Feature = np.arange(len(Feature))
		#++++++++++++++++++++++++++++++TRAIN; VALID; TEST SETS		
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Test = Feature[int(config['train']*len(Feature)) : ]
		x_Test = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		#++++++++++++++++++++++++++++++ARCHITECTURE OF ANN		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train) + 1)		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		if config['hidden_layers'][0] == 0:
			print('Auto config of layers')
			if config['hidden_layers'][1] == 1:			
				config['hidden_layers'] = (int(0.5*(n_pre - m_post) + m_post))
			elif config['hidden_layers'][1] == 2:			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post) + m_post), int(0.25*(n_pre - m_post) + m_post))
			elif config['hidden_layers'][1] == 3:			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post) + m_post), int(0.50*(n_pre - m_post) + m_post), int(0.25*(n_pre - m_post) + m_post))
			else:
				print('WITh coded LAYERS !!!!!!!')
				lay = uncode_tuple_layers(config['hidden_layers'][1])
				if len(lay) == 2:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				elif len(lay) == 1:
					config['hidden_layers'] = (int(lay[0]),)
				elif len(lay) == 3:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]), int(lay[2]))
				else:
					print('fatal error 557575')
					sys.exit()
	
			print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
		else:
			print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
			
		
		#++++++++++++++++++++++++++++++ MODEL
		clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=config['rs'], activation=config['activation'], verbose=False)
		
		
		#++++++++++++++++++++++++++++++ TRAINING EXAMPLES
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):
			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])

	
		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		MSE_Train = []
		MSE_Test = []
		#++++++++++++++++++++++++++++++ TRAINING
		for i in range(config['epochs']):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			random_rs = randint(0,1000)
			print('random_rs, ', random_rs)
			kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=random_rs)
			
			numb = 0
			for train_index, test_index in kf.split(T_Inputs):
				print('+++++++++++Batch', numb)
				T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				clf.partial_fit(T_Inputs_test, T_Outputs_test)
				numb += 1
			count += 1


			print('Calculating score epoch....')			
			
			Predict_Test = []
			It_TrainValid = list(Train)
			for k in range(len(x_Test) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_TrainValid[n_ex + k: n_ex + n_pre + k]
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
			
			
			MSE_Train.append(np.sum((Predict_Train - Train)**2.0)/len(Train))
			MSE_Test.append(np.sum((Predict_Test - Test)**2.0)/len(Test))
			
		#++++++++++++++++++++++++++++++ SAVING
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Train':MSE_Train, 'MSE_Test':MSE_Test})
		save_pickle('config_' + config['name'] + '.pkl', config)
		

		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		
			fig, ax = plt.subplots()
			ax.set_xlabel('Time', fontsize=13)
			ax.set_ylabel('Series', fontsize=13)
			ax.set_title('ANN', fontsize=13)
			ax.plot(x_Feature, Feature, 'b', label='Real')
			ax.plot(x_Test, Predict_Test, 'r', label='Test')
			ax.plot(x_Train, Train, 'm', label='Train')
			ax.legend()
		if config['save_plot'] == 'ON':
			plt.savefig('fig_' + config['name'] + '.png')
		else:
			x_Feature = x_Feature*5/60/60.
			x_Test = x_Test*5/60/60.
			x_Train = x_Train*5/60/60.
			
			style = {'xlabel':'Accumulated operating hours', 'ylabel':'Rate of AE bursts', 'legend':['Real', 'Test', 'Training'], 'title':None, 'customxlabels':None, 'xlim':None, 'ylim':None, 'color':['b', 'r', 'm'], 'loc_legend':'upper left', 'legend_line':'ON'}
			data = {'x':[x_Feature, x_Test, x_Train], 'y':[Feature, Predict_Test, Train]}
			
			plot1_thesis(data, style)
			# plt.show()
	
	elif config['mode'] == 'ann_test_with_model':		
		#++++++++++++++++++++++++++++++LOAD DATA
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		print('select model...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_clf = filedialog.askopenfilename()
		root.destroy()
		#++++++++++++++++++++++++++++++APPEND DATA TO ARRAY
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
				
		#++++++++++++++++++++++++++++++MOVING AVERAGE
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		# Feature = np.arange(len(Feature))
		#++++++++++++++++++++++++++++++TRAIN; VALID; TEST SETS		
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Test = Feature[int(config['train']*len(Feature)) : ]
		x_Test = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		#++++++++++++++++++++++++++++++ARCHITECTURE OF ANN		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train) + 1)		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		if config['hidden_layers'][0] == 0:
			print('Auto config of layers')
			if config['hidden_layers'][1] == 1:			
				config['hidden_layers'] = (int(0.5*(n_pre - m_post) + m_post))
			elif config['hidden_layers'][1] == 2:			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post) + m_post), int(0.25*(n_pre - m_post) + m_post))
			elif config['hidden_layers'][1] == 3:			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post) + m_post), int(0.50*(n_pre - m_post) + m_post), int(0.25*(n_pre - m_post) + m_post))
			else:
				print('WITh coded LAYERS !!!!!!!')
				lay = uncode_tuple_layers(config['hidden_layers'][1])
				if len(lay) == 2:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				elif len(lay) == 1:
					config['hidden_layers'] = (int(lay[0]),)
				elif len(lay) == 3:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]), int(lay[2]))
				else:
					print('fatal error 557575')
					sys.exit()
	
			print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
		else:
			print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
			
		
		#++++++++++++++++++++++++++++++ MODEL
		# clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=config['rs'], activation=config['activation'], verbose=False)
		
		clf = read_pickle(filepath_clf)
		
		
		#++++++++++++++++++++++++++++++ TRAINING EXAMPLES
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):
			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])

	
		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)
		count = 0
		MSE_Train = []
		MSE_Test = []
		#++++++++++++++++++++++++++++++ TRAINING
		for i in range(1):
			# print('++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			# random_rs = randint(0,1000)
			# print('random_rs, ', random_rs)
			# kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=random_rs)
			
			# numb = 0
			# for train_index, test_index in kf.split(T_Inputs):
				# print('+++++++++++Batch', numb)
				# T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				# T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				# clf.partial_fit(T_Inputs_test, T_Outputs_test)
				# numb += 1
			# count += 1


			print('Calculating score epoch....')			
			
			Predict_Test = []
			It_TrainValid = list(Train)
			for k in range(len(x_Test) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_TrainValid[n_ex + k: n_ex + n_pre + k]
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
			
			
			MSE_Train.append(np.sum((Predict_Train - Train)**2.0)/len(Train))
			MSE_Test.append(np.sum((Predict_Test - Test)**2.0)/len(Test))
			
		#++++++++++++++++++++++++++++++ SAVING
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Train':MSE_Train, 'MSE_Test':MSE_Test})
		save_pickle('config_' + config['name'] + '.pkl', config)
		

		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		
			fig, ax = plt.subplots()
			ax.set_xlabel('Time', fontsize=13)
			ax.set_ylabel('Series', fontsize=13)
			ax.set_title('ANN', fontsize=13)
			ax.plot(x_Feature, Feature, 'b', label='Real')
			ax.plot(x_Test, Predict_Test, 'r', label='Test')
			ax.plot(x_Train, Train, 'm', label='Train')
			ax.legend()
		if config['save_plot'] == 'ON':
			plt.savefig('fig_' + config['name'] + '.png')
		else:
			x_Feature = x_Feature*5/60/60.
			x_Test = x_Test*5/60/60.
			x_Train = x_Train*5/60/60.
			
			name = 'Predict70_Ini'
			path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
			path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
			path_1b = path_1 + name + '.svg'
			path_2b = path_2 + name + '.pdf'
			
			style = {'xlabel':'Cumulative operating hours', 'ylabel':'Max. cross-correlation', 'legend':['Real', 'Predicted', 'Training'], 'title':None, 'customxlabels':None, 'xlim':[0, 10.5], 'ylim':[0.05, 0.25], 'color':['b', 'r', 'm'], 'loc_legend':'upper left', 'legend_line':'ON', 'vlines':None, 'output':'save', 'path_1':path_1b, 'path_2':path_2b}
			data = {'x':[x_Feature, x_Test, x_Train], 'y':[Feature, Predict_Test, Train]}
			
			plot1_thesis(data, style)
			# plt.show()
	
	elif config['mode'] == 'ann_validation':
		#++++++++++++++++++++++++++++++LOAD DATA
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		#++++++++++++++++++++++++++++++APPEND DATA TO ARRAY
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
				
		#++++++++++++++++++++++++++++++MOVING AVERAGE
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		#++++++++++++++++++++++++++++++TRAIN; VALID; TEST SETS		
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Valid = Feature[int(config['train']*len(Feature)) : int(config['train']*len(Feature)) + int(config['valid']*len(Feature))]
		x_Valid = np.linspace(len(Train), len(Train) + len(Valid), num=len(Valid), endpoint=False)
		
		Test = Feature[int(config['train']*len(Feature)) + int(config['valid']*len(Feature)) : ]
		x_Test = np.linspace(len(Train) + len(Valid), len(Feature), num=len(Feature) - len(Train) - len(Valid), endpoint=False)
		
		
		#++++++++++++++++++++++++++++++ARCHITECTURE OF ANN		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train) + 1)		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
		
		if config['hidden_layers'][0] == 0:
			print('Auto config of layers')
			if config['hidden_layers'][1] == 1:			
				config['hidden_layers'] = (int(0.5*(n_pre - m_post) + m_post))
			elif config['hidden_layers'][1] == 2:			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post) + m_post), int(0.25*(n_pre - m_post) + m_post))
			elif config['hidden_layers'][1] == 3:			
				config['hidden_layers'] = (int(0.75*(n_pre - m_post) + m_post), int(0.50*(n_pre - m_post) + m_post), int(0.25*(n_pre - m_post) + m_post))
			else:
				print('WITh coded LAYERS !!!!!!!')
				lay = uncode_tuple_layers(config['hidden_layers'][1])
				if len(lay) == 2:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]))
				elif len(lay) == 1:
					config['hidden_layers'] = (int(lay[0]),)
				elif len(lay) == 3:
					config['hidden_layers'] = (int(lay[0]), int(lay[1]), int(lay[2]))
				else:
					print('fatal error 557575')
					sys.exit()
	
			print('+++++++++++++Info: Hidden Layers = ', config['hidden_layers'])
		else:
			print('WITHOUT AUTO CONFIG LAYERS !!!!!!!')
			
		
		#++++++++++++++++++++++++++++++ MODEL
		print(config['hidden_layers'])
		clf = MLPRegressor(solver='adam', alpha=config['alpha'], hidden_layer_sizes=config['hidden_layers'], random_state=config['rs'], activation=config['activation'], verbose=False)
		
		
		#++++++++++++++++++++++++++++++ TRAINING EXAMPLES
		T_Inputs = []
		T_Outputs = []
		
		
		for k in range(n_ex):
			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])

	
		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)

		count = 0
		MSE_Valid = []
		MSE_Train = []
		MSE_Test = []
		CRC_Valid = []
		CRC_Train = []
		CRC_Test = []
		PEN1_Valid = []
		PEN1_Train = []
		PEN1_Test = []
		PEN2_Valid = []
		PEN2_Train = []
		PEN2_Test = []
		STDp_Valid = []
		STDp_Train = []
		STDp_Test = []
		STDr_Valid = []
		STDr_Train = []
		STDr_Test = []
		
		#++++++++++++++++++++++++++++++ TRAINING
		for i in range(config['epochs']):
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++Epoch', count)
			random_rs = randint(0,1000)
			print('random_rs, ', random_rs)
			kf = KFold(n_splits=config['mini_batches'], shuffle=True, random_state=random_rs)
			numb = 0
			print('minibatches ', config['mini_batches'])
			print(len(T_Inputs))
			print(len(T_Inputs[0]))
			for train_index, test_index in kf.split(T_Inputs):
				print('+++++++++++Batch', numb)
				T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
				T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
				if m_post == 1:
					T_Outputs_test = np.ravel(T_Outputs_test)
				clf.partial_fit(T_Inputs_test, T_Outputs_test)
				# print(T_Inputs_test)
				numb += 1
			count += 1

			print('Calculating score epoch....')
			Predict_Valid = []
			It_Train = list(Train)
			for k in range(len(x_Valid) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_Train[n_ex + k : n_ex + n_pre + k]
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]
					if m_post == 1:
						Predict_Valid.append(P_Output)
						It_Train.append(P_Output)
					else:
						Predict_Valid.append(P_Output[-1])
						It_Train.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Valid.append(P_Output[0])
					It_Train.append(P_Output[0])
			
			if m_post == 1:
				Predict_Valid = np.array(Predict_Valid)
			else:
				Predict_Valid = np.array(Predict_Valid[:-(m_post-1)])
			
			if config['train_test_ref'] == 'ON':
				Predict_Test = []
				It_TrainValid = list(Train) + list(Valid)
				for k in range(len(x_Test) + m_post - 1):		
					if config['predict_node'] == 'last':
						P_Input = It_TrainValid[n_ex + k + len(Valid): n_ex + n_pre + len(Valid) + k]
						# print(P_Input)
						# b = input('...')		
						P_Output = clf.predict([P_Input])
						P_Output = P_Output[0]
						if m_post == 1:
							Predict_Test.append(P_Output)
							It_TrainValid.append(P_Output)
						else:						
							Predict_Test.append(P_Output[-1])
							It_TrainValid.append(P_Output[-1])
					elif config['predict_node'] == 'first':				
						P_Input = It_TrainValid[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
						P_Output = clf.predict([P_Input])
						P_Output = P_Output[0]			
						Predict_Test.append(P_Output[0])
						It_TrainValid.append(P_Output[0])
				
				if m_post == 1:
					Predict_Test = np.array(Predict_Test)
				else:
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
						if m_post == 1:
							Predict_Train.append(P_Output)
							It_TrainFirst.append(P_Output)
						else:
							Predict_Train.append(P_Output[-1])
							It_TrainFirst.append(P_Output[-1])
					elif config['predict_node'] == 'first':				
						P_Input = It_TrainFirst[k : n_pre + k]			
						P_Output = clf.predict([P_Input])
						P_Output = P_Output[0]			
						Predict_Train.append(P_Output[0])
						It_TrainFirst.append(P_Output[0])

				if m_post == 1:
					Predict_Train = np.array(Predict_Train)
				else:
					Predict_Train = np.array(Predict_Train[:-(m_post-1)])
				
				MSE_Train.append(np.sum((Predict_Train - Train)**2.0)/len(Train))
				MSE_Test.append(np.sum((Predict_Test - Test)**2.0)/len(Test))
				
				# CRC_Train.append(np.corrcoef(Predict_Train, Train)[0][1])
				# CRC_Test.append(np.corrcoef(Predict_Test, Test)[0][1])
				
				# noise = np.random.normal(loc=0., scale=0.0001, size=len(Train))
				# PEN1_Train.append(np.absolute(np.corrcoef(Predict_Train, noise+np.ones(len(Train)))[0][1]))			

				
				# noise = np.random.normal(loc=0., scale=0.0001, size=len(Test))
				# PEN1_Test.append(np.absolute(np.corrcoef(Predict_Test, noise+np.ones(len(Test)))[0][1]))
				
				# PEN2_Train.append(np.absolute(np.corrcoef(Predict_Train, np.arange(len(Train)))[0][1]))
				# PEN2_Test.append(np.absolute(np.corrcoef(Predict_Test, np.arange(len(Test)))[0][1]))
				
				# STDp_Train.append(np.std(Predict_Train))
				# STDp_Test.append(np.std(Predict_Test))
				# STDr_Train.append(np.std(Train))
				# STDr_Test.append(np.std(Test))
				
			mse_iter = np.sum((Predict_Valid - Valid)**2.0)/len(Valid)
			MSE_Valid.append(mse_iter)
			
			# CRC_Valid.append(np.corrcoef(Predict_Valid, Valid)[0][1])
			# noise = np.random.normal(loc=0., scale=0.0001, size=len(Valid))
			# PEN1_Valid.append(np.absolute(np.corrcoef(Predict_Valid, noise+np.ones(len(Valid)))[0][1]))
			# PEN2_Valid.append(np.absolute(np.corrcoef(Predict_Valid, np.arange(len(Valid)))[0][1]))
			# STDp_Valid.append(np.std(Predict_Valid))
			# STDr_Valid.append(np.std(Valid))
			
			if config['early_stop'] == 'ON':
				if i == 0:
					before_mse = mse_iter
					acum = 0
				else:
					before_mse = mse_iter
					if mse_iter < before_mse:						
						acum = 0
					else:
						acum += 1
				if acum >= config['stop_epochs']:
					break
			else:
				acum = i
						
		print('final epochs = ', i)			
			
		#++++++++++++++++++++++++++++++ SAVING
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Valid':MSE_Valid, 'MSE_Test':MSE_Test, 'MSE_Train':MSE_Train, 'CRC_Valid':CRC_Valid, 'CRC_Test':CRC_Test, 'CRC_Train':CRC_Train, 'PEN1_Valid':PEN1_Valid, 'PEN1_Test':PEN1_Test, 'PEN1_Train':PEN1_Train, 'PEN2_Valid':PEN2_Valid, 'PEN2_Test':PEN2_Test, 'PEN2_Train':PEN2_Train, 'STDp_Train':STDp_Train, 'STDp_Test':STDp_Test, 'STDp_Valid':STDp_Valid, 'STDr_Train':STDr_Train, 'STDr_Test':STDr_Test, 'STDr_Valid':STDr_Valid})
		save_pickle('config_' + config['name'] + '.pkl', config)
		

		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		if config['save_plot'] == 'ON':
			fig, ax = plt.subplots()
			ax.set_xlabel('Time', fontsize=13)
			ax.set_ylabel('Series', fontsize=13)
			ax.set_title('ANN', fontsize=13)
			ax.plot(x_Feature, Feature, 'b', label='Real')
			ax.plot(x_Valid, Predict_Valid, 'g', label='Validation')
			if config['train_test_ref'] == 'ON':
				# print(x_Test)
				# print(Predict_Test)
				ax.plot(x_Test, Predict_Test, 'r', label='Test')
				ax.plot(x_Train, Predict_Train, 'm', label='Train')
			ax.legend()
			plt.savefig('fig_' + config['name'] + '.png')
		
	
	

	
	elif config['mode'] == 'tree_validation':
		#++++++++++++++++++++++++++++++LOAD DATA
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		#++++++++++++++++++++++++++++++APPEND DATA TO ARRAY
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
				
		#++++++++++++++++++++++++++++++MOVING AVERAGE
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		#++++++++++++++++++++++++++++++TRAIN; VALID; TEST SETS		
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Valid = Feature[int(config['train']*len(Feature)) : int(config['train']*len(Feature)) + int(config['valid']*len(Feature))]
		x_Valid = np.linspace(len(Train), len(Train) + len(Valid), num=len(Valid), endpoint=False)
		
		Test = Feature[int(config['train']*len(Feature)) + int(config['valid']*len(Feature)) : ]
		x_Test = np.linspace(len(Train) + len(Valid), len(Feature), num=len(Feature) - len(Train) - len(Valid), endpoint=False)
		
		
		#++++++++++++++++++++++++++++++ARCHITECTURE OF ANN		
		n_pre = int(config['n_pre']*len(Train))
		m_post = int(config['m_post']*len(Train) + 1)		

		
		n_ex = len(Train) - n_pre - m_post + 1
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		config['n_ex'] = n_ex
			
		
		#++++++++++++++++++++++++++++++ MODEL
		
		clf = SVR(kernel=config['linear'], C=config['C'], degree=config['degree'])
		
		
		#++++++++++++++++++++++++++++++ TRAINING EXAMPLES
		T_Inputs = []
		T_Outputs = []

		for k in range(n_ex):
			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])

	
		T_Inputs = np.array(T_Inputs)
		T_Outputs = np.array(T_Outputs)

		MSE_Valid = []
		
		#++++++++++++++++++++++++++++++ TRAINING
		print('+++++++++++++++++++++++++++++++Ini Training')
		# print(T_Inputs)
		# print(T_Outputs)
		clf.fit(T_Inputs, T_Outputs)


		# for train_index, test_index in kf.split(T_Inputs):
			# print('+++++++++++Batch', numb)
			# T_Inputs_train, T_Inputs_test = T_Inputs[train_index], T_Inputs[test_index]
			# T_Outputs_train, T_Outputs_test = T_Outputs[train_index], T_Outputs[test_index]
			# if m_post == 1:
				# T_Outputs_test = np.ravel(T_Outputs_test)
			# clf.partial_fit(T_Inputs_test, T_Outputs_test)
			# # print(T_Inputs_test)
			# numb += 1
		# count += 1

		print('Calculating MSE Valid....')
		Predict_Valid = []
		It_Train = list(Train)
		for k in range(len(x_Valid) + m_post - 1):		
			if config['predict_node'] == 'last':
				P_Input = It_Train[n_ex + k : n_ex + n_pre + k]
				# print('+++, ', P_Input)
				P_Output = clf.predict([P_Input])
				# print('ooo, ', P_Output)
				P_Output = P_Output[0]
				if m_post == 1:
					Predict_Valid.append(P_Output)
					It_Train.append(P_Output)
				else:
					Predict_Valid.append(P_Output[-1]+0.1)
					It_Train.append(P_Output[-1]+0.1)
					# print(P_Output[-1]+0.1)
			elif config['predict_node'] == 'first':				
				P_Input = It_Train[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
				P_Output = clf.predict([P_Input])
				P_Output = P_Output[0]			
				Predict_Valid.append(P_Output[0])
				It_Train.append(P_Output[0])
		
		if m_post == 1:
			Predict_Valid = np.array(Predict_Valid)
		else:
			Predict_Valid = np.array(Predict_Valid[:-(m_post-1)])
		
		if config['train_test_ref'] == 'ON':
			Predict_Test = []
			It_TrainValid = list(Train) + list(Valid)
			for k in range(len(x_Test) + m_post - 1):		
				if config['predict_node'] == 'last':
					P_Input = It_TrainValid[n_ex + k + len(Valid): n_ex + n_pre + len(Valid) + k]
					# print(P_Input)
					# b = input('...')		
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]
					if m_post == 1:
						Predict_Test.append(P_Output)
						It_TrainValid.append(P_Output)
					else:						
						Predict_Test.append(P_Output[-1])
						It_TrainValid.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_TrainValid[n_ex + k + m_post - 1 : n_ex + n_pre + k + m_post]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Test.append(P_Output[0])
					It_TrainValid.append(P_Output[0])
			
			if m_post == 1:
				Predict_Test = np.array(Predict_Test)
			else:
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
					if m_post == 1:
						Predict_Train.append(P_Output)
						It_TrainFirst.append(P_Output)
					else:
						Predict_Train.append(P_Output[-1])
						It_TrainFirst.append(P_Output[-1])
				elif config['predict_node'] == 'first':				
					P_Input = It_TrainFirst[k : n_pre + k]			
					P_Output = clf.predict([P_Input])
					P_Output = P_Output[0]			
					Predict_Train.append(P_Output[0])
					It_TrainFirst.append(P_Output[0])
			if m_post == 1:
				Predict_Train = np.array(Predict_Train)
			else:
				Predict_Train = np.array(Predict_Train[:-(m_post-1)])
			
		mse_iter = np.sum((Predict_Valid - Valid)**2.0)/len(Valid)
		MSE_Valid.append(mse_iter)
		

						
			
		#++++++++++++++++++++++++++++++ SAVING
		save_pickle('Scores_' + config['name'] + '.pkl', {'MSE_Valid':MSE_Valid})
		save_pickle('config_' + config['name'] + '.pkl', config)
		

		if config['save_model'] == 'ON':
			save_pickle('model_' + config['name'] + '.pkl', clf)
			
		if config['save_plot'] == 'ON':
			fig, ax = plt.subplots()
			ax.set_xlabel('Time', fontsize=13)
			ax.set_ylabel('Series', fontsize=13)
			ax.set_title('SVR', fontsize=13)
			ax.plot(x_Feature, Feature, 'b', label='Real')
			ax.plot(x_Valid, Predict_Valid, 'g', label='Validation')
			if config['train_test_ref'] == 'ON':
				# print(x_Test)
				# print(Predict_Test)
				ax.plot(x_Test, Predict_Test, 'r', label='Test')
				ax.plot(x_Train, Predict_Train, 'm', label='Train')
			ax.legend()
			plt.savefig('fig_' + config['name'] + '.png')
		
  	

	elif config['mode'] == 'eval_mse':		
		print('Select PKL Files with MSE')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()		
		
		Combi, Epoch_Combi = calculate_score_regression(Filepaths, 'mse_train_valid')		
		
		plt.figure(0)
		plt.plot(Combi, '-o', color='red', label='Combi')
		plt.legend()
		
		plt.figure(1)
		plt.plot(Epoch_Combi, '-s', color='black', label='epoch')
		plt.legend()		
		
		plt.show()
	
	elif config['mode'] == 'check_epochs':		
		print('Select PKL Files with MSE')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		
		data = read_pickle(filepath)
		

		
		plt.figure(1)
		plt.plot(data['MSE_Valid'], '-o')
		
		plt.show()
	
	elif config['mode'] == 'generate_genetic':

		print('Select PKL Files with SCORES')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
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
		
		Combi, Epoch_Combi = calculate_score_regression(Filepaths, 'mse_train_valid')
		Idx_Combi = list(np.arange(len(Combi)))
		
		Combi_sorted = sorted(Combi)
		Epoch_Combi_sorted = sort_X_based_on_Y(Epoch_Combi, Combi)
		Idx_Combi_sorted = sort_X_based_on_Y(Idx_Combi, Combi)
		
		n_bests = config['n_bests']
		best_combis = Idx_Combi_sorted[-n_bests:]
		

		best_epoch_combis = Epoch_Combi_sorted[-n_bests:]		
		
		Filepaths_Configs_bests = [Filepaths_Configs[i] for i in best_combis]
		Filepaths_bests = [Filepaths[i] for i in best_combis]
		

		
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
		
		print(Best_Hyper[:,0])
		offspring = mutation_ann(offspring, weight)

		
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
			
			# mypath = 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Tri_Analysis\\Idx14'
			
			mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\Congreso_Lyon\\Relative_Counting_08mV_THR_300_300'
			
			count = 0
			
			# print(Layers)
			# sys.exit()
			for layer, alpha, n_pre, m_post, minibatches, epochs in zip(Layers, Alphas, N_Pres, M_Posts, MiniBatches, Epochs):
				if (float(n_pre) + float(m_post) >= 1.):
					n_pre_f = float(n_pre)/2.
					m_post_f = float(m_post)/2.
					n_pre = str(n_pre_f)
					m_post = str(m_post_f)
				# print('++++++++++++++++...', layer)
				if count != -99999:
					os.system('python M_Regression7.py --plot OFF --train_test_ref ON --rs 0 --hidden_layers 0 ' + layer + ' --mode ann_validation --name Gen' + opt + '_CORR_t70_Idx14_ANN_Idx_' + str(count) + ' --activation identity --alpha ' + alpha + ' --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg 12 --train 0.5 --feature CORR' + ' --predict_node last --valid 0.2 --mini_batches ' + minibatches + ' --epochs ' + epochs + ' --mypath ' + mypath)
				count += 1
	
	
	elif config['mode'] == 'linear':
		#++++++++++++++++++++++++++++++LOAD DATA
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		
		
		#++++++++++++++++++++++++++++++APPEND DATA TO ARRAY
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
				
		#++++++++++++++++++++++++++++++MOVING AVERAGE
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		# Feature = np.arange(len(Feature))
		#++++++++++++++++++++++++++++++TRAIN; VALID; TEST SETS		
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))
		
		Test = Feature[int(config['train']*len(Feature)) : ]
		x_Test = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		
		slope, intercept, r_value, p_value, std_err = stats.linregress(x_Train, Train)
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)

		Predict = slope*x_Predict + intercept
		
		
		
		
		
		x_Feature = x_Feature*5/60/60.
		x_Predict = x_Predict*5/60/60.
		x_Train = x_Train*5/60/60.
		
		style = {'xlabel':'Accumulated operating hours', 'ylabel':'Rate of AE bursts', 'legend':['Real', 'Test', 'Training'], 'title':None, 'customxlabels':None, 'xlim':None, 'ylim':None, 'color':['b', 'r', 'm'], 'loc_legend':'upper left', 'legend_line':'ON'}
		data = {'x':[x_Feature, x_Predict, x_Train], 'y':[Feature, Predict, Train]}
		
		plot1_thesis(data, style)
		
		
		
		# fig, ax = plt.subplots()
		# ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		# ax.set_ylabel('Health Index', fontsize=13)
		# # ax.set_title('Linear Regression', fontsize=13)
		# ax.set_title('Neural Network', fontsize=13)
		# fact = 5./3600.
		# ax.plot(x_Feature*fact, Feature, 'b', label='Real')
		# ax.plot(x_Predict*fact, Predict, 'r', label='Prediction')
		# ax.plot(x_Train*fact, Train, 'k', label='Training')
		# ax.legend()
		# plt.show()
	
		
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
	config['stop_epochs'] = int(config['stop_epochs'])
	
	
	
	config['C'] = float(config['C'])
	config['degree'] = int(config['degree'])
	

	
	
	
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



def calculate_score_regression(Filepaths, score):	
	Combi = []
	Epoch_Combi = []
	count = 0
	for filepath in Filepaths:
		data = read_pickle(filepath)
		mini_combi = []
		if score == 'mse_valid':
			for mse_v in data['MSE_Valid']:			
				valid = 1/(mse_v)			
				mini_combi.append(valid)
		elif score == 'crc_mse_valid':
			for crc_v, mse_v in zip(data['CRC_Valid'], data['MSE_Valid']):			
				valid = crc_v/mse_v			
				mini_combi.append(valid)
		elif score == 'mse_train_valid':
			idx = 0
			for mse_t, mse_v in zip(data['MSE_Train'], data['MSE_Valid']):			
				valid = 1./mse_v
				train = 1./mse_t
				prom = (valid + train)/2.
				mini_combi.append(prom)
				# if count == 30: #58
					# if idx == 20: #63
						# print('!!!!', prom)
						# print('mse_v', mse_v)
						# print('mse_t',mse_t)
				
				idx += 1
				# print(prom)
		elif score == 'crc_mse_train_valid':
			for crc_t, crc_v, mse_t, mse_v in zip(data['CRC_Train'], data['CRC_Valid'], data['MSE_Train'], data['MSE_Valid']):			
				valid = crc_v/mse_v
				train = crc_t/mse_t
				prom = (valid + train)/2.
				mini_combi.append(prom)
		elif score == 'crc_mse_pen2_train_valid':
			for crc_t, crc_v, mse_t, mse_v, pen2_t, pen2_v in zip(data['CRC_Train'], data['CRC_Valid'], data['MSE_Train'], data['MSE_Valid'], data['PEN2_Train'], data['PEN2_Valid']):			
				valid = (crc_v/mse_v)*(pen2_v)
				train = (crc_t/mse_t)*(pen2_t)
				prom = (valid + train)/2.
				mini_combi.append(prom)
		# if count == 30:
			# print(np.array(mini_combi))
			# print(len(np.array(mini_combi)))
			# print(np.max(np.array(mini_combi)))
			# print(np.argmax(np.array(mini_combi)))
		Combi.append(np.max(np.array(mini_combi)))
		Epoch_Combi.append(np.argmax(np.array(mini_combi)))
		count += 1
	return Combi, Epoch_Combi



def mutation_pos(value, weight):

	random_value = np.random.uniform(-weight, weight, None)
	value = value * (1. + random_value)
	while value < 0:
		random_value = np.random.uniform(-weight, weight, None)
		value = value * (1. + random_value)

	return value

def mutation_pos_codedlayer(value, weight):
	tuple_layers = uncode_tuple_layers(value)
	random_value = np.random.uniform(-weight, weight, None)
	
	new_value0 = tuple_layers[0] * (1. + random_value)
	while new_value0 <= 0:
		print(new_value0)
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
		print(idx)
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


