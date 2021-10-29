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
import graphviz 
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
Inputs = ['mode', 'channel', 'fs']


InputsOpt_Defaults = {'plot':'OFF', 'rs':0, 'save':'ON', 'scaler':None, 'pca_comp':0, 'mypath':None, 'name':'auto', 'feature':'RMS', 'train':0.60, 'n_mov_avg':1, 'predict_node':'last', 'mini_batches':10, 'activation':'identity', 'alpha':0.001, 'epochs':10, 'hidden_layers':[50], 'valid':0.2, 'n_pre':0.3, 'm_post':0.05, 'auto_layers':None, 'save_plot':'ON', 'n_bests':3, 'weight':0.05, 'n_children':7, 'save_model':'OFF', 'train_test_ref':'OFF', 'early_stop':'OFF', 'stop_epochs':10, 'kernel':'linear', 'degree':3, 'C':1.}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	

	
	if config['mode'] == 'plot_wfm':		
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
		
		count = 0
		for filepath in Filepaths:
			print('plotting... ', count/len(Filepaths))
			x = load_signal(filepath, channel=config['channel'])
			n = len(x)
			dt = 1./config['fs']
			t = np.array([i*dt for i in range(n)])
			filename = os.path.basename(filepath)[:-5]
				
			if config['save_plot'] == 'ON':
				fig, ax = plt.subplots()
				fig.set_size_inches(11,5)
				ax.set_xlabel('Time [s]', fontsize=13)
				ax.set_ylabel('Amplitude [V]', fontsize=13)
				ax.set_title('Tachometer ' + filename, fontsize=13)
				ax.plot(t, x, 'b')
				plt.savefig('wfm_Drehzahl' + filename + '.png')
			del x
			del t
			del fig
			del ax
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
	config['stop_epochs'] = int(config['stop_epochs'])
	
	
	
	config['fs'] = float(config['fs'])
	config['degree'] = int(config['degree'])
	

	
	
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config
	


	

			

if __name__ == '__main__':
	main(sys.argv)


