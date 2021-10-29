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
from tkinter import filedialog

import os
import sys
sys.path.insert(0, './lib') #to open user-defined functions

from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_pattern import *
from m_denois import *
from m_det_features import *
from m_processing import *
import pickle
import argparse

import pandas as pd
import datetime

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser

plt.rcParams['savefig.directory'] = os.chdir(os.path.dirname('C:'))

plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from random import randint
Inputs = ['mode']

# 'mypath':R'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Master'

InputsOpt_Defaults = {'mypath':None, 'name':'auto', 'feature':'RMS', 'train':0.60, 'n_mov_avg':0, 'predict_node':'last', 'auto_layers':None, 'save_plot':'ON', 'n_bests':3, 'weight':0.05, 'n_children':7, 'save_model':'OFF', 'features_array':['RMS', 'MAX', 'KURT', 'CORR', 'sqr10h_f_f_r', 'sqr5h_f_g', 'sqr5h_sb_f_g_c', 'Temp', 'acRMS', 'acMAX', 'acKURT', 'bursts']}
# 'acRMS', 'acMAX', 'acKURT'

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	

	
	if config['mode'] == 'lin_system':		
		if config['mypath'] == None:
			print('Select XLS Files with Feature')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'xlsx']
		

		mydict = pd.read_excel(filepath)			
		
		Features, n_samples, n_features = narray_features_nma(filepath, config['features_array'], config['n_mov_avg'])
		
		# for k in range(n_features):
			# plt.plot(Features[:,k], label=config['features_array'][k])		
		# plt.legend()
		# plt.show()
		
		scaler = StandardScaler()
		scaler.fit(Features)
		Features = scaler.transform(Features)
		
		for k in range(n_features):
			if config['features_array'][k] == 'Temp':
				temp = Features[:,k]
		
		# for k in range(n_features):
			# plt.plot(Features[:,k], label=config['features_array'][k])		
		# plt.legend()
		# plt.show()
		
		pca = PCA(n_components=n_features)
		pca.fit(Features)
		print(pca.explained_variance_ratio_) 
		Features = pca.transform(Features) 
		
		
		
		ref = np.arange(n_samples)
		print('scores lineal')
		for k in range(n_features):
			score = np.corrcoef(Features[:,k], ref)[0][1]
			print(score)
		
		print('scores temp')
		for k in range(n_features):
			score = np.corrcoef(Features[:,k], temp)[0][1]
			print(score)
		
		plt.figure(1)
		plt.plot(temp, label='temp')
		plt.plot(Features[:,0], label='tempPCA')
		
		plt.figure(2)
		plt.plot(ref, label='ref')
		plt.plot(Features[:,1], label='refPCA')
		
		plt.legend()
		plt.show()
		
		exit()
		
		
		size_population = 50
		ini_population = np.random.uniform(low=-1, high=1., size=(n_features, size_population))
		
		weights = np.random.uniform(low=-1, high=1., size=(n_features, 1))
		bias = np.random.uniform(low=-1, high=1., size=(n_samples, 1))
		
		fusion = np.dot(Features, weights) + bias
		ref = np.arange(n_samples)

		score = np.corrcoef(fusion[:,0], ref)[0][1]
		
		print(score)
		
		offspring = general_crossover_single(parents, n_children, n_features)
		offspring = general_mutation(offspring, weight, n_features)
		
		
		plt.plot(fusion)
		plt.show()
	
	
	
		
	else:
		print('unknown mode')

	
	return

def narray_features_nma(filepath, names_features, nma):				
	Dict_Features = {}
	for feature in names_features:
		Dict_Features[feature] = []
	mydict = pd.read_excel(filepath)			
	mydict = mydict.to_dict(orient='list')			
	for element in names_features:
		data = movil_avg(np.array(mydict[element]), nma)
		Dict_Features[element] += list(data)
		
	n_samples = len(Dict_Features[names_features[0]])
	n_features = len(names_features)		
	Features = np.zeros((n_samples, n_features))
	count = 0
	for feature in names_features:
		Features[:, count] = Dict_Features[feature]
		count += 1
	
	for k in range(count):
		for i in range(len(Features[:,k])):
			caca = 0
			while np.isnan(Features[i,k]) == True:
				caca += 1
				Features[i,k] = Features[i + caca,k]	
	
	return Features, n_samples, n_features


def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'files' or element == 'features_array':
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

	
	
	# config['clusters'] = int(config['clusters'])
	config['n_mov_avg'] = int(config['n_mov_avg'])
	
	
	
	

	
	config['train'] = float(config['train'])
	# config['valid'] = float(config['valid'])
	

	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config
	


	

def mutation_pos(value, weight):

	random_value = np.random.uniform(-weight, weight, None)
	value = value * (1. + random_value)
	while value <= 0:
		random_value = np.random.uniform(-weight, weight, None)
		value = value * (1. + random_value)

	return value



def mutation_ann(offspring, weight):
	for idx in range(offspring.shape[0]):
		offspring[idx, 0] = int(round(mutation_pos_codedlayer(offspring[idx, 0], weight))) #layer 
		offspring[idx, 1] = mutation_pos(offspring[idx, 1], weight) #alpha
		offspring[idx, 2] = mutation_pos(offspring[idx, 2], weight) #npre
		offspring[idx, 3] = mutation_pos(offspring[idx, 3], weight) #mpost
		offspring[idx, 4] = int(round(mutation_pos(offspring[idx, 4], weight))) #minibatches
		offspring[idx, 5] = int(round(mutation_pos(offspring[idx, 5], weight))) #epochs
	return offspring

def general_mutation(offspring, weight, n_features):
	for idx in range(offspring.shape[0]):	
		for j in range(n_features):
			offspring[idx, j] = mutation_pos(offspring[idx, j], weight)

	return offspring

# def code_tuple_layers(tuple_layers):
	# if len(tuple_layers) == 1:
		# codlayers = str(1) + str(len(str(tuple_layers[0]))) + '0' + str(tuple_layers[0])
	# elif len(tuple_layers) == 2:
		# codlayers = str(2) + str(len(str(tuple_layers[0]))) + str(len(str(tuple_layers[1]))) + str(tuple_layers[0]) + str(tuple_layers[1])
	# return codlayers
	
# def uncode_tuple_layers(code_layers):
	# code = str(code_layers)
	# if code[0] == '1':
		# uncode_layers = (float(code[3:]),)
	# elif code[0] == '2':
		# uncode_layers = (float(code[3:3+int(code[1])]), float(code[3+int(code[1]):]))
	# else:
		# print('fatal error 515485')
		# sys.exit()
	# return uncode_layers

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

def general_crossover_single(parents, n_children, n_features):
	offspring_size = (n_children, n_features) 		
	offspring = np.empty(offspring_size)
	
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]
		rint = randint(1, n_features-1)
		
		for j in range(n_features):
			if j < rint:
				offspring[k, j] = parents[parent1_idx, j]
			else:
				offspring[k, j] = parents[parent2_idx, j]
		
		
	return offspring
			

if __name__ == '__main__':
	main(sys.argv)


