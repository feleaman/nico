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
from os.path import isfile, join
import pickle
import argparse
from os import chdir
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


Inputs = ['mode']


InputsOpt_Defaults = {'plot':'OFF', 'rs':0, 'save':'ON', 'scaler':None, 'pca_comp':0, 'mypath':None, 'name':'auto', 'clusters':5, 'batch_size':20000}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'mode1':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)
		
		# print(Dict_Features['count'])
		
		Features = np.zeros((n_samples, n_features))
		count = 0
		for feature in names_features:
			Features[:, count] = Dict_Features[feature]
			count += 1
		# print(Features[:,1])
		# print(Features[:,1][3])
		scaler = StandardScaler()
		scaler.fit(Features)
		Features = scaler.transform(Features)
		
		pca = PCA(n_components=2)
		pca.fit(Features)
		print(pca.explained_variance_ratio_*100.)		
		NewFeatures = pca.transform(Features)		
		# NewFeatures = Features
		
		# print(NewFeatures)
		# plt.scatter(NewFeatures[:,0], NewFeatures[:,1])
		# plt.show()
		# sys.exit()
		colors = ['red', 'blue', 'black', 'green', 'magenta', 'yellow', 'deepskyblue', 'darkred']
		clusters = 5
		print('+++++++++++++++++++++++++++++++ clusters: ', clusters)		
		kmeans = KMeans(n_clusters=clusters, random_state=0)
		kmeans.fit(NewFeatures)	
		Labels = kmeans.predict(NewFeatures)
		Index_labels = []
		for k in range(clusters):
			it_index_labels = []
			count = 0
			for label in Labels:
				if label == k:
					it_index_labels.append(count)
				count += 1
			Index_labels.append(it_index_labels)
		for k in range(clusters):
			plt.scatter(NewFeatures[:,0][Index_labels[k]], NewFeatures[:,1][Index_labels[k]], color=colors[k])
		plt.show()
		
		
		sys.exit()
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()		
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		for filepath in Filepaths:
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
		Features = scaler.transform(Features)
		# NewFeatures = pca.transform(Features)
		NewFeatures = Features
		
		Labels = kmeans.predict(NewFeatures)
		Index_labels = []
		for k in range(clusters):
			it_index_labels = []
			count = 0
			for label in Labels:
				if label == k:
					it_index_labels.append(count)
				count += 1
			Index_labels.append(it_index_labels)
		for k in range(clusters):
			print('class ', k)
			print(len(Index_labels[k]))
		# for k in range(clusters):
			# plt.scatter(NewFeatures[:,0][Index_labels[k]], NewFeatures[:,1][Index_labels[k]], color=colors[k])
			# # plt.scatter(NewFeatures[:,0][Index_labels[1]], NewFeatures[:,1][Index_labels[1]], color='blue')
		# plt.show()
		
		
		
		# print(kmeans.cluster_centers_)
		
		
		# print('3 clusters ++++++++++++++++++++++++++++++++')
		# clusters = 3
		# kmeans = KMeans(n_clusters=clusters, random_state=0)
		# kmeans.fit(NewFeatures)
		# Labels = kmeans.predict(NewFeatures)
		# Index_labels = []
		# for k in range(clusters):
			# it_index_labels = []
			# count = 0
			# for label in Labels:
				# if label == k:
					# it_index_labels.append(count)
				# count += 1
			# Index_labels.append(it_index_labels)
		
		# plt.scatter(NewFeatures[:,0][Index_labels[0]], NewFeatures[:,1][Index_labels[0]], color='red')
		# plt.scatter(NewFeatures[:,0][Index_labels[1]], NewFeatures[:,1][Index_labels[1]], color='blue')
		# plt.scatter(NewFeatures[:,0][Index_labels[2]], NewFeatures[:,1][Index_labels[2]], color='green')
		# plt.show()

		# print('4 clusters ++++++++++++++++++++++++++++++++')
		# clusters = 4
		# kmeans = KMeans(n_clusters=clusters, random_state=0)
		# kmeans.fit(NewFeatures)
		# Labels = kmeans.predict(NewFeatures)
		# Index_labels = []
		# for k in range(clusters):
			# it_index_labels = []
			# count = 0
			# for label in Labels:
				# if label == k:
					# it_index_labels.append(count)
				# count += 1
			# Index_labels.append(it_index_labels)
		
		# plt.scatter(NewFeatures[:,0][Index_labels[0]], NewFeatures[:,1][Index_labels[0]], color='red')
		# plt.scatter(NewFeatures[:,0][Index_labels[1]], NewFeatures[:,1][Index_labels[1]], color='blue')
		# plt.scatter(NewFeatures[:,0][Index_labels[2]], NewFeatures[:,1][Index_labels[2]], color='green')
		# plt.scatter(NewFeatures[:,0][Index_labels[3]], NewFeatures[:,1][Index_labels[3]], color='magenta')
		# plt.show()
		
		
		# print('5 clusters ++++++++++++++++++++++++++++++++')
		# clusters = 5
		# kmeans = KMeans(n_clusters=clusters, random_state=0)
		# kmeans.fit(NewFeatures)
		# Labels = kmeans.predict(NewFeatures)
		# Index_labels = []
		# for k in range(clusters):
			# it_index_labels = []
			# count = 0
			# for label in Labels:
				# if label == k:
					# it_index_labels.append(count)
				# count += 1
			# Index_labels.append(it_index_labels)
		
		# plt.scatter(NewFeatures[:,0][Index_labels[0]], NewFeatures[:,1][Index_labels[0]], color='red')
		# plt.scatter(NewFeatures[:,0][Index_labels[1]], NewFeatures[:,1][Index_labels[1]], color='blue')
		# plt.scatter(NewFeatures[:,0][Index_labels[2]], NewFeatures[:,1][Index_labels[2]], color='green')
		# plt.scatter(NewFeatures[:,0][Index_labels[3]], NewFeatures[:,1][Index_labels[3]], color='magenta')
		# plt.scatter(NewFeatures[:,0][Index_labels[4]], NewFeatures[:,1][Index_labels[4]], color='black')
		# plt.show()

	# elif config['mode'] == 'mode2':
		# print('Select xls from class 1 (gut)')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		# print('Select xls from class -1 (schlecht)')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths2 = filedialog.askopenfilenames()
		# root.destroy()
		
		
		# names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		# Dict_Features = {}
		# for feature in names_features:
			# Dict_Features[feature] = []
		
		# Labels = []
		# for filepath in Filepaths:
			# mydict = pd.read_excel(filepath)			
			# mydict = mydict.to_dict(orient='list')			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
			# Labels += [1 for i in range(len(mydict[element]))]
		# for filepath in Filepaths2:
			# mydict = pd.read_excel(filepath)			
			# mydict = mydict.to_dict(orient='list')			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
			# Labels += [-1 for i in range(len(mydict[element]))]

		# n_samples = len(Dict_Features[names_features[0]])
		# n_features = len(names_features)		
		
		# Features = np.zeros((n_samples, n_features))
		# count = 0
		# for feature in names_features:
			# Features[:, count] = Dict_Features[feature]
			# count += 1
		
		# # Labels = np.zeros(n_samples) - 1
		# # for k in range(len(Filepaths)):
			# # Labels[k] = 1
		
		
		# Idx_Gut = [i for i in range(n_samples) if Labels[i] == 1]
		# Idx_Schlecht = [i for i in range(n_samples) if Labels[i] == -1]
		# # print(Labels)
		# print('Gut: ', len(Idx_Gut))
		# print('Schlecht: ', len(Idx_Schlecht))
		# # sys.exit()

		# scaler = StandardScaler()
		# scaler.fit(Features)
		# Features = scaler.transform(Features)
		
		# pca = PCA(n_components=4)
		# pca.fit(Features)
		# print(pca.explained_variance_ratio_*100.)		
		# NewFeatures = pca.transform(Features)

		# plt.scatter(NewFeatures[:,0][Idx_Gut], NewFeatures[:,1][Idx_Gut], color='blue')
		# plt.scatter(NewFeatures[:,0][Idx_Schlecht], NewFeatures[:,1][Idx_Schlecht], color='red')
		
		
		# plt.show()
	
	elif config['mode'] == 'obtain_features_narray':
		#Obtain Filepaths
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']


		#Construct Features NArray
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		for filepath in Filepaths:
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
		
		#Scaler
		if config['scaler'] == 'standard':
			scaler = StandardScaler()
			scaler.fit(Features)
			Features = scaler.transform(Features)
		else:
			print('no scaling')
		
		#PCA
		if config['pca_comp'] != 0:
			pca = PCA(n_components=config['pca_comp'])
			pca.fit(Features)
			print('Explained PCA variance: ', pca.explained_variance_ratio_*100.)		
			Features = pca.transform(Features)
		else:
			print('no PCA')
		
		save_pickle('Features_NArray_' + config['name'] + '.pkl', Features)
		save_pickle('config_Features_NArray_' + config['name'] + '.pkl', config)
		
	
	
	elif config['mode'] == 'feed_model':
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']


			
		

		
		#Load Filepaths Features
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']

		
		#Invoke Scaler
		if config['scaler'] == 'standard':
			scaler = StandardScaler()
		else:
			print('++++++no scaling') 
		if config['scaler'] != 'OFF':
			for filepath in Filepaths:
				Features = narray_features(filepath, names_features)
				scaler.partial_fit(Features)
				# Features = scaler.transform(Features)
		if config['scaler'] != 'OFF':
			save_pickle('scaler_KMeans_' + config['name'] + '.pkl', scaler)
		
		#Invoke PCA
		if config['pca_comp'] != 0:
			pca = IncrementalPCA(n_components=config['pca_comp'], batch_size=None)
			for filepath in Filepaths:
				Features = narray_features(filepath, names_features)
				if config['scaler'] != None:
					Features = scaler.transform(Features)
				print(os.path.basename(filepath))
				try:
					pca.partial_fit(Features)
				except:
					print('exception handled!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
					print(Features.shape[0])
					vec = np.isfinite(Features)		
					rows = []
					for i in range(vec.shape[0]):
						for element in vec[i,:]:
							if element == False:
								rows.append(i)
					Features = np.delete(Features, np.array(rows), 0)
					print(rows)
					print(Features.shape[0])
					
					print(Features.shape[0])
					vec = np.isnan(Features)		
					rows = []
					for i in range(vec.shape[0]):
						for element in vec[i,:]:
							if element == True:
								rows.append(i)
					Features = np.delete(Features, np.array(rows), 0)
					print(rows)
					print(Features.shape[0])
					
					
					vec = np.isinf(Features)		
					rows = []
					for i in range(vec.shape[0]):
						for element in vec[i,:]:
							if element == True:
								rows.append(i)
					Features = np.delete(Features, np.array(rows), 0)
					print(rows)
					print(Features.shape[0])
					
					pca.partial_fit(Features)
					print('sucess!')
					
						
					
		else:
			print('++++++no PCA')
		if config['pca_comp'] != 0:
			save_pickle('pca_KMeans_' + config['name'] + '.pkl', pca)


		#Fit Model
		kmeans = MiniBatchKMeans(n_clusters=config['clusters'], random_state=config['rs'], batch_size=config['batch_size'])
		n_files = len(Filepaths)
		count = 0
		for filepath in Filepaths:
			print('Fit model % ', count/n_files)
			count += 1
			Features = narray_features(filepath, names_features)
			
			if config['scaler'] != 'OFF':
				Features = scaler.transform(Features)
			
			if config['pca_comp'] != 0:
				Features = pca.transform(Features)			

			
			kmeans = kmeans.partial_fit(Features)

		
		save_pickle('KMeans_' + config['name'] + '.pkl', kmeans)
		save_pickle('config_KMeans_' + config['name'] + '.pkl', config)
		
		
		
	
	elif config['mode'] == 'clf_from_model':
		#Open Model
		print('Select model')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_model = filedialog.askopenfilename()			
		root.destroy()
		kmeans = load_signal_2(filepath_model)
		# print(type(kmeans))
		# sys.exit()
		
		if config['scaler'] != None:
			print('Select scaler')
			root = Tk()
			root.withdraw()
			root.update()
			filepath_scaler = filedialog.askopenfilename()			
			root.destroy()
			scaler = load_signal_2(filepath_scaler)
			Features = scaler.transform(Features)
		
		if config['pca_comp'] != 0:
			print('Select pca')
			root = Tk()
			root.withdraw()
			root.update()
			filepath_pca = filedialog.askopenfilename()			
			root.destroy()
			pca = load_signal_2(filepath_pca)
			Features = pca.transform(Features)
		
		
		
		
		#Obtain Filepaths Features
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			print('Select file with features')
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
		
		#Construct Features NArray
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		for filepath in Filepaths:
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
		
		
		
		#Predict Clusters from Features
		Labels = kmeans.predict(Features)
		Index_labels = []
		for k in range(config['clusters']):
			it_index_labels = []
			count = 0
			for label in Labels:
				if label == k:
					it_index_labels.append(count)
				count += 1
			Index_labels.append(it_index_labels)
		for k in range(config['clusters']):
			print('class ', k)
			print(len(Index_labels[k]))
	
	
	elif config['mode'] == 'bursts_per_file':


		#Obtain Filepaths Features
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			print('Select file with features')
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
		
		#Construct Features NArray
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		for filepath in Filepaths:
			filename = os.path.basename(filepath)
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index)
			# print(rownames[-1])
			# print(rownames[-1][-20])
			

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
			
			# if config['scaler'] != None:
				# Features = scaler.transform(Features)
		
			# if config['pca_comp'] != 0:
				# Features = pca.transform(Features)

			#Check same files with multiple bursts
			Idx_Same_Files = {}
			for element in rownames:
				if element[-20:-5] not in Idx_Same_Files.keys():
					Idx_Same_Files[element[-20:-5]] = []
			for i in range(len(rownames)):
				for key in Idx_Same_Files.keys():
					if key == rownames[i][-20:-5]:
						Idx_Same_Files[key].append(i)
						break
			# print(Idx_Same_Files)
			
			Burst_per_File = {'burst_per_file':[]}
			# filesnames = []
			for key in Idx_Same_Files.keys():
				# filesnames.append(key)
				Burst_per_File['burst_per_file'].append(len(Idx_Same_Files[key]))
			
			
			writer = pd.ExcelWriter('Simple_Condensed_' + filename)

			DataFr_max = pd.DataFrame(data=Burst_per_File, index=Idx_Same_Files.keys())		
			
			DataFr_max.to_excel(writer, sheet_name='Condensed_Bursts')		

			writer.close()
	
	# elif config['mode'] == 'predict_clusters_per_file':

		
		# print('Select model')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath_model = filedialog.askopenfilename()			
		# root.destroy()
		# kmeans = load_signal_2(filepath_model)
		# # print(type(kmeans))
		# # sys.exit()
		
		# if config['scaler'] != None:
			# print('Select scaler')
			# root = Tk()
			# root.withdraw()
			# root.update()
			# filepath_scaler = filedialog.askopenfilename()			
			# root.destroy()
			# scaler = load_signal_2(filepath_scaler)
		
		# if config['pca_comp'] != 0:
			# print('Select pca')
			# root = Tk()
			# root.withdraw()
			# root.update()
			# filepath_pca = filedialog.askopenfilename()			
			# root.destroy()
			# pca = load_signal_2(filepath_pca)
		
		
		
		
		# #Obtain Filepaths Features
		# if config['mypath'] == None:
			# root = Tk()
			# root.withdraw()
			# root.update()
			# print('Select file with features')
			# Filepaths = filedialog.askopenfilenames()			
			# root.destroy()
		# else:		
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
		
		# #Construct Features NArray
		# names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		# Dict_Features = {}
		# for feature in names_features:
			# Dict_Features[feature] = []
		
		# for filepath in Filepaths:
			# filename = os.path.basename(filepath)
			# mydict = pd.read_excel(filepath)
			# rownames = list(mydict.index)
			# # print(rownames[-1])
			# # print(rownames[-1][-20])
			

			# mydict = mydict.to_dict(orient='list')
			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
				
			# n_samples = len(Dict_Features[names_features[0]])
			# n_features = len(names_features)		
			# Features = np.zeros((n_samples, n_features))
			# count = 0
			# for feature in names_features:
				# Features[:, count] = Dict_Features[feature]
				# count += 1
			
			# if config['scaler'] != None:
				# Features = scaler.transform(Features)
		
			# if config['pca_comp'] != 0:
				# Features = pca.transform(Features)
			
			
			# Labels = kmeans.predict(Features)
			# Index_labels = []
			# for k in range(config['clusters']):
				# it_index_labels = []
				# count = 0
				# for label in Labels:
					# if label == k:
						# it_index_labels.append(count)
					# count += 1
				# Index_labels.append(it_index_labels)
				
			# for k in range(config['clusters']):
				# print('class ', k)
				# print(len(Index_labels[k]))

			# #Check same files with multiple bursts
			# Idx_Same_Files = {}
			# for element in rownames:
				# if element[-20:-5] not in Idx_Same_Files.keys():
					# Idx_Same_Files[element[-20:-5]] = []
			# for i in range(len(rownames)):
				# for key in Idx_Same_Files.keys():
					# if key == rownames[i][-20:-5]:
						# Idx_Same_Files[key].append(i)
						# break
			# # print(Idx_Same_Files)
			
			# Burst_per_File = {'burst_per_file':[]}
			# for k in range(config['clusters']):
				# Burst_per_File['cluster_' + str(k)] = []
			# # filesnames = []
			
			# for key in Idx_Same_Files.keys():
				# # filesnames.append(key)
				# Burst_per_File['burst_per_file'].append(len(Idx_Same_Files[key]))
				# for k in range(config['clusters']):
					# vec = []
					# for element in Idx_Same_Files[key]:						
						# if element in Index_labels[k]:
							# vec.append(element)
					# Burst_per_File['cluster_' + str(k)].append(len(vec))
					
				
			
			# writer = pd.ExcelWriter('Condensed_' + filename)

			# DataFr_max = pd.DataFrame(data=Burst_per_File, index=Idx_Same_Files.keys())		
			
			# DataFr_max.to_excel(writer, sheet_name='Condensed_Bursts')		

			# writer.close()
	
	elif config['mode'] == 'predict_clusters_per_file':

		
		print('Select model')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_model = filedialog.askopenfilename()			
		root.destroy()
		kmeans = load_signal_2(filepath_model)
		# print(type(kmeans))
		# sys.exit()
		
		if config['scaler'] != 'OFF':
			print('Select scaler')
			root = Tk()
			root.withdraw()
			root.update()
			filepath_scaler = filedialog.askopenfilename()			
			root.destroy()
			scaler = load_signal_2(filepath_scaler)
		
		if config['pca_comp'] != 0:
			print('Select pca')
			root = Tk()
			root.withdraw()
			root.update()
			filepath_pca = filedialog.askopenfilename()			
			root.destroy()
			pca = load_signal_2(filepath_pca)
		
		
		
		
		#Obtain Filepaths Features
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			print('Select file with features')
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
		
		#Construct Features NArray
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		for filepath in Filepaths:
			filename = os.path.basename(filepath)
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index)
			# print(rownames[-1])
			# print(rownames[-1][-20])
			

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
			
			if config['scaler'] != 'OFF':
				Features = scaler.transform(Features)
		
			if config['pca_comp'] != 0:
				Features = pca.transform(Features)
			
			
			
			
			
			#Check same files with multiple bursts
			Idx_Same_Files = {}
			for element in rownames:
				if element[-20:-5] not in Idx_Same_Files.keys():
					Idx_Same_Files[element[-20:-5]] = []
			for i in range(len(rownames)):
				for key in Idx_Same_Files.keys():
					if key == rownames[i][-20:-5]:
						Idx_Same_Files[key].append(i)
						break
			
			Clusters = {}
			for i in range(config['clusters']):
				Clusters['cluster_' + str(i)] = []
			
			for key in Idx_Same_Files.keys():
				accum = np.array([0 for j in range(config['clusters'])])
				for i in Idx_Same_Files[key]:
					# print(Features[Idx_Same_Files[key][i]])
					# print(Features[0])
					# print(kmeans.predict([Features[0]]))
					# print(kmeans.predict(Features))
					# sys.exit()
					clf = kmeans.predict([Features[i]])
					
					for k in range(config['clusters']):
						if clf[0] == k:
							accum[k] += 1
							break
				for h in range(config['clusters']):
					Clusters['cluster_' + str(h)].append(accum[h])
			
			
			
			
			
			
			# Index_labels = []
			# for k in range(config['clusters']):
				# it_index_labels = []
				# count = 0
				# for label in Labels:
					# if label == k:
						# it_index_labels.append(count)
					# count += 1
				# Index_labels.append(it_index_labels)
				
			# for k in range(config['clusters']):
				# print('class ', k)
				# print(len(Index_labels[k]))

			
			# # print(Idx_Same_Files)
			
			# Burst_per_File = {'burst_per_file':[]}
			# for k in range(config['clusters']):
				# Burst_per_File['cluster_' + str(k)] = []
			# # filesnames = []
			
			# for key in Idx_Same_Files.keys():
				# # filesnames.append(key)
				# Burst_per_File['burst_per_file'].append(len(Idx_Same_Files[key]))
				# for k in range(config['clusters']):
					# vec = []
					# for element in Idx_Same_Files[key]:						
						# if element in Index_labels[k]:
							# vec.append(element)
					# Burst_per_File['cluster_' + str(k)].append(len(vec))
					
			total = np.zeros(len(Idx_Same_Files.keys()))
			for k in range(config['clusters']):
				total += np.array(Clusters['cluster_' + str(k)])
				
			Clusters['total'] = total
			
			writer = pd.ExcelWriter('Condensed_' + filename)

			DataFr_max = pd.DataFrame(data=Clusters, index=Idx_Same_Files.keys())		
			
			DataFr_max.to_excel(writer, sheet_name='Condensed_Bursts')		

			writer.close()



		
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
			if element == 'files':
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

	if config['rs'] != None:
		config['rs'] = int(config['rs'])
	config['pca_comp'] = int(config['pca_comp'])
	config['clusters'] = int(config['clusters'])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config
	


	
	
	
	






if __name__ == '__main__':
	main(sys.argv)


