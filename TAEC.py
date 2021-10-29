
#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os.path
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser
import pandas as pd
import os.path
from os import listdir


# from THR_Burst_Detection import thr_burst_detector
# from THR_Burst_Detection import thr_burst_detector_rev
from THR_Burst_Detection import full_thr_burst_detector
from THR_Burst_Detection import read_threshold
from THR_Burst_Detection import plot_burst_rev


import math
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler  
import datetime
import scipy

#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']

InputsOpt_Defaults = {'channel_ae':'AE1', 'fs_ae':2.e6, 'channel_trigger':'Trigger2', 'fs_trigger':1.e3, 'threshold':7., 'seg_lockout':1.}


# InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'plot':'OFF', 'n_files':1, 'title_plot':None, 'thr_value':30., 'thr_mode':'fixed_value', 'window_time':0.05, 'save_plot':'OFF', 'file':'OFF', 'time_segments':1., 'stella':1500, 'lockout':3000, 'highpass':20.e3, 'mv':'ON', 'mypath':None}

# InputsOpt_Defaults = {'name':'auto', 'plot':'ON', 'n_files':1, 'title_plot':None, 'thr_value':3.6, 'thr_mode':'factor_rms', 'window_time':0.001, 'stella':300, 'lockout':300, 'filter':['highpass', 5.e3, 3], 'mv':'ON', 'mypath':None, 'save':'ON', 'save_plot':'OFF', 'amp_db':'OFF', 'db_pa':37.}
# gearbox mio: thr_60, wt_0.001, hp_70k, stella_100, lcokout 200


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	if config['mode'] == 'get_abbrennen':
		print('Select Signals AE')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()	

		print('Select Signals Trigger')
		root = Tk()
		root.withdraw()
		root.update()
		FilepathsT = filedialog.askopenfilenames()			
		root.destroy()

		T_inis = []
		T_ends = []
		Durations = []
		for filepath, filepathT in zip(Filepaths, FilepathsT):
			filename = os.path.basename(filepath)[:-5]
			signal = load_signal(filepath, config['channel_ae'])
			trigger = load_signal(filepathT, config['channel_trigger'])
			fig, ax = plt.subplots(nrows=2, ncols=1)
			
			
			length_signal = len(signal)
			length_trigger = len(trigger)
			
			t_signal = np.arange(length_signal)/config['fs_ae']
			t_trigger = np.arange(length_trigger)/config['fs_trigger']
			
			# ax[0].plot(t_signal, signal)
			# ax[1].plot(t_trigger, trigger)
			# print(config['fs_trigger'])
			
			
			flag = 'ON'
			idx_ini = -1
			while flag == 'ON':
				idx_ini += 1
				if trigger[idx_ini] <= config['threshold']:
					t_ini = t_trigger[idx_ini]
					flag = 'OFF'
					
			flag = 'ON'
			idx_end = idx_ini + int(config['seg_lockout']*config['fs_trigger'])
			while flag == 'ON':
				idx_end += 1
				if trigger[idx_end] > config['threshold']:
					t_end = t_trigger[idx_end]
					flag = 'OFF'
			# print(t_ini)
			# print(t_end)
			# plt.show()
			T_inis.append(t_ini)
			T_ends.append(t_end)
			Durations.append(t_end-t_ini)
			abbrennen = signal[int(t_ini*config['fs_ae']) : int(t_end*config['fs_ae'])]		

			mydict = {config['channel_ae']:abbrennen}
			scipy.io.savemat(filename + '_Abbrennen' + '.mat', mydict)
		print('Initial_Times: ', T_inis)
		print('Ending Times: ', T_ends)
		print('Durations: ', Durations)
	
	
	elif config['mode'] == 'get_nachbehandlung':
		print('Select Signals AE')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()	

		print('Select Signals Trigger')
		root = Tk()
		root.withdraw()
		root.update()
		FilepathsT = filedialog.askopenfilenames()			
		root.destroy()

		T_inis = []
		T_ends = []
		Durations = []
		for filepath, filepathT in zip(Filepaths, FilepathsT):
			filename = os.path.basename(filepath)[:-5]
			signal = load_signal(filepath, config['channel_ae'])
			trigger = load_signal(filepathT, config['channel_trigger'])
			fig, ax = plt.subplots(nrows=2, ncols=1)
			
			
			length_signal = len(signal)
			length_trigger = len(trigger)
			
			t_signal = np.arange(length_signal)/config['fs_ae']
			t_trigger = np.arange(length_trigger)/config['fs_trigger']
			
			# ax[0].plot(t_signal, signal)
			# ax[1].plot(t_trigger, trigger)
			# print(config['fs_trigger'])
			
			
			flag = 'ON'
			idx_ini = -1
			while flag == 'ON':
				idx_ini += 1
				if trigger[idx_ini] <= config['threshold']:
					t_ini = t_trigger[idx_ini]
					flag = 'OFF'
					
			flag = 'ON'
			idx_end = idx_ini + int(config['seg_lockout']*config['fs_trigger'])
			while flag == 'ON':
				idx_end += 1
				if trigger[idx_end] > config['threshold']:
					t_end = t_trigger[idx_end]
					flag = 'OFF'

			T_inis.append(t_ini)
			T_ends.append(t_end)
			Durations.append(t_end-t_ini)
			# abbrennen = signal[int(t_ini*config['fs_ae']) : int(t_end*config['fs_ae'])]
			nachbehandlung = signal[int(t_end*config['fs_ae']) : int((t_end+10)*config['fs_ae'])]

			mydict = {config['channel_ae']:nachbehandlung}
			scipy.io.savemat(filename + '_Nachbehandlung10' + '.mat', mydict)
			
		print('Initial_Times: ', T_inis)
		print('Ending Times: ', T_ends)
		print('Durations: ', Durations)
	

	
	elif config['mode'] == 'plot_features':
		print('Waehlen XLS von Klasse 1 (gut)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Waehlen XLS von Klasse 1 -1 (schlecht)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		fig, ax = plt.subplots(nrows=2, ncols=2)
		# names_features = ['count', 'dura']
		# plt.xlabel('Überschwingungen', fontsize=14), plt.ylabel('Dauer [us]', fontsize=14)
		# names_features = ['amax', 'rms']
		# plt.xlabel('Max. Amplitude [mV]', fontsize=14), plt.ylabel('RMS [mV]', fontsize=14)
		# names_features = ['rise', 'kurt']
		# plt.xlabel('Anstiegszeit [us]', fontsize=14), plt.ylabel('Kurtosis', fontsize=14)
		# names_features = ['crest', 'freq']
		# plt.xlabel('Crest Faktor', fontsize=14), plt.ylabel('Hauptfrequenz [kHz]', fontsize=14)
		
		
		
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [1 for i in range(len(mydict[element]))]
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [-1 for i in range(len(mydict[element]))]

		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)		
		
		Idx_Gut = [i for i in range(n_samples) if Labels[i] == 1]
		Idx_Schlecht = [i for i in range(n_samples) if Labels[i] == -1]
		
		alpha = 0.5
		ax[0][0].scatter(np.array(Dict_Features['count'])[Idx_Gut], np.array(Dict_Features['dura'])[Idx_Gut], color='blue', marker='s', alpha=alpha, label='Gut')
		ax[0][0].scatter(np.array(Dict_Features['count'])[Idx_Schlecht], np.array(Dict_Features['dura'])[Idx_Schlecht], color='red', alpha=alpha, label='Schlecht')
		ax[0][0].set_xlabel('Überschwingungen [-]', fontsize=14), ax[0][0].set_ylabel('Dauer [us]', fontsize=14)
		
		ax[1][0].scatter(np.array(Dict_Features['amax'])[Idx_Gut], np.array(Dict_Features['rise'])[Idx_Gut], color='blue', marker='s', alpha=alpha, label='Gut')
		ax[1][0].scatter(np.array(Dict_Features['amax'])[Idx_Schlecht], np.array(Dict_Features['rise'])[Idx_Schlecht], color='red', alpha=alpha, label='Schlecht')
		ax[1][0].set_xlabel('Max. Amplitude [mV]', fontsize=14), ax[1][0].set_ylabel('Anstiegszeit [us]', fontsize=14)
		
		ax[0][1].scatter(np.array(Dict_Features['kurt'])[Idx_Gut], np.array(Dict_Features['rms'])[Idx_Gut], color='blue', marker='s', alpha=alpha, label='Gut')
		ax[0][1].scatter(np.array(Dict_Features['kurt'])[Idx_Schlecht], np.array(Dict_Features['rms'])[Idx_Schlecht], color='red', alpha=alpha, label='Schlecht')
		ax[0][1].set_xlabel('Kurtosis [-]', fontsize=14), ax[0][1].set_ylabel('RMS Wert [mV]', fontsize=14)
		
		ax[1][1].scatter(np.array(Dict_Features['crest'])[Idx_Gut], np.array(Dict_Features['freq'])[Idx_Gut], color='blue', marker='s', alpha=alpha, label='Gut')
		ax[1][1].scatter(np.array(Dict_Features['crest'])[Idx_Schlecht], np.array(Dict_Features['freq'])[Idx_Schlecht], color='red', alpha=alpha, label='Schlecht')
		ax[1][1].set_xlabel('Crest Faktor [-]', fontsize=14), ax[1][1].set_ylabel('Hauptfrequenz [kHz]', fontsize=14)
		
		
		ax[0][0].tick_params(axis='both', labelsize=12)
		ax[1][0].tick_params(axis='both', labelsize=12)
		ax[0][1].tick_params(axis='both', labelsize=12)
		ax[1][1].tick_params(axis='both', labelsize=12)
		
		ax[0][0].legend(fontsize=12)
		ax[0][1].legend(fontsize=12)
		ax[1][0].legend(fontsize=12)
		ax[1][1].legend(fontsize=12)
		plt.tight_layout()
		plt.show()
	
	elif config['mode'] == 'svm_one_class_valid':

		
		print('Waehlen XLS von Klasse 1 (gut)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print('Waehlen XLS von Klasse 1 -1 (schlecht)')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths2 = filedialog.askopenfilenames()
		# root.destroy()
		

		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		# Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			# Labels += [1 for i in range(len(mydict[element]))]
			
		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)
		
		Features = np.zeros((n_samples, n_features))
		count = 0
		for feature in names_features:
			Features[:, count] = Dict_Features[feature]
			count += 1

		scaler = StandardScaler()
		scaler.fit(Features)
		Features = scaler.transform(Features)
		
		from sklearn.svm import OneClassSVM
		from sklearn.model_selection import train_test_split
		
		Normals = []
		Anormals = []
		for i in range(10):
			print('Validation n°', i)
			X_train, X_valid = train_test_split(Features, test_size=0.25, random_state=None)		
			clf = OneClassSVM(nu=0.5, kernel='sigmoid')
			clf.fit(X_train)
			y_pred = clf.predict(X_valid)
			
			normal = 0
			anormal = 0
			for element in y_pred:
				if element == 1:
					normal += 1
				elif element == -1:
					anormal += 1
				else:
					print('error 9475')
					sys.exit()
			normal = normal / len(y_pred)
			anormal = anormal / len(y_pred)
			# print('normal ', normal)
			# print('anormal ', anormal)
			Normals.append(normal)
			Anormals.append(anormal)
		print('Normal rate: ', np.mean(np.array(Normals)))
		print('Anormal rate: ', np.mean(np.array(Anormals)))

		
		# for filepath in Filepaths2:
			# mydict = pd.read_excel(filepath)			
			# mydict = mydict.to_dict(orient='list')			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
		
		
	elif config['mode'] == 'svm_one_class_test':

		
		print('Waehlen XLS von Klasse 1 (gut)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		

		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		# Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			# Labels += [1 for i in range(len(mydict[element]))]
			
		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)
		
		Features = np.zeros((n_samples, n_features))
		count = 0
		for feature in names_features:
			Features[:, count] = Dict_Features[feature]
			count += 1

		scaler = StandardScaler()
		scaler.fit(Features)
		Features = scaler.transform(Features)
		
		from sklearn.svm import OneClassSVM
		from sklearn.model_selection import train_test_split
		
	
		clf = OneClassSVM(nu=0.01, kernel='sigmoid')
		clf.fit(Features)
		

		print('Waehlen XLS von Klasse -1 (schlecht)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
		Dict_Features2 = {}
		for feature in names_features:
			Dict_Features2[feature] = []
		
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				
				Dict_Features2[element] += mydict[element]
			
		n_samples2 = len(Dict_Features2[names_features[0]])
		n_features2 = len(names_features)
		
		Features2 = np.zeros((n_samples2, n_features2))
		count = 0
		for feature in names_features:
			fact = np.random.randn()
			print(fact)
			Features2[:, count] = np.array(Dict_Features2[feature])*fact*1000
			count += 1

		# scaler = StandardScaler()
		# scaler.fit(Features2)
		Features2 = scaler.transform(Features2)
		
		y_pred = clf.predict(Features2)
			
		normal = 0
		anormal = 0
		for element in y_pred:
			if element == 1:
				normal += 1
			elif element == -1:
				anormal += 1
			else:
				print('error 9475')
				sys.exit()
		normal = normal / len(y_pred)
		anormal = anormal / len(y_pred)
		
		print('Normal rate: ', normal)
		print('Anormal rate: ', anormal)
	
	elif config['mode'] == 'novelty_valid':

		
		print('Waehlen XLS von Klasse 1 (gut)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print('Waehlen XLS von Klasse 1 -1 (schlecht)')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths2 = filedialog.askopenfilenames()
		# root.destroy()
		

		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		# Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			# Labels += [1 for i in range(len(mydict[element]))]
			
		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)
		
		Features = np.zeros((n_samples, n_features))
		count = 0
		for feature in names_features:
			Features[:, count] = Dict_Features[feature]
			count += 1

		# scaler = StandardScaler()
		# scaler.fit(Features)
		# Features = scaler.transform(Features)
		
		# from sklearn.decomposition import PCA
		# pca = PCA(n_components=4)
		# Features = pca.fit_transform(Features)
		# print(pca.explained_variance_ratio_ )
		
		from sklearn.neighbors import LocalOutlierFactor
		from sklearn.model_selection import train_test_split
		
	
		# clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
		# clf.fit(Features)
		
		Normals = []
		Anormals = []
		for i in range(20):
			print('Validation n°', i)
			X_train, X_valid = train_test_split(Features, test_size=0.25, random_state=None)		
			clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
			clf.fit(X_train)
			y_pred = clf.predict(X_valid)
			
			normal = 0
			anormal = 0
			for element in y_pred:
				if element == 1:
					normal += 1
				elif element == -1:
					anormal += 1
				else:
					print('error 9475')
					sys.exit()
			normal = normal / len(y_pred)
			anormal = anormal / len(y_pred)
			# print('normal ', normal)
			# print('anormal ', anormal)
			Normals.append(normal)
			Anormals.append(anormal)
		print('Normal rate: ', np.mean(np.array(Normals)))
		print('Anormal rate: ', np.mean(np.array(Anormals)))
		
		print('Normal STD: ', np.std(np.array(Normals)))
		print('Anormal STD: ', np.std(np.array(Anormals)))

		
		# for filepath in Filepaths2:
			# mydict = pd.read_excel(filepath)			
			# mydict = mydict.to_dict(orient='list')			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
	
	elif config['mode'] == 'novelty_test':
		# import sklearn
		# print('The scikit-learn version is {}.'.format(sklearn.__version__))
		# sys.exit()
		print('Waehlen XLS von Klasse 1 (gut)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		

		names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'kurt', 'ra','rise', 'rms']
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		# Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			# Labels += [1 for i in range(len(mydict[element]))]
			
		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)
		
		Features = np.zeros((n_samples, n_features))
		count = 0
		for feature in names_features:
			Features[:, count] = Dict_Features[feature]
			count += 1

		# scaler = StandardScaler()
		# scaler.fit(Features)
		# Features = scaler.transform(Features)
		
		
		
		# from sklearn.decomposition import PCA
		# pca = PCA(n_components=4)
		# Features = pca.fit_transform(Features)
		# print(pca.explained_variance_ratio_ )
		
		from sklearn.neighbors import LocalOutlierFactor
		from sklearn.model_selection import train_test_split
		
	
		clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
		clf.fit(Features)
		

		print('Waehlen XLS von Klasse -1 (schlecht)')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
		Dict_Features2 = {}
		for feature in names_features:
			Dict_Features2[feature] = []
		
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				
				Dict_Features2[element] += mydict[element]
			
		n_samples2 = len(Dict_Features2[names_features[0]])
		n_features2 = len(names_features)
		
		Features2 = np.zeros((n_samples2, n_features2))
		count = 0
		for feature in names_features:
			# fact = np.random.randn()
			# print(fact)
			Features2[:, count] = np.array(Dict_Features2[feature])
			count += 1

		# scaler = StandardScaler()
		# scaler.fit(Features2)
		# Features2 = scaler.transform(Features2)
		

		# Features2 = pca.transform(Features2)
		
		y_pred = clf.predict(Features2)
		
		normal = 0
		anormal = 0
		for element in y_pred:
			if element == 1:
				normal += 1
			elif element == -1:
				anormal += 1
			else:
				print('error 9475')
				sys.exit()
		normal = normal / len(y_pred)
		anormal = anormal / len(y_pred)
		
		print('Normal rate: ', normal)
		print('Anormal rate: ', anormal)

		
	else:
		print('wrong_mode')
		
		
		
	return

def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element' or element == 'filter':
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
	# if config['power2'] != 'auto' and config['power2'] != 'OFF':
		# config['power2'] = int(config['power2'])
	# config['mode'] = float(config['fs_tacho'])
	config['fs_ae'] = float(config['fs_ae'])
	config['fs_trigger'] = float(config['fs_trigger'])
	# config['n_files'] = int(config['n_files'])
	# config['stella'] = int(config['stella'])
	config['threshold'] = float(config['threshold'])
	config['seg_lockout'] = float(config['seg_lockout'])
	# config['highpass'] = float(config['highpass'])
	# config['window_time'] = float(config['window_time'])
	# # config['time_segments'] = float(config['time_segments'])
	# config['lockout'] = int(config['lockout'])
	
	
	
	# if config['filter'][0] != 'OFF':
		# if config['filter'][0] == 'bandpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2]), float(config['filter'][3])]
		# elif config['filter'][0] == 'highpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		# elif config['filter'][0] == 'lowpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		# else:
			# print('error filter 87965')
			# sys.exit()
	
	
	
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config

def max_norm_correlation(signal1, signal2):
	correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')
	return np.max(correlation)

if __name__ == '__main__':
	main(sys.argv)
