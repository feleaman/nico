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
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import RobustScaler

from sklearn.tree import DecisionTreeClassifier
import datetime
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))

from sklearn import tree
import graphviz 
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'


Inputs = ['mode', 'save', 'files', 'classifications']


InputsOpt_Defaults = {'power2':20, 'plot':'OFF', 'title':None, 'title2':None, 'fs':1.e6, 'alpha':1.e-1, 'tol':1.e-6, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[500], 'solver':'adam', 'rs':1, 'activation':'tanh', 'data_norm':'OFF', 'processing':'OFF', 'demod_filter':'OFF', 'demod_prefilter':'OFF', 'demod_rect':'only_positives', 'demod_dc':'without_dc', 'diff':'OFF', 'window_time':0.001, 'rs_cv':2}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'mode1':
		
		Features = []
		Classifications = []
		Classifications_bin = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			old_classification = info_classification['classification']
			
			classification = old_classification
			# classification = []
			# vImpar = old_classification[1::2]
			# vPar = old_classification[0::2]			
			# for epar, eimpar in zip(vPar, vImpar):
				# if (epar == 1) or (epar == 2) or (eimpar == 1) or (eimpar == 2):
					# classification.append(1)
				# else:
					# classification.append(0)
			# # sys.exit()
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]

			
			if config['data_norm'] == 'per_rms':
				signal = np.array(signal)
				signal = signal / signal_rms(signal)
				signal = signal.tolist()
			else:
				print('without norm')
			
			if config['processing'] != 'OFF':
				print('with processing')
				signal = signal_processing(signal, config)
			else:
				print('without processing')
			
			if config['diff'] != 'OFF':
				print('with diff')
				signal= diff_signal_eq(signal, config['diff'])
			
			# plt.plot(signal)
			# plt.show()
			checked_windows_in_file = len(classification)
			window_points = int(config['window_time']*config['fs'])
			
			# print('RMS value+++++++++')
			# print(signal_rms(signal))
			# a = input('pause')
			caca = 0
			for count in range(checked_windows_in_file):
				signal_window = signal[count*window_points:(count+1)*window_points]			
				values = sorted(signal_window)
				Features.append(values)
				
				if classification[count] == 2:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
					caca += 1
				elif classification[count] == 1:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
					caca += 1
				elif classification[count] == 0:
					Classifications.append([0, 0])
					Classifications_bin.append(0)
				else:
					print('fatal error 4978')
					sys.exit()
			
			# print('BURST value+++++++++')
			# print(caca)
			# a = input('pause')
			# sys.exit()
				

		
		
		features_cross, features_test, classes_cross, classes_test = train_test_split(Features, Classifications_bin, test_size=0.3, random_state=config['rs_cv'], stratify=Classifications_bin)
		
		classes_cross = np.array(classes_cross)
		features_cross = np.array(features_cross)

		skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=False)
		Scores_cv = []
		for train_index, test_index in skf.split(features_cross, classes_cross):
			features_train_cv, features_test_cv = features_cross[train_index], features_cross[test_index]
			classes_train_cv, classes_test_cv = classes_cross[train_index], classes_cross[test_index]
			
			scaler_cv = StandardScaler()
			# scaler_cv = RobustScaler()
			scaler_cv.fit(features_train_cv)
			features_train_cv = scaler_cv.transform(features_train_cv)	
			features_test_cv = scaler_cv.transform(features_test_cv)

			clf_cv = MLPClassifier(solver=config['solver'], alpha=config['alpha'], hidden_layer_sizes=config['layers'], random_state=config['rs'], activation=config['activation'], tol=config['tol'], verbose=True, max_iter=config['max_iter'])
			
			clf_cv.fit(features_train_cv, classes_train_cv)
			
			classes_prediction_cv = clf_cv.predict(features_test_cv)
			
			scores = obtain_bin_scores(classes_prediction_cv, classes_test_cv)
			Scores_cv.append(scores)
		
		scaler = StandardScaler()
		# scaler = RobustScaler()
		scaler.fit(features_cross)
		features_cross = scaler.transform(features_cross)	
		features_test = scaler.transform(features_test)

		clf = MLPClassifier(solver=config['solver'], alpha=config['alpha'], hidden_layer_sizes=config['layers'], random_state=config['rs'], activation=config['activation'], tol=config['tol'], verbose=True, max_iter=config['max_iter'])
		
		clf.fit(features_cross, classes_cross)
		classes_prediction = clf.predict(features_test)
		scores_test = obtain_bin_scores(classes_prediction, classes_test)
		# print(scores_test)
		Output = {'clf':clf, 'scores_cv':Scores_cv, 'scores_test':scores_test, 'scaler':scaler, 'config':config}
		# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		save_pickle('clf_' + config['title'] + '_' + config['title2'] + '.pkl', Output)
	
	
	elif config['mode'] == 'test_tree':
		Features = [18, 10, 11, 12, 15, 13, 15, 18, 19, 11, 1, 5, 8, 3, 9, 6, 7, 6, 9, 9]
		Classes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		
		Features = [[element] for element in Features]
		
		print(len(Features))
		print(len(Classes))
		
		clf = DecisionTreeClassifier(random_state=0, min_samples_split=2, max_features=1, max_depth=1)
		
		clf.fit(Features, Classes)
		
		# # tree = clf.tree_

		
		tree = clf.tree_
		print(tree.threshold[0])
		
		# sys.exit()
		from sklearn import tree
		import graphviz 
		dot_data = tree.export_graphviz(clf, out_file=None) 
		
		print(dot_data)
		print(dot_data[1])
		
		graph = graphviz.Source(dot_data) 
		
		file = open('caca.txt', 'w')

		file.write(str(graph)) 
		# file.write(“To add more lines.”)

		file.close()
		print(graph)
		graph
		sys.exit()
		
		
		graph.render("airis") 
		
	
	elif config['mode'] == 'mode1thr':
		
		Features = []
		Classifications = []
		Classifications_bin = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			classification = info_classification['classification']
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]

			
			if config['data_norm'] == 'per_rms':
				signal = np.array(signal)
				signal = signal / signal_rms(signal)
				signal = signal.tolist()
			else:
				print('without demod')
			
			# if config['processing'] != 'OFF':
				# print('with processing')
				# signal = signal_processing(signal, config)
			# else:
				# print('without processing')
			
			# if config['diff'] != 'OFF':
				# print('with diff')
				# signal= diff_signal_eq(signal, config['diff'])
			
			# plt.plot(signal)
			# plt.show()
			checked_windows_in_file = len(classification)
			window_points = int(config['window_time']*config['fs'])
			
			for count in range(checked_windows_in_file):
				signal_window = signal[count*window_points:(count+1)*window_points]			
				values = np.max(signal_window)
				Features.append([values])
				
				if classification[count] == 2:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 1:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 0:
					Classifications.append([0, 0])
					Classifications_bin.append(0)
				else:
					print('fatal error 4978')
					sys.exit()
				

		features_cross, features_test, classes_cross, classes_test = train_test_split(Features, Classifications_bin, test_size=0.3, random_state=config['rs_cv'], stratify=Classifications_bin)
		
		classes_cross = np.array(classes_cross)
		features_cross = np.array(features_cross)

		skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=False)
		Scores_cv = []
		Thresholds_cv = []
		count = 0
		for train_index, test_index in skf.split(features_cross, classes_cross):
			features_train_cv, features_test_cv = features_cross[train_index], features_cross[test_index]
			classes_train_cv, classes_test_cv = classes_cross[train_index], classes_cross[test_index]
			
			# scaler_cv = StandardScaler()
			# scaler_cv.fit(features_train_cv)
			# features_train_cv = scaler_cv.transform(features_train_cv)	
			# features_test_cv = scaler_cv.transform(features_test_cv)

			# clf_cv = MLPClassifier(solver=config['solver'], alpha=config['alpha'], hidden_layer_sizes=config['layers'], random_state=config['rs'], activation=config['activation'], tol=config['tol'], verbose=True, max_iter=config['max_iter'])
			
			clf_cv = DecisionTreeClassifier(random_state=0, min_samples_split=2, max_features=1, max_depth=1)
			
			clf_cv.fit(features_train_cv, classes_train_cv)
			
			classes_prediction_cv = clf_cv.predict(features_test_cv)
			
			
			scores = obtain_bin_scores(classes_prediction_cv, classes_test_cv)
			Scores_cv.append(scores)
			
			arbol = clf_cv.tree_
			Thresholds_cv.append(arbol.threshold[0])
			
			# dot_data = tree.export_graphviz(clf, out_file=None) 
			# graph = graphviz.Source(dot_data) 
			# graph.render('tree_' + str(count))
			# count += 1
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(features_cross)
		# features_cross = scaler.transform(features_cross)	
		# features_test = scaler.transform(features_test)
		
		
		# threshold =  = DecisionTreeClassifier(random_state=0, min_samples_split=2, max_features=1, max_depth=1)
		print(Thresholds_cv)
		print(np.mean(np.array(Thresholds_cv)))
		
		# print(Scores_cv)
		
		sum = 0.
		for mydict in Scores_cv:
			sum += mydict['mcc']
		mean = sum/10.
		
		sum2 = 0.
		for mydict in Scores_cv:
			sum2 += (mydict['mcc'] - mean)**2.0
		std = (sum2/10)**0.5
		
		print('mcc mean cv', mean)
		print('mcc std cv', std)
		
		
		# clf.fit(features_cross, classes_cross)
		# classes_prediction = clf.predict(features_test)
		# scores_test = obtain_bin_scores(classes_prediction, classes_test)
		
		# Output = {'clf':clf, 'scores_cv':Scores_cv, 'scores_test':scores_test, 'scaler':scaler, 'config':config}
		# # stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		# save_pickle('clf_' + config['title'] + '_' + config['title2'] + '.pkl', Output)
	
	
	elif config['mode'] == 'mode1thr_man':
		
		Features = []
		Classifications = []
		Classifications_bin = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			classification = info_classification['classification']
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]

			
			if config['data_norm'] == 'per_rms':
				signal = np.array(signal)
				signal = signal / signal_rms(signal)
				signal = signal.tolist()
			else:
				print('without demod')
			
			# if config['processing'] != 'OFF':
				# print('with processing')
				# signal = signal_processing(signal, config)
			# else:
				# print('without processing')
			
			# if config['diff'] != 'OFF':
				# print('with diff')
				# signal= diff_signal_eq(signal, config['diff'])
			
			# plt.plot(signal)
			# plt.show()
			checked_windows_in_file = len(classification)
			window_points = int(config['window_time']*config['fs'])
			
			for count in range(checked_windows_in_file):
				signal_window = signal[count*window_points:(count+1)*window_points]			
				values = np.max(signal_window)
				Features.append([values])
				
				if classification[count] == 2:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 1:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 0:
					Classifications.append([0, 0])
					Classifications_bin.append(0)
				else:
					print('fatal error 4978')
					sys.exit()
				

		features_cross, features_test, classes_cross, classes_test = train_test_split(Features, Classifications_bin, test_size=0.3, random_state=config['rs_cv'], stratify=Classifications_bin)
		
		classes_cross = np.array(classes_cross)
		features_cross = np.array(features_cross)

		skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=False)
		Scores_cv = []
		Thresholds_cv = []
		count = 0
		for train_index, test_index in skf.split(features_cross, classes_cross):
			features_train_cv, features_test_cv = features_cross[train_index], features_cross[test_index]
			classes_train_cv, classes_test_cv = classes_cross[train_index], classes_cross[test_index]
			

			
			# clf_cv = DecisionTreeClassifier(random_state=0, min_samples_split=2, max_features=1, max_depth=1)
			
			best_thr_cv = test_values_limit(ini=1., fin=7., step=0.01, features=features_train_cv, classes=classes_train_cv)
			# print(best_thr)
			# a = input('pause')
			# def test_values_limit(ini, fin, step, features, classes):
				# test_thr = list(np.arange(1, 7, 0.01))
				# function_error = []			
				# for limit in test_thr:
					# error = 0
					# for array_feat, array_class in zip(features_train_cv, classes_train_cv):
						# if array_feat[0] < limit:
							# prediction = 0
						# else:
							# prediction = 1
						# error = error + (array_class - prediction)**2.0
					# function_error.append(error)
				# best_index = np.argmin(np.array(function_error))
				# best_thr = test_thr[best_index]
			
			
			# clf_cv.fit(features_train_cv, classes_train_cv)
			classes_prediction_cv = []
			for array_feat_test in features_test_cv:
				if array_feat_test[0] < best_thr_cv:
					classes_prediction_cv.append(0)
				else:
					classes_prediction_cv.append(1)
			
			
			
			scores = obtain_bin_scores(classes_prediction_cv, classes_test_cv)
			Scores_cv.append(scores)
			
			# arbol = clf_cv.tree_
			Thresholds_cv.append(best_thr_cv)
			
			# dot_data = tree.export_graphviz(clf, out_file=None) 
			# graph = graphviz.Source(dot_data) 
			# graph.render('tree_' + str(count))
			# count += 1
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(features_cross)
		# features_cross = scaler.transform(features_cross)	
		# features_test = scaler.transform(features_test)
		
		
		# threshold =  = DecisionTreeClassifier(random_state=0, min_samples_split=2, max_features=1, max_depth=1)
		print(Thresholds_cv)
		print(np.mean(np.array(Thresholds_cv)))
		
		# print(Scores_cv)
		
		sum = 0.
		for mydict in Scores_cv:
			sum += mydict['mcc']
		mean = sum/10.
		
		sum2 = 0.
		for mydict in Scores_cv:
			sum2 += (mydict['mcc'] - mean)**2.0
		std = (sum2/10)**0.5
		
		print('mcc mean cv', mean)
		print('mcc std cv', std)
		
		
		best_thr_train = test_values_limit(ini=1., fin=7., step=0.01, features=features_cross, classes=classes_cross)
		
		classes_prediction_test = []
		for array_feat_test in features_test:
			if array_feat_test[0] < best_thr_train:
				classes_prediction_test.append(0)
			else:
				classes_prediction_test.append(1)
		
		
		# clf.fit(features_cross, classes_cross)
		# classes_prediction = clf.predict(features_test)
		scores_test = obtain_bin_scores(classes_prediction_test, classes_test)
		
		Output = {'scores_cv':Scores_cv, 'scores_test':scores_test, 'config':config, 'thr_cv':Thresholds_cv, 'thr_test':best_thr_train}
		# # stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		save_pickle(config['title'] + '_' + config['title2'] + '.pkl', Output)
	
	
	elif config['mode'] == 'mode1thr_man_kurt':
		
		Features = []
		Classifications = []
		Classifications_bin = []
		
		Super_Signal = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			classification = info_classification['classification']
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]

			
			if config['data_norm'] == 'per_rms':
				signal = np.array(signal)
				# signal = signal / signal_rms(signal)
				signal = signal.tolist()
			else:
				print('without demod')
			
			Super_Signal = Super_Signal + signal
			
			# if config['processing'] != 'OFF':
				# print('with processing')
				# signal = signal_processing(signal, config)
			# else:
				# print('without processing')
			
			# if config['diff'] != 'OFF':
				# print('with diff')
				# signal= diff_signal_eq(signal, config['diff'])
			
			# plt.plot(signal)
			# plt.show()
		print('kurtosis')
		print(scipy.stats.kurtosis(Super_Signal))
		
		print('rms')
		print(signal_rms(Super_Signal))
			
			
			
	
	elif config['mode'] == 'mode2thr':
		
		Features = []
		Classifications = []
		Classifications_bin = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			classification = info_classification['classification']
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]

			
			if config['data_norm'] == 'per_rms':
				signal = np.array(signal)
				signal = signal / signal_rms(signal)
				signal = signal.tolist()
			else:
				print('without demod')
			
			# if config['processing'] != 'OFF':
				# print('with processing')
				# signal = signal_processing(signal, config)
			# else:
				# print('without processing')
			
			# if config['diff'] != 'OFF':
				# print('with diff')
				# signal= diff_signal_eq(signal, config['diff'])
			
			# plt.plot(signal)
			# plt.show()
			checked_windows_in_file = len(classification)
			window_points = int(config['window_time']*config['fs'])
			
			for count in range(checked_windows_in_file):
				signal_window = signal[count*window_points:(count+1)*window_points]			
				values = np.max(signal_window)
				Features.append([values])
				
				if classification[count] == 2:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 1:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 0:
					Classifications.append([0, 0])
					Classifications_bin.append(0)
				else:
					print('fatal error 4978')
					sys.exit()
				

		features_cross, features_test, classes_cross, classes_test = train_test_split(Features, Classifications_bin, test_size=0.3, random_state=config['rs_cv'], stratify=Classifications_bin)
		
		# print(features_test)
		
		classes_prediction = []
		for element in features_test:
			if element[0] >= 5.2:
				classes_prediction.append(1)
			else:
				classes_prediction.append(0)
		
		# classes_prediction = clf.predict(features_test)
		scores_test = obtain_bin_scores(classes_prediction, classes_test)
		print(scores_test)
		# Output = {'clf':clf, 'scores_cv':Scores_cv, 'scores_test':scores_test, 'scaler':scaler, 'config':config}
		# # stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		# save_pickle('clf_' + config['title'] + '_' + config['title2'] + '.pkl', Output)
		
		
		
		
	elif config['mode'] == 'mode2':
		print('caca!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		Features = []
		Classifications = []
		Classifications_bin = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = 1000*load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			classification = info_classification['classification']
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]
			
			if config['data_norm'] == 'per_rms':
				signal = signal / signal_rms(signal)
				signal = signal.tolist()
			else:
				print('without demod')
			
			if config['processing'] != 'OFF':
				print('with processing')
				signal = signal_processing(signal, config)
			else:
				print('without processing')
			
			
			if config['diff'] != 'OFF':
				print('with diff')
				signal= diff_signal_eq(signal, config['diff'])
			# plt.plot(signal)
			# plt.show()
			t = [i/config['fs'] for i in range(len(signal))]
			
			fig, ax = plt.subplots()
			ax.plot(t, signal, color='darkblue')
			ax.set_xlabel('Time [s]', fontsize=13)
			ax.set_ylabel('Amplitude [mV]', fontsize=13)
			ax.set_title('Signal 1: 1500 [RPM] / 80% Load', fontsize=13)
			ax.tick_params(axis='both', labelsize=12)
			ax.set_xlim(left=0, right=1.048)
			
			
			checked_windows_in_file = len(classification)
			window_points = int(config['window_time']*config['fs'])
			
			for count in range(checked_windows_in_file):
				
				if (classification[count] == 2 or classification[count] == 1):			
					ax.axvspan(xmin=count*window_points/config['fs'], xmax=(count+1)*window_points/config['fs'], facecolor='y', alpha=0.5)
				else:
					print('!!!!!!!!')
			
			
			plt.show()
		
		# fig, ax = plt.subplots()
		# ax.boxplot(Features_Pos)
		# ax.boxplot(Features_Neg)
		# ax.legend()
		# plt.show()
	
	elif config['mode'] == 'mode3':
		
		Features = []
		Classifications = []
		Classifications_bin = []
	
		for filepath_signal, filepath_classification in zip(config['files'], config['classifications']):
			signal = 1000*load_signal(filepath_signal)
			
			info_classification = load_signal(filepath_classification)
			classification = info_classification['classification']
			
			if info_classification['filename'] != os.path.basename(filepath_signal):
				print('Wrong filename!!!')
				sys.exit()
			
			if config['power2'] != 'OFF':
				signal = signal[0:2**config['power2']]

			
			if config['data_norm'] == 'per_rms':
				signal = signal / signal_rms(signal)
				signal = signal.tolist()
				print('rms normalization')
			else:
				print('without norm')
			
			if config['processing'] != 'OFF':
				print('with processing')
				signal = signal_processing(signal, config)
			else:
				print('without processing')
			
			if config['diff'] != 'OFF':
				print('with diff')
				signal= diff_signal_eq(signal, config['diff'])
			
			# plt.plot(signal)
			# plt.show()
			checked_windows_in_file = len(classification)
			window_points = int(config['window_time']*config['fs'])
			
			for count in range(checked_windows_in_file):
				signal_window = signal[count*window_points:(count+1)*window_points]			
				values = sorted(signal_window)
				# plt.plot(values)
				# plt.show()
				Features.append(values)
				
				if classification[count] == 2:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 1:
					Classifications.append([1, 0])
					Classifications_bin.append(1)
				elif classification[count] == 0:
					Classifications.append([0, 0])
					Classifications_bin.append(0)
				else:
					print('fatal error 4978')
					sys.exit()
		
		
		# scaler = StandardScaler()
		# scaler = MinMaxScaler()
		# scaler = MaxAbsScaler()
		# scaler = RobustScaler()
		# # scaler = Normalizer()
		
		
		# scaler.fit(Features)
		# Features = scaler.transform(Features)
		
		
		
		Features_Pos = []
		Features_Neg = []
		count = 0
		for element in Features:
			if Classifications_bin[count] == 0:
				Features_Neg.append(element)
			elif Classifications_bin[count] == 1:
				Features_Pos.append(element)
			else:
				print('error 53365')
				sys.exit()
			count += 1

		
		Features_Pos = list(map(list, zip(*Features_Pos)))
		Features_Neg = list(map(list, zip(*Features_Neg)))
		
		

		
		
		Features_Pos_50 = []
		Features_Neg_50 = []	

		Features_Pos_75 = []
		Features_Neg_75 = []	

		Features_Pos_25 = []
		Features_Neg_25 = []	
		
		for element_pos in Features_Pos:
			# value_pos = np.mean(np.array(element_pos))
			# Features_Pos_Mean.append(value_pos)	
			v50 = np.percentile(np.absolute(np.array(element_pos)), 50)
			v25 = np.percentile(np.absolute(np.array(element_pos)), 25)
			v75 = np.percentile(np.absolute(np.array(element_pos)), 75)
			Features_Pos_50.append(v50)
			Features_Pos_25.append(v25)	
			Features_Pos_75.append(v75)				
			
		for element_neg in Features_Neg:
			# value_neg = np.mean(np.array(element_neg))
			# Features_Neg_Mean.append(value_neg)
			v50 = np.percentile(np.absolute(np.array(element_neg)), 50)
			v25 = np.percentile(np.absolute(np.array(element_neg)), 25)
			v75 = np.percentile(np.absolute(np.array(element_neg)), 75)
			Features_Neg_50.append(v50)
			Features_Neg_25.append(v25)	
			Features_Neg_75.append(v75)
			
		# fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
		# ax[0].plot(Features_Pos_Mean, label='pos')
		# ax[1].plot(Features_Neg_Mean, label='neg')
		# ax[0].legend()
		# ax[1].legend()
		# plt.show()
		
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
		ax.plot(Features_Neg_50, label='Median', linewidth=2)
		ax.plot(Features_Neg_25, label='First quartile', marker='^', markevery=99)
		ax.plot(Features_Neg_75, label='Third quartile', marker='o', markevery=99)
		ax.set_title('Pattern distribution of segments without a burst', fontsize=13.5)
		ax.set_xlabel('N° Feature', fontsize=13.5)
		ax.set_ylabel('Amplitude envelope [-]', fontsize=13.5)
		ax.tick_params(axis='both', labelsize=12)
		
		ax.legend()		
		plt.show()
		
		
		
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
		ax.plot(Features_Pos_50, label='Median', linewidth=2)
		ax.plot(Features_Pos_25, label='First quartile', marker='^', markevery=99)
		ax.plot(Features_Pos_75, label='Third quartile', marker='o', markevery=99)
		ax.set_title('Pattern distribution of segments with a burst', fontsize=13.5)
		ax.set_xlabel('N° Feature', fontsize=13.5)
		ax.set_ylabel('Amplitude envelope [-]', fontsize=13.5)
		ax.tick_params(axis='both', labelsize=12)
		
		ax.legend()		
		plt.show()
		
		
		# fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
		# ax[0].plot(Features_Pos_50, label='Median')
		# ax[0].plot(Features_Pos_25, label='First quartile')
		# ax[0].plot(Features_Pos_75, label='Third quartile')
		# ax[0].set_title('Pattern distribution of segments with a burst', fontsize=13)
		# ax[0].set_xlabel('N° Feature', fontsize=13)
		# ax[0].set_ylabel('Amplitude envelope [-]', fontsize=13)
		# ax[0].tick_params(axis='both', labelsize=12)
		
		
		# ax[1].plot(Features_Neg_50, label='Median')
		# ax[1].plot(Features_Neg_25, label='First quartile')
		# ax[1].plot(Features_Neg_75, label='Third quartile')
		# ax[1].set_title('Pattern distribution of segments without a burst', fontsize=13)
		# ax[1].set_xlabel('N° Feature', fontsize=13)
		# ax[1].tick_params(axis='both', labelsize=12)
		
		# ax[0].legend()
		# ax[1].legend()
		
		# fig.set_size_inches(10, 4)
		# fig.tight_layout()
		# plt.show()
		
		
		
		
		# fig, ax = plt.subplots(ncols=2, nrows=1)
		# ax[0].boxplot(Features_Pos, 0, '')
		# ax[1].boxplot(Features_Neg, 0, '')		

		plt.show()

	
	return

def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'files' or element == 'classifications' or element == 'demod_filter' or element == 'demod_prefilter' or element == 'layers':
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
	if config['power2'] != 'auto' and config['power2'] != 'OFF':
		config['power2'] = int(config['power2'])
	config['fs'] = float(config['fs'])
	config['window_time'] = float(config['window_time'])
	
	config['alpha'] = float(config['alpha'])
	config['tol'] = float(config['tol'])	
	config['learning_rate_init'] = float(config['learning_rate_init'])	
	#Type conversion to int	
	config['max_iter'] = int(config['max_iter'])
	config['rs_cv'] = int(config['rs_cv'])
	
	if config['diff'] != 'OFF':
		config['diff'] = int(config['diff'])
	# Variable conversion	
	correct_layers = tuple([int(element) for element in (config['layers'])])
	config['layers'] = correct_layers
	if config['rs'] != None:
		config['rs'] = int(config['rs'])

	if config['demod_prefilter'] != 'OFF':
		if config['demod_prefilter'][0] == 'bandpass':
			config['demod_prefilter'] = [config['demod_prefilter'][0], [float(config['demod_prefilter'][1]), float(config['demod_prefilter'][2])], float(config['demod_prefilter'][3])]
		elif config['demod_prefilter'][0] == 'highpass':
			config['demod_prefilter'] = [config['demod_prefilter'][0], float(config['demod_prefilter'][1]), float(config['demod_prefilter'][2])]
		else:
			print('error prefilter')
			sys.exit()
	if config['demod_filter'] != 'OFF':
		config['demod_filter'] = [config['demod_filter'][0], float(config['demod_filter'][1]), float(config['demod_filter'][2])]
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config
	
	
# def read_parser(argv, Inputs, Inputs_opt, Defaults):
	# parser = argparse.ArgumentParser()
	# for element in (Inputs + Inputs_opt):
		# print(element)
		# if element == 'classifications' or element == 'layers' or element == 'demod_prefilter' or element == 'demod_filter':
			# parser.add_argument('--' + element, nargs='+')
		# else:
			# parser.add_argument('--' + element, nargs='?')
	
	# args = parser.parse_args()
	# config = {}
	# for element in Inputs:
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Required:', element)
			# sys.exit()

	# for element, value in zip(Inputs_opt, Defaults):
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Default ' + element + ' = ', value)
			# config[element] = value
	# #Type conversion to float

	
	
	
	






if __name__ == '__main__':
	main(sys.argv)


