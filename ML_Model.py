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
from m_denois import *
from m_det_features import *
from m_processing import *
from os.path import isfile, join
import pickle
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler  
import datetime
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


from sklearn.model_selection import train_test_split

Inputs = ['mode', 'save', 'file']
Inputs_opt = ['solver', 'alpha', 'rs', 'activation', 'tol', 'max_iter', 'NN_name', 'learning_rate_init', 'layers', 'output', 'file_test']
Defaults = ['lbfgs', 1.e-1, 1, 'tanh', 1.e-6, 400000, 'auto', 0.001, [4], '2n', None]


def main(argv):
	config = read_parser(argv, Inputs, Inputs_opt, Defaults)
	
	if config['mode'] == 'mode1':

		print(config['file'])
		mydict = read_pickle(config['file'])
		
		features = mydict['features']
		classification = mydict['classification']
		features_select = mydict['features_select']
		
		n_pos = 0
		for i in range(len(classification)):
			if classification[i] == 1:
				n_pos += 1
		
		# classification = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		# features = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]]	
		# classification = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
		
			
		
		# print(features_train)
		# print(features_test)
		# print(classes_train)
		# print(classes_test)

		scaler = StandardScaler()
		scaler.fit(features)
		features = scaler.transform(features)	
		
		if config['output'] == '2n':
			new_classification = []
			for k in range(len(classification)):
				if classification[k] == 1:
					new_classification.append([1, 0])
				elif classification[k] == 0:
					new_classification.append([0, 1])
				else:
					print('error 8889')
			classification = new_classification
		else:
			print('only one neuron out++++++++++++++++++++++++')
		
		
		# print(features)
		
		
		
		features_train, features_test, classes_train, classes_test = train_test_split(features, classification, test_size=0.4, random_state=50, stratify=classification)
		
		
		clf = MLPClassifier(solver=config['solver'], alpha=config['alpha'],
		hidden_layer_sizes=config['layers'], random_state=config['rs'],
		activation=config['activation'], tol=config['tol'], verbose=False,
		max_iter=config['max_iter'])
		
		# clf = SVC(tol=1.e-6)
		
		# clf = DecisionTreeClassifier()	
		
		
		clf.fit(features_train, classes_train)
		
		
		
		# clf = KMeans()
		# clf.fit(features_train)

		clf_pickle_info = {}
		clf_pickle_info['config'] = config
		clf_pickle_info['clf'] = clf
		clf_pickle_info['scaler'] = scaler	
		
		if config['save'] == 'ON':
			stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")	
			save_pickle('clf_' + stamp + '.pkl', clf_pickle_info)
			print('++++model saved')
			
		
		classes_prediction = clf.predict(features_test)
		
		TP = 0
		FP = 0
		FN = 0
		TN = 0
		if config['output'] == '2n':
			for i in range(len(classes_prediction)):
				if classes_test[i][0] == 1:
					if classes_prediction[i][0] == 1:
						TP = TP + 1
					else:
						FN = FN + 1
				elif classes_test[i][0] == 0:
					if classes_prediction[i][0] == 0:
						TN = TN + 1
					else:
						FP = FP + 1
		else:
			for i in range(len(classes_prediction)):
				if classes_test[i] == 1:
					if classes_prediction[i] == 1:
						TP = TP + 1
					else:
						FN = FN + 1
				elif classes_test[i] == 0:
					if classes_prediction[i] == 0:
						TN = TN + 1
					else:
						FP = FP + 1
		print('Recall: ', 100 * TP / (TP+FN))
		print('Precision: ', 100 * TP / (TP+FP))
		# print('FPR: ', 100 * FP / (FP+TN))
		print('Accuracy: ', 100 * (TP + TN) / (TP + TN + FP + FN))
		print('TNR: ', 100 * TN / (TN+FP))
		
		print('+++Examples in train: ', len(classes_train))
		print('+++Examples in test: ', len(classes_test))
		print('Positives in test: ', TP + FN)
		print('Negatives in test: ', TN + FP)
		print('Positives total: ', n_pos)
		
		
		# print(features)
		
		
		
		
		sys.exit()
		
	elif config['mode'] == 'mode2':
	
		print(config['file'])
		print('one class!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		mydict = read_pickle(config['file'])
		
		classification = mydict['classification']
		# features = mydict['features_select']
		features = mydict['features']


		


		scaler = StandardScaler()
		scaler.fit(features)
		features = scaler.transform(features)	
		
		# if config['output'] == '2n':
			# new_classification = []
			# for k in range(len(classification)):
				# if classification[k] == 1:
					# new_classification.append([1, 0])
				# elif classification[k] == 0:
					# new_classification.append([0, 1])
				# else:
					# print('error 8889')
			# classification = new_classification
		# else:
			# print('only one neuron out++++++++++++++++++++++++')
		
		
		# print(features)
		
		
		
		features_train, features_test, classes_train, classes_test = train_test_split(features, classification, test_size=0.15, random_state=30)
		print(len(features_train))
		print(len(features_train[0]))
		
		# clf = MLPClassifier(solver=config['solver'], alpha=config['alpha'],
		# hidden_layer_sizes=config['layers'], random_state=config['rs'],
		# activation=config['activation'], tol=config['tol'], verbose=False,
		# max_iter=config['max_iter'])
		
		# clf = SVC(tol=1.e-6)
		
		# clf = DecisionTreeClassifier()	
		
		nu = 0.1
		
		clf = OneClassSVM(nu=nu, verbose=True, kernel='sigmoid', coef0=1., tol=1.e-6, degree=3)
		
		clf.fit(features_train)
		# clf.fit(features)
		
		
		
		# clf = KMeans()
		# clf.fit(features_train)

		clf_pickle_info = {}
		clf_pickle_info['config'] = config
		clf_pickle_info['clf'] = clf
		clf_pickle_info['scaler'] = scaler	
		
		if config['save'] == 'ON':
			stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")	
			save_pickle('clf_' + stamp + '.pkl', clf_pickle_info)
			print('++++model saved')
			
		
		classes_prediction = clf.predict(features_test)
		
		# print(classes_prediction)
		
		count = 0
		for element in classes_prediction:
			if element == -1:
				count += 1
		print('errores in train')
		print(count)
		
		
		
		clf = OneClassSVM(nu=nu, verbose=True, kernel='sigmoid', coef0=1., tol=1.e-6)		
		clf.fit(features)
		
		
		
		if config['file_test'] == None:
			print('Select Files to test: ')
			root = Tk()
			root.withdraw()
			root.update()
			Filepath_test = filedialog.askopenfilenames()			
			root.destroy()
		else:
			filepath_test = config['file_test']
		
		for filepath_test in Filepath_test:	
			print(os.path.basename(filepath_test))
			mydict2 = read_pickle(filepath_test)		
			classification2 = mydict2['classification']
			# features2 = mydict2['features_select']
			features2 = mydict2['features']
			features2 = scaler.transform(features2)

			classes_prediction2 = clf.predict(features2)		
			
			count = 0
			for element in classes_prediction2:
				if element == -1:
					count += 1
			print('errores in test')
			print(count)
		
		
		
		
		
		sys.exit()
		
		TP = 0
		FP = 0
		FN = 0
		TN = 0
		if config['output'] == '2n':
			for i in range(len(classes_prediction)):
				if classes_test[i][0] == 1:
					if classes_prediction[i][0] == 1:
						TP = TP + 1
					else:
						FN = FN + 1
				elif classes_test[i][0] == 0:
					if classes_prediction[i][0] == 0:
						TN = TN + 1
					else:
						FP = FP + 1
		else:
			for i in range(len(classes_prediction)):
				if classes_test[i] == 1:
					if classes_prediction[i] == 1:
						TP = TP + 1
					else:
						FN = FN + 1
				elif classes_test[i] == 0:
					if classes_prediction[i] == 0:
						TN = TN + 1
					else:
						FP = FP + 1
		print('Recall: ', 100 * TP / (TP+FN))
		print('Precision: ', 100 * TP / (TP+FP))
		# print('FPR: ', 100 * FP / (FP+TN))
		print('Accuracy: ', 100 * (TP + TN) / (TP + TN + FP + FN))
		print('TNR: ', 100 * TN / (TN+FP))
		
		print('+++Examples in train: ', len(classes_train))
		print('+++Examples in test: ', len(classes_test))
		print('Positives in test: ', TP + FN)
		print('Negatives in test: ', TN + FP)
		print('Positives total: ', n_pos)
		
		
		# print(features)
		
		
		
		
		sys.exit()
	
	
	elif config['mode'] == 'mode_burst':
		
		Features = []
		Classifications = []
		
		for signal, classification in zip()

		
		print(config['file'])
		mydict = read_pickle(config['file'])
		
		classification = mydict['classification']
		# features = mydict['features_select']
		features = mydict['features']


		


		scaler = StandardScaler()
		scaler.fit(features)
		features = scaler.transform(features)	
		
		# if config['output'] == '2n':
			# new_classification = []
			# for k in range(len(classification)):
				# if classification[k] == 1:
					# new_classification.append([1, 0])
				# elif classification[k] == 0:
					# new_classification.append([0, 1])
				# else:
					# print('error 8889')
			# classification = new_classification
		# else:
			# print('only one neuron out++++++++++++++++++++++++')
		
		
		# print(features)
		
		
		
		features_train, features_test, classes_train, classes_test = train_test_split(features, classification, test_size=0.15, random_state=30)
		print(len(features_train))
		print(len(features_train[0]))
		
		# clf = MLPClassifier(solver=config['solver'], alpha=config['alpha'],
		# hidden_layer_sizes=config['layers'], random_state=config['rs'],
		# activation=config['activation'], tol=config['tol'], verbose=False,
		# max_iter=config['max_iter'])
		
		# clf = SVC(tol=1.e-6)
		
		# clf = DecisionTreeClassifier()	
		
		nu = 0.1
		
		clf = OneClassSVM(nu=nu, verbose=True, kernel='sigmoid', coef0=1., tol=1.e-6, degree=3)
		
		clf.fit(features_train)
		# clf.fit(features)
		
		
		
		# clf = KMeans()
		# clf.fit(features_train)

		clf_pickle_info = {}
		clf_pickle_info['config'] = config
		clf_pickle_info['clf'] = clf
		clf_pickle_info['scaler'] = scaler	
		
		if config['save'] == 'ON':
			stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")	
			save_pickle('clf_' + stamp + '.pkl', clf_pickle_info)
			print('++++model saved')
			
		
		classes_prediction = clf.predict(features_test)
		
		# print(classes_prediction)
		
		count = 0
		for element in classes_prediction:
			if element == -1:
				count += 1
		print('errores in train')
		print(count)
		
		
		
		clf = OneClassSVM(nu=nu, verbose=True, kernel='sigmoid', coef0=1., tol=1.e-6)		
		clf.fit(features)
		
		
		
		if config['file_test'] == None:
			print('Select Files to test: ')
			root = Tk()
			root.withdraw()
			root.update()
			Filepath_test = filedialog.askopenfilenames()			
			root.destroy()
		else:
			filepath_test = config['file_test']
		
		for filepath_test in Filepath_test:	
			print(os.path.basename(filepath_test))
			mydict2 = read_pickle(filepath_test)		
			classification2 = mydict2['classification']
			# features2 = mydict2['features_select']
			features2 = mydict2['features']
			features2 = scaler.transform(features2)

			classes_prediction2 = clf.predict(features2)		
			
			count = 0
			for element in classes_prediction2:
				if element == -1:
					count += 1
			print('errores in test')
			print(count)
		
		
		
		
		
		sys.exit()
		
		TP = 0
		FP = 0
		FN = 0
		TN = 0
		if config['output'] == '2n':
			for i in range(len(classes_prediction)):
				if classes_test[i][0] == 1:
					if classes_prediction[i][0] == 1:
						TP = TP + 1
					else:
						FN = FN + 1
				elif classes_test[i][0] == 0:
					if classes_prediction[i][0] == 0:
						TN = TN + 1
					else:
						FP = FP + 1
		else:
			for i in range(len(classes_prediction)):
				if classes_test[i] == 1:
					if classes_prediction[i] == 1:
						TP = TP + 1
					else:
						FN = FN + 1
				elif classes_test[i] == 0:
					if classes_prediction[i] == 0:
						TN = TN + 1
					else:
						FP = FP + 1
		print('Recall: ', 100 * TP / (TP+FN))
		print('Precision: ', 100 * TP / (TP+FP))
		# print('FPR: ', 100 * FP / (FP+TN))
		print('Accuracy: ', 100 * (TP + TN) / (TP + TN + FP + FN))
		print('TNR: ', 100 * TN / (TN+FP))
		
		print('+++Examples in train: ', len(classes_train))
		print('+++Examples in test: ', len(classes_test))
		print('Positives in test: ', TP + FN)
		print('Negatives in test: ', TN + FP)
		print('Positives total: ', n_pos)
		
		
		# print(features)
		
		
		
		
		sys.exit()
	
	
	return

	
	
	
def read_parser(argv, Inputs, Inputs_opt, Defaults):
	parser = argparse.ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'classifications' or element == 'layers' or element == 'demod_prefilter' or element == 'demod_filter':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config_input = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config_input[element] = value
	#Type conversion to float

	config_input['alpha'] = float(config_input['alpha'])
	config_input['tol'] = float(config_input['tol'])	
	config_input['learning_rate_init'] = float(config_input['learning_rate_init'])	
	#Type conversion to int	
	config_input['max_iter'] = int(config_input['max_iter'])

	# Variable conversion	
	correct_layers = tuple([int(element) for element in (config_input['layers'])])
	config_input['layers'] = correct_layers
	if config_input['rs'] != None:
		config_input['rs'] = int(config_input['rs'])	
	
	
	
	return config_input






if __name__ == '__main__':
	main(sys.argv)


