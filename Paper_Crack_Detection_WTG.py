
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
from os.path import join, isdir, basename, dirname, isfile
from os import listdir
import pywt


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


InputsOpt_Defaults = {'name':'auto'}
# gearbox mio: thr_60, wt_0.001, hp_70k, stella_100, lcokout 200


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	if config['mode'] == 'model_train_test':
		
		print('Select data training')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()		
		Feature_1 = []
		Feature_2 = []
		for filepath in Filepaths:		
			mydata = pd.read_excel(filepath)
			mydata = mydata.to_dict(orient='list')
			Feature_1 += mydata['crest']
			Feature_2 += mydata['fmax']
		Feature_1 = np.array(Feature_1)
		Feature_2 = np.array(Feature_2)
		n_train = len(Feature_1)
		X_train = np.array([Feature_1, Feature_2, np.ones(len(Feature_1))])
		X_train = np.transpose(X_train)
		
		print('Select data test')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()		
		Feature_1 = []
		Feature_2 = []
		for filepath in Filepaths:		
			mydata = pd.read_excel(filepath)
			mydata = mydata.to_dict(orient='list')
			Feature_1 += mydata['crest']
			Feature_2 += mydata['fmax']
		Feature_1 = np.array(Feature_1)
		Feature_2 = np.array(Feature_2)
		n_test = len(Feature_1)
		X_test = np.array([Feature_1, Feature_2, np.ones(len(Feature_1))])
		X_test = np.transpose(X_test)
		

		
		
		# SVM-one-class		
		from sklearn.preprocessing import StandardScaler
		from sklearn.svm import OneClassSVM		
		
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)		
		
		Nus = [0.01, 0.1, 0.5]
		Tolerances = [1.e-2, 1.e-4, 1.e-6]
		
		clf = OneClassSVM(kernel='linear', nu=0.01, degree=3, verbose=True, coef0=0).fit(X_train)		
		labels_train = clf.predict(X_train)
		labels_test = clf.predict(X_test)
		# print(kmeans.labels_)
		
		# X = scaler.inverse_transform(X)
		
		plt.title('Train')
		for i in range(n_train):
			if labels_train[i] == 1:
				plt.scatter(X_train[i,0], X_train[i,1], color='red')
			else:
				plt.scatter(X_train[i,0], X_train[i,1], color='blue')
		plt.show()
		
		plt.title('Test')
		for i in range(n_test):
			if labels_test[i] == 1:
				plt.scatter(X_test[i,0], X_test[i,1], color='red')
			else:
				plt.scatter(X_test[i,0], X_test[i,1], color='blue')
		plt.show()
		
		
	

	elif config['mode'] == 'model_train_test':	
		
		print('caca')
		
		
		
	
	else:
		print('Must give mode')

		
	return

def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element':
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
	# config['fs'] = float(config['fs'])
	# # config['n_files'] = int(config['n_files'])
	# config['stella'] = int(config['stella'])
	# config['idx_fp1'] = int(config['idx_fp1'])
	# config['idx_fp2'] = int(config['idx_fp2'])
	
	# config['n_clusters'] = int(config['n_clusters'])
	
	# config['thr_value'] = float(config['thr_value'])
	# # config['highpass'] = float(config['highpass'])
	# config['window_time'] = float(config['window_time'])
	# # config['time_segments'] = float(config['time_segments'])
	# config['lockout'] = int(config['lockout'])
	# config['pdt'] = int(config['pdt'])
	# config['level'] = int(config['level'])
	
	# if config['range'] != None:
		# config['range'][0] = float(config['range'][0])
		# config['range'][1] = float(config['range'][1])
	
	# if config['db_out'] != 'OFF':
		# config['db_out'] = int(config['db_out'])
	
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


if __name__ == '__main__':
	main(sys.argv)
