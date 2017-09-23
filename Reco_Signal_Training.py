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
from sklearn.preprocessing import StandardScaler  
import datetime
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes



Inputs = ['channel', 'save', 'files', 'classifications', 'layers', 'features']
Inputs_opt = ['window_time', 'overlap', 'data_norm', 'solver', 'alpha', 'rs', 'activation', 'tol', 'max_iter', 'filter']
Defaults = [0.001, 0, 'per_signal', 'lbfgs', 1.e-3, 1, 'relu', 1.e-6, 400000, 'OFF']

def main(argv):
	config = read_parser(argv, Inputs, Inputs_opt, Defaults)
	
	print('Fs = 1 MHz for AE')
	fs = 1000000.0
	n_files = len(config['files'])
	
	Filepaths = []
	Filenames = []
	Signals = []
	Classifications_per_file = []
	Windows_per_file = []

	Flags_start_in = []	
	for i in range(n_files):
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
		print('Select file, then pickle classification!!! ')
		#Paths
		filepath = config['files'][i]
		Filepaths.append(filepath)
		
		#Classifications
		classification_pickle = config['classifications'][i]
		info_classification_pickle = read_pickle(classification_pickle)
		classification_in_file = info_classification_pickle['classification']
		Classifications_per_file.append(classification_in_file)
		checked_windows_in_file = len(classification_in_file)
		
		print('Windows checked in file: ', checked_windows_in_file)
		
		#N_Windows	
		windows_in_file = info_classification_pickle['n_windows']
		window_duration = info_classification_pickle['config_analysis']['WindowTime']
		points_in_file = int(windows_in_file*window_duration*fs)
		print('Power2 in file: ', np.log2(points_in_file))
		print('Total Windows total in file: ', windows_in_file)		
		checked_points_in_file = int(checked_windows_in_file*window_duration*fs)
		
		#Signals
		print(filepath)
		x = load_signal(filepath)
		x = x[0:checked_points_in_file]		
		
		if info_classification_pickle['config_analysis']['start_in'] != 0:
			start_in_time = info_classification_pickle['config_analysis']['start_in']
			print(start_in_time)
			print('with start in!!!!+++++++++++')
			# Flags_start_in.append(True)
			print(int(start_in_time*fs))
			print(len(x))
			print(x.shape)
			x = x[int(start_in_time*fs):]
			print(len(x))
			print(x.shape)
			Windows_per_file.append(checked_windows_in_file-1)
			# print(int(start_in_time*fs))
		else:
			print('Classification pickel not having start in')
			Windows_per_file.append(checked_windows_in_file)
			print(len(x))
			print(x.shape)
		
		
		if config['filter'] == 'ON':
			sys.exit()
			print('+++Filter:')
			if config_filter['type'] == 'bandpass':
				print('Bandpass')
				f_nyq = 0.5*fs
				order = config_filter['params'][1]
				freqs_bandpass = [config_filter['params'][0][0]/f_nyq, config_filter['params'][0][1]/f_nyq]
				b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
				x = signal.filtfilt(b, a, x)
			elif config_filter['type'] == 'median':
				print('Median')
				x = scipy.signal.medfilt(x, kernel_size=config_filter['median_kernel'])

		Signals.append(x)
		
		#Filenames
		filename = os.path.basename(filepath)
		Filenames.append(filename)
		print(filename)
		print(info_classification_pickle['filename'])
		if info_classification_pickle['filename'] != filename:
			print('Wrong filename!!!')
			sys.exit()
	print('fin')
	
	
	features = []
	master_classification = []
	window_time = config['window_time']
	window_points = int(window_time*fs)
	window_advance = int(window_points*config['overlap'])

	for x, classification, n_windows in zip(Signals, Classifications_per_file, Windows_per_file):
		if config['data_norm'] == 'per_signal':
			x = x / np.max(np.absolute(x))
			x = x.tolist()
			print('normalization per signal!!!!!!')
		
		for count in range(n_windows):
			if config['overlap'] != 0:
				sys.exit()
				current_window = x[count*window_advance:window_points+window_advance*count]	
				
				if config['data_norm'] == 'per_window':
					print('normalization per window')
					current_window = current_window / np.max(np.absolute(current_window))
				
				if config['features'] == 'interval10_stats_nomean':
					values = interval10_stats_nomean(current_window)
				elif config['features'] == 'interval10_stats':
					values = interval10_stats(current_window)
				elif config['features'] == 'interval5_stats_nomean':
					values = interval5_stats_nomean(current_window)
				else:
					print('error name features')
					sys.exit()
				# points_intervals = n_per_intervals_left_right(current_window, [-1., 1.], 5)
				
				# values = basic_stats_sides
				# values = np.array(values)
				features.append(values)
				
			else:
				print('without overlap!!!')
				current_window = x[count*window_points:(count+1)*window_points]
				
				if config['data_norm'] == 'per_window':
					sys.exit()
					current_window = current_window / np.max(np.absolute(current_window))
					print('normalization per window')
				
				if config['features'] == 'interval10_stats_nomean':
					values = interval10_stats_nomean(current_window)
				elif config['features'] == 'interval10_stats':
					values = interval10_stats(current_window)
				elif config['features'] == 'interval5_stats_nomean':
					values = interval5_stats_nomean(current_window)
				else:
					print('error name features')
					sys.exit()
				
				# values = basic_stats_sides		
				features.append(values)

			master_classification.append(classification[count])
	print('total number windows = ', len(features))
	

	
	clf = MLPClassifier(solver=config['solver'], alpha=config['alpha'],
	hidden_layer_sizes=config['layers'], random_state=config['rs'],
	activation=config['activation'], tol=config['tol'], verbose=True,
	max_iter=config['max_iter'])	

	
	#Scale
	scaler = StandardScaler()
	scaler.fit(features)
	features = scaler.transform(features)
	
	print(len(features))
	print(len(features[0]))
	print(len(master_classification))
	clf.fit(features, master_classification)

	# Save pickle model
	clf_pickle_info = [config, clf, scaler]
	stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	save_pickle('clf_' + stamp + '_' + '.pkl', clf_pickle_info)
	return

	
	
	
def read_parser(argv, Inputs, Inputs_opt, Defaults):
	parser = argparse.ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'files' or element == 'classifications' or element == 'layers':
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
	config_input['overlap'] = float(config_input['overlap'])
	config_input['window_time'] = float(config_input['window_time'])
	config_input['alpha'] = float(config_input['alpha'])
	config_input['tol'] = float(config_input['tol'])	
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


