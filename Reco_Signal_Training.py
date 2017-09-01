# Reco_Signal_Training.py
# Last updated: 24.08.2017 by Felix Leaman
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

import datetime
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes



#+++++++++++++++++++++++++++PARSER++++++++++++++++++++++++++++++++++++++++++
parser = argparse.ArgumentParser()
parser.add_argument('--channel', nargs='?')
# channel = 'AE_Signal'
# channel = 'Koerperschall'
# channel = 'Drehmoment'
# parser.add_argument('--power2', nargs='?')
parser.add_argument('--showplot', nargs='?')
parser.add_argument('--type', nargs='?')
args = parser.parse_args()

if args.channel != None:
	channel = args.channel
else:
	print('Required: Channel')
	sys.exit()

# if args.power2 != None:
	# n_points = 2**int(args.power2)

if args.showplot != None:
	showplot = args.showplot


if channel == 'Koerperschall':
	fs = 1000.0
elif channel == 'Drehmoment':
	fs = 1000.0
elif channel == 'AE_Signal':
	fs = 1000000.0
else:
	print('Error fs assignment')
	
#+++++++++++++++++++++++++++FUNCTIONS++++++++++++++++++++++++++++++++++++++++++
def leftright_stats(window):
	pos_max = np.argmax(window)
	left_window = window[0:pos_max]
	right_window = window[pos_max:]

	if (len(left_window) != 0 and len(right_window) != 0):
		values = [np.max(window), 
		np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window)), stats.kurtosis(np.array(left_window), fisher=True), 
		np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	elif (len(left_window) == 0 and len(right_window) != 0):
		values = [np.max(window), 
		0., 0., 0., 0., 0., 
		np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	elif (len(left_window) != 0 and len(right_window) == 0):
		values = [np.max(window), 
		np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window)), stats.kurtosis(np.array(left_window), fisher=True), 
		0., 0., 0., 0., 0.]
	else:
		print('error lens windows left and right+++++++++++++++++++++')
	return values

# def leftright_stats_quantils(window):
	# pos_max = np.argmax(window)
	# left_window = window[0:pos_max]
	# right_window = window[pos_max:]

	# if (len(left_window) != 0 and len(right_window) != 0):
		# values = [np.max(window), 
		# np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window)), stats.kurtosis(np.array(left_window), fisher=True), 
		# np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	# elif (len(left_window) == 0 and len(right_window) != 0):
		# values = [np.max(window), 
		# 0., 0., 0., 0., 0., 
		# np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	# elif (len(left_window) != 0 and len(right_window) == 0):
		# values = [np.max(window), 
		# np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window)), stats.kurtosis(np.array(left_window), fisher=True), 
		# 0., 0., 0., 0., 0.]
	# else:
		# print('error lens windows left and right+++++++++++++++++++++')
	# return values


def n_per_intervals(data, interval, divisions):
	data = sorted(data)
	save = 0
	print(data)
	values = np.zeros(divisions)
	interval_length = (interval[1] - interval[0])/divisions
	for i in range(divisions):		
		cont = 0
		for k in range(len(data)-save):
			if (data[k+save] <= interval[0] + interval_length*(i+1)):
				cont = cont + 1
			else:
				break
		values[i] = cont
		if cont != 0:
			save = cont + save
	return values
	

def save_pickle(pickle_name, pickle_data):
	pik = open(pickle_name, 'wb')
	pickle.dump(pickle_data, pik)
	pik.close()

def read_pickle(pickle_name):
	pik = open(pickle_name, 'rb')
	pickle_data = pickle.load(pik)
	return pickle_data


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
hh = [-1., 0.21, 0.3 ,0.8, 0.4, -0.3, 0., 0.9, 0.6, 0.8, -0.3, -0.9, -0.96]
juju = n_per_intervals(hh, [-1., 1.], 10)
print(juju)
print(np.sum(juju))
print(len(hh))
sys.exit()








config_analysis = {'WindowTime':0.001, 'Overlap':False, 'WindowAdvance':0.4, 'savepik':True, 'power2':'various',
'channel':args.channel}

config_filter = {'analysis':False, 'type':'median', 'median_kernel':5,
'mode':'bandpass', 'params':[[70.0e3, 350.0e3], 3]}###

# config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [70.0e3, 170.0e3] , 3], 
'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warming':False,
'warming_points':20000}
#When hilbert is selected, the other parameters are ignored

config_diff = {'analysis':False, 'length':1, 'same':True}


config_NNmodel = {'normalization': 'per_signal', 
'solver':'lbfgs', 'alpha':1e-5, 'hidden_layer_sizes':(300, 30, 3),
'random_state':1, 'activation':'tanh', 'tol':1.e-8, 'max_iter':200000}

#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
n_files = input('Number of files to use for training: ')
n_files = int(n_files)
root = Tk()
root.withdraw()
root.update()

Filepaths = []
Filenames = []
# classification_pickles = []
Signals = []
Classifications_per_file = []
Windows_per_file = []

# info_classification = read_pickle(pickle_classification)
# if info_classification['filename'] != filename1:
	# print('Wrong filename!!!')
	# sys.exit()
# classification = info_classification['classification']

Flags_start_in = []
for i in range(n_files):
	print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
	print('Select file, then pickle classification!!! ')
	#Paths
	filepath = filedialog.askopenfilename()
	Filepaths.append(filepath)
	
	#Classifications
	classification_pickle = filedialog.askopenfilename()
	info_classification_pickle = read_pickle(classification_pickle)

	classification_in_file = info_classification_pickle['classification']
	Classifications_per_file.append(classification_in_file)
	checked_windows_in_file = len(classification_in_file)
	# print(classification_in_file)
	print('Windows checked in file: ', checked_windows_in_file)

	
	#N_Windows	
	windows_in_file = info_classification_pickle['n_windows']
	window_duration = info_classification_pickle['config_analysis']['WindowTime']
	points_in_file = int(windows_in_file*window_duration*fs)
	print('Power2 in file: ', np.log2(points_in_file))
	print('Total Windows total in file: ', windows_in_file)
	
	checked_points_in_file = int(checked_windows_in_file*window_duration*fs)
	
	#Signals
	x = f_open_mat(filepath, channel)
	# print(x)
	# x = np.ndarray.flatten(x)
	x = np.ravel(x)
	# print(x)
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
	
	
	if config_filter['analysis'] == True:
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
	
	
	
	
	
	# sys.exit()
	Signals.append(x)
	
	#Filenames
	filename = os.path.basename(filepath)
	Filenames.append(filename)
	
	if info_classification_pickle['filename'] != filename:
		print('Wrong filename!!!')
		sys.exit()
	
	
config_NNmodel['files'] = Filenames
config_NNmodel['windows_per_file'] = Windows_per_file

	
# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
# ax[0].plot(Signals[0])
# ax[1].plot(Signals[1])
# plt.show()
# sys.exit()

root.destroy()

# print(windows_per_file)
# x = f_open_mat(filename, channel)
# x = np.ndarray.flatten(x)



# filename = os.path.basename(filename) #changes from path to file



# mypath = 'C:/Felix/Data/CNs_Getriebe/Paper_Bursts/n1500_M80/train'
# filename1 = join(mypath, 'V1_9_n1500_M80_AE_Signal_20160928_144737.mat')
# pickle_classification = 'classification_20170825_131154_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl'







#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
# point_index = filename1.find('.')
# extension = filename1[point_index+1] + filename1[point_index+2] + filename1[point_index+3]

# if extension == 'mat':
	# x1 = f_open_mat(filename1, channel)
	# x1 = np.ndarray.flatten(x1)

# elif extension == 'tdm': #tdms
	# x1 = f_open_tdms(filename1, channel)

# filename1 = os.path.basename(filename1) #changes from path to file

#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++


# if args.power2 == None:
	# n_points = 2**(max_2power(len(x1)))
# x1 = x1[0:n_points]
# x1raw = x1

# dt = 1.0/fs
# n_points = len(x1)
# tr = n_points*dt
# t = np.array([i*dt for i in range(n_points)])
# traw = t


#++++++++++++++++++++++SIGNAL PROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #Filter
# if config_filter['analysis'] == True:
	# print('+++Filter:')
	# if config_filter['type'] == 'bandpass':
		# print('Bandpass')
		# f_nyq = 0.5*fs
		# order = config_filter['params'][1]
		# freqs_bandpass = [config_filter['params'][0][0]/f_nyq, config_filter['params'][0][1]/f_nyq]
		# b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
		# x1 = signal.filtfilt(b, a, x1)
	# elif config_filter['type'] == 'median':
		# print('Median')
		# x1 = scipy.signal.medfilt(x1)

# #Autocorrelation
# # if config_autocorr['analysis'] == True:
	# # print('+++Filter:')
	# # if config_autocorr['type'] == 'definition':
		# # x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
		# # x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
	
	# # elif config_autocorr['type'] == 'wiener':	
		# # fftx1 = np.fft.fft(x1)
		# # x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))
		
		# # fftx2 = np.fft.fft(x2)
		# # x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))

		
# #Demodulation
# if config_demod['analysis'] == True:
	# print('+++Demodulation:')
	# if config_demod['mode'] == 'hilbert':
		# x1 = hilbert_demodulation(x1)
	# elif config_demod['mode'] == 'butter':
		# x1 = butter_demodulation(x=x1, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
		# type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
	# else:
		# print('Error assignment demodulation')


# #Differentiation
# if config_diff['analysis'] == True:
	# print('+++Differentiation:')
	# if config_diff['same'] == True:
		# x1 = diff_signal_eq(x=x1, length_diff=config_diff['length'])
	# elif config_diff['same'] == False:
		# x1 = diff_signal(x=x1, length_diff=config_diff['length'])
	# else:
		# print('Error assignment diff')	

# if (config_demod['analysis'] == True or config_filter['analysis'] == True):
	# if (config_demod['warming'] == True and config_demod['mode'] == 'butter'):
		# print('Warm Warning')
		# warm = config_demod['warming_points']
		# x1 = x1[warm:]
		# t = t[warm:]
		# warm = float(warm)



#++++++++++++++++++++++ TRAINING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# info_classification = read_pickle(pickle_classification)
# if info_classification['filename'] != filename1:
	# print('Wrong filename!!!')
	# sys.exit()
# classification = info_classification['classification']




features = []
master_classification = []
window_time = config_analysis['WindowTime']
window_points = int(window_time*fs)
window_advance = int(window_points*config_analysis['WindowAdvance'])

for x, classification, n_windows in zip(Signals, Classifications_per_file, Windows_per_file):
	if config_NNmodel['normalization'] == 'per_signal':
		x = x / np.max(np.absolute(x))
		x = x.tolist()
		print('normalization per signal!!!!!!')
	
	for count in range(n_windows):
		if config_analysis['Overlap'] == True:
			current_window = x[count*window_advance:window_points+window_advance*count]	
			
			if config_NNmodel['normalization'] == 'per_window':
				print('normalization per window')
				current_window = current_window / np.max(np.absolute(current_window))	
				
			values = leftright_stats(current_window)
			# values = current_window
			features.append(values)
		else:
			current_window = x[count*window_points:(count+1)*window_points]
			
			
			if config_NNmodel['normalization'] == 'per_window':
				current_window = current_window / np.max(np.absolute(current_window))
				print('normalization per window')
			
			
			values = leftright_stats(current_window)
			# values = current_window
			
			features.append(values)

		master_classification.append(classification[count])
print('total number windows = ', len(features))
# check = 0
# while check != -1:
	# print('Classification: ', master_classification[check])
	# plt.plot(features[check])
	# plt.show()
	# check = input('Window to check: ')
	# check = int(check)
	
	
# Neuronal Network
clf = MLPClassifier(solver=config_NNmodel['solver'], alpha=config_NNmodel['alpha'],
hidden_layer_sizes=config_NNmodel['hidden_layer_sizes'], random_state=config_NNmodel['random_state'],
activation=config_NNmodel['activation'], tol=config_NNmodel['tol'], verbose=True,
max_iter=config_NNmodel['max_iter'])

# 
# # 

# print(type(features[0]))

# print(type(features[0][0]))

# print(type(master_classification[0]))



# print(len(features[0]))
# print(len(master_classification[0]))
# print(type(features))
# print(type(features[0]))
# print(features)

# caca = []
# for i in range(len(features)):
	# # caca[i] = [i, i+1, i+2]
	# # print(features[i])
	# # print(type(features[i]))
	# # print(type(features[i][i]))
	# # queso = [float(i), float(i+1), float(i+2)]
	# queso = features[i]
	# # print(queso)
	# caca.append(queso)

# features = caca


# sys.exit()
# print(features)
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# print(type(features))
# print(type(master_classification))
# print(type(features[0]))
# print(len(features))
# print(len(master_classification))

clf.fit(features, master_classification)

# Save pickle model
clf_pickle_info = [config_NNmodel, clf]

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_pickle('clf_' + stamp + '_' + '.pkl', clf_pickle_info)



sys.exit()




