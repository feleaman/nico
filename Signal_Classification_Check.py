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
import os.path
import sys
sys.path.insert(0, './lib') #to open user-defined functions

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


plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes



#+++++++++++++++++++++++++++PARSER++++++++++++++++++++++++++++++++++++++++++
parser = argparse.ArgumentParser()
parser.add_argument('--channel', nargs='?')
# channel = 'AE_Signal'
# channel = 'Koerperschall'
# channel = 'Drehmoment'
parser.add_argument('--power2', nargs='?')
parser.add_argument('--showplot', nargs='?')
parser.add_argument('--type', nargs='?')
args = parser.parse_args()

if args.channel != None:
	channel = args.channel
else:
	print('Required: Channel')
	sys.exit()

if args.power2 != None:
	n_points = 2**int(args.power2)

if args.showplot != None:
	showplot = args.showplot


#+++++++++++++++++++++++++++FUNCTIONS++++++++++++++++++++++++++++++++++++++++++
def save_pickle(pickle_name, pickle_data):
	pik = open(pickle_name, 'wb')
	pickle.dump(pickle_data, pik)
	pik.close()

def read_pickle(pickle_name):
	pik = open(pickle_name, 'rb')
	pickle_data = pickle.load(pik)
	return pickle_data


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
# mypath = 'C:/Felix/Data/CNs_Getriebe/Paper_Bursts/n1500_M80/'
# filename1 = join(mypath, 'V3_9_n1500_M80_AE_Signal_20160928_154159.mat')

mypath = 'C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault'
filename1 = join(mypath, 'V1_9_n1500_M80_AE_Signal_20160928_144737.mat')

pickle_classification = 'classification_20170830_141254_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl'


config_analysis = {'WindowTime':0.001, 'Overlap':False, 'WindowAdvance':0.4, 'savepik':True}

config_filter = {'analysis':False, 'type':'median', 'mode':'bandpass', 'params':[[70.0e3, 350.0e3], 3]}###

# config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [70.0e3, 170.0e3] , 3], 
'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warming':False,
'warming_points':20000}
#When hilbert is selected, the other parameters are ignored

config_diff = {'analysis':False, 'length':1, 'same':True}



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
point_index = filename1.find('.')
extension = filename1[point_index+1] + filename1[point_index+2] + filename1[point_index+3]

if extension == 'mat':
	x1 = f_open_mat(filename1, channel)
	x1 = np.ndarray.flatten(x1)

elif extension == 'tdm': #tdms
	x1 = f_open_tdms(filename1, channel)

filename1 = os.path.basename(filename1) #changes from path to file

#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++
if channel == 'Koerperschall':
	fs = 1000.0
elif channel == 'Drehmoment':
	fs = 1000.0
elif channel == 'AE_Signal':
	fs = 1000000.0
else:
	print('Error fs assignment')

if args.power2 == None:
	n_points = 2**(max_2power(len(x1)))
x1 = x1[0:n_points]
x1raw = x1

dt = 1.0/fs
n_points = len(x1)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])
traw = t


#++++++++++++++++++++++SIGNAL PROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Filter
if config_filter['analysis'] == True:
	print('+++Filter:')
	if config_filter['type'] == 'bandpass':
		print('Bandpass')
		f_nyq = 0.5*fs
		order = config_filter['params'][1]
		freqs_bandpass = [config_filter['params'][0][0]/f_nyq, config_filter['params'][0][1]/f_nyq]
		b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
		x1 = signal.filtfilt(b, a, x1)
	elif config_filter['type'] == 'median':
		print('Median')
		x1 = scipy.signal.medfilt(x1)

#Autocorrelation
# if config_autocorr['analysis'] == True:
	# print('+++Filter:')
	# if config_autocorr['type'] == 'definition':
		# x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
		# x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
	
	# elif config_autocorr['type'] == 'wiener':	
		# fftx1 = np.fft.fft(x1)
		# x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))
		
		# fftx2 = np.fft.fft(x2)
		# x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))

		
#Demodulation
if config_demod['analysis'] == True:
	print('+++Demodulation:')
	if config_demod['mode'] == 'hilbert':
		x1 = hilbert_demodulation(x1)
	elif config_demod['mode'] == 'butter':
		x1 = butter_demodulation(x=x1, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
		type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
	else:
		print('Error assignment demodulation')


#Differentiation
if config_diff['analysis'] == True:
	print('+++Differentiation:')
	if config_diff['same'] == True:
		x1 = diff_signal_eq(x=x1, length_diff=config_diff['length'])
	elif config_diff['same'] == False:
		x1 = diff_signal(x=x1, length_diff=config_diff['length'])
	else:
		print('Error assignment diff')	

if (config_demod['analysis'] == True or config_filter['analysis'] == True):
	if (config_demod['warming'] == True and config_demod['mode'] == 'butter'):
		print('Warm Warning')
		warm = config_demod['warming_points']
		x1 = x1[warm:]
		t = t[warm:]
		warm = float(warm)



#++++++++++++++++++++++ CHECK +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
info_classification = read_pickle(pickle_classification)
if info_classification['filename'] != filename1:
	print('Wrong filename!!!')
	sys.exit()
classification = info_classification['classification']
print(len(classification))
print(len([item for item in classification if item==1]))
print(len([item for item in classification if item==2]))
print(len([item for item in classification if item==0]))



windows = []
window_time = config_analysis['WindowTime']
window_points = int(window_time*fs)
window_advance = int(window_points*config_analysis['WindowAdvance'])
if config_analysis['Overlap'] == True:
	n_windows = int((n_points - window_points)/window_advance) + 1
else:
	n_windows = int(n_points/window_points)
print('Number of windows: ', n_windows)
print(type(info_classification['n_windows']))
print(type(n_windows))
print(info_classification['n_windows'])
print(n_windows)
if info_classification['n_windows'] != n_windows:
	print('Wrong n_windows!!!')
	sys.exit()

for count in range(n_windows):
	if config_analysis['Overlap'] == True:
		windows.append(x1[count*window_advance:window_points+window_advance*count])
	else:
		windows.append(x1[count*window_points:(count+1)*window_points])

	

# Check
window_check = 0
while window_check != -1:	
	count = window_check
	plt.plot(windows[window_check])
	print(classification[window_check])
	plt.show()
	window_check = input('Window to check: ')
	window_check = int(window_check)
