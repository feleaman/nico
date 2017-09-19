# Burst_Detection.py
# Last updated: 14.09.2017 by Felix Leaman
# Description:
# Code for opening 2 x a .mat or .tdms data files with single channel and detecting Bursts
# The file and channel is selected by the user
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively
# Power2 option let the user to analyze only 2^Power2 points of each file
# Only and only one detection method must be selected

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from tkinter import filedialog
from tkinter import Tk
from skimage import img_as_uint
import skimage.filters
import os.path
import sys
sys.path.insert(0, './lib') #to open user-defined functions
import argparse
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
parser = argparse.ArgumentParser()
parser.add_argument('--channel', nargs='?')
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

#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++

root = Tk()
root.withdraw()
root.update()
filename1 = filedialog.askopenfilename()
filename2 = filedialog.askopenfilename()
root.destroy()

point_index = filename1.find('.')
extension = filename1[point_index+1] + filename1[point_index+2] + filename1[point_index+3]

if extension == 'mat':
	x1 = f_open_mat(filename1, channel)
	x1 = np.ndarray.flatten(x1)
	x2 = f_open_mat(filename2, channel)
	x2 = np.ndarray.flatten(x2)

elif extension == 'tdm': #tdms
	x1 = f_open_tdms(filename1, channel)
	x2 = f_open_tdms(filename2, channel)


filename1 = os.path.basename(filename1) #changes from path to file
filename2 = os.path.basename(filename2) #changes from path to file

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
x2 = x2[0:n_points]	
x1raw = x1
x2raw = x2


dt = 1.0/fs
n_points = len(x1)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])
traw = t

#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++
#Methods
Config_Methods = {'Threshold_WFM':False, 'NeuronalNetwork':True}
config_threshold_wfm = {'mode':'fixed_value', 'value':0.3, 'min_t_burst':0.001}

if Config_Methods['NeuronalNetwork'] == True:
	print('Select NN Model:')
	root = Tk()
	root.withdraw()
	root.update()
	path_info_model = filedialog.askopenfilename()
	info_model = read_pickle(path_info_model)
	print('Info Model: ')
	print(info_model)
	root.destroy()
	clf = info_model[1]
	config_model = info_model[0]
	plt.show()
else:
	clf = None
config_neuronal = {'Model':clf, 'WindowTime':0.001, 'RateOverlap':0, 'normalization':'per_signal', 'feat_normalization':'standard'}
if config_neuronal['feat_normalization'] == 'standard':
	print('Select Scale:')
	root = Tk()
	root.withdraw()
	root.update()
	path_info_scale = filedialog.askopenfilename()
	info_scale = read_pickle(path_info_scale)
	print('Info scale: ')
	print(info_scale)
	root.destroy()
	scaler = info_scale[1]
	config_scale = info_scale[0]
	plt.show()
else:
	scaler = 0


#Pre-processing
config_filter = {'analysis':False, 'type':'median', 'mode':'bandpass', 'params':[[70.0e3, 350.0e3], 3]}

config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_diff = {'analysis':False, 'length':1, 'same':True}

config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [70.0e3, 170.0e3] , 3], 
'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warm_points':20000}
#When hilbert is selected, the other parameters are ignored


#++++++++++++++++++++++CHECKS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if Config_Methods['NeuronalNetwork'] == True:

	if config_model['normalization'] != config_neuronal['normalization']:
		print('error normalization model NN')
		sys.exit()


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
		x2 = signal.filtfilt(b, a, x2)
	elif config_filter['type'] == 'median':
		print('Median')
		x1 = scipy.signal.medfilt(x1)
		x2 = scipy.signal.medfilt(x2)

#Autocorrelation
if config_autocorr['analysis'] == True:
	print('+++Filter:')
	if config_autocorr['type'] == 'definition':
		x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
		x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
	
	elif config_autocorr['type'] == 'wiener':	
		fftx1 = np.fft.fft(x1)
		x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))		
		fftx2 = np.fft.fft(x2)
		x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))
	else:
		print('Error assignment autocorrelation')

#Demodulation
if config_demod['analysis'] == True:
	print('+++Demodulation:')
	if config_demod['mode'] == 'hilbert':
		x1 = hilbert_demodulation(x1)
		x2 = hilbert_demodulation(x2)
	elif config_demod['mode'] == 'butter':
		x1 = butter_demodulation(x=x1, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
		type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
		x2 = butter_demodulation(x=x2, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
		type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
	else:
		print('Error assignment demodulation')


#Differentiation
if config_diff['analysis'] == True:
	print('+++Differentiation:')
	if config_diff['same'] == True:
		x1 = diff_signal_eq(x=x1, length_diff=config_diff['length'])
		x2 = diff_signal_eq(x=x2, length_diff=config_diff['length'])
	elif config_diff['same'] == False:
		x1 = diff_signal(x=x1, length_diff=config_diff['length'])
		x2 = diff_signal(x=x2, length_diff=config_diff['length'])
	else:
		print('Error assignment diff')	

warm = 0.0
if (config_demod['warm_points'] != 0 and config_demod['mode'] == 'butter' and config_demod['analysis'] == True):
	x1 = x1[config_demod['warm_points']:]
	x2 = x2[config_demod['warm_points']:]
	t = t[config_demod['warm_points']:]
	warm = float(config_demod['warm_points'])



#++++++++++++++++++++++ BURST DETECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
perro = 0
for element in Config_Methods:
	if Config_Methods[element] == True:
		perro = perro + 1
		Name = element
		Method = element
if perro != 1:
	print('One method and only one method must be selected')
	sys.exit()

if Config_Methods[Method] == True:
	if Method == 'Threshold_WFM':	
		if config_threshold_wfm['mode'] == 'factor_rms':
			threshold1 = config_threshold_wfm['value']*signal_rms(x1)
			threshold2 = config_threshold_wfm['value']*signal_rms(x2)
		elif config_threshold_wfm['mode'] == 'fixed_value':
			threshold1 = config_threshold_wfm['value']
			threshold2 = config_threshold_wfm['value']
		
		n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=fs, threshold=threshold1, t_window=config_threshold_wfm['min_t_burst'])
		
		
		t_burst_corr1 = t_burst_corr1 + np.ones(len(t_burst_corr1))*warm*dt

		
		n_burst_corr2, t_burst_corr2, amp_burst_corr2, t_burst2, amp_burst2 = id_burst_threshold(x=x2, fs=fs, threshold=threshold2, t_window=config_threshold_wfm['min_t_burst'])
		
		t_burst_corr2 = t_burst_corr2 + np.ones(len(t_burst_corr2))*warm*dt

	
	elif Method == 'NeuronalNetwork':
		print('Not yet')
		
		Windows1 = []
		Windows2 = []
		window_time = config_neuronal['WindowTime']
		window_points = int(window_time*fs)
		window_advance = int(window_points*config_neuronal['RateOverlap'])
		if config_neuronal['RateOverlap'] != 0:
			print('Windows with overlap')
			n_windows = int((n_points - window_points)/window_advance) + 1
		else:
			n_windows = int(n_points/window_points)
		print('Number of windows: ', n_windows)
		
		if config_neuronal['normalization'] == 'per_signal':
			x1 = x1 / np.max(np.absolute(x1))
			x2 = x2 / np.max(np.absolute(x2))
			print('Normalization per signal')		
		
		for count in range(n_windows):
			if config_neuronal['RateOverlap'] != 0:
				Windows1.append(x1[count*window_advance:window_points+window_advance*count])
				Windows2.append(x2[count*window_advance:window_points+window_advance*count])
			else:
				Windows1.append(x1[count*window_points:(count+1)*window_points])
				Windows2.append(x2[count*window_points:(count+1)*window_points])
		
		Predictions1 = []
		Predictions2 = []
		features_fault = []
		features_ok = []
		numero = 0
		for window1, window2 in zip(Windows1, Windows2):
			if config_neuronal['normalization'] == 'per_window':
				print('Normalization per window')
				window1 = window1 / np.max(np.absolute(window1))
				window2 = window2 / np.max(np.absolute(window2))
			
			basic_stats_sides = interval10_stats_nomean(window1)
			# points_intervals = n_per_intervals_left_right(window1, [-1., 1.], 5)
			values = basic_stats_sides
			
			values = scaler.transform(values)
			
			features_fault.append(values)
			prediction = clf.predict(values)
			
			# if numero == 3:
				# print(values)
				# print(prediction)
				# plt.plot(window1)
				# plt.show()
				# sys.exit()
			
			
			Predictions1.append(prediction[0])
		
			basic_stats_sides = interval10_stats_nomean(window2)
			# points_intervals = n_per_intervals_left_right(window2, [-1., 1.], 5)
			
			values = basic_stats_sides
			values = scaler.transform(values)
			
			features_ok.append(values)
			prediction = clf.predict(values)
			Predictions2.append(prediction[0])
			numero = numero + 1
			
		t_burst_corr1 = []
		amp_burst_corr1 = []
		for i in range(len(Predictions1)):
			if Predictions1[i] != 0:
				t_burst_corr1.append(i*window_time)
				amp_burst_corr1.append(x1raw[int(i*window_time*fs)])
			
		t_burst_corr2 = []
		amp_burst_corr2 = []
		for i in range(len(Predictions2)):
			if Predictions2[i] != 0:
				t_burst_corr2.append(i*window_time)
				amp_burst_corr2.append(x2raw[int(i*window_time*fs)])
		

#++++++++++++++++++++++ PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_filter['analysis'] == True:
	Name = Name + ' FIL'
if config_autocorr['analysis'] == True:
	Name = Name + ' ACR'
if config_demod['analysis'] == True:
	Name = Name + ' ENV'		
if config_diff['analysis'] == True:
	Name = Name + ' DIF'

fig = [[], []]	
fig[0], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
if Config_Methods['Threshold_WFM'] == True:
	ax[0].axhline(threshold1, color='k')
	ax[0].plot(t_burst_corr1, amp_burst_corr1, 'ro')
elif Config_Methods['NeuronalNetwork'] == True:
	for i in range(len(t_burst_corr1)):
		ax[0].axvspan(xmin=t_burst_corr1[i], xmax=t_burst_corr1[i]+config_neuronal['WindowTime'], facecolor='y', alpha=0.5)
ax[0].plot(t, x1, color='darkblue')		
ax[0].set_title(channel + ' ' + Name + '\n' + filename1, fontsize=10)
ax[0].set_ylabel('Amplitude')
	
if Config_Methods['Threshold_WFM'] == True:
	ax[1].axhline(threshold2, color='k')
	ax[1].plot(t_burst_corr2, amp_burst_corr2, 'ro')
elif Config_Methods['NeuronalNetwork'] == True:
	for i in range(len(t_burst_corr2)):
		ax[1].axvspan(xmin=t_burst_corr2[i], xmax=t_burst_corr2[i]+config_neuronal['WindowTime'], facecolor='y', alpha=0.5)
ax[1].plot(t, x2, color='darkblue')
ax[1].set_title(filename2, fontsize=10)
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Time s')


#++++++++++++++++++++++ BURSTS BACK IN RAW SIGNALS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig[1], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
amp_burst_corr1 = np.array([x1raw[int(time*fs)] for time in t_burst_corr1])
amp_burst_corr2 = np.array([x2raw[int(time*fs)] for time in t_burst_corr2])
ax[0].plot(traw, x1raw)
ax[0].plot(t_burst_corr1, amp_burst_corr1, 'ro')
ax[0].set_title(channel + ' ' + 'Raw WFM' + '\n' + filename1, fontsize=10)
ax[0].set_ylabel('Amplitude')
# ax[0].axvspan(xmin=0.078, xmax=0.079, facecolor='y', alpha=0.5)

ax[1].plot(traw, x2raw)
ax[1].plot(t_burst_corr2, amp_burst_corr2, 'ro')
ax[1].set_title(filename2, fontsize=10)
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Time s')

print('Detected Burst')
print(len(t_burst_corr1))
print(len(t_burst_corr2))


plt.show()

np.savetxt('features_fault.txt', features_fault)
np.savetxt('features_ok.txt', features_ok)
