# Reco_Signal_Test.py
# Last updated: 30.08.2017 by Felix Leaman
# Description:
# 

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
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
from scipy.signal import medfilt
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from os.path import isfile, join
import pickle
import argparse

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from sklearn.neural_network import MLPClassifier

from tkinter import filedialog
from tkinter import Tk

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



root = Tk()
root.withdraw()
root.update()


path_info_model = filedialog.askopenfilename()
info_model = read_pickle(path_info_model)
print('Info Model: ')
print(info_model)
bool = input('Continue? Y/N: ')
if bool == 'N':
	sys.exit()
clf = info_model[1]
config_model = info_model[0]

filepath = filedialog.askopenfilename()

x = f_open_mat(filepath, channel)
x = np.ndarray.flatten(x)


# info_classification = read_pickle(pickle_name_classification)

# if info_classification['filename'] != filename1:
	# print('Wrong filename!!!')
	# sys.exit()
# classification = info_classification['classification']



#Filenames
filename = os.path.basename(filepath)

# if info_classification_pickle['filename'] != filename:
	# print('Wrong filename!!!')
	# sys.exit()


root.destroy()





# mypath = 'C:/Felix/Data/CNs_Getriebe/Paper_Bursts/n1500_M80/'
# filename1 = join(mypath, 'V3_9_n1500_M80_AE_Signal_20160928_154159.mat')
# filename1 = join(mypath, 'V3_9_n1500_M80_AE_Signal_20160506_152625.mat')

config_analysis = {'WindowTime':0.001, 'Overlap':False, 'WindowAdvance':0.4, 'savepik':True, 'power2':args.power2,
'channel':args.channel}

config_filter = {'analysis':False, 'type':'median', 'median_kernel':5,
'mode':'bandpass', 'params':[[70.0e3, 350.0e3], 3]}###

# config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [70.0e3, 170.0e3] , 3], 
'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warming':False,
'warming_points':20000}
#When hilbert is selected, the other parameters are ignored

config_diff = {'analysis':False, 'length':1, 'same':True}


# pickle_name_model = 'clf_20170825_132430_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl'
# pickle_name_classification = 'classification_20170825_130754_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl'
# pickle_name_classification = 'classification_20170825_131643_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl'

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
x = x[0:n_points]
# x1raw = x1


dt = 1.0/fs
n_points = len(x)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])
# traw = t

xraw = x

if config_model['normalization'] == 'per_signal':
	x = x / np.max(np.absolute(x))
	print('normalization per signal')



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
		x = signal.filtfilt(b, a, x)
	elif config_filter['type'] == 'median':
		print('Median')
		x = scipy.signal.medfilt(x, kernel_size=config_filter['median_kernel'])

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

if (config_demod['analysis'] == True or (config_filter['analysis'] == True and config_filter['type'] == 'bandpass')):
	if (config_demod['offwarming'] == True and config_demod['mode'] == 'butter'):
		print('Warm Warning')
		warm = config_demod['warming_points']
		x1 = x1[warm:]
		t = t[warm:]
		warm = float(warm)


#++++++++++++++++++++++ TEST +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# info_model = read_pickle(pickle_name_model)
# clf = info_model[1]
# info_classification = read_pickle(pickle_name_classification)

# if info_classification['filename'] != filename1:
	# print('Wrong filename!!!')
	# sys.exit()
# classification = info_classification['classification']


windows = []
window_time = config_analysis['WindowTime']
window_points = int(window_time*fs)
window_advance = int(window_points*config_analysis['WindowAdvance'])
if config_analysis['Overlap'] == True:
	n_windows = int((n_points - window_points)/window_advance) + 1
else:
	n_windows = int(n_points/window_points)
print('Number of windows: ', n_windows)
# if info_classification['n_windows'] != n_windows:
	# print('Wrong n_windows!!!')
	# sys.exit()



for count in range(n_windows):
	if config_analysis['Overlap'] == True:
		windows.append(x[count*window_advance:window_points+window_advance*count])
	else:
		windows.append(x[count*window_points:(count+1)*window_points])

predictions = []
for window in windows:
	if config_model['normalization'] == 'per_window':
		print('normalization per window')
		window = window / np.max(np.absolute(window))
	
	
	# current_window = window
	# pos_max = np.argmax(current_window)
	# left_window = current_window[0:pos_max]
	# right_window = current_window[pos_max:]
	# if (len(left_window) != 0 and len(right_window) != 0):
		# current_window = [np.max(current_window), 
		# np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window)), stats.kurtosis(np.array(left_window), fisher=True), 
		# np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	# elif (len(left_window) == 0 and len(right_window) != 0):
		# current_window = [np.max(current_window), 
		# 0., 0., 0., 0., 0., 
		# np.min(right_window), np.mean(right_window), np.std(right_window), stats.skew(np.array(right_window)), stats.kurtosis(np.array(right_window), fisher=True)]
	# elif (len(left_window) != 0 and len(right_window) == 0):
		# current_window = [np.max(current_window), 
		# np.min(left_window), np.mean(left_window), np.std(left_window), stats.skew(np.array(left_window)), stats.kurtosis(np.array(left_window), fisher=True), 
		# 0., 0., 0., 0., 0.]
	# else:
		# print('error lens windows left and right+++++++++++++++++++++')
	
	# window = current_window
	# values = window
	
	basic_stats_sides = leftright_stats(window)			
	points_intervals = n_per_intervals(window, [-1., 1.], 10)			
	values = basic_stats_sides + points_intervals			
	# features.append(values)
	
	
	
	prediction = clf.predict(values)
	# print(prediction)
	# print(type(prediction))
	# print(type(prediction[0]))
	predictions.append(prediction[0])
t_burst = []
amp_burst = []
for i in range(len(predictions)):
	if predictions[i] != 0:
		t_burst.append(i*window_time)
		amp_burst.append(xraw[int(i*window_time*fs)])


fig, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(t, xraw)
ax.plot(t_burst, amp_burst, 'ro')
# ax[0][0].set_title(channel + ' ' + 'Raw WFM' + '\n' + file_ok_train_1, fontsize=10)
ax.set_title(filename, fontsize=10)

ax.set_ylabel('Amplitude')
ax.set_xlabel('Time s')
plt.show()

sys.exit()


# fp = 0
# fn = 0
# tp = 0
# tn = 0
# for window, label in zip(windows, classification):
	# prediction = clf.predict(window)
	# print('+++++++++++')
	# print('Prediction: ', prediction[0])
	# print('Reality: ', label)
	# print(type(prediction[0]))
	# print(type(label))
	# if prediction[0] == int(label):
		# print('!!!!!!!!!!!!!!!!!!!!!!!')
		# if int(label) == 1:
			# tp = tp + 1
		# elif int(label) == 0:
			# tn = tn + 1
		# else:
			# print('Problem with labels')
			# sys.exit()
	# else:
		# print('......................')
		# if int(label) == 1:
			# fn = fn + 1
		# elif int(label) == 0:
			# fp = fp + 1
		# else:
			# print('Problem with labels')
			# sys.exit()
# print('False Negatives: ', fn)
# print('False Positives: ', fp)
# print('True Negatives: ', tn)
# print('True Positives: ', tp)
# print('Total: ', len(classification))
# recall = 
# precision = 
# accuracy =

# info_results = ['FN', fn, 'FP', fp, 'TN', tn, 'TP', tp, 'Total', len(classification),
# 'Recall', 'Precision', 'Accuracy']

# sys.exit()