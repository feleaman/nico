# Main_Analysis.py
# Last updated: 15.08.2017 13:13 by Felix Leaman
# Description:
# Code for opening a .mat or .tdms data file with single channel and plotting different types of analysis
# The file and channel is selected by the user
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

import os.path
import sys


sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--channel', nargs='?')
# channel = 'AE_Signal'
# channel = 'Koerperschall'
# channel = 'Drehmoment'


parser.add_argument('--power2', nargs='?')
# n_points = 2**power2

parser.add_argument('--type', nargs='?')
args = parser.parse_args()

if args.channel != None:
	channel = args.channel
else:
	print('Required: Channel')
	sys.exit()

if args.power2 != None:
	n_points = 2**int(args.power2)

# print(channel)
# print(n_points)

# print(args.pos)
# print(args.type)
# sys.exit()

#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++

root = Tk()
root.withdraw()
root.update()
filename = filedialog.askopenfilename()
root.destroy()

point_index = filename.find('.')
extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]

if extension == 'mat':
	x = f_open_mat(filename, channel)
	x = np.ndarray.flatten(x)

elif extension == 'tdm': #tdms
	x = f_open_tdms(filename, channel)

elif extension == 'txt': #tdms
	x = np.loadtxt(filename)
filename = os.path.basename(filename) #changes from path to file


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
	print('jiji')
	n_points = 2**(max_2power(len(x)))
print(n_points)
x = x[0:n_points]	
# nemo = 10.0
# count = 0
# while nemo > 1.0:
	# count = count + 1
	# nemo = n / 2.0**count
# count = count - 1
	
# x = x[0:n] #reduces number of points to a power of 2 to improve speed
# x = x[0:10000]	
dt = 1.0/fs
n = len(x)
tr = n*dt
t = np.array([i*dt for i in range(n)])

#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

config_analysis = {'WFM':True, 'FFT':True, 'PSD':False, 'STFT':False, 'STPSD':False, 'Cepstrum':False}

config_demod = {'analysis':False, 'mode':'hilbert', 'prefilter':['highpass', 60.0e3, 3], 
'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 50.0, 3]}
#When hilbert is selected, the other parameters are ignored

config_diff = {'analysis':False, 'length':1, 'same':True}

config_stft = {'segments':100, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}

config_stPSD = {'segments':100, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}


#++++++++++++++++++++++ SIGNAL DEFINITION ++++++++++++++++++++++++++++++++++++++++++++++++++++++

if config_demod['analysis'] == True:
	if config_demod['mode'] == 'hilbert':
		x = hilbert_demodulation(x)
	elif config_demod['mode'] == 'butter':
		x = butter_demodulation(x=x, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
		type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
	else:
		print('Error assignment demodulation')

if config_diff['analysis'] == True:
	if config_diff['same'] == True:
		x = diff_signal_eq(x=x, length_diff=config_diff['length'])
	elif config_diff['same'] == False:
		x = diff_signal(x=x, length_diff=config_diff['length'])
	else:
		print('Error assignment diff')


#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['FFT'] == True:
	magX, f, df = mag_fft(x, fs)


#++++++++++++++++++++++ STFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['STFT'] == True:
	segments = config_stft['segments']
	window = config_stft['window']
	mode = config_stft['mode']
	
	stftX, f_stft, df_stft, t_stft = shortFFT(x, fs, segments, window, mode)
	if config_stft['log-scale'] == True:
		stftX = np.log(stftX)


#++++++++++++++++++++++ PSD +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['PSD'] == True:
	f_psd, psdX = signal.periodogram(x, fs, return_onesided=True, scaling='density')

#++++++++++++++++++++++ STPSD +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['STPSD'] == True:
	segments = config_stPSD['segments']
	window = config_stPSD['window']
	mode = config_stPSD['mode']
	
	stPSDX, f_stPSD, df, t_stPSD = shortPSD(x, fs, segments)	
	if config_stPSD['log-scale'] == True:
		stPSDX = np.log(stPSDX)


#++++++++++++++++++++++ CEPSTRUM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if config_analysis['Cepstrum'] == True:
	cepstrumX, tc, dtc = cepstrum_real(x, fs)



#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
count = 0
fig = []
for element in config_analysis:
	if config_analysis[element] == True:
		# count = count + 1
		fig.append(plt.figure(count))
		
		# plt.figure(count)
		# plt.title(channel + ' ' + element)
		# plt.suptitle(filename, fontsize=10)
		if element == 'WFM':
			ax_wfm = fig[count].add_subplot(1,1,1)
			ax_wfm.set_title(channel + ' ' + element + '\n' + filename)
			ax_wfm.set_label('AE')
			ax_wfm.plot(t, x)
			ax_wfm.set_xlabel('Time s')
			ax_wfm.set_ylabel('Amplitude')

		elif element == 'FFT':
			ax_fft = fig[count].add_subplot(1,1,1)
			ax_fft.set_title(channel + ' ' + element + '\n' + filename)
			ax_fft.plot(f, magX)
			ax_fft.set_xlabel('Frequency Hz')
			ax_fft.set_ylabel('Amplitude')
		elif element == 'PSD':
			ax_psd = fig[count].add_subplot(1,1,1)
			ax_psd.set_title(channel + ' ' + element)
			ax_psd.plot(f_psd, psdX)
			ax_psd.set_xlabel('Frequency Hz')
			ax_psd.set_ylabel('Amplitude')
			# ax.set_xticklabels(['{:.}'.format(int(x)) for x in ax.get_xticks().tolist()])
		elif element == 'STFT':
			ax_stft = fig[count].add_subplot(1,1,1)
			ax_stft.set_title(channel + ' ' + element)
			ax_stft.pcolormesh(t_stft, f_stft, stftX)
			ax_stft.set_xlabel('Time s')
			ax_stft.set_ylabel('Frequency Hz')

		elif element == 'STPSD':
			ax_stpsd = fig[count].add_subplot(1,1,1)
			ax_stpsd.set_title(channel + ' ' + element)
			ax_stpsd.pcolormesh(t_stPSD, f_stPSD, stPSDX)
			ax_stpsd.set_xlabel('Time s')
			ax_stpsd.set_ylabel('Frequency Hz')
		elif element == 'Cepstrum':
			ax_cepstrum = fig[count].add_subplot(1,1,1)
			ax_cepstrum.set_title(channel + ' ' + element)
			ax_cepstrum.plot(tc, cepstrumX)
			ax_cepstrum.set_xlabel('Quefrency s')
			ax_cepstrum.set_ylabel('Amplitude')

		count = count + 1

plt.show()



# count = 0
# for element in config_analysis:
	# if config_analysis[element] == True:
		# count = count + 1
		# plt.figure(count)
		# plt.title(channel + ' ' + element)
		# plt.suptitle(filename, fontsize=10)
		# if element == 'WFM':
			# plt.xlabel('Time s')
			# plt.ylabel('Amplitude')
			# plt.plot(t, x)
		# elif element == 'FFT':
			# plt.xlabel('Frequency Hz')
			# plt.ylabel('Amplitude')
			# plt.plot(f, magX, 'r')
		# elif element == 'PSD':
			# plt.xlabel('Frequency Hz')
			# plt.ylabel('Amplitude')
			# plt.plot(f_psd, psdX, 'g')
		# elif element == 'STFT':
			# plt.xlabel('Time s')
			# plt.ylabel('Frequency Hz')
			# plt.pcolormesh(t_stft, f_stft, stftX)	
		# elif element == 'STPSD':
			# plt.xlabel('Time s')
			# plt.ylabel('Frequency Hz')
			# plt.pcolormesh(t_stPSD, f_stPSD, stPSDX)
		# elif element == 'Cepstrum':
			# plt.xlabel('Quefrency s')
			# plt.ylabel('Amplitude')
			# plt.plot(tc, cepstrumX)

# plt.show()


