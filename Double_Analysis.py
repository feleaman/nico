# Double_Features.py
# Last updated: 15.08.2017 by Felix Leaman
# Description:
# Code for opening 2 x a .mat or .tdms data files with single channel and plotting different types of analysis
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
	x2 = f_open_tdms(filename1, channel)


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


dt = 1.0/fs
n_points = len(x1)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])

#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

config_analysis = {'WFM':True, 'FFT':True, 'PSD':False, 'STFT':False, 'STPSD':False,
'Cepstrum':False, 'Quantile':False, 'Hist':False}



config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_diff = {'analysis':False, 'length':1, 'same':True}


config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['highpass', 80.0e3, 3], 
'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 200.0, 3]}
#When hilbert is selected, the other parameters are ignored




config_stft = {'segments':1000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}

config_stPSD = {'segments':1000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}

quantile = 1000
#++++++++++++++++++++++ SIGNAL DEFINITION ++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_autocorr['analysis'] == True:
	if config_autocorr['type'] == 'definition':
		x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
		x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
	elif config_autocorr['type'] == 'wiener':		
		# fftx1, f, df = mag_fft(x=x1, fs=fs)
		# x1 = np.fft.ifft(fftx1**2.0)
		# x1 = x1*len(x1)
		
		# fftx2, f, df = mag_fft(x=x2, fs=fs)
		# x2 = np.fft.ifft(fftx2**2.0)
		# x2 = x2*len(x2)
		
		
		
		fftx1 = np.fft.fft(x1)
		x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))
		
		fftx2 = np.fft.fft(x2)
		x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))
		
		

if config_demod['analysis'] == True:
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

if config_diff['analysis'] == True:
	if config_diff['same'] == True:
		x1 = diff_signal_eq(x=x1, length_diff=config_diff['length'])
		x2 = diff_signal_eq(x=x2, length_diff=config_diff['length'])
	elif config_diff['same'] == False:
		x1 = diff_signal(x=x1, length_diff=config_diff['length'])
		x2 = diff_signal(x=x2, length_diff=config_diff['length'])
	else:
		print('Error assignment diff')

# print(signal_rms(x1))
# print(np.std(x1))
# print(np.mean(x1))


# print(signal_rms(x2))
# print(np.std(x2))
# print(np.mean(x2))



#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['FFT'] == True:
	magX1, f, df = mag_fft(x1, fs)
	magX2, f, df = mag_fft(x2, fs)


#++++++++++++++++++++++ STFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['STFT'] == True:
	segments = config_stft['segments']
	window = config_stft['window']
	mode = config_stft['mode']
	
	stftX1, f_stft, df_stft, t_stft = shortFFT(x1[1000:len(x1)-1000], fs, segments, window, mode)
	stftX2, f_stft, df_stft, t_stft = shortFFT(x2[1000:len(x2)-1000], fs, segments, window, mode)
	if config_stft['log-scale'] == True:
		stftX1 = np.log(stftX1)
		stftX2 = np.log(stftX2)


#++++++++++++++++++++++ PSD +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['PSD'] == True:
	f_psd, psdX1 = signal.periodogram(x1, fs, return_onesided=True, scaling='density')
	f_psd, psdX2 = signal.periodogram(x2, fs, return_onesided=True, scaling='density')

#++++++++++++++++++++++ STPSD +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['STPSD'] == True:
	segments = config_stPSD['segments']
	window = config_stPSD['window']
	mode = config_stPSD['mode']
	
	stPSDX1, f_stPSD, df, t_stPSD = shortPSD(x1, fs, segments)
	stPSDX2, f_stPSD, df, t_stPSD = shortPSD(x2, fs, segments)	
	if config_stPSD['log-scale'] == True:
		stPSDX1 = np.log(stPSDX1)
		stPSDX2 = np.log(stPSDX2)


#++++++++++++++++++++++ CEPSTRUM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if config_analysis['Cepstrum'] == True:
	cepstrumX1, tc, dtc = cepstrum_real(x1, fs)
	cepstrumX2, tc, dtc = cepstrum_real(x2, fs)

#++++++++++++++++++++++ QUANTILE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if config_analysis['Quantile'] == True:
	quant_x1 = np.percentile(x1, np.arange(0, 100, 100/quantile))
	quant_x2 = np.percentile(x2, np.arange(0, 100, 100/quantile))

	
#++++++++++++++++++++++ AUTOCORR +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
count = 0
fig = [[] for element in config_analysis if config_analysis[element] == True]
for element in config_analysis:
	if config_analysis[element] == True:
		# count = count + 1
		# fig.append(plt.figure(count))
		
		# plt.figure(count)
		# plt.title(channel + ' ' + element)
		# plt.suptitle(filename, fontsize=10)
		if element == 'WFM':
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			ax[0].plot(t, x1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(t, x2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Time s')

		elif element == 'FFT':
		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].plot(f, magX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(f, magX2)
			# ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Frequency Hz')
			
			
			
			
			
		elif element == 'PSD':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].plot(f_psd, psdX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(f_psd, psdX2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Frequency Hz')	
		
		
		
		
		elif element == 'STFT':			
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].pcolormesh(t_stft, f_stft, stftX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Frequency Hz')
			if (config_demod['analysis'] == True and config_demod['mode'] == 'butter'):
				ax[0].set_ylim((0, config_demod['filter'][1]))
			

			ax[1].pcolormesh(t_stft, f_stft, stftX2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Frequency Hz')
			ax[1].set_xlabel('Time s')
			if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
				ax[1].set_ylim((0, config_demod['filter'][1]))
			
		elif element == 'STPSD':			
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].pcolormesh(t_stPSD, f_stPSD, stPSDX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Frequency Hz')
			if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
				ax[0].set_ylim((0, config_demod['filter'][1]))

			ax[1].pcolormesh(t_stPSD, f_stPSD, stPSDX2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Frequency Hz')
			ax[1].set_xlabel('Time s')
			if (config_demod['analysis'] == True and config_demod['mode'] == 'butter'):
				ax[1].set_ylim((0, config_demod['filter'][1]))
			
			
			
			
		elif element == 'Cepstrum':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].plot(tc, cepstrumX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(tc, cepstrumX2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Quefrency s')	
			
			
		elif element == 'Quantile':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].plot(np.arange(quantile), quant_x1, '-o')
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Number')

			ax[1].plot(np.arange(quantile), quant_x2, '-o')
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Number')
			ax[1].set_xlabel('Quantile')	
			
		elif element == 'Hist':	
			# hist1, jiji1 = np.histogram(x1, bins=100)
			# hist2, jiji2 = np.histogram(x2, bins=100)
			# hist1 = [hist1[i+1] - hist1[i] for i in range(len(hist1)-1)]
			# hist2 = [hist2[i+1] - hist2[i] for i in range(len(hist2)-1)]		
			# fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			# ax[0].plot(jiji1[2:], hist1)
			# ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			# ax[0].set_ylabel('Number')
			# ax[1].plot(jiji2[2:],hist2)
			# ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			# ax[1].set_ylabel('Number')
			# ax[1].set_xlabel('Bins')
			
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].hist(x1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Number')
			ax[1].hist(x2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Number')
			ax[1].set_xlabel('Bins')

			
			
			

		count = count + 1

plt.show()



