# Burst_Detection.py
# Last updated: 17.08.2017 by Felix Leaman
# Description:
# Code for opening 2 x a .mat or .tdms data files with single channel and plotting different types of analysis
# The file and channel is selected by the user
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
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
x1raw = x1
x2raw = x2
# traw

np.savetxt('defect_v3_n1500_m80.txt', x1)
np.savetxt('ok_v3_n1500_m80.txt', x2)


dt = 1.0/fs
n_points = len(x1)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])

#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

config_analysis = {'WFM':True, 'FFT':True, 'PSD':False, 'STFT':False, 'STPSD':False,
'Cepstrum':False, 'Hist':False}


config_filter = {'analysis':False, 'type':'median', 'mode':'bandpass', 'params':[[180.0e3, 350.0e3], 3]}


config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_diff = {'analysis':True, 'length':1, 'same':True}


config_demod = {'analysis':True, 'mode':'butter', 'prefilter':['bandpass',[100.0e3, 350.0e3], 3], 
'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 80.0, 3], 'offwarming':False}
#When hilbert is selected, the other parameters are ignored

config_stft = {'segments':1000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False, 'type':'binary', 'color':'gray'}
#colors= gray, inferno, Spectral, copper...

config_stPSD = {'segments':1000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}




#++++++++++++++++++++++ RAW SIGNAL ++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig = [[] for element in config_analysis if config_analysis[element] == True]
# fig.append([])
# fig[0], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
# # n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst = id_burst_threshold(x=x1, fs=fs, threshold=8*signal_rms(x1), t_window=0.002)
# ax[0].plot(t, x1, 'b')
# # ax[0].plot(t_burst_corr, amp_burst_corr, 'ro')
# ax[0].set_title(channel + ' ' + 'Raw WFM' + '\n' + filename1, fontsize=10)
# ax[0].set_ylabel('Amplitude')

# # n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst = id_burst_threshold(x=x2, fs=fs, threshold=8*signal_rms(x2), t_window=0.002)
# ax[1].plot(t, x2, 'b')
# # ax[1].plot(t_burst_corr, amp_burst_corr, 'ro')
# ax[1].set_title(filename2, fontsize=10)
# ax[1].set_ylabel('Amplitude')
# ax[1].set_xlabel('Time s')
# # plt.show()

# # sys.exit()


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

# print(scipy.signal.find_peaks_cwt(vector=x1, widths=np.arange(1, 4))*dt)

if config_demod['offwarming'] == True:
	# x1 = x1[20000:]
	# x2 = x2[20000:]
	# t = t[20000:]
	x1[0:20000] = np.zeros(20000)
	x2[0:20000] = np.zeros(20000)
	# t = t[20000:]

	threshold1 = 4*signal_rms(x1[2000:])
	threshold2 = 4*signal_rms(x2[2000:])
else:
	threshold1 = 1.3*signal_rms(x1)
	threshold2 = 1.3*signal_rms(x2)

t_window1 = 0.002
t_window2 = 0.002

#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['FFT'] == True:
	magX1, f, df = mag_fft(x1, fs)
	magX2, f, df = mag_fft(x2, fs)


#++++++++++++++++++++++ STFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['STFT'] == True:
	segments = config_stft['segments']
	window = config_stft['window']
	mode = config_stft['mode']
	
	stftX1, f_stft, df_stft, t_stft = shortFFT(x=x1, fs=fs, segments=segments, window=window, mode=mode)
	stftX2, f_stft, df_stft, t_stft = shortFFT(x=x2, fs=fs, segments=segments, window=window, mode=mode)
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
	
	stPSDX1, f_stPSD, df, t_stPSD = shortPSD(x=x1, fs=fs, segments=segments)
	stPSDX2, f_stPSD, df, t_stPSD = shortPSD(x=x2, fs=fs, segments=segments)	
	if config_stPSD['log-scale'] == True:
		stPSDX1 = np.log(stPSDX1)
		stPSDX2 = np.log(stPSDX2)


#++++++++++++++++++++++ CEPSTRUM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if config_analysis['Cepstrum'] == True:
	cepstrumX1, tc, dtc = cepstrum_real(x1, fs)
	cepstrumX2, tc, dtc = cepstrum_real(x2, fs)

	
#++++++++++++++++++++++ BURST DETECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst = id_burst_threshold(x=x1, fs=fs, threshold=threshold1, t_window=t_window1)
# ax[0].plot(t, x1)
# ax[0].plot(t_burst_corr, amp_burst_corr, 'ro')

# n_burst_corr, t_burst_corr, amp_burst_corr, t_burst, amp_burst = id_burst_threshold(x=x2, fs=fs, threshold=threshold2, t_window=t_window2)
# ax[1].plot(t, x2)
# ax[1].plot(t_burst_corr, amp_burst_corr, 'ro')



#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
count = 0
# fig = [[] for element in config_analysis if config_analysis[element] == True]
for element in config_analysis:
	Name = element
	if config_filter['analysis'] == True:
		Name = Name + ' FIL'
	if config_autocorr['analysis'] == True:
		Name = Name + ' ACR'

	if config_demod['analysis'] == True:
		Name = Name + ' ENV'
		
	if config_diff['analysis'] == True:
		Name = Name + ' DIF'
	
	if config_analysis[element] == True:
		if element == 'WFM':
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=fs, threshold=threshold1, t_window=t_window1)
			print(n_burst_corr1)
			ax[0].axhline(threshold1, color='k')
			ax[0].plot(t, x1, color='darkblue')
			ax[0].plot(t_burst_corr1, amp_burst_corr1, 'ro')
			
			
			ax[0].set_title(channel + ' ' + Name + '\n' + filename1, fontsize=10)
			ax[0].set_ylabel('Amplitude')
			
			
			n_burst_corr2, t_burst_corr2, amp_burst_corr2, t_burst2, amp_burst2 = id_burst_threshold(x=x2, fs=fs, threshold=threshold2, t_window=t_window2)
			print(n_burst_corr2)
			ax[1].axhline(threshold2, color='k')
			ax[1].plot(t, x2, color='darkblue')
			ax[1].plot(t_burst_corr2, amp_burst_corr2, 'ro')
			
			ax[1].set_title(filename2, fontsize=10)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Time s')
			
			
			
			
			
			
			

		elif element == 'FFT':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			ax[0].plot(f, magX1, 'r')
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(f, magX2, 'r')
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Frequency Hz')
			
		
		elif element == 'PSD':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			ax[0].plot(f_psd, psdX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(f_psd, psdX2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Frequency Hz')	
			
		
		elif element == 'STFT':
			if config_stft['type'] == 'colormesh':
				fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
				map = ax[0].pcolormesh(t_stft, f_stft, stftX1, cmap=config_stft['color'])
				ax[0].set_title(channel + ' ' + element + '\n' + filename1)
				ax[0].set_ylabel('Frequency Hz')			
				fig[count].colorbar(map, ax=ax[0], extendrect=True, extend='both', extendfrac=0)
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[0].set_ylim((0, 1.5*config_demod['filter'][1]))
				
				map = ax[1].pcolormesh(t_stft, f_stft, stftX2, cmap=config_stft['color'])
				ax[1].set_title(channel + ' ' + element + '\n' + filename2)
				ax[1].set_ylabel('Frequency Hz')
				ax[1].set_xlabel('Time s')
				fig[count].colorbar(map, ax=ax[1], extendrect=True, extend='both', extendfrac=0)
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[1].set_ylim((0, 1.5*config_demod['filter'][1]))
			
			
			elif config_stft['type'] == 'image':
				fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
				map = ax[0].imshow(stftX1, aspect='auto', interpolation='bilinear', cmap=config_stft['color'], vmax=abs(stftX2).max(), vmin=-abs(stftX2).max(), origin='lower')
				ax[0].set_title(channel + ' ' + element + '\n' + filename1)
				ax[0].set_ylabel('Frequency Hz')			
				fig[count].colorbar(map, ax=ax[0], extendrect=True, extend='both', extendfrac=0)
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[0].set_ylim((0, 1.5*config_demod['filter'][1]))
				
				map = ax[1].imshow(stftX2, aspect='auto', interpolation='bilinear', cmap=config_stft['color'], vmax=abs(stftX2).max(), vmin=-abs(stftX2).max(), origin='lower')
				ax[1].set_title(channel + ' ' + element + '\n' + filename2)
				ax[1].set_ylabel('Frequency Hz')
				ax[1].set_xlabel('Time s')
				fig[count].colorbar(map, ax=ax[0], extendrect=True, extend='both', extendfrac=0)
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[1].set_ylim((0, 1.5*config_demod['filter'][1]))
			
			
			
			elif config_stft['type'] == 'binary':
				# stftX1 = stftX1 / np.max(np.abs(stftX1))

				stftX1 = img_as_uint(stftX1)
				stftX1 = skimage.filters.median(image=stftX1, selem=np.ones((3, 3)))

				
				stftX2 = img_as_uint(stftX2)
				stftX2 = skimage.filters.median(image=stftX2, selem=np.ones((3, 3)))
				
				
				thr1 = skimage.filters.threshold_otsu(stftX1)
				thr2 = skimage.filters.threshold_otsu(stftX2)
				
				
				
				for i in range(len(stftX1)):
					for j in range(len(stftX1[0])):
						if stftX1[i][j] > thr1:
							stftX1[i][j] = 1
						else:
							stftX1[i][j] = 0
						if stftX2[i][j] > thr2:
							stftX2[i][j] = 1
						else:
							stftX2[i][j] = 0
				
				fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
				map = ax[0].pcolormesh(t_stft, f_stft, stftX1, cmap=config_stft['color'])
				ax[0].set_title(channel + ' ' + element + '\n' + filename1)
				ax[0].set_ylabel('Frequency Hz')			
				fig[count].colorbar(map, ax=ax[0], extendrect=True, extend='both', extendfrac=0)
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[0].set_ylim((0, 1.5*config_demod['filter'][1]))
				
				map = ax[1].pcolormesh(t_stft, f_stft, stftX2, cmap=config_stft['color'])
				ax[1].set_title(channel + ' ' + element + '\n' + filename2)
				ax[1].set_ylabel('Frequency Hz')
				ax[1].set_xlabel('Time s')
				fig[count].colorbar(map, ax=ax[1], extendrect=True, extend='neither')
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[1].set_ylim((0, 1.5*config_demod['filter'][1]))
				
				# binis = stftX1[0]				
				# bursts = 0
				# for s in range(len(binis)-1):
					# if (binis[s] == 0 and binis[s+1] == 1):
						# bursts = bursts + 1
				# print(bursts)
			
			
			
			
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
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True)
			ax[0].hist(x1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Number')
			ax[1].hist(x2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Number')
			ax[1].set_xlabel('Bins')

			
			
			

		count = count + 1
#++++++++++++++++++++++ BURST DETECTION RAW +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig.append([])






fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

# 

# ax[1].plot(t, x2, 'b')
# # ax[1].plot(t_burst_corr, amp_burst_corr, 'ro')
# 

# # plt.show()
amp_burst_corr1 = np.array([x1raw[int(time*fs)] for time in t_burst_corr1])
amp_burst_corr2 = np.array([x2raw[int(time*fs)] for time in t_burst_corr2])


ax[0].plot(t, x1raw)
ax[0].plot(t_burst_corr1, amp_burst_corr1, 'ro')
ax[0].set_title(channel + ' ' + 'Raw WFM' + '\n' + filename1, fontsize=10)
ax[0].set_ylabel('Amplitude')
# ax[0].axhline(threshold1, color='k')

ax[1].plot(t, x2raw)
ax[1].plot(t_burst_corr2, amp_burst_corr2, 'ro')
ax[1].set_title(filename2, fontsize=10)
# ax[1].axhline(threshold2, color='k')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Time s')
# plt.show()
# sys.exit()


plt.show()



