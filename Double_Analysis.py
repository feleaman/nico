# Double_Features.py
# Last updated: 14.09.2017 by Felix Leaman
# Description:
# Code for opening 2 x a .mat or .tdms data files with single channel and plotting different types of analysis
# The files and channel must be selected by the user
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively
# Power2 option let the user to analyze only 2^Power2 points of each file

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

# point_index = filename1.find('.')
# extension = filename1[point_index+1] + filename1[point_index+2] + filename1[point_index+3]

# if extension == 'mat':
	# # x1 = f_open_mat(filename1, channel)
	# # x1 = np.ndarray.flatten(x1)
	# # x2 = f_open_mat(filename2, channel)
	# # x2 = np.ndarray.flatten(x2)
	
	# x1 = f_open_mat_2(filename1)
	# x1 = np.ndarray.flatten(x1[0])
	# x2 = f_open_mat_2(filename2)
	# x2 = np.ndarray.flatten(x2[0])

# elif extension == 'tdm': #tdms
	# x1 = f_open_tdms(filename1, channel)
	# x2 = f_open_tdms(filename2, channel)

# elif extension == 'txt': #tdms
	# x1 = np.loadtxt(filename1)
	# x2 = np.loadtxt(filename2)

# elif extension == 'pkl': #tdms
	# x1 = read_pickle(filename1)
	# x2 = read_pickle(filename2)

channel1 = 'KS2' + channel
channel2 = 'KS2' + channel
channel2 = 'DG'


x1 = f_open_tdms(filename1, channel1)
# x2 = f_open_tdms(filename2, channel2)
x2 = read_pickle(filename2)

print(len(x1))
print(len(x2))

filename1 = os.path.basename(filename1) #changes from path to file
filename2 = os.path.basename(filename2)

#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# if channel == 'Koerperschall' or channel == 'AC_0':
	# fs = 50000.
# elif channel == 'Drehmoment':
	# fs = 1000.0
# elif channel == 'AE_Signal' or channel == 'AE_0':
	# fs = 1000000.0
# elif channel == 'AE_Signal' or channel == 'AE1':
	# fs = 2000000.0
# else:
	# print('Error fs assignment')
fs = 50000.

if args.power2 == None:
	# n_points = 2**(max_2power(len(x1)))
	n_points = len(x1)

# x1 = x1*10.
# x2 = x2*10.

# x1 = np.zeros(len(x1))
# x2 = np.zeros(len(x2))


# x1[100000] = 1.
# x1[100001] = 1.
# x1[100002] = 1.
# x1[100003] = 1.
# x1[100004] = 1.
# x1[100005] = 1.


# x1[200003] = 1.
# x1[200004] = 1.
# x1[200005] = 1.
# x1[200006] = 1.
# x1[200007] = 1.
# x1[200008] = 1.

# x1[300005] = 1.
# x1[300006] = 1.
# x1[300007] = 1.
# x1[300008] = 1.
# x1[300009] = 1.
# x1[300010] = 1.






dt = 1.0/fs
n_points = len(x1)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])

#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++
config_analysis = {'WFM':True, 'WFMzoom':False,  'FFT':False, 'PSD':False, 'STFT':False, 'STPSD':False,
'Cepstrum':False, 'Hist':False, 'CyclicSpectrum':False}

config_norm = {'analysis':False, 'type':'rms'}


# config_filter = {'analysis':True, 'type':'highpass', 'mode':'aa', 'params':[[20.e3], 3]}

config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}



config_diff = {'analysis':False, 'length':1, 'same':True}

config_denois = {'analysis':False, 'mode':'butter_highpass', 'freq':5.e3}
# config_denois = {'analysis':True, 'mode':'butter_bandpass', 'freq':[95.e3, 140.e3]}


config_demod = {'analysis':False, 'mode':'hilbert', 'prefilter':['bandpass', [103518, 200581], 3], 
'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 2000.0, 3], 'warm_points':0}

# config_demod = {'analysis':False, 'mode':'hilbert', 'prefilter':['highpass', 20.e3, 3], 
# 'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warm_points':0}
#When hilbert is selected, the other parameters are ignored

config_stft = {'segments':500, 'window':'hanning', 'mode':'magnitude', 'log-scale':False, 'type':'colormesh', 'color':'inferno'}
#colors= gray, inferno, Spectral, copper...

config_stPSD = {'segments':1000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}

config_CyclicSpectrum = {'segments':25, 'freq_range':[20.e3, 450.e3], 'window':'boxcar', 'mode':'magnitude', 'log':False, 'off_PSD':True,
'kHz':True, 'warm_points':None}

#++++++++++++++++++++++ SIGNAL DEFINITION ++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig = [[] for element in config_analysis if config_analysis[element] == True]
fig.append([])
fig[0], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
ax[0].plot(t, x1)
ax[0].set_title(channel1 + ' WFM ' + filename1[:-5], fontsize=10)
ax[0].set_ylabel('Amplitude [g]', fontsize=13)

n_points2 = len(x2)
t2 = np.array([i*dt for i in range(n_points2)])

ax[1].plot(t2, x2)
ax[1].set_title(channel2 + ' WFM ' + filename2[:-5], fontsize=10)
ax[1].set_ylabel('Amplitude', fontsize=13)
ax[1].set_xlabel('Dauer [s]', fontsize=13)
# ax[1].set_xlim(left=95.35, right=95.45)


ax[0].tick_params(axis='both', labelsize=12)
ax[1].tick_params(axis='both', labelsize=12)

ax[0].grid(axis='both')
ax[1].grid(axis='both')

#Filter
if config_norm['analysis'] == True:
	print('+++ normalization')
	if config_norm['type'] == 'rms':
		x1 = x1 / signal_rms(x1)
		x2 = x2 / signal_rms(x2)
	else:
		print('unknown norm type')
	# print('new rms')
	# print(signal_rms(x1))
	# print(signal_rms(x2))

# if config_filter['analysis'] == True:
	# print('+++Filter:')
	# if config_filter['type'] == 'bandpass':
		# print('Bandpass')
		# f_nyq = 0.5*fs
		# order = config_filter['params'][1]
		# freqs_bandpass = [config_filter['params'][0][0]/f_nyq, config_filter['params'][0][1]/f_nyq]
		# b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
		# x1 = signal.filtfilt(b, a, x1)
		# x2 = signal.filtfilt(b, a, x2)
	# elif config_filter['type'] == 'median':
		# print('Median')
		# x1 = scipy.signal.medfilt(x1, kernel_size=5)
		# x2 = scipy.signal.medfilt(x2, kernel_size=5)
	# elif config_filter['type'] == 'highpass':
		# print('highpass!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		# f_nyq = 0.5*fs
		# order = config_filter['params'][1]
		# freq_highpass = config_filter['params'][0][0]/f_nyq
		# b, a = signal.butter(order, freq_highpass, btype='highpass')
		# x1 = signal.filtfilt(b, a, x1)
		# x2 = signal.filtfilt(b, a, x2)
# plt.plot(x1)
# plt.show()
if config_denois['analysis'] == True:
		print('with filter')
		if config_denois['mode'] == 'butter_highpass':
			print('highpass!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			x1 = butter_highpass(x=x1, fs=fs, freq=config_denois['freq'], order=3, warm_points=None)
			x2 = butter_highpass(x=x2, fs=fs, freq=config_denois['freq'], order=3, warm_points=None)
		elif config_denois['mode'] == 'butter_bandpass':
			x1 = butter_bandpass(x=x1, fs=fs, freqs=config_denois['freq'], order=3, warm_points=None)
			x2 = butter_bandpass(x=x2, fs=fs, freqs=config_denois['freq'], order=3, warm_points=None)
		elif config_denois['mode'] == 'butter_lowpass':
			x1 = butter_lowpass(x=x1, fs=fs, freq=config_denois['freq'], order=3, warm_points=None)
			x2 = butter_lowpass(x=x2, fs=fs, freq=config_denois['freq'], order=3, warm_points=None)

		else:
			print('Error assignment denois')

# plt.plot(x1)
# plt.show()

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

# x1 = x1**2.0	
# x2 = x2**2.0	

		
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

#Autocorrelation
if config_autocorr['analysis'] == True:
	print('+++Autocorr:')
	if config_autocorr['type'] == 'definition':
		x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
		x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
	
	elif config_autocorr['type'] == 'wiener':	
		fftx1 = np.fft.fft(x1)
		x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))
		# x1 = x1[0:int(len(x1)/2)]
		fftx2 = np.fft.fft(x2)
		x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))
	else:
		print('Error assignment autocorrelation')

warm = 0.0
if (config_demod['warm_points'] != 0 and config_demod['mode'] == 'butter' and config_demod['analysis'] == True):
	x1 = x1[config_demod['warm_points']:]
	x2 = x2[config_demod['warm_points']:]
	t = t[config_demod['warm_points']:]
	warm = float(config_demod['warm_points'])

# print('+++RMS')
# print(filename1, signal_rms(x1))
# print(filename2, signal_rms(x2))
# sys.exit()
#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['FFT'] == True:
	magX1, f, df = mag_fft_hanning(x1, fs)
	magX2, f, df = mag_fft_hanning(x2, fs)


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


#++++++++++++++++++++++ CYCLIC SPECTRUM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if config_analysis['CyclicSpectrum'] == True:
	segments = config_CyclicSpectrum['segments']
	window = config_CyclicSpectrum['window']
	mode = config_CyclicSpectrum['mode']
	freq_range = config_CyclicSpectrum['freq_range']
	
	CyclicSpectrum1, a_CyclicSpectrum1, f_CyclicSpectrum1 = Cyclic_Spectrum(x=x1, fs=fs, segments=segments, freq_range=freq_range, warm_points=config_CyclicSpectrum['warm_points'])
	CyclicSpectrum2, a_CyclicSpectrum2, f_CyclicSpectrum2 = Cyclic_Spectrum(x=x2, fs=fs, segments=segments, freq_range=freq_range, warm_points=config_CyclicSpectrum['warm_points'])
	if config_CyclicSpectrum['off_PSD'] == True:
		for i in range(len(CyclicSpectrum1)):
			CyclicSpectrum1[i][0] = 0.			
		for i in range(len(CyclicSpectrum2)):
			CyclicSpectrum2[i][0] = 0.

	if config_CyclicSpectrum['log'] == True:
		CyclicSpectrum1 = np.log(CyclicSpectrum1)
		CyclicSpectrum2 = np.log(CyclicSpectrum2)

	
print(len(x1))
print(len(x2))

#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
count = 1

for element in config_analysis:
	if config_analysis[element] == True:
		if element == 'WFM':
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			# tx = np.linspace(0, t[len(t)-1], len(x1))
			# x1 = np.interp(t, tx, x1)
			ax[0].plot(t, x1)
			# ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_title('Faulty Case Train Signal: 1500RPM / 80% Load', fontsize=10)
			if config_norm['analysis'] == True:
				ax[0].set_ylabel('Norm. Amplitude')
			else:
				ax[0].set_ylabel('Amplitude')


			ax[1].plot(t2, x2)
			# ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_title('Healthy Case Train Signal: 1500RPM / 80% Load', fontsize=10)
			if config_norm['analysis'] == True:
				ax[1].set_ylabel('Norm. Amplitude')
			else:
				ax[1].set_ylabel('Amplitude')
			ax[1].set_xlabel('Time s')
		
		elif element == 'WFMzoom':
			fig[count], ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
			# tx = np.linspace(0, t[len(t)-1], len(x1))
			# x1 = np.interp(t, tx, x1)
			ax[0][0].plot(t, x1)
			# ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0][0].set_title('Faulty Train Signal: 1500RPM / 80% Load', fontsize=10)
			ax[0][0].set_ylabel('Amplitude')

			ax[1][0].plot(t, x2)
			# ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1][0].set_title('Healthy Train Signal: 1500RPM / 80% Load', fontsize=10)
			ax[1][0].set_ylabel('Amplitude')
			ax[1][0].set_xlabel('Time s')
			
			ax[0][1].plot(t, x1)
			# ax[0].set_title(channel + ' ' + element + '\n' + filename1)

			ax[0][1].set_title('Faulty Train Signal: 1500RPM / 80% Load', fontsize=10)
			ax[0][1].set_ylabel('Amplitude')

			ax[1][1].plot(t, x2)
			# ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1][1].set_title('Healthy Train Signal: 1500RPM / 80% Load', fontsize=10)
			ax[1][1].set_ylabel('Amplitude')
			ax[1][1].set_xlabel('Time s')
			
			

		elif element == 'FFT':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			ax[0].plot(f, magX1, 'r')
			ax[0].set_title(channel1 + ' FFT ' + filename1[:-5], fontsize=10)
			ax[0].set_ylabel('Magnitude [g]', fontsize=13)

			ax[1].plot(f, magX2, 'r')
			ax[1].set_title(channel2 + ' FFT ' + filename2[:-5], fontsize=10)
			ax[1].set_ylabel('Magnitude [g]', fontsize=13)
			ax[1].set_xlabel('Frequenz [Hz]', fontsize=13)
			
			ax[0].grid(axis='both')
			ax[1].grid(axis='both')
			
			ax[0].ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
			ax[1].ticklabel_format(style='sci', scilimits=(-2, 2), axis='y')
		
		elif element == 'PSD':		
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			ax[0].plot(f_psd, psdX1)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Amplitude')

			ax[1].plot(f_psd, psdX2)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Magnitude [g]', fontsize=13)
			ax[1].set_xlabel('Frequency Hz', fontsize=13)	
			ax[0].tick_params(axis='both', labelsize=12)
			ax[1].tick_params(axis='both', labelsize=12)
			
		elif element == 'STFT':
			if config_stft['type'] == 'colormesh':
				fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
				map = ax[0].pcolormesh(t_stft, f_stft, stftX1, cmap=config_stft['color'])
				ax[0].set_title(channel1 + ' FFT ' + filename1[:-5], fontsize=10)
				ax[0].set_ylabel('Frequenz [Hz]', fontsize=13)			
				fig[count].colorbar(map, ax=ax[0], extendrect=True, extend='both', extendfrac=0)
				if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
					ax[0].set_ylim((0, 1.5*config_demod['filter'][1]))
				
				map = ax[1].pcolormesh(t_stft, f_stft, stftX2, cmap=config_stft['color'])
				ax[1].set_title(channel2 + ' FFT ' + filename2[:-5], fontsize=10)
				ax[1].set_ylabel('Frequenz [Hz]', fontsize=13)
				ax[1].set_xlabel('Dauer [s]', fontsize=13)
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
			ax[0].hist(x1, bins=50)
			ax[0].set_title(channel + ' ' + element + '\n' + filename1)
			ax[0].set_ylabel('Number')
			ax[1].hist(x2, bins=50)
			ax[1].set_title(channel + ' ' + element + '\n' + filename2)
			ax[1].set_ylabel('Number')
			ax[1].set_xlabel('Bins')

			
		elif element == 'CyclicSpectrum':
			map = []
			vmax = np.max([max_cspectrum(CyclicSpectrum1), max_cspectrum(CyclicSpectrum2)])
			fig[count], ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
			if config_CyclicSpectrum['kHz'] == True:
				fact = 1000.
				yscalename = 'Frequency kHz'
			else:
				fact = 1
				yscalename = 'Frequency Hz'
			
			map.append(ax[0].pcolormesh(a_CyclicSpectrum1, f_CyclicSpectrum1/fact, CyclicSpectrum1, cmap='Purples', vmax=vmax))

			name1 = 'Faulty Case Traininig Signal: '
			name2 = 'Healthy Case Traininig Signal: '
			flag = filename1.find('1500')
			if flag != -1:
				flag2 = filename1.find('80')
				if flag2 != -1:
					name1 = name1 + '1500RPM / 80% Load'
					name2 = name2 + '1500RPM / 80% Load'
				else:
					name1 = name1 + '1500RPM / 40% Load'
					name2 = name2 + '1500RPM / 40% Load'
			else:
				name1 = name1 + '1000RPM / 80% Load'
				name2 = name2 + '1000RPM / 80% Load'
			ax[0].set_title(name1, fontsize=10)
			# ax[0].set_title(filename1)
			ax[0].set_ylabel(yscalename)
			# ax[0].set_xlabel('Cyclic Frequency Hz')
			if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
				ax[0].set_ylim((0, config_demod['filter'][1]))
			# for tik in ax[0].get_xticklabels():
				# tik.set_visible(True)

			
			map.append(ax[1].pcolormesh(a_CyclicSpectrum2, f_CyclicSpectrum2/fact, CyclicSpectrum2, cmap='Purples', vmax=vmax))
			ax[1].set_title(name2, fontsize=10)
			ax[1].set_ylabel(yscalename)
			ax[1].set_xlabel('Cyclic Frequency Hz')
			if (config_demod['analysis'] == True and config_demod['mode'] == 'butter' and config_diff['analysis'] == False):
				ax[1].set_ylim((0, config_demod['filter'][1]))
			for tik in ax[1].get_yticklabels():
				tik.set_visible(True)
			for tik in ax[1].get_xticklabels():
				tik.set_visible(True)
				
			indmax = np.argmax([max_cspectrum(CyclicSpectrum1), max_cspectrum(CyclicSpectrum2)])
			fig[count].colorbar(map[indmax], ax=ax.ravel().tolist())
			
			

		count = count + 1

plt.show()



