import numpy as np
from scipy import signal
import scipy
import math
import sys



def butter_demodulation(x, fs, filter, prefilter=None, type_rect=None, dc_value=None):
	n = len(x)
	f_nyq = 0.5*fs
	
	#Defaults
	if not dc_value:
		dc_value = 'without_dc'
	
	if not type_rect:
		type_rect = 'only_positives'
	
	if not prefilter:
		prefilter = ['none', 0, 0]
	
	#Pre-filter
	type_prefilter = prefilter[0]
	freq_prefilter = prefilter[1] #normalized freqs
	order_prefilter = prefilter[2]
	if type_prefilter == 'highpass':
		freq_highpass = freq_prefilter/f_nyq
		b, a = signal.butter(order_prefilter, freq_highpass, btype='highpass')
		x_prefilter = signal.filtfilt(b, a, x)
	elif type_prefilter == 'bandpass':
		freqs_bandpass = [freq_prefilter[0]/f_nyq, freq_prefilter[1]/f_nyq]
		b, a = signal.butter(order_prefilter, freqs_bandpass, btype='bandpass')
		x_prefilter = signal.filtfilt(b, a, x)
	else:
		print('Info: Demodulation without pre-filter')
		x_prefilter = x

	#Rectification
	if type_rect == 'only_positives':
		x_rect = np.zeros(n)
		for i in range(n):
			if x_prefilter[i] > 0:
				x_rect[i] = x_prefilter[i]
	elif type_rect == 'absolute_value':
		x_rect = np.abs(x_prefilter)
	else:
		print('Info: Demodulation without rectification')
		x_rect = x_prefilter

	#DC Value		
	if dc_value == 'without_dc':
		x_rect = x_rect - np.mean(x_rect)
	else: 
		print('Info: Demodulation without DC value elimination')

	#Lowpass
	type_filter = filter[0]
	freq_filter = filter[1]/f_nyq #normalized freqs
	order_filter = filter[2]
	if type_filter == 'lowpass':
		f_lowpass = freq_filter
		b, a = signal.butter(order_filter, f_lowpass)
		x_demod = signal.filtfilt(b, a, x_rect)
	else:
		x_demod = x_rect
		print('No Filter Lowpass')
	
	return x_demod

def hilbert_demodulation(x):		
	x_ana = signal.hilbert(x)
	x_demod = np.abs(x_ana)
	return x_demod
	
def mixed_demodulation(x, fs, prefilter, type_rect=None, dc_value=None):
	n = len(x)
	f_nyq = 0.5*fs
	
	#Defaults
	if not dc_value:
		dc_value = 'without_dc'
	
	if not type_rect:
		type_rect = 'only_positives'
	
	if not prefilter:
		prefilter = ['none', 0, 0]
	
	#Filter
	type_prefilter = prefilter[0]
	freq_prefilter = prefilter[1] #normalized freqs
	order_prefilter = prefilter[2]
	if type_prefilter == 'highpass':
		freq_highpass = freq_prefilter/f_nyq
		b, a = signal.butter(order_prefilter, freq_highpass, btype='highpass')
		x_prefilter = signal.filtfilt(b, a, x)
	elif type_prefilter == 'bandpass':
		freqs_bandpass = [freq_prefilter[0]/f_nyq, freq_prefilter[1]/f_nyq]
		b, a = signal.butter(order_prefilter, freqs_bandpass, btype='bandpass')
		x_prefilter = signal.filtfilt(b, a, x)
	else:
		print('Info: Demodulation without pre-filter')
		x_prefilter = x

	# #Rectification
	# if type_rect == 'only_positives':
		# x_rect = np.zeros(n)
		# for i in range(n):
			# if x_prefilter[i] > 0:
				# x_rect[i] = x_prefilter[i]
	# elif type_rect == 'absolute_value':
		# x_rect = np.abs(x_prefilter)
	# else:
		# print('Info: Demodulation without rectification')
		# x_rect = x_prefilter

	# #DC Value		
	# if dc_value == 'without_dc':
		# x_rect = x_rect - np.mean(x_rect)
	# else: 
		# print('Info: Demodulation without DC value elimination')
	x_rect = x_prefilter
	#Hilbert
	x_demod = hilbert_demodulation(x_rect)
	
	return x_demod