import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math

from m_fft import *
from m_demodulation import *


def sort_X_based_on_Y(X, Y):
	return [x for _,x in sorted(zip(Y,X))]

def movil_avg(feature, movil_avg):
	print('yuhuuuuuuuuuuuuuuuuuuuuuuuuu')
	if movil_avg != 0:
		print('WITH AVERAGE MOVIL')
		for i in range(len(feature)):
			if i >= movil_avg:
				count = 0
				for k in range(movil_avg):
					count = count + feature[i-k]
				feature[i] = count / movil_avg
		
		for i in range(movil_avg+1):
			# feature[i] = (feature[i] + feature[movil_avg] + feature[movil_avg+1] + feature[movil_avg+2])/4.
			count = 0
			for k in range(movil_avg):
				count = count + feature[movil_avg+k-i]
			feature[movil_avg - i] = count / movil_avg
				
	return feature

def sum_values_dict(max_Freq_Dict):
	sum = 0
	for key in max_Freq_Dict:
		# print(max_Freq_Dict[key])
		sum += max_Freq_Dict[key]
	return sum

def multi_sum_values_dict(mydict):
	# myarray = np.zeros(len(mydict[mydict.keys()[0]]))
	myarray = np.zeros(len(mydict[list(mydict.keys())[0]]))
	# sys.exit()

	for key in mydict:
		# print(mydict[key])
		myarray += np.array(mydict[key])
	return list(myarray)

def add_sum_dict(mydict):
	for key in mydict:

		sum = np.sum(np.array(mydict[key]))
		mydict[key].append(sum)
	return mydict

def add_mean_dict(mydict):
	for key in mydict:
		mylist = [element for element in mydict[key] if isinstance(element, float) is True]
		
		sum = np.sum(np.array(mylist))
		n = len(mylist)
		mydict[key].append(sum/n)
	return mydict





def obtain_syn_avg(signal, tacho, times_pulses, config):

	n_segments = len(times_pulses) - 1
	print('!!')
	print(n_segments)
	Synchro_Segments = []
	for k in range(n_segments):
		to_append = signal[int(np.rint(times_pulses[k]*config['fs_signal'])) : int(np.rint(times_pulses[k+1]*config['fs_signal']))]
		Synchro_Segments.append(to_append)
		if k == 0:
			min_it = len(to_append)
		if len(to_append) < min_it:
			min_it = len(to_append)
	print(min_it)
	
	# print(Synchro_Segments)
	
	Synchro_Segments_Eq = []
	signal_avg = np.zeros(min_it)
	for k in range(n_segments):
		segment = Synchro_Segments[k]
		new_x = np.linspace(0., 1., num=min_it)
		old_x = np.linspace(0., 1., num=len(segment))
		
		# print(new_x, old_x, segment)
		# print(old_x)
		# print(segment)
		
		segment_eq = np.interp(new_x, old_x, segment)
		Synchro_Segments_Eq.append(segment_eq)
		signal_avg = signal_avg + segment_eq
	signal_avg = signal_avg / n_segments
	
	# t_avg = np.array([i/fs_signal for i in range(len(signal_avg))])
	
	return signal_avg, Synchro_Segments_Eq

def autocorr_fft(x):
	y = np.fft.fft(x)
	y_conj = np.conj(y)
	y_mult = y*y_conj
	y_inv = np.fft.ifft(y_mult)
	
	return y_inv
	
def xcorr_fft(x, y):
	x_fft = np.fft.fft(x)
	y_fft = np.fft.fft(y)
	
	y_fft_conj = np.conj(y_fft)
	
	mult = x_fft*y_fft_conj
	
	mult_inv = np.fft.ifft(mult)
	
	return mult_inv


def xcorrelation_sum(array1, array2, point):#
	n = min(len(array1), len(array2))
	sum = 0.0	
	for j in range(n):
		if (((j+point)>= 0) and ((j+point)<n)):
			sum = sum + (array1[j])*array2[j+point] #
	return sum

def xcorrelation_same(array1, array2, point):
	n = len(array1)
	xcorr = np.correlate(array1, array2, mode='same')
	return xcorr[int(n/2.0) - point]

def xcorrelation_full(array1, array2, point):
	n = len(array1)
	xcorr = np.correlate(array1, array2, mode='full')
	return xcorr[n - point]

def autocorrelation2p_sum(array, point1, point2):#
	n = len(array)
	sum = 0.0	
	for j in range(n):
		if (((j-point1)>= 0) and ((j-point2)>= 0) and ((j-point1)<n) and ((j-point2)<n)):
			sum = sum + (array[j-point1])*array[j-point2]
	return sum

def convolution_sum(array1, array2, point): #
	n = min(len(array1), len(array2))
	sum = 0.0	
	for k in range(n):
		if (((point-k)>= 0) and ((point-k)<n)):
			sum = sum + (array1[k])*array2[point-k]
	return sum

def diff_signal(x, length_diff):
	#Differentiation/Derivative
	n = len(x)
	dx = np.zeros(n-length_diff)
	n_diff = len(dx)
	for i in range(n_diff):
		dx[i] = x[i+length_diff] - x[i]

	return dx

def monotonicity(x):
	#Differentiation/Derivative
	dx = diff_signal(x, 1)
	n = len(dx)	
	ones = 0.
	Mones = 0.
	zeros = 0.
	for elem in dx:
		if elem > 0.:
			ones += 1
		elif elem < 0.:
			Mones += 1
		else:
			zeros += 1
	mono = ones - Mones	
	return mono/n

def monotonicity_poly(x, n_poly):
	import matplotlib.pyplot as plt
	#Differentiation/Derivative
	
	
	t = np.arange(len(x))
	z = np.polyfit(t, x, n_poly)
	p = np.poly1d(z)
	x_poly = p(t)
	# plt.plot(t, x_poly, 'r')
	# plt.plot(t, x, 'b')
	# plt.show()
	
	dx = diff_signal(x_poly, 1)
	n = len(dx)	
	ones = 0.
	Mones = 0.
	zeros = 0.
	for elem in dx:
		if elem > 0.:
			ones += 1
		elif elem < 0.:
			Mones += 1
		else:
			zeros += 1
	mono = ones - Mones	
	return mono/n

def diff_end_thesis(x):	
	xend = x[-1]	
	center = x[1:-1]
	# print('end xxxxxxxxx xend', xend)
	# print('end xxxxxxxxx center', center)
	mean = np.mean(center)	
	# value = xend / mean	
	value = (xend-mean) / mean	
	return value

def diff_start_thesis(x):	
	xstart = x[0]	
	center = x[1:-1]
	# print('start yyyyyy xstart', xstart)
	# print('start yyyyyy center', center)
	mean = np.mean(center)	
	value = (mean-xstart) / xstart	
	return value
		
def deriv_signal(x, fs, length_diff):
	#Differentiation/Derivative
	n = len(x)
	dt = 1.0/fs
	dx = np.zeros(n-length_diff)
	n_diff = len(dx)
	for i in range(n_diff):
		dx[i] = x[i+length_diff] - x[i]
	dxdt = dx/(length_diff*dt)	
	return dxdt

def env_up_data(x):
	n = len(x)
	x_up = []
	x_up.append(x[0])
	for i in range(n-2):
		if (x[i+1] > x[i] and x[i+2] < x[i+1]):
			x_up.append(x[i+1])
	x_up.append(x[n-1])
	x_up = np.array(x_up)
	
	return x_up

def env_up(t, x):
	n = len(x)
	x_up = []
	t_up = []
	x_up.append(x[0])
	t_up.append(t[0])
	for i in range(n-2):
		if (x[i+1] > x[i] and x[i+2] < x[i+1]):
			x_up.append(x[i+1])
			t_up.append(t[i+1])
	x_up.append(x[n-1])
	t_up.append(t[n-1])
	x_up = np.array(x_up)
	t_up = np.array(t_up)
	
	return t_up, x_up

	
def diff_signal_eq(x, length_diff):
	#Differentiation/Derivative
	n = len(x)
	if length_diff != 'auto':

		dx = np.zeros(n-length_diff)
		n_diff = len(dx)
		for i in range(n_diff):
			dx[i] = x[i+length_diff] - x[i]
		t_diff = np.linspace(0, n_diff-1, num=n_diff)
		t_signal = np.linspace(0, n_diff-1, num=n)
	else:
		temp = 0
		ind = 0
		for length_diff in range(50):
			dx = np.zeros(n-length_diff)
			n_diff = len(dx)
			for i in range(n_diff):
				dx[i] = x[i+length_diff] - x[i]
			max = np.max(dx)
			if max >= temp:
				ind = length_diff
				temp = max
		
		length_diff = ind
		dx = np.zeros(n-length_diff)
		n_diff = len(dx)
		for i in range(n_diff):
			dx[i] = x[i+length_diff] - x[i]		
		t_diff = np.linspace(0, n_diff-1, num=n_diff)
		t_signal = np.linspace(0, n_diff-1, num=n)
	
	
	
	
	# print(length_diff)
	# print(len(dx))
	dx_eq = np.interp(t_signal, t_diff, dx)
	# print(len(dx_eq))
	# print(len(x))
	# a = input('pause--')
	return dx_eq

def lower_to_up(Feature, Feature2):
	n_f = len(Feature)
	n_f2 = len(Feature2)
	if n_f2 > n_f:
		xold = np.linspace(0., 1., n_f)
		xnew = np.linspace(0., 1., n_f2)				
		Feature = np.interp(x=xnew, xp=xold, fp=np.array(Feature))
	elif n_f > n_f2:
		xold = np.linspace(0., 1., n_f2)
		xnew = np.linspace(0., 1., n_f)				
		Feature2 = np.interp(x=xnew, xp=xold, fp=np.array(Feature2))
	return Feature, Feature2

# def up_to_low(Feature, Feature2):
	# n_f = len(Feature)
	# n_f2 = len(Feature2)
	# if n_f2 > n_f:
		# xold = np.linspace(0., 1., n_f)
		# xnew = np.linspace(0., 1., n_f2)				
		# Feature = np.interp(x=xnew, xp=xold, fp=np.array(Feature))
	# elif n_f > n_f2:
		# xold = np.linspace(0., 1., n_f2)
		# xnew = np.linspace(0., 1., n_f)				
		# Feature2 = np.interp(x=xnew, xp=xold, fp=np.array(Feature2))
	# return Feature, Feature2


def deriv_signal_eq(x, fs, length_diff):
	#Differentiation/Derivative
	n = len(x)
	dt = 1.0/fs
	dx = np.zeros(n-length_diff)
	n_diff = len(dx)
	for i in range(n_diff):
		dx[i] = x[i+length_diff] - x[i]
	dxdt = dx/(length_diff*dt)
	t_diff = np.linspace(0, n_diff-1, num=n_diff)
	t_signal = np.linspace(0, n_diff-1, num=n)
	dxdt_eq = np.interp(t_signal, t_diff, dxdt)	
	return dxdt_eq

def max_2power(n_x):
	print(n_x)
	aux = 10.0
	count = 0
	while aux > 1.0:
		count = count + 1
		aux = n_x / 2.0**count
	count = count - 1
	# n__max2power = 2**count
	power2 = count
	return power2

def to_dBAE(signal, amp_factor):
	signal_dBAE = np.zeros(len(signal))
	for i in range(len(signal)):
		current_input_V = np.absolute(signal[i]/(10.**(amp_factor/20.)))
		signal_dBAE[i] = 20*np.log10(current_input_V/1.e-6)
	signal_dBAE = signal_dBAE.tolist()
	return signal_dBAE

def under_sampling(x, factor):

	ind_old = [i for i in range(len(x))]
	ind_new = [factor*i for i in range(int(len(x)/factor))]
	
	
	x = np.interp(ind_new, ind_old, x)
	# t = np.interp(ind_new, ind_old, t)
	
	return x
	
def signal_processing(x, config):
	processing = config['processing']
	if processing == 'demod_hilbert':
		x = hilbert_demodulation(x)
	elif processing == 'times_demod_hilbert':
		x = hilbert_demodulation(x)*x
	elif processing == 'square':
		x = x**2.0
	elif processing == 'df_log':
		x = np.log10(1. + np.absolute(x))
	elif processing == 'butter_demod':
		x = butter_demodulation(x=x, fs=config['fs'], filter=config['demod_filter'], prefilter=config['demod_prefilter'], type_rect=config['demod_rect'], dc_value=config['demod_dc'])
	else:
		print('unknown processing')
		sys.exit()
	return x

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
	if prefilter[0] != 'OFF':
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
			# sys.exit()
	else:
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
	elif type_filter == 'highpass':
		f_highpass = freq_filter
		b, a = signal.butter(order_filter, f_highpass, btype='highpass')
		x_demod = signal.filtfilt(b, a, x_rect)
	else:
		x_demod = x_rect
		print('No Filter Lowpass')
	
	return x_demod