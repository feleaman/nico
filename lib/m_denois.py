import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math
from m_fft import *
from m_open_extension import *

from m_kurtogram3 import Fast_Kurtogram_filters
from os.path import join, isdir, basename, dirname, isfile
from os import listdir
# def mag_fft(x, fs):
	# fftx = np.fft.fft(x)
	# fftx = np.abs(fftx)/len(fftx)
	# fftx = 2*fftx[0:int(len(fftx)/2)]
	# tr = len(x)/fs
	# df = 1.0/tr
	# f = np.array([i*df for i in range(len(fftx))])
	# magX = fftx
	# return magX, f, df

def fourier_filter(x, fs, type, freqs):
	n_x = len(x)
	dt = 1.0/fs
	magX, f, df = mag_fft(x, fs)
	phaX = np.angle(np.fft.fft(x))
	phaX = phaX[0:int(len(phaX)/2)]
	n_magX = len(magX)
	x_filt = np.zeros(n_x)
	
	if type == 'bandpass':
		f_low = freqs[0]
		f_high = freqs[1]
		n_harmonics = int((f_high - f_low)/df)
		
		for i in range(n_x):
			print(i/n_x*100)
			sum = 0
			for j in range(n_harmonics):
				sum = sum + magX[int(j+int(f_low/df))]*np.cos(2*math.pi*(j*df+f_low)*(i*dt) + phaX[int(j+int(f_low/df))])
			x_filt[i] = sum
	
	elif type == 'one_component':
		frec = freqs
		n_harmonics = 1
		
		for i in range(n_x):
			print(i/n_x*100)
			sum = 0
			for j in range(n_harmonics):
				sum = sum + magX[int(j+int(frec/df))]*np.cos(2*math.pi*(j*df+frec)*(i*dt) + phaX[int(j+int(frec/df))])
			x_filt[i] = sum
			
	elif type == 'lowpass':
		f_lowcut = freqs
		n_harmonics = int((f_lowcut)/df)
		
		for i in range(n_x):
			sum = 0
			for j in range(n_harmonics):
				sum = sum + magX[j]*np.cos(2*math.pi*(j*df)*(i*dt) + phaX[j])
			x_filt[i] = sum
			
	elif type == 'highpass':
		f_highpass = freqs
		f_max = n_magX*df
		n_harmonics = int((f_max - f_highpass)/df)
		
		for i in range(n_x):
			
			sum = 0
			for j in range(n_harmonics):
				sum = sum + magX[j+f_highpass/df]*np.cos(2*math.pi*(j*df+f_highpass)*(i*dt) + phaX[j+f_highpass/df])
			x_filt[i] = sum
	
	elif type == 'nobandpass':
		f_low = freqs[0]
		f_high = freqs[1]
		n_harmonics = n_magX
		
		for i in range(n_x):
			sum = 0
			for j in range(n_harmonics):
				if ((j*df > f_high) and (j*df < f_low)):
					sum = sum + magX[j]*np.cos(2*math.pi*(j*df)*(i*dt) + phaX[j])
			x_filt[i] = sum
		
	else:
		print('Filter type error')
	return x_filt
	
	

def butter_bandpass(x, fs, freqs, order, warm_points=None):
	f_nyq = 0.5*fs
	
	
	#Pre-filter
	freqs_bandpass = [freqs[0]/f_nyq, freqs[1]/f_nyq]
	b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
	x_filt = signal.filtfilt(b, a, x)
	
	if warm_points != None:
		x_filt = x_filt[warm_points:]
	
	return x_filt

def butter_bandstop(x, fs, freqs, order, warm_points=None):
	f_nyq = 0.5*fs
	
	
	#Pre-filter
	freqs_bandstop = [freqs[0]/f_nyq, freqs[1]/f_nyq]
	b, a = signal.butter(order, freqs_bandstop, btype='bandstop')
	x_filt = signal.filtfilt(b, a, x)
	
	if warm_points != None:
		x_filt = x_filt[warm_points:]
	
	return x_filt


def butter_lowpass(x, fs, freq, order, warm_points=None):
	f_nyq = 0.5*fs
	
	
	# #Lowpass
	# type_filter = filter[0]
	# freq_filter = filter[1]/f_nyq #normalized freqs
	# order_filter = filter[2]
	# if type_filter == 'lowpass':
		# f_lowpass = freq_filter
		# b, a = signal.butter(order_filter, f_lowpass)
		# x_demod = signal.filtfilt(b, a, x_rect)
	
	
	#Pre-filter
	freq = freq/f_nyq
	b, a = signal.butter(order, freq, btype='lowpass')
	x_filt = signal.filtfilt(b, a, x)
	
	if warm_points != None:
		x_filt = x_filt[warm:]
	
	return x_filt

def butter_highpass(x, fs, freq, order, warm_points=None):
	f_nyq = 0.5*fs
	
	freq = freq/f_nyq
	b, a = signal.butter(order, freq, btype='highpass')

	x_filt = signal.filtfilt(b, a, x)
	
	if warm_points != None:
		x_filt = x_filt[warm:]
	
	return x_filt

def median_filter(data, points, same_length):
	n = len(data)
	new_n = n + 1 - points
	new_data = np.ones(new_n)
	for i in range(new_n):
		short_array = data[i:i+points]
		new_data[i] = np.median(short_array)
	if same_length == True:
		print('Same length median filter ON')
		new_data = np.interp(x=np.linspace(0, new_n-1, num=n), xp=np.linspace(0, new_n-1, num=new_n), fp=new_data)
	return new_data

def	signal_denois(x, denois, med_kernel=None, fs=None, freq=None, order=None, warm_points=None):
	if denois == 'median':
		x = scipy.signal.medfilt(x, kernel_size=med_kernel)
	elif denois == 'highpass':
		x = butter_highpass(x, fs, freq, order, warm_points=None)
	else:
		print('unknown denois')
		sys.exit()
	return x

def multi_filter(x, config, filename=None):
	if config['filter'] == 'butter_hp':
		print('Highpass Butter Filter')
		x = butter_highpass(x=x, fs=config['fs'], freq=config['freq_hp'], order=3, warm_points=None)
	elif config['filter'] == 'butter_bp':
		print('Bandpass Butter Filter')
		x = butter_bandpass(x=x, fs=config['fs'], freqs=[config['freq_lp'], config['freq_hp']], order=3, warm_points=None)
	elif config['filter'] == 'butter_lp':
		print('Lowpass Butter Filter')
		x = butter_lowpass(x=x, fs=config['fs'], freq=config['freq_lp'], order=3, warm_points=None)
	elif config['filter'] == 'kurtogram':
		print('Kurtogram Filter')
		lp, hp, max_kurt = Fast_Kurtogram_filters(x, config['level'], config['fs'])
		if hp >= config['fs']/2:
			print('Warning!!!! hp == nyquist')
			hp = np.floor(hp) - 1.
		else:
			print('no funcionaaaaaaaaaaaaa')
			print(hp)
			print(config['fs']/2)
			
		if lp == 0.:
			print('Warning!!!! lp == 0')
			lp = lp + 1.
		
		x = butter_bandpass(x=x, fs=config['fs'], freqs=[lp, hp], order=3, warm_points=None)
	elif config['filter'] == 'bp_from_pkl':
		print('Bandpass Filter from PKL')
		
		Filepaths_Filter = [join(config['filter_path'], f) for f in listdir(config['filter_path']) if isfile(join(config['filter_path'], f)) if f[-3:] == 'pkl']
		count = 0
		for filepath_filter in Filepaths_Filter:
			# print(count)
			filename_filter = basename(filepath_filter)
			
			# print(filename_filter)
			# print(filename[:-5])
			# a = input('...')
			
			if filename_filter.find(filename[:-5]) != -1:
				pik = read_pickle(filepath_filter)
				print(pik)
				lp = pik[0][0] - pik[0][1]/2.
				hp = pik[0][0] + pik[0][1]/2.
				print(count)
				break			
			count += 1
			
		x = butter_bandpass(x=x, fs=config['fs'], freqs=[lp, hp], order=3, warm_points=None)
	else:
		print('Error assignment denois')
		sys.exit()
			
	return x

def	butter_filter(x, fs, filter):
	if filter[0] == 'highpass':
		x = butter_highpass(x=x, fs=fs, freq=filter[1], order=filter[2], warm_points=None)
	
	elif filter[0] == 'lowpass':
		x = butter_lowpass(x=x, fs=fs, freq=filter[1], order=filter[2], warm_points=None)
	
	elif filter[0] == 'bandpass':
		x = butter_bandpass(x=x, fs=fs, freqs=[filter[1], filter[2]], order=filter[3], warm_points=None)
	
	else:
		print('unknown denois')
		sys.exit()
	return x