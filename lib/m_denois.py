import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math
from nico.m_fft import *


def fourier_filter(x, fs, type, freqs):
	n_x = len(x)
	dt = 1.0/fs
	magX, f, df = mag_fft(x, fs)
	phaX = np.angle(np.fft.fft(x))
	phaX = phaX[0:len(phaX)/2]
	n_magX = len(magX)
	x_filt = np.zeros(n_x)
	
	if type == 'bandpass':
		f_low = freqs[0]
		f_high = freqs[1]
		n_harmonics = int((f_high - f_low)/df)
		
		for i in range(n_x):
			sum = 0
			for j in range(n_harmonics):
				sum = sum + magX[j+f_low/df]*np.cos(2*math.pi*(j*df+f_low)*(i*dt) + phaX[j+f_low/df])
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
	