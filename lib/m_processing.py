import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy
import math

from m_fft import *


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
	
	
	
	
	print(length_diff)
	dx_eq = np.interp(t_signal, t_diff, dx)	
	return dx_eq
	
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

def signal_processing(x, processing):
	if processing == 'demod_hilbert':
		x = hilbert_demodulation(x)
	elif processing == 'times_demod_hilbert':
		x = hilbert_demodulation(x)*x
	elif processing == 'square':
		x = x**2.0
	else:
		print('unknown processing')
		sys.exit()
	return x