import numpy as np
import sys
sys.path.insert(0, './lib')
import pickle
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
from scipy import interpolate
import time
start_time = time.time()
from os.path import isfile, join, basename
import scipy.io
import argparse
from m_open_extension import *
from m_demodulation import *

from m_denois import butter_highpass

from argparse import ArgumentParser

Inputs = ['file', 'channel', 'power2', 'save', 'fs']
Inputs_opt = ['file_h1', 'min_iter', 'max_iter', 's_number', 'tolerance', 'file_h2', 'file_h3', 'plot', 'file_h4', 'file_h5', 'file_h6', 'file_h7']
Defaults = [None, 1000, 10000, 2, 3.0, None, None, 'OFF', None, None, None, None]

def main(argv):
	config_input = read_parser(argv, Inputs, Inputs_opt, Defaults)
	
	# path = config_input['path']
	file = config_input['file']
	channel = config_input['channel']
	power2 = config_input['power2']
	save = config_input['save']
	
	file_h1 = config_input['file_h1']
	min_iter = int(config_input['min_iter'])
	max_iter = int(config_input['max_iter'])
	s_number = int(config_input['s_number'])
	tolerance = float(config_input['tolerance'])
	file_h2 = config_input['file_h2']
	file_h3 = config_input['file_h3']
	file_h4 = config_input['file_h4']
	file_h5 = config_input['file_h5']
	file_h6 = config_input['file_h6']
	file_h7 = config_input['file_h7']

	# filepath_x = join(path, file)
	# filename = filepath_x
	# # x = f_open_mat(filepath_x, channel)
	# # x = np.ndarray.flatten(x)
	
	# point_index = filepath_x.find('.')
	# extension = filepath_x[point_index+1] +filepath_x[point_index+2] + filepath_x[point_index+3]
	
	
	# # x = load_signal(filename, channel)
	# if extension == 'mat':
		# # x, channel = f_open_mat_2(filename)
		# # x = f_open_mat(filename, channel)
		# # x = np.ndarray.flatten(x)

		# print(filename)
		# print(channel)
		# x = f_open_mat_2(filename, channel)
		# x = np.ndarray.flatten(x)

	# elif extension == 'tdm': #tdms
		# x = f_open_tdms(filename, channel)
		# # x = f_open_tdms_2(filename)


	# elif extension == 'txt': #tdms
		# x = np.loadtxt(filename)
	# # filename = os.path.basename(filename) #changes from path to file
	# print(filepath_x)
	from Main_Analysis import invoke_signal
	
	
	
	
	dict_int = invoke_signal(config_input)
	x = dict_int['signal']
	filename = dict_int['filename']
	

	
	
	
	
	
	
	
	
	if power2 == 'auto':
		n_points = 2**(max_2power(len(x)))
	elif power2 == 'OFF':
		n_points = len(x)
	else:
		n_points = 2**power2
	
	x = x[0:n_points]
	
	
	# n_points = 2**int(power2)
	# x = x[0:n_points]
	
	if file_h1 != None and file_h2 == None:
		print('To calculate: h2')
		name_out = 'h2_'
		filepath_h1 = file_h1
		h1 = read_pickle(filepath_h1)
		x = x - h1
	elif file_h1 != None and file_h2 != None and file_h3 == None:
		print('To calculate: h3')
		name_out = 'h3_'
		filepath_h1 = file_h1
		h1 = read_pickle(filepath_h1)
		filepath_h2 = file_h2
		h2 = read_pickle(filepath_h2)
		x = x - h1 - h2
	elif file_h1 != None and file_h2 != None and file_h3 != None and file_h4 == None:
		print('To calculate: h4')
		name_out = 'h4_'
		filepath_h1 = file_h1
		h1 = read_pickle(filepath_h1)
		filepath_h2 = file_h2
		h2 = read_pickle(filepath_h2)
		filepath_h3 = file_h3
		h3 = read_pickle(filepath_h3)
		x = x - h1 - h2 - h3
	elif file_h1 != None and file_h2 != None and file_h3 != None and file_h4 != None and file_h5 == None:
		print('To calculate: h5')
		name_out = 'h5_'
		filepath_h1 = file_h1
		h1 = read_pickle(filepath_h1)
		filepath_h2 = file_h2
		h2 = read_pickle(filepath_h2)
		filepath_h3 = file_h3
		h3 = read_pickle(filepath_h3)
		filepath_h4 = file_h4
		h4 = read_pickle(filepath_h4)
		x = x - h1 - h2 - h3 - h4
	elif file_h1 != None and file_h2 != None and file_h3 != None and file_h4 != None and file_h5 != None and file_h6 == None:
		print('To calculate: h6')
		name_out = 'h6_'
		filepath_h1 = file_h1
		h1 = read_pickle(filepath_h1)
		filepath_h2 = file_h2
		h2 = read_pickle(filepath_h2)
		filepath_h3 = file_h3
		h3 = read_pickle(filepath_h3)
		filepath_h4 = file_h4
		h4 = read_pickle(filepath_h4)
		filepath_h5 = file_h5
		h5 = read_pickle(filepath_h5)
		x = x - h1 - h2 - h3 - h4 - h5
	elif file_h1 != None and file_h2 != None and file_h3 != None and file_h4 != None and file_h5 != None and file_h6 != None and file_h7 == None:
		print('To calculate: h7')
		name_out = 'h7_'
		filepath_h1 = file_h1
		h1 = read_pickle(filepath_h1)
		filepath_h2 = file_h2
		h2 = read_pickle(filepath_h2)
		filepath_h3 = file_h3
		h3 = read_pickle(filepath_h3)
		filepath_h4 = file_h4
		h4 = read_pickle(filepath_h4)
		filepath_h5 = file_h5
		h5 = read_pickle(filepath_h5)
		filepath_h6 = file_h6
		h6 = read_pickle(filepath_h6)
		x = x - h1 - h2 - h3 - h4 - h5 - h6
	else:
		print('To calculate: h1')
		name_out = 'h1_'
		
	print('Fs = 1 MHz for AE')
	fs = 1000000.
	dt = 1/fs
	n = len(x)
	t = np.array([i*dt for i in range(n)])

	#++++++++++++++++++++++++++++ENVELOPES AND MEAN
	
	# x = butter_highpass(x=x, fs=1.e6, freq=80.e3, order=3, warm_points=None)
	
	# x = butter_demodulation(x=x, fs=10.e6, filter=['lowpass', 5000., 3], prefilter=['highpass', 80.e3, 3], type_rect='only_positives', dc_value='without_dc')
	# plt.plot(x)
	# plt.show()
	# h1 = sifting_iteration(t, x, min_iter, max_iter, s_number, tolerance)
	h1 = sifting_iteration_sd(t, x, min_iter, max_iter, tolerance)
	# h1 = sifting_iteration_sd_opt(t, x, min_iter, max_iter, tolerance)
	file = basename(file)
	
	if save == 'ON':
		print('Saving...')
		save_pickle(name_out + file[:-4] + '.pkl', h1)
		# np.savetxt(name_out + file[:-4] + '.txt', h1)

	print("--- %s seconds ---" % (time.time() - start_time))
	sys.exit()
	#++++++++++++++++++++++++++++++++COMMENTS


def read_parser(argv, Inputs, Inputs_opt, Defaults):
	parser = argparse.ArgumentParser()	
	for element in (Inputs + Inputs_opt):
		parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config_input = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config_input[element] = value
	
	#conversion
	if config_input['power2'] != 'auto' and config_input['power2'] != 'OFF':
		config_input['power2'] = int(config_input['power2'])
	config_input['min_iter'] = int(config_input['min_iter'])
	config_input['max_iter'] = int(config_input['max_iter'])
	config_input['s_number'] = int(config_input['s_number'])
	config_input['tolerance'] = float(config_input['tolerance'])
	config_input['fs'] = float(config_input['fs'])
	
	return config_input
	

def f_open_mat(filename, channel):
	file = scipy.io.loadmat(filename)
	data = file[channel]
	return data

def sifting_iteration(t, x, min_iter, max_iter, s_number, tolerance):
	# s_number = 2
	# tolerance = 2	
	# max_iter = 5
	# min_iter = 1
	cont = 0
	error = 5000000
	s_iter = 10
	while (error > tolerance or s_iter < s_number):
		print('++++++++iteration ', cont)
		h1, extrema_x = sifting2(t, x)
		if cont > min_iter:
			error_extrema = extrema(h1)-extrema_x
			error_xzeros = xzeros(h1)-xzeros(x)		
			error = error_extrema + error_xzeros

			print(error)
			if error < tolerance:
				s_iter = s_iter + 1
			else:
				s_iter = 0
			print('Conseq. it = ', s_iter)
		x = h1
		cont = cont + 1
		if cont > max_iter:
			break
		if s_iter > 15:
			break
	return h1
	
	
def sifting_iteration_sd(t, x, min_iter, max_iter, tolerance):
	# s_number = 2
	# tolerance = 2	
	# max_iter = 5
	# min_iter = 1
	cont = 0
	sd = 5000000
	while (sd > tolerance):
		print('++++++++iteration ', cont)
		h1, extrema_x = sifting2(t, x)
		if cont > min_iter:
			sum = 0.
			for i in range(len(h1)):
				# print(x[i])
				sum = sum + ((x[i] - h1[i])**2.0)/(1+(x[i])**2.0)
			sd = sum
			print('SD = ', sd)
		x = h1
		cont = cont + 1
		if cont > max_iter:
			break

	return h1

def sifting_iteration_sd_opt(t, x, min_iter, max_iter, tolerance):
	# s_number = 2
	# tolerance = 2	
	# max_iter = 5
	# min_iter = 1
	cont = 0
	sd = 5000000
	while (sd > tolerance):
		print('++++++++iteration ', cont)
		h1, extrema_x = sifting2_opt(t, x)
		if cont > min_iter:
			sum = 0.
			for i in range(len(h1)):
				# print(x[i])
				sum = sum + ((x[i] - h1[i])**2.0)/(1+(x[i])**2.0)
			sd = sum
			print('SD = ', sd)
		x = h1
		cont = cont + 1
		if cont > max_iter:
			break

	return h1

def extrema(x):
	n = len(x)
	n_extrema = 0
	for i in range(n-2):
		if (x[i+1] < x[i] and x[i+2] > x[i+1]):
			n_extrema = n_extrema + 1
	for i in range(n-2):
		if (x[i+1] > x[i] and x[i+2] < x[i+1]):
			n_extrema = n_extrema + 1

	return n_extrema

def xzeros(x):
	n = len(x)
	n_xzeros = 0
	for i in range(n-1):
		if (x[i] > 0 and x[i+1] < 0):
			n_xzeros = n_xzeros + 1
	for i in range(n-1):
		if (x[i] < 0 and x[i+1] > 0):
			n_xzeros = n_xzeros + 1

	return n_xzeros

def extrema_xzeros(x):
	n = len(x)
	n_xzeros = 0
	n_extrema = 0
	
	for i in range(n-2):
		if (x[i+1] < x[i] and x[i+2] > x[i+1]) or (x[i+1] > x[i] and x[i+2] < x[i+1]):
			n_extrema = n_extrema + 1
			
		if (x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0):
			n_xzeros = n_xzeros + 1


	return n_extrema, n_xzeros
	
def env_down(t, x):
	n = len(x)
	x_down = []
	t_down = []
	x_down.append(x[0])
	t_down.append(t[0])
	for i in range(n-2):
		if (x[i+1] < x[i] and x[i+2] > x[i+1]):
			x_down.append(x[i+1])
			t_down.append(t[i+1])
	x_down.append(x[n-1])
	t_down.append(t[n-1])
	x_down = np.array(x_down)
	t_down = np.array(t_down)

	return t_down, x_down


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








def env_up_down_opt(t, x):
	# t = t[0:len(t)-2]
	# t = t[1:]
	px = diff_uno_eq(x)
	px = sign_fun(px)
	px = diff_uno_eq(px)
	# x = np.absolute(x)
	t_peaks_up = []
	amp_peaks_up = []
	t_peaks_down = []
	amp_peaks_down = []
	for i in range(len(px)-1):
		if px[i] < 0:
			t_peaks_up.append(t[i+1])
			amp_peaks_up.append(x[i+1])
		elif px[i] > 0:
			t_peaks_down.append(t[i+1])
			amp_peaks_down.append(x[i+1])
	
	return t_peaks_up, amp_peaks_up, t_peaks_down, amp_peaks_down

def env_up_down_opt_no_eq(t, x):
	# t = t[0:len(t)-2]
	# t = t[1:]
	px = diff_uno(x)
	px = sign_fun(px)
	px = diff_uno(px)
	# x = np.absolute(x)
	t_peaks_up = []
	amp_peaks_up = []
	t_peaks_down = []
	amp_peaks_down = []
	for i in range(len(px)-1):
		if px[i] < 0:
			t_peaks_up.append(t[i+1])
			amp_peaks_up.append(x[i+1])
		elif px[i] > 0:
			t_peaks_down.append(t[i+1])
			amp_peaks_down.append(x[i+1])
	
	return t_peaks_up, amp_peaks_up, t_peaks_down, amp_peaks_down

def sign_fun(x):
	output = np.zeros(len(x))
	for i in range(len(x)):
		if x[i] > 0:
			output[i] = 1
		elif x[i] < 0:
			output[i] = -1
	return output

def diff_uno_eq(x):
	#Differentiation/Derivative
	n = len(x)
	dx = np.zeros(n-1)
	n_diff = len(dx)
	for i in range(n_diff):
		dx[i] = x[i+1] - x[i]
	
	t_diff = np.linspace(0, n_diff-1, num=n_diff)
	t_signal = np.linspace(0, n_diff-1, num=n)
	dx_eq = np.interp(t_signal, t_diff, dx)

	return dx

def diff_uno(x):
	#Differentiation/Derivative
	n = len(x)
	dx = np.zeros(n-1)
	n_diff = len(dx)
	for i in range(n_diff):
		dx[i] = x[i+1] - x[i]


	return dx

def sifting(t, x):
	t_up, x_up = env_up(t, x)
	t_down, x_down = env_down(t, x)

	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h

def sifting2(t, x):
	t_up, x_up = env_up(t, x)
	t_down, x_down = env_down(t, x)
	extrema_x = len(x_up) + len(x_down)
	
	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h, extrema_x

def sifting2_opt(t, x):
	# t_up, x_up = env_up(t, x)
	# t_down, x_down = env_down(t, x)	
	t_up, x_up, t_down, x_down = env_up_down_opt(t, x)
	extrema_x = len(x_up) + len(x_down)
	
	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h, extrema_x

if __name__ == '__main__':
	main(sys.argv)