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
from Main_AnalysisEMD import invoke_signal
from m_denois import butter_highpass

from argparse import ArgumentParser
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.signal import argrelextrema
Inputs = ['channel', 'power2', 'fs', 'n_imf']
Inputs_opt = ['file', 'min_iter', 'max_iter', 's_number', 'tolerance', 'plot', 'save', 'file_h1', 'file_h2']
Defaults = [None, 0, 500000, 2, 0.2, 'OFF', 'ON', None, None]

def main(argv):
	config_input = read_parser(argv, Inputs, Inputs_opt, Defaults)
	
	file = config_input['file']
	if file == None:
		root = Tk()
		root.withdraw()
		root.update()
		file = filedialog.askopenfilename()
		config_input['file'] = file
		root.destroy()
	
	channel = config_input['channel']
	power2 = config_input['power2']
	save = config_input['save']
	
	min_iter = int(config_input['min_iter'])
	max_iter = int(config_input['max_iter'])
	s_number = int(config_input['s_number'])
	tolerance = float(config_input['tolerance'])


	dict_int = invoke_signal(config_input)
	x = dict_int['signal']
	filename = dict_int['filename']
	
	

	# if power2 == 'auto':
		# n_points = 2**(max_2power(len(x)))
	# elif power2 == 'OFF':
		# n_points = len(x)
	# else:
		# n_points = 2**power2
	
	x = x[0:int(len(x)/2)]
	
	dt = 1./config_input['fs']
	n = len(x)
	t = np.array([i*dt for i in range(n)])
	
	
	
	
	
	for count in range(config_input['n_imf']):
		if config_input['file_h1'] != None and config_input['file_h2'] == None:
			count += 2
			print(basename(config_input['file_h1']))	
		
			print('To calculate: h' + str(count))
			name_out = 'h' + str(count) + '_'
			h1 = load_signal(config_input['file_h1'], channel=None)
			x = x - h1
			
		elif config_input['file_h1'] != None and config_input['file_h2'] != None:
			count += 3	
		
			print('To calculate: h' + str(count))
			name_out = 'h' + str(count) + '_'
			h1 = load_signal(config_input['file_h1'], channel=None)
			h2 = load_signal(config_input['file_h2'], channel=None)
			x = x - h1 - h2
		
		else:
			count += 1	
		
			print('To calculate: h' + str(count))
			name_out = 'h' + str(count) + '_'
		
		
		
		
		print("--- %s seconds Inicia Sifting ---" % (time.time() - start_time))
		
		# h = sifting_iteration_sd(t, x, min_iter, max_iter, tolerance)
		
		
		# h = sifting_iteration_ez_opt(t, x, min_iter, max_iter, s_number, tolerance)
		h = sifting_iteration_s_corr(t, x, min_iter, max_iter, s_number, tolerance)
	
		file = basename(file)
	
		print('Saving...')
		myname = name_out + file[:-4] + '.pkl'
		save_pickle(myname, h)

		print("--- %s seconds ---" % (time.time() - start_time))	
	
		x = x - h
	
	
	
	
	
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
	config_input['n_imf'] = int(config_input['n_imf'])
	
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
		# start_time = time.time()
		# print('+++++++++++++++++++++++++++++++++++iteration ', cont)
		# print("--- %s seconds Inicia sifting2 ---" % (time.time() - start_time))
		h1, extrema_x = sifting2(t, x)
		# print("--- %s seconds Termina sifting2 ---" % (time.time() - start_time))
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
		# print("--- %s seconds Inicia chequeo del residuo ---" % (time.time() - start_time))
	return h1

def sifting_iteration_s_corr(t, x, min_iter, max_iter, s_number, tolerance):
	
	# s_number = 2
	# tolerance = 2	
	# max_iter = 5
	# min_iter = 1
	cont = 0
	error = 5000000
	s_iter = 10
	while (error > tolerance or s_iter < s_number):
		# start_time = time.time()
		print('+++++++++++++++++++++++++++++++++++iteration ', cont)
		# print("--- %s seconds Inicia sifting2 ---" % (time.time() - start_time))
		h1, extrema_x = sifting2(t, x)
		# print("--- %s seconds Termina sifting2 ---" % (time.time() - start_time))
		if cont > min_iter:
			# n_extrema = extrema(h1)
			# n_xzeros = xzeros(h1)
			# error = np.absolute(n_extrema - n_xzeros)
			error = dif_extrema_xzeros(h1)

			print(error)
			if error <= tolerance:
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
		# print("--- %s seconds Inicia chequeo del residuo ---" % (time.time() - start_time))
	return h1

	

def sifting_iteration_ez_opt(t, x, min_iter, max_iter, s_number, tolerance):
	# s_number = 2
	# tolerance = 2	
	# max_iter = 5
	# min_iter = 1
	cont = 0
	error = 5000000
	s_iter = 10
	while (error > tolerance or s_iter < s_number):
		print('++++++++iteration ', cont)
		h1, extrema_x = sifting2_lin_opt(t, x)
		if cont > min_iter:
			n_extrema_h1, n_xzeros_h1 = extrema_xzeros(h1)
			
			error_extrema = n_extrema_h1 - extrema_x
			error_xzeros = n_xzeros_h1 - xzeros(x)
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
	sd = 5.e17
	while (sd > tolerance):
		print('++++++++iteration ', cont)
		h1, extrema_x = sifting2_argrel(t, x)
		if cont > min_iter:
			sum = 0.
			# for element in x:
				# if element**2.0 == 0.:
					# print('value:', element)
					# print(len(h1))
					# print(len(x))
					# a = input('pause')
					
			
			for i in range(len(h1)):
				if x[i] != 0.:
					sum = sum + ((x[i] - h1[i])**2.0)/((x[i])**2.0)
				
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

def dif_extrema_xzeros(x):
	n = len(x)
	n_xzeros = 0
	n_extrema = 0
	
	for i in range(n-2):
		if (x[i+1] < x[i] and x[i+2] > x[i+1]) or (x[i+1] > x[i] and x[i+2] < x[i+1]):
			n_extrema = n_extrema + 1
			
		if (x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0):
			n_xzeros = n_xzeros + 1


	return np.absolute(n_extrema - n_xzeros)

	
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

def sifting2_argrel(t, x):

	t_up = argrelmax(x)
	t_up = t_up[0]
	x_up = x[t_up]
	
	t_down = argrelmin(x)
	t_down = t_down[0]
	x_down = x[t_down]
	
	
	extrema_x = len(x_up) + len(x_down)
	
	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h, extrema_x

# def sifting2_cwt(t, x):


	
	# t_up_down = argrelextrema(x)
	# t_up_down = t_up_down[0]
	# x_up_down = x[t_up_down]
	
	# t_up = t_up_down[::2]
	# t_down = t_up_down[1::2]
	
	# x_up = x_up_down[::2]
	# x_down = x_up_down[1::2]

	
	
	# extrema_x = len(x_up) + len(x_down)
	
	# tck = interpolate.splrep(t_up, x_up)
	# x_up = interpolate.splev(t, tck)
	# tck = interpolate.splrep(t_down, x_down)
	# x_down = interpolate.splev(t, tck)

	# x_mean = (x_up + x_down)/2
	# h = x - x_mean
	# return h, extrema_x


	
def sifting2_argrel_lin(t, x):

	t_up = argrelmax(x)
	t_up = t_up[0]
	x_up = x[t_up]
	
	t_down = argrelmin(x)
	t_down = t_down[0]
	x_down = x[t_down]
	
	
	extrema_x = len(x_up) + len(x_down)
	
	x_up = np.interp(t, t_up, x_up)	
	x_down = np.interp(t, t_down, x_down)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h, extrema_x



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



def sifting2_lin(t, x):
	t_up, x_up = env_up(t, x)
	t_down, x_down = env_down(t, x)
	
	# print("--- %s seconds Termina envolventes ---" % (time.time() - start_time))
	extrema_x = len(x_up) + len(x_down)
	
	# tck = interpolate.splrep(t_up, x_up)
	# x_up = interpolate.splev(t, tck)
	
	x_up = np.interp(t, t_up, x_up)	
	x_down = np.interp(t, t_down, x_down)
	# print("--- %s seconds Termina interpolacion ---" % (time.time() - start_time))
	
	# tck = interpolate.splrep(t_down, x_down)
	# x_down = interpolate.splev(t, tck)

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

def sifting2_lin_opt(t, x):
	
	t_up, x_up, t_down, x_down = env_up_down_opt_no_eq(t, x)
	extrema_x = len(x_up) + len(x_down)
	
	# tck = interpolate.splrep(t_up, x_up)
	# x_up = interpolate.splev(t, tck)
	# tck = interpolate.splrep(t_down, x_down)
	# x_down = interpolate.splev(t, tck)
	x_up = np.interp(t, t_up, x_up)	
	x_down = np.interp(t, t_down, x_down)
	

	x_mean = (x_up + x_down)/2.
	h = x - x_mean
	return h, extrema_x


if __name__ == '__main__':
	main(sys.argv)