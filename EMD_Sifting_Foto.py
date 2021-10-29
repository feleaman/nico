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
from os.path import isfile, join, basename, dirname
import scipy.io
import argparse
from m_open_extension import *
from m_demodulation import *
from Main_Analysis import invoke_signal
from m_denois import butter_highpass
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))
plt.rcParams['savefig.dpi'] = 1200
plt.rcParams['savefig.format'] = 'png'

from argparse import ArgumentParser

Inputs = ['channel', 'power2', 'save', 'fs', 'n_imf']
Inputs_opt = ['file', 'min_iter', 'max_iter', 's_number', 'tolerance', 'plot', ]
Defaults = [None, 0, 2, 2, 2.0, 'OFF']

def main(argv):
	config_input = read_parser(argv, Inputs, Inputs_opt, Defaults)
	
	# file = config_input['file']
	# if file == None:
		# root = Tk()
		# root.withdraw()
		# root.update()
		# file = filedialog.askopenfilename()
		# config_input['file'] = file
		# root.destroy()
	
	channel = config_input['channel']
	power2 = config_input['power2']
	save = config_input['save']
	
	min_iter = int(config_input['min_iter'])
	max_iter = int(config_input['max_iter'])
	s_number = int(config_input['s_number'])
	tolerance = float(config_input['tolerance'])


	# dict_int = invoke_signal(config_input)
	# x = dict_int['signal']
	t = np.arange(0, 5., 0.01)
	x = 0.21*np.sin(2*np.pi*3*t) + 0.11*np.sin(2*np.pi*1.25*t+2) + 0.004*2*np.pi*t + 0.03*np.sin(2*np.pi*0.91*t-1)
	
	# filename = dict_int['filename']
	

	# if power2 == 'auto':
		# n_points = 2**(max_2power(len(x)))
	# elif power2 == 'OFF':
		# n_points = len(x)
	# else:
		# n_points = 2**power2
	
	# x = x[0:n_points]
	
	dt = 1./config_input['fs']
	n = len(x)
	t = np.array([i*dt for i in range(n)])
	
	
	for count in range(config_input['n_imf']):
		count += 1		
		
		print('To calculate: h' + str(count))
		name_out = 'h' + str(count) + '_'

		h = sifting_iteration_sd(t, x, min_iter, max_iter, tolerance)
	
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
	
	
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=0.14, top=0.95, right=0.95, left=0.14)
	# tt = np.arange()
	# plt.subplots_adjust(left=0.15)
	ax.plot(t*10000, x, color='k', label= 'signal')
	ax.set_ylim(bottom=-0.4, top=0.8)
	ax.set_xlim(left=0, right=5)
	ax.plot(t*10000, x_mean, color='r', linestyle = '--', label= 'mean envelope')
	ax.plot(t*10000, x_up, color='g', linestyle = '-.', label= 'upper envelope')
	ax.plot(t*10000, x_down, color='b', linestyle = ':', label= 'lower envelope')
	# plt.legend(loc='lower right', fontsize=12)
	plt.legend(loc='upper left', fontsize=12.5+2+2, ncol=2)	
	ax.set_xlabel('Time [s]', fontsize=15+2+3)
	ax.set_ylabel('Amplitude [V]', fontsize=15+2+3)
	ax.tick_params(axis='both', labelsize=13+2+3)	
	plt.show()
	
	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=0.14, top=0.95, right=0.95, left=0.14)
	# plt.subplots_adjust(left=0.15)
	ax.plot(t*10000, x, color='k', label= 'signal')
	# ax.set_ylim(bottom=-0.35, top=0.55)
	# ax.plot(t*10000, x_mean, color='r', linestyle = '--', label= 'mean envelope')
	# ax.plot(t*10000, x_up, color='g', linestyle = '-.', label= 'upper envelope')
	# ax.plot(t*10000, x_down, color='b', linestyle = '-.', label= 'lower envelope')
	plt.legend(loc='lower right', fontsize=12.5+2)	
	ax.set_xlabel('Time [s]', fontsize=15+2)
	ax.set_ylabel('Amplitude [V]', fontsize=15+2)
	ax.tick_params(axis='both', labelsize=13+2)	
	plt.show()
	
	fig, ax = plt.subplots()
	plt.subplots_adjust(bottom=0.14, top=0.95, right=0.95, left=0.14)
	# plt.subplots_adjust(left=0.15)
	ax.plot(t*10000, h, color='darkblue', label= 'h$_{11}$')
	ax.set_ylim(bottom=-0.4, top=0.8)
	ax.set_xlim(left=0, right=5)
	# ax.plot(t*10000, x_mean, color='r', linestyle = '--', label= 'mean envelope')
	# ax.plot(t*10000, x_up, color='g', linestyle = '-.', label= 'upper envelope')
	# ax.plot(t*10000, x_down, color='b', linestyle = '-.', label= 'lower envelope')
	# plt.legend(loc='lower right', fontsize=12)
	plt.legend(loc='upper left', fontsize=12.5+2+2)		
	ax.set_xlabel('Time [s]', fontsize=15+2+3)
	ax.set_ylabel('Amplitude [V]', fontsize=15+2+3)
	ax.tick_params(axis='both', labelsize=13+2+3)	
	plt.show()
	
	sys.exit()
	
	
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