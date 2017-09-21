import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
import time
start_time = time.time()
from os.path import isfile, join
import scipy.io
import argparse

Inputs = ['path', 'file_x', 'channel', 'power2', 'save']
Inputs_opt = ['file_h1', 'min_iter', 'max_iter', 's_number', 'tolerance', 'file_h2']
Defaults = [None, 100, 40000, 2, 2, None]

def main(argv):
	config_input = read_parser(argv, Inputs, Inputs_opt, Defaults)
	
	path = config_input['path']
	file_x = config_input['file_x']
	channel = config_input['channel']
	power2 = config_input['power2']
	save = config_input['save']
	
	file_h1 = config_input['file_h1']
	min_iter = int(config_input['min_iter'])
	max_iter = int(config_input['max_iter'])
	s_number = int(config_input['s_number'])
	tolerance = int(config_input['tolerance'])
	file_h2 = config_input['file_h2']

	filepath_x = join(path, file_x)
	x = f_open_mat(filepath_x, channel)
	x = np.ndarray.flatten(x)
	n_points = 2**int(power2)
	x = x[0:n_points]
	
	if file_h1 != None and file_h2 == None:
		print('To calculate: h2')
		name_out = 'h2_'
		filepath_h1 = join(path, file_h1)
		h1 = np.loadtxt(filepath_h1)
		x = x - h1
	elif file_h1 != None and file_h2 != None:
		print('To calculate: h3')
		name_out = 'h3_'
		filepath_h1 = join(path, file_h1)
		h1 = np.loadtxt(filepath_h1)
		filepath_h2 = join(path, file_h2)
		h2 = np.loadtxt(filepath_h2)
		x = x - h1 - h2
	else:
		print('To calculate: h1')
		name_out = 'h1_'
		
	print('Fs = 1 MHz for AE')
	fs = 1000000.0
	dt = 1/fs
	n = len(x)
	t = np.array([i*dt for i in range(n)])

	#++++++++++++++++++++++++++++ENVELOPES AND MEAN
	h1 = sifting_iteration(t, x, min_iter, max_iter, s_number, tolerance)
	
	if save == 'ON':
		print('Saving...')
		np.savetxt(name_out + file_x[:-4] + '.txt', h1)

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



if __name__ == '__main__':
	main(sys.argv)