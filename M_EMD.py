
#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import sys
from os import chdir
from os.path import join, isdir, basename, dirname, isfile
from os import listdir
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))
sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from argparse import ArgumentParser
import pandas as pd



import math
import time
import scipy

#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'channel', 'file']


InputsOpt_Defaults = {'name':'auto', 'plot':'OFF', 'n_imf':10, 'min_iter':500, 'max_iter':8000, 's_number':2, 'tolerance':1., 'filter':['OFF'], 'fs':1.e6, 'range':None}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	if config['mode'] == 'emd_from_raw':
		if config['file'] == 'OFF':
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()			
		else:		
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			filepath = config['file']	
		
		config['start_time'] = time.time()
		
		try:
			x = load_signal(filepath, channel=config['channel'])
		except:
			x = f_open_mat(filepath, channel=config['channel'])
		
		if config['range'] != None:
			x = x[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		
		
		dt = 1./config['fs']
		n = len(x)
		t = np.array([i*dt for i in range(n)])
		
		filename = basename(filepath)
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			x = butter_filter(x, config['fs'], config['filter'])	
			
		for count in range(config['n_imf']):		
			count += 1				
			print('To calculate: h' + str(count))
			name_out = 'h' + str(count) + '_'			
			
			h = sifting_iteration_s_corr(t, x, config)
		
			# file = basename(file)
		
			print('Saving...')
			myname = name_out + filename[:-4] + '.pkl'
			save_pickle(myname, h)

			# print("--- %s seconds ---" % (time.time() - start_time))	
		
			x = x - h
		
		config['end_time'] = time.time()
		config['duration_s'] = config['end_time'] - config['start_time']
		save_pickle('configEMD_' + filename[:-4] + '.pkl', config)
		
	else:
		print('unknown mode!!')
		
		
		
	return

def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element' or element == 'filter' or element == 'range':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float
	# if config['power2'] != 'auto' and config['power2'] != 'OFF':
		# config['power2'] = int(config['power2'])
	# config['mode'] = float(config['fs_tacho'])
	config['fs'] = float(config['fs'])
	# config['n_files'] = int(config['n_files'])
	config['n_imf'] = int(config['n_imf'])
	config['min_iter'] = int(config['min_iter'])
	# config['highpass'] = float(config['highpass'])
	config['max_iter'] = int(config['max_iter'])
	# config['time_segments'] = float(config['time_segments'])
	config['tolerance'] = float(config['tolerance'])
	config['s_number'] = int(config['s_number'])
	
	
	if config['filter'][0] != 'OFF':
		if config['filter'][0] == 'bandpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2]), float(config['filter'][3])]
		elif config['filter'][0] == 'highpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		elif config['filter'][0] == 'lowpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		else:
			print('error filter 87965')
			sys.exit()
	
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config

def sifting_iteration_s_corr(t, x, config):
	cont = 0
	error = 5000000
	s_iter = 10
	while (error > config['tolerance'] or s_iter < config['s_number']):
		# start_time = time.time()
		print('+++++++++++++++++++++++++++++++++++iteration ', cont)
		# print("--- %s seconds Inicia sifting2 ---" % (time.time() - start_time))
		h1, extrema_x = sifting2(t, x)
		# print("--- %s seconds Termina sifting2 ---" % (time.time() - start_time))
		if cont > config['min_iter']:
			# pre_time = time.time()
			error = dif_extrema_xzeros(h1)
			# print('dif_extrema_xzeros: ', time.time() - pre_time)
			print(error)
			if error <= config['tolerance']:
				s_iter = s_iter + 1
			else:
				s_iter = 0
			print('Conseq. it = ', s_iter)
		x = h1
		cont = cont + 1
		if cont > config['max_iter']:
			break
		if s_iter > 15:
			break
	return h1

def sifting2(t, x):
	# pre_time = time.time()	
	t_up, x_up = env_up(t, x)
	# print('env_up: ', time.time() - pre_time)
	t_down, x_down = env_down(t, x)
	extrema_x = len(x_up) + len(x_down)
	
	# pre_time = time.time()	
	tck = interpolate.splrep(t_up, x_up)
	x_up = interpolate.splev(t, tck)
	# print('spline_up: ', time.time() - pre_time)
	tck = interpolate.splrep(t_down, x_down)
	x_down = interpolate.splev(t, tck)

	x_mean = (x_up + x_down)/2
	h = x - x_mean
	return h, extrema_x

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


if __name__ == '__main__':
	main(sys.argv)
