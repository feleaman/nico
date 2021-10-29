# import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))
# from sys import exit
# from sys.path import path.insert
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import numpy as np
# import pandas as pd
from m_open_extension import *
from m_det_features import *

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import RobustScaler  


from Kurtogram3 import Fast_Kurtogram_filters
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'name':'name', 'mypath':None, 'fs':1.e6, 'channel':'AE_0', 'thr_value':100., 'thr_mode':'fixed_value', 'window_time':0.001, 'save_plot':'OFF', 'file':'OFF', 'stella':600, 'lockout':600, 'harm_fc':1., 'fact_red':1000., 'lag':232000, 'side_points':0, 'filter':['OFF'], 'amp_db':'OFF', 'db_pa':37., 'mv':'ON', 'level':5, 'plot':'OFF', 'n_files':1, 'range':None, 'time_reg':0., 'units':'v', 'db_out':'OFF'}
#stella 100 gearbox amt
#lockout 200 gearbox amt



from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'mode1_dev':

		time_reg = 5.
		fs = 1000.
		high_amplitude = 1.13
		spacing = 0.5
		
		x = boolean_burst(time_reg, fs, high_amplitude, spacing)
		
		plt.plot(x, 'bo-')
		print(len(x))
		y = np.array(list(x[3:]) + list(x[0:3]))
		for i in range(int(len(y)/4)):
			y[i] = y[i]*1.
		plt.plot(y, 'rs-')
		print(len(y))
		plt.show()
		print(np.corrcoef(x,y)[0][1])
		print(max_norm_correlation_lag(x, y, lag=5))
	
	elif config['mode'] == 'mode2_dev':

		time_reg = 5.
		fs = config['fs']
		high_amplitude = 1.
		spacing = 0.23077/3.
		
		x, index_x = triangle_burst(time_reg, fs, high_amplitude, spacing)
		
		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = 1000*load_signal(filepath, config['channel'])
		
		t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
		
		index_burst = list(np.array(t_burst_corr)*config['fs'])
		burst = np.zeros(len(signal))
		for element in index_burst:
			burst[int(element)] = 1

		fig, ax = plt.subplots(ncols=1, nrows=2)
		ax[0].plot(x, 'bo-')
		ax[1].plot(burst, 'ro-')
		plt.show()
		
		index_max = np.max(index_burst + index_x)
		print(index_max)
		factor_red = 1000.
		index_max_red = int(index_max/factor_red)
		
		
		index_burst_red = []
		for element in index_burst:
			index_burst_red.append(int(element/factor_red))

		burst_red = np.zeros(index_max_red+1)
		for i in range(index_max_red+1):
			if i in index_burst_red:
				burst_red[i] = 1.
				
				
		index_x_red = []
		for element in index_x:

			index_x_red.append(int(element/factor_red))
		print(index_max_red)
		print(index_x_red)
		# sys.exit()
		
		x_red = np.zeros(index_max_red+1)
		for i in range(index_max_red+1):
			if i in index_x_red:
				x_red[i] = 1.
			
		del(x)
		del(burst)
		del(signal)
		
		fig, ax = plt.subplots(ncols=1, nrows=2)
		ax[0].plot(x_red, 'bs-')
		ax[1].plot(burst_red, 'rs-')
		plt.show()

		lag = int(116000/factor_red/3)
		print(max_norm_correlation_lag(x_red, burst_red, lag=lag))
		

	elif config['mode'] == 'mode3_dev':

		time_reg = 5.
		fs = config['fs']
		high_amplitude = 1.
		spacing = 0.23077/3.
		
		x, index_x = triangle_burst(time_reg, fs, high_amplitude, spacing)
		
		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		signal = 1000*load_signal(filepath, config['channel'])
		
		t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
		index_burst = list(np.array(t_burst_corr)*config['fs'])
		
		burst = boolean_burst(t_burst_corr, config['fs'], len(signal))


		fig, ax = plt.subplots(ncols=1, nrows=2)
		ax[0].plot(x, 'bo-')
		ax[1].plot(burst, 'ro-')
		plt.show()
		
		
		
		index_max = np.max(index_burst + index_x)
		print(index_max)
		factor_red = 1000.
		index_max_red = int(index_max/factor_red)
		
		
		burst_red = reduce_fact(index_burst, factor_red, index_max_red)
		x_red = reduce_fact(index_x, factor_red, index_max_red)

			
		del(x)
		del(burst)
		del(signal)
		
		fig, ax = plt.subplots(ncols=1, nrows=2)
		ax[0].plot(x_red, 'bs-')
		ax[1].plot(burst_red, 'rs-')
		plt.show()

		lag = int(116000/factor_red/3)
		print(max_norm_correlation_lag(x_red, burst_red, lag=lag))
	
	elif config['mode'] == 'mode4_dev':

		time_reg = 5.
		high_amplitude = 1.
		spacing = 0.23077/3.
		factor_red = 1000.
		lag = int(116000/factor_red/3)
		
		x, index_x = triangle_burst(time_reg, config['fs'], high_amplitude, spacing)
		
		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		CORR= []
		count = 0
		for filepath in Filepaths:
		
			signal = 1000*load_signal(filepath, config['channel'])/70.8
			
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			index_burst = list(np.array(t_burst_corr)*config['fs'])
			
			burst = boolean_burst(t_burst_corr, config['fs'], len(signal))

			index_max = np.max(index_burst + index_x)
			
			index_max_red = int(index_max/factor_red)

			burst_red = reduce_fact(index_burst, factor_red, index_max_red)
			x_red = reduce_fact(index_x, factor_red, index_max_red)
			
			if count == 0:
				del(x)
			del(burst)
			del(signal)
			# del(index_x)
			del(index_burst)
			

			count += 1
			
			CORR.append(max_norm_correlation_lag(x_red, burst_red, lag=lag))
		
		row_names = [basename(filepath) for filepath in Filepaths]
		mydict = {}
		mydict['CORR'] = CORR


		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter('CORR_out' + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Corr')	
		print('Result in Excel table')
		
		
	elif config['mode'] == 'mode5':
		print('mode5_dev start running..........')
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		
		
		time_reg = config['time_reg']

		
		
		# high_amplitude = 1.
		harm_fc = config['harm_fc']
		
		if Filepaths[0].find('20181102') != -1: #medicion Q
			spacing = 0.225564/harm_fc
			print('medicion Q!!!!!!!!!!!!!!!!!!!!!!!!999999999999999999999999999')
		else:
			# spacing = 0.23077/harm_fc #getriebe amt
			spacing = 0.2873/harm_fc # bochum
			# spacing = 0.258276/harm_fc # schottland
			
			
			
		# spacing = 0.225564/harm_fc
		factor_red = config['fact_red']
		# lag = int(116000/factor_red/harm_fc)
		lag = int(config['lag']/factor_red/harm_fc)
		
		# x, index_x = triangle_pulse(time_reg, config['fs'], high_amplitude, spacing)		
		# x, index_x = trape_pulse(time_reg, config['fs'], high_amplitude, spacing, 3)
		x, index_x = boolean_pulse(time_reg, config['fs'], 1., spacing)
		# plt.plot(x, 'k')
		# plt.show

		
		CORR= []
		count = 0
		total = len(Filepaths)
		for filepath in Filepaths:
			
			filename = basename(filepath)

				
			signal = load_signal(filepath, channel=config['channel'])
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
				
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
		
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			
			
			
			
			if config['filter'][0] != 'OFF':
				
				if config['filter'][0] == 'Kurtogram': 
					print('Kurtogram based filter!!!!!!')
					lp, hp = Fast_Kurtogram_filters(signal, config['level'], config['fs'])
					signal = butter_bandpass(signal, config['fs'], freqs=[lp, hp], order=3)
				else:
					signal = butter_filter(signal, config['fs'], config['filter'])
					
			else:
				print('info:              no filter')
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)			
			# signal = signal**2.0
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=20.e3, order=3)
			# signal = hilbert_demodulation(signal)			
			# signal = butter_demodulation(x=signal, fs=config['fs'], filter=['lowpass', 2000., 3], prefilter=['highpass', 20.e3, 3], type_rect=None, dc_value=None)			
			
			
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			
			print('!!!! 1')
			if config['plot'] == 'ON':
				from THR_Burst_Detection import plot_burst_rev		
				t = [i/config['fs'] for i in range(len(signal))]
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			print('!!!! 2')
			
			
			
			index_burst = list(np.array(t_burst_corr)*config['fs'])
			
			# burst = triangle_burst(t_burst_corr, config['fs'], len(signal))
			# burst = trape_burst(t_burst_corr, config['fs'], len(signal), 3)
			burst = boolean_burst(t_burst_corr, config['fs'], len(signal))

			index_max = np.max(index_burst + index_x)
			
			index_max_red = int(index_max/factor_red)

			# burst_red = reduce_fact_bool(index_burst, factor_red, index_max_red)
			# x_red = reduce_fact_bool(index_x, factor_red, index_max_red)
			
			burst_red = reduce_fact_rect(index_burst, factor_red, index_max_red+1, config['side_points'])
			x_red = reduce_fact_rect(index_x, factor_red, index_max_red+1, config['side_points'])
			
			# plt.plot(burst_red, 'bo-')
			# plt.plot(x_red, 'ro-')
			# plt.show()
			
			
			if count == 0:
				del(x)
			del(burst)
			del(signal)
			# del(index_x)
			del(index_burst)
			

			count += 1
			# print('avance: ', count/total)
			
			CORR.append(max_norm_correlation_lag(x_red, burst_red, lag=lag))
		
		row_names = [basename(filepath) for filepath in Filepaths]
		mydict = {}
		mydict['CORR'] = CORR


		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Corr')	
		print('Result in Excel table')
		
		# Output = {'clf':clf, 'scores_cv':Scores_cv, 'scores_test':scores_test, 'scaler':scaler, 'config':config}
		# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		
		
		
		

	else:
		print('unknown mode')
		sys.exit()

		
	return




# def boolean_burst(time_reg, fs, high_amplitude, spacing):
	# n_points = int(time_reg * fs)
	# times_pulses = np.array([spacing*i for i in range(int(time_reg/spacing)+1)])
	# index_pulses = list(times_pulses*fs)
	# x = np.zeros(n_points)
	# for element in index_pulses:
		# x[int(element)] = high_amplitude

	# return x, index_pulses

def triangle_pulse(time_reg, fs, high_amplitude, spacing):
	n_points = int(time_reg * fs)
	times_pulses = np.array([spacing*i for i in range(int(time_reg/spacing)+1)])
	index_pulses = list(times_pulses*fs)
	x = np.zeros(n_points)
	for element in index_pulses:
		x[int(element)] = high_amplitude
		x[int(element+1)] = high_amplitude/2.
		x[int(element-1)] = high_amplitude/2.

	return x, index_pulses

def trape_pulse(time_reg, fs, high_amplitude, spacing, points):
	n_points = int(time_reg * fs)
	times_pulses = np.array([spacing*i for i in range(int(time_reg/spacing)+1)])
	index_pulses = list(times_pulses*fs)
	x = np.zeros(n_points)
	for element in index_pulses:
		for k in range(points):
			x[int(element+k)] = high_amplitude
		x[int(element+1+k)] = high_amplitude/2.
		x[int(element-1)] = high_amplitude/2.

	return x, index_pulses

def boolean_burst(t_burst, fs, n):
	index_burst = list(np.array(t_burst)*fs)
	burst = np.zeros(n)
	for element in index_burst:
		burst[int(element)] = 1
	return burst

def boolean_pulse(time_reg, fs, high_amplitude, spacing):
	n_points = int(time_reg * fs)
	times_pulses = np.array([spacing*i for i in range(int(time_reg/spacing)+1)])
	index_pulses = list(times_pulses*fs)
	x = np.zeros(n_points)
	for element in index_pulses:
		
		x[int(element)] = high_amplitude


	return x, index_pulses

def triangle_burst(t_burst, fs, n):
	index_burst = list(np.array(t_burst)*fs)
	burst = np.zeros(n)
	for element in index_burst:
		burst[int(element)] = 1
		burst[int(element+1)] = 0.5
		burst[int(element-1)] = 0.5
	return burst

def trape_burst(t_burst, fs, n, points):
	index_burst = list(np.array(t_burst)*fs)
	burst = np.zeros(n)
	for element in index_burst:
		for k in range(points):
			burst[int(element + k)] = 1
		burst[int(element+1+k)] = 0.5
		burst[int(element-1)] = 0.5
	return burst



def reduce_fact_bool(index_bool, factor, index_max_red):
	index_bool_red = []
	for element in index_bool:
		index_bool_red.append(int(element/factor))

	bool_red = np.zeros(index_max_red+1)
	for i in range(index_max_red+1):
		if i in index_bool_red:
			bool_red[i] = 1.
	return bool_red

def reduce_fact_rect(index_bool, factor, index_max_red, side_points):
	index_bool_red = []
	for element in index_bool:
		index_bool_red.append(int(element/factor))

	bool_red = np.zeros(index_max_red+1)
	for i in range(index_max_red+1):
		if i in index_bool_red:			
			bool_red[i] = 1.
			for k in range(side_points):
				bool_red[int(i-1-k)] = 1.
				bool_red[int(i+1+k)] = 1.
	return bool_red


def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'layers' or element == 'filter' or element == 'range':
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
	
	config['fs'] = float(config['fs'])
	config['harm_fc'] = float(config['harm_fc'])
	config['fact_red'] = float(config['fact_red'])
	config['lag'] = int(config['lag'])
	config['level'] = int(config['level'])
	config['side_points'] = int(config['side_points'])
	config['db_pa'] = float(config['db_pa'])
	
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
		
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# config['n_batches'] = int(config['n_batches'])
	# config['db'] = int(config['db'])
	# config['divisions'] = int(config['divisions'])
	# config['n_mov_avg'] = int(config['n_mov_avg'])
	# config['train'] = float(config['train'])
	config['thr_value'] = float(config['thr_value'])
	config['time_reg'] = float(config['time_reg'])
	if config['db_out'] != 'OFF':
		config['db_out'] = int(config['db_out'])
		
	
	if config['filter'][0] != 'OFF':
		if config['filter'][0] == 'bandpass':
			config['filter'] = [config['filter'][0], [float(config['filter'][1]), float(config['filter'][2])], float(config['filter'][3])]
		elif config['filter'][0] == 'highpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		elif config['filter'][0] == 'lowpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		elif config['filter'][0] == 'Kurtogram':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
			print('BP Filter based on Kurtogram ON')
		else:
			print('error filter 87965')
			sys.exit()
	
	
	# config['ylabel'] = config['ylabel'].replace('_', ' ')
	# config['zlabel'] = config['zlabel'].replace('_', ' ')
	# config['title'] = config['title'].replace('_', ' ')

	
	# Variable conversion
	
	# Variable conversion
	# if config['sheet'] == 'OFF':
		# config['sheet'] = 0
	
	return config


	
if __name__ == '__main__':
	main(sys.argv)
