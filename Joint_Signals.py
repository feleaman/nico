# Main_Analysis.py
# Last updated: 15.08.2017 13:13 by Felix Leaman
# Description:
# Code for opening a .mat or .tdms data file with single channel and plotting different types of analysis
# The file and channel is selected by the user
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

import os.path
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))

sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


# from os import chdir
# plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpeg'
 
#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs']
InputsOpt_Defaults = {'power2':'OFF', 'plot':'OFF', 'title_plot':None, 'file':'OFF'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	x1, filename1 = invoke_signal(config)
	x2, filename2 = invoke_signal(config)
	
	magX1, f1, df1 = mag_fft(x1, config['fs'])
	magX2, f2, df2 = mag_fft(x1, config['fs'])
	
	
	# freq_value = 21.4
	freq_value = 35.8
	print(amp_component_zone(X=magX1, df=df1, freq=freq_value, tol=1.0))
	print(amp_component_zone(X=magX2, df=df2, freq=freq_value, tol=1.0))
	
	# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True)
	# ax[0].plot(x1, 'bo-')
	# ax[0].set_title(filename1)
	# ax[1].plot(x2, 'ro-')
	# ax[1].set_title(filename2)
	# plt.show()
	
	values = []
	# iter = int(250000/100)
	iter = int(2500/2)
	for i in range(iter):
	
		print('avance: ', i/iter)
		i = i*2
		signal_fas = list(x2[i:])
		
		# x3 = np.array(list(x1) + list(x2[i:]))
		x3 = np.array(list(x1) + signal_fas)
		
		# n_points = 2**(max_2power(len(x3)))
		# x3 = x3[0:n_points]
		
	
		
		magX3, f3, df3 = mag_fft(x3, config['fs'])
		# magX3, f3, df3 = mag_fft_hanning(x3, config['fs'])
		
		
		values.append(amp_component_zone(X=magX3, df=df3, freq=freq_value, tol=1.0))
	
	max_value = np.max(np.array(values))
	max_index = np.argmax(np.array(values))
	print(max_value)
	print(max_index)
	
	# x3b = np.array(list(x1) + list(x2[max_index:]))
	x3b = np.array(list(x1) + list(x2[max_index:]) + list(x2[0:max_index]))
	
	magX3b, f3b, df3b = mag_fft(x3b, config['fs'])
	
	fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
	ax[0].plot(f1, magX1, 'bo-')
	ax[0].set_title(filename1)
	ax[1].plot(f2, magX2, 'ro-')
	ax[1].set_title(filename2)
	ax[2].plot(f3b, magX3b, 'go-')
	ax[2].set_title('combi')
	plt.show()
	
	
	
	
	return 

def invoke_signal(config):

	channel = config['channel']
	fs = config['fs']
	
	if config['file'] == 'OFF':
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
	else:
		filename = config['file']


	x = load_signal(filename, channel)
	filename = os.path.basename(filename) #changes from path to file
	print(filename)

	#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++


		
	dt = 1.0/fs
	

	
	if config['power2'] == 'auto':
		n_points = 2**(max_2power(len(x)))
	elif config['power2'] == 'OFF':
		n_points = len(x)
	else:
		n_points = 2**config['power2']
	

	n = len(x)
	n_points = n
	tr = n*dt
	t = np.array([i*dt for i in range(n)])
	
	xraw = x
	
	# x = x*1000.
	
	
	#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

	config_analysis = {'WFM':True, 'FFT':False, 'PSD':False, 'STFT':False, 'STPSD':False, 'Cepstrum':False, 'CyclicSpectrum':False, 'Kurtogram':False}

	config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [95.e3, 140.e3], 3],
	'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 2000., 3]}

	#When hilbert is selected, the other parameters are ignored

	config_diff = {'analysis':False, 'length':1, 'same':True}

	config_stft = {'segments':500, 'window':'boxcar', 'mode':'magnitude', 'log-scale':False, 'SPK':True}
	
	
	config_kurtogram = {'levels':5}

	config_stPSD = {'segments':500, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}
	
	# config_denois = {'analysis':True, 'mode':'butter_highpass', 'freq':70.e3}
	# config_denois = {'analysis':True, 'mode':'butter_highpass', 'freq':20.e3}
	# config_denois = {'analysis':True, 'mode':'butter_lowpass', 'freq':5.}
	# config_denois = {'analysis':False, 'mode':'butter_bandpass', 'freq':[95.e3, 140.e3]}
	config_denois = {'analysis':False, 'mode':'butter_lowpass', 'freq':30.}
	# x = fourier_filter(x=x, fs=config['fs'], type='bandpass', freqs=[21.0, 23.8])
	# x = fourier_filter(x=x, fs=config['fs'], type='one_component', freqs=22.4)
	# x = x / np.max(x)


	# dict_out = {'time':t, 'signal':x, 'freq':f, 'mag_fft':magX, 'filename':filename, 'signal_raw':xraw}
	return x, filename
	# return t, x, f, magX, filename
	# return t, x, f, magX, xraw, filename

# plt.show()
def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'no_element':
				parser.add_argument('--' + element, nargs='+')
			else:
				parser.add_argument('--' + element, nargs='?')
		print(parser.parse_args())
		args = parser.parse_args()

		
	except:
		# args = argv
		arguments = [element for element in argv if element[0:2] == '--']
		values = [element for element in argv if element[0:2] != '--']

		# from argparse import ArgumentParser
		# from ArgumentParser import Namespace
		parser = ArgumentParser()
		for element in arguments:
			parser.add_argument(element)

		args = parser.parse_args(argv)

		# print(test)
		# sys.exit()
		
	config = {}	
		
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float
	if config['power2'] != 'auto' and config['power2'] != 'OFF':
		config['power2'] = int(config['power2'])
	config['fs'] = float(config['fs'])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	print('caca')
	return config


if __name__ == '__main__':
	main(sys.argv)
