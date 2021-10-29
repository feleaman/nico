
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
from argparse import ArgumentParser
import pandas as pd
from os.path import join, isdir, basename, dirname, isfile
from os import listdir


# from THR_Burst_Detection import thr_burst_detector
# from THR_Burst_Detection import thr_burst_detector_rev
from THR_Burst_Detection import full_thr_burst_detector
from THR_Burst_Detection import read_threshold
from THR_Burst_Detection import plot_burst_rev


import math
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler  
import datetime
import scipy

#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'channel', 'fs']
# InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'plot':'OFF', 'n_files':1, 'title_plot':None, 'thr_value':30., 'thr_mode':'fixed_value', 'window_time':0.05, 'save_plot':'OFF', 'file':'OFF', 'time_segments':1., 'stella':1500, 'lockout':3000, 'highpass':20.e3, 'mv':'ON', 'mypath':None}

InputsOpt_Defaults = {'name':'auto', 'plot':'ON', 'n_files':1, 'title_plot':None, 'thr_value':3.6, 'thr_mode':'factor_rms', 'window_time':0.001, 'stella':300, 'lockout':300, 'filter':['highpass', 5.e3, 3], 'mv':'ON', 'mypath':None, 'save':'ON', 'save_plot':'OFF', 'amp_db':'OFF', 'db_pa':37.}
# gearbox mio: thr_60, wt_0.001, hp_70k, stella_100, lcokout 200


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	if config['mode'] == 'energy_band':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
	
		intervals, Intervals_Dict, str_intervals = Lgenerate_n_intervals_noinf(0.e3, 500.e3, 22)
	
		
		print(str_intervals)

		for filepath in Filepaths:			
		
			signal = 1000*load_signal(filepath, channel=config['channel'])/70.8
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
				
			filename = basename(filepath)

			magX, f, df = mag_fft(x=signal, fs=config['fs'])
			# mag2 = (magX)**2.0
			# rms = ((np.sum(mag2))**0.5)
			# print('!!!!!!!!!!!!!!!!!!!!!!!!! rms = ', rms*0.707)
			# print(filename)
			# a = input('pause ... ')
			
			for i in range(len(intervals)-1):				
				Intervals_Dict[str_intervals[i]].append(energy_in_band(magX, df, intervals[i], intervals[i+1]))


		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		
		# if config['rms_on'] == 'ON':
			# exponent = 0.5
			# fact = 0.707
		# else:
			# exponent = 1.
			# fact = 1.
		
		for element in str_intervals:

			mydict[element] = Intervals_Dict[element]



		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Freq_Bands')	
		print('Result in Excel table')
	
	elif config['mode'] == 'plot_bands_taec':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
		params = {'mathtext.default': 'regular' }          
		fig, ax = plt.subplots(nrows=2, sharey=True, sharex=True)
		colors = ['pink', 'lightgreen']
		count = 0
		for filepath in Filepaths:
			Boxes = []
			Labels = []
			
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')
			
			for key in mydict.keys():
				Boxes.append(mydict[key])
				Labels.append(key)


			NewLabels = []
			for num in Labels:
				NewLabels.append(str(int(float(num[num.find('/')+1 : num.find('_')])/1000.)) + '-' + str(int(float(num[num.find('_')+1 : ])/1000.)))


			box = ax[count].boxplot(Boxes[:25], patch_artist=True)

			print(np.sum(np.array(Boxes[:25])))
			
			ax[count].set_xticklabels(NewLabels[:25], rotation=45)
			ax[count].grid(color='k', linestyle='-', linewidth=0.1)
			
			ax[count].set_ylabel('Energie [mV]', fontsize=13)
			
			
			plt.rcParams.update(params)
			ax[count].set_ylabel('Energie [m$V^{2}$]', fontsize=14)
			
			for patch in box['boxes']:
				patch.set_facecolor(colors[count])

			if os.path.basename(filepath).find('Gut') != -1:
				lbl = 'Gut'
			else:
				lbl = 'Schlecht'
			ax[count].legend(box['boxes'], [lbl], loc='upper right', fontsize=13)
			ax[count].set_yscale('log')
			count += 1
		ax[0].tick_params(labelbottom=False)
		ax[1].set_xlabel('Frequenzbereich [kHz]', fontsize=14)
		ax[0].tick_params(axis='both', labelsize=12)
		ax[1].tick_params(axis='both', labelsize=12)
		
		
		
		# plt.legend()
		plt.subplots_adjust(hspace=0.05)
		plt.show()
	
	
	elif config['mode'] == 'plot_bands':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
		params = {'mathtext.default': 'regular' }          
			
		
		count = 0
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')
			Labels = []			
			for key in mydict.keys():
				Labels.append(key)
			
			if count == 0:				
				NewLabels = []
				for num in Labels:
					NewLabels.append(str(int(float(num[num.find('/')+1 : num.find('_')])/1000.)) + '-' + str(int(float(num[num.find('_')+1 : ])/1000.)))

				labdict = {}
				for element in NewLabels:
					labdict[element] = []
			
			
			for newelement, element in zip(NewLabels, Labels):
				labdict[newelement] += mydict[element]
			
			count += 1

		fig, ax = plt.subplots()
		for i in range(len(NewLabels)):
			ax.plot(labdict[NewLabels[i]], '-o', label=NewLabels[i])
		ax.legend()
		plt.show()
	
	elif config['mode'] == 'plot_bands_box':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
		params = {'mathtext.default': 'regular' }          
			
		
		count = 0
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')
			Labels = []			
			for key in mydict.keys():
				Labels.append(key)
			
			if count == 0:				
				NewLabels = []
				for num in Labels:
					NewLabels.append(str(int(float(num[num.find('/')+1 : num.find('_')])/1000.)) + '-' + str(int(float(num[num.find('_')+1 : ])/1000.)))

				labdict = {}
				for element in NewLabels:
					labdict[element] = []
			
			
			for newelement, element in zip(NewLabels, Labels):
				labdict[newelement] += mydict[element]
			
			count += 1

		fig, ax = plt.subplots()
		Boxes = []
		Labels = []
		for i in range(len(NewLabels)):
			Boxes.append(labdict[NewLabels[i]])
			Labels.append(NewLabels[i])
			# ax.plot(labdict[NewLabels[i]], '-o', label=NewLabels[i])
		ax.boxplot(Boxes)
		ax.set_xticklabels(Labels, rotation=45)
		ax.set_yscale('log')
		ax.grid(color='k', linestyle='-', linewidth=0.1)
		plt.show()
		
		
		
	return

def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element' or element == 'filter':
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
	config['stella'] = int(config['stella'])
	config['thr_value'] = float(config['thr_value'])
	# config['highpass'] = float(config['highpass'])
	config['window_time'] = float(config['window_time'])
	# config['time_segments'] = float(config['time_segments'])
	config['lockout'] = int(config['lockout'])
	
	
	
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
	
	
	
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config

def max_norm_correlation(signal1, signal2):
	correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')
	return np.max(correlation)

if __name__ == '__main__':
	main(sys.argv)
