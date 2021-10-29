# import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import isnan
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

plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.7, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature_cond':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature5':'RMS', 'feature_array':['RMS']}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == '1x4_two_scale_plot':
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'
		print('Select Features files')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		print('Select Feature Condition...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath2 = filedialog.askopenfilename()			
		root.destroy()
		
		
		
		mydict2 = pd.read_excel(filepath2, sheetname=config['sheet'])
		mydict2 = mydict2.to_dict(orient='list')
		Feature2 = np.array(mydict2[config['feature_cond']])
		
		index_nonan = []
		count = 0
		for element in Feature2:
			if not isnan(element):
				index_nonan.append(count)
			count +=1
		Feature2 = Feature2[index_nonan]
		if config['feature_cond'] == 'n':
			Feature2 = Feature2/117.18
		
		fig, ax = plt.subplots(nrows=1, ncols=2)
		# ax2 = ax.twinx()
		
		# count = 0
		titles = ['AE-1', 'AE-2', 'AE-3', 'AE-4']
		features = ['MAX', 'RMS']
		fontsize_big = 18
		fontsize_little = 17
		fontsize_legend = 14
		for k in range(2):
			count = 0
			for filepath in Filepaths:
			
				mydict = pd.read_excel(filepath, sheetname=config['sheet'])
				mydict = mydict.to_dict(orient='list')
				Feature = mydict[features[k]]
				
				Feature = 1000*np.array(Feature)/141.3
				
				if config['n_mov_avg'] != 0:
					Feature = movil_avg(Feature, config['n_mov_avg'])			
				
				
				time = np.array([i*10. for i in range(len(Feature))])/60.
				time2 = time[index_nonan]
				
				ax[k].plot(time, Feature, label=titles[count])
				
				ax[k].set_xlabel('Time [min]', fontsize=fontsize_big)
				
				if features[k] == 'RMS':
					ax[k].set_ylabel('RMS value [mV]', color='k', fontsize=fontsize_big)
					ax[k].set_ylim(bottom=0, top=0.8)
				else:
					ax[k].set_ylabel('Maximum value [mV]', color='k', fontsize=fontsize_big)
					ax[k].set_ylim(bottom=0, top=40)
				
				ax[k].tick_params('y', colors='k')				
				
				ax[k].legend(fontsize=fontsize_legend, loc='best')
				ax[k].tick_params(axis='both', labelsize=fontsize_little)
				
				ax[k].set_xlim(left=0, right=60)
				count += 1	
				
			ax2 = ax[k].twinx()			
			ax2.set_xlim(left=0, right=60)
			# ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
			
			if config['feature_cond'] == 'T':
				ax2.set_ylim(bottom=25, top=55)
				ax2.set_ylabel('Temperature [°C]', color='m', fontsize=fontsize_big)
			elif config['feature_cond'] == 'n':
				
				ax2.set_ylabel('Rotational speed [CPM]', color='m', fontsize=fontsize_big)
				# Feature2 = Feature2/117.18
				ax2.set_ylim(bottom=0, top=15)
				ax2.set_yticks([0, 3, 6, 9, 12, 15])
			
			elif config['feature_cond'] == 'M':
				
				ax2.set_ylabel('Torque [Nm]', color='m', fontsize=fontsize_big)
				# Feature2 = Feature2/117.18
				ax2.set_ylim(bottom=0, top=18)
				ax2.set_yticks([0, 3, 6, 9, 12, 15, 18])
				# ax2.set_yticks([0, 3, 6, 9, 12, 15])
			# ax2.set_ylabel('Rotational speed [CPM]', color='r', fontsize=15)
			
			ax2.plot(time2, Feature2, 'sm')			
			
			ax2.tick_params('y', colors='m')
			ax2.tick_params(axis='both', labelsize=fontsize_little)
			


		fig.set_size_inches(12, 4.5)


		plt.subplots_adjust(wspace=0.4, left=0.065, right=0.935, bottom=0.15, top=0.94)
		# plt.tight_layout()
		plt.show()
	
	elif config['mode'] == 'calculate_correlation':

		print('Select Features files')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		print('Select Feature Condition...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath2 = filedialog.askopenfilename()			
		root.destroy()
		
		
		
		mydict2 = pd.read_excel(filepath2, sheetname=config['sheet'])
		mydict2 = mydict2.to_dict(orient='list')
		Feature2 = np.array(mydict2[config['feature_cond']])
		
		index_nonan = []
		count = 0
		for element in Feature2:
			if not isnan(element):
				index_nonan.append(count)
			count +=1
		Feature2 = Feature2[index_nonan]
		if config['feature_cond'] == 'n':
			Feature2 = Feature2/117.18
		
		# ax2 = ax.twinx()
		mydict2 = {}
		count = 0
		titles = ['AE-1', 'AE-2', 'AE-3', 'AE-4']
		features = ['MAX', 'RMS']
		for k in range(2):
			vec = []
			for filepath in Filepaths:
				
				mydict = pd.read_excel(filepath, sheetname=config['sheet'])
				mydict = mydict.to_dict(orient='list')
				Feature = mydict[features[k]]
				
				Feature = 1000*np.array(Feature)/141.3
				
				if config['n_mov_avg'] != 0:
					Feature = movil_avg(Feature, config['n_mov_avg'])			
				
				
				time = np.array([i*10. for i in range(len(Feature))])/60.
				time2 = time[index_nonan]
				
				Feature2int = np.interp(time, time2, Feature2)
				
				# plt.plot(time, Feature, 'b', time, Feature2int, 'or')
				# plt.show()
				print(titles[count], ' = ' , np.corrcoef(Feature, Feature2int)[0][1])
				vec.append(np.corrcoef(Feature, Feature2int)[0][1])
			mydict2[features[k]] = vec
			



		print(mydict2)
		DataFr = pd.DataFrame(data=mydict2, index=titles)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Correlations')	
		print('Result in Excel table')
			


	
	
	
	
	
			




	else:
		print('unknown mode')
		sys.exit()

		
	return






def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'layers' or element == 'feature_array':
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
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# config['n_batches'] = int(config['n_batches'])
	# config['db'] = int(config['db'])
	# config['divisions'] = int(config['divisions'])
	config['n_mov_avg'] = int(config['n_mov_avg'])
	config['train'] = float(config['train'])
	
	config['n_pre'] = float(config['n_pre'])
	config['m_post'] = float(config['m_post'])
	
	config['alpha'] = float(config['alpha'])
	config['tol'] = float(config['tol'])	
	config['learning_rate_init'] = float(config['learning_rate_init'])	
	#Type conversion to int	
	config['max_iter'] = int(config['max_iter'])
	config['rs'] = int(config['rs'])

	# Variable conversion	
	correct_layers = tuple([int(element) for element in (config['layers'])])
	config['layers'] = correct_layers
	
	config['ylabel'] = config['ylabel'].replace('_', ' ')
	config['zlabel'] = config['zlabel'].replace('_', ' ')
	config['title'] = config['title'].replace('_', ' ')

	
	# Variable conversion
	
	# Variable conversion
	if config['sheet'] == 'OFF':
		config['sheet'] = 0
	
	return config


	
if __name__ == '__main__':
	main(sys.argv)
