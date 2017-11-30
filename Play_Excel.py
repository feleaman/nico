# import os
from os import listdir
from os.path import join, isdir, basename
import sys
# from sys import exit
# from sys.path import path.insert
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
from m_open_extension import *
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from m_open_extension import *
from m_det_features import signal_rms

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

Inputs = ['mode']
InputsOpt_Defaults = {'n_batches':1}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)	
	
	if config['mode'] == 'combine':
		print('Select father group')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_father = filedialog.askopenfilenames()			
		root.destroy()
		
		print('Select son group')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_son = filedialog.askopenfilenames()			
		root.destroy()
		
		RMS_long = []
		count = 0
		for filepath_father, filepath_son in zip(Filepaths_father, Filepaths_son):
			count = count + 1
			table_father = pd.read_excel(filepath_father)
			table_son = pd.read_excel(filepath_son)
			# print(type(table_father))
			RMS = table_father['RMS'].values			
			MAX = table_son['MAX'].values
			rows = table_father.axes[0].tolist()
			# table_father['MAX_dBAE'].values = table_son['RMS'].values
			# print(table_father)			
			mydict = {}
			mydict['RMS'] = RMS
			mydict['MAX'] = MAX
			
			DataFr = pd.DataFrame(data=mydict, index=rows)
			writer = pd.ExcelWriter('Batch_7_Features_OK_AE_' + str(count) + '.xlsx')		
			DataFr.to_excel(writer, sheet_name='Sheet1')	
			print('Result in Excel table')
			
			
			
			# RMS_long.append(table['RMS'].values)
		

		# for count in range(config['n_batches']):
			# print('Batch ', count)
			# root = Tk()
			# root.withdraw()
			# root.update()
			# Filepaths = filedialog.askopenfilenames()			
			# root.destroy()
			# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
			# # RMS = [signal_rms(signal) for signal in Data]
			# RMS = [signal_rms(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
			# MAX = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
			# save_pickle('rms_filt_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('max_filt_batch_' + str(count) + '.pkl', MAX)
			# mydict = {}
			
			# row_names = [basename(filepath) for filepath in Filepaths]
			# mydict['RMS'] = RMS
			# mydict['MAX'] = MAX
			
			# DataFr = pd.DataFrame(data=mydict, index=row_names)
			# writer = pd.ExcelWriter('to_use_batch_' + str(count) + '.xlsx')

		
			# DataFr.to_excel(writer, sheet_name='Sheet1')	
			# print('Result in Excel table')
			
			# # mean_mag_fft = read_pickle('mean_5_fft.pkl')
			# # corrcoefMAGFFT = [np.corrcoef(mag_fft(signal, config['fs'])[0], mean_mag_fft) for signal in Data]
			# # save_pickle('fftcorrcoef_batch_' + str(count) + '.pkl', corrcoefMAGFFT)
			
			
			
			
			# # plt.boxplot(RMS)
			# # plt.show()
	
	

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
		if element == 'no_element':
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
	
	# config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# config['n_batches'] = int(config['n_batches'])
	# Variable conversion
	return config


	
if __name__ == '__main__':
	main(sys.argv)
