# import os
from os import listdir
import matplotlib.pyplot as plt

from Kurtogram3 import Fast_Kurtogram_filters
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
from Genetic_Filter import *

# from THR_Burst_Detection import full_thr_burst_detector
# from THR_Burst_Detection import read_threshold
# from THR_Burst_Detection import plot_burst_rev

Inputs = ['mode', 'fs', 'channel']
InputsOpt_Defaults = {'n_batches':1, 'name_feature':'RMS', 'name_condition':'T', 'db':0, 'divisions':10, 'name':'name', 'mypath':None, 'level':5, 'parents':4, 'clusters':20, 'mutation':0.1, 'generations':3}
from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'two_scale_plot':
		fig, ax1 = plt.subplots()
		ax1.set_xlabel('Time s')
		for k in range(2):
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			data = np.loadtxt(filepath)
			data = data.tolist()
			time = [i/config['fs'] for i in range(len(data))]				
			path_basename = basename(filepath)
			if path_basename.find('torque') != -1:
				ax1.plot(time, data, '-b')
				ax1.set_ylabel('Torque kNm', color='b')
				ax1.tick_params('y', colors='b')				
			elif path_basename.find('rpm') != -1:
				# data = data * 1000
				ax2 = ax1.twinx()
				ax2.plot(time, data, '-r')
				ax2.set_ylabel('RPM', color='r')
				ax2.tick_params('y', colors='r')				
			else:
				name_label = None

		fig.tight_layout()
		plt.show()
	
	elif config['mode'] == 'long_analysis_features':

		for count in range(config['n_batches']):
			print('Batch ', count)
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
			Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
			# RMS = [signal_rms(signal) for signal in Data]
			# MAX = [np.max(signal) for signal in Data]
			RMS = [signal_rms(butter_highpass(x=signal, fs=config['fs'], freq=70.e3, order=3)) for signal in Data]
			# MAX = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=70.e3, order=3)) for signal in Data]
			# config['thr_value'] = 3.226
			# config['window_time'] = 0.0001
			# config['thr_mode'] = 'factor_rms'
			# from m_det_features import id_burst_threshold
			# BPS = [burst_per_second(butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3), config) for signal in Data]
			
			# CRF = []
			# # RMS = []
			# for signal in Data:
				# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
				# # signal = signal / np.max(np.absolute(signal))
				# CRF.append(np.max(np.absolute(signal)) / signal_rms(signal))
				# # RMS.append(signal_rms(signal))
				
			
			
			# save_pickle('rms_filt_norm_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('rms_filt_norm_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('MAX_hp70k_batch_' + str(count) + '.pkl', MAX)
			save_pickle('RMS_hp70k_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('max_filt_batch_' + str(count) + '.pkl', MAX)
			# save_pickle('bps_filt_perrms_batch_' + str(count) + '.pkl', BPS)
			mydict = {}
			
			row_names = [basename(filepath) for filepath in Filepaths]
			# mydict['RMS'] = RMS
			mydict['MAX'] = MAX
			# mydict['BPS'] = BPS
			# mydict['CRF'] = CRF

			
			DataFr = pd.DataFrame(data=mydict, index=row_names)
			writer = pd.ExcelWriter('to_use_batch_' + str(count) + '.xlsx')

		
			DataFr.to_excel(writer, sheet_name='Sheet1')	
			print('Result in Excel table')
			
			# mean_mag_fft = read_pickle('mean_5_fft.pkl')
			# corrcoefMAGFFT = [np.corrcoef(mag_fft(signal, config['fs'])[0], mean_mag_fft) for signal in Data]
			# save_pickle('fftcorrcoef_batch_' + str(count) + '.pkl', corrcoefMAGFFT)
			
			
			
			
			# plt.boxplot(RMS)
			# plt.show()
	
	elif config['mode'] == 'new_long_analysis_features':
	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		RMS = []
		MAX= []

		for filepath in Filepaths:
			
			signal = load_signal(filepath, channel=config['channel'])
			# RMS = [signal_rms(signal) for signal in Data]
			# MAX = [np.max(signal) for signal in Data]
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=70.e3, order=3)
			RMS.append(signal_rms(signal))
			MAX.append(np.max(signal))
			# MAX = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=70.e3, order=3)) for signal in Data]
			# config['thr_value'] = 3.226
			# config['window_time'] = 0.0001
			# config['thr_mode'] = 'factor_rms'
			# from m_det_features import id_burst_threshold
			# BPS = [burst_per_second(butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3), config) for signal in Data]
			
			# CRF = []
			# # RMS = []
			# for signal in Data:
				# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
				# # signal = signal / np.max(np.absolute(signal))
				# CRF.append(np.max(np.absolute(signal)) / signal_rms(signal))
				# # RMS.append(signal_rms(signal))
				
			
			
			# save_pickle('rms_filt_norm_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('rms_filt_norm_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('MAX_hp70k_batch_' + str(count) + '.pkl', MAX)
			# save_pickle('RMS_hp70k_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('max_filt_batch_' + str(count) + '.pkl', MAX)
			# save_pickle('bps_filt_perrms_batch_' + str(count) + '.pkl', BPS)
		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		mydict['RMS'] = RMS
		mydict['MAX'] = MAX


		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter('Result' + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Sheet1')	
		print('Result in Excel table')
	
	elif config['mode'] == 'new_long_analysis_features_wfm':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:
			if config['channel'].find('AE') != -1:
				idf = 'E'
			elif config['channel'].find('AC') != -1:
				idf = 'D'
			else:
				print('error auto file selection 454356')
				sys.exit()
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == idf if f[-4:] == 'tdms']
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			

		
		MAX= []
		P2P = []
		RMS = []
		# STD = []
		CREST = []
		# SKEW = []
		KURT = []
		

		
		
		for filepath in Filepaths:
			
			signal = 1000*load_signal_2(filepath, channel=config['channel'])/70.8
			# signal = load_signal_2(filepath, channel=config['channel'])
			
			# if filepath.find('20181102') == -1:
				# signal = signal*10.
			# signal = 141.3*signal/281.8
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=10.e3, order=3)
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			
			# signal = signal**2.0
			
			# signal = hilbert_demodulation(signal)
			
			
			value_rms = signal_rms(signal)
			value_max = np.max(np.absolute(signal))
			value_min = np.min(signal)
			abs_max = np.max(np.array([np.absolute(value_max), np.absolute(value_min)]))
			
			MAX.append(abs_max)
			P2P.append(value_max - value_min)
			RMS.append(value_rms)
			CREST.append(abs_max/value_rms)
			# # STD.append(np.std(signal))
			# SKEW.append(scipy.stats.skew(signal))
			KURT.append(scipy.stats.kurtosis(signal, fisher=False))

			
			
			

			
			# CRF = []
			# # RMS = []
			# for signal in Data:
				# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
				# # signal = signal / np.max(np.absolute(signal))
				# CRF.append(np.max(np.absolute(signal)) / signal_rms(signal))
				# # RMS.append(signal_rms(signal))
		
		# RMS.append(np.mean(np.array(RMS)))
		# MAX.append(np.mean(np.array(MAX)))
		# P2P.append(np.mean(np.array(P2P)))
		# CREST.append(np.mean(np.array(CREST)))
		# # STD.append(np.mean(np.array(STD)))
		# SKEW.append(np.mean(np.array(SKEW)))
		# KURT.append(np.mean(np.array(KURT)))
		# for freq_value in all_frequencies:
			# f_mean = np.mean(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_mean)
		# LP6.append(np.mean(np.array(LP6)))
		# LP7.append(np.mean(np.array(LP7)))
		# LP16.append(np.mean(np.array(LP16)))
		# LP17.append(np.mean(np.array(LP17)))
		# LP21.append(np.mean(np.array(LP21)))
		
		
		# RMS.append(np.std(np.array(RMS)))
		# MAX.append(np.std(np.array(MAX)))
		# P2P.append(np.std(np.array(P2P)))
		# CREST.append(np.std(np.array(CREST)))
		# # STD.append(np.std(np.array(STD)))
		# SKEW.append(np.std(np.array(SKEW)))
		# KURT.append(np.std(np.array(KURT)))

		# LP6.append(np.std(np.array(LP6)))
		# LP7.append(np.std(np.array(LP7)))
		# LP16.append(np.std(np.array(LP16)))
		# LP17.append(np.std(np.array(LP17)))
		# LP21.append(np.std(np.array(LP21)))
		

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')
		mydict['RMS'] = RMS
		mydict['MAX'] = MAX
		mydict['P2P'] = P2P
		mydict['CREST'] = CREST
		# # mydict['STD'] = STD
		# mydict['SKEW'] = SKEW
		mydict['KURT'] = KURT
		# mydict['LP6'] = LP6
		# mydict['LP7'] = LP7
		# mydict['LP16'] = LP16
		# mydict['LP17'] = LP17
		# mydict['LP21'] = LP21
		# mydict['LP24'] = LP24

		# print(mydict)
		# sys.exit()
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Wfm_Features')	
		print('Result in Excel table')
	
	elif config['mode'] == 'new_long_analysis_features_cross':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		Filenames = [basename(filepath) for filepath in Filepaths]

		corr_dict = {}
		# for filename in Filenames:
			# corr_dict[filename[0:-5]] = []

		# RMS = []
		# STD = []
		# CREST = []
		# SKEW = []
		# KURT = []
		
		# LP4 = []
		# LP6 = []
		# LP7 = []
		# LP12 = []
		# LP13 = []
		# LP17 = []
		# LP21 = []
		# LP24 = []
		
		
		for filepath in Filepaths:
			
			signal = load_signal_2(filepath, channel=config['channel'])
			# signal = 141.3*signal/281.8
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=20.e3, order=3)
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			
			# signal = signal**2.0
			
			# signal = hilbert_demodulation(signal)
			
			
			# value_rms = signal_rms(signal)
			corr_dict[basename(filepath)[0:-4]] = []
			for filepath2 in Filepaths:
				signal2 = load_signal_2(filepath2, channel=config['channel'])
				corr = max_norm_correlation(signal, signal2)
				corr_dict[basename(filepath)[0:-4]].append(corr)
			# RMS.append(value_rms)
			# CREST.append(abs_max/value_rms)
			# # STD.append(np.std(signal))
			# SKEW.append(scipy.stats.skew(signal))
			# KURT.append(scipy.stats.kurtosis(signal, fisher=False))
			
			# LP6.append(lout_featp6(signal))
			# LP7.append(lout_featp7(signal))
			# LP16.append(lout_featp16(signal, dp=1./config['fs']))
			# LP17.append(lout_featp17(signal, dp=1./config['fs']))
			# LP21.append(lout_featp21(signal, dp=1./config['fs']))
			# LP24.append(lout_featp24(signal, dp=1./config['fs']))
			
			
			

			
			# CRF = []
			# # RMS = []
			# for signal in Data:
				# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
				# # signal = signal / np.max(np.absolute(signal))
				# CRF.append(np.max(np.absolute(signal)) / signal_rms(signal))
				# # RMS.append(signal_rms(signal))
		
		# RMS.append(np.mean(np.array(RMS)))
		# MAX.append(np.mean(np.array(MAX)))
		# P2P.append(np.mean(np.array(P2P)))
		# CREST.append(np.mean(np.array(CREST)))
		# # STD.append(np.mean(np.array(STD)))
		# SKEW.append(np.mean(np.array(SKEW)))
		# KURT.append(np.mean(np.array(KURT)))
		# for freq_value in all_frequencies:
			# f_mean = np.mean(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_mean)
		# LP6.append(np.mean(np.array(LP6)))
		# LP7.append(np.mean(np.array(LP7)))
		# LP16.append(np.mean(np.array(LP16)))
		# LP17.append(np.mean(np.array(LP17)))
		# LP21.append(np.mean(np.array(LP21)))
		
		
		# RMS.append(np.std(np.array(RMS)))
		# MAX.append(np.std(np.array(MAX)))
		# P2P.append(np.std(np.array(P2P)))
		# CREST.append(np.std(np.array(CREST)))
		# # STD.append(np.std(np.array(STD)))
		# SKEW.append(np.std(np.array(SKEW)))
		# KURT.append(np.std(np.array(KURT)))

		# LP6.append(np.std(np.array(LP6)))
		# LP7.append(np.std(np.array(LP7)))
		# LP16.append(np.std(np.array(LP16)))
		# LP17.append(np.std(np.array(LP17)))
		# LP21.append(np.std(np.array(LP21)))
		

		mydict = corr_dict
			
		row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')
		# mydict['RMS'] = RMS
		# mydict['MAX'] = MAX
		# mydict['P2P'] = P2P
		# mydict['CREST'] = CREST
		# # mydict['STD'] = STD
		# mydict['SKEW'] = SKEW
		# mydict['KURT'] = KURT
		# mydict['LP6'] = LP6
		# mydict['LP7'] = LP7
		# mydict['LP16'] = LP16
		# mydict['LP17'] = LP17
		# mydict['LP21'] = LP21
		# mydict['LP24'] = LP24

		# print(mydict)
		# sys.exit()
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['channel'] + '_' + config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Burst_Cor')	
		print('Result in Excel table')	
	
	
	elif config['mode'] == 'new_long_analysis_features_loutas':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			

		

		
		# LP4 = []
		# LP6 = []
		# LP7 = []
		LP12 = []
		LP13 = []
		LP16 = []
		LP17 = []
		LP21 = []
		LP24 = []
		
		
		for filepath in Filepaths:
			
			signal = load_signal_2(filepath, channel=config['channel'])
			# signal = signal*1000.
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=20.e3, order=3)
			signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			
			signal, f, df = mag_fft(x=signal, fs=config['fs'])
			
			# signal = signal**2.0
			
			# signal = hilbert_demodulation(signal)
			
			
			# value_rms = signal_rms(signal)
			# value_max = np.max(np.absolute(signal))
			# value_min = np.min(signal)
			# abs_max = np.max(np.array([np.absolute(value_max), np.absolute(value_min)]))
			
			# MAX.append(value_max)
			# # P2P.append(value_max - value_min)
			# RMS.append(value_rms)
			# CREST.append(abs_max/value_rms)
			# # STD.append(np.std(signal))
			# SKEW.append(scipy.stats.skew(signal))
			# KURT.append(scipy.stats.kurtosis(signal, fisher=False))
			
			# LP6.append(lout_featp6(signal))
			# LP7.append(lout_featp7(signal))
			
			lp12 = lout_featp12(signal)
			LP12.append(lp12)
			
			lp13 = lout_featp13(signal, lp12)
			LP13.append(lp13)
			
			lp16 = lout_featp16(signal, dp=1./config['fs'])
			LP16.append(lp16)
			
			lp17 = lout_featp17(signal, dp=1./config['fs'], lp16=lp16)
			LP17.append(lp17)
			
			lp21 = lout_featp21(signal, dp=1./config['fs'], lp16=lp16, lp17=lp17)			
			LP21.append(lp21)
			
			lp24 = lout_featp24(signal, dp=1./config['fs'], lp16=lp16, lp17=lp17)
			LP24.append(lp24)
			
			
			

			
			# CRF = []
			# # RMS = []
			# for signal in Data:
				# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
				# # signal = signal / np.max(np.absolute(signal))
				# CRF.append(np.max(np.absolute(signal)) / signal_rms(signal))
				# # RMS.append(signal_rms(signal))
		
		# RMS.append(np.mean(np.array(RMS)))
		# MAX.append(np.mean(np.array(MAX)))
		# P2P.append(np.mean(np.array(P2P)))
		# CREST.append(np.mean(np.array(CREST)))
		# # STD.append(np.mean(np.array(STD)))
		# SKEW.append(np.mean(np.array(SKEW)))
		# KURT.append(np.mean(np.array(KURT)))
		# for freq_value in all_frequencies:
			# f_mean = np.mean(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_mean)
		LP12.append(np.mean(np.array(LP12)))
		LP13.append(np.mean(np.array(LP13)))
		LP16.append(np.mean(np.array(LP16)))
		LP17.append(np.mean(np.array(LP17)))
		LP21.append(np.mean(np.array(LP21)))
		LP24.append(np.mean(np.array(LP24)))
		
		
		# RMS.append(np.std(np.array(RMS)))
		# MAX.append(np.std(np.array(MAX)))
		# P2P.append(np.std(np.array(P2P)))
		# CREST.append(np.std(np.array(CREST)))
		# # STD.append(np.std(np.array(STD)))
		# SKEW.append(np.std(np.array(SKEW)))
		# KURT.append(np.std(np.array(KURT)))

		LP12.append(np.std(np.array(LP12)))
		LP13.append(np.std(np.array(LP13)))
		LP16.append(np.std(np.array(LP16)))
		LP17.append(np.std(np.array(LP17)))
		LP21.append(np.std(np.array(LP21)))
		LP24.append(np.std(np.array(LP24)))
		

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		row_names.append('Mean')
		row_names.append('Std')
		# mydict['RMS'] = RMS
		# mydict['MAX'] = MAX
		# mydict['P2P'] = P2P
		# mydict['CREST'] = CREST
		# # mydict['STD'] = STD
		# mydict['SKEW'] = SKEW
		# mydict['KURT'] = KURT
		# mydict['LP6'] = LP6
		# mydict['LP7'] = LP7
		mydict['LP12'] = LP12
		mydict['LP13'] = LP13
		mydict['LP16'] = LP16
		mydict['LP17'] = LP17
		mydict['LP21'] = LP21
		mydict['LP24'] = LP24

		# print(mydict)
		# sys.exit()
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='AE_Wfm')	
		print('Result in Excel table')	
		
	elif config['mode'] == 'new_long_analysis_features_freq':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:
			if config['channel'].find('AE') != -1:
				idf = 'E'
			elif config['channel'].find('AC') != -1:
				idf = 'C'
			else:
				print('error auto file selection 454356')
				sys.exit()
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == idf if f[-4:] == 'tdms']
			



		
		frequencies1 = [21.67, 7.67, 4.33, 312., 52., 12., 13., 299.01, 324.99, 286.02, 337.98, 273.03, 350.97, 8.67, 17.33, 4.34, 21.66, 307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65]
		frequencies2 = [43.34, 15.34, 8.66, 624., 104., 24., 26., 611.01, 636.99, 598.02, 649.98, 585.03, 662.97, 21.7, 30.33, 17.34, 34.66, 619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65]
		frequencies3 = [65.01, 23.01, 12.99, 936., 156., 36., 39., 923.01, 948.99, 910.02, 961.98, 897.03, 974.97, 34.67, 43.33, 30.34, 47.66, 931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65]
		frequencies4 = [86.68, 30.68, 17.32, 1248.0, 208., 48., 52.01, 1235.01, 1260.99, 1222.02, 1273.98, 1209.03, 1286.97, 47.67, 56.33, 43.341, 60.66, 1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35]
		frequencies5 = [108.35, 38.85, 21.65, 1560., 260., 60., 65., 1547.01, 1572.99, 1534.02, 1585.98, 1521.03, 1598.97, 60.67, 69.33, 56.34, 73.66, 1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65]
		all_frequencies_names = frequencies1 + frequencies2 + frequencies3 + frequencies4 + frequencies5
		
		#1300
		int_freqs = [4.33, 8.66, 12.99, 17.32, 21.65, 26.0, 30.33, 34.67, 39.00, 43.33]		
		int_freqs2 = [312., 624., 936., 1248., 1560.]
		int_freqs2 += [307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] +  [619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65] + [931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65] + [1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35] + [1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65]
		
		int_freqs2 += int_freqs
		
		# #1000
		# all_frequencies = [3.33, 6.66, 9.99, 13.32, 16.65, 19.98, 23.32, 26.65, 29.98, 33.21]
		# print('1000++++++')
			
		# #700
		# all_frequencies = [2.33, 4.66, 6.99, 9.33, 11.66, 13.99, 16.32, 18.65, 20.98, 23.32]
		# print('700++++++')
		
		# #400
		# all_frequencies = [1.33, 2.66, 4.00, 5.33, 6.66, 7.99, 9.33, 10.66, 11.99, 13.32]
		# print('400++++++')

		dict_freq = {}
		for freq_value in all_frequencies_names:
			dict_freq[str(freq_value)] = []
			
		# for freq_value in all_frequencies:
			# try:
				# print(dict_freq[str(freq_value)])
				# print(str(freq_value))
			# except:
				# dict_freq[str(freq_value)] = []
		
		best_results = {}
		for filepath in Filepaths:
			filename = basename(filepath)
			if filename.find('20181102') != -1:
				all_frequencies = list(np.array(all_frequencies_names)*1.023077)
				# int_freqs = list(np.array(int_freqs)*1.023077)
				int_freqs2 = list(np.array(int_freqs2)*1.023077)
				# freq_range = list(np.array([3., 45.])*1.023077)
				freq_range2 = list(np.array([3., 1600.])*1.023077)
			else:
				all_frequencies = all_frequencies_names
				# freq_range = list(np.array([3., 45.]))
				freq_range2 = list(np.array([3., 1600.]))
				
			signal = load_signal(filepath, channel=config['channel'])
			
			# if filepath.find('20181102') == -1:
				# signal = signal*10.
			
			signal = signal*1000./70.8
			# print('without amplification!')
			
			
			# lp, hp = genetic_optimal_filter_A(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=4, freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], inter=30)
			
			# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			# print(config['level'])
			# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			
			lp, hp = genetic_optimal_filter_C(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=config['parents'], freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], clusters=config['clusters'], filename=filename[:-5])
			
			# lp, hp = genetic_optimal_filter_B(x=signal, fs=config['fs'], levels=config['level'], num_generations=3, num_parents_mating=4, freq_values1=int_freqs, freq_range1=freq_range, freq_values2=int_freqs2, freq_range2=freq_range2, weight_mutation=0.1, inter=12)
			
			best_results[config['name'] + '_' + filename[:-5]] = [lp, hp]
			
			# save_pickle('best_gen_' + config['name'] + '_' + filename[:-5]+ '.pkl', [lp, hp])
			
			# lp, hp, max_kurt = Fast_Kurtogram_filters(signal, config['level'], config['fs'])
			# lp = 90.e3
			# hp = 150.e3
			
			# magSignal, f, df = mag_fft(signal, config['fs'])
			# pre_energy_band = np.sum((magSignal[int(lp/df) : int(hp/df)])**2.0)
			# signal = signal/pre_energy_band
			# 
			# 
			
			# signal = signal/(scipy.stats.kurtosis(signal, fisher=False))	
			# signal = signal/scipy.stats.kurtosis(signal, fisher=False)
			

			signal = butter_bandpass(signal, config['fs'], freqs=[lp, hp], order=3)
			
			# signal = signal**2.0
			
			
			
			
			
			
			
			# 
			
			

			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 125.e3], order=3)
			
			signal = hilbert_demodulation(signal)			
			# signal = butter_demodulation(x=signal, fs=config['fs'], filter=['lowpass', 2000., 3], prefilter=None, type_rect=None, dc_value=None)
			
			magX, f, df = mag_fft(x=signal, fs=config['fs'])

			

			
			for freq_value, freq_name in zip(all_frequencies, all_frequencies_names):
				vv = amp_component_zone(X=magX, df=df, freq=freq_value, tol=2.0)
				# print(vv)
				dict_freq[str(freq_name)].append(vv)
			# plt.plot(f, magX)
			# plt.show()

		
		# for freq_value in all_frequencies:
			# f_mean = np.mean(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_mean)

		

		# for freq_value in all_frequencies:
			# f_std = np.std(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_std)

		save_pickle('best_gen_' + config['name'] + '.pkl', best_results)

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')

		for freq_name in all_frequencies_names:
			mydict[str(freq_name)] = dict_freq[str(freq_name)]

		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		# DataFr.to_excel(writer, sheet_name='Envelope_Fft')
		DataFr.to_excel(writer, sheet_name='Fft')			
		print('Result in Excel table')
	
	
	elif config['mode'] == 'new_long_analysis_features_imf':
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
	
		# intervals, Intervals_Dict, str_intervals = Lgenerate_n_intervals_noinf(50.e3, 450.e3, 20)
		# intervals, Intervals_Dict, str_intervals = Lgenerate_n_intervals_noinf(3., 45., 3)
		# intervals, Intervals_Dict, str_intervals = Lgenerate_n_intervals_noinf(280., 1590., 4)
		
		#1300
		intervals = [3., 45., 280., 1590.]
		Intervals_Dict = {'0/3.0_45.0':[], '1/45.0_280.0':[], '2/280.0_1590.0':[]}
		str_intervals = ['0/3.0_45.0', '1/45.0_280.0', '2/280.0_1590.0']
		
		# #1000
		# intervals = [2.3, 34.6, 215.4, 1223.1]
		# Intervals_Dict = {'0/2.3_34.6':[], '1/34.6_215.4':[], '2/215.4_1223.1':[]}
		# str_intervals = ['0/2.3_34.6', '1/34.6_215.4', '2/215.4_1223.1']
		
		# #700
		# intervals = [1.6, 24.2, 150.7, 856.2]
		# Intervals_Dict = {'0/1.6_24.2':[], '1/24.2_150.7':[], '2/150.7_856.2':[]}
		# str_intervals = ['0/1.6_24.2', '1/24.2_150.7', '2/150.7_856.2']
		
		# #400
		# intervals = [0.9, 13.9, 86.1, 489.2]
		# Intervals_Dict = {'0/0.9_13.9':[], '1/13.9_86.1':[], '2/86.1_489.2':[]}
		# str_intervals = ['0/0.9_13.9', '1/13.9_86.1', '2/86.1_489.2']
		
		print(intervals, Intervals_Dict, str_intervals)
		# intervals, Intervals_Dict, str_intervals = Lgenerate_n_intervals(50.e3, 450.e3, 20)
		
		# print(intervals)
		# sys.exit()
		for filepath in Filepaths:
			signal = load_signal(filepath, channel=config['channel'])
			filename = basename(filepath)
			if filename.find('h1') == -1:
				signal = signal*1000.
			
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=20.e3, order=3)
			signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			
			# signal = signal**2.0
			
			signal = hilbert_demodulation(signal)
			

			
			
			# signal = butter_demodulation(x=signal, fs=config['fs'], filter=['lowpass', 2000., 3], prefilter=['highpass', 20.e3, 3], type_rect=None, dc_value=None)
			
			
			# signal = hilbert_demodulation(butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 125.e3], order=3))
			magX, f, df = mag_fft(x=signal, fs=config['fs'])
			
			for i in range(len(intervals)-1):				
				Intervals_Dict[str_intervals[i]].append(sum_in_band(magX, df, intervals[i], intervals[i+1]))
			
			# print(Intervals_Dict)
			# plt.plot(f, magX, 'r')
			# plt.show()

		
		# sys.exit()
		for key in Intervals_Dict:
			Intervals_Dict[key].append(np.mean(Intervals_Dict[key]))
			Intervals_Dict[key].append(np.std(Intervals_Dict[key]))

		

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		row_names.append('Mean')
		row_names.append('Std')
		
		for element in str_intervals:
			mydict[element] = Intervals_Dict[element]



		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='AE_Fft')	
		print('Result in Excel table')	

	
	elif config['mode'] == 'long_analysis_features_2':
		MASTER_FILEPATH = []
		# Channels = ['AE_1', 'AE_2', 'AE_3', 'AE_4']
		# Channels = ['AE_1', 'AE_2', 'AE_3']
		Channels = ['AE_0']
		# ref_dbAE = 1.e-6
		# amp_factor = 43.
		ini_count = 0
		
		for count in range(config['n_batches']):
			print('Select Batch ', count+ini_count)
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			MASTER_FILEPATH.append(Filepaths)
			root.destroy()
		
		# import time
		from time import time
		start_time = time()

		for count in range(config['n_batches']):
			Filepaths = MASTER_FILEPATH[count]
			
			print('Processing Batch ', count+ini_count)			
			row_names = [basename(filepath) for filepath in Filepaths]
			save_pickle('names_batch_' + str(count+ini_count) + '.pkl', row_names)
			for channel in Channels:
				# Data = [load_signal(filepath, channel=channel) for filepath in Filepaths]
				Data = [butter_highpass(x=load_signal(filepath, channel=channel), fs=config['fs'], freq=80.e3, order=3) for filepath in Filepaths]
				# for i in range(len(Data)):
					# Data[i] = butter_highpass(x=Data[i], fs=config['fs'], freq=80.e3, order=3)
				# RMS = [signal_rms(signal) for signal in Data]
				MAX = [np.max(np.absolute(signal)) for signal in Data]
				# save_pickle('rms_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', RMS)
				save_pickle('max_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', MAX)
				
				
				
				# Data_dBAE = []
				# for signal in Data:
					# signal_dBAE = np.zeros(len(signal))
					# for i in range(len(signal)):
						# current_input_V = np.absolute(signal[i]/(10.**(amp_factor/20.)))
						# signal_dBAE[i] = 20*np.log10(current_input_V/ref_dbAE)

					# Data_dBAE.append(signal_dBAE)
				# RMS_dBAE = [signal_rms(signal) for signal in Data_dBAE]
				# MAX_dBAE = [np.max(signal) for signal in Data_dBAE]
				
				# save_pickle('rms_dBAE_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', RMS_dBAE)
				# save_pickle('max_dBAE_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', MAX_dBAE)
				
				mydict = {}			
				
				
				
				
				# mydict['RMS'] = RMS
				mydict['MAX'] = MAX	
				# mydict['RMS_dBAE'] = RMS_dBAE
				# mydict['MAX_dBAE'] = MAX_dBAE			
				DataFr = pd.DataFrame(data=mydict, index=row_names)
				# writer = pd.ExcelWriter('to_use_batch_' + str(count+ini_count) + '_channel_' + channel + '.xlsx')
				writer = pd.ExcelWriter('Batch_' + str(count+ini_count) + '_Features_OK_' + channel + '.xlsx')				
				DataFr.to_excel(writer, sheet_name='Sheet1')
				print('Result in Excel table')
			
			# mean_mag_fft = read_pickle('mean_5_fft.pkl')
			# corrcoefMAGFFT = [np.corrcoef(mag_fft(signal, config['fs'])[0], mean_mag_fft) for signal in Data]
			# save_pickle('fftcorrcoef_batch_' + str(count) + '.pkl', corrcoefMAGFFT)
			
		print("--- %s seconds ---" % (time() - start_time))
	
			
			
			# plt.boxplot(RMS)
			# plt.show()
	
	
	elif config['mode'] == 'long_analysis_plot':
		# RMS_long = [[] for i in range(config['n_batches'])]
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()
		# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		# RMS_long = [read_pickle(filepath) for filepath in Filepaths]
		
		
		# rows = table.axes[0].tolist()		
		# max_V = table['MAX'].values
		# rms_V = table['RMS'].values
		
		Filepaths = []
		for i in range(2):
			# root = Tk()
			# root.withdraw()
			# root.update()
			# filepath = filedialog.askopenfilename()			
			# root.destroy()
			
			i = i + 22
			base = 'M:\Betriebsmessungen\WEA-Getriebe Eickhoff\Durchführung\Auswertung\Batch_' + str(i)
			filepath = join(base, 'Batch_' + str(i) + '_Features_OK_AE_1.xlsx')
			
			
			Filepaths.append(filepath)
		
		
		FEATURE = []
		for filepath in Filepaths:
			table = pd.read_excel(filepath)	
			FEATURE.append(table['MAX'].values)
		
		FEATURE = np.array(FEATURE)
		FEATURE = (FEATURE / 281.8) * 1000
		FEATURE = FEATURE.tolist()
		# save_pickle('rms_batch.pkl', RMS)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.boxplot(FEATURE)
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
		
		ax.set_title('AE_1', fontsize=12)
		
		# ax.set_xticklabels(['M=25%\nn=100%', 'M=50%\nn=100%', 'M=75%\nn=100%', 'M=100%\nn=25%', 'M=100%\nn=50%', 'M=100%\nn=75%', 'M=100%\nn=100%'])
		# ax.set_xticklabels(['Point A', 'Point B', 'Point C', 'Point D', 'Point E', 'Point F'])
		# ax.set_xticklabels(['Tooth 1', 'Tooth 2'])
		# ax.set_xticklabels(['Max High', 'Min High'])
		ax.set_xticklabels(['Point E', 'Point F'])
		# for label in ax.get_xmajorticklabels():
			# label.set_rotation(45)
			# label.set_horizontalalignment("right")

		# ax.set_xticklabels(['14:06', '14:16', '14:30', '15:00', '15:30', '16:00', '16:20', '17:00', '17:30', '18:00', '18:30', '19:00', '23:30', '00:03'])
		# ax.set_xlabel('Time on 20171020')
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax.set_ylabel('Max. Amplitude (m$V_{in}$)')
		# ax.set_ylim(bottom=1.e-4, top=4.e-4)
		
		# plt.boxplot(RMS_long)
		plt.show()
	
	elif config['mode'] == 'new_long_analysis_plot':
		# RMS_long = [[] for i in range(config['n_batches'])]
		print('Select table with features: ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		# feature = read_pickle(filepath)
		
		# amp_factor = input('Input amplification factor dB: ')
		# amp_factor = float(amp_factor)
		fig, ax = plt.subplots(nrows=1, ncols=1)
		for filepath in Filepaths:
			table = pd.read_excel(filepath)	
			rows = table.axes[0].tolist()

			
			feature = table[config['name_feature']].values
			# feature = table['RMS'].values
			
			if config['db'] == 49:
				feature = (feature / 281.8) * 1000
			elif config['db'] == 43:
				feature = (feature / 141.25) * 1000
			elif config['db'] == 0:
				feature = (feature / 1.) * 1000
			
			if config['name_feature'] == 'MAX':
				print('WITH AVERAGE MOVIL')
				movil_avg = 5
				for i in range(len(feature)):
					if i >= movil_avg:
						count = 0
						for k in range(movil_avg):
							count = count + feature[i-k]
						feature[i] = count / movil_avg
				
				for i in range(movil_avg):
					feature[i] = (feature[i] + feature[movil_avg] + feature[movil_avg+1] + feature[movil_avg+2])/4.
					
					
					
			
			# print(rms_V)
			# print(max_V)
			# sys.exit()
			
			# times = [rows[i][25:31] for i in range(len(rows))]
			times = [rows[i][28:34] for i in range(len(rows))]
			times = [time[0:2] + ":" + time[2:4] + ":" + time[4:6] for time in times]
			
			filename = basename(filepath)
			index = filename.find('.')
			label = filename[index-4:index]
			label = label.replace('_', '-')
			ax.plot(feature, label=label)
			# ax.plot(feature, label=label)
			# ax.plot(feature, label=label)
		
		
		ax.xaxis.grid(True)
		plt.grid()
		
		
		divisions = config['divisions']
		ax.set_xticks( [i*divisions for i in range(int(len(times)/divisions))] + [len(times)-1])
		# ax.set_xticklabels(times)
		ax.set_xticklabels( [times[i*divisions] for i in range(int(len(times)/divisions))] + [times[len(times)-1]])
		ax.tick_params(axis='y', labelsize=14)
		ax.tick_params(axis='x', labelsize=13)
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		
		if config['name_feature'] == 'RMS':
			ax.set_ylabel('RMS-Wert [m$V_{in}$]', fontsize=16)
		elif config['name_feature'] == 'MAX':
			ax.set_ylabel('Max. Amplitude [m$V_{in}$]', fontsize=16)
		elif config['name_feature'] == 'BPS':
			ax.set_ylabel('1000 x Bursts per Second [-]', fontsize=16)

		
		for label in ax.get_xmajorticklabels():
			label.set_rotation(37.5)
			label.set_horizontalalignment("right")
		
		
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()		
		# root.destroy()
		# table = pd.read_excel(filepath)	
		# rows = table.axes[0].tolist()	
		

		# externe = table[config['name_condition']].values
		
		# ax2 = ax.twinx()
		# ax2.plot(externe, 'om')
		# ax2.tick_params(axis='y', labelsize=14)
		

		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)

		
		# if config['name_condition'] == 'T':
			# ax2.set_ylabel('Temperatur [°C]', color='m', fontsize=16)
		# elif config['name_condition'] == 'n':
			# ax2.set_ylabel('$Drehzahl_{out}$ $[min^{-1}]$', color='m', fontsize=16)
		# elif config['name_condition'] == 'M':
			# ax2.set_ylabel('$Drehmoment_{out}$ [kNm]', color='m', fontsize=16)

		# ax2.tick_params('y', colors='m')
		# fig.set_size_inches(8, 6)
		# ax.legend(fontsize=11.5, loc='best')
		# # ax.set_ylim(bottom=0, top=0.75)
		# # ax.set_ylim(bottom=0, top=45)
		# # ax2.set_ylim(bottom=25., top=55.)
		# plt.tight_layout()
		plt.show()

		

	elif config['mode'] == 'mean_mag_fft':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		meanFFT = np.zeros(len(Data[0])/2)
		
		for k in range(len(Data)):
			magX, f, df = mag_fft(Data[k], config['fs'])
			meanFFT = meanFFT + magX
		meanFFT = meanFFT / len(Data)
		
		save_pickle('mean_5_fft.pkl', meanFFT)
		
		plt.plot(meanFFT)
		plt.show()
	
	elif config['mode'] == 'plot_extended_signal':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		Extended_Signal = []
		
		count = 0
		for filepath in Filepaths:
			# Extended_Signal += list(butter_lowpass(x=load_signal(filepath, channel=config['channel']), fs=config['fs'], freq=5., order=3))
			if filepath.find('210405') != -1:
				print(count)
			Extended_Signal += list(load_signal(filepath, channel=config['channel']))
			count += 1
		
		Extended_Signal = butter_lowpass(x=np.array(Extended_Signal), fs=config['fs'], freq=5., order=3)
		
		t = [i/config['fs'] + 210. for i in range(len(Extended_Signal))]
		fig, ax = plt.subplots()
		ax.plot(t, np.array(Extended_Signal)*10.)
		ax.set_xlabel('Dauer [s]', fontsize=12)
		ax.set_ylabel('Kraft [kN]', fontsize=12)
		ax.tick_params(axis='both', labelsize=11)
		
		plt.show()

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
	
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	config['n_batches'] = int(config['n_batches'])
	config['level'] = int(config['level'])
	config['db'] = int(config['db'])
	config['divisions'] = int(config['divisions'])
	
	config['parents'] = int(config['parents'])
	config['clusters'] = int(config['clusters'])
	config['generations'] = int(config['generations'])
	config['mutation'] = float(config['mutation'])
	# Variable conversion
	return config


	
if __name__ == '__main__':
	main(sys.argv)
