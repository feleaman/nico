# import os
from os import listdir
import matplotlib.pyplot as plt

from Kurtogram3 import Fast_Kurtogram_filters
from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('D:'))
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
from M_Wavelet import *

# from THR_Burst_Detection import full_thr_burst_detector
# from THR_Burst_Detection import read_threshold
# from THR_Burst_Detection import plot_burst_rev

Inputs = ['mode', 'fs', 'channel']
InputsOpt_Defaults = {'n_batches':1, 'name_feature':'RMS', 'name_condition':'T', 'db':0, 'divisions':10, 'name':'name', 'mypath':None, 'level':5, 'parents':4, 'clusters':20, 'mutation':0.1, 'generations':3, 'wv_deco':'OFF', 'wv_mother':'db1', 'wv_crit':'kurt_sen', 'wv_approx':'OFF', 'db_out':'OFF', 'units':'v', 'extension':'dms', 'plot':'OFF', 'filter':'OFF', 'freq_lp':'OFF', 'freq_hp':'OFF', 'sqr_envelope':'OFF', 'filter_path':None, 'features_g1':'OFF', 'features_g2':'OFF', 'features_g3':'OFF', 'features_g4':'OFF', 'features_g5':'OFF', 'demodulation':'OFF', 'fourier':'ON', 'calc_range':'avg', 'binary_count':'OFF', 'times_std':2., 'range':None, 'idx_levels':None, 'add_mean_std':'OFF', 'norm_max':'OFF'}

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
		STD = []
		MAX = []
		KURT = []

		for filepath in Filepaths:
			
			signal = load_signal(filepath, channel=config['channel'])
			
			signal = hilbert_demodulation(signal) #!!!!!!!!!!!!!!!!!!!!!!!!!!!
			signal = signal/np.max(signal) #!!!!!!!!!!!!!!!!!!!!!!!!!!!

			STD.append(np.std(signal))
			MAX.append(np.max(signal))
			KURT.append(scipy.stats.kurtosis(signal, fisher=False))
			
			

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		mydict['STD'] = STD
		mydict['MAX'] = MAX
		mydict['KURT'] = KURT


		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		
		#writer = pd.ExcelWriter('TEST_Result' + '.xlsx')
		#DataFr.to_excel(writer, sheet_name='Sheet1')	
		DataFr.to_excel('CC_ENV_nmax_OvFeatures_GUT.xlsx')	
		
		print('Result in Excel table')
		
	
	
	elif config['mode'] == 'overall_features':
		if config['mypath'] == None:
			print('Select signals...')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
			
			if config['channel'].find('AE') != -1:
				idf = 'E'
			elif config['channel'].find('AC') != -1:
				idf = 'D'
			
		else:
			if config['channel'].find('AE') != -1:
				idf = 'E'
				Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == idf if f[-4:] == 'tdms']
			elif config['channel'].find('AC') != -1:
				idf = 'D'
				Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == idf if f[-4:] == 'tdms']
			else:
				print('error auto file selection 454356')
				idf = 'NN'
				Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f))]
			
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-3:] == config['extension']]
			
		
		
		MAX = []
		RMS = []
		CREST = []		
		KURT = []
		SEN = []		
		
		IIN = []
		CLF = []
		EOP = []
		STD = []
		
		AVF = []
		CEF = []
		RMF = []
		STF = []
		
		LP12 = []
		LP13 = []
		LP16 = []
		LP17 = []
		LP21 = []
		LP24 = []
		
		LP12F = []
		LP13F = []
		LP16F = []
		LP17F = []
		LP21F = []
		LP24F = []

		
		for filepath in Filepaths:
			# print()
			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if idf == 'E':
				if config['db_out'] != 'OFF':
					if config['db_out'] == 37:
						signal = signal/70.8
					elif config['db_out'] == 49:
						print('db Schottland mit Lack!')
						signal = signal/281.8
					elif config['db_out'] == 43:
						print('db out bochum')
						signal = signal/141.25
				
				if config['units'] != 'v':
					if config['units'] == 'uv':
						signal = signal*1000.*1000.
					elif config['units'] == 'mv':
						signal = signal*1000.
			elif idf == 'D':
				if filepath.find('20180227') != -1:
					signal = signal*10.
					print('AC signal was multiplied by 10')
				else:
					print('AC signal already in g')
			else:
				print('no sensibility')
			
			# Wavelet
			if config['wv_deco'] == 'DWT':
				if config['wv_crit'] != 'avg_mpr':
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'])
					print('best level = ', best_lvl)
				else:
					print('not implemented here!!')
					sys.exit()
			elif config['wv_deco'] == 'WPD':
				print('Wavelet Packet!!')
				if config['wv_crit'] == 'inverse_levels':
					print('Inverse Wavelet Multi Levels')
					signal = return_iwv_PACKET_fix_levels(x=signal, fs=config['fs'], max_level=config['level'], mother_wv=config['wv_mother'], idx_levels=config['idx_levels'])
				elif config['wv_crit'] == 'coef_one_level':
					print('Wavelet Coefs. One Levels')
					signal = return_wv_PACKET_one_level(x=signal, fs=config['fs'], max_level=config['level'], mother_wv=config['wv_mother'], idx_level=config['idx_levels'])
				elif config['wv_crit'] == 'kurt':
					signal, best_level_idx, new_fs = return_best_wv_level_idx_PACKET(x=signal, fs=config['fs'], sqr='OFF', levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx='OFF', freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None)
			
			# Filter
			if config['filter'] != 'OFF':
				signal = multi_filter(signal, config, filename=None)
			#Sqr
			if config['sqr_envelope'] == 'ON':
				signal = signal**2.0
			

			n = len(signal)
			
			if config['demodulation'] == 'ON':
				print('envelope calculation!')
				signal = hilbert_demodulation(signal)
			
			if config['wv_deco'] == 'DWT' or config['wv_deco'] == 'WPD':
				print('info: wavelet affects FFT fs')
				Magnitude, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!
			else:
				Magnitude, f, df = mag_fft(x=signal, fs=config['fs'])
			

			
			# Magnitude, f, df = mag_fft(signal, fs=config['fs'])
			
			value_max = np.max(np.absolute(signal))
			print('max ', value_max)
			value_min = np.min(signal)
			print('min ', value_min)
			abs_max = np.max(np.array([np.absolute(value_max), np.absolute(value_min)]))
			
			
			if config['norm_max'] == 'ON':
				signal = signal / abs_max
				print('max normed!!')
			
			if config['features_g1'] == 'ON':
				value_rms = signal_rms(signal)	
				
				
					
				px = ((value_rms)**2.0)*n
				ent = 0.
				for i in range(n):
					# if (signal[i]**2.0)/px > 1.e-15:						
					ent += ((signal[i]**2.0)/px)*np.log2((signal[i]**2.0)/px)
				ent = - ent	
				
				MAX.append(abs_max)
				RMS.append(value_rms)
				CREST.append(abs_max/value_rms)
				KURT.append(scipy.stats.kurtosis(signal, fisher=False))
				SEN.append(ent)
			else:
				print('no_group_1')
				MAX.append(0)
				RMS.append(0)
				CREST.append(0)
				KURT.append(0)
				SEN.append(0)
			
			
			if config['features_g2'] == 'ON':
			
				resignal = np.zeros(n)
				for i in range(n):
					if i == 0:
						resignal[i] = signal[i]**2.0 - signal[n-1]*signal[i+1]
					elif i == n-1:
						resignal[i] = signal[i]**2.0 - signal[i-1]*signal[0]
					else:
						resignal[i] = signal[i]**2.0 - signal[i-1]*signal[i+1]
				
				sum_io = 0.
				sum_cl = 0.
				for i in range(n):
					sum_io += np.absolute(signal[i])
					sum_cl += (np.absolute(signal[i]))**0.5
				sum_io = sum_io / n
				sum_cl = (sum_cl / n)**0.5
					
				
				IIN.append(abs_max/sum_cl)
				CLF.append(abs_max/sum_io)
				EOP.append(scipy.stats.kurtosis(resignal, fisher=False))
				STD.append(np.std(signal))
			else:
				print('no_group_2')
				IIN.append(0)
				CLF.append(0)
				EOP.append(0)
				STD.append(0)
			
			
			if config['features_g3'] == 'ON':
				
				
				nf = len(Magnitude)
				sum_mag = np.sum(Magnitude)
				
				sum_fc = 0.
				sum_rmsf = 0.
				for i in range(nf):
					sum_fc += Magnitude[i]*df*i
					sum_rmsf += Magnitude[i]*(df*i)**2.
				cef = sum_fc/sum_mag
				
				sum_stf = 0.
				for i in range(nf):
					sum_stf += Magnitude[i]*(df*i-cef)**2.0			
				
				
				AVF.append(sum_mag/nf)			
				CEF.append(cef)
				RMF.append((sum_rmsf/sum_mag)**0.5)
				STF.append((sum_stf/sum_mag)**0.5)
			
			else:
				print('no_group_3')
				AVF.append(0)			
				CEF.append(0)
				RMF.append(0)
				STF.append(0)
			
			
			
			
			if config['features_g4'] == 'ON':
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
			else:
				print('no_group_4')
				LP12.append(0)	
				LP13.append(0)	
				LP16.append(0)			
				LP17.append(0)
				LP21.append(0)
				LP24.append(0)
			
			
			
			if config['features_g5'] == 'ON':
				lp12F = lout_featp12(Magnitude)
				LP12F.append(lp12F)
				
				lp13F = lout_featp13(Magnitude, lp12F)
				LP13F.append(lp13F)
				
				lp16F = lout_featp16(Magnitude, dp=df)
				LP16F.append(lp16F)
				
				lp17F = lout_featp17(Magnitude, dp=df, lp16=lp16F)
				LP17F.append(lp17F)
				
				lp21F = lout_featp21(Magnitude, dp=df, lp16=lp16F, lp17=lp17F)
				LP21F.append(lp21F)
				
				lp24F = lout_featp24(Magnitude, dp=df, lp16=lp16F, lp17=lp17F)
				LP24F.append(lp24F)
			else:
				print('no_group_5')
				LP12F.append(0)	
				LP13F.append(0)	
				LP16F.append(0)			
				LP17F.append(0)
				LP21F.append(0)
				LP24F.append(0)

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		
		
		mydict['MAX'] = MAX
		mydict['RMS'] = RMS
		mydict['CREST'] = CREST
		mydict['KURT'] = KURT
		mydict['SEN'] = SEN
		
		
		mydict['IIN'] = IIN
		mydict['CLF'] = CLF
		mydict['EOP'] = EOP
		mydict['STD'] = STD
		
		mydict['AVF'] = AVF 
		mydict['CEF'] = CEF
		mydict['RMF'] = RMF
		mydict['STF'] = STF
		
		mydict['LP12'] = LP12
		mydict['LP13'] = LP13
		mydict['LP16'] = LP16
		mydict['LP17'] = LP17
		mydict['LP21'] = LP21
		mydict['LP24'] = LP24
		
		mydict['LP12F'] = LP12F
		mydict['LP13F'] = LP13F
		mydict['LP16F'] = LP16F
		mydict['LP17F'] = LP17F
		mydict['LP21F'] = LP21F
		mydict['LP24F'] = LP24F
		
		if config['add_mean_std'] == 'ON':	
			row_names.append('Mean')
			row_names.append('Std')
			for key in mydict.keys():
				mean = np.mean(mydict[key])
				std = np.std(mydict[key])
				mydict[key].append(mean)
				mydict[key].append(std)
		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		#writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(config['name'] + '.xlsx')	
		print('Result in Excel table')
		
	
	elif config['mode'] == 'overall_features_CORR':
		if config['mypath'] == None:
			print('Select signals...')
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
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-3:] == config['extension']]
			
		
		
		MAX = []
		RMS = []
		CREST = []		
		KURT = []
		SEN = []		
		
		IIN = []
		CLF = []
		EOP = []
		STD = []
		
		AVF = []
		CEF = []
		RMF = []
		STF = []
		
		LP12 = []
		LP13 = []
		LP16 = []
		LP17 = []
		LP21 = []
		LP24 = []
		
		LP12F = []
		LP13F = []
		LP16F = []
		LP17F = []
		LP21F = []
		LP24F = []

		
		for filepath in Filepaths:
			
			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					signal = signal/70.8
				elif config['db_out'] == 49:
					print('db Schottland mit Lack!')
					signal = signal/281.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			
			# Wavelet
			if config['wv_deco'] == 'DWT':
				if config['wv_crit'] != 'avg_mpr':
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'], int_points=True)
					print('best level = ', best_lvl)
				else:
					print('not implemented here!!')
					sys.exit()
			elif config['wv_deco'] == 'WPD':
				print('Wavelet Packet!!')
				if config['wv_crit'] == 'inverse_levels':
					print('Inverse Wavelet Multi Levels')
					signal = return_iwv_PACKET_fix_levels(x=signal, fs=config['fs'], max_level=config['level'], mother_wv=config['wv_mother'], idx_levels=config['idx_levels'])
				elif config['wv_crit'] == 'coef_one_level':
					print('Wavelet Coefs. One Levels')
					signal = return_wv_PACKET_one_level(x=signal, fs=config['fs'], max_level=config['level'], mother_wv=config['wv_mother'], idx_level=config['idx_levels'])
			
			# Filter
			if config['filter'] != 'OFF':
				signal = multi_filter(signal, config, filename=None)
			#Sqr
			if config['sqr_envelope'] == 'ON':
				signal = signal**2.0
			

			n = len(signal)
			
			if config['demodulation'] == 'ON':
				print('envelope calculation!')
				signal = hilbert_demodulation(signal)
			
			if config['wv_deco'] == 'DWT':
				print('info: wavelet affects FFT fs')
				Magnitude, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!
			else:
				Magnitude, f, df = mag_fft(x=signal, fs=config['fs'])
			

			
			# Magnitude, f, df = mag_fft(signal, fs=config['fs'])
			
			value_max = np.max(np.absolute(signal))
			value_min = np.min(signal)
			abs_max = np.max(np.array([np.absolute(value_max), np.absolute(value_min)]))
			
			
			if config['features_g1'] == 'ON':
				# value_rms = signal_rms(signal)	
				
				
					
				# px = ((value_rms)**2.0)*n
				# ent = 0.
				# for i in range(n):
					# # if (signal[i]**2.0)/px > 1.e-15:						
					# ent += ((signal[i]**2.0)/px)*np.log2((signal[i]**2.0)/px)
				# ent = - ent	
				
				# MAX.append(abs_max)
				# RMS.append(value_rms)
				# CREST.append(abs_max/value_rms)
				# KURT.append(scipy.stats.kurtosis(signal, fisher=False))
				# SEN.append(ent)
				
				MAX.append(0)
				RMS.append(0)
				CREST.append(0)
				KURT.append(0)
				SEN.append(0)
				
			else:
				print('no_group_1')
				MAX.append(0)
				RMS.append(0)
				CREST.append(0)
				KURT.append(0)
				SEN.append(0)
			
			
			if config['features_g2'] == 'ON':
			
				# resignal = np.zeros(n)
				# for i in range(n):
					# if i == 0:
						# resignal[i] = signal[i]**2.0 - signal[n-1]*signal[i+1]
					# elif i == n-1:
						# resignal[i] = signal[i]**2.0 - signal[i-1]*signal[0]
					# else:
						# resignal[i] = signal[i]**2.0 - signal[i-1]*signal[i+1]
				
				# sum_io = 0.
				# sum_cl = 0.
				# for i in range(n):
					# sum_io += np.absolute(signal[i])
					# sum_cl += (np.absolute(signal[i]))**0.5
				# sum_io = sum_io / n
				# sum_cl = (sum_cl / n)**0.5
					
				
				# IIN.append(abs_max/sum_cl)
				# CLF.append(abs_max/sum_io)
				# EOP.append(scipy.stats.kurtosis(resignal, fisher=False))
				# STD.append(np.std(signal))
				
				IIN.append(0)
				CLF.append(0)
				EOP.append(0)
				STD.append(0)
				
			else:
				print('no_group_2')
				IIN.append(0)
				CLF.append(0)
				EOP.append(0)
				STD.append(0)
			
			
			if config['features_g3'] == 'ON':
				
				
				# nf = len(Magnitude)
				# sum_mag = np.sum(Magnitude)
				
				# sum_fc = 0.
				# # sum_rmsf = 0.
				# for i in range(nf):
					# sum_fc += Magnitude[i]*df*i
					# # sum_rmsf += Magnitude[i]*(df*i)**2.
				# cef = sum_fc/sum_mag
				
				# sum_stf = 0.
				# for i in range(nf):
					# sum_stf += Magnitude[i]*(df*i-cef)**2.0			
				
				
				# AVF.append(sum_mag/nf)			
				# CEF.append(cef)
				CEF.append(0)
				# RMF.append((sum_rmsf/sum_mag)**0.5)
				# STF.append((sum_stf/sum_mag)**0.5)
				STF.append(0)
				
				AVF.append(0)	
				RMF.append(0)
				
			else:
				print('no_group_3')
				AVF.append(0)			
				CEF.append(0)
				RMF.append(0)
				STF.append(0)
			
			
			
			
			if config['features_g4'] == 'ON':

				
				LP12.append(0)	
				LP13.append(0)	
				LP16.append(0)			
				LP17.append(0)
				LP21.append(0)
				LP24.append(0)
				
				
			else:
				print('no_group_4')
				LP12.append(0)	
				LP13.append(0)	
				LP16.append(0)			
				LP17.append(0)
				LP21.append(0)
				LP24.append(0)
			
			
			
			if config['features_g5'] == 'ON':
				# lp12F = lout_featp12(Magnitude)
				# LP12F.append(lp12F)
				
				# lp13F = lout_featp13(Magnitude, lp12F)
				# LP13F.append(lp13F)
				
				lp16F = lout_featp16(Magnitude, dp=df)
				# LP16F.append(lp16F)
				
				
				LP12F.append(0)	
				LP13F.append(0)	
				LP16F.append(0)
				
				lp17F = lout_featp17(Magnitude, dp=df, lp16=lp16F)
				LP17F.append(lp17F)
				
				lp21F = lout_featp21(Magnitude, dp=df, lp16=lp16F, lp17=lp17F)
				LP21F.append(lp21F)
				
				lp24F = lout_featp24(Magnitude, dp=df, lp16=lp16F, lp17=lp17F)
				LP24F.append(lp24F)
				# LP24F.append(0)
			else:
				print('no_group_5')
				LP12F.append(0)	
				LP13F.append(0)	
				LP16F.append(0)			
				LP17F.append(0)
				LP21F.append(0)
				LP24F.append(0)

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]

		
		mydict['MAX'] = MAX
		mydict['RMS'] = RMS
		mydict['CREST'] = CREST
		mydict['KURT'] = KURT
		mydict['SEN'] = SEN
		
		
		mydict['IIN'] = IIN
		mydict['CLF'] = CLF
		mydict['EOP'] = EOP
		mydict['STD'] = STD
		
		mydict['AVF'] = AVF 
		mydict['CEF'] = CEF
		mydict['RMF'] = RMF
		mydict['STF'] = STF
		
		mydict['LP12'] = LP12
		mydict['LP13'] = LP13
		mydict['LP16'] = LP16
		mydict['LP17'] = LP17
		mydict['LP21'] = LP21
		mydict['LP24'] = LP24
		
		mydict['LP12F'] = LP12F
		mydict['LP13F'] = LP13F
		mydict['LP16F'] = LP16F
		mydict['LP17F'] = LP17F
		mydict['LP21F'] = LP21F
		mydict['LP24F'] = LP24F

		
		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='OV_Features')	
		print('Result in Excel table')
	
	
	elif config['mode'] == 'overall_features_select':
		if config['mypath'] == None:
			print('Select signals...')
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
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == idf if f[-4:] == 'tdms']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-3:] == config['extension']]
			
		
		
		RMS = []
		MAX = []
		# STD = []	
		# LP12 = [] #avg
		# LP13 = [] #var		
		# LP12F = [] #avs
		# LP17F = [] #p17
		# LP21F = [] #p21
		# LP24F = [] #p24

		
		for filepath in Filepaths:
			
			signal = load_signal(filepath, channel=config['channel'])
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					signal = signal/70.8
				elif config['db_out'] == 49:
					print('db Schottland mit Lack!')
					signal = signal/281.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			
			# Wavelet
			if config['wv_deco'] == 'ON':
				if config['wv_crit'] != 'avg_mpr':					
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'])					
				else:
					print('not implemented here!')
					sys.exit()		
			# Filter
			if config['filter'] != 'OFF':
				signal = multi_filter(signal, config, filename=None)
			#Sqr
			if config['sqr_envelope'] == 'ON':
				signal = signal**2.0
			

			n = len(signal)
			
			if config['demodulation'] == 'ON':
				print('envelope calculation!')
				signal = hilbert_demodulation(signal)
			
			if config['wv_deco'] == 'ON':
				print('info: wavelet affects FFT fs')
				Magnitude, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!
			else:
				Magnitude, f, df = mag_fft(x=signal, fs=config['fs'])
			
			
			
			
			value_max = np.max(np.absolute(signal))
			# value_min = np.min(signal)
			# abs_max = np.max(np.array([np.absolute(value_max), np.absolute(value_min)]))
			MAX.append(value_max)
			
			value_rms = signal_rms(signal)	
			RMS.append(value_rms)
			
			# STD.append(np.std(signal))
			
			
			# lp12 = lout_featp12(signal) #avg
			# # LP12.append(lp12)
			
			
			# lp13 = lout_featp13(signal, lp12) #var
			# LP13.append(lp13)
			
			
			# lp12F = lout_featp12(Magnitude) #avs
			# LP12F.append(lp12F)
			
			
			# lp16F = lout_featp16(Magnitude, dp=df)			
			# lp17F = lout_featp17(Magnitude, dp=df, lp16=lp16F) #p17
			# LP17F.append(lp17F)
			
			# lp21F = lout_featp21(Magnitude, dp=df, lp16=lp16F, lp17=lp17F) #p21
			# LP21F.append(lp21F)
			
			# lp24F = lout_featp24(Magnitude, dp=df, lp16=lp16F, lp17=lp17F) #p24
			# LP24F.append(lp24F)
			
		

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]

		
		mydict['RMS'] = RMS
		mydict['MAX'] = MAX
		# mydict['STD'] = STD		
		# mydict['AVG'] = LP12
		# mydict['VAR'] = LP13		
		# mydict['AVS'] = LP12F
		# mydict['P17'] = LP17F
		# mydict['P21'] = LP21F
		# mydict['P24'] = LP24F

		print(mydict)
		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='OV_Features')	
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

		
	elif config['mode'] == 'magnitude_freq_comps':
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

			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-3:] == config['extension']  if f[1] == idf]

		# #++++++++Schottland 1
		# all_frequencies_names = [3.872, 7.744, 11.615, 15.487, 19.359, 23.231, 27.103, 30.974, 34.847, 38.718] #ffr	
		# int_freqs2 = all_frequencies_names
		
		# #++++++++Bochum Z
		# all_frequencies_names = [3.48, 6.96, 10.44, 13.92, 17.4, 20.88, 24.36, 27.85, 31.32, 34.81] #ffr	
		# int_freqs2 = all_frequencies_names
		
		
		# #CWD-Damage
		# frequencies1 = [2.0714285, 0.55769, 0.3625, 35.8875, 5.1267857, 0.9201923, 1.0875, 35.525, 36.25, 35.1625, 36.6125, 34.8, 36.975, 34.4375, 37.3375, 34.075, 37.7, 33.7125, 38.0625]
		# frequencies2 = [4.142857, 1.11538, 0.725, 71.775, 10.2535714, 1.8403846, 2.175, 71.4125, 72.1375, 71.05, 72.5, 70.6875, 72.8625, 70.325, 73.225, 69.9625, 73.5875, 69.6, 73.95]
		# frequencies3 = [6.2142855, 1.67307, 1.0875, 107.6625, 15.3803571, 2.7605769, 3.2625, 107.3, 108.025, 106.9375, 108.3875, 106.575, 108.75, 106.2125, 109.1125, 105.85, 109.475, 105.4875, 109.8375]
		# frequencies4 = [8.285714, 2.23076, 1.45, 143.55, 20.5071428, 3.6807692, 4.35, 143.1875, 143.9125, 142.825, 144.275, 142.4625, 144.6375, 142.1, 145., 141.7375, 145.3625, 141.375, 145.725]
		# frequencies5 = [10.3571425, 2.78845, 1.8125, 179.4375, 25.6339285, 4.6009615, 5.4375, 179.075, 179.8, 178.7125, 180.1625, 178.35, 180.525, 177.9875, 180.8875, 177.625, 181.25, 177.2625, 181.6125]
		# frequencies6 = [2.175, 2.5375, 2.9, 3.2625, 3.625]

		# all_frequencies_names = frequencies1 + frequencies2 + frequencies3 + frequencies4 + frequencies5
		
		# int_freqs = [0.3625, 0.725, 1.0875, 1.45, 1.8125, 2.175, 2.5375, 2.9, 3.2625, 3.625]		
		# int_freqs2 = [35.8875, 71.775, 107.6625, 143.55, 179.4375]
		# int_freqs2 += [35.525, 36.25, 35.1625, 36.6125, 34.8, 36.975, 34.4375, 37.3375, 34.075, 37.7, 33.7125, 38.0625, 71.4125, 72.1375, 71.05, 72.5, 70.6875, 72.8625, 70.325, 73.225, 69.9625, 73.5875, 69.6, 73.95, 107.3, 108.025, 106.9375, 108.3875, 106.575, 108.75, 106.2125, 109.1125, 105.85, 109.475, 105.4875, 109.8375, 143.1875, 143.9125, 142.825, 144.275, 142.4625, 144.6375, 142.1, 145, 141.7375, 145.3625, 141.375, 145.725, 179.075, 179.8, 178.7125, 180.1625, 178.35, 180.525, 177.9875, 180.8875, 177.625, 181.25, 177.2625, 181.6125]		
		# int_freqs2 += int_freqs
		


		#TestBench
		frequencies1 = [21.67, 7.67, 4.33, 312., 52., 12., 13., 299.01, 324.99, 286.02, 337.98, 273.03, 350.97, 8.67, 17.33, 4.34, 21.66, 307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] #25
		frequencies2 = [43.34, 15.34, 8.66, 624., 104., 24., 26., 611.01, 636.99, 598.02, 649.98, 585.03, 662.97, 21.7, 30.33, 17.34, 34.66, 619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65]
		frequencies3 = [65.01, 23.01, 12.99, 936., 156., 36., 39., 923.01, 948.99, 910.02, 961.98, 897.03, 974.97, 34.67, 43.33, 30.34, 47.66, 931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65]
		frequencies4 = [86.68, 30.68, 17.32, 1248.0, 208., 48., 52.01, 1235.01, 1260.99, 1222.02, 1273.98, 1209.03, 1286.97, 47.67, 56.33, 43.341, 60.66, 1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35]
		frequencies5 = [108.35, 38.85, 21.65, 1560., 260., 60., 65., 1547.01, 1572.99, 1534.02, 1585.98, 1521.03, 1598.97, 60.67, 69.33, 56.34, 73.66, 1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65]
		all_frequencies_names = frequencies1 + frequencies2 + frequencies3 + frequencies4 + frequencies5
		
		
		
		# ffr_freqs = [4.33, 8.66, 12.99, 17.32, 21.65, 26.0, 30.33, 34.67, 39.00, 43.33] # 10
		# first_freqs = ffr_freqs
		# fg_freqs = [312., 624., 936., 1248., 1560.] # 5
		# sb_freqs = [307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] +  [619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65] + [931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65] + [1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35] + [1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65] + [299.01, 611.01, 923.01, 1235.01, 1547.01, 324.99, 636.99, 948.99, 1260.99, 1572.99] + [286.02, 598.02, 910.02, 1222.02, 1534.02, 337.98, 649.98, 961.98, 1273.98, 1585.98] #60
		# second_freqs = fg_freqs + sb_freqs
		# one_second_freqs = fg_freqs + sb_freqs + ffr_freqs
		
		# ffr_freqs = [4.33, 8.66, 13.00, 17.33, 21.67, 26.00, 30.33, 34.67, 39.00, 43.33, 47.67, 52.00, 56.33, 60.67, 65.00, 69.33, 73.67, 78.00, 82.33, 86.67, 91.00, 95.33, 99.67] # 23
		# first_freqs = ffr_freqs
		# fg_freqs = [312., 624., 936., 1248., 1560.] # 5
		# sb_freqs = [316.33, 320.67, 325.00, 329.33, 333.67, 338.00, 342.33, 346.67, 351.00, 355.33, 359.67, 364.00, 368.33, 372.67, 377.00, 381.33, 385.67, 390.00, 394.33, 398.67] + [628.33, 632.67, 637.00, 641.33, 645.67, 650.00, 654.33, 658.67, 663.00, 667.33, 671.67, 676.00, 680.33, 684.67, 689.00, 693.33, 697.67, 702.00, 706.33, 710.67] + [940.33, 944.67, 949.00, 953.33, 957.67, 962.00, 966.33, 970.67, 975.00, 979.33, 983.67, 988.00, 992.33, 996.67, 1001.00, 1005.33, 1009.67, 1014.00, 1018.33, 1022.67] + [1252.33, 1256.67, 1261.00, 1265.33, 1269.67, 1274.00, 1278.33, 1282.67, 1287.00, 1291.33, 1295.67, 1300.00, 1304.33, 1308.67, 1313.00, 1317.33, 1321.67, 1326.00, 1330.33, 1334.67] + [1564.33, 1568.67, 1573.00, 1577.33, 1581.67, 1586.00, 1590.33, 1594.67, 1599.00, 1603.33, 1607.67, 1612.00, 1616.33, 1620.67, 1625.00, 1629.33, 1633.67, 1638.00, 1642.33, 1646.67] + [225.33, 229.67, 234.00, 238.33, 242.67, 247.00, 251.33, 255.67, 260.00, 264.33, 268.67, 273.00, 277.33, 281.67, 286.00, 290.33, 294.67, 299.00, 303.33, 307.67] + [537.33, 541.67, 546.00, 550.33, 554.67, 559.00, 563.33, 567.67, 572.00, 576.33, 580.67, 585.00, 589.33, 593.67, 598.00, 602.33, 606.67, 611.00, 615.33, 619.67] + [849.33, 853.67, 858.00, 862.33, 866.67, 871.00, 875.33, 879.67, 884.00, 888.33, 892.67, 897.00, 901.33, 905.67, 910.00, 914.33, 918.67, 923.00, 927.33, 931.67] + [1161.33, 1165.67, 1170.00, 1174.33, 1178.67, 1183.00, 1187.33, 1191.67, 1196.00, 1200.33, 1204.67, 1209.00, 1213.33, 1217.67, 1222.00, 1226.33, 1230.67, 1235.00, 1239.33, 1243.67] + [1473.33, 1477.67, 1482.00, 1486.33, 1490.67, 1495.00, 1499.33, 1503.67, 1508.00, 1512.33, 1516.67, 1521.00, 1525.33, 1529.67, 1534.00, 1538.33, 1542.67, 1547.00, 1551.33, 1555.67] #200	
		
		# second_freqs = fg_freqs + sb_freqs
		# one_second_freqs = fg_freqs + sb_freqs + ffr_freqs
		# all_frequencies_names = fg_freqs + sb_freqs + ffr_freqs


		dict_freq = {}
		for freq_value in all_frequencies_names:
			dict_freq[str(freq_value)] = []
		
		best_results = {}
		for filepath in Filepaths:
			filename = basename(filepath)
			if filename.find('20181102') != -1:
				print('warning change frequencies')
				# if filename.find('20181102') != -1:
				all_frequencies = list(np.array(all_frequencies_names)*1.023077)
				one_second_freqs = list(np.array(one_second_freqs)*1.023077)
				one_second_range = list(np.array([3., 1700.])*1.023077)
				
				first_range = list(np.array([3., 100.])*1.023077)
				second_range = list(np.array([100., 1700.])*1.023077)
				first_freqs = list(np.array(first_freqs)*1.023077)
				second_freqs = list(np.array(second_freqs)*1.023077)
				
			else:
				all_frequencies = all_frequencies_names
				# freq_range = list(np.array([3., 45.]))
				one_second_range = list(np.array([3., 1700.])) # TestBench
				# freq_range2 = list(np.array([0.1, 190.])) # CWD Damage
				
				first_range = list(np.array([3., 100.]))
				second_range = list(np.array([100., 1700.]))
				
			signal = load_signal(filepath, channel=config['channel'])
			
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					signal = signal/70.8
			
			if config['units'] != 'OFF':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.

			
			# lp, hp = genetic_optimal_filter_A(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=4, freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], inter=30)
			
			# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			# print(config['level'])
			# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			
			# lp, hp = genetic_optimal_filter_C(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=config['parents'], freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], clusters=config['clusters'], filename=filename[:-5])
			
			# lp, hp = genetic_optimal_filter_B(x=signal, fs=config['fs'], levels=config['level'], num_generations=3, num_parents_mating=4, freq_values1=int_freqs, freq_range1=freq_range, freq_values2=int_freqs2, freq_range2=freq_range2, weight_mutation=0.1, inter=12)
			
			# best_results[config['name'] + '_' + filename[:-5]] = [lp, hp]
			
			# save_pickle('best_gen_' + config['name'] + '_' + filename[:-5]+ '.pkl', [lp, hp])
			
			# Wavelet
			if config['wv_deco'] == 'ON':
				if config['wv_crit'] != 'avg_mpr':
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'])
				else:
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'], freq_values=first_freqs, freq_range=first_range, freq_values_2=second_freqs, freq_range_2=second_range)
			
			
			if config['filter'] != 'OFF':
				signal = multi_filter(signal, config, filename=filename)
			
			if config['sqr_envelope'] == 'ON':
				signal = signal**2.0
				
			
			# lp, hp, max_kurt = Fast_Kurtogram_filters(signal, config['level'], config['fs'])
			# lp = 90.e3 #cyclo
			# hp = 150.e3 #cyclo
			
			# lp = 95.e3 #emd
			# hp = 140.e3 #emd

			
			# signal = signal/(scipy.stats.kurtosis(signal, fisher=False))	
			# signal = signal/scipy.stats.kurtosis(signal, fisher=False)
			
			

			

			# signal = butter_bandpass(signal, config['fs'], freqs=[lp, hp], order=3)


			
			
			# signal = signal**2.0 #SQR ANTES DE DEMODULATION
			if config['demodulation'] == 'ON':
				signal = hilbert_demodulation(signal)			
			
			if config['fourier'] == 'ON':
				if config['wv_deco'] == 'ON':
					print('info: wavelet affects FFT fs')
					magX, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!
				else:
					magX, f, df = mag_fft(x=signal, fs=config['fs'])
			else:
				print('Warning!!! No Fourier. Signal as Dict...')
				magX = signal['fft']
				f = signal['f']
				df = f[2] - f[1]

			# magX = 10*np.log10(magX/1.)

			
			for freq_value, freq_name in zip(all_frequencies, all_frequencies_names):
				vv = amp_component_zone(X=magX, df=df, freq=freq_value, tol=2.0)
				# print(vv)
				if config['binary_count'] != 'ON':
					dict_freq[str(freq_name)].append(vv)
				else:
					if (freq_value >= first_range[0]) and (freq_value < first_range[1]):
						print('freq in first range')
						intervals = first_range
					elif (freq_value >= second_range[0]) and (freq_value < second_range[1]):
						print('freq in second range')
						intervals = second_range
					else:
						print('error: freq in no range')
						sys.exit()
					avg = avg_in_band(magX, df, intervals[0], intervals[1])
					std = std_in_band(magX, df, intervals[0], intervals[1])
					limit = avg + config['times_std']*std
					if vv >= limit:
						dict_freq[str(freq_name)].append(1)
					else:
						dict_freq[str(freq_name)].append(0)	
						
						
			
			if config['plot'] == 'ON':
				plt.plot(f, magX)
				plt.show()



		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]


		for freq_name in all_frequencies_names:
			mydict[str(freq_name)] = dict_freq[str(freq_name)]

		# DataFr = pd.DataFrame(data=mydict, index=row_names)
		# writer = pd.ExcelWriter(config['name'] + '.xlsx')
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		# DataFr.to_excel(writer, sheet_name='Envelope_Fft')
		# DataFr.to_excel(writer, sheet_name='Fft')


		writer = pd.ExcelWriter(config['name'] + '.xlsx')
		DataFr_max = pd.DataFrame(data=mydict, index=row_names)				
		DataFr_max.to_excel(writer, sheet_name='Comps')
		writer.close()
		
		print('Result in Excel table')
	
	

	
	elif config['mode'] == 'new_long_analysis_features_freq_CWD_damage':
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
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[4] == idf if f[-3:] == 'pkl']
			
		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD
		#CWD-Damage
		# frequencies1 = [2.0714285, 0.55769, 0.3625, 35.8875, 5.1267857, 0.9201923, 1.0875, 35.525, 36.25, 35.1625, 36.6125, 34.8, 36.975, 34.4375, 37.3375, 34.075, 37.7, 33.7125, 38.0625]
		# frequencies2 = [4.142857, 1.11538, 0.725, 71.775, 10.2535714, 1.8403846, 2.175, 71.4125, 72.1375, 71.05, 72.5, 70.6875, 72.8625, 70.325, 73.225, 69.9625, 73.5875, 69.6, 73.95]
		# frequencies3 = [6.2142855, 1.67307, 1.0875, 107.6625, 15.3803571, 2.7605769, 3.2625, 107.3, 108.025, 106.9375, 108.3875, 106.575, 108.75, 106.2125, 109.1125, 105.85, 109.475, 105.4875, 109.8375]
		# frequencies4 = [8.285714, 2.23076, 1.45, 143.55, 20.5071428, 3.6807692, 4.35, 143.1875, 143.9125, 142.825, 144.275, 142.4625, 144.6375, 142.1, 145., 141.7375, 145.3625, 141.375, 145.725]
		# frequencies5 = [10.3571425, 2.78845, 1.8125, 179.4375, 25.6339285, 4.6009615, 5.4375, 179.075, 179.8, 178.7125, 180.1625, 178.35, 180.525, 177.9875, 180.8875, 177.625, 181.25, 177.2625, 181.6125]
		# frequencies6 = [2.175, 2.5375, 2.9, 3.2625, 3.625]
		
		frequencies1 = [0.3625, 35.8875, 5.1267857, 0.9201923, 1.0875, 35.525, 36.25, 35.1625, 36.6125, 34.8, 36.975, 34.4375, 37.3375, 34.075, 37.7, 33.7125, 38.0625]
		frequencies2 = [0.725, 71.775, 10.2535714, 1.8403846, 2.175, 71.4125, 72.1375, 71.05, 72.5, 70.6875, 72.8625, 70.325, 73.225, 69.9625, 73.5875, 69.6, 73.95]
		frequencies3 = [1.0875, 107.6625, 15.3803571, 2.7605769, 3.2625, 107.3, 108.025, 106.9375, 108.3875, 106.575, 108.75, 106.2125, 109.1125, 105.85, 109.475, 105.4875, 109.8375]
		frequencies4 = [1.45, 143.55, 20.5071428, 3.6807692, 4.35, 143.1875, 143.9125, 142.825, 144.275, 142.4625, 144.6375, 142.1, 145., 141.7375, 145.3625, 141.375, 145.725]
		frequencies5 = [1.8125, 179.4375, 25.6339285, 4.6009615, 5.4375, 179.075, 179.8, 178.7125, 180.1625, 178.35, 180.525, 177.9875, 180.8875, 177.625, 181.25, 177.2625, 181.6125]
		
		frequencies6 = [2.175, 2.5375, 2.9, 3.2625, 3.625]

		all_frequencies_names = frequencies1 + frequencies2 + frequencies3 + frequencies4 + frequencies5 + frequencies6
		
		print(len(all_frequencies_names))
		
		int_freqs = [0.3625, 0.725, 1.0875, 1.45, 1.8125, 2.175, 2.5375, 2.9, 3.2625, 3.625]		
		int_freqs2 = [35.8875, 71.775, 107.6625, 143.55, 179.4375]
		int_freqs2 += [35.525, 36.25, 35.1625, 36.6125, 34.8, 36.975, 34.4375, 37.3375, 34.075, 37.7, 33.7125, 38.0625, 71.4125, 72.1375, 71.05, 72.5, 70.6875, 72.8625, 70.325, 73.225, 69.9625, 73.5875, 69.6, 73.95, 107.3, 108.025, 106.9375, 108.3875, 106.575, 108.75, 106.2125, 109.1125, 105.85, 109.475, 105.4875, 109.8375, 143.1875, 143.9125, 142.825, 144.275, 142.4625, 144.6375, 142.1, 145, 141.7375, 145.3625, 141.375, 145.725, 179.075, 179.8, 178.7125, 180.1625, 178.35, 180.525, 177.9875, 180.8875, 177.625, 181.25, 177.2625, 181.6125]		
		int_freqs2 += int_freqs
		
		all_frequencies_names = int_freqs2
		
		print(len(int_freqs2))
		aaa = {str(caca):[] for caca in int_freqs2}
		print(aaa)
		print(len(list(aaa.keys())))
		# sys.exit()
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD
		

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

			all_frequencies = all_frequencies_names

			freq_range2 = list(np.array([0.1, 190.])) # CWD Damage
				
			signal = load_signal(filepath, channel=config['channel'])

			
			signal = signal*1000./70.8 #for AE
			
			
			# lp, hp = genetic_optimal_filter_A(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=4, freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], inter=30)
			
			#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD
			
			# lp, hp = genetic_optimal_filter_C(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=config['parents'], freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], clusters=config['clusters'], filename=filename[:-5])
			
			# lp, hp = genetic_optimal_filter_B(x=signal, fs=config['fs'], levels=config['level'], num_generations=3, num_parents_mating=4, freq_values1=int_freqs, freq_range1=freq_range, freq_values2=int_freqs2, freq_range2=freq_range2, weight_mutation=0.1, inter=12)
			
			# best_results[config['name'] + '_' + filename[:-5]] = [lp, hp]
			
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
			
			
			# # Wavelet
			# signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], levels=config['level'], mother_wv=config['mother_wv'], freq_values=int_freqs2, freq_range=freq_range2)
			

			# signal = butter_bandpass(signal, config['fs'], freqs=[lp, hp], order=3)
			

			#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD
			
			signal = signal**2.0 #SQR ANTES DE DEMODULATION
			# signal = signal /(np.max(signal) - np.min(signal))
			
			signal = hilbert_demodulation(signal)			
			
			magX, f, df = mag_fft(x=signal, fs=config['fs'])
			# magX, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!

			

			
			for freq_value, freq_name in zip(all_frequencies, all_frequencies_names):
				vv = amp_component_zone(X=magX, df=df, freq=freq_value, tol=2.0)
				# vv = amp_component(X=magX, df=df, freq=freq_value)
				# print(vv)
				dict_freq[str(freq_name)].append(vv)
			# plt.plot(f, magX)
			# plt.show()

		
		# for freq_value in all_frequencies:
			# f_mean = np.mean(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_mean)

		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD

		# for freq_value in all_frequencies:
			# f_std = np.std(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_std)

		# save_pickle('best_gen_' + config['name'] + '.pkl', best_results)

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')

		for freq_name in all_frequencies_names:
			mydict[str(freq_name)] = dict_freq[str(freq_name)]
		print('!!!!!!!!!!!!!!!!!!!!!')
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')
		save_pickle('config_' + config['name'] + '.pkl', config)
		
		# DataFr.to_excel(writer, sheet_name='Envelope_Fft')
		DataFr.to_excel(writer, sheet_name='Fft')
		writer.close()		
		print('Result in Excel table')
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD
	
	
	elif config['mode'] == 'new_long_analysis_features_freq_CWD_NOdamage':
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
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[4] == idf if f[-3:] == 'pkl']
			
		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD NO DAMAGE
		#CWD-  NOOOO Damage

		
		int_freqs = [30.6075, 61.215, 91.8225, 122.43, 153.0375]		
		int_freqs2 = [0.3091667, 0.6183334, 0.9275001, 1.2366668, 1.5458335, 1.8550002, 2.1641669, 2.4733336, 2.7825003, 3.091667]
		int_freqs2 += [30.2983333, 30.9166667, 29.9891666, 31.2258334, 29.6799999, 31.5350001, 29.3708332, 31.8441668, 29.0616665, 32.1533335, 28.7524998, 32.4625002, 60.9058333, 61.5241667, 60.5966666, 61.8333334, 60.2874999, 62.1425001, 59.9783332, 62.4516668, 59.6691665, 62.7608335, 59.3599998, 63.0700002, 91.5133333, 92.1316667, 91.2041666, 92.4408334, 90.8949999, 92.7500001, 90.5858332, 93.0591668, 90.2766665, 93.3683335, 89.9674998, 93.6775002, 122.1208333, 122.7391667, 121.8116666, 123.0483334, 121.5024999, 123.3575001, 121.1933332, 123.6666668, 120.8841665, 123.9758335, 120.5749998, 124.2850002, 152.7283333, 153.3466667, 152.4191666, 153.6558334, 152.1099999, 153.9650001, 151.8008332, 154.2741668, 151.4916665, 154.5833335, 151.1824998, 154.8925002]		
		int_freqs2 += int_freqs
		
		all_frequencies_names = int_freqs2
		

		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD NO DAMAGE
		

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

			all_frequencies = all_frequencies_names

			freq_range2 = list(np.array([0.1, 190.])) # CWD NOOOOOOOOO DAMAGE
				
			signal = load_signal(filepath, channel=config['channel'])

			
			signal = signal*1000./70.8 #for AE
			
			
			# lp, hp = genetic_optimal_filter_A(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=4, freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], inter=30)
			
			#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD
			
			# lp, hp = genetic_optimal_filter_C(x=signal, fs=config['fs'], levels=config['level'], num_generations=config['generations'], num_parents_mating=config['parents'], freq_values=int_freqs2, freq_range=freq_range2, weight_mutation=config['mutation'], clusters=config['clusters'], filename=filename[:-5])
			
			# lp, hp = genetic_optimal_filter_B(x=signal, fs=config['fs'], levels=config['level'], num_generations=3, num_parents_mating=4, freq_values1=int_freqs, freq_range1=freq_range, freq_values2=int_freqs2, freq_range2=freq_range2, weight_mutation=0.1, inter=12)
			
			# best_results[config['name'] + '_' + filename[:-5]] = [lp, hp]
			
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
			
			
			# # Wavelet
			# signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], levels=config['level'], mother_wv=config['mother_wv'], freq_values=int_freqs2, freq_range=freq_range2)
			

			# signal = butter_bandpass(signal, config['fs'], freqs=[lp, hp], order=3)
			

			#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD NOOOOOOO DAMAGE
			
			signal = signal**2.0 #SQR ANTES DE DEMODULATION
			# signal = signal /(np.max(signal) - np.min(signal))
			
			signal = hilbert_demodulation(signal)			
			
			magX, f, df = mag_fft(x=signal, fs=config['fs'])
			# magX, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!

			

			
			for freq_value, freq_name in zip(all_frequencies, all_frequencies_names):
				# vv = amp_component_zone(X=magX, df=df, freq=freq_value, tol=2.0)
				vv = amp_component(X=magX, df=df, freq=freq_value)
				# print(vv)
				dict_freq[str(freq_name)].append(vv)
			# plt.plot(f, magX)
			# plt.show()

		
		# for freq_value in all_frequencies:
			# f_mean = np.mean(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_mean)

		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD NOOOOOOOO DAMAGE

		# for freq_value in all_frequencies:
			# f_std = np.std(np.array(dict_freq[str(freq_value)]))
			# dict_freq[str(freq_value)].append(f_std)

		# save_pickle('best_gen_' + config['name'] + '.pkl', best_results)

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
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++CWD NOOOOOOO DAMAGE
	
	elif config['mode'] == 'magnitude_freq_ranges':
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
			
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-3:] == config['extension']]

		
	
		# ffr_freqs = [4.33, 8.66, 12.99, 17.32, 21.65, 26.0, 30.33, 34.67, 39.00, 43.33]
		# first_freqs = ffr_freqs
		# fg_freqs = [312., 624., 936., 1248., 1560.]
		# sb_freqs = [307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] +  [619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65] + [931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65] + [1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35] + [1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65] + [299.01, 611.01, 923.01, 1235.01, 1547.01, 324.99, 636.99, 948.99, 1260.99, 1572.99] + [286.02, 598.02, 910.02, 1222.02, 1534.02, 337.98, 649.98, 961.98, 1273.98, 1585.98]		
		# second_freqs = fg_freqs + sb_freqs
		ffr_freqs = [4.33, 8.66, 13.00, 17.33, 21.67, 26.00, 30.33, 34.67, 39.00, 43.33, 47.67, 52.00, 56.33, 60.67, 65.00, 69.33, 73.67, 78.00, 82.33, 86.67, 91.00, 95.33, 99.67] # 23
		first_freqs = ffr_freqs
		fg_freqs = [312., 624., 936., 1248., 1560.] # 5
		sb_freqs = [316.33, 320.67, 325.00, 329.33, 333.67, 338.00, 342.33, 346.67, 351.00, 355.33, 359.67, 364.00, 368.33, 372.67, 377.00, 381.33, 385.67, 390.00, 394.33, 398.67] + [628.33, 632.67, 637.00, 641.33, 645.67, 650.00, 654.33, 658.67, 663.00, 667.33, 671.67, 676.00, 680.33, 684.67, 689.00, 693.33, 697.67, 702.00, 706.33, 710.67] + [940.33, 944.67, 949.00, 953.33, 957.67, 962.00, 966.33, 970.67, 975.00, 979.33, 983.67, 988.00, 992.33, 996.67, 1001.00, 1005.33, 1009.67, 1014.00, 1018.33, 1022.67] + [1252.33, 1256.67, 1261.00, 1265.33, 1269.67, 1274.00, 1278.33, 1282.67, 1287.00, 1291.33, 1295.67, 1300.00, 1304.33, 1308.67, 1313.00, 1317.33, 1321.67, 1326.00, 1330.33, 1334.67] + [1564.33, 1568.67, 1573.00, 1577.33, 1581.67, 1586.00, 1590.33, 1594.67, 1599.00, 1603.33, 1607.67, 1612.00, 1616.33, 1620.67, 1625.00, 1629.33, 1633.67, 1638.00, 1642.33, 1646.67] + [225.33, 229.67, 234.00, 238.33, 242.67, 247.00, 251.33, 255.67, 260.00, 264.33, 268.67, 273.00, 277.33, 281.67, 286.00, 290.33, 294.67, 299.00, 303.33, 307.67] + [537.33, 541.67, 546.00, 550.33, 554.67, 559.00, 563.33, 567.67, 572.00, 576.33, 580.67, 585.00, 589.33, 593.67, 598.00, 602.33, 606.67, 611.00, 615.33, 619.67] + [849.33, 853.67, 858.00, 862.33, 866.67, 871.00, 875.33, 879.67, 884.00, 888.33, 892.67, 897.00, 901.33, 905.67, 910.00, 914.33, 918.67, 923.00, 927.33, 931.67] + [1161.33, 1165.67, 1170.00, 1174.33, 1178.67, 1183.00, 1187.33, 1191.67, 1196.00, 1200.33, 1204.67, 1209.00, 1213.33, 1217.67, 1222.00, 1226.33, 1230.67, 1235.00, 1239.33, 1243.67] + [1473.33, 1477.67, 1482.00, 1486.33, 1490.67, 1495.00, 1499.33, 1503.67, 1508.00, 1512.33, 1516.67, 1521.00, 1525.33, 1529.67, 1534.00, 1538.33, 1542.67, 1547.00, 1551.33, 1555.67] #200		
		second_freqs = fg_freqs + sb_freqs
		# one_second_freqs = fg_freqs + sb_freqs + ffr_freqs
		
		
		
		# #1300 Test Bench
		intervals = [3., 45., 280.]
		Intervals_Dict = {'0/3.0_45.0':[], '1/45.0_280.0':[]}
		str_intervals = ['0/3.0_45.0', '1/45.0_280.0']
		
		# intervals = [60., 100., 140., 180., 220., 260., 300.]
		# Intervals_Dict = {'0/60.0_100.0':[], '1/100.0_140.0':[], '2/140.0_180.0':[], '3/180.0_220.0':[], '4/220.0_260.0':[], '5/260.0_300.0':[]}
		# str_intervals = ['0/60.0_100.0', '1/100.0_140.0', '2/140.0_180.0', '3/180.0_220.0', '4/220.0_260.0', '5/260.0_300.0']
		

		for filepath in Filepaths:		
			filename = basename(filepath)
			print(filename)
			
			# if filename.find('20181102') != -1 and idf == 'E':
				# print('warning change frequencies')
				# intervals = [3.069231, 51.15385, 1636.9232]				
				# # freq_range2 = list(np.array([3., 1600.])*1.023077)
				# first_range = list(np.array([3., 50.])*1.023077)
				# second_range = list(np.array([50., 1600.])*1.023077)
				# first_freqs = list(np.array(first_freqs)*1.023077)
				# second_freqs = list(np.array(second_freqs)*1.023077)
				
				
			# else:
				# intervals = [3., 50., 1600.]
				# first_range = list(np.array([3., 50.]))
				# second_range = list(np.array([50., 1600.]))
			
			
			
			signal = load_signal(filepath, channel=config['channel'])
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					signal = signal/70.8
			
			if config['units'] != 'OFF':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.

			# filename = basename(filepath)			
			
			# lp, hp, max_kurt = Fast_Kurtogram_filters(signal, config['level'], config['fs'])
			
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=20.e3, order=3)
			# lp = 95.e3
			# hp = 140.e3
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[lp, hp], order=3)
			
			# Wavelet
			if config['wv_deco'] == 'ON':
				if config['wv_crit'] != 'avg_mpr':
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'])
				else:
					signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'], freq_values=first_freqs, freq_range=first_range, freq_values_2=second_freqs, freq_range_2=second_range)
				
			
			if config['filter'] != 'OFF':
				signal = multi_filter(signal, config, filename=filename)
			
			if config['sqr_envelope'] == 'ON':
				signal = signal**2.0
			
			if config['demodulation'] == 'ON':
				signal = hilbert_demodulation(signal)

			
			# signal = butter_demodulation(x=signal, fs=config['fs'], filter=['lowpass', 2000., 3], prefilter=['highpass', 20.e3, 3], type_rect=None, dc_value=None)
			
			
			if config['fourier'] == 'ON':
				if config['wv_deco'] == 'ON':
					print('info: wavelet affects FFT fs')
					magX, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!
				else:
					magX, f, df = mag_fft(x=signal, fs=config['fs'])
			else:
				print('Warning!!! No Fourier. Signal as Dict...')
				magX = signal['fft']
				f = signal['f']
				df = f[2] - f[1]
			
			
			
			for i in range(len(intervals)-1):
				if config['calc_range'] == 'avg':
					value = avg_in_band(magX, df, intervals[i], intervals[i+1])
				elif config['calc_range'] == 'energy':
					value = energy_in_band(magX, df, intervals[i], intervals[i+1])
				elif config['calc_range'] == 'std':
					value = std_in_band(magX, df, intervals[i], intervals[i+1])
				# value = sum_in_band(magX, df, intervals[i], intervals[i+1])
				else:
					print('error with calc range')
					sys.exit()
				print(value)				
				Intervals_Dict[str_intervals[i]].append(value)
			
			# print(Intervals_Dict)
			if config['plot'] == 'ON':
				plt.plot(f, magX, 'r')
				plt.show()

		
		# for key in Intervals_Dict:
			# Intervals_Dict[key].append(np.mean(Intervals_Dict[key]))
			# Intervals_Dict[key].append(np.std(Intervals_Dict[key]))
		
		# print('!!!')
		# print(str_intervals[0])
		# print(Intervals_Dict[str_intervals[0]])

		mydict = {}
			
		row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')
		
		for element in str_intervals:
			mydict[element] = Intervals_Dict[element]



		# DataFr = pd.DataFrame(data=mydict, index=row_names)
		# writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		# DataFr.to_excel(writer, sheet_name='Mag_Range')
		
		
		writer = pd.ExcelWriter(config['name'] + '.xlsx')
		DataFr_max = pd.DataFrame(data=mydict, index=row_names)				
		DataFr_max.to_excel(writer, sheet_name='Range')
		writer.close()
		
		
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
		if element == 'no_element' or element == 'range' or element == 'idx_levels':
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
	config['times_std'] = float(config['times_std'])
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	if config['freq_lp'] != 'OFF':
		config['freq_lp'] = float(config['freq_lp'])
	
	if config['freq_hp'] != 'OFF':
		config['freq_hp'] = float(config['freq_hp'])
	
	if config['db_out'] != 'OFF':
		config['db_out'] = int(config['db_out'])
	
	if config['idx_levels'] != None:
		for i in range(len(config['idx_levels'])):
			config['idx_levels'][i] = int(config['idx_levels'][i])
	
	
	# Variable conversion
	return config


	
if __name__ == '__main__':
	main(sys.argv)
