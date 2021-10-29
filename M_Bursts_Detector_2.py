
#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import os.path
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('D:'))
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
import pywt


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

InputsOpt_Defaults = {'name':'auto', 'plot':'ON', 'n_files':1, 'title_plot':None, 'thr_value':3.6, 'thr_mode':'factor_rms', 'window_time':0.001, 'stella':600, 'lockout':600, 'pdt':200, 'filter':['highpass', 5.e3, 3], 'mv':'ON', 'mypath':None, 'save':'ON', 'save_plot':'OFF', 'amp_db':'OFF', 'db_pa':37., 'range':None, 'db_out':'OFF', 'units':'v', 'n_clusters':0., 'level':0, 'wv_mother':'db6', 'inverse_wv':'OFF', 'idx_fp1':0,  'idx_fp2':0, 'short_burst':'OFF'}
# gearbox mio: thr_60, wt_0.001, hp_70k, stella_100, lcokout 200


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	if config['mode'] == 'mode1':
		
		print('Select signals')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		mydict = {}
		mydict['arrival'] = []
		mydict['arrival_acum'] = []
		mydict['rms'] = []
		mydict['amax'] = []
		mydict['count'] = []
		mydict['rise'] = []
		mydict['dura'] = []
		mydict['freq'] = []
		rownames = []
		
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
		
			signal = 1000*load_signal(filepath, channel=config['channel'])
			
			t = [i/config['fs'] for i in range(len(signal))]
			signal = butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)

			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
			from THR_Burst_Detection import plot_burst_rev
			
			plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
			ax_0.set_title(filename)
			ax_0.set_ylabel('Amplitude [V]', fontsize=13)
			ax_0.set_xlabel('Time [s]', fontsize=13)
			ax_0.tick_params(axis='both', labelsize=12)
			plt.show()
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
					value = t_ini1
					
					# burst = signal[int(t_ini1*config['fs']): int(t_fin1*config['fs'])]
					# spec, f, df = mag_fft(burst, config['fs'])
					# plt.plot(f, spec, 'k')
					# plt.show()
					
					mydict['arrival'].append(value)
					mydict['arrival_acum'].append(value+t_acum)
					rownames.append(str(perro) + '_' + filename)

					dict_feat = single_burst_features2(signal, t_ini1, t_fin1, config)		


					mydict['rms'].append(dict_feat['rms'])
					mydict['count'].append(dict_feat['count'])
					mydict['amax'].append(dict_feat['amax'])
					mydict['freq'].append(dict_feat['freq'])
					mydict['dura'].append(dict_feat['dura'])
					mydict['rise'].append(dict_feat['rise'])

					
					
					
					
					
					
					perro += 1
				count += 1
			else:
				
				mydict['arrival'].append('')
				mydict['arrival_acum'].append('')

				mydict['rms'].append('')
				mydict['amax'].append('')
				mydict['count'].append('')
				mydict['freq'].append('')
				mydict['rise'].append('')
				mydict['dura'].append('')
				
				
				
				
				rownames.append(str(0) + '_' + filename)
			t_acum += len(signal)/config['fs']
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('Arrival_Times_Thr_' + config['channel'] + config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)		
			
			DataFr_max.to_excel(writer, sheet_name='Arrival_Times_Thr')		

			
			writer.close()
		newdict = {}
		arrival = []
		newdict['amax'] = []
		newdict['count'] = []
		newdict['acum'] = []
		newdict['rms'] = []
		newdict['rms_acu'] = []
		newdict['freq'] = []
		newdict['rise'] = []
		newdict['dura'] = []
		cont = 0
		rms_sum = 0.
		for i in range(len(mydict['arrival'])):
			if mydict['arrival_acum'][i] != '':
				cont += 1
				arrival.append(mydict['arrival_acum'][i])
				newdict['acum'].append(cont)
				newdict['amax'].append(mydict['amax'][i])
				newdict['count'].append(mydict['count'][i])
				newdict['rms'].append(mydict['rms'][i])
				
				newdict['rise'].append(mydict['rise'][i])
				newdict['dura'].append(mydict['dura'][i])
				newdict['freq'].append(mydict['freq'][i])
				rms_sum += mydict['rms'][i]
				newdict['rms_acu'].append(rms_sum)
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('Feat_' + config['channel'] + config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=newdict, index=arrival)		
			
			DataFr_max.to_excel(writer, sheet_name='Arrival_Times_Thr')		

			
			writer.close()
	
	elif config['mode'] == 'mode2':

		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []
		Times = []
		fig, ax = plt.subplots()
		count = 0		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)
			
			if count == 0:
				Times += list(mydict.index)
				mydict = mydict.to_dict(orient='list')
				Feature += mydict[config['feature']]
			else:
				Times += list(np.array(list(mydict.index)) + Times[-1])
				if config['feature'].find('acu') != -1:
					mydict = mydict.to_dict(orient='list')
					Feature += list(np.array(list(mydict[config['feature']])) + Feature[-1])
				else:
					mydict = mydict.to_dict(orient='list')
					Feature += mydict[config['feature']]

			
			count += 1
		
		# Feature = list(np.nan_to_num(Feature))
		# min_dif = 1000.
		# for i in range(len(Feature)-1):
			# act_dif = Feature[i+1] - Feature[i]
			# if act_dif < min_dif:
				# min_dif = act_dif
		
		# newTimes = np.arange(np.min(Times), np.max(Times), min_dif)
		# newFeature = list(np.interp(newTimes, np.array(Times), np.array(Feature)))
		
		# newFeature = diff_signal_eq(newFeature, 1)
		# Feature += [0]
		ax.plot(Times, Feature, 'o')
		
		# magX, f, df = 
		
		FeatureTREND = movil_avg(Feature, 5)
		
		
		
		# print(Feature)
		
		
		n_files = len(Filepaths)
		
		
		
		
		
		
		# ax.set_xticks([(i)*180 for i in range(n_files+1)])
		# ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		# ax.set_xlabel('N° Campaign', fontsize=13)
		
		
		
		
		ax.plot(Times, FeatureTREND, 'r')
		# ax.set_xlabel('Time [h]', fontsize=12)
		# ax.set_ylabel('RMS Value [mV]', fontsize=12)
		
		if config['feature'] == 'rms':
			ax.set_ylabel('RMS Value [mV]', fontsize=13)
		elif config['feature'] == 'acum':
			ax.set_ylabel('N° AE Bursts', fontsize=13)
		elif config['feature'] == 'ra':
			ax.set_ylabel('RA [V/us]', fontsize=13)
		elif config['feature'] == 'dc':
			ax.set_ylabel('DC [1/us]', fontsize=13)
		elif config['feature'] == 'amax':
			ax.set_ylabel('Max. Amplitude [mV]', fontsize=13)
		elif config['feature'] == 'dura':
			ax.set_ylabel('Duration [us]', fontsize=13)
		elif config['feature'] == 'rise_corr':
			ax.set_ylabel('Rise Time [us]', fontsize=13)
		elif config['feature'] == 'count_corr':
			ax.set_ylabel('Counts [-]', fontsize=13)
		elif config['feature'] == 'freq':
			ax.set_ylabel('Main Frequency [kHz]', fontsize=13)
		elif config['feature'] == 'crest':
			ax.set_ylabel('Crest Factor [-]', fontsize=13)
		
		
		ax.set_xlabel('Time [s]', fontsize=13)


		ax.tick_params(axis='both', labelsize=12)
		print('caca')
		# plt.savefig(config['title'] + '_' + config['feature'] + '.png')
		plt.show()
		# print(Feature)
		
		

	
	elif config['mode'] == 'mode3':
	
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		mydict = {}
		mydict['arrival'] = []
		mydict['arrival_acum'] = []
		mydict['rms'] = []
		mydict['amax'] = []
		mydict['count'] = []
		mydict['rise'] = []
		mydict['dura'] = []
		mydict['freq'] = []
		mydict['ra'] = []
		mydict['dc'] = []
		mydict['crest'] = []
		rownames = []
		
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
		
			signal = 1*load_signal(filepath, channel=config['channel'])
			
			t = [i/config['fs'] for i in range(len(signal))]
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				

			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [V]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
					value = t_ini1
					
					# burst = signal[int(t_ini1*config['fs']): int(t_fin1*config['fs'])]
					# spec, f, df = mag_fft(burst, config['fs'])
					# plt.plot(f, spec, 'k')
					# plt.show()
					
					mydict['arrival'].append(value)
					mydict['arrival_acum'].append(value+t_acum)
					rownames.append(str(perro) + '_' + filename)

					dict_feat = single_burst_features3(signal, t_ini1, t_fin1, config)		


					mydict['rms'].append(dict_feat['rms'])
					mydict['count'].append(dict_feat['count'])
					mydict['amax'].append(dict_feat['amax'])
					mydict['freq'].append(dict_feat['freq'])
					
					mydict['dura'].append(dict_feat['dura'])
					mydict['rise'].append(dict_feat['rise'])
					
					mydict['ra'].append(dict_feat['amax'] / dict_feat['rise'])					
					mydict['dc'].append(dict_feat['count'] / dict_feat['dura'])					
					mydict['crest'].append(dict_feat['amax'] / dict_feat['rms'])
					
					perro += 1
				count += 1
			else:
				
				mydict['arrival'].append('')
				mydict['arrival_acum'].append('')

				mydict['rms'].append('')
				mydict['amax'].append('')
				mydict['count'].append('')
				mydict['freq'].append('')
				
				mydict['rise'].append('')				
				mydict['dura'].append('')
				
				mydict['ra'].append('')
				mydict['dc'].append('')
				mydict['crest'].append('')
				
				
				
				
				rownames.append(str(0) + '_' + filename)
			t_acum += len(signal)/config['fs']
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_AE_Burst_Features.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts')
			writer.close()
		
		newdict = {}
		arrival = []
		newdict['acum'] = []
		newdict['amax'] = []
		newdict['count'] = []
		
		newdict['rms'] = []
		newdict['rms_acu'] = []		
		newdict['freq'] = []
		
		newdict['rise'] = []
		newdict['dura'] = []
		
		newdict['ra'] = []
		newdict['dc'] = []
		newdict['crest'] = []
		cont = 0
		rms_sum = 0.
		for i in range(len(mydict['arrival'])):
			if mydict['arrival_acum'][i] != '':
				cont += 1
				arrival.append(mydict['arrival_acum'][i])
				newdict['acum'].append(cont)
				newdict['amax'].append(mydict['amax'][i])
				newdict['count'].append(mydict['count'][i])
				
				newdict['rms'].append(mydict['rms'][i])
				rms_sum += mydict['rms'][i]
				newdict['rms_acu'].append(rms_sum)
				newdict['freq'].append(mydict['freq'][i])

				newdict['rise'].append(mydict['rise'][i])
				newdict['dura'].append(mydict['dura'][i])

				newdict['ra'].append(mydict['ra'][i])
				newdict['dc'].append(mydict['dc'][i])
				newdict['crest'].append(mydict['crest'][i])
				
				
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_CorrAE_Burst_Features.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=newdict, index=arrival)		
			
			DataFr_max.to_excel(writer, sheet_name='Corr_Bursts')		

			writer.close()
			# Output = {'config':config}
			# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pickle('config_' + config['name'] +'.pkl', config)
	
	elif config['mode'] == 'detector_thr_7exp':
	
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		# print(Filenames)
		mydict = {}
		mydict['arrival'] = []
		mydict['arrival_acum'] = []
		mydict['rms'] = []
		mydict['amax'] = []
		mydict['count'] = []
		mydict['rise'] = []
		mydict['dura'] = []
		mydict['freq'] = []
		mydict['kurt'] = []
		mydict['ra'] = []
		mydict['dc'] = []
		mydict['crest'] = []
		
		mydict['cef'] = []
		mydict['stf'] = []
		
		mydict['rmsf'] = []
		mydict['kurtf'] = []
		
		mydict['amaxt'] = []
		mydict['rmst'] = []
		mydict['kurtt'] = []
		mydict['crestt'] = []
		
		
		mydict['senenvt'] = []
		mydict['kurtenvt'] = []
		mydict['stdenvt'] = []
		mydict['amaxenvt'] = []
		mydict['sent'] = []
		mydict['sen'] = []
		mydict['cetenv'] = []
		mydict['rmscetenv'] = []
		mydict['amaxenv'] = []
		mydict['stdenv'] = []
		mydict['kurtenv'] = []
		mydict['senenv'] = []
		mydict['cet'] = []
		mydict['rmscet'] = []
		
		
		rownames = []
		
		count_burst = 0.
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			# signal = 1000*1000*load_signal(filepath, channel=config['channel'])/141.25 #BOCHUM
			signal = load_signal(filepath, channel=config['channel'])
			# X, f, df = mag_fft(signal, config['fs'])
			# plt.plot(f, X, 'k')
			# plt.show()
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt')
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]

			
			
			# mydict2 = read_pickle(filepath)
			# signal = mydict2['x']
			# config['fs'] = mydict2['fs']
			# dt = 1./config['fs']
			# t = dt*np.arange(len(signal))
			
			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			# num_bursts = len(t_burst_corr1)
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				
				plt.show()
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):
					count_burst += 1
					value = t_ini1
					
					# burst = signal[int(t_ini1*config['fs']): int(t_fin1*config['fs'])]
					# spec, f, df = mag_fft(burst, config['fs'])
					# plt.plot(f, spec, 'k')
					# plt.show()
					
					mydict['arrival'].append(value)
					mydict['arrival_acum'].append(value+t_acum)
					rownames.append(str(perro) + '_' + filename)

					# dict_feat = single_burst_features4_vs(signal, t_ini1, t_fin1, config, threshold=threshold)
					# dict_feat = single_burst_features5_exp(signal, t_ini1, t_fin1, config, threshold=threshold, num_bursts_abs=count_burst, filename=filename, num_bursts_rel=perro)
					dict_feat = single_burst_features7_exp(signal, t_ini1, t_fin1, config, threshold=threshold, num_bursts_abs=count_burst, filename=filename, num_bursts_rel=perro)

					mydict['rms'].append(dict_feat['rms'])
					mydict['count'].append(dict_feat['count'])
					mydict['amax'].append(dict_feat['amax'])

					
					mydict['dura'].append(dict_feat['dura'])
					mydict['rise'].append(dict_feat['rise'])
					
					mydict['ra'].append(dict_feat['amax'] / dict_feat['rise'])					
					mydict['dc'].append(dict_feat['count'] / dict_feat['dura'])					
					mydict['crest'].append(dict_feat['amax'] / dict_feat['rms'])
					
					
					
					mydict['cef'].append(dict_feat['cef'])
					mydict['stf'].append(dict_feat['stf'])
					mydict['freq'].append(dict_feat['freq'])
					mydict['kurt'].append(dict_feat['kurt'])
					
					mydict['rmsf'].append(dict_feat['rmsf'])
					mydict['kurtf'].append(dict_feat['kurtf'])
					
					
					
					mydict['rmst'].append(dict_feat['rmst'])
					mydict['amaxt'].append(dict_feat['amaxt'])
					mydict['kurtt'].append(dict_feat['kurtt'])
					mydict['crestt'].append(dict_feat['amaxt'] / dict_feat['rmst'])
					
					
					
					mydict['senenvt'].append(dict_feat['senenvt'])
					mydict['kurtenvt'].append(dict_feat['kurtenvt'])
					mydict['stdenvt'].append(dict_feat['stdenvt'])
					mydict['amaxenvt'].append(dict_feat['amaxenvt'])
					mydict['sent'].append(dict_feat['sent'])
					mydict['sen'].append(dict_feat['sen'])
					mydict['cetenv'].append(dict_feat['cetenv'])
					mydict['rmscetenv'].append(dict_feat['rmscetenv'])
					mydict['amaxenv'].append(dict_feat['amaxenv'])
					mydict['stdenv'].append(dict_feat['stdenv'])
					mydict['kurtenv'].append(dict_feat['kurtenv'])
					mydict['senenv'].append(dict_feat['senenv'])
					mydict['cet'].append(dict_feat['cet'])
					mydict['rmscet'].append(dict_feat['rmscet'])
					
					
					
					
					perro += 1
				count += 1
			else:
				
				mydict['arrival'].append('')
				mydict['arrival_acum'].append('')

				mydict['rms'].append('')
				mydict['amax'].append('')
				mydict['count'].append('')
				# mydict['freq'].append('')
				# mydict['kurt'].append('')
				
				mydict['rise'].append('')				
				mydict['dura'].append('')
				
				mydict['ra'].append('')
				mydict['dc'].append('')
				mydict['crest'].append('')
				
				
				mydict['cef'].append('')
				mydict['stf'].append('')
				mydict['freq'].append('')
				mydict['kurt'].append('')
				
				mydict['rmsf'].append('')
				mydict['kurtf'].append('')
				
				mydict['rmst'].append('')
				mydict['amaxt'].append('')
				mydict['kurtt'].append('')
				mydict['crestt'].append('')
				
				
				mydict['senenvt'].append('')
				mydict['kurtenvt'].append('')
				mydict['stdenvt'].append('')
				mydict['amaxenvt'].append('')
				mydict['sent'].append('')
				mydict['sen'].append('')
				mydict['cetenv'].append('')
				mydict['rmscetenv'].append('')
				mydict['amaxenv'].append('')
				mydict['stdenv'].append('')
				mydict['kurtenv'].append('')
				mydict['senenv'].append('')
				mydict['cet'].append('')
				mydict['rmscet'].append('')
					
				
				rownames.append(str(0) + '_' + filename)
			t_acum += len(signal)/config['fs']
			
				

				
		
		if config['save'] == 'ON':
			# writer = pd.ExcelWriter(config['name'] + '_AE_Burst_Features.xlsx')
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts')
			writer.close()
		
		newdict = {}
		arrival = []
		newdict['acum'] = []
		newdict['amax'] = []
		newdict['count'] = []
		
		newdict['rms'] = []
		newdict['rms_acu'] = []		
		# newdict['freq'] = []
		
		newdict['rise'] = []
		newdict['dura'] = []
		
		newdict['ra'] = []
		newdict['dc'] = []
		newdict['crest'] = []
		
		
		newdict['cef'] = []
		newdict['stf'] = []
		newdict['freq'] = []
		newdict['kurt'] = []
		
		newdict['rmsf'] = []
		newdict['kurtf'] = []
		
		
		newdict['amaxt'] = []
		newdict['rmst'] = []
		newdict['kurtt'] = []
		newdict['crestt'] = []
		
		
		
		newdict['senenvt'] = []
		newdict['kurtenvt'] = []
		newdict['stdenvt'] = []
		newdict['amaxenvt'] = []
		newdict['sent'] = []
		newdict['sen'] = []
		newdict['cetenv'] = []
		newdict['rmscetenv'] = []
		newdict['amaxenv'] = []
		newdict['stdenv'] = []
		newdict['kurtenv'] = []
		newdict['senenv'] = []
		newdict['cet'] = []
		newdict['rmscet'] = []
		
		
		cont = 0
		rms_sum = 0.
		for i in range(len(mydict['arrival'])):
			if mydict['arrival_acum'][i] != '':
				cont += 1
				arrival.append(mydict['arrival_acum'][i])
				newdict['acum'].append(cont)
				newdict['amax'].append(mydict['amax'][i])
				newdict['count'].append(mydict['count'][i])
				
				newdict['rms'].append(mydict['rms'][i])
				rms_sum += mydict['rms'][i]
				newdict['rms_acu'].append(rms_sum)
				# newdict['freq'].append(mydict['freq'][i])

				newdict['rise'].append(mydict['rise'][i])
				newdict['dura'].append(mydict['dura'][i])

				newdict['ra'].append(mydict['ra'][i])
				newdict['dc'].append(mydict['dc'][i])
				newdict['crest'].append(mydict['crest'][i])
				
				
				
				newdict['cef'].append(mydict['cef'][i])
				newdict['stf'].append(mydict['stf'][i])
				newdict['freq'].append(mydict['freq'][i])
				newdict['kurt'].append(mydict['kurt'][i])
				
				newdict['rmsf'].append(mydict['rmsf'][i])
				newdict['kurtf'].append(mydict['kurtf'][i])
				
				newdict['rmst'].append(mydict['rmst'][i])
				newdict['kurtt'].append(mydict['kurtt'][i])
				newdict['amaxt'].append(mydict['amaxt'][i])
				newdict['crestt'].append(mydict['crestt'][i])
				
				
				newdict['senenvt'].append(mydict['senenvt'][i])
				newdict['kurtenvt'].append(mydict['kurtenvt'][i])
				newdict['stdenvt'].append(mydict['stdenvt'][i])
				newdict['amaxenvt'].append(mydict['amaxenvt'][i])
				newdict['sent'].append(mydict['sent'][i])
				newdict['sen'].append(mydict['sen'][i])
				newdict['cetenv'].append(mydict['cetenv'][i])
				newdict['rmscetenv'].append(mydict['rmscetenv'][i])
				newdict['amaxenv'].append(mydict['amaxenv'][i])
				newdict['stdenv'].append(mydict['stdenv'][i])
				newdict['kurtenv'].append(mydict['kurtenv'][i])
				newdict['senenv'].append(mydict['senenv'][i])
				newdict['cet'].append(mydict['cet'][i])
				newdict['rmscet'].append(mydict['rmscet'][i])
		
				
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_CorrAE_Burst_Features.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=newdict, index=arrival)		
			
			DataFr_max.to_excel(writer, sheet_name='Corr_Bursts')		

			writer.close()
			# Output = {'config':config}
			# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pickle('config_' + config['name'] +'.pkl', config)
	
	elif config['mode'] == 'detector_thr':
	
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
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		# print(Filenames)
		mydict = {}
		mydict['arrival'] = []
		mydict['arrival_acum'] = []
		mydict['rms'] = []
		mydict['amax'] = []
		mydict['count'] = []
		mydict['rise'] = []
		mydict['dura'] = []
		mydict['freq'] = []
		mydict['kurt'] = []
		mydict['ra'] = []
		mydict['dc'] = []
		mydict['crest'] = []
		
		mydict['cef'] = []
		mydict['stf'] = []
		
		mydict['rmsf'] = []
		mydict['kurtf'] = []
		
		# mydict['amaxt'] = []
		# mydict['rmst'] = []
		# mydict['kurtt'] = []
		# mydict['crestt'] = []
		
		rownames = []
		
		count_burst = 0.
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			# signal = 1000*1000*load_signal(filepath, channel=config['channel'])/141.25 #BOCHUM
			signal = load_signal(filepath, channel=config['channel'])
			# X, f, df = mag_fft(signal, config['fs'])
			# plt.plot(f, X, 'k')
			# plt.show()
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt OHNE LACK')
					signal = signal/70.8
				elif config['db_out'] == 49:
					print('db out schottland MIT LACK')
					signal = signal/281.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]

			
			
			# mydict2 = read_pickle(filepath)
			# signal = mydict2['x']
			# config['fs'] = mydict2['fs']
			# dt = 1./config['fs']
			# t = dt*np.arange(len(signal))
			
			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			# num_bursts = len(t_burst_corr1)
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				
				plt.show()
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):
					count_burst += 1
					value = t_ini1
					
					# burst = signal[int(t_ini1*config['fs']): int(t_fin1*config['fs'])]
					# spec, f, df = mag_fft(burst, config['fs'])
					# plt.plot(f, spec, 'k')
					# plt.show()
					
					mydict['arrival'].append(value)
					mydict['arrival_acum'].append(value+t_acum)
					rownames.append(str(perro) + '_' + filename)

					if config['short_burst'] == 'ON':
						print('with short burst!')
						dict_feat = single_burst_features6_exp_short(signal, t_ini1, t_fin1, config, threshold=threshold, num_bursts_abs=count_burst, filename=filename, num_bursts_rel=perro)
					else:
						dict_feat = single_burst_features6_exp(signal, t_ini1, t_fin1, config, threshold=threshold, num_bursts_abs=count_burst, filename=filename, num_bursts_rel=perro)

					mydict['rms'].append(dict_feat['rms'])
					mydict['count'].append(dict_feat['count'])
					mydict['amax'].append(dict_feat['amax'])

					
					mydict['dura'].append(dict_feat['dura'])
					mydict['rise'].append(dict_feat['rise'])
					
					mydict['ra'].append(dict_feat['amax'] / dict_feat['rise'])					
					mydict['dc'].append(dict_feat['count'] / dict_feat['dura'])					
					mydict['crest'].append(dict_feat['amax'] / dict_feat['rms'])
					
					
					
					mydict['cef'].append(dict_feat['cef'])
					mydict['stf'].append(dict_feat['stf'])
					mydict['freq'].append(dict_feat['freq'])
					mydict['kurt'].append(dict_feat['kurt'])
					
					mydict['rmsf'].append(dict_feat['rmsf'])
					mydict['kurtf'].append(dict_feat['kurtf'])
					
					
					
					# mydict['rmst'].append(dict_feat['rmst'])
					# mydict['amaxt'].append(dict_feat['amaxt'])
					# mydict['kurtt'].append(dict_feat['kurtt'])
					# mydict['crestt'].append(dict_feat['amaxt'] / dict_feat['rmst'])
					
					perro += 1
				count += 1
			else:
				
				mydict['arrival'].append('')
				mydict['arrival_acum'].append('')

				mydict['rms'].append('')
				mydict['amax'].append('')
				mydict['count'].append('')
				# mydict['freq'].append('')
				# mydict['kurt'].append('')
				
				mydict['rise'].append('')				
				mydict['dura'].append('')
				
				mydict['ra'].append('')
				mydict['dc'].append('')
				mydict['crest'].append('')
				
				
				mydict['cef'].append('')
				mydict['stf'].append('')
				mydict['freq'].append('')
				mydict['kurt'].append('')
				
				mydict['rmsf'].append('')
				mydict['kurtf'].append('')
				
				# mydict['rmst'].append('')
				# mydict['amaxt'].append('')
				# mydict['kurtt'].append('')
				# mydict['crestt'].append('')
				
				rownames.append(str(0) + '_' + filename)
			t_acum += len(signal)/config['fs']
			
				

				
		
		if config['save'] == 'ON':
			# writer = pd.ExcelWriter(config['name'] + '_AE_Burst_Features.xlsx')
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts')
			writer.close()
		
		newdict = {}
		arrival = []
		newdict['acum'] = []
		newdict['amax'] = []
		newdict['count'] = []
		
		newdict['rms'] = []
		newdict['rms_acu'] = []		
		# newdict['freq'] = []
		
		newdict['rise'] = []
		newdict['dura'] = []
		
		newdict['ra'] = []
		newdict['dc'] = []
		newdict['crest'] = []
		
		
		newdict['cef'] = []
		newdict['stf'] = []
		newdict['freq'] = []
		newdict['kurt'] = []
		
		newdict['rmsf'] = []
		newdict['kurtf'] = []
		
		
		# newdict['amaxt'] = []
		# newdict['rmst'] = []
		# newdict['kurtt'] = []
		# newdict['crestt'] = []
		
		
		cont = 0
		rms_sum = 0.
		for i in range(len(mydict['arrival'])):
			if mydict['arrival_acum'][i] != '':
				cont += 1
				arrival.append(mydict['arrival_acum'][i])
				newdict['acum'].append(cont)
				newdict['amax'].append(mydict['amax'][i])
				newdict['count'].append(mydict['count'][i])
				
				newdict['rms'].append(mydict['rms'][i])
				rms_sum += mydict['rms'][i]
				newdict['rms_acu'].append(rms_sum)
				# newdict['freq'].append(mydict['freq'][i])

				newdict['rise'].append(mydict['rise'][i])
				newdict['dura'].append(mydict['dura'][i])

				newdict['ra'].append(mydict['ra'][i])
				newdict['dc'].append(mydict['dc'][i])
				newdict['crest'].append(mydict['crest'][i])
				
				
				
				newdict['cef'].append(mydict['cef'][i])
				newdict['stf'].append(mydict['stf'][i])
				newdict['freq'].append(mydict['freq'][i])
				newdict['kurt'].append(mydict['kurt'][i])
				
				newdict['rmsf'].append(mydict['rmsf'][i])
				newdict['kurtf'].append(mydict['kurtf'][i])
				
				# newdict['rmst'].append(mydict['rmst'][i])
				# newdict['kurtt'].append(mydict['kurtt'][i])
				# newdict['amaxt'].append(mydict['amaxt'][i])
				# newdict['crestt'].append(mydict['crestt'][i])
				
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_CorrAE_Burst_Features.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=newdict, index=arrival)		
			
			DataFr_max.to_excel(writer, sheet_name='Corr_Bursts')		

			writer.close()
			# Output = {'config':config}
			# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pickle('config_' + config['name'] +'.pkl', config)
	
	elif config['mode'] == 'detector_spectra_thr':
	
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		
		
		rownames = []
		
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			# signal = 1000*1000*load_signal(filepath, channel=config['channel'])/141.25 #BOCHUM
			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt')
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]


			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini, t_fin in zip(t_burst_corr1, t_burst_corr_rev1):			

					burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
					
					t_it = np.arange(len(burst_it))/config['fs']
					
					
					mBurst_it, f_it, df_it = mag_fft(burst_it, config['fs'])
					
					fig, ax = plt.subplots(nrows=2, ncols=1)
					ax[0].plot(t_it, burst_it, 'b')
					ax[1].plot(f_it, mBurst_it, 'r')
					ax[0].set_title(config['channel'] + '_Time ini ' + str(t_ini))
					print(filename[:-5] + '_' + str(perro) + '_' + str(t_ini) + '.png')
					plt.savefig(filename[:-5] + '_' + str(perro) + '_' + str(t_ini) + '.png')
					
					
					
					
					perro += 1
				count += 1
	
	elif config['mode'] == 'detector_cwt_thr':
		print('++++++++cwt')
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		
		
		rownames = []
		
		count_abs = 1
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			# signal = 1000*1000*load_signal(filepath, channel=config['channel'])/141.25 #BOCHUM
			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt')
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]

			

			# dt = 1./config['fs']
			# max_width = 201
			# widths = np.arange(1, max_width)
			# mother_wv = 'morl'
			# real_freq = pywt.scale2frequency(mother_wv, widths)/dt
			# cwtmatrALL, freqs = pywt.cwt(data=signal, scales=widths, wavelet=mother_wv, sampling_period=dt)
			# cwtmatrALL = cwtmatrALL**2.0
			# maxALL = np.max(cwtmatrALL)
			# minALL = np.min(cwtmatrALL)
			
			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			# # maxALL = 0.
			# # minALL = 0.
			# perro = 0
			# if len(t_burst_corr1) != 0:
				# for t_ini, t_fin in zip(t_burst_corr1, t_burst_corr_rev1):			

					# burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
					# t_it = np.arange(len(burst_it))/config['fs']

					
					# dt = 1./config['fs']
					# max_width = 201
					# widths = np.arange(1, max_width)
					# mother_wv = 'morl'
					# real_freq = pywt.scale2frequency(mother_wv, widths)/dt
					# cwtmatrALL, freqs = pywt.cwt(data=burst_it, scales=widths, wavelet=mother_wv, sampling_period=dt)
					# cwtmatrALL = cwtmatrALL**2.0
					# # maxALL = np.max(cwtmatrALL)
					# # minALL = np.min(cwtmatrALL)
					# if perro == 0:
						# maxALL = np.max(cwtmatrALL)
						# minALL = np.min(cwtmatrALL)
					# else:
						# max_it = np.max(cwtmatrALL)
						# min_it = np.min(cwtmatrALL)
						# if max_it > maxALL:
							# maxALL = max_it
						# if min_it < minALL:
							# minALL = min_it					
					
					
					# perro += 1
				# count += 1	
			
			# print('min = ', minALL)
			# print('max = ', maxALL)
			
			
			
			
			
			
			
			
			
			
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini, t_fin in zip(t_burst_corr1, t_burst_corr_rev1):			

					burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
					t_it = np.arange(len(burst_it))/config['fs']

					
					dt = 1./config['fs']
					max_width = 101
					widths = np.arange(1, max_width)
					mother_wv = 'morl'
					real_freq = pywt.scale2frequency(mother_wv, widths)/dt
					cwtmatr, freqs = pywt.cwt(data=burst_it, scales=widths, wavelet=mother_wv, sampling_period=dt)
					# cwtmatr = cwtmatr**2.0
					# y = real_freq
					# x = t
					extent_ = [-1, 1, 1, max_width]
					if filename[0] == '_' or filename.find('20181102') != -1 or filename.find('20180511') != -1:
						colormap_ = 'PRGn_r'
						print('with damage!')
					else:
						colormap_ = 'PRGn'
					
					fig, ax = plt.subplots()
					
					# ax.imshow(cwtmatr, extent=extent_, cmap=colormap_, aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), interpolation='bilinear')
					# ax.set_title('Time ini ' + str(t_ini))
					# ax.set_xlim(left=-1 , right=1)
					# ax.set_ylim(bottom=180 , top=max_width)
					
					
					# levels_ = list(np.linspace(np.min(cwtmatr), np.max(cwtmatr), num=20))
					# levels_ = list(np.linspace(minALL, maxALL, num=20))
					# ax.contour(t_it, real_freq, cwtmatr, levels=levels_)
					ax.contour(t_it, real_freq, cwtmatr)
					ax.set_ylim(bottom=0 , top=500000)
					# ax.set_title(config['channel'] + '_Time ini ' + str(t_ini))
					ax.set_title(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '_t_ini' + str(t_ini))
					

					
					# sum = np.zeros(len(cwtmatr))
					# for i in range(len(cwtmatr)):
						# sum[i] = np.sum((cwtmatr[i]))
					# ax.plot(real_freq, sum, 'ro-')
					# # print(sum)
					
					# plt.plot(sum, 'b')
					# plt.show()
					# sys.exit()
					
					# ax.set_xlim(left=0 , right=500000)
					# ax.set_title(config['channel'] + '_Time ini ' + str(t_ini))
					
					
					
					
					print(config['channel'] + '_' + filename[:-5] + '_' + str(perro) + '_' + str(t_ini) + '.png')
					plt.savefig(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
					
					count_abs += 1

					
					perro += 1
				count += 1
	
	
	elif config['mode'] == 'detector_wpd_thr':
		print('++++++++WPD')
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		ref_spectrum = np.zeros(2**config['level'])
		ref_spectrum[config['idx_fp1']] = 1.
		ref_spectrum[config['idx_fp1']-1] = 0.5
		ref_spectrum[config['idx_fp1']+1] = 0.5
		# ref_spectrum[config['idx_fp1']-2] = 0.33
		# ref_spectrum[config['idx_fp1']+2] = 0.33
		ref_spectrum[config['idx_fp2']] = 1.
		ref_spectrum[config['idx_fp2']-1] = 0.5
		ref_spectrum[config['idx_fp2']+1] = 0.5
		# ref_spectrum[config['idx_fp2']-2] = 0.33
		# ref_spectrum[config['idx_fp2']+2] = 0.33
		
		mydict = {}
		mydict['fmax'] = []
		mydict['cef'] = []
		mydict['rmsf'] = []
		mydict['stf'] = []
		
		mydict['arrival'] = []
		mydict['arrival_acum'] = []
		
		mydict['xcorr'] = []
		mydict['fp1'] = []
		mydict['fp2'] = []
		
		rownames = []
		
		count_abs = 1
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt')
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]

			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			
			
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini, t_fin in zip(t_burst_corr1, t_burst_corr_rev1):			
					value = t_ini
					if config['short_burst'] == 'ON':
						print('with short burst!')
						burst_it = signal[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
					elif config['short_burst'] == 'PDT':
						print('with PDT burst!')
						burst_it = signal[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['pdt'])]
					elif config['short_burst'] == 'PDT_left':
						print('with PDT left burst!')
						burst_it = signal[int(t_ini*config['fs']) - int(config['pdt']/5) : int(t_ini*config['fs']) + int(config['pdt'])]
					else:
						burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
					
					# plt.plot(burst_it)
					# plt.show()
					t_it = np.arange(len(burst_it))/config['fs']
					n_it = len(burst_it)
					
					dt = 1./config['fs']
					
					
					select_level = config['level']
		
					wavelet_mother = config['wv_mother']
					nico = pywt.WaveletPacket(data=burst_it, wavelet=wavelet_mother, maxlevel=select_level)
					
					mylevels = [node.path for node in nico.get_level(select_level, 'freq')]
					
					
					myfreq = np.arange(len(mylevels))*config['fs']/(2**(1+config['level']))
					myfreq += config['fs']/(2**(2+config['level']))
					# print(myfreq)
					# aa = input('pause....')
					
					cwtmatr = []
					count = 0
					for level in mylevels:
						mywav = nico[level].data
						
						if config['inverse_wv'] != 'ON':
							xold = np.linspace(0., 1., len(mywav))
							xnew = np.linspace(0., 1., n_it)				
							mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)	
						else:
							print('inverse WV!')
							gato = pywt.WaveletPacket(data=None, wavelet=wavelet_mother, maxlevel=config['level'])
							gato[level] = nico[level].data
							mywav_int = gato.reconstruct(update=False)
							
						cwtmatr.append(mywav_int)
					cwtmatr = np.array(cwtmatr)
					# print(cwtmatr)
					
					
					
					

					# fig, ax = plt.subplots()
					# extent_ = [0, np.max(t_it), 0, 500]
					# ax.contour(cwtmatr, extent=extent_)					
					# plt.show()
					

					
					sum = np.zeros(len(cwtmatr))
					for i in range(len(cwtmatr)):
						sum[i] = np.sum((cwtmatr[i])**2.0)
					
					sum[0:10] = 0 #ALTERED
					
					fmax = myfreq[np.argmax(sum)]					
					sum_fc = 0.
					sum_rmsf = 0.
					sum_sum = np.sum(sum)
					for i in range(len(mylevels)):
						sum_fc += sum[i]*myfreq[i]
						sum_rmsf += sum[i]*(myfreq[i])**2.0
					cef = sum_fc/sum_sum
					rmsf = (sum_rmsf/sum_sum)**0.5
					
					
					sum_stf = 0.
					for i in range(len(mylevels)):
						sum_stf += sum[i]*(myfreq[i]-cef)**2.0
					stf = (sum_stf/sum_sum)**0.5
					
					
					mydict['arrival'].append(value)
					mydict['arrival_acum'].append(value+t_acum)
					rownames.append(str(perro) + '_' + filename)
					
					mydict['fmax'].append(fmax)
					mydict['cef'].append(cef)
					mydict['rmsf'].append(rmsf)
					mydict['stf'].append(stf)
					
					
					# xcorr = max_norm_correlation_lag(sum/np.max(sum), ref_spectrum, 2)
					xcorr = max_norm_correlation(sum/np.max(sum), ref_spectrum)
					mydict['xcorr'].append(xcorr)
					
					fpeaks = np.zeros(2)
					med_level = int(np.median(np.arange(len(mylevels))))
					
					sum_p1 = sum[0:med_level]
					sum_p2 = sum[med_level:]
					idx_p1 = np.arange(med_level)
					idx_p2 = np.arange(med_level, len(mylevels))

					lim = 0.9
					if np.argmax(sum) in list(idx_p1):					
						fpeaks[0] = fmax
						if np.max(sum_p2) >= lim*np.max(sum):
							fpeaks[1] = myfreq[med_level + np.argmax(sum_p2)]
						else:
							fpeaks[1] = 0.
						
					elif np.argmax(sum) in list(idx_p2):
						fpeaks[1] = fmax
						if np.max(sum_p1) >= lim*np.max(sum):
							fpeaks[0] = myfreq[np.argmax(sum_p1)]
						else:
							fpeaks[0] = 0.
						
					else:
						print('warning 588!!!')
						fpeaks[0] = fmax
						if np.max(sum_p2) >= lim*np.max(sum):
							fpeaks[1] = myfreq[med_level + np.argmax(sum_p2)]
						else:
							fpeaks[1] = 0.
					
					fp1 = fpeaks[0]
					fp2 = fpeaks[1]
					
					
					
					# mean_sum = np.mean(sum)
					# std_sum = np.std(sum)
					# med_level = int(np.median(np.arange(len(mylevels))))
					
					# sum_p1 = sum[0:med_level]
					# sum_p2 = sum[med_level:]
					
					# max_p1 = np.max(sum_p1)
					# max_p2 = np.max(sum_p2)
					
					# mean_sum1 = np.mean(sum_p1)
					# mean_sum2 = np.mean(sum_p2)
					# std_sum1 = np.std(sum_p1)
					# std_sum2 = np.std(sum_p2)
					
					# lim1 = mean_sum + 3*std_sum
					# lim2 = lim1
					
					# print('tini+++++++ ', t_ini)
					# if max_p1 >= lim1:
						# print(max_p1, ' higher than ', lim1)
						# fp1 = myfreq[np.argmax(sum_p1)]
					# else:
						# fp1 = 0.
						# print(max_p1, ' lower than ', lim1)
						
					# if max_p2 >= lim2:
						# fp2 = myfreq[med_level + np.argmax(sum_p2)]
						# print(max_p2, ' higher than ', lim2)
					# else:
						# fp2 = 0.
						# print(max_p2, ' lower than ', lim2)
					
					mydict['fp1'].append(fp1)
					mydict['fp2'].append(fp2)
					
					if config['save_plot'] == 'ON':
						fig, ax = plt.subplots()
						ax.set_title(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '_t_ini' + str(t_ini))
						ax.plot(sum, 'ro-')					
					
						print(config['channel'] + '_' + filename[:-5] + '_' + str(perro) + '_' + str(t_ini) + '.png')
						plt.savefig(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
					
					count_abs += 1

					
					perro += 1
				count += 1
			
			t_acum += len(signal)/config['fs']
		
		if config['save'] == 'ON':
			# writer = pd.ExcelWriter(config['name'] + '_AE_Burst_Features.xlsx')
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts_WPD')
			writer.close()
		
		
		newdict = {}
		arrival = []
		newdict['acum'] = []
		newdict['fmax'] = []
		newdict['cef'] = []
		newdict['rmsf'] = []
		newdict['stf'] = []
		
		newdict['xcorr'] = []
		newdict['fp1'] = []
		newdict['fp2'] = []
		
		cont = 0
		rms_sum = 0.
		for i in range(len(mydict['arrival'])):
			if mydict['arrival_acum'][i] != '':
				cont += 1
				arrival.append(mydict['arrival_acum'][i])
				newdict['acum'].append(cont)
				newdict['fmax'].append(mydict['fmax'][i])
				newdict['cef'].append(mydict['cef'][i])
				newdict['rmsf'].append(mydict['rmsf'][i])
				newdict['stf'].append(mydict['stf'][i])
				
				newdict['xcorr'].append(mydict['xcorr'][i])
				newdict['fp1'].append(mydict['fp1'][i])
				newdict['fp2'].append(mydict['fp2'][i])
				
				
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_CorrAE_Burst_Features.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=newdict, index=arrival)		
			
			DataFr_max.to_excel(writer, sheet_name='Corr_Bursts')		

			writer.close()
			# Output = {'config':config}
			# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pickle('config_' + config['name'] +'.pkl', config)
		
	
	elif config['mode'] == 'detector_wpd_thr_nocorr':
		print('++++++++WPD')
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
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']

			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == idf if f[-4:] == 'tdms']
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-3:] == config['extension']]
			
		
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		
		
		mydict = {}
		mydict['fmax'] = []
		mydict['cef'] = []
		mydict['rmsf'] = []
		mydict['stf'] = []
		
		mydict['arrival'] = []
		mydict['arrival_acum'] = []
		
		mydict['fp1'] = []
		mydict['fp2'] = []
		
		rownames = []
		
		count_abs = 1
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt')
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]

			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			
			
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini, t_fin in zip(t_burst_corr1, t_burst_corr_rev1):			
					value = t_ini
					if config['short_burst'] == 'ON':
						print('with short burst!')
						burst_it = signal[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
					elif config['short_burst'] == 'PDT':
						print('with PDT burst!')
						burst_it = signal[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['pdt'])]
					elif config['short_burst'] == 'PDT_left':
						print('with PDT left burst!')
						burst_it = signal[int(t_ini*config['fs']) - int(config['pdt']/5) : int(t_ini*config['fs']) + int(config['pdt'])]
					else:
						burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
						if len(burst_it) == 0:
							burst_it = signal[int(t_ini*config['fs']): int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
						if len(burst_it) == 0:
							burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']/2*config['fs'])]
							
					# plt.plot(burst_it)
					# plt.show()
					t_it = np.arange(len(burst_it))/config['fs']
					n_it = len(burst_it)
					
					dt = 1./config['fs']
					
					
					select_level = config['level']
		
					wavelet_mother = config['wv_mother']
					nico = pywt.WaveletPacket(data=burst_it, wavelet=wavelet_mother, maxlevel=select_level)
					
					# print('ini, fin, len, burst ++++++')
					# print(t_ini)
					# print(t_fin)
					# print(len(burst_it))
					
					mylevels = [node.path for node in nico.get_level(select_level, 'freq')]
					
					
					myfreq = np.arange(len(mylevels))*config['fs']/(2**(1+config['level']))
					myfreq += config['fs']/(2**(2+config['level']))
					# print(myfreq)
					# aa = input('pause....')
					
					cwtmatr = []
					count = 0
					for level in mylevels:
						mywav = nico[level].data
						
						if config['inverse_wv'] != 'ON':
							xold = np.linspace(0., 1., len(mywav))
							xnew = np.linspace(0., 1., n_it)				
							mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)	
						else:
							print('inverse WV!')
							gato = pywt.WaveletPacket(data=None, wavelet=wavelet_mother, maxlevel=config['level'])
							gato[level] = nico[level].data
							mywav_int = gato.reconstruct(update=False)
							
						cwtmatr.append(mywav_int)
					cwtmatr = np.array(cwtmatr)
					# print(cwtmatr)
					
					
					
					

					# fig, ax = plt.subplots()
					# extent_ = [0, np.max(t_it), 0, 500]
					# ax.contour(cwtmatr, extent=extent_)					
					# plt.show()
					

					
					sum = np.zeros(len(cwtmatr))
					for i in range(len(cwtmatr)):
						sum[i] = np.sum((cwtmatr[i])**2.0)
					

					
					# sumalt = np.zeros(len(sum)) 
					# sumalt[12:] = sum[12:]			
					# fmax = myfreq[np.argmax(sumalt)]	


					fmax = myfreq[np.argmax(sum)]

					
					sum_fc = 0.
					sum_rmsf = 0.
					sum_sum = np.sum(sum)
					for i in range(len(mylevels)):
						sum_fc += sum[i]*myfreq[i]
						sum_rmsf += sum[i]*(myfreq[i])**2.0
					cef = sum_fc/sum_sum
					rmsf = (sum_rmsf/sum_sum)**0.5
					
					
					sum_stf = 0.
					for i in range(len(mylevels)):
						sum_stf += sum[i]*(myfreq[i]-cef)**2.0
					stf = (sum_stf/sum_sum)**0.5
					
					
					mydict['arrival'].append(value)
					mydict['arrival_acum'].append(value+t_acum)
					rownames.append(str(perro) + '_' + filename)
					
					mydict['fmax'].append(fmax)
					mydict['cef'].append(cef)
					mydict['rmsf'].append(rmsf)
					mydict['stf'].append(stf)
					
					
					
					
					fpeaks = np.zeros(2)
					med_level = int(np.median(np.arange(len(mylevels))))
					
					sum_p1 = sum[0:med_level]
					sum_p2 = sum[med_level:]
					idx_p1 = np.arange(med_level)
					idx_p2 = np.arange(med_level, len(mylevels))

					lim = 0.9
					if np.argmax(sum) in list(idx_p1):					
						fpeaks[0] = fmax
						if np.max(sum_p2) >= lim*np.max(sum):
							fpeaks[1] = myfreq[med_level + np.argmax(sum_p2)]
						else:
							fpeaks[1] = 0.
						
					elif np.argmax(sum) in list(idx_p2):
						fpeaks[1] = fmax
						if np.max(sum_p1) >= lim*np.max(sum):
							fpeaks[0] = myfreq[np.argmax(sum_p1)]
						else:
							fpeaks[0] = 0.
						
					else:
						print('warning 588!!!')
						fpeaks[0] = fmax
						if np.max(sum_p2) >= lim*np.max(sum):
							fpeaks[1] = myfreq[med_level + np.argmax(sum_p2)]
						else:
							fpeaks[1] = 0.
					
					fp1 = fpeaks[0]
					fp2 = fpeaks[1]
					

					
					mydict['fp1'].append(fp1)
					mydict['fp2'].append(fp2)
					
					if config['save_plot'] == 'ON':
						fig, ax = plt.subplots()
						ax.set_title(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '_t_ini' + str(t_ini))
						ax.plot(sum, 'ro-')					
					
						print(config['channel'] + '_' + filename[:-5] + '_' + str(perro) + '_' + str(t_ini) + '.png')
						plt.savefig(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
					
					count_abs += 1

					
					perro += 1
				count += 1
			
			t_acum += len(signal)/config['fs']
		
		if config['save'] == 'ON':
			# writer = pd.ExcelWriter(config['name'] + '_AE_Burst_Features.xlsx')
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts_WPD')
			writer.close()
		
		
		newdict = {}
		arrival = []
		newdict['acum'] = []
		newdict['fmax'] = []
		newdict['cef'] = []
		newdict['rmsf'] = []
		newdict['stf'] = []
		
		newdict['fp1'] = []
		newdict['fp2'] = []
		
		cont = 0
		rms_sum = 0.
		for i in range(len(mydict['arrival'])):
			if mydict['arrival_acum'][i] != '':
				cont += 1
				arrival.append(mydict['arrival_acum'][i])
				newdict['acum'].append(cont)
				newdict['fmax'].append(mydict['fmax'][i])
				newdict['cef'].append(mydict['cef'][i])
				newdict['rmsf'].append(mydict['rmsf'][i])
				newdict['stf'].append(mydict['stf'][i])
				
				newdict['fp1'].append(mydict['fp1'][i])
				newdict['fp2'].append(mydict['fp2'][i])
				
				
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_CorrAE_Burst_Features.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=newdict, index=arrival)		
			
			DataFr_max.to_excel(writer, sheet_name='Corr_Bursts')		

			writer.close()
			# Output = {'config':config}
			# stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			save_pickle('config_' + config['name'] +'.pkl', config)
		
		
	
	elif config['mode'] == 'detector_wpd_features_thr':
		print('++++++++cwt')
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		
		
		rownames = []
		
		count_abs = 1
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			signal = load_signal(filepath, channel=config['channel'])
			
			if config['range'] != None:
				signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
			
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
					print('db out schottland amt')
					signal = signal/70.8
				elif config['db_out'] == 43:
					print('db out bochum')
					signal = signal/141.25
			
			if config['units'] != 'v':
				if config['units'] == 'uv':
					signal = signal*1000.*1000.
				elif config['units'] == 'mv':
					signal = signal*1000.
			
			t = [i/config['fs'] for i in range(len(signal))]

			
			
			
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			from THR_Burst_Detection import plot_burst_rev
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			
			
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini, t_fin in zip(t_burst_corr1, t_burst_corr_rev1):			
					
					if config['short_burst'] == 'ON':
						print('with short burst!')
						burst_it = signal[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
					else:					
						burst_it = signal[int(t_ini*config['fs']) - int(config['window_time']/2*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
					
					t_it = np.arange(len(burst_it))/config['fs']
					n_it = len(burst_it)
					
					dt = 1./config['fs']
					
					
					select_level = config['level']
		
					wavelet_mother = 'db6'
					nico = pywt.WaveletPacket(data=burst_it, wavelet=wavelet_mother, maxlevel=select_level)
					
					mylevels = [node.path for node in nico.get_level(select_level, 'freq')]
					
					myfreq = np.arange(len(mylevels))
					myfreq = (myfreq + 1)*config['fs']/(2**(2+config['level']))
					
					cwtmatr = []
					count = 0
					for level in mylevels:
						mywav = nico[level].data
						
						xold = np.linspace(0., 1., len(mywav))
						xnew = np.linspace(0., 1., n_it)				
						mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)		
						
						cwtmatr.append(mywav_int)
					cwtmatr = np.array(cwtmatr)
					print(cwtmatr)
					
					
					
					

					
					# extent_ = [0, np.max(t_it), 0, 500]
					# ax.contour(cwtmatr, extent=extent_)					
					
					

					
					sum = np.zeros(len(cwtmatr))
					for i in range(len(cwtmatr)):
						sum[i] = sum[i] = np.sum((cwtmatr[i])**2.0)
					
					
					fmax = myfreq[np.argmax(sum)]					
					sum_fc = 0.
					for i in range(len(mylevels)):
						sum_fc += sum[i]*myfreq[i]
					cef = sum_fc/np.sum(sum)
					
					

					
					

					
					
					
					
					print(config['channel'] + '_' + filename[:-5] + '_' + str(perro) + '_' + str(t_ini) + '.png')
					plt.savefig(str(count_abs) + '_rel_' + str(perro) + '_' + config['channel'] + '_' + filename[:-5] + '.png')
					
					count_abs += 1

					
					perro += 1
				count += 1
	
	
	elif config['mode'] == 'freqs_bursts':
	
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		# mydict = {}
		# mydict['arrival'] = []
		# mydict['arrival_acum'] = []
		# mydict['rms'] = []
		# mydict['amax'] = []
		# mydict['count'] = []
		# mydict['rise'] = []
		# mydict['dura'] = []
		# mydict['freq'] = []
		# mydict['kurt'] = []
		# mydict['ra'] = []
		# mydict['dc'] = []
		# mydict['crest'] = []
		
		intervals, Intervals_Dict, str_intervals = Lgenerate_n_intervals_noinf(10.e3, 1000.e3, 50)
		
		
		rownames = []
		
		count = 0
		myrows = {}
		t_acum = 0.
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			signal = 1000*load_signal(filepath, channel=config['channel'])/70.8
			
			t = [i/config['fs'] for i in range(len(signal))]
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
				
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				from THR_Burst_Detection import plot_burst_rev
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
					value = t_ini1
					rownames.append(str(perro) + '_' + filename)
					# burst = signal[int(t_ini1*config['fs']): int(t_fin1*config['fs'])]
					# spec, f, df = mag_fft(burst, config['fs'])
					# plt.plot(f, spec, 'k')
					# plt.show()
					burst = signal[int(t_ini1*config['fs']) : int(t_fin1*config['fs'])]
					
					magX, f, df = mag_fft(x=burst, fs=config['fs'])
			
					for i in range(len(intervals)-1):				
						Intervals_Dict[str_intervals[i]].append(energy_in_band(magX, df, intervals[i], intervals[i+1]))
				
					perro += 1
					
			
		mydict = {}
			
		# row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')
		
		for element in str_intervals:
			mydict[element] = Intervals_Dict[element]



		DataFr = pd.DataFrame(data=mydict, index=rownames)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='AE_Fft')	
		print('Result in Excel table XXX')		

				
	elif config['mode'] == 'bursts_per_file':
	
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
	
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]

		mydict = {}
		mydict['bursts'] = []

		
		rownames = []
		
		count = 0
		myrows = {}
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			signal = 1000*load_signal(filepath, channel=config['channel'])/70.8
			
			t = [i/config['fs'] for i in range(len(signal))]
			# signal = butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			# signal = butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			if config['filter'][0] != 'OFF':
				print(config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
			
			# print(scipy.stats.kurtosis(signal, fisher=False)**0.25)
			# signal = signal / scipy.stats.kurtosis(signal, fisher=False)**0.25
			
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0, threshold=threshold)
			
			
			
			
			if config['plot'] == 'ON':
				from THR_Burst_Detection import plot_burst_rev
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			num = len(t_burst_corr1)

			mydict['bursts'].append(num)
				
				
				
				
				
			rownames.append(filename)
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts')
			writer.close()		
		
			# save_pickle('config_' + config['name'] +'.pkl', config)	

	elif config['mode'] == 'plot_features_oneclass_var1':
		print('Waehlen XLS von Klasse ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print('Waehlen XLS von Klasse 1 -1 (schlecht)')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths2 = filedialog.askopenfilenames()
		# root.destroy()
		
		names_features = ['crest', 'dc', 'ra']
		fig, ax = plt.subplots(nrows=3, ncols=3)
		# names_features = ['count', 'dura']
		# plt.xlabel('Überschwingungen', fontsize=14), plt.ylabel('Dauer [us]', fontsize=14)
		# names_features = ['amax', 'rms']
		# plt.xlabel('Max. Amplitude [mV]', fontsize=14), plt.ylabel('RMS [mV]', fontsize=14)
		# names_features = ['rise', 'kurt']
		# plt.xlabel('Anstiegszeit [us]', fontsize=14), plt.ylabel('Kurtosis', fontsize=14)
		# names_features = ['crest', 'freq']
		# plt.xlabel('Crest Faktor', fontsize=14), plt.ylabel('Hauptfrequenz [kHz]', fontsize=14)
		
		
		
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [1 for i in range(len(mydict[element]))]
		# for filepath in Filepaths2:
			# mydict = pd.read_excel(filepath)			
			# mydict = mydict.to_dict(orient='list')			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
			# Labels += [-1 for i in range(len(mydict[element]))]

		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)		
		
		# Idx_Gut = [i for i in range(n_samples) if Labels[i] == 1]
		# Idx_Schlecht = [i for i in range(n_samples) if Labels[i] == -1]
		
		alpha = 0.5
		max_dc = 250000
		max_ra = 30000
		fontsize_big = 13
		
		ax[0][1].scatter(np.array(Dict_Features['crest']), np.array(Dict_Features['dc']), color='red', alpha=alpha)
		ax[0][1].set_xlabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[0][1].set_ylabel('DC [1/s]', fontsize=fontsize_big)
		ax[0][1].set_xlim(left=0, right=10)
		ax[0][1].set_ylim(bottom=0, top=max_dc)
		
		ax[0][2].scatter(np.array(Dict_Features['crest']), np.array(Dict_Features['ra']), color='red', alpha=alpha)
		ax[0][2].set_xlabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[0][2].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		ax[0][2].set_xlim(left=0, right=10)
		ax[0][2].set_ylim(bottom=0, top=max_ra)
		
		
		ax[1][0].scatter(np.array(Dict_Features['dc']), np.array(Dict_Features['crest']), color='red', alpha=alpha)
		ax[1][0].set_xlabel('DC [1/s]', fontsize=fontsize_big)
		ax[1][0].set_ylabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[1][0].set_xlim(left=0, right=max_dc)
		ax[1][0].set_ylim(bottom=0, top=10)
		
		ax[1][2].scatter(np.array(Dict_Features['dc']), np.array(Dict_Features['ra']), color='red', alpha=alpha)
		ax[1][2].set_xlabel('DC [1/s]', fontsize=fontsize_big)
		ax[1][2].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		ax[1][2].set_xlim(left=0, right=max_dc)
		ax[1][2].set_ylim(bottom=0, top=max_ra)
		
		
		ax[2][0].scatter(np.array(Dict_Features['ra']), np.array(Dict_Features['crest']), color='red', alpha=alpha)
		ax[2][0].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[2][0].set_ylabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[2][0].set_xlim(left=0, right=max_ra)
		ax[2][0].set_ylim(bottom=0, top=10)
		
		ax[2][1].scatter(np.array(Dict_Features['ra']), np.array(Dict_Features['dc']), color='red', alpha=alpha)
		ax[2][1].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[2][1].set_ylabel('DC [1/s]', fontsize=fontsize_big)
		ax[2][1].set_xlim(left=0, right=max_ra)
		ax[2][1].set_ylim(bottom=0, top=max_dc)
		
		ax[0][0].hist(np.array(Dict_Features['crest']), color='blue', log=True)
		ax[0][0].set_xlabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[0][0].set_ylabel('Ocurrence', fontsize=fontsize_big)
		ax[0][0].set_xlim(left=0, right=10)
		
		ax[1][1].hist(np.array(Dict_Features['dc']), color='blue', log=True)
		ax[1][1].set_xlabel('DC [1/s]', fontsize=fontsize_big)
		ax[1][1].set_ylabel('Ocurrence', fontsize=fontsize_big)
		ax[1][1].set_xlim(left=0, right=max_dc)
		
		ax[2][2].hist(np.array(Dict_Features['ra']), color='blue', log=True)
		ax[2][2].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[2][2].set_ylabel('Ocurrence', fontsize=fontsize_big)
		ax[2][2].set_xlim(left=0, right=max_ra)
		
		
		
		for i in range(3):
			for j in range(3):
				ax[i][j].tick_params(axis='both', labelsize=12)
				if i != j:
					ax[i][j].ticklabel_format(style='sci', scilimits=(-2, 2))
				else:
					ax[i][j].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		# ax[0][0].legend(fontsize=12)
		# ax[0][1].legend(fontsize=12)
		# ax[1][0].legend(fontsize=12)
		# ax[1][1].legend(fontsize=12)
		fig.set_size_inches(13.5, 8.5)
		plt.tight_layout()
		plt.show()
	
	elif config['mode'] == 'bursts_clustering':

		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		

		
		Names_Features_1 = ['amax', 'rms']
		Names_Features_2 = ['cef', 'freq', 'kurtf', 'rmsf', 'stf']
		Names_Features_3 = ['crest', 'kurt']
		Names_Features_4 = ['count', 'dc', 'dura']
		Names_Features_5 = ['rise', 'ra']
		Names_Features_6 = ['amaxt', 'rmst', 'kurtt', 'crestt']
		Names_Features_7 = ['senenvt', 'kurtenvt', 'stdenvt', 'amaxenvt', 'sent', 'sen', 'cetenv', 'rmscetenv', 'amaxenv', 'stdenv', 'kurtenv', 'senenv', 'cet', 'rmscet']
		
		# Names_Features = Names_Features_1 + Names_Features_2 + Names_Features_3 + Names_Features_4 + Names_Features_5 + Names_Features_6 + Names_Features_7 #all
		
		Names_Features = ['freq', 'crest', 'kurtf', 'count', 'dura', 'amaxt', 'stdenvt', 'sen', 'cetenv', 'rmscetenv', 'amaxenv', 'stdenv', 'senenv', 'cet', 'rmscet']  #sel1
		
		# Names_Features = Names_Features_2 + Names_Features_3 + Names_Features_4 + Names_Features_5
		
		# Names_Features = Names_Features_A
		
		n_features_ = len(Names_Features)
		
		n_clusters_ = config['n_clusters']
		Features = []
		
		mydict = pd.read_excel(filepath)		
		mydict = mydict.to_dict(orient='list')
		# print(mydict)
		n_samples_ = len(mydict[Names_Features[0]])
		
		Features = []
		for i in range(n_samples_):
			data = []
			for name in Names_Features:
				data.append(mydict[name][i])
			Features.append(data)
		Features = np.array(Features)
		
		
		from sklearn.preprocessing import StandardScaler
		scaler = StandardScaler()
		scaler.fit(Features)
		Features = scaler.transform(Features)
		
		# from sklearn.decomposition import PCA
		# pca_comps_ = 5
		# pca = PCA(n_components=pca_comps_)
		# pca.fit(Features)  
		# print('Variance_Ratio = ', pca.explained_variance_ratio_)
		# Features = pca.transform(Features)
		# n_features_ = pca_comps_
		# # print(Features)
		# # sys.exit()
		
		
		
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=config['n_clusters'], random_state=0)
		kmeans.fit(Features)
		
		print('\n++++++++++++++++++++')
		Cardinal = np.zeros(config['n_clusters'])
		Dict_Bursts = {'0':[], '1':[], '2':[], '3':[], '4':[]}
		for i in range(n_samples_):
			clf = kmeans.predict([Features[i]])
			Cardinal[clf[0]] += 1
			Dict_Bursts[str(clf[0])].append(i+1)
		print('Percentage per Cluster: ', 100*Cardinal/np.sum(Cardinal))
		
		# print(np.linalg.norm(np.array([0, 0]) - np.array([2, 2])))
		# print((8)**0.5)
		# sys.exit()
		
		print('\n++++++++++++++++++++')
		Centers = kmeans.cluster_centers_
		Repre = np.zeros(config['n_clusters'])
		for k in range(config['n_clusters']):			
			dist = np.zeros(n_samples_)
			for i in range(n_samples_):
				dist[i] = np.linalg.norm(Features[i] - Centers[k])
			Repre[k] = np.argmin(dist)
		print('Represent per Cluster: ', Repre+1)
		
		print('\n++++++++++')
		mydict2 = {}
		for k in range(config['n_clusters']):
			print('--------Repre Cluster ', k, ' = ', Repre[k]+1)
			mydict = {}
			for i in range(n_features_):
				mydict[Names_Features[i]] = round(Features[int(Repre[k])][i], 5)
				if k != 0:
					mydict2[Names_Features[i]].append(Features[int(Repre[k])][i])
				else:
					mydict2[Names_Features[i]] = []
					mydict2[Names_Features[i]].append(Features[int(Repre[k])][i])
			print(mydict)
		# print(mydict2)
		DataFr = pd.DataFrame(data=mydict2, index=np.arange(config['n_clusters']))
		writer = pd.ExcelWriter('info' + '.xlsx')
		
		DataFr.to_excel(writer, sheet_name='OV_Features')
		writer.close()				
		print('Result in Excel table')
		
		
		
		# print(Features[80])
		# print(Features[81])
		# print(Features[82])
		print("Inertia : {:.3e}".format(kmeans.inertia_))
		
		print(Dict_Bursts)
		sys.exit()
		# data = []
		# for name in Names_Features:
			# data += 
		# Feature.append(mydict[config['feature']][:-2]
		
		
		Feature = list(np.nan_to_num(Feature))		
		n = len(Feature)		
		if config['feature'] == 'Avg':
			for i in range(len(Feature)):
				if Feature[i] == 0:
					Feature[i] = Feature[i+1]
					print('!!!!!!!!!!!')		
		feature_raw = movil_avg(Feature, 1)

		
		
		
		
		Atributo = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
	
	
	elif config['mode'] == 'bursts_features_eval':

		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		

		
		Names_Features_1 = ['amax', 'rms']
		Names_Features_2 = ['cef', 'freq', 'kurtf', 'rmsf', 'stf']
		Names_Features_3 = ['crest', 'kurt']
		Names_Features_4 = ['count', 'dc', 'dura']
		Names_Features_5 = ['rise', 'ra']
		Names_Features_6 = ['amaxt', 'rmst', 'kurtt', 'crestt']
		Names_Features_7 = ['senenvt', 'kurtenvt', 'stdenvt', 'amaxenvt', 'sent', 'sen', 'cetenv', 'rmscetenv', 'amaxenv', 'stdenv', 'kurtenv', 'senenv', 'cet', 'rmscet']
		
		Names_Features = Names_Features_1 + Names_Features_2 + Names_Features_3 + Names_Features_4 + Names_Features_5 + Names_Features_6 + Names_Features_7 #all
		
		# Names_Features = Names_Features_2 + Names_Features_3 + Names_Features_4 + Names_Features_5
		
		# Names_Features = Names_Features_A
		
		n_features_ = len(Names_Features)
		
		Features = []
		
		mydict = pd.read_excel(filepath)		
		mydict = mydict.to_dict(orient='list')
		n_samples_ = len(mydict[Names_Features[0]])
		
		Features = []
		for i in range(n_samples_):
			data = []
			for name in Names_Features:
				data.append(mydict[name][i])
			Features.append(data)
		Features = np.array(Features)
		
		
		from sklearn.preprocessing import StandardScaler
		scaler = StandardScaler()
		scaler.fit(Features)
		Features = scaler.transform(Features)
		

		
		
		target = np.zeros(n_samples_)
		idx_pos = np.array([10, 44, 68, 71, 72, 77, 86, 130, 134, 137, 145, 150, 153, 163, 165, 171, 180, 187, 189, 193, 195, 206, 207, 220, 227, 231, 241, 257])-1
		target[idx_pos] = 1.
		from sklearn.svm import SVC
		clf = SVC(kernel='linear')
		
		from sklearn.feature_selection import RFE
		rfe = RFE(estimator=clf, step=1)
		rfe = rfe.fit(Features, target)
		
		print(rfe.ranking_)
		print(Features[0])
		print(Names_Features)
	
	elif config['mode'] == 'plot_features_threeclass_var1':
		print('Waehlen XLS von Klasse 1')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Waehlen XLS von Klasse 2')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
		print('Waehlen XLS von Klasse 3')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths3 = filedialog.askopenfilenames()
		root.destroy()
		
		names_features = ['crest', 'dc', 'ra']
		fig, ax = plt.subplots(nrows=3, ncols=3)

		
		
		
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [1 for i in range(len(mydict[element]))]
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [2 for i in range(len(mydict[element]))]
		for filepath in Filepaths3:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [3 for i in range(len(mydict[element]))]

		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)		
		
		Idx_1 = [i for i in range(n_samples) if Labels[i] == 1]
		Idx_2 = [i for i in range(n_samples) if Labels[i] == 2]
		Idx_3 = [i for i in range(n_samples) if Labels[i] == 3]
		
		alpha = 0.5
		alpha_2 = 0.5
		max_dc = 250000
		max_ra = 30000
		fontsize_big = 13
		color_1 = 'red'
		color_2 = 'green'
		color_3 = 'blue'
		
		# color_1 = 'green' 
		# color_2 = 'blue' 
		# color_3 = 'red' 
		
		
		color_hist = 'gray'
		label_1 = 'AE-1'
		label_2 = 'AE-2'
		label_3 = 'AE-3'
		max_hist = 3.e3
		max_crest = 10
		
		ax[0][1].scatter(np.array(Dict_Features['crest'])[Idx_1], np.array(Dict_Features['dc'])[Idx_1], color=color_1, alpha=alpha, label=label_1)
		ax[0][1].scatter(np.array(Dict_Features['crest'])[Idx_2], np.array(Dict_Features['dc'])[Idx_2], color=color_2, alpha=alpha, label=label_2)
		ax[0][1].scatter(np.array(Dict_Features['crest'])[Idx_3], np.array(Dict_Features['dc'])[Idx_3], color=color_3, alpha=alpha, label=label_3)
		ax[0][1].set_xlabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[0][1].set_ylabel('DC [1/s]', fontsize=fontsize_big)
		ax[0][1].set_xlim(left=0, right=max_crest)
		ax[0][1].set_ylim(bottom=0, top=max_dc)
		
		ax[0][2].scatter(np.array(Dict_Features['crest'])[Idx_1], np.array(Dict_Features['ra'])[Idx_1], color=color_1, alpha=alpha, label=label_1)
		ax[0][2].scatter(np.array(Dict_Features['crest'])[Idx_2], np.array(Dict_Features['ra'])[Idx_2], color=color_2, alpha=alpha, label=label_2)
		ax[0][2].scatter(np.array(Dict_Features['crest'])[Idx_3], np.array(Dict_Features['ra'])[Idx_3], color=color_3, alpha=alpha, label=label_3)
		ax[0][2].set_xlabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[0][2].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		ax[0][2].set_xlim(left=0, right=max_crest)
		ax[0][2].set_ylim(bottom=0, top=max_ra)
		
		
		ax[1][0].scatter(np.array(Dict_Features['dc'])[Idx_1], np.array(Dict_Features['crest'])[Idx_1], color=color_1, alpha=alpha, label=label_1)
		ax[1][0].scatter(np.array(Dict_Features['dc'])[Idx_2], np.array(Dict_Features['crest'])[Idx_2], color=color_2, alpha=alpha, label=label_2)
		ax[1][0].scatter(np.array(Dict_Features['dc'])[Idx_3], np.array(Dict_Features['crest'])[Idx_3], color=color_3, alpha=alpha, label=label_3)
		ax[1][0].set_xlabel('DC [1/s]', fontsize=fontsize_big)
		ax[1][0].set_ylabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[1][0].set_xlim(left=0, right=max_dc)
		ax[1][0].set_ylim(bottom=0, top=max_crest)
		
		ax[1][2].scatter(np.array(Dict_Features['dc'])[Idx_1], np.array(Dict_Features['ra'])[Idx_1], color=color_1, alpha=alpha, label=label_1)
		ax[1][2].scatter(np.array(Dict_Features['dc'])[Idx_2], np.array(Dict_Features['ra'])[Idx_2], color=color_2, alpha=alpha, label=label_2)
		ax[1][2].scatter(np.array(Dict_Features['dc'])[Idx_3], np.array(Dict_Features['ra'])[Idx_3], color=color_3, alpha=alpha, label=label_3)
		ax[1][2].set_xlabel('DC [1/s]', fontsize=fontsize_big)
		ax[1][2].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		ax[1][2].set_xlim(left=0, right=max_dc)
		ax[1][2].set_ylim(bottom=0, top=max_ra)
		
		
		ax[2][0].scatter(np.array(Dict_Features['ra'])[Idx_1], np.array(Dict_Features['crest'])[Idx_1], color=color_1, alpha=alpha, label=label_1)
		ax[2][0].scatter(np.array(Dict_Features['ra'])[Idx_2], np.array(Dict_Features['crest'])[Idx_2], color=color_2, alpha=alpha, label=label_2)
		ax[2][0].scatter(np.array(Dict_Features['ra'])[Idx_3], np.array(Dict_Features['crest'])[Idx_3], color=color_3, alpha=alpha, label=label_3)
		ax[2][0].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[2][0].set_ylabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[2][0].set_xlim(left=0, right=max_ra)
		ax[2][0].set_ylim(bottom=0, top=max_crest)
		
		ax[2][1].scatter(np.array(Dict_Features['ra'])[Idx_1], np.array(Dict_Features['dc'])[Idx_1], color=color_1, alpha=alpha, label=label_1)
		ax[2][1].scatter(np.array(Dict_Features['ra'])[Idx_2], np.array(Dict_Features['dc'])[Idx_2], color=color_2, alpha=alpha, label=label_2)
		ax[2][1].scatter(np.array(Dict_Features['ra'])[Idx_3], np.array(Dict_Features['dc'])[Idx_3], color=color_3, alpha=alpha, label=label_3)
		ax[2][1].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[2][1].set_ylabel('DC [1/s]', fontsize=fontsize_big)
		ax[2][1].set_xlim(left=0, right=max_ra)
		ax[2][1].set_ylim(bottom=0, top=max_dc)
		
		ax[0][0].hist(np.array(Dict_Features['crest'])[Idx_1], color=color_1, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_1)
		ax[0][0].hist(np.array(Dict_Features['crest'])[Idx_2], color=color_2, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_2)
		ax[0][0].hist(np.array(Dict_Features['crest'])[Idx_3], color=color_3, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_3)
		ax[0][0].set_xlabel('Crest Factor [-]', fontsize=fontsize_big)
		ax[0][0].set_ylabel('Ocurrence', fontsize=fontsize_big)
		ax[0][0].set_xlim(left=0, right=max_crest)
		ax[0][0].set_ylim(bottom=0, top=max_hist)
		
		ax[1][1].hist(np.array(Dict_Features['dc'])[Idx_1], color=color_1, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_1)
		ax[1][1].hist(np.array(Dict_Features['dc'])[Idx_2], color=color_2, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_2)
		ax[1][1].hist(np.array(Dict_Features['dc'])[Idx_3], color=color_3, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_3)
		ax[1][1].set_xlabel('DC [1/s]', fontsize=fontsize_big)
		ax[1][1].set_ylabel('Ocurrence', fontsize=fontsize_big)
		ax[1][1].set_xlim(left=0, right=max_dc)
		ax[1][1].set_ylim(bottom=0, top=max_hist)
		
		ax[2][2].hist(np.array(Dict_Features['ra'])[Idx_1], color=color_1, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_1)
		ax[2][2].hist(np.array(Dict_Features['ra'])[Idx_2], color=color_2, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_2)
		ax[2][2].hist(np.array(Dict_Features['ra'])[Idx_3], color=color_3, log=True, alpha=alpha_2, ec='black', histtype='stepfilled', label=label_3)
		ax[2][2].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[2][2].set_ylabel('Ocurrence', fontsize=fontsize_big)
		ax[2][2].set_xlim(left=0, right=max_ra)
		ax[2][2].set_ylim(bottom=0, top=max_hist)
		
		
		
		for i in range(3):
			for j in range(3):
				ax[i][j].tick_params(axis='both', labelsize=12)
				
				if i != j:
					ax[i][j].ticklabel_format(style='sci', scilimits=(-2, 2))
				else:
					ax[i][j].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
					ax[i][j].legend(fontsize=10)
		# ax[0][0].legend(fontsize=12)
		# ax[0][1].legend(fontsize=12)
		# ax[1][0].legend(fontsize=12)
		# ax[1][1].legend(fontsize=12)
		fig.set_size_inches(13.5, 8.5)
		plt.tight_layout()
		plt.show()
	
	
	elif config['mode'] == 'plot_features_oneclass_var2':
		print('Waehlen XLS von Klasse ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print('Waehlen XLS von Klasse 1 -1 (schlecht)')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths2 = filedialog.askopenfilenames()
		# root.destroy()
		
		names_features = ['ra', 'amax', 'rms', 'rise']
		fig, ax = plt.subplots(nrows=4, ncols=4)
		# names_features = ['count', 'dura']
		# plt.xlabel('Überschwingungen', fontsize=14), plt.ylabel('Dauer [us]', fontsize=14)
		# names_features = ['amax', 'rms']
		# plt.xlabel('Max. Amplitude [mV]', fontsize=14), plt.ylabel('RMS [mV]', fontsize=14)
		# names_features = ['rise', 'kurt']
		# plt.xlabel('Anstiegszeit [us]', fontsize=14), plt.ylabel('Kurtosis', fontsize=14)
		# names_features = ['crest', 'freq']
		# plt.xlabel('Crest Faktor', fontsize=14), plt.ylabel('Hauptfrequenz [kHz]', fontsize=14)
		
		
		
		Dict_Features = {}
		for feature in names_features:
			Dict_Features[feature] = []
		
		Labels = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)			
			mydict = mydict.to_dict(orient='list')			
			for element in names_features:
				Dict_Features[element] += mydict[element]
			Labels += [1 for i in range(len(mydict[element]))]
		# for filepath in Filepaths2:
			# mydict = pd.read_excel(filepath)			
			# mydict = mydict.to_dict(orient='list')			
			# for element in names_features:
				# Dict_Features[element] += mydict[element]
			# Labels += [-1 for i in range(len(mydict[element]))]

		n_samples = len(Dict_Features[names_features[0]])
		n_features = len(names_features)		
		
		# Idx_Gut = [i for i in range(n_samples) if Labels[i] == 1]
		# Idx_Schlecht = [i for i in range(n_samples) if Labels[i] == -1]
		
		fontsize_big = 13
		alpha = 0.5
		ax[0][1].scatter(np.array(Dict_Features['ra']), np.array(Dict_Features['amax']), color='red', alpha=alpha)
		ax[0][1].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[0][1].set_ylabel('Max. [V]', fontsize=fontsize_big)
		
		ax[0][2].scatter(np.array(Dict_Features['ra']), np.array(Dict_Features['rms']), color='red', alpha=alpha)
		ax[0][2].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[0][2].set_ylabel('RMS Value [V]', fontsize=fontsize_big)
		
		ax[0][3].scatter(np.array(Dict_Features['ra']), np.array(Dict_Features['rise']), color='red', alpha=alpha)
		ax[0][3].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		ax[0][3].set_ylabel('Rise [s]', fontsize=fontsize_big)
		
		
		
		ax[1][0].scatter(np.array(Dict_Features['amax']), np.array(Dict_Features['ra']), color='red', alpha=alpha)
		ax[1][0].set_xlabel('Max. [V]', fontsize=fontsize_big)
		ax[1][0].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		
		ax[1][2].scatter(np.array(Dict_Features['amax']), np.array(Dict_Features['rms']), color='red', alpha=alpha)
		ax[1][2].set_xlabel('Max. [V]', fontsize=fontsize_big)
		ax[1][2].set_ylabel('RMS Value [V]', fontsize=fontsize_big)
		
		ax[1][3].scatter(np.array(Dict_Features['amax']), np.array(Dict_Features['rise']), color='red', alpha=alpha)
		ax[1][3].set_xlabel('Max. [V]', fontsize=fontsize_big)
		ax[1][3].set_ylabel('Rise [s]', fontsize=fontsize_big)
		
		
		
		ax[2][0].scatter(np.array(Dict_Features['rms']), np.array(Dict_Features['ra']), color='red', alpha=alpha)
		ax[2][0].set_xlabel('RMS Value [V]', fontsize=fontsize_big)
		ax[2][0].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		
		ax[2][1].scatter(np.array(Dict_Features['rms']), np.array(Dict_Features['amax']), color='red', alpha=alpha)
		ax[2][1].set_xlabel('RMS Value [V]', fontsize=fontsize_big)
		ax[2][1].set_ylabel('Max. Amplitude [V]', fontsize=fontsize_big)
		
		ax[2][3].scatter(np.array(Dict_Features['rms']), np.array(Dict_Features['rise']), color='red', alpha=alpha)
		ax[2][3].set_xlabel('RMS Value [V]', fontsize=fontsize_big)
		ax[2][3].set_ylabel('Rise [s]', fontsize=fontsize_big)
		
		
		
		
		ax[3][0].scatter(np.array(Dict_Features['rise']), np.array(Dict_Features['ra']), color='red', alpha=alpha)
		ax[3][0].set_xlabel('Rise [s]', fontsize=fontsize_big)
		ax[3][0].set_ylabel('RA [V/s]', fontsize=fontsize_big)
		
		ax[3][1].scatter(np.array(Dict_Features['rise']), np.array(Dict_Features['amax']), color='red', alpha=alpha)
		ax[3][1].set_xlabel('Rise [s]', fontsize=fontsize_big)
		ax[3][1].set_ylabel('Max. Amplitude [V]', fontsize=fontsize_big)
		
		ax[3][2].scatter(np.array(Dict_Features['rise']), np.array(Dict_Features['rms']), color='red', alpha=alpha)
		ax[3][2].set_xlabel('Rise [s]', fontsize=fontsize_big)
		ax[3][2].set_ylabel('RMS Value [V]', fontsize=fontsize_big)
		
		
		
		
		
		
		
		ax[0][0].hist(np.array(Dict_Features['ra']), color='blue')
		ax[0][0].set_xlabel('RA [V/s]', fontsize=fontsize_big)
		
		ax[1][1].hist(np.array(Dict_Features['amax']), color='blue')
		ax[1][1].set_xlabel('Max. Amplitude [V]', fontsize=fontsize_big)
		
		ax[2][2].hist(np.array(Dict_Features['rms']), color='blue')
		ax[2][2].set_xlabel('RMS Value [V]', fontsize=fontsize_big)
		
		ax[3][3].hist(np.array(Dict_Features['rise']), color='blue')
		ax[3][3].set_xlabel('Rise [s]', fontsize=fontsize_big)
		
		
		
		for i in range(3):
			for j in range(3):
				ax[i][j].tick_params(axis='both', labelsize=12)


		
		# ax[0][0].legend(fontsize=12)
		# ax[0][1].legend(fontsize=12)
		# ax[1][0].legend(fontsize=12)
		# ax[1][1].legend(fontsize=12)
		plt.tight_layout()
		plt.show()
	
	elif config['mode'] == 'mode4':

		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []
		Times = []
		
		count = 0
		Dict_Feat = {}		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath)
			
			mydict = mydict.to_dict(orient='list')
			if count == 0 :
				# names_features = list(mydict.keys())
				names_features = ['amax', 'count', 'crest', 'dc', 'dura', 'freq', 'ra','rise', 'rms', 'acum']
				for element in names_features:
					Dict_Feat[element] = []

			for element in names_features:
				Dict_Feat[element].append(mydict[element])

			count += 1
		

		n_features = len(names_features)
	
		fig, ax = plt.subplots(nrows=2, ncols=5)
		
		count = 0
		for i in range(2):
			for j in range(5):
				print(names_features[count])
				if names_features[count] != 'acum':
					ax[i][j].boxplot(Dict_Feat[names_features[count]])
				else:
					max_bursts = []
					for vec in Dict_Feat[names_features[count]]:
						max_bursts.append(np.max(np.array(vec)))
					ax[i][j].bar(np.array([1,2,3]), max_bursts)
					
				ax[i][j].tick_params(axis='both', labelsize=12)
				ax[i][j].set_xticklabels(['N.F.', 'I.F.', 'D.F.'])
				if names_features[count] == 'rms':
					ax[i][j].set_title('RMS Value [mV]', fontsize=13)
				
				elif names_features[count] == 'acum':
					ax[i][j].set_title('N° AE Bursts', fontsize=13)
					
				elif names_features[count] == 'ra':
					ax[i][j].set_title('RA [V/us]', fontsize=13)
					
				elif names_features[count] == 'dc':
					ax[i][j].set_title('DC [1/us]', fontsize=13)
					
				elif names_features[count] == 'amax':
					print('!!!!!!!!!!!!!!!!!!!')
					ax[i][j].set_title('Max. Amplitude [mV]', fontsize=13)
					
				elif names_features[count] == 'dura':
					ax[i][j].set_title('Duration [us]', fontsize=13)
					
				elif names_features[count] == 'rise':
					ax[i][j].set_title('Rise Time [us]', fontsize=13)
					
				elif names_features[count] == 'count':
					ax[i][j].set_title('Counts [-]', fontsize=13)

				elif names_features[count] == 'freq':
					ax[i][j].set_title('Main Frequency [kHz]', fontsize=13)
					
				elif names_features[count] == 'crest':
					ax[i][j].set_title('Crest Factor [-]', fontsize=13)
				
				count += 1
		
		

		plt.show()
		# print(Feature)
		
		
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
	config['stella'] = int(config['stella'])
	config['idx_fp1'] = int(config['idx_fp1'])
	config['idx_fp2'] = int(config['idx_fp2'])
	
	config['n_clusters'] = int(config['n_clusters'])
	
	config['thr_value'] = float(config['thr_value'])
	# config['highpass'] = float(config['highpass'])
	config['window_time'] = float(config['window_time'])
	# config['time_segments'] = float(config['time_segments'])
	config['lockout'] = int(config['lockout'])
	config['pdt'] = int(config['pdt'])
	config['level'] = int(config['level'])
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	if config['db_out'] != 'OFF':
		config['db_out'] = int(config['db_out'])
	
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
