
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
Inputs = ['mode', 'save', 'channel']
# InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'plot':'OFF', 'n_files':1, 'title_plot':None, 'thr_value':30., 'thr_mode':'fixed_value', 'window_time':0.05, 'save_plot':'OFF', 'file':'OFF', 'time_segments':1., 'stella':1500, 'lockout':3000, 'highpass':20.e3, 'mv':'ON', 'mypath':None}

InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'plot':'ON', 'n_files':1, 'title_plot':None, 'thr_value':60., 'thr_mode':'fixed_value', 'window_time':0.001, 'save_plot':'OFF', 'file':'OFF', 'time_segments':1., 'stella':100, 'lockout':200, 'highpass':20.e3, 'mv':'ON', 'mypath':None, 'mypath2':None, 'fs_tacho':50.e3}
# gearbox: thr_60, wt_0.001, hp_70k, stella_100, lcokout 200


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	from Main_Analysis_Mod import invoke_signal
	
	if config['mv'] == 'ON':
		amp_factor = 1000.
	else:
		amp_factor = 1.
	
	if config['mode'] == 'burst_detection_features_multi':
		print('Select signals...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		# Data = []
		Bursts_Ini = []
		Bursts_End = []
		Features = []
		for filepath in Filepaths:
			signal = load_signal(filepath, channel=config['channel'])
			# signal = 1000*butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
			signal = 1000*butter_highpass(x=signal, fs=config['fs'], freq=70.e3, order=3)
			# Data.append(signal)
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout(signal, config, count=0)
			
			Dict_Ini = {'times':t_burst_corr, 'amplitudes':amp_burst_corr}
			Dict_End = {'times':t_burst_corr_rev, 'amplitudes':amp_burst_corr_rev}
			Bursts_Ini.append(Dict_Ini)
			Bursts_End.append(Dict_End)		
			
			Dict_Feat = bursts_features(signal, t_burst_corr, t_burst_corr_rev, config)
			Features.append(Dict_Feat)
		
		save_pickle(config['name'] + '.pkl', Features)
	
	elif config['mode'] == 'burst_detection_features_multi_intervals':
		print('Select signals...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		# Data = []
		# Bursts_Ini = []
		# Bursts_End = []
		# superdict = {'max':[0., 250], 'crest':[0., 14.], 'fas':[0., 3.], 'rise':[0., 150.], 'count':[0., 70.], 'rms':[0., 70.], 'freq':[70., 350.], 'p2p':[0., 500.], 'area':[0., 25000.], 'dura':[0., 800.]}

		# abc , Main_max_Freq_Dict, cba = generate_10_intervals(superdict['max'][0], superdict['max'][1])
		# abc , Main_crest_Freq_Dict, cba = generate_10_intervals(superdict['crest'][0], superdict['crest'][1])
		# abc , Main_fas_Freq_Dict, cba = generate_10_intervals(superdict['fas'][0], superdict['fas'][1])
		# abc , Main_rise_Freq_Dict, cba = generate_10_intervals(superdict['rise'][0], superdict['rise'][1])
		# abc , Main_count_Freq_Dict, cba = generate_10_intervals(superdict['count'][0], superdict['count'][1])
		
		# abc , Main_rms_Freq_Dict, cba = generate_10_intervals(superdict['rms'][0], superdict['rms'][1])
		# abc , Main_freq_Freq_Dict, cba = generate_10_intervals(superdict['freq'][0], superdict['freq'][1])
		# abc , Main_p2p_Freq_Dict, cba = generate_10_intervals(superdict['p2p'][0], superdict['p2p'][1])
		# abc , Main_area_Freq_Dict, cba = generate_10_intervals(superdict['area'][0], superdict['area'][1])
		# abc , Main_dura_Freq_Dict, cba = generate_10_intervals(superdict['dura'][0], superdict['dura'][1])
		
		
		for filepath in Filepaths:
			Main_Features = []
			filename = os.path.basename(filepath)
			signal = load_signal(filepath, channel=config['channel'])
			# signal = 1000*butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
			if config['mv'] == 'ON':
				amp_factor = 1000.
			else:
				amp_factor = 1.
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			# Data.append(signal)
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout(signal, config, count=0)
			
			# fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
			# from THR_Burst_Detection import plot_burst_rev
			# t = [i/config['fs'] for i in range(len(signal))]
			# plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
			# plt.show()
			
			for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				dict_feat = single_burst_features(signal, t_ini, t_fin, config)
				Main_Features.append(dict_feat)
			if config['save'] == 'ON':
				save_pickle('MAIN_bursts_' + filename[:-4] + 'pkl', Main_Features)
				save_pickle('config_bursts.pkl', config)
				
			
			superdict = {'max':[0., 250], 'crest':[0., 14.], 'fas':[0., 3.], 'rise':[0., 150.], 'count':[0., 70.], 'rms':[0., 70.], 'freq':[70., 350.], 'p2p':[0., 500.], 'area':[0., 25000.], 'dura':[0., 800.]}
			for key, value in superdict.items():			
				max_intervals, max_Freq_Dict, max_str_intervals = generate_10_intervals(value[0], value[1])
				for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
					dict_feat = single_burst_features(signal, t_ini, t_fin, config)
					# Main_Features.append(dict_feat)				
					max_Freq_Dict = acum_in_dict(key, dict_feat, max_intervals, max_Freq_Dict, max_str_intervals)
					# Main_max_Freq_Dict = acum_in_dict(key, dict_feat, max_intervals, Main_max_Freq_Dict, max_str_intervals)

					# print(key + '!!!!!!!!!!!!!!!!!!!!!!!!!!')
					# print(max_Freq_Dict)
				if config['save'] == 'ON':
					save_pickle(key + '_bursts_freq_' + filename[:-4] + 'pkl', max_Freq_Dict)
			
			# Dict_Feat = bursts_features(signal, t_burst_corr, t_burst_corr_rev, config)
			# Features.append(Dict_Feat)
		# if config['save'] == 'ON':
			# save_pickle('MAIN_max_bursts_freq_.pkl', Main_max_Freq_Dict)
	
	
	
	
	elif config['mode'] == 'burst_detection_features_multi_intervals_3':
		# print('Select signals...')	
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()

		
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		#gearbox  LAST
		superdict = {'max':[32., 250], 'crest':[1., 14.], 'fas':[1., 5.], 'rise':[2., 200.], 'count':[2., 50.], 'rms':[4., 60.], 'freq':[50., 450.], 'p2p':[32., 500.], 'area':[500., 25000.], 'dura':[50., 700.]}
		
		# superdict = {'max':[0., 2500], 'crest':[0., 14.], 'fas':[0., 5.], 'rise':[0., 2000.], 'count':[0., 120.], 'rms':[0., 250.], 'freq':[50., 400.], 'p2p':[0., 5000.], 'area':[0., 500000.], 'dura':[0., 30000.]}

		Lmax_intervals, Lmax_Freq_Dict, Lmax_str_intervals = Lgenerate_n_intervals(superdict['max'][0], superdict['max'][1], 20)
		
			
		
		Lcrest_intervals , Lcrest_Freq_Dict, Lcrest_str_intervals = Lgenerate_n_intervals(superdict['crest'][0], superdict['crest'][1], 20)
		Lfas_intervals, Lfas_Freq_Dict, Lfas_str_intervals = Lgenerate_n_intervals(superdict['fas'][0], superdict['fas'][1], 20)
		Lrise_intervals, Lrise_Freq_Dict, Lrise_str_intervals = Lgenerate_n_intervals(superdict['rise'][0], superdict['rise'][1], 20)
		Lcount_intervals , Lcount_Freq_Dict, Lcount_str_intervals = Lgenerate_n_intervals(superdict['count'][0], superdict['count'][1], 20)
		
		Lrms_intervals, Lrms_Freq_Dict, Lrms_str_intervals = Lgenerate_n_intervals(superdict['rms'][0], superdict['rms'][1], 20)
		Lfreq_intervals, Lfreq_Freq_Dict, Lfreq_str_intervals = Lgenerate_n_intervals(superdict['freq'][0], superdict['freq'][1], 20)
		Lp2p_intervals, Lp2p_Freq_Dict, Lp2p_str_intervals = Lgenerate_n_intervals(superdict['p2p'][0], superdict['p2p'][1], 20)
		Larea_intervals, Larea_Freq_Dict, Larea_str_intervals = Lgenerate_n_intervals(superdict['area'][0], superdict['area'][1], 20)
		Ldura_intervals, Ldura_Freq_Dict, Ldura_str_intervals = Lgenerate_n_intervals(superdict['dura'][0], superdict['dura'][1], 20)
		
		Lmax_Freq_Dict['all'] = []		
		Lcrest_Freq_Dict['all'] = []
		Lfas_Freq_Dict['all'] = []
		Lrise_Freq_Dict['all'] = []
		Lfreq_Freq_Dict['all'] = []
		Lp2p_Freq_Dict['all'] = []
		Larea_Freq_Dict['all'] = []
		Ldura_Freq_Dict['all'] = []
		Lrms_Freq_Dict['all'] = []
		Lcount_Freq_Dict['all'] = []
		
		# if config['save'] == 'ON':
			# save_pickle('config_bursts.pkl', config)
		
		
		for filepath in Filepaths:
			max_intervals, max_Freq_Dict, max_str_intervals = generate_n_intervals(superdict['max'][0], superdict['max'][1], 20)
			crest_intervals , crest_Freq_Dict, crest_str_intervals = generate_n_intervals(superdict['crest'][0], superdict['crest'][1], 20)
			fas_intervals, fas_Freq_Dict, fas_str_intervals = generate_n_intervals(superdict['fas'][0], superdict['fas'][1], 20)
			rise_intervals, rise_Freq_Dict, rise_str_intervals = generate_n_intervals(superdict['rise'][0], superdict['rise'][1], 20)
			count_intervals , count_Freq_Dict, count_str_intervals = generate_n_intervals(superdict['count'][0], superdict['count'][1], 20)
			
			rms_intervals, rms_Freq_Dict, rms_str_intervals = generate_n_intervals(superdict['rms'][0], superdict['rms'][1], 20)
			freq_intervals, freq_Freq_Dict, freq_str_intervals = generate_n_intervals(superdict['freq'][0], superdict['freq'][1], 20)
			p2p_intervals, p2p_Freq_Dict, p2p_str_intervals = generate_n_intervals(superdict['p2p'][0], superdict['p2p'][1], 20)
			area_intervals, area_Freq_Dict, area_str_intervals = generate_n_intervals(superdict['area'][0], superdict['area'][1], 20)
			dura_intervals, dura_Freq_Dict, dura_str_intervals = generate_n_intervals(superdict['dura'][0], superdict['dura'][1], 20)
			
			
			
			
			
			
			
			
			Main_Features = []
			filename = os.path.basename(filepath)
			signal = load_signal(filepath, channel=config['channel'])
			
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				title = str(config['channel']) + ' ' + 'Bursts' + ' ' + filename
				title = title.replace('_', '-')
				ax_0.set_title(title, fontsize=11)
				from THR_Burst_Detection import plot_burst_rev
				t = [i/config['fs'] for i in range(len(signal))]
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
				plt.show()
			

			
			# for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				# dict_feat = single_burst_features(signal, t_ini, t_fin, config)
				# Main_Features.append(dict_feat)
				
			# if config['save'] == 'ON':
				# save_pickle('MAIN_bursts_' + filename[:-4] + 'pkl', Main_Features)
				
			

			for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				
				dict_feat = single_burst_features(signal, t_ini, t_fin, config)
				Main_Features.append(dict_feat)
				max_Freq_Dict = acum_in_dict20('max', dict_feat, max_intervals, max_Freq_Dict, max_str_intervals)
				
				crest_Freq_Dict = acum_in_dict20('crest', dict_feat, crest_intervals, crest_Freq_Dict, crest_str_intervals)
				fas_Freq_Dict = acum_in_dict20('fas', dict_feat, fas_intervals, fas_Freq_Dict, fas_str_intervals)
				rise_Freq_Dict = acum_in_dict20('rise', dict_feat, rise_intervals, rise_Freq_Dict, rise_str_intervals)
				count_Freq_Dict = acum_in_dict20('count', dict_feat, count_intervals, count_Freq_Dict, count_str_intervals)
				
				rms_Freq_Dict = acum_in_dict20('rms', dict_feat, rms_intervals, rms_Freq_Dict, rms_str_intervals)
				freq_Freq_Dict = acum_in_dict20('freq', dict_feat, freq_intervals, freq_Freq_Dict, freq_str_intervals)
				p2p_Freq_Dict = acum_in_dict20('p2p', dict_feat, p2p_intervals, p2p_Freq_Dict, p2p_str_intervals)
				area_Freq_Dict = acum_in_dict20('area', dict_feat, area_intervals, area_Freq_Dict, area_str_intervals)
				dura_Freq_Dict = acum_in_dict20('dura', dict_feat, dura_intervals, dura_Freq_Dict, dura_str_intervals)
				
			# print(max_Freq_Dict)

			max_Freq_Dict['all'] = sum_values_dict(max_Freq_Dict)
			crest_Freq_Dict['all'] = sum_values_dict(crest_Freq_Dict)
			fas_Freq_Dict['all'] = sum_values_dict(fas_Freq_Dict)
			rise_Freq_Dict['all'] = sum_values_dict(rise_Freq_Dict)
			freq_Freq_Dict['all'] = sum_values_dict(freq_Freq_Dict)
			p2p_Freq_Dict['all'] = sum_values_dict(p2p_Freq_Dict)
			area_Freq_Dict['all'] = sum_values_dict(area_Freq_Dict)
			dura_Freq_Dict['all'] = sum_values_dict(dura_Freq_Dict)
			rms_Freq_Dict['all'] = sum_values_dict(rms_Freq_Dict)
			count_Freq_Dict['all'] = sum_values_dict(count_Freq_Dict)
			
			for key, value in max_Freq_Dict.items():
				max_Freq_Dict[key] = [value]
			for key, value in crest_Freq_Dict.items():
				crest_Freq_Dict[key] = [value]				
			for key, value in fas_Freq_Dict.items():
				fas_Freq_Dict[key] = [value]
			for key, value in rise_Freq_Dict.items():
				rise_Freq_Dict[key] = [value]
			for key, value in count_Freq_Dict.items():
				count_Freq_Dict[key] = [value]
			
			for key, value in rms_Freq_Dict.items():
				rms_Freq_Dict[key] = [value]
			for key, value in freq_Freq_Dict.items():
				freq_Freq_Dict[key] = [value]
			for key, value in p2p_Freq_Dict.items():
				p2p_Freq_Dict[key] = [value]
			for key, value in area_Freq_Dict.items():
				area_Freq_Dict[key] = [value]
			for key, value in dura_Freq_Dict.items():
				dura_Freq_Dict[key] = [value]
			
			# print(Lmax_Freq_Dict)
			# print(max_Freq_Dict)
			Lmax_Freq_Dict = {k: Lmax_Freq_Dict.get(k, 0) + max_Freq_Dict.get(k, 0) for k in set(Lmax_Freq_Dict) & set(max_Freq_Dict)}
			# print(set(Lmax_Freq_Dict) & set(max_Freq_Dict))
			# print(Lmax_Freq_Dict)
			# a = input('pauseee')
			Lcrest_Freq_Dict = {k: Lcrest_Freq_Dict.get(k, 0) + crest_Freq_Dict.get(k, 0) for k in set(Lcrest_Freq_Dict) & set(crest_Freq_Dict)}
			
			Lfas_Freq_Dict = {k: Lfas_Freq_Dict.get(k, 0) + fas_Freq_Dict.get(k, 0) for k in set(Lfas_Freq_Dict) & set(fas_Freq_Dict)}
			
			Lrise_Freq_Dict = {k: Lrise_Freq_Dict.get(k, 0) + rise_Freq_Dict.get(k, 0) for k in set(Lrise_Freq_Dict) & set(rise_Freq_Dict)}
			
			Lcount_Freq_Dict = {k: Lcount_Freq_Dict.get(k, 0) + count_Freq_Dict.get(k, 0) for k in set(Lcount_Freq_Dict) & set(count_Freq_Dict)}
			
			Lrms_Freq_Dict = {k: Lrms_Freq_Dict.get(k, 0) + rms_Freq_Dict.get(k, 0) for k in set(Lrms_Freq_Dict) & set(rms_Freq_Dict)}
			
			Lfreq_Freq_Dict = {k: Lfreq_Freq_Dict.get(k, 0) + freq_Freq_Dict.get(k, 0) for k in set(Lfreq_Freq_Dict) & set(freq_Freq_Dict)}

			Lp2p_Freq_Dict = {k: Lp2p_Freq_Dict.get(k, 0) + p2p_Freq_Dict.get(k, 0) for k in set(Lp2p_Freq_Dict) & set(p2p_Freq_Dict)}
			
			Larea_Freq_Dict = {k: Larea_Freq_Dict.get(k, 0) + area_Freq_Dict.get(k, 0) for k in set(Larea_Freq_Dict) & set(area_Freq_Dict)}
			
			Ldura_Freq_Dict = {k: Ldura_Freq_Dict.get(k, 0) + dura_Freq_Dict.get(k, 0) for k in set(Ldura_Freq_Dict) & set(dura_Freq_Dict)}
			
			

		Lmax_Freq_Dict = add_sum_dict(Lmax_Freq_Dict)
		Lcrest_Freq_Dict = add_sum_dict(Lcrest_Freq_Dict)
		Lfas_Freq_Dict = add_sum_dict(Lfas_Freq_Dict)
		Lrise_Freq_Dict = add_sum_dict(Lrise_Freq_Dict)
		Lfreq_Freq_Dict = add_sum_dict(Lfreq_Freq_Dict)
		Lp2p_Freq_Dict = add_sum_dict(Lp2p_Freq_Dict)
		Larea_Freq_Dict = add_sum_dict(Larea_Freq_Dict)
		Ldura_Freq_Dict = add_sum_dict(Ldura_Freq_Dict)
		Lrms_Freq_Dict = add_sum_dict(Lrms_Freq_Dict)
		Lcount_Freq_Dict = add_sum_dict(Lcount_Freq_Dict)
		Filenames.append('Sum')
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			
			DataFr_max = pd.DataFrame(data=Lmax_Freq_Dict, index=Filenames)		
			DataFr_max.to_excel(writer, sheet_name='max')		
			
			DataFr_crest = pd.DataFrame(data=Lcrest_Freq_Dict, index=Filenames)
			DataFr_crest.to_excel(writer, sheet_name='crest')
			
			DataFr_fas = pd.DataFrame(data=Lfas_Freq_Dict, index=Filenames)
			DataFr_fas.to_excel(writer, sheet_name='fas')

			DataFr_rise = pd.DataFrame(data=Lrise_Freq_Dict, index=Filenames)
			DataFr_rise.to_excel(writer, sheet_name='rise')
			
			DataFr_count = pd.DataFrame(data=Lcount_Freq_Dict, index=Filenames)
			DataFr_count.to_excel(writer, sheet_name='count')
			
			
			
			DataFr_rms = pd.DataFrame(data=Lrms_Freq_Dict, index=Filenames)
			DataFr_rms.to_excel(writer, sheet_name='rms')		
			
			DataFr_freq = pd.DataFrame(data=Lfreq_Freq_Dict, index=Filenames)
			DataFr_freq.to_excel(writer, sheet_name='freq')
			
			DataFr_p2p = pd.DataFrame(data=Lp2p_Freq_Dict, index=Filenames)
			DataFr_p2p.to_excel(writer, sheet_name='p2p')
			
			DataFr_area = pd.DataFrame(data=Larea_Freq_Dict, index=Filenames)
			DataFr_area.to_excel(writer, sheet_name='area')
			
			DataFr_dura = pd.DataFrame(data=Ldura_Freq_Dict, index=Filenames)
			DataFr_dura.to_excel(writer, sheet_name='dura')

			
			# writer = pd.ExcelWriter(config['name'] + '.xlsx')
			
			# DataFr_max = pd.DataFrame(data=Lmax_Freq_Dict, index=Filenames)		
			# DataFr_max.to_excel(writer, sheet_name='max')	
			writer.close()
	
	
	elif config['mode'] == 'burst_detection_per_tacho':
		# print('Select signals...')	
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()

		
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			print('Select AE Signals')
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
			
			root = Tk()
			root.withdraw()
			root.update()
			print('Select Tacho Signals')
			Filepaths_Tacho = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
			Filepaths_Tacho = [join(config['mypath2'], f) for f in listdir(config['mypath2']) if isfile(join(config['mypath2'], f)) if f[-3:] == 'pkl']
		

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		
		Teeth = list(np.arange(72))
		mydict = {}
		for element in Teeth:
			mydict[str(element)] = []
		
		
		
		

		for filepath, filepath_tacho, filename in zip(Filepaths, Filepaths_Tacho, Filenames):
			print(os.path.basename(filepath))
			print(os.path.basename(filepath_tacho))
			signal = load_signal(filepath, channel=config['channel'])*1000/70.8
			# signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			
			tacho = load_signal(filepath_tacho, channel=None)
			# plt.plot(tacho)
			# plt.show()
			Times_Tacho = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] == 1]
			
			# print(Times_Tacho)
			# a = input('pause')
			
			Rev_Signals = []
			for i in range(len(Times_Tacho)-1):
				Rev_Signals.append(signal[int( np.rint(Times_Tacho[i]*config['fs']) ) : int( np.rint(Times_Tacho[i+1]*config['fs']) )])
			
			Teeth_Burst = {}
			for element in Teeth:
				Teeth_Burst[str(element)] = 0
			for rev_signal in Rev_Signals:
				time_length = len(rev_signal)/config['fs']
				
				t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(rev_signal, config, count=0)
				# full_thr_burst_detector_stella_lockout2
				
				if config['plot'] == 'ON':
					fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
					title = str(config['channel']) + ' ' + 'Bursts' + ' ' + filename
					title = title.replace('_', '-')
					ax_0.set_title(title, fontsize=11)
					from THR_Burst_Detection import plot_burst_rev
					t = [i/config['fs'] for i in range(len(rev_signal))]
					plot_burst_rev(fig_0, ax_0, 0, t, rev_signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
					plt.show()
				
				
				for t_ini in t_burst_corr:				
					tooth_burst = int(np.rint(t_ini*71/time_length))					
					Teeth_Burst[str(tooth_burst)] += 1
					
			for key in mydict:
				# print(Teeth_Burst[key])
				# a = input('pause')
				mydict[key].append(Teeth_Burst[key])
					
					
					# dict_feat = single_burst_features(signal, t_ini, t_fin, config)
					# Main_Features.append(dict_feat)
					# max_Freq_Dict = acum_in_dict20('max', dict_feat, max_intervals, max_Freq_Dict, max_str_intervals)
					
					
				
			# print(max_Freq_Dict)

			
			

		# Lmax_Freq_Dict = add_sum_dict(Lmax_Freq_Dict)
		# Lcrest_Freq_Dict = add_sum_dict(Lcrest_Freq_Dict)
		# Lfas_Freq_Dict = add_sum_dict(Lfas_Freq_Dict)
		# Lrise_Freq_Dict = add_sum_dict(Lrise_Freq_Dict)
		# Lfreq_Freq_Dict = add_sum_dict(Lfreq_Freq_Dict)
		# Lp2p_Freq_Dict = add_sum_dict(Lp2p_Freq_Dict)
		# Larea_Freq_Dict = add_sum_dict(Larea_Freq_Dict)
		# Ldura_Freq_Dict = add_sum_dict(Ldura_Freq_Dict)
		# Lrms_Freq_Dict = add_sum_dict(Lrms_Freq_Dict)
		# Lcount_Freq_Dict = add_sum_dict(Lcount_Freq_Dict)
		
		
		# Filenames.append('Sum')
		mydict_mod = {}
		count = 0
		letters = ['aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal', 'aam', 'aan', 'aao', 'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay', 'aaz', 'aba', 'abb', 'abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'abi', 'abj', 'abk', 'abl', 'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acb', 'acc', 'acd', 'ace', 'acf', 'acg', 'ach', 'aci', 'acj', 'ack', 'acl', 'acm', 'acn', 'aco', 'acp', 'acq', 'acr', 'acs', 'act']
		for key, values in mydict.items():
			mydict_mod[letters[count] + '_T' + key] = values
			count += 1
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			
			# DataFr = pd.DataFrame(data=mydict)	
			DataFr = pd.DataFrame(data=mydict_mod, index=Filenames)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')		
			
			

			
			
			writer.close()
	
	
	elif config['mode'] == 'burst_features_per_tacho':
		# print('Select signals...')	
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()

		
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			print('Select AE Signals')
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
			
			root = Tk()
			root.withdraw()
			root.update()
			print('Select Tacho Signals')
			Filepaths_Tacho = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			
			Filepaths_Tacho = [join((config['mypath'] + '\\Tacho'), f) for f in listdir(config['mypath'] + '\\Tacho') if isfile(join((config['mypath'] + '\\Tacho'), f)) if f[1] == 'u' if f[-4:] == '.pkl']

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		Filenames_Tacho = [os.path.basename(filepath) for filepath in Filepaths_Tacho]
		
		# for name, name_tacho in zip(Filenames, Filenames_Tacho):
			# if name[3:15] != name_tacho[9:21]:
				# print(name[3:16])
				# print(name_tacho[9:24])
				# print('error names tacho and signal 89147')
				# sys.exit()
		
		
		
		Teeth = list(np.arange(72))
		mydict_burst = {}
		mydict_energy = {}
		mydict_freq = {}
		mydict_max = {}
		mydict_crest = {}
		mydict_kurt = {}
		mydict_count = {}
		mydict_dura = {}
		mydict_rise = {}
		mydict_area = {}
		mydict_fas = {}
		mydict_rms = {}
		mydict_p2p = {}
		
		mydict_summax = {}
		mydict_sumcount = {}
		mydict_sumarea = {}
		mydict_sump2p = {}
		
		for element in Teeth:
			mydict_burst[str(element)] = []
			mydict_energy[str(element)] = []
			mydict_freq[str(element)] = []
			mydict_max[str(element)] = []
			mydict_crest[str(element)] = []
			mydict_kurt[str(element)] = []
			mydict_count[str(element)] = []
			mydict_dura[str(element)] = []
			mydict_rise[str(element)] = []
			mydict_area[str(element)] = []
			mydict_fas[str(element)] = []
			mydict_rms[str(element)] = []
			mydict_p2p[str(element)] = []
			
			mydict_summax[str(element)] = []
			mydict_sumcount[str(element)] = []
			mydict_sumarea[str(element)] = []
			mydict_sump2p[str(element)] = []
		
		
		
		

		for filepath, filepath_tacho, filename in zip(Filepaths, Filepaths_Tacho, Filenames):
		
			signal = load_signal(filepath, channel=config['channel'])
			signal = 1000.*butter_bandpass(x=signal, fs=config['fs'], freqs=[95.e3, 140.e3], order=3)
			# signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			
			tacho = load_signal(filepath_tacho, channel=None)
			# plt.plot(tacho)
			# plt.show()
			Times_Tacho = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] == 1]
			
			# print(Times_Tacho)
			# a = input('pause')
			
			Rev_Signals = []
			for i in range(len(Times_Tacho)-1):
				Rev_Signals.append(signal[int( np.rint(Times_Tacho[i]*config['fs']) ) : int( np.rint(Times_Tacho[i+1]*config['fs']) )])
			
			Teeth_Burst = {}
			Teeth_Energy = {}
			Teeth_Freq = {}
			Teeth_Max = {}
			Teeth_Crest = {}
			Teeth_Kurt = {}
			Teeth_Count = {}
			Teeth_Dura = {}
			Teeth_Rise = {}
			Teeth_Area = {}
			Teeth_Fas = {}
			Teeth_Rms = {}
			Teeth_P2p = {}
			
			Teeth_SumMax = {}
			Teeth_SumCount = {}
			Teeth_SumArea = {}
			Teeth_SumP2p = {}
			# contador = 0.
			for element in Teeth:
				Teeth_Burst[str(element)] = 0
				Teeth_Energy[str(element)] = 0
				Teeth_Freq[str(element)] = 0
				Teeth_Max[str(element)] = 0
				Teeth_Crest[str(element)] = 0
				Teeth_Kurt[str(element)] = 0
				Teeth_Count[str(element)] = 0
				Teeth_Dura[str(element)] = 0
				Teeth_Rise[str(element)] = 0
				Teeth_Area[str(element)] = 0
				Teeth_Fas[str(element)] = 0
				Teeth_Rms[str(element)] = 0
				Teeth_P2p[str(element)] = 0
				
				Teeth_SumMax[str(element)] = 0
				Teeth_SumCount[str(element)] = 0
				Teeth_SumArea[str(element)] = 0
				Teeth_SumP2p[str(element)] = 0
				
			for rev_signal in Rev_Signals:
				time_length = len(rev_signal)/config['fs']
				
				t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(rev_signal, config, count=0)

				
				if config['plot'] == 'ON':
					fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
					title = str(config['channel']) + ' ' + 'Bursts' + ' ' + filename
					title = title.replace('_', '-')
					ax_0.set_title(title, fontsize=11)
					from THR_Burst_Detection import plot_burst_rev
					t = [i/config['fs'] for i in range(len(rev_signal))]
					plot_burst_rev(fig_0, ax_0, 0, t, rev_signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
					plt.show()
				
				
				for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):				
					tooth_burst = int(np.rint(t_ini*71/time_length))
					
					
					dict_feat = single_burst_features2(rev_signal, t_ini, t_fin, config)		

					# print(dict_feat['freq'])
					# print(dict_feat['max'])
					# a = input('pause')
					Teeth_Burst[str(tooth_burst)] += 1			
					Teeth_Energy[str(tooth_burst)] += dict_feat['rms']
					
					Teeth_Freq[str(tooth_burst)] += dict_feat['freq']
					Teeth_Max[str(tooth_burst)] += dict_feat['amax']
					Teeth_Crest[str(tooth_burst)] += dict_feat['crest']
					Teeth_Kurt[str(tooth_burst)] += dict_feat['kurt']
					Teeth_Count[str(tooth_burst)] += dict_feat['count']
					Teeth_Dura[str(tooth_burst)] += dict_feat['dura']
					Teeth_Rise[str(tooth_burst)] += dict_feat['rise']
					Teeth_Area[str(tooth_burst)] += dict_feat['area']
					Teeth_Fas[str(tooth_burst)] += dict_feat['fas']
					Teeth_Rms[str(tooth_burst)] += dict_feat['rms']
					Teeth_P2p[str(tooth_burst)] += dict_feat['p2p']
					
					Teeth_SumMax[str(tooth_burst)] += dict_feat['amax']
					Teeth_SumCount[str(tooth_burst)] += dict_feat['count']
					Teeth_SumArea[str(tooth_burst)] += dict_feat['area']
					Teeth_SumP2p[str(tooth_burst)] += dict_feat['p2p']
					# contador += 1
					
			for key in mydict_burst:
				# print(Teeth_Burst[key])
				# a = input('pause')
				# print(contador)
				# a = input('pause')
				num = Teeth_Burst[key]
				mydict_burst[key].append(num)
				mydict_energy[key].append(Teeth_Energy[key])
				
				mydict_summax[key].append(Teeth_SumMax[key])
				mydict_sumcount[key].append(Teeth_SumCount[key])
				mydict_sumarea[key].append(Teeth_SumArea[key])
				mydict_sump2p[key].append(Teeth_SumP2p[key])
				
				if num != 0:
					# mydict_freq[key].append(Teeth_Freq[key]/num)
					# mydict_max[key].append(Teeth_Max[key]/num)
					# mydict_crest[key].append(Teeth_Crest[key]/num)
					# mydict_kurt[key].append(Teeth_Kurt[key]/num)
					# mydict_count[key].append(Teeth_Count[key]/num)
					# mydict_dura[key].append(Teeth_Dura[key]/num)
					# mydict_rise[key].append(Teeth_Rise[key]/num)
					# mydict_area[key].append(Teeth_Area[key]/num)
					# mydict_fas[key].append(Teeth_Fas[key]/num)
					# mydict_rms[key].append(Teeth_Rms[key]/num)
					# mydict_p2p[key].append(Teeth_P2p[key]/num)
					
					mydict_freq[key].append(Teeth_Freq[key])
					mydict_max[key].append(Teeth_Max[key])
					mydict_crest[key].append(Teeth_Crest[key])
					mydict_kurt[key].append(Teeth_Kurt[key])
					mydict_count[key].append(Teeth_Count[key])
					mydict_dura[key].append(Teeth_Dura[key])
					mydict_rise[key].append(Teeth_Rise[key])
					mydict_area[key].append(Teeth_Area[key])
					mydict_fas[key].append(Teeth_Fas[key])
					mydict_rms[key].append(Teeth_Rms[key])
					mydict_p2p[key].append(Teeth_P2p[key])
				else:
					mydict_freq[key].append('')
					mydict_max[key].append('')
					mydict_crest[key].append('')
					mydict_kurt[key].append('')
					mydict_count[key].append('')
					mydict_dura[key].append('')
					mydict_rise[key].append('')
					mydict_area[key].append('')
					mydict_fas[key].append('')
					mydict_rms[key].append('')
					mydict_p2p[key].append('')
					
					


			
			


		
		
		# Filenames.append('Sum')
		mydict_mod_burst = {}
		mydict_mod_energy = {}
		mydict_mod_freq = {}
		mydict_mod_max = {}
		mydict_mod_crest = {}
		mydict_mod_kurt = {}
		mydict_mod_count = {}
		mydict_mod_dura = {}
		mydict_mod_rise = {}
		mydict_mod_area = {}
		mydict_mod_fas = {}
		mydict_mod_rms = {}
		mydict_mod_p2p = {}
		
		mydict_mod_summax = {}
		mydict_mod_sumcount = {}
		mydict_mod_sumarea = {}
		mydict_mod_sump2p = {}
		
		letters = ['aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal', 'aam', 'aan', 'aao', 'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay', 'aaz', 'aba', 'abb', 'abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'abi', 'abj', 'abk', 'abl', 'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acb', 'acc', 'acd', 'ace', 'acf', 'acg', 'ach', 'aci', 'acj', 'ack', 'acl', 'acm', 'acn', 'aco', 'acp', 'acq', 'acr', 'acs', 'act']
		count = 0
		for key, values in mydict_burst.items():
			mydict_mod_burst[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_energy.items():
			mydict_mod_energy[letters[count] + '_T' + key] = values
			count += 1
			
		count = 0
		for key, values in mydict_freq.items():
			mydict_mod_freq[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_max.items():
			mydict_mod_max[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_crest.items():
			mydict_mod_crest[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_kurt.items():
			mydict_mod_kurt[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_count.items():
			mydict_mod_count[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_dura.items():
			mydict_mod_dura[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_rise.items():
			mydict_mod_rise[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_area.items():
			mydict_mod_area[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_fas.items():
			mydict_mod_fas[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_rms.items():
			mydict_mod_rms[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_p2p.items():
			mydict_mod_p2p[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_summax.items():
			mydict_mod_summax[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_sumcount.items():
			mydict_mod_sumcount[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_sumarea.items():
			mydict_mod_sumarea[letters[count] + '_T' + key] = values
			count += 1
		
		count = 0
		for key, values in mydict_sump2p.items():
			mydict_mod_sump2p[letters[count] + '_T' + key] = values
			count += 1
		
		
		mydict_mod_burst = add_sum_dict(mydict_mod_burst)
		mydict_mod_energy = add_sum_dict(mydict_mod_energy)
		mydict_mod_summax = add_sum_dict(mydict_mod_summax)
		mydict_mod_sumcount = add_sum_dict(mydict_mod_sumcount)
		mydict_mod_sumarea = add_sum_dict(mydict_mod_sumarea)
		mydict_mod_sump2p = add_sum_dict(mydict_mod_sump2p)
		
		mydict_mod_freq = add_mean_dict(mydict_mod_freq)
		mydict_mod_max = add_mean_dict(mydict_mod_max)
		mydict_mod_crest = add_mean_dict(mydict_mod_crest)
		mydict_mod_kurt = add_mean_dict(mydict_mod_kurt)
		
		mydict_mod_count = add_mean_dict(mydict_mod_count)
		mydict_mod_dura = add_mean_dict(mydict_mod_dura)
		mydict_mod_rise = add_mean_dict(mydict_mod_rise)
		mydict_mod_area = add_mean_dict(mydict_mod_area)
		mydict_mod_fas = add_mean_dict(mydict_mod_fas)
		mydict_mod_rms = add_mean_dict(mydict_mod_rms)
		mydict_mod_p2p = add_mean_dict(mydict_mod_p2p)
		

		Filenames_Sum = Filenames + ['Sum']
		# Filenames_Sum.append('Sum')
		
		Filenames_Mean = Filenames + ['Mean']
		# Filenames_Mean.append('Mean')
		print(Filenames_Sum)
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '_TeethBurst' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_burst, index=Filenames_Sum)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name'] + '_SumRms' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_energy, index=Filenames_Sum)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			
			
			writer = pd.ExcelWriter(config['name']+ '_SumMax' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_summax, index=Filenames_Sum)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ '_SumCount' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_sumcount, index=Filenames_Sum)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ '_SumArea' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_sumarea, index=Filenames_Sum)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ '_SumP2p' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_sump2p, index=Filenames_Sum)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			
			
			type = '_Sum'
			
			
			
			
			writer = pd.ExcelWriter(config['name']+ type + 'Freq' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_freq, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Max' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_max, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Crest' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_crest, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Kurt' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_kurt, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Count' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_count, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
		
			writer = pd.ExcelWriter(config['name']+ type + 'Dura' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_dura, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Rise' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_rise, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Area' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_area, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Fas' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_fas, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'Rms' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_rms, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			
			writer = pd.ExcelWriter(config['name']+ type + 'P2p' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod_p2p, index=Filenames_Mean)					
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')			
			writer.close()
			

	
	
	elif config['mode'] == 'obtain_burst_features':
		# 
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()

		
		if config['mypath'] == None:
			print('Select signals...')	
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		
		Main_Features = []
		for filepath, filename in zip(Filepaths, Filenames):
			

			
			signal = load_signal(filepath, channel=config['channel'])
			
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				title = str(config['channel']) + ' ' + 'Bursts' + ' ' + filename
				title = title.replace('_', '-')
				ax_0.set_title(title, fontsize=11)
				from THR_Burst_Detection import plot_burst_rev
				t = [i/config['fs'] for i in range(len(signal))]
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
				plt.show()
			
			

			for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				
				dict_feat = single_burst_features2(signal, t_ini, t_fin, config)
				Main_Features.append(dict_feat)
				

			
		
		
		if config['save'] == 'ON':
			save_pickle('Main_Features_' + config['channel'] + '_' + config['name'] + '.pkl', Main_Features)
	
	
	elif config['mode'] == 'obtain_burst_spectrum':

		
		if config['mypath'] == None:
			print('Select signals...')	
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		
		Main_Features = []
		
		for filepath, filename in zip(Filepaths, Filenames):
			count = 0
			
			signal = load_signal(filepath, channel=config['channel'])
			
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				title = str(config['channel']) + ' ' + 'Bursts' + ' ' + filename
				title = title.replace('_', '-')
				ax_0.set_title(title, fontsize=11)
				from THR_Burst_Detection import plot_burst_rev
				t = [i/config['fs'] for i in range(len(signal))]
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
				plt.show()
			
			

			for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				
				fig, ax = plt.subplots()
				burst_signal = signal[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']/10*config['fs'])]
				# plt.plot(burst_signal)
				# plt.show()
				max_signal = np.max(np.absolute(burst_signal))
				
				# magX, f, df = mag_fft(burst_signal, config['fs'])
				# kurt_spectrum = scipy.stats.kurtosis(magX, fisher=False)
				# std_spectrum = np.std(magX)
				
				
				
				
				print(filename)
				f, magX = scipy.signal.periodogram(signal, config['fs'], return_onesided=True, scaling='density')				
				f = f[250000:]
				magX = magX[250000:]
				kurt_spectrum = scipy.stats.kurtosis(magX, fisher=False)
				std_spectrum = np.std(magX)
				
				print('KURT_XXX: ', max_signal/kurt_spectrum)
				print('STD_XXX: ', 0.001*max_signal/std_spectrum)
				print('\n')
				
				ax.plot(f/1000., magX, 'green')
				ax.set_xlabel('Frequenz [kHz]', fontsize=12)
				ax.set_ylabel('Amplitude [mV]', fontsize=12)
				title = str(count) + '_' + filename
				ax.set_title(title, fontsize=11)				
				ax.tick_params(axis='both', labelsize=11)
				ax.set_xlim(left=0., right=500.)
				ax.set_ylim(bottom=0., top=None)
				
				plt.savefig('PSD_' + title + '.png')
				# plt.show()
				# dict_feat = single_burst_features2(signal, t_ini, t_fin, config)
				# Main_Features.append(dict_feat)
				

			
				count += 1
		
		if config['save'] == 'ON':
			save_pickle('Main_Features_' + config['channel'] + '_' + config['name'] + '.pkl', Main_Features)
			

	
	elif config['mode'] == 'plot_main_features':


		if config['mypath'] == None:
			print('Select signals...')	
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		main_dura = []
		main_crest = []
		main_count = []
		main_rise = []		
		main_rms = []
		main_freq = []
		main_amax = []
		main_kurt = []		
		
		
		for filepath, filename in zip(Filepaths, Filenames):
			print(filename)
			dura = []
			crest = []
			count = []
			rise = []		
			rms = []
			freq = []
			amax = []
			kurt = []
			
			Main_Features_It = read_pickle(filepath)
			
			for element in Main_Features_It:
				dura.append(element['dura']/1000.)
				crest.append(element['crest'])
				count.append(element['count']/1000.)
				rise.append(element['rise']/1000.)
				
				rms.append(element['rms'])
				freq.append(element['freq'])
				amax.append(element['amax'])
				kurt.append(element['kurt'])
			
			main_dura.append(dura)
			main_crest.append(crest)
			main_count.append(count)
			main_rise.append(rise)	
			main_rms.append(rms)
			main_freq.append(freq)
			main_amax.append(amax)
			main_kurt.append(kurt)
		
		mylabel = ['37-L', '106-L', '130-LoR', '37-LoR', '121-LoR']
		# mylabel = ['Z106-L']
		fig, ax = plt.subplots(nrows=2, ncols=4)
		
		caja = ax[0][0].boxplot(main_rms)
		ax[0][0].set_title('RMS [mV]')
		ax[0][0].set_xticklabels(mylabel)
		
		caja = ax[0][1].boxplot(main_dura)
		ax[0][1].set_title('Dauer [ms]')
		ax[0][1].set_xticklabels(mylabel)
		
		caja = ax[0][2].boxplot(main_crest)
		ax[0][2].set_title('Crest [-]')
		ax[0][2].set_xticklabels(mylabel)
		
		caja = ax[0][3].boxplot(main_count)
		ax[0][3].set_title('Count [k]')
		ax[0][3].set_xticklabels(mylabel)
		
		
		
		caja = ax[1][0].boxplot(main_kurt)
		ax[1][0].set_title('Kurtosis [-]')
		ax[1][0].set_xticklabels(mylabel)
		
		caja = ax[1][1].boxplot(main_rise)
		ax[1][1].set_title('Anstiegszeit [ms]')
		ax[1][1].set_xticklabels(mylabel)
		
		caja = ax[1][2].boxplot(main_freq)
		ax[1][2].set_title('Hauptfrequenz [kHz]')
		ax[1][2].set_xticklabels(mylabel)
		
		caja = ax[1][3].boxplot(main_amax)
		ax[1][3].set_title('Maximum [mV]')
		ax[1][3].set_xticklabels(mylabel)
		
		
		
		# print(rms)
		plt.show()
		
		

		
	
	
	elif config['mode'] == 'burst_detection_features_multi_intervals_3_10bin':
		# print('Select signals...')	
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()

		
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']

		Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		##gearbox
		superdict = {'max':[0., 250], 'crest':[0., 14.], 'fas':[0., 5.], 'rise':[0., 150.], 'count':[0., 60.], 'rms':[0., 70.], 'freq':[20., 500.], 'p2p':[0., 500.], 'area':[0., 25000.], 'dura':[0., 800.]}
		
		# superdict = {'max':[0., 2500], 'crest':[0., 14.], 'fas':[0., 5.], 'rise':[0., 2000.], 'count':[0., 120.], 'rms':[0., 250.], 'freq':[50., 400.], 'p2p':[0., 5000.], 'area':[0., 500000.], 'dura':[0., 30000.]}

		Lmax_intervals, Lmax_Freq_Dict, Lmax_str_intervals = Lgenerate_10_intervals(superdict['max'][0], superdict['max'][1])
		
			
		
		Lcrest_intervals , Lcrest_Freq_Dict, Lcrest_str_intervals = Lgenerate_10_intervals(superdict['crest'][0], superdict['crest'][1])
		Lfas_intervals, Lfas_Freq_Dict, Lfas_str_intervals = Lgenerate_10_intervals(superdict['fas'][0], superdict['fas'][1])
		Lrise_intervals, Lrise_Freq_Dict, Lrise_str_intervals = Lgenerate_10_intervals(superdict['rise'][0], superdict['rise'][1])
		Lcount_intervals , Lcount_Freq_Dict, Lcount_str_intervals = Lgenerate_10_intervals(superdict['count'][0], superdict['count'][1])
		
		Lrms_intervals, Lrms_Freq_Dict, Lrms_str_intervals = Lgenerate_10_intervals(superdict['rms'][0], superdict['rms'][1])
		Lfreq_intervals, Lfreq_Freq_Dict, Lfreq_str_intervals = Lgenerate_10_intervals(superdict['freq'][0], superdict['freq'][1])
		Lp2p_intervals, Lp2p_Freq_Dict, Lp2p_str_intervals = Lgenerate_10_intervals(superdict['p2p'][0], superdict['p2p'][1])
		Larea_intervals, Larea_Freq_Dict, Larea_str_intervals = Lgenerate_10_intervals(superdict['area'][0], superdict['area'][1])
		Ldura_intervals, Ldura_Freq_Dict, Ldura_str_intervals = Lgenerate_10_intervals(superdict['dura'][0], superdict['dura'][1])
		
		Lmax_Freq_Dict['all'] = []		
		Lcrest_Freq_Dict['all'] = []
		Lfas_Freq_Dict['all'] = []
		Lrise_Freq_Dict['all'] = []
		Lfreq_Freq_Dict['all'] = []
		Lp2p_Freq_Dict['all'] = []
		Larea_Freq_Dict['all'] = []
		Ldura_Freq_Dict['all'] = []
		Lrms_Freq_Dict['all'] = []
		Lcount_Freq_Dict['all'] = []
		
		# if config['save'] == 'ON':
			# save_pickle('config_bursts.pkl', config)
		
		
		for filepath in Filepaths:
			max_intervals, max_Freq_Dict, max_str_intervals = generate_10_intervals(superdict['max'][0], superdict['max'][1])
			crest_intervals , crest_Freq_Dict, crest_str_intervals = generate_10_intervals(superdict['crest'][0], superdict['crest'][1])
			fas_intervals, fas_Freq_Dict, fas_str_intervals = generate_10_intervals(superdict['fas'][0], superdict['fas'][1])
			rise_intervals, rise_Freq_Dict, rise_str_intervals = generate_10_intervals(superdict['rise'][0], superdict['rise'][1])
			count_intervals , count_Freq_Dict, count_str_intervals = generate_10_intervals(superdict['count'][0], superdict['count'][1])
			
			rms_intervals, rms_Freq_Dict, rms_str_intervals = generate_10_intervals(superdict['rms'][0], superdict['rms'][1])
			freq_intervals, freq_Freq_Dict, freq_str_intervals = generate_10_intervals(superdict['freq'][0], superdict['freq'][1])
			p2p_intervals, p2p_Freq_Dict, p2p_str_intervals = generate_10_intervals(superdict['p2p'][0], superdict['p2p'][1])
			area_intervals, area_Freq_Dict, area_str_intervals = generate_10_intervals(superdict['area'][0], superdict['area'][1])
			dura_intervals, dura_Freq_Dict, dura_str_intervals = generate_10_intervals(superdict['dura'][0], superdict['dura'][1])
			
			
			
			
			
			
			
			
			Main_Features = []
			filename = os.path.basename(filepath)
			signal = load_signal(filepath, channel=config['channel'])
			
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			if config['plot'] == 'ON':
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				title = str(config['channel']) + ' ' + 'Bursts' + ' ' + filename
				title = title.replace('_', '-')
				ax_0.set_title(title, fontsize=11)
				from THR_Burst_Detection import plot_burst_rev
				t = [i/config['fs'] for i in range(len(signal))]
				plot_burst_rev(fig_0, ax_0, 0, t, signal, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
				plt.show()
			

			
			# for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				# dict_feat = single_burst_features(signal, t_ini, t_fin, config)
				# Main_Features.append(dict_feat)
				
			# if config['save'] == 'ON':
				# save_pickle('MAIN_bursts_' + filename[:-4] + 'pkl', Main_Features)
				
			

			for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
				
				dict_feat = single_burst_features(signal, t_ini, t_fin, config)
				Main_Features.append(dict_feat)
				max_Freq_Dict = acum_in_dict10('max', dict_feat, max_intervals, max_Freq_Dict, max_str_intervals)
				
				crest_Freq_Dict = acum_in_dict10('crest', dict_feat, crest_intervals, crest_Freq_Dict, crest_str_intervals)
				fas_Freq_Dict = acum_in_dict10('fas', dict_feat, fas_intervals, fas_Freq_Dict, fas_str_intervals)
				rise_Freq_Dict = acum_in_dict10('rise', dict_feat, rise_intervals, rise_Freq_Dict, rise_str_intervals)
				count_Freq_Dict = acum_in_dict10('count', dict_feat, count_intervals, count_Freq_Dict, count_str_intervals)
				
				rms_Freq_Dict = acum_in_dict10('rms', dict_feat, rms_intervals, rms_Freq_Dict, rms_str_intervals)
				freq_Freq_Dict = acum_in_dict10('freq', dict_feat, freq_intervals, freq_Freq_Dict, freq_str_intervals)
				p2p_Freq_Dict = acum_in_dict10('p2p', dict_feat, p2p_intervals, p2p_Freq_Dict, p2p_str_intervals)
				area_Freq_Dict = acum_in_dict10('area', dict_feat, area_intervals, area_Freq_Dict, area_str_intervals)
				dura_Freq_Dict = acum_in_dict10('dura', dict_feat, dura_intervals, dura_Freq_Dict, dura_str_intervals)
				
			# print(max_Freq_Dict)

			max_Freq_Dict['all'] = sum_values_dict(max_Freq_Dict)
			crest_Freq_Dict['all'] = sum_values_dict(crest_Freq_Dict)
			fas_Freq_Dict['all'] = sum_values_dict(fas_Freq_Dict)
			rise_Freq_Dict['all'] = sum_values_dict(rise_Freq_Dict)
			freq_Freq_Dict['all'] = sum_values_dict(freq_Freq_Dict)
			p2p_Freq_Dict['all'] = sum_values_dict(p2p_Freq_Dict)
			area_Freq_Dict['all'] = sum_values_dict(area_Freq_Dict)
			dura_Freq_Dict['all'] = sum_values_dict(dura_Freq_Dict)
			rms_Freq_Dict['all'] = sum_values_dict(rms_Freq_Dict)
			count_Freq_Dict['all'] = sum_values_dict(count_Freq_Dict)
			
			for key, value in max_Freq_Dict.items():
				max_Freq_Dict[key] = [value]
			for key, value in crest_Freq_Dict.items():
				crest_Freq_Dict[key] = [value]				
			for key, value in fas_Freq_Dict.items():
				fas_Freq_Dict[key] = [value]
			for key, value in rise_Freq_Dict.items():
				rise_Freq_Dict[key] = [value]
			for key, value in count_Freq_Dict.items():
				count_Freq_Dict[key] = [value]
			
			for key, value in rms_Freq_Dict.items():
				rms_Freq_Dict[key] = [value]
			for key, value in freq_Freq_Dict.items():
				freq_Freq_Dict[key] = [value]
			for key, value in p2p_Freq_Dict.items():
				p2p_Freq_Dict[key] = [value]
			for key, value in area_Freq_Dict.items():
				area_Freq_Dict[key] = [value]
			for key, value in dura_Freq_Dict.items():
				dura_Freq_Dict[key] = [value]
			
			# print(Lmax_Freq_Dict)
			# print(max_Freq_Dict)
			Lmax_Freq_Dict = {k: Lmax_Freq_Dict.get(k, 0) + max_Freq_Dict.get(k, 0) for k in set(Lmax_Freq_Dict) & set(max_Freq_Dict)}
			# print(set(Lmax_Freq_Dict) & set(max_Freq_Dict))
			# print(Lmax_Freq_Dict)
			# a = input('pauseee')
			Lcrest_Freq_Dict = {k: Lcrest_Freq_Dict.get(k, 0) + crest_Freq_Dict.get(k, 0) for k in set(Lcrest_Freq_Dict) & set(crest_Freq_Dict)}
			
			Lfas_Freq_Dict = {k: Lfas_Freq_Dict.get(k, 0) + fas_Freq_Dict.get(k, 0) for k in set(Lfas_Freq_Dict) & set(fas_Freq_Dict)}
			
			Lrise_Freq_Dict = {k: Lrise_Freq_Dict.get(k, 0) + rise_Freq_Dict.get(k, 0) for k in set(Lrise_Freq_Dict) & set(rise_Freq_Dict)}
			
			Lcount_Freq_Dict = {k: Lcount_Freq_Dict.get(k, 0) + count_Freq_Dict.get(k, 0) for k in set(Lcount_Freq_Dict) & set(count_Freq_Dict)}
			
			Lrms_Freq_Dict = {k: Lrms_Freq_Dict.get(k, 0) + rms_Freq_Dict.get(k, 0) for k in set(Lrms_Freq_Dict) & set(rms_Freq_Dict)}
			
			Lfreq_Freq_Dict = {k: Lfreq_Freq_Dict.get(k, 0) + freq_Freq_Dict.get(k, 0) for k in set(Lfreq_Freq_Dict) & set(freq_Freq_Dict)}

			Lp2p_Freq_Dict = {k: Lp2p_Freq_Dict.get(k, 0) + p2p_Freq_Dict.get(k, 0) for k in set(Lp2p_Freq_Dict) & set(p2p_Freq_Dict)}
			
			Larea_Freq_Dict = {k: Larea_Freq_Dict.get(k, 0) + area_Freq_Dict.get(k, 0) for k in set(Larea_Freq_Dict) & set(area_Freq_Dict)}
			
			Ldura_Freq_Dict = {k: Ldura_Freq_Dict.get(k, 0) + dura_Freq_Dict.get(k, 0) for k in set(Ldura_Freq_Dict) & set(dura_Freq_Dict)}
			
			

		Lmax_Freq_Dict = add_sum_dict(Lmax_Freq_Dict)
		Lcrest_Freq_Dict = add_sum_dict(Lcrest_Freq_Dict)
		Lfas_Freq_Dict = add_sum_dict(Lfas_Freq_Dict)
		Lrise_Freq_Dict = add_sum_dict(Lrise_Freq_Dict)
		Lfreq_Freq_Dict = add_sum_dict(Lfreq_Freq_Dict)
		Lp2p_Freq_Dict = add_sum_dict(Lp2p_Freq_Dict)
		Larea_Freq_Dict = add_sum_dict(Larea_Freq_Dict)
		Ldura_Freq_Dict = add_sum_dict(Ldura_Freq_Dict)
		Lrms_Freq_Dict = add_sum_dict(Lrms_Freq_Dict)
		Lcount_Freq_Dict = add_sum_dict(Lcount_Freq_Dict)
		Filenames.append('Sum')
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			
			DataFr_max = pd.DataFrame(data=Lmax_Freq_Dict, index=Filenames)		
			DataFr_max.to_excel(writer, sheet_name='max')		
			
			DataFr_crest = pd.DataFrame(data=Lcrest_Freq_Dict, index=Filenames)
			DataFr_crest.to_excel(writer, sheet_name='crest')
			
			DataFr_fas = pd.DataFrame(data=Lfas_Freq_Dict, index=Filenames)
			DataFr_fas.to_excel(writer, sheet_name='fas')

			DataFr_rise = pd.DataFrame(data=Lrise_Freq_Dict, index=Filenames)
			DataFr_rise.to_excel(writer, sheet_name='rise')
			
			DataFr_count = pd.DataFrame(data=Lcount_Freq_Dict, index=Filenames)
			DataFr_count.to_excel(writer, sheet_name='count')
			
			
			
			DataFr_rms = pd.DataFrame(data=Lrms_Freq_Dict, index=Filenames)
			DataFr_rms.to_excel(writer, sheet_name='rms')		
			
			DataFr_freq = pd.DataFrame(data=Lfreq_Freq_Dict, index=Filenames)
			DataFr_freq.to_excel(writer, sheet_name='freq')
			
			DataFr_p2p = pd.DataFrame(data=Lp2p_Freq_Dict, index=Filenames)
			DataFr_p2p.to_excel(writer, sheet_name='p2p')
			
			DataFr_area = pd.DataFrame(data=Larea_Freq_Dict, index=Filenames)
			DataFr_area.to_excel(writer, sheet_name='area')
			
			DataFr_dura = pd.DataFrame(data=Ldura_Freq_Dict, index=Filenames)
			DataFr_dura.to_excel(writer, sheet_name='dura')

			
			
			writer.close()
	
	
	elif config['mode'] == 'arrival_times':
		
		print('Select signals')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]


		
		mydict = {}
		mydict[config['channel']] = []
		# mydict['AE_1'] = []
		rownames = []


		
		count = 0
		myrows = {}
		for filepath, filename in zip(Filepaths, Filenames):
		
			signal = load_signal(filepath, channel=config['channel'])				
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)

			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout2(signal, config, count=0)
			
			perro = 0
			if len(t_burst_corr1) != 0:
				for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
					# mydict[str(perro) + '_' + filename] = []

					value = t_ini1
					
					# mydict[str(perro) + '_' + filename].append(value)
					mydict[config['channel']].append(value)
					rownames.append(str(perro) + '_' + filename)
					
					perro += 1
				count += 1
			else:
				mydict[config['channel']].append(-1)
				rownames.append(str(0) + '_' + filename)
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('Arrival_Times_Thr_' + config['channel'] + config['name'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)		
			
			DataFr_max.to_excel(writer, sheet_name='Arrival_Times_Thr')		

			
			writer.close()
	
	
	elif config['mode'] == 'arrival_times_edge':
		
		print('Select signals')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames = [os.path.basename(filepath) for filepath in Filepaths]


		
		mydict = {}
		mydict[config['channel']] = []
		# mydict['AE_1'] = []
		rownames = []


		
		count = 0
		myrows = {}
		for filepath, filename in zip(Filepaths, Filenames):
		
			signal = load_signal(filepath, channel=config['channel'])				
			signal = amp_factor*butter_highpass(x=signal, fs=config['fs'], freq=config['highpass'], order=3)
		
			signal = butter_demodulation(x=signal, fs=config['fs'], filter = ['lowpass', 2000., 3], prefilter=['highpass', 20.e3, 3], type_rect='absolute_value', dc_value='without_dc')
			signal = diff_signal_eq(signal, 1)
			
			t_ini = (np.argmax(signal[5000:])+5000)/config['fs']	

			# t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout(signal, config, count=0)
			
			value = t_ini
			mydict[config['channel']].append(value)
			rownames.append(str(0) + '_' + filename)
			
			# perro = 0
			# for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
				# # mydict[str(perro) + '_' + filename] = []

				
				
				# # mydict[str(perro) + '_' + filename].append(value)
				
				
				
				# perro += 1
			count += 1
				
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('Arrival_Times_Thr_' + config['channel'] + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)		
			
			DataFr_max.to_excel(writer, sheet_name='Arrival_Times_Thr')		

			
			writer.close()
		
	
	
	
	
	elif config['mode'] == 'burst_multi_correlation':
		print('Select signals TYPE ONE lack...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths1 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames1 = [os.path.basename(filepath) for filepath in Filepaths1]
		
		print('Select signals TYPE TWO unknown...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames2 = [os.path.basename(filepath) for filepath in Filepaths2]
		
		mydict = {}
		mydict = {key:[] for key in Filenames1}
		# if config['save'] == 'ON':
			# save_pickle('config_bursts.pkl', config)
		
		
		for filepath1, filename1 in zip(Filepaths1, Filenames1):
			filename1 = os.path.basename(filepath1)
			signal1 = load_signal(filepath1, channel=config['channel'])				
			signal1 = amp_factor*butter_highpass(x=signal1, fs=config['fs'], freq=config['highpass'], order=3)				
			# t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout(signal1, config, count=0)
			for filepath2 in Filepaths2:

				# Main_Features = []
				filename2 = os.path.basename(filepath2)
				signal2 = load_signal(filepath2, channel=config['channel'])				
				signal2 = amp_factor*butter_highpass(x=signal2, fs=config['fs'], freq=config['highpass'], order=3)				
				# t_burst_corr2, amp_burst_corr2, t_burst_corr_rev2, amp_burst_corr_rev2 = full_thr_burst_detector_stella_lockout(signal2, config, count=0)
				
				value = max_norm_correlation(signal1, signal2)
				
				mydict[filename1].append(value)
				
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('caca5' + '.xlsx')
			
			DataFr_max = pd.DataFrame(data=mydict, index=Filenames2)		
			DataFr_max.to_excel(writer, sheet_name='Max_CrossCorr')		

			
			writer.close()
	
	elif config['mode'] == 'burst_multi_correlation_2':
		print('Select signals TYPE ONE lack...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths1 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames1 = [os.path.basename(filepath) for filepath in Filepaths1]
		
		print('Select signals TYPE TWO unknown...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames2 = [os.path.basename(filepath) for filepath in Filepaths2]
		
		mydict = {}
		mydict = {key:[] for key in Filenames1}
		# if config['save'] == 'ON':
			# save_pickle('config_bursts.pkl', config)
		
		count = 0
		for filepath1, filename1 in zip(Filepaths1, Filenames1):
			print('Calculating... % ', 100*count/len(Filepaths1))
			filename1 = os.path.basename(filepath1)
			signal1 = load_signal(filepath1, channel=config['channel'])				
			signal1 = amp_factor*butter_highpass(x=signal1, fs=config['fs'], freq=config['highpass'], order=3)				
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout(signal1, config, count=0)
			for filepath2 in Filepaths2:

				# Main_Features = []
				filename2 = os.path.basename(filepath2)
				signal2 = load_signal(filepath2, channel=config['channel'])				
				signal2 = amp_factor*butter_highpass(x=signal2, fs=config['fs'], freq=config['highpass'], order=3)				
				t_burst_corr2, amp_burst_corr2, t_burst_corr_rev2, amp_burst_corr_rev2 = full_thr_burst_detector_stella_lockout(signal2, config, count=0)
				
				# burst1 = signal1[int(t_burst_corr1[0]*config['fs']) : int(t_burst_corr_rev1[0]*config['fs'])]
				# burst2 = signal2[int(t_burst_corr2[0]*config['fs']) : int(t_burst_corr_rev2[0]*config['fs'])]
				add_time = 150.e-6
				burst1 = signal1[int(t_burst_corr1[0]*config['fs']) : int((t_burst_corr1[0] + add_time)*config['fs'])]
				burst2 = signal2[int(t_burst_corr2[0]*config['fs']) : int((t_burst_corr2[0] + add_time)*config['fs'])]
				
				
				value = max_norm_correlation(burst1, burst2)
				
				mydict[filename1].append(value)
			count += 1
				
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('caca5' + '.xlsx')
			
			DataFr_max = pd.DataFrame(data=mydict, index=Filenames2)		
			DataFr_max.to_excel(writer, sheet_name='Max_CrossCorr')		

			
			writer.close()
	
	elif config['mode'] == 'burst_multi_correlation_3_edge':
		print('Select signals TYPE ONE lack...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths1 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames1 = [os.path.basename(filepath) for filepath in Filepaths1]
		
		print('Select signals TYPE TWO unknown...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames2 = [os.path.basename(filepath) for filepath in Filepaths2]
		

		
		mydict = {}
		rownames = []

		
		
		# if config['save'] == 'ON':
			# save_pickle('config_bursts.pkl', config)
		
		count = 0
		myrows = {}
		for filepath1, filename1 in zip(Filepaths1, Filenames1):
			print('Calculating... % ', 100*count/len(Filepaths1))
			filename1 = os.path.basename(filepath1)
			signal1 = load_signal(filepath1, channel=config['channel'])				
			signal1 = amp_factor*butter_highpass(x=signal1, fs=config['fs'], freq=config['highpass'], order=3)

			signal1raw = signal1
			signal1 = butter_demodulation(x=signal1, fs=config['fs'], filter = ['lowpass', 2500., 3], prefilter=['highpass', 20.e3, 3], type_rect='absolute_value', dc_value='without_dc')
			signal1 = diff_signal_eq(signal1, 1)

			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout(signal1, config, count=0)
			
			perro = 0
			for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
				mydict[str(perro) + '_' + filename1] = []
				
			
				for filepath2, filename2 in zip(Filepaths2, Filenames2):

					# Main_Features = []
					filename2 = os.path.basename(filepath2)
					signal2 = load_signal(filepath2, channel=config['channel'])
					# signal2 = load_signal(filepath2, channel='AE_1')
					# print('Warning!! Channel AE_1 set for Group 2')
					signal2 = amp_factor*butter_highpass(x=signal2, fs=config['fs'], freq=config['highpass'], order=3)


					signal2raw = signal2
					signal2 = butter_demodulation(x=signal2, fs=config['fs'], filter = ['lowpass', 2500., 3], prefilter=['highpass', 20.e3, 3], type_rect='absolute_value', dc_value='without_dc')
					signal2 = diff_signal_eq(signal2, 1)

					
					t_burst_corr2, amp_burst_corr2, t_burst_corr_rev2, amp_burst_corr_rev2 = full_thr_burst_detector_stella_lockout(signal2, config, count=0)
					
					if filename2 not in myrows.keys():
						myrows[filename2] = 0
					
					gato = 0
					for t_ini2, t_fin2 in zip(t_burst_corr2, t_burst_corr_rev2):
						if count == 0:
							rownames.append(str(gato) + '_' + filename2)					
							gato += 1
						if count == 0:
							myrows[filename2] += 1
						burst1 = signal1raw[int(t_ini1*config['fs']) : int(t_fin1*config['fs'])]
						burst2 = signal2raw[int(t_ini2*config['fs']) : int(t_fin2*config['fs'])]
						
						# add_time = 150.e-6
						# burst1 = signal1raw[int(t_ini1*config['fs']) : int((t_ini1 + add_time)*config['fs'])]
						# burst2 = signal2raw[int(t_ini2*config['fs']) : int((t_ini2 + add_time)*config['fs'])]
					
					
						value = max_norm_correlation(burst1, burst2)
					
						mydict[str(perro) + '_' + filename1].append(value)
						
				perro += 1
			count += 1
				
			
				

				
		
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('Correlation_Edge' + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)		
			
			DataFr_max.to_excel(writer, sheet_name='Max_CrossCorr')		

			
			writer.close()
	
	
	elif config['mode'] == 'burst_multi_correlation_3':
		# print('Warning!! Channel AE_1 set for Group 2')
		print('Select signals TYPE ONE lack...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths1 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames1 = [os.path.basename(filepath) for filepath in Filepaths1]
		
		print('Select signals TYPE TWO unknown...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()			
		root.destroy()		
		Filenames2 = [os.path.basename(filepath) for filepath in Filepaths2]
		

		
		mydict = {}
		rownames = []

		
		
		# if config['save'] == 'ON':
			# save_pickle('config_bursts.pkl', config)
		
		dcount = 0
		myrows = {}
		for filepath1, filename1 in zip(Filepaths1, Filenames1):
			print('Calculating... % ', 100*dcount/len(Filepaths1))
			# filename1 = os.path.basename(filepath1)
			signal1 = load_signal(filepath1, channel=config['channel'])				
			signal1 = amp_factor*butter_highpass(x=signal1, fs=config['fs'], freq=config['highpass'], order=3)				
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = full_thr_burst_detector_stella_lockout(signal1, config, count=0)
			
			perro = 0
			for t_ini1, t_fin1 in zip(t_burst_corr1, t_burst_corr_rev1):			
				mydict[str(perro) + '_' + filename1] = []
				
			
				for filepath2, filename2 in zip(Filepaths2, Filenames2):
					# print('new+++++++++++++++++++')

					# Main_Features = []
					# filename2 = os.path.basename(filepath2)
					signal2 = load_signal(filepath2, channel=config['channel'])
					# signal2 = load_signal(filepath2, channel='AE_1')
					# print('Warning!! Channel AE_1 set for Group 2')
					signal2 = amp_factor*butter_highpass(x=signal2, fs=config['fs'], freq=config['highpass'], order=3)				
					t_burst_corr2, amp_burst_corr2, t_burst_corr_rev2, amp_burst_corr_rev2 = full_thr_burst_detector_stella_lockout(signal2, config, count=0)
					# print(t_burst_corr2)
					# print(filename2)
					# a = input('pause--')
					if filename2 not in myrows.keys():
						myrows[filename2] = 0
					
					gato = 0
					for t_ini2, t_fin2 in zip(t_burst_corr2, t_burst_corr_rev2):
						# print('!!!!!')
						# print(dcount)
						# a = input('pause2--')
						
						if dcount == 0:
							rownames.append(str(gato) + '_' + filename2)					
							gato += 1
						if dcount == 0:
							myrows[filename2] += 1
						# burst1 = signal1[int(t_ini1*config['fs']) : int(t_fin1*config['fs'])]
						# burst2 = signal2[int(t_ini2*config['fs']) : int(t_fin2*config['fs'])]
						add_time = 150.e-6
						burst1 = signal1[int(t_ini1*config['fs']) : int((t_ini1 + add_time)*config['fs'])]
						burst2 = signal2[int(t_ini2*config['fs']) : int((t_ini2 + add_time)*config['fs'])]
					
					
						value = max_norm_correlation(burst1, burst2)
					
						mydict[str(perro) + '_' + filename1].append(value)
						
				perro += 1
				dcount += 1
				
			
				

				
		print(rownames)
		print(myrows)
		if config['save'] == 'ON':
			writer = pd.ExcelWriter('CrossCorr_' + config['channel'] + '_' + config['name'] + '_150ms' + '.xlsx')
			# print(myrows)
			# print(rownames)
			# DataFr_max = pd.DataFrame(data=mydict)	
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)		
			
			DataFr_max.to_excel(writer, sheet_name='Max_CrossCorr')		

			
			writer.close()
		

	
	elif config['mode'] == 'read_from_pkl_hist':
		print('Select pkl hists...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		Filenames = []
		for filepath in Filepaths:
			element = os.path.basename(filepath)
			element.replace('MAIN_bursts_', 'hist_')
			Filenames.append(element)
		
		superdict = {'max':[0., 250], 'crest':[0., 14.], 'fas':[0., 3.], 'rise':[0., 150.], 'count':[0., 70.], 'rms':[0., 70.], 'freq':[70., 350.], 'p2p':[0., 500.], 'area':[0., 25000.], 'dura':[0., 800.]}
		
		# for filepath in Filepaths:
			# mylist_ = read_pickle(filepath)
			# if len(mylist) != 0:
				# for mydict in mylist:
				
					
		superdict = {'max':[0., 250], 'crest':[0., 14.], 'fas':[0., 3.], 'rise':[0., 150.], 'count':[0., 70.], 'rms':[0., 70.], 'freq':[70., 350.], 'p2p':[0., 500.], 'area':[0., 25000.], 'dura':[0., 800.]}
		
		for key, value in superdict.items():			
			max_intervals, max_Freq_Dict, max_str_intervals = generate_10_intervals(value[0], value[1])
			# for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
			for filepath in Filepaths:
				dict_feat = read_pickle(filepath)
				# dict_feat = single_burst_features(signal, t_ini, t_fin, config)
				# Main_Features.append(dict_feat)				
				max_Freq_Dict = acum_in_dict(key, dict_feat, max_intervals, max_Freq_Dict, max_str_intervals)
				# Main_max_Freq_Dict = acum_in_dict(key, dict_feat, max_intervals, Main_max_Freq_Dict, max_str_intervals)

				# print(key + '!!!!!!!!!!!!!!!!!!!!!!!!!!')
				# print(max_Freq_Dict)
			if config['save'] == 'ON':
				save_pickle(key + '_bursts_freq_' + filename[:-4] + 'pkl', max_Freq_Dict)
			
		
	
	elif config['mode'] == 'burst_detection_features_one':
		print('Select signal...')		
		dict_int = invoke_signal(config)
		
		x = dict_int['signal']
		x = amp_factor*x
		t = dict_int['time']
		filename_signal = dict_int['filename']
		xraw = x
		
		# t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector(x, config, count=0)
		x = butter_highpass(x=x, fs=config['fs'], freq=config['highpass'], order=3)
		t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_lockout(x, config, count=0)
		
		if len(t_burst_corr) != len(t_burst_corr_rev):
			print('fatal error 1224')
			sys.exit()

		
		fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
		from THR_Burst_Detection import plot_burst_rev
		plot_burst_rev(fig_0, ax_0, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		plt.show()


		# sys.exit()
		
		
		
		
		dura = []
		crest = []
		count = []
		rise = []
		p2p = []
		rms = []
		freq = []
		max = []
		n_id = []
		
		
		kurt = []
		skew = []
		difp2 = []		
		maxdif = []		
		per25 = []
		per50 = []
		per75 = []		
		area = []
		
		fas = []
		narea = []
		nqua = []
		
		# xnew = xraw		
		
		xnew = np.zeros(len(xraw))	
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):		
			signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			signal_complete = xraw[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
			# if len(signal) >= 5:
			for i in range(len(signal_complete)):
				xnew[i + int(t_ini*config['fs'])] = signal_complete[i]


	
		index = 0
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
			signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			signal_complete = xraw[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
			print(len(signal))
			
			# if len(signal) >= 5:
			max_it = np.max(signal)
			dura_it = (t_fin - t_ini)*1000.*1000.
			rms_it = signal_rms(signal)
			rise_it = (np.argmax(signal)/config['fs'])*1000.*1000.
			p2p_it = max_it - np.min(signal)
			crest_it = np.max(np.absolute(signal))/rms_it
			magX_it, f_it, df_it = mag_fft(signal_complete, config['fs'])
			freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.
			# freq_it = (np.argmax(magX_it[1:])/(t_fin - t_ini))/1000.

			
			dura.append(dura_it)				
			rms.append(rms_it)				
			rise.append(rise_it)
			p2p.append(p2p_it)
			crest.append(crest_it)				
			freq.append(freq_it)
			max.append(max_it)
			n_id.append(index)
			
			
			contador = 0
			for u in range(len(signal)-1):			
				if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
					contador = contador + 1
			count.append(contador)
			
			
			
			kurt.append(scipy.stats.kurtosis(signal, fisher=False))
			skew.append(scipy.stats.skew(signal))					
			maxdif.append(np.max(diff_signal(signal, 1)))
			per50.append(np.percentile(np.absolute(signal), 50))
			per75.append(np.percentile(np.absolute(signal), 75))
			per25.append(np.percentile(np.absolute(signal), 25))
			area_it = 1000.*1000*np.sum(np.absolute(signal))/config['fs']
			area.append(area_it)					
			
			nqua.append((np.percentile(np.absolute(signal), 75) - np.percentile(np.absolute(signal), 25))/np.percentile(np.absolute(signal), 50))
			
			index_signal_it = [i for i in range(len(signal))]
			tonto, envolve_up = env_up(index_signal_it, signal)					
			index_triangle_it = [0, int(len(signal)/2), len(signal)]
			triangle_it = [np.max(signal_complete), 0., np.max(signal)]					
			index_signal_it = np.array(index_signal_it)
			index_triangle_it = np.array(index_triangle_it)
			triangle_up_it = np.array(triangle_it)					
			poly2_coef = np.polyfit(index_triangle_it, triangle_it, 2)
			p2 = np.poly1d(poly2_coef)
			poly2 = p2(tonto)					
			difp2.append(np.sum(np.absolute(poly2 - envolve_up)))
			
			fas_it = dura_it/p2p_it
			fas.append(fas_it)
			narea.append(area_it/fas_it)
			
			
			
			index += 1
		

		# fig_1, ax_1 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_1, ax_1, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		# ax_1b = ax_1.twinx()
		# ax_1b.plot(t_burst_corr, rms, '-r', marker='s')
		# ax_1b.set_ylabel('RMS', color='r')
		# ax_1b.tick_params('y', colors='r')
		
		# plt.show()
		from Box_Plot import box_10_plot
		names = ['Rise time (us)', 'Aspect Factor (us/mV)', 'Crest Factor (-)', 'Count (-)', 'RMS (mV)', 'Dura (us)', 'Main Freq. (kHz)', 'Area (us-mV)', 'P2P (mV)', 'Peak (mV)', 'Initial Time (s)']
		features = [rise, fas, crest, count, rms, dura, freq, area, p2p, max]
		box_10_plot('Bursts', names, features)
		
		
		# names = ['Kurtosis', 'Skewness', 'Diff P2', 'Max Dif', 'Per 25', 'Per 50', 'Per 75', 'Area', 'Initial Time (s)']
		# features = [kurt, skew, difp2, maxdif, per25, per50, per75, area]
		# box_8_plot('Bursts', names, features)

	elif config['mode'] == 'find_anormal_features_multi':
		print('Select signals...')	
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		Data = []
		Bursts_Ini = []
		Bursts_End = []
		Features = []
		anormals = []
		for filepath in Filepaths:
			signal = load_signal(filepath, channel=config['channel'])
			signal = 1000*butter_bandpass(x=signal, fs=config['fs'], freqs=[70.e3, 170.e3], order=3)
			Data.append(signal)
			t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector_stella_corr(signal, config, count=0)
			
			Dict_Ini = {'times':t_burst_corr, 'amplitudes':amp_burst_corr}
			Dict_End = {'times':t_burst_corr_rev, 'amplitudes':amp_burst_corr_rev}
			Bursts_Ini.append(Dict_Ini)
			Bursts_End.append(Dict_End)		
			
			Dict_Feat = bursts_features(signal, t_burst_corr, t_burst_corr_rev, config)
			
			fas = Dict_Feat['fas']
			kurt = Dict_Feat['kurt']
			skew = Dict_Feat['skew']
			crest = Dict_Feat['crest']
			
			# fig, ax = plt.subplots(ncols=1, nrows=4)
			# ax[0].plot(fas, 'o')
			# ax[1].plot(kurt, 'o')
			# ax[2].plot(skew, 'o')
			# ax[3].plot(crest, 'o')			
			# ax[0].axhline(y=1.5)
			# ax[1].axhline(y=12)
			# ax[2].axhline(y=2.8)
			# ax[3].axhline(y=4.5)			
			# ax[0].set_ylabel('fas')
			# ax[1].set_ylabel('kurt')
			# ax[2].set_ylabel('skew')
			# ax[3].set_ylabel('crest')			
			# ax[0].set_title(os.path.basename(filepath))			
			# plt.show()
			fas_lim = 1.5
			kurt_lim = 12
			skew_lim = 2.8
			crest_lim = 4.5
			
			for k in range(len(fas)):
				if fas[k] >= fas_lim:
					anormals.append([os.path.basename(filepath), k, 'fas', fas[k], t_burst_corr[k]])
			for k in range(len(kurt)):
				if kurt[k] >= kurt_lim:
					anormals.append([os.path.basename(filepath), k, 'kurt', kurt[k], t_burst_corr[k]])
			for k in range(len(skew)):
				if skew[k] >= skew_lim:
					anormals.append([os.path.basename(filepath), k, 'skew', skew[k], t_burst_corr[k]])
			for k in range(len(crest)):
				if crest[k] >= crest_lim:
					anormals.append([os.path.basename(filepath), k, 'crest', crest[k], t_burst_corr[k]])
			
			
			
			Features.append(Dict_Feat)
		print(anormals)
		save_pickle(config['name'] + '.pkl', anormals)
	
		
	elif config['mode'] == 'model_one_class':
		
		print('Select features for model...')	
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		
		Features = read_pickle(filepath)
		
		rise = []
		fas = []
		count = []
		kurt = []
		freq = []
		skew = []
		nqua = []
		crest = []		
		for feature_dict in Features:
			rise.append(feature_dict['rise'])

			fas.append(feature_dict['fas'])
			count.append(feature_dict['count'])
			kurt.append(feature_dict['kurt'])
			freq.append(feature_dict['freq'])
			skew.append(feature_dict['skew'])
			nqua.append(feature_dict['nqua'])
			crest.append(feature_dict['crest'])
		rise = [item for sublist in rise for item in sublist]
		fas = [item for sublist in fas for item in sublist]
		count = [item for sublist in count for item in sublist]
		kurt = [item for sublist in kurt for item in sublist]
		freq = [item for sublist in freq for item in sublist]
		skew = [item for sublist in skew for item in sublist]
		nqua = [item for sublist in nqua for item in sublist]
		crest = [item for sublist in crest for item in sublist]

		features_to_go = []
		for k in range(len(rise)):
			# features_to_go.append([rise[k], fas[k], count[k], kurt[k], freq[k], skew[k], nqua[k], crest[k]])
			features_to_go.append([rise[k], kurt[k], freq[k], crest[k]])

	
		scaler = StandardScaler()
		# print(features_to_go)
		scaler.fit(features_to_go)
		features_to_go = scaler.transform(features_to_go)	
		
		
		# features_train, features_test, classes_train, classes_test = train_test_split(features, classification, test_size=0.4, random_state=50, stratify=classification)
		
		nu = 0.01
		kernel = 'sigmoid'
		clf = OneClassSVM(nu=nu, kernel=kernel, max_iter=100000)

		clf.fit(features_to_go)


		clf_pickle_info = {}
		clf_pickle_info['nu'] = nu
		clf_pickle_info['kernel'] = nu
		clf_pickle_info['clf'] = clf
		clf_pickle_info['scaler'] = scaler	
		
		if config['save'] == 'ON':
			stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")	
			save_pickle('clf_' + stamp + '.pkl', clf_pickle_info)
			print('++++model saved')
	
	elif config['mode'] == 'test_one_class':
		
		print('Select model...')	
		root = Tk()
		root.withdraw()
		root.update()
		filepath_model = filedialog.askopenfilename()			
		root.destroy()
		
		clf_pickle_info = read_pickle(filepath_model)
		clf = clf_pickle_info['clf']
		scaler = clf_pickle_info['scaler']
		
		print('Select features to test...')	
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		
		Features = read_pickle(filepath)
		
		rise = []
		fas = []
		count = []
		kurt = []
		freq = []
		skew = []
		nqua = []
		crest = []		
		for feature_dict in Features:
			rise.append(feature_dict['rise'])
			fas.append(feature_dict['fas'])
			count.append(feature_dict['count'])
			kurt.append(feature_dict['kurt'])
			freq.append(feature_dict['freq'])
			skew.append(feature_dict['skew'])
			nqua.append(feature_dict['nqua'])
			crest.append(feature_dict['crest'])
		rise = [item for sublist in rise for item in sublist]
		fas = [item for sublist in fas for item in sublist]
		count = [item for sublist in count for item in sublist]
		kurt = [item for sublist in kurt for item in sublist]
		freq = [item for sublist in freq for item in sublist]
		skew = [item for sublist in skew for item in sublist]
		nqua = [item for sublist in nqua for item in sublist]
		crest = [item for sublist in crest for item in sublist]
		features_to_go = []
		for k in range(len(rise)):
			# features_to_go.append([rise[k], fas[k], count[k], kurt[k], freq[k], skew[k], nqua[k], crest[k]])
			features_to_go.append([rise[k], kurt[k], freq[k], crest[k]])

		
		features_to_go = scaler.transform(features_to_go)
		
		predictions = clf.predict(features_to_go)
		count = 0
		for element in predictions:
			if element == 1:
				count += 1
		print('outlyers')
		print(count)
		
		print('inlyers')
		print(len(predictions) - count)
	
	
	elif config['mode'] == 'plot_features':
		

		
		print('Select features...')	
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		
		Features = read_pickle(filepath)
		
		rise = []
		fas = []
		count = []
		kurt = []
		freq = []
		skew = []
		nqua = []
		crest = []		
		for feature_dict in Features:
			rise.append(feature_dict['rise'])
			fas.append(feature_dict['fas'])
			count.append(feature_dict['count'])
			kurt.append(feature_dict['kurt'])
			freq.append(feature_dict['freq'])
			skew.append(feature_dict['skew'])
			nqua.append(feature_dict['nqua'])
			crest.append(feature_dict['crest'])
		rise = [item for sublist in rise for item in sublist]
		fas = [item for sublist in fas for item in sublist]
		count = [item for sublist in count for item in sublist]
		kurt = [item for sublist in kurt for item in sublist]
		freq = [item for sublist in freq for item in sublist]
		skew = [item for sublist in skew for item in sublist]
		nqua = [item for sublist in nqua for item in sublist]
		crest = [item for sublist in crest for item in sublist]
		features_to_go = []
		

		
		from Box_Plot import box_8_plot
		names = ['Rise time', 'Aspect Factor', 'Count', 'Kurtosis', 'Frequency', 'Skewness', 'N-Q', 'Crest Factor', 'Initial Time (s)']
		features = [rise, fas, count, kurt, freq, skew, nqua, crest]
		box_8_plot('Bursts', names, features)
		
		
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
	if config['power2'] != 'auto' and config['power2'] != 'OFF':
		config['power2'] = int(config['power2'])
	# config['mode'] = float(config['fs_tacho'])
	config['fs'] = float(config['fs'])
	config['fs_tacho'] = float(config['fs_tacho'])
	config['n_files'] = int(config['n_files'])
	config['stella'] = int(config['stella'])
	config['thr_value'] = float(config['thr_value'])
	config['highpass'] = float(config['highpass'])
	config['window_time'] = float(config['window_time'])
	config['time_segments'] = float(config['time_segments'])
	config['lockout'] = int(config['lockout'])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config

def max_norm_correlation(signal1, signal2):
	correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')
	return np.max(correlation)

if __name__ == '__main__':
	main(sys.argv)
