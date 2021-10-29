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
from THR_Burst_Detection import full_thr_burst_detector

#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser
from Main_Analysis import invoke_signal
from THR_Burst_Detection import read_threshold as read_threshold
from THR_Burst_Detection import plot_burst_rev as plot_burst_rev
from THR_Burst_Detection import thr_burst_detector
from THR_Burst_Detection import thr_burst_detector_rev
from Box_Plot import box_8_plot
from Box_Plot import box_8_plot_vs
#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'save', 'channel']
InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'n_signals':1, 'plot':'OFF', 'only_first':'OFF', 'file':'OFF'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	
	
	
	if config['mode'] == 'boxplot_features':
		print('Select normal signal..')
		dict_signal_normal = invoke_signal(config)
		
		x_normal = dict_signal_normal['signal']
		t_normal = dict_signal_normal['time']
		filename_signal_normal = dict_signal_normal['filename']
		
		dict_features_normal = calculate_features_2(t_normal, x_normal, config)
		
		print('Select ANORMAL signal!!.')
		dict_signal_anormal = invoke_signal(config)
		
		x_anormal = dict_signal_anormal['signal']
		t_anormal = dict_signal_anormal['time']
		filename_signal_anormal = dict_signal_anormal['filename']
		
		dict_features_anormal = calculate_features_2(t_anormal, x_anormal, config)
		
		
		
		
		fig1, ax1 = plt.subplots()
		ax1.boxplot([dict_features_normal['crest'], dict_features_anormal['crest']])
		ax1.set_xticklabels(['Gearbox-Source', 'WB-Source'], fontsize=11)
		ax1.set_ylabel('Crest Factor (-)', fontsize=11)		
		plt.tight_layout()
		plt.savefig(config['channel'] + '_crest_hp_80_box_thr_3_wt_1000_n_1.png')
		
		fig1d, ax1d = plt.subplots()
		ax1d.boxplot([dict_features_normal['crest'], dict_features_anormal['crest']])
		ax1d.set_xticklabels(['Getriebe-Quelle', 'DB-Quelle'], fontsize=11)
		ax1d.set_ylabel('Crest-Faktor (-)', fontsize=11)		
		plt.tight_layout()
		plt.savefig('DE_' + config['channel'] + '_crest_hp_80_box_thr_3_wt_1000_n_1.png')
		
		
		
		
		
		fig2, ax2 = plt.subplots()
		ax2.boxplot([dict_features_normal['rms'], dict_features_anormal['rms']])
		ax2.set_xticklabels(['Gearbox-Source', 'WB-Source'], fontsize=11)
		ax2.set_ylabel('RMS Value (m$V_{in}$)', fontsize=11)		
		plt.tight_layout()
		plt.savefig(config['channel'] + '_rms_hp_80_box_thr_3_wt_1000_n_1.png')
		
		fig2d, ax2d = plt.subplots()
		ax2d.boxplot([dict_features_normal['rms'], dict_features_anormal['rms']])
		ax2d.set_xticklabels(['Getriebe-Quelle', 'DB-Quelle'], fontsize=11)
		ax2d.set_ylabel('RMS-Wert (m$V_{in}$)', fontsize=11)		
		plt.tight_layout()
		plt.savefig('DE_' + config['channel'] + '_rms_hp_80_box_thr_3_wt_1000_n_1.png')
		
		
		
		
		
		fig3, ax3 = plt.subplots()
		ax3.boxplot([dict_features_normal['count'], dict_features_anormal['count']])
		ax3.set_xticklabels(['Gearbox-Source', 'WB-Source'], fontsize=11)
		ax3.set_ylabel('Cycles Count (-)', fontsize=11)		
		plt.tight_layout()
		plt.savefig(config['channel'] + '_count_hp_80_box_thr_3_wt_1000_n_1.png')
		
		fig3, ax3d = plt.subplots()
		ax3d.boxplot([dict_features_normal['count'], dict_features_anormal['count']])
		ax3d.set_xticklabels(['Getriebe-Quelle', 'DB-Quelle'], fontsize=11)
		ax3d.set_ylabel('Zyklusanzahl (-)', fontsize=11)		
		plt.tight_layout()
		plt.savefig('DE_' + config['channel'] + '_count_hp_80_box_thr_3_wt_1000_n_1.png')

		


	
	elif config['mode'] == 'plot_three_burst':
		
		
		Xs = []
		Ts = []
		Filenames = []
		for i in range(3):
			dict_int = invoke_signal(config)
			x = dict_int['signal']
			t = dict_int['time']
			f = dict_int['freq']
			magX = dict_int['mag_fft']
			filename_x = dict_int['filename']
			
			Xs.append(x)
			Ts.append(t)
			Filenames.append(filename_x)
		
		fig, ax = plt.subplots(nrows=1, ncols=3)
		for i in range(3):
			ax[i].plot(Ts[i], Xs[i])			
			ax[i].set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
			
			title = config['channel'] + ' WFM Burst an F (Wdh. ' + str(i+1) + ')'
			title = title.replace('_', '-')
			
			ax[i].set_title(title, fontsize=12)			
			ax[i].set_xlabel('Dauer [s]', fontsize=12)
			ax[i].tick_params(axis='both', labelsize=11)
			ax[i].ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))
		fig.set_size_inches(16, 4)
		plt.show()
		
	
		
	elif config['mode'] == 'wfm_features':
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		
		dict_signal = invoke_signal(config)
		
		x = dict_signal['signal']
		t = dict_signal['time']
		filename = dict_signal['filename']
		
		dict_features = calculate_features_2(t, x, config)
		
		
		fig1, ax1 = plt.subplots()
		ax1.plot(t, x)
		ax1b = ax1.twinx()		
		ax1.set_ylabel('Amplitude (m$V_{in}$)')
		ax1.set_xlabel('Time (s)')
		ax1.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)		
		ax1b.plot(dict_features['ini_time'], dict_features['crest'], 'ro')		
		ax1b.set_ylabel('Crest Factor (-)', color='r')
		ax1b.tick_params('y', colors='r')
		plt.tight_layout()
		plt.savefig(config['channel'] + '_crest_hp_80_source_brush_thr_3_wt_1000_n_1.png')
		
		
		fig2, ax2 = plt.subplots()
		ax2.plot(t, x)
		ax2b = ax2.twinx()		
		ax2.set_ylabel('Amplitude (m$V_{in}$)')
		ax2.set_xlabel('Zeit (s)')
		ax2.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)		
		ax2b.plot(dict_features['ini_time'], dict_features['crest'], 'ro')		
		ax2b.set_ylabel('Crest-Faktor (-)', color='r')
		ax2b.tick_params('y', colors='r')
		plt.tight_layout()
		plt.savefig('DE_' + config['channel'] + '_crest_hp_80_source_brush_thr_3_wt_1000_n_1.png')
		
		
		
		
		
		
		
		
		
		
		fig3, ax3 = plt.subplots()
		ax3.plot(t, x)
		ax3b = ax3.twinx()		
		ax3.set_ylabel('Amplitude (m$V_{in}$)')
		ax3.set_xlabel('Time (s)')
		ax3.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)		
		ax3b.plot(dict_features['ini_time'], dict_features['rms'], 'ro')		
		ax3b.set_ylabel('RMS Value (m$V_{in}$)', color='r')
		ax3b.tick_params('y', colors='r')
		plt.tight_layout()
		plt.savefig(config['channel'] + '_rms_hp_80_source_brush_thr_3_wt_1000_n_1.png')
		
		
		fig4, ax4 = plt.subplots()
		ax4.plot(t, x)
		ax4b = ax4.twinx()		
		ax4.set_ylabel('Amplitude (m$V_{in}$)')
		ax4.set_xlabel('Zeit (s)')
		ax4.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)		
		ax4b.plot(dict_features['ini_time'], dict_features['rms'], 'ro')		
		ax4b.set_ylabel('RMS Wert (m$V_{in}$)', color='r')
		ax4b.tick_params('y', colors='r')
		plt.tight_layout()
		plt.savefig('DE_' + config['channel'] + '_rms_hp_80_source_brush_thr_3_wt_1000_n_1.png')
		
		
		
		
		
		
		
		
		
		fig5, ax5 = plt.subplots()
		ax5.plot(t, x)
		ax5b = ax5.twinx()		
		ax5.set_ylabel('Amplitude (m$V_{in}$)')
		ax5.set_xlabel('Time (s)')
		ax5.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)		
		ax5b.plot(dict_features['ini_time'], dict_features['count'], 'ro')		
		ax5b.set_ylabel('Cycles Count (-)', color='r')
		ax5b.tick_params('y', colors='r')
		plt.tight_layout()
		plt.savefig(config['channel'] + '_count_hp_80_source_brush_thr_3_wt_1000_n_1.png')
		
		
		fig6, ax6 = plt.subplots()
		ax6.plot(t, x)
		ax6b = ax6.twinx()		
		ax6.set_ylabel('Amplitude (m$V_{in}$)')
		ax6.set_xlabel('Zeit (s)')
		ax6.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)		
		ax6b.plot(dict_features['ini_time'], dict_features['count'], 'ro')		
		ax6b.set_ylabel('Zyklusanzahl (-)', color='r')
		ax6b.tick_params('y', colors='r')
		plt.tight_layout()
		plt.savefig('DE_' + config['channel'] + '_count_hp_80_source_brush_thr_3_wt_1000_n_1.png')
		
	
	elif config['mode'] == 'obtain_features':
		rise = []
		dura = []
		crest = []
		count = []
		maxcorr = []
		rms = []
		freq = []
		max = []

		print('Select reference burst')
		root = Tk()
		root.withdraw()
		root.update()
		filename_burst = filedialog.askopenfilename()
		root.destroy()	
		reference = read_pickle(filename_burst)
		
		for i in range(config['n_signals']):
			print('Select signal..')
			dict_signal = invoke_signal(config)
			
			x = dict_signal['signal']
			t = dict_signal['time']
			filename_signal = dict_signal['filename']
			
			dict_features = calculate_features(t, x, reference, config)
			
			rise.append(dict_features['rise'])
			dura.append(dict_features['dura'])
			crest.append(dict_features['crest'])
			count.append(dict_features['count'])
			maxcorr.append(dict_features['maxcorr'])
			rms.append(dict_features['rms'])
			freq.append(dict_features['freq'])
			max.append(dict_features['max'])
		
		rise = [element for list_in in rise for element in list_in]
		dura = [element for list_in in dura for element in list_in]
		crest = [element for list_in in crest for element in list_in]
		count = [element for list_in in count for element in list_in]
		maxcorr = [element for list_in in maxcorr for element in list_in]
		rms = [element for list_in in rms for element in list_in]
		freq = [element for list_in in freq for element in list_in]
		max = [element for list_in in max for element in list_in]
		
		
		caca = 'AA'
		names = ['Rise time (us)', 'Duration (us)', 'Crest Factor (-)', 'Count (-)', 'Max. Cross-Corr. (-)', 'RMS Value (mV)', 'Main Freq. (kHz)', 'Max. Value (mV)', 'Initial Time (s)']
		features = [rise, dura, crest, count, maxcorr, rms, freq, max]
		box_8_plot(caca, names, features)
		
		dict_out = {'features':features, 'names':names}
		
		save_pickle(config['channel'] + '_features_thr_05_wt_500_HSN_point_E.pkl', dict_out)
		
	elif config['mode'] == 'plot_8_box_vs':
		print('Select features normal pickle')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()		
		dict_normal = read_pickle(filename)
		
		print('Select features Anormal pickle...')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()		
		dict_anormal = read_pickle(filename)
		
		features_normal = dict_normal['features']
		features_anormal = dict_anormal['features']
		
		names = dict_normal['names']
		labels = ['Gearbox', 'HSN']		
		box_8_plot_vs(labels, names, features_normal, features_anormal)
		
		names = ['Steigungszeit [us]', 'Dauer [us]', 'Crest-Faktor [-]', 'Überschwingungen [-]', 'Max. Kreuzkorrelation [-]', 'RMS-Wert [mV]', 'Frequenz [kHz]', 'Max. Wert [mV]', 'Initial Time (s)']
		labels = ['Getriebe', 'HSN']		
		box_8_plot_vs(labels, names, features_normal, features_anormal)
	
	
	else:
		print('unknown mode')
		
		
		
	return

# plt.show()
def calculate_features(t, x, reference_raw, config):
	

	dura = []
	crest = []
	count = []
	rise = []
	maxcorr = []
	rms = []
	max = []
	freq = []
	
	# kurt = []
	# skew = []
	# difp2 = []		
	# maxdif = []		
	# per25 = []
	# per50 = []
	# per75 = []		
	# area = []
	
	times_features = []
	

	# t, x, f, magX, filename_x = invoke_signal(config)				
	# fig0, ax0 = plt.subplots(nrows=1, ncols=2)
	# ax0[0].plot(t, x)
	# ax0[0].set_title('Signal')			
	# ax0[1].plot([i/config['fs'] for i in range(len(reference))], reference)
	# ax0[1].set_title('Reference')
	# plt.show()			
	
	x_raw = x
	
	# config_detector = {'fs':config['fs'], 'thr_value':1.0, 'thr_mode':'fixed_value', 'window_time':500, 'n_files':1, 'save_plot':'OFF'}		
	# config_detector['window_time'] = config_detector['window_time'] / 1000000.
	
	config['thr_value'] = 1.0
	config['thr_mode'] = 'fixed_value'
	config['window_time'] = 500.
	# config['n_files'] = 1
	config['save_plot'] = 'OFF'
	
	config['window_time'] = config['window_time'] / 1000000.

	
	# t_burst_x, amp_burst_x = thr_burst_detector(x, config_detector, count=0)		
	# t_burst_rev_x, amp_burst_rev_x = thr_burst_detector_rev(x, config_detector, count=0)
	
	
	t_burst_x, amp_burst_x, t_burst_rev_x, amp_burst_rev_x = full_thr_burst_detector(x, config, count=0)
	
	if config['only_first'] == 'ON':
		t_burst_x = t_burst_x[0:1]
		amp_burst_x = amp_burst_x[0:1]
		t_burst_rev_x = t_burst_rev_x[0:1]
		amp_burst_rev_x = amp_burst_rev_x[0:1]
	
	
	
	if len(t_burst_x) != len(t_burst_rev_x):
		print('fatal error 1224')
		sys.exit()
	
	
	t_burst_ref, amp_burst_ref = thr_burst_detector(reference_raw, config, count=0)		
	t_burst_rev_ref, amp_burst_rev_ref = thr_burst_detector_rev(reference_raw, config, count=0)
	# t_burst_ref, amp_burst_ref, t_burst_rev_ref, amp_burst_rev_ref = full_thr_burst_detector(reference_raw, config, count=0)
	
	
	if len(t_burst_ref) != len(t_burst_rev_ref):
		print('fatal error 1234')
		sys.exit()
	if len(t_burst_ref) != 1:
		print('fatal error 9234')
		sys.exit()
	
	
	# reference = reference_raw[int(t_burst_ref[0]*config['fs']) : int(t_burst_rev_ref[0]*config['fs'])]
	reference = reference_raw
	
	fig1, ax1 = plt.subplots(nrows=1, ncols=2)
	ax1[0].plot(t, x)
	ax1[0].plot(t_burst_x, amp_burst_x, 'ro')
	ax1[0].plot(t_burst_rev_x, amp_burst_rev_x, 'go')
	ax1[0].set_title('Signal')			
	
	ax1[1].plot([i/config['fs'] for i in range(len(reference_raw))], reference_raw)
	ax1[1].plot(t_burst_ref, amp_burst_ref, 'ro')
	ax1[1].plot(t_burst_rev_ref, amp_burst_rev_ref, 'go')
	ax1[1].set_title('Reference')
	plt.show()


	for t_ini, t_fin in zip(t_burst_x, t_burst_rev_x):
		signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
		signal_complete = x[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]			
		
		if len(signal) >= 10:
			max_it = np.max(signal)
			dura_it = (t_fin - t_ini)*1000.*1000.
			# rms_it = signal_rms(signal)
			rms_it = signal_rms(signal_complete)
			rise_it = (np.argmax(signal)/config['fs'])*1000.*1000.
			# maxcorr_it = max_norm_correlation(signal, reference)
			maxcorr_it = max_norm_correlation(signal_complete, reference)
			crest_it = np.max(np.absolute(signal_complete))/rms_it
			# crest_it = np.max(np.absolute(signal))/rms_it
			magX_it, f_it, df_it = mag_fft(signal_complete, config['fs'])
			freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.
			# magX_it, f_it, df_it = mag_fft(signal, config['fs'])
			# freq_it = (np.argmax(magX_it[1:])/(t_fin - t_ini))/1000.
			
			dura.append(dura_it)				
			rms.append(rms_it)				
			rise.append(rise_it)
			maxcorr.append(maxcorr_it)
			crest.append(crest_it)				
			freq.append(freq_it)
			max.append(np.max(signal))
			
			contador = 0
			for u in range(len(signal)-1):			
				if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
					contador = contador + 1
			count.append(contador)
			
			
			# kurt.append(scipy.stats.kurtosis(signal, fisher=False))
			# skew.append(scipy.stats.skew(signal))					
			# maxdif.append(np.max(diff_signal(signal, 1)))
			# per50.append(np.percentile(np.absolute(signal), 50))
			# per75.append(np.percentile(np.absolute(signal), 75))
			# per25.append(np.percentile(np.absolute(signal), 25))					
			# area.append(1000.*1000*np.sum(np.absolute(signal))/config['fs'])					
			
			
			# index_signal_it = [i for i in range(len(signal))]
			# tonto, envolve_up = env_up(index_signal_it, signal)					
			# index_triangle_it = [0, int(len(signal)/2), len(signal)]
			# triangle_it = [np.max(signal), 0., np.max(signal)]					
			# index_signal_it = np.array(index_signal_it)
			# index_triangle_it = np.array(index_triangle_it)
			# triangle_up_it = np.array(triangle_it)					
			# poly2_coef = np.polyfit(index_triangle_it, triangle_it, 2)
			# p2 = np.poly1d(poly2_coef)
			# poly2 = p2(tonto)					
			# difp2.append(np.sum(np.absolute(poly2 - envolve_up)))
			times_features.append(t_ini)

			
		else:
			print('do nothing: too few points at: ', t_ini)
		


	# from Box_Plot import box_8_plot
	# names = ['Rise time (us)', 'Duration (us)', 'Crest Factor (-)', 'Count (-)', 'Max. Cross-Corr. (-)', 'RMS Value (mV)', 'Main Freq. (kHz)', 'Max. Value (mV)', 'Initial Time (s)']
	# features = [rise, dura, crest, count, maxcorr, rms, freq, max, times_features]
	dict_features = {'rise':rise, 'dura':dura, 'crest':crest, 'count':count, 'maxcorr':maxcorr, 'rms':rms, 'freq':freq, 'max':max, 'ini_time':times_features}


		
	return dict_features
	
def calculate_features_2(t, x, config):

			
	dura = []
	crest = []
	count = []
	rise = []
	rms = []
	max = []
	freq = []
	
	# kurt = []
	# skew = []
	# difp2 = []		
	# maxdif = []		
	# per25 = []
	# per50 = []
	# per75 = []		
	# area = []
	
	times_features = []
	

	for i in range(config['n_signals']):
		print('Select signal')
		# t, x, f, magX, filename_x = invoke_signal(config)				
		# fig0, ax0 = plt.subplots(nrows=1, ncols=2)
		# ax0[0].plot(t, x)
		# ax0[0].set_title('Signal')			
		# ax0[1].plot([i/config['fs'] for i in range(len(reference))], reference)
		# ax0[1].set_title('Reference')
		# plt.show()			
		
		x_raw = x
		
		# config_detector = {'fs':config['fs'], 'thr_value':1.0, 'thr_mode':'fixed_value', 'window_time':500, 'n_files':1, 'save_plot':'OFF'}		
		# config_detector['window_time'] = config_detector['window_time'] / 1000000.
		
		config['thr_value'] = 0.5
		config['thr_mode'] = 'fixed_value'
		config['window_time'] = 500.
		config['n_files'] = 1
		config['save_plot'] = 'OFF'
		
		config['window_time'] = config['window_time'] / 1000000.

		
		# t_burst_x, amp_burst_x = thr_burst_detector(x, config_detector, count=0)		
		# t_burst_rev_x, amp_burst_rev_x = thr_burst_detector_rev(x, config_detector, count=0)
		
		
		t_burst_x, amp_burst_x, t_burst_rev_x, amp_burst_rev_x = full_thr_burst_detector(x, config, count=0)
		
		if config['only_first'] == 'ON':
			t_burst_x = t_burst_x[0:1]
			amp_burst_x = amp_burst_x[0:1]
			t_burst_rev_x = t_burst_rev_x[0:1]
			amp_burst_rev_x = amp_burst_rev_x[0:1]
		
		
		
		if len(t_burst_x) != len(t_burst_rev_x):
			print('fatal error 1224')
			sys.exit()
		
		
		
		
		fig1, ax1 = plt.subplots(nrows=1, ncols=1)
		ax1.plot(t, x)
		ax1.plot(t_burst_x, amp_burst_x, 'ro')
		ax1.plot(t_burst_rev_x, amp_burst_rev_x, 'go')
		ax1.set_title('Signal')			

		plt.show()

	
		for t_ini, t_fin in zip(t_burst_x, t_burst_rev_x):
			signal = x[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			signal_complete = x[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]			
			
			if len(signal) >= 10:
				max_it = np.max(signal)
				dura_it = (t_fin - t_ini)*1000.*1000.
				# rms_it = signal_rms(signal)
				rms_it = signal_rms(signal_complete)
				rise_it = (np.argmax(signal)/config['fs'])*1000.*1000.
				crest_it = np.max(np.absolute(signal_complete))/rms_it
				# crest_it = np.max(np.absolute(signal))/rms_it
				magX_it, f_it, df_it = mag_fft(signal_complete, config['fs'])
				freq_it = (np.argmax(magX_it[1:])/(config['window_time']))/1000.
				# magX_it, f_it, df_it = mag_fft(signal, config['fs'])
				# freq_it = (np.argmax(magX_it[1:])/(t_fin - t_ini))/1000.
				
				dura.append(dura_it)				
				rms.append(rms_it)				
				rise.append(rise_it)
				crest.append(crest_it)				
				freq.append(freq_it)
				max.append(np.max(signal))
				
				contador = 0
				for u in range(len(signal)-1):			
					if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
						contador = contador + 1
				count.append(contador)
				
				
				# kurt.append(scipy.stats.kurtosis(signal, fisher=False))
				# skew.append(scipy.stats.skew(signal))					
				# maxdif.append(np.max(diff_signal(signal, 1)))
				# per50.append(np.percentile(np.absolute(signal), 50))
				# per75.append(np.percentile(np.absolute(signal), 75))
				# per25.append(np.percentile(np.absolute(signal), 25))					
				# area.append(1000.*1000*np.sum(np.absolute(signal))/config['fs'])					
				
				
				# index_signal_it = [i for i in range(len(signal))]
				# tonto, envolve_up = env_up(index_signal_it, signal)					
				# index_triangle_it = [0, int(len(signal)/2), len(signal)]
				# triangle_it = [np.max(signal), 0., np.max(signal)]					
				# index_signal_it = np.array(index_signal_it)
				# index_triangle_it = np.array(index_triangle_it)
				# triangle_up_it = np.array(triangle_it)					
				# poly2_coef = np.polyfit(index_triangle_it, triangle_it, 2)
				# p2 = np.poly1d(poly2_coef)
				# poly2 = p2(tonto)					
				# difp2.append(np.sum(np.absolute(poly2 - envolve_up)))
				times_features.append(t_ini)

				
			else:
				print('do nothing: too few points at: ', t_ini)
			

	
	# from Box_Plot import box_8_plot
	# names = ['Rise time (us)', 'Duration (us)', 'Crest Factor (-)', 'Count (-)', 'Max. Cross-Corr. (-)', 'RMS Value (mV)', 'Main Freq. (kHz)', 'Max. Value (mV)', 'Initial Time (s)']
	# features = [rise, dura, crest, count, maxcorr, rms, freq, max, times_features]
	dict_features = {'rise':rise, 'dura':dura, 'crest':crest, 'count':count, 'rms':rms, 'freq':freq, 'max':max, 'ini_time':times_features}


		
	return dict_features	
	

	
	
	

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
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	config['n_signals'] = int(config['n_signals'])
	# Variable conversion
	return config

def max_norm_correlation(signal1, signal2):
	correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')
	return np.max(correlation)

if __name__ == '__main__':
	main(sys.argv)
