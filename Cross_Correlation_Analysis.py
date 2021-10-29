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




from os.path import join, isdir, basename, dirname, isfile

import os.path
import sys

from os import chdir, listdir
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


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'save', 'channel']
InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'plot':'ON', 'n_files':1, 'title_plot':None, 'file':'OFF', 'mypath':None}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	from Main_Analysis import invoke_signal
	
	if config['mode'] == 'export_window':

		# t, x, f, magX, xraw, filename_signal = invoke_signal(['--channel', 'AE_2', '--fs', str(config['fs']), '--power2', 'OFF', '--plot', 'ON'])
		# t, x, f, magX, filename_signal = invoke_signal(config)
		
		
		# dict_int = invoke_signal(config)
		# xraw = dict_int['signal_raw']
		# x = dict_int['signal']
		# t = dict_int['time']
		# filename_signal = dict_int['filename']
		
		
		
		# plt.close()
		
		print('Select burst')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_burst = filedialog.askopenfilename()
		root.destroy()
		
		x = load_signal_2(filepath_burst, channel='AE_1')
		t = np.arange(len(x))/config['fs']
		
		# t1 = 0.51918
		# t2 = 0.51933
		
		# t1 = 0.42845
		# t2 = 0.42850
		
		# t1 = 0.42840
		# t2 = 0.42865
		
		# t1 = 0.38388
		# t2 = 0.38395
		
		# t1 = 0.51939
		# t2 = 0.51947
		
		# t1 = 0.63658
		# t2 = 0.63665
		
		# t1 = 0.63610
		# t2 = 0.63618
		
		# t1 = 0.45421
		# t2 = 0.45429
		
		# t1 = 0.47436
		# t2 = 0.47444
		
		# t1 = 0.3854
		# t2 = 0.3856
		
		t1 = 1.029
		t2 = 1.059
		
		t1 = 1.03012
		t2 = 1.03012 + 0.0002
		
		x = x[int(t1*config['fs']) : int(t2*config['fs'])]
		t = t[int(t1*config['fs']) : int(t2*config['fs'])]
		
		if config['save'] == 'ON':
			save_pickle('Burst_AE_1_191539_02ms.pkl', x)
		
		plt.plot(t, x)
		plt.show()
		
		# ind_old = [i for i in range(len(x))]
		# ind_new = [10*i for i in range(int(len(x)/10))]
		
		# print(ind_new)
		# print(ind_old)
		# print(len(ind_new))
		# print(len(ind_old))
		
		# x = np.interp(ind_new, ind_old, x)
		# t = np.interp(ind_new, ind_old, t)
		
		# if config['save'] == 'ON':
			# save_pickle('Burst_AE_3_Punkt_F_Draht_1mhz_01.pkl', x)
		
		# plt.figure(1)
		# plt.plot(t, x)
		# plt.title('1')
		# plt.show()
	
	elif config['mode'] == 'multi_bursts_segments':

		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
			# Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		# Filenames = [os.path.basename(filepath) for filepath in Filepaths]
		
		
		for filepath in Filepaths:
			signal = load_signal_2(filepath, channel=config['channel'])
			filename = os.path.basename(filepath)
			if np.max(signal) >= np.absolute(np.min(signal)):
				idx_amax = np.argmax(signal)
			else:
				idx_amax = np.argmin(signal)
			
			
			before_ms = 0.05
			after_ms = 0.15
			
			before_points = int(before_ms/1000.*config['fs'])
			after_points = int(after_ms/1000.*config['fs'])
			
			
			signal = signal[idx_amax-before_points:idx_amax+after_points]
			
			
			
			t = np.array([i/config['fs'] for i in range(len(signal))])
			
			if config['save'] == 'ON':
				save_pickle('Burst_' + config['channel'] + '_' + filename[0:-5] + '.pkl', signal)
			
			# plt.figure(0)
			# plt.plot(t, signal)
			
			# plt.show()
		
	
	elif config['mode'] == 'simple_cross_correlation':
		
		print('Select signal AE')
		root = Tk()
		root.withdraw()
		root.update()
		filepath1 = filedialog.askopenfilename()
		root.destroy()		
		signal1 = load_signal(filepath1, channel='AE_0')
		
		filename = basename(filepath1)
		
		print('Select signal AC')
		root = Tk()
		root.withdraw()
		root.update()
		filepath2 = filedialog.askopenfilename()
		root.destroy()		
		signal2 = load_signal(filepath2, channel='ACC_0')
		
		signal1 = np.absolute(signal1)
		signal2 = np.absolute(signal2)
		
		# signal1 = (signal1)**2.0
		# signal2 = (signal2)**2.0		
		
		signal1 = signal1/np.max(signal1)
		signal2 = signal2/np.max(signal2)
		
		# signal1 = signal1[1:100000]
		# signal2 = signal2[1:100000]
		
		# signal1 = signal1[0:int(len(signal1)/5)]
		# signal2 = signal2[0:int(len(signal2)/5)]
		
		xold = np.linspace(0, 1, len(signal2))
		xnew = np.linspace(0, 1, len(signal1))		
		print(len(signal2))
		signal2 = np.interp(xnew, xold, signal2)
		print(len(signal2))

		
		correlation = xcorr_fft(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5)
		
		# correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')		
		# correlation = correlation.tolist()
		# correlation = np.array(correlation[::-1])
		
		save_pickle('CC_' + filename[3:-5] + '.pkl', correlation)
		
		plt.plot(correlation)
		plt.show()
		
		# print('Select signal CC')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()
		# root.destroy()		
		# signal = load_signal(filepath, channel='xxx')
		# plt.plot(signal, 'r')
		# plt.show()
	
	elif config['mode'] == 'auto_cross_correlation':
		
		if config['mypath'] == None:
			print('Select signals AE')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths_AE = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths_AE = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
		if config['mypath'] == None:
			print('Select signals AC')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths_AC = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths_AC = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'D' if f[-4:] == 'tdms']
		
		
		for filepath1, filepath2 in zip(Filepaths_AE, Filepaths_AC):
			
			signal1 = load_signal(filepath1, channel='AE_0')			
			filename = basename(filepath1)
			signal2 = load_signal(filepath2, channel='AC_0')
			filename2 = basename(filepath2)
			print(filename + '   ' + filename2)
			
			# #SQR
			# signal1_sqr = (signal1)**2.0
			# signal2_sqr = (signal2)**2.0		
			
			# signal1_sqr = signal1_sqr/np.max(signal1_sqr)
			# signal2_sqr = signal2_sqr/np.max(signal2_sqr)
			
			# # signal1_sqr = signal1_sqr/np.std(signal1_sqr)
			# # signal2_sqr = signal2_sqr/np.std(signal2_sqr)
			
			# xold = np.linspace(0, 1, len(signal2_sqr))
			# xnew = np.linspace(0, 1, len(signal1_sqr))		
			# signal2_sqr = np.interp(xnew, xold, signal2_sqr)
			
			# correlation_sqr = xcorr_fft(signal1_sqr/(np.sum(signal1_sqr**2))**0.5, signal2_sqr/(np.sum(signal2_sqr**2))**0.5)			
			
			# mult_sqr = signal1_sqr*signal2_sqr
			
			# save_pickle('CC_SQRnmax_' + filename[3:-5] + '.pkl', correlation_sqr)
			# save_pickle('MM_SQRnmax_' + filename[3:-5] + '.pkl', mult_sqr)
			
			
			#ENV
			signal1_env = hilbert_demodulation(signal1)
			signal2_env = hilbert_demodulation(signal2)
			
			
			
			signal1_env = signal1_env/np.max(signal1_env)
			signal2_env = signal2_env/np.max(signal2_env)
			
			# signal1_env = signal1_env/np.std(signal1_env)
			# signal2_env = signal2_env/np.std(signal2_env)
			
			
			xold = np.linspace(0, 1, len(signal2_env))
			xnew = np.linspace(0, 1, len(signal1_env))		
			signal2_env = np.interp(xnew, xold, signal2_env)
			
			
			
			correlation_env = xcorr_fft(signal1_env/(np.sum(signal1_env**2))**0.5, signal2_env/(np.sum(signal2_env**2))**0.5)
			correlation_env = np.abs(correlation_env)
			
			#mult_env = signal1_env*signal2_env
			
			# plt.plot(signal1_env, 'r')
			# plt.plot(signal2_env, 'b')
			# plt.plot(correlation_env, 'k')
			# plt.show()
			
			save_pickle('CC_ENVnmax_' + filename[3:-5] + '.pkl', correlation_env)
			#save_pickle('MM_ENVnmax_' + filename[3:-5] + '.pkl', mult_env)
			
			# print('+++')
			# print(len(correlation_sqr))
			# print(len(mult_sqr))
			# print(len(correlation_env))
			# print(len(mult_env))
	
	elif config['mode'] == 'auto_cross_correlation_double':
		
		if config['mypath'] == None:
			print('Select signals AE')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths_AE = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths_AE = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'E' if f[-4:] == 'tdms']
		
		if config['mypath'] == None:
			print('Select signals AC')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths_AC = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths_AC = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[1] == 'D' if f[-4:] == 'tdms']
		
		
		for filepath1, filepath2 in zip(Filepaths_AE, Filepaths_AC):
			
			signal1 = load_signal(filepath1, channel='AE_0')			
			filename = basename(filepath1)
			signal2 = load_signal(filepath2, channel='AC_0')
			filename2 = basename(filepath2)
			print(filename + '   ' + filename2)
			
			
			
			#ENV
			signal1_env = hilbert_demodulation(signal1)
			signal2_env = hilbert_demodulation(signal2)
			
			
			
			signal1_env = signal1_env/np.max(signal1_env)
			signal2_env = signal2_env/np.max(signal2_env)
			

			
			
			xold = np.linspace(0, 1, len(signal2_env))
			xnew = np.linspace(0, 1, len(signal1_env))		
			signal2_env = np.interp(xnew, xold, signal2_env)
			
			
			
			correlation_env = xcorr_fft(signal1_env/(np.sum(signal1_env**2))**0.5, signal2_env/(np.sum(signal2_env**2))**0.5)
			correlation_env = np.abs(correlation_env)
			
			
			length = len(correlation_env)
			correlation_env_1 = correlation_env[0:int(length/2)]
			correlation_env_2 = correlation_env[int(length/2) :]
			
			# time = np.arange(length)
			# time_1 = time[0:int(length/2)]
			# time_2 = time[int(length/2) :]
			# plt.plot(time_1, correlation_env_1, color='red', marker='o')
			# plt.plot(time_2, correlation_env_2, color='blue', marker='s')
			# plt.plot(time, correlation_env, color='green')
			# plt.show()
			
			save_pickle('CC_ENVnmax_' + filename[3:-5] + '_1.pkl', correlation_env_1)
			
			save_pickle('CC_ENVnmax_' + filename[3:-5] + '_2.pkl', correlation_env_2)


			


	elif config['mode'] == 'cross_correlation':
		print('Select signal')
		t, x, f, magX, xraw, filename_signal = invoke_signal(['--channel', 'AE_2', '--fs', str(config['fs']), '--power2', 'OFF', '--plot', 'ON'])
		
		print('Select burst')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		burst = read_pickle(filename)
		
		x = np.absolute(x)
		burst = np.absolute(burst)
		
		# x = x/np.max(x)
		# burst = burst/np.max(burst)

		
		correlation = np.correlate(burst/(np.sum(burst**2))**0.5, x/(np.sum(x**2))**0.5, mode='same')
		
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		
		save_pickle('cross_correlation_' + filename_signal[:-3] + '.pkl', correlation)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(t, correlation)
		ax.set_xlabel('Zeit (s)')
		ax.set_ylabel('Amplitude (-)')
		ax.set_title('AE_2' + ' ' + 'Kreuzkorrelation' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		plt.show()
	
	elif config['mode'] == 'simple_cross_correlation':
		print('Select signal')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()

		
		print('Select burst')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_burst = filedialog.askopenfilename()
		root.destroy()	
		
		x = load_signal_2(filepath, channel=config['channel']) / 70.8
		# x = load_signal_2(filepath, channel=config['channel']) / 141.25
		t = np.arange(len(x))/config['fs']
		burst = load_signal_2(filepath_burst, channel='AE-1')
		
		print(len(x))
		print(len(burst))
		
		x = np.absolute(x)
		burst = np.absolute(burst)
		
		x = x / np.max(x)
		burst = burst / np.max(burst)
	
		# x = x - np.mean(x)
		# burst = burst - np.mean(burst)
		
		# x = x / np.max(x)
		# burst = burst / np.max(burst)

		
		# correlation = np.correlate(burst/(np.sum(burst**2))**0.5, x/(np.sum(x**2))**0.5, mode='same')
		# correlation = np.correlate(burst/np.std(burst), x/np.std(x), mode='same')
		correlation = np.correlate(burst, x, mode='same')
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		
		if config['save'] == 'ON':
			save_pickle('cross_correlation_' + config['name'] + '_' + config['channel'] + '.pkl', correlation)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(t, correlation)
		ax.set_xlabel('Time [s]', fontsize=13)
		ax.set_ylabel('Amplitude [-]', fontsize=13)
		# ax.set_title('AE_2' + ' ' + 'Kreuzkorrelation' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		# ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		plt.show()

		
		
	elif config['mode'] == 'double_correlation':
		
		print('Select burst 1')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		
		burst1 = read_pickle(filename)
		
		print('Select burst 2')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		
		burst2 = read_pickle(filename)
		
		
		burst1 = np.absolute(burst2)
		
		burst2 = np.absolute(burst2)
		
		correlation = np.correlate(burst1/(np.sum(burst1**2))**0.5, burst2/(np.sum(burst2**2))**0.5, mode='same')
		
		
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		

		plt.figure(0)
		plt.plot(correlation)
		plt.xlabel('Zeit (s)')
		plt.ylabel('Correlation')
		# plt.title('Cross Correlation')
		plt.show()
		
		
		if config['save'] == 'ON':
			save_pickle('Second_Correlation_01.pkl', correlation)
	
	elif config['mode'] == 'double_cross_correlation':
		print('Select signal')
		t, x, f, magX, xraw, filename_signal = invoke_signal(['--channel', 'AE_3', '--fs', str(config['fs']), '--power2', 'OFF', '--plot', 'ON'])
		
		print('Select burst')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		burst = read_pickle(filename)
		
		x = np.absolute(x)
		burst = np.absolute(burst)

		
		correlation = np.correlate(burst/(np.sum(burst**2))**0.5, x/(np.sum(x**2))**0.5, mode='same')
		
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		

		plt.figure(0)
		plt.plot(t, correlation)
		plt.xlabel('Zeit (s)')
		plt.ylabel('Correlation')
		# plt.title('Cross Correlation')
		plt.show()
		
		
		
		
		print('Select  2 burst')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		burst2 = read_pickle(filename)
		
		x = correlation
		burst = burst2

		
		correlation = np.correlate(burst/(np.sum(burst**2))**0.5, x/(np.sum(x**2))**0.5, mode='same')
		
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		

		plt.figure(0)
		plt.plot(t, correlation)
		plt.xlabel('Zeit (s)')
		plt.ylabel('Correlation 2')
		# plt.title('Cross Correlation')
		plt.show()
		
	elif config['mode'] == 'cross_correlation_feature':
		print('Select signal')
		t, x, f, magX, xraw, filename_signal = invoke_signal(['--channel', 'AE_1', '--fs', str(config['fs']), '--power2', 'OFF', '--plot', 'ON'])
		
		print('Select burst')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		burst = read_pickle(filename)
		
		x = np.absolute(x)
		burst = np.absolute(burst)

		
		correlation = np.correlate(burst/(np.sum(burst**2))**0.5, x/(np.sum(x**2))**0.5, mode='same')
		
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		
		save_pickle('cross_correlation_' + filename_signal[:-3] + '.pkl', correlation)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		# ax.plot(t, correlation)
		ax.set_xlabel('Zeit (s)')
		ax.set_ylabel('Amplitude (-)')
		ax.set_title('AE_1' + ' ' + 'Kreuzkorrelation' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		# plt.show()
		
		
		from THR_Burst_Detection import thr_burst_detector as thr_burst_detector
		from THR_Burst_Detection import read_threshold as read_threshold
		from THR_Burst_Detection import plot_burst_rev as plot_burst_rev
		
		x = correlation
		xraw = correlation
		
		config = {'fs':1.e6, 'thr_value':0.015, 'thr_mode':'fixed_value', 'window_time':0.001, 'n_files':1, 'save_plot':'OFF'}
		
		x, t_burst_corr, amp_burst_corr = thr_burst_detector(xraw, config, count=0)		
		xrev, t_burst_corr_rev, amp_burst_corr_rev = thr_burst_detector(xraw[::-1], config, count=0)
		
		tr = len(t)/config['fs']
		t_burst_corr_rev = t_burst_corr_rev[::-1]
		t_burst_corr_rev = tr*np.ones(len(t_burst_corr_rev)) - np.array(t_burst_corr_rev)
		t_burst_corr_rev = t_burst_corr_rev.tolist()
		amp_burst_corr_rev = amp_burst_corr_rev[::-1]
		
		
		
		
		
		# fig, ax = plt.subplots(nrows=1, ncols=1)			
		plot_burst_rev(fig, ax, 0, t, correlation, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		
		
		
		
		dura = []
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
			dura.append(t_fin - t_ini)
		
		ax2 = ax.twinx()
		ax2.plot(t_burst_corr, dura, '-r')
		ax2.set_ylabel('Dauer', color='r')
		ax2.tick_params('y', colors='r')
		
		plt.show()
		
	
	elif config['mode'] == 'no_correlation_feature':
		print('Select signal')
		# t, x, f, magX, filename_signal = invoke_signal(config)
		
		
		dict_int = invoke_signal(config)
		# xraw = dict_int['signal_raw']
		x = dict_int['signal']
		t = dict_int['time']
		filename_signal = dict_int['filename']
		
		
		
		print('Select reference burst for feature correlation')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		reference = read_pickle(filename)
		
		

		
		
		
		fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
		# ax.plot(t, correlation)
		ax_0.set_xlabel('Time (s)')
		ax_0.set_ylabel('Amplitude (-)')
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax_0.set_ylabel('Amplitude (m$V_{in}$)')
					
					
		ax_0.set_title('AE_3' + ' ' + 'WFM' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		ax_0.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		# plt.show()
		
		
		from THR_Burst_Detection import thr_burst_detector
		from THR_Burst_Detection import thr_burst_detector_rev
		from THR_Burst_Detection import full_thr_burst_detector
		from THR_Burst_Detection import read_threshold as read_threshold
		from THR_Burst_Detection import plot_burst_rev as plot_burst_rev
		
		xraw = x
		
		config['thr_value'] = 1.0
		config['thr_mode'] = 'fixed_value'
		config['window_time'] = 500
		config['n_files'] = 1
		config['save_plot'] = 'OFF'
		
		config['window_time'] = config['window_time'] / 1000000.
		
		
		# x, t_burst_corr, amp_burst_corr = thr_burst_detector(xraw, config, count=0)		
		# xrev, t_burst_corr_rev, amp_burst_corr_rev = thr_burst_detector(xraw[::-1], config, count=0)
		
		# t_burst_corr, amp_burst_corr = thr_burst_detector(x, config, count=0)		
		# t_burst_corr_rev, amp_burst_corr_rev = thr_burst_detector_rev(x, config, count=0)
		
		t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev = full_thr_burst_detector(x, config, count=0)
		
		if len(t_burst_corr) != len(t_burst_corr_rev):
			print('fatal error 1224')
			sys.exit()
		
		
		# tr = len(t)/config['fs']
		# t_burst_corr_rev = t_burst_corr_rev[::-1]
		# t_burst_corr_rev = tr*np.ones(len(t_burst_corr_rev)) - np.array(t_burst_corr_rev)
		# t_burst_corr_rev = t_burst_corr_rev.tolist()
		# amp_burst_corr_rev = amp_burst_corr_rev[::-1]
		
		
		
		plot_burst_rev(fig_0, ax_0, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		
		# # AE-2-A-n25
		# count_lim_down = 3.
		# count_lim_up = 12.
		
		# maxcorr_lim_down = 0.55
		
		# crest_lim_down = 6.0
		# crest_lim_up = 10.0
		
		# dura_lim_up = 100. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.15
		# rms_lim_up = 0.85
		
		# rise_lim_up = 30.
		
		# max_lim_up = 6.
		# max_lim_down = 1.
		
		# freq_lim_down = 250.
		# freq_lim_up = 460.
		
		
		# # AE-2-A
		# count_lim_down = 0.
		# count_lim_up = 17.
		
		# maxcorr_lim_down = 0.5
		
		# crest_lim_down = 4.
		# crest_lim_up = 14.0
		
		# dura_lim_up = 130. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.12
		# rms_lim_up = 3.0
		
		# rise_lim_up = 40.
		
		# max_lim_up = 8.
		# max_lim_down = 0.
		
		# freq_lim_down = 230.
		# freq_lim_up = 460.
		
		
		
		# # # # # # #AE-1-C
		# # # # # # count_lim_down = -1.
		# # # # # # count_lim_up = 12.
		
		# # # # # # maxcorr_lim_down = 0.56		
		
		# # # # # # crest_lim_down = 1.0
		# # # # # # crest_lim_up = 4.0
		
		# # # # # # dura_lim_up = 200. 
		# # # # # # dura_lim_down = 5. 
		
		# # # # # # rms_lim_down = 0.2
		# # # # # # rms_lim_up = 3.5
		
		# # # # # # rise_lim_up = 50.
		
		# # # # # # max_lim_up = 6.
		
		# # # # # # freq_lim_down = 200.
		# # # # # # freq_lim_up = 450.
		
		
		# # # # # # #AE-3-F
		# # # # # # count_lim_down = -1.
		# # # # # # count_lim_up = 12.
		
		# # # # # # maxcorr_lim_down = 0.56		
		
		# # # # # # crest_lim_down = 1.0
		# # # # # # crest_lim_up = 5.1
		
		# # # # # # dura_lim_up = 220. 
		# # # # # # dura_lim_down = 5. 
		
		# # # # # # rms_lim_down = 0.2
		# # # # # # rms_lim_up = 3.5
		
		# # # # # # rise_lim_up = 50.
		
		# # # # # # max_lim_up = 6.
		
		# # # # # # freq_lim_down = 200.
		# # # # # # freq_lim_up = 450.
		
		
		
		# #AE-3-D
		# count_lim_down = 1.
		# count_lim_up = 12.
		
		# maxcorr_lim_down = 0.42
		
		# crest_lim_down = 5.5
		# crest_lim_up = 14.
		
		# dura_lim_up = 30. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.1
		# rms_lim_up = 1.8
		
		# rise_lim_up = 20.
		
		# max_lim_up = 4.0
		
		# freq_lim_down = 280.
		# freq_lim_up = 460.
		
		
		
		# #AE-3-C
		# count_lim_down = 3.
		# count_lim_up = 20.
		
		# maxcorr_lim_down = 0.42
		
		# crest_lim_down = 4.0
		# crest_lim_up = 10.
		
		# dura_lim_up = 120. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.05
		# rms_lim_up = 1.0
		
		# rise_lim_up = 40.
		
		# max_lim_up = 4.0
		
		# freq_lim_down = 250.
		# freq_lim_up = 460.
		
		
		# #AE-3-B
		# count_lim_down = 6.
		# count_lim_up = 22.
		
		# maxcorr_lim_down = 0.4
		
		# crest_lim_down = 4.0
		# crest_lim_up = 10.
		
		# dura_lim_up = 300. 
		# dura_lim_down = 70. 
		
		# rms_lim_down = 0.05
		# rms_lim_up = 1.0
		
		# rise_lim_up = 40.
		
		# max_lim_up = 3.0
		
		# freq_lim_down = 280.
		# freq_lim_up = 420.
		
		
		# #AE-1-C
		# count_lim_down = -1
		# count_lim_up = 20.
		
		# maxcorr_lim_down = 0.5	
		
		# crest_lim_down = 4.0
		# crest_lim_up = 14.
		
		# dura_lim_up = 200. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.15
		# rms_lim_up = 2.7
		
		# rise_lim_up = 30.
		
		# max_lim_up = 7.0
		
		# freq_lim_down = 220.
		# freq_lim_up = 460.
		
		
		# #AE-1-C-n25
		# count_lim_down = -1
		# count_lim_up = 20.
		
		# maxcorr_lim_down = 0.43	
		
		# crest_lim_down = 5.0
		# crest_lim_up = 10.
		
		# dura_lim_up = 250. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.15
		# rms_lim_up = 2.1
		
		# rise_lim_up = 40.
		
		# max_lim_up = 5.0
		# max_lim_down = 1.2
		
		# freq_lim_down = 250.
		# freq_lim_up = 460.
		
		
		#AE-3-F-E
		count_lim_down = 3
		count_lim_up = 20.
		
		maxcorr_lim_down = 0.55
		
		crest_lim_down = 4.0
		crest_lim_up = 14.
		
		dura_lim_up = 230. 
		dura_lim_down = 5. 
		
		rms_lim_down = 0.3
		rms_lim_up = 2.7
		
		rise_lim_up = 40.
		
		max_lim_up = 7.0
		max_lim_down = 0.
		
		freq_lim_down = 220.
		freq_lim_up = 460.
		
		
		# #AE-3-E-n25
		# count_lim_down = 4
		# count_lim_up = 20.
		
		# maxcorr_lim_down = 0.55
		
		# crest_lim_down = 5.0
		# crest_lim_up = 9.
		
		# dura_lim_up = 180. 
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.25
		# rms_lim_up = 0.7
		
		# rise_lim_up = 40.
		
		# max_lim_up = 4.0
		# max_lim_down = 1.5
		
		# freq_lim_down = 220.
		# freq_lim_up = 460.
		
		
		# #AE-3-D-n25
		# count_lim_down = 1.
		# count_lim_up = 10.
		
		# maxcorr_lim_down = 0.31
		
		# crest_lim_down = 5.5
		# crest_lim_up = 8.5
		
		# dura_lim_up = 50.
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.20
		# rms_lim_up = 0.33
		
		# rise_lim_up = 25.
		
		# max_lim_up = 2.8
		# max_lim_down = 0.8
		
		# freq_lim_down = 280.
		# freq_lim_up = 430.
		
		
		# #AE-3-C-n25
		# count_lim_down = 2.
		# count_lim_up = 20.
		
		# maxcorr_lim_down = 0.55
		
		# crest_lim_down = 5.0
		# crest_lim_up = 9.5
		
		# dura_lim_up = 420.
		# dura_lim_down = 5. 
		
		# rms_lim_down = 0.05
		# rms_lim_up = 0.45
		
		# rise_lim_up = 40.
		
		# max_lim_up = 2.5
		# max_lim_down = 0.4
		
		# freq_lim_down = 260.
		# freq_lim_up = 460.
		

		
		
		
		
		
		dura = []
		crest = []
		count = []
		rise = []
		maxcorr = []
		rms = []
		freq = []
		max = []
		n_id = []
		
		
		# kurt = []
		# skew = []
		# difp2 = []		
		# maxdif = []		
		# per25 = []
		# per50 = []
		# per75 = []		
		# area = []
		
		# xnew = xraw		
		
		xnew = np.zeros(len(xraw))	
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):		
			signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			signal_complete = xraw[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
			if len(signal) >= 10:
				for i in range(len(signal_complete)):
					xnew[i + int(t_ini*config['fs'])] = signal_complete[i]

		
		
		
	
		index = 0
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
			signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			signal_complete = xraw[int(t_ini*config['fs']) : int(t_ini*config['fs']) + int(config['window_time']*config['fs'])]
			print(len(signal))
			
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
				n_id.append(index)
				
				
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
				
				
				
				
				
				length_to_cut = len(signal_complete)
				if contador <= count_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.						
				if contador >= count_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				
				if maxcorr_it <= maxcorr_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				
				if crest_it <= crest_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.				
				if crest_it >= crest_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				
				if dura_it <= dura_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.				
				if dura_it >= dura_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
						
						
				if rise_it >= rise_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				
				if rms_it >= rms_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				if rms_it <= rms_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				
				if freq_it >= freq_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				if freq_it <= freq_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				
				if max_it >= max_lim_up:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				if max_it <= max_lim_down:
					for i in range(length_to_cut):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				

				
			else:
				dura.append(0)
				crest.append(0)
				rise.append(0)
				count.append(0)
				maxcorr.append(0)
				rms.append(0)
				max.append(0)
				freq.append(0)
				n_id.append(-1)
				
				
				# kurt.append(0)
				# skew.append(0)
				# difp2.append(0)
				# maxdif.append(0)
				# per25.append(0)
				# per50.append(0)
				# per75.append(0)
				# area.append(0)
			index += 1
		

		fig_1, ax_1 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_1, ax_1, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		ax_1b = ax_1.twinx()
		ax_1b.plot(t_burst_corr, dura, '-r', marker='s')
		ax_1b.set_ylabel('Dauer', color='r')
		ax_1b.tick_params('y', colors='r')
		
		
		
		fig_2, ax_2 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_2, ax_2, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_2b = ax_2.twinx()
		ax_2b.plot(t_burst_corr, crest, '-r', marker='s')
		ax_2b.set_ylabel('Crest', color='r')
		ax_2b.tick_params('y', colors='r')
		
		
		fig_3, ax_3 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_3, ax_3, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_3b = ax_3.twinx()
		ax_3b.plot(t_burst_corr, count, '-r', marker='s')
		ax_3b.set_ylabel('Count', color='r')
		ax_3b.tick_params('y', colors='r')
		
		
		fig_4, ax_4 = plt.subplots(nrows=1, ncols=1)
		ax_4b = ax_4.twinx()
		plot_burst_rev(fig_4, ax_4, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		
		ax_4b.plot(t_burst_corr, rise, '-r', marker='s')
		ax_4b.set_ylabel('Rise', color='r')
		ax_4b.tick_params('y', colors='r')
		
		
		fig_5, ax_5 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_5, ax_5, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_5b = ax_5.twinx()
		ax_5b.plot(t_burst_corr, maxcorr, '-r', marker='s')
		ax_5b.set_ylabel('MaxCorr', color='r')
		ax_5b.tick_params('y', colors='r')
		
		
		fig_6, ax_6 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_6, ax_6, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_6b = ax_6.twinx()
		ax_6b.plot(t_burst_corr, rms, '-r', marker='s')
		ax_6b.set_ylabel('RMS', color='r')
		ax_6b.tick_params('y', colors='r')
		
		
		fig_7, ax_7 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_7, ax_7, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_7b = ax_7.twinx()
		ax_7b.plot(t_burst_corr, max, '-r', marker='s')
		ax_7b.set_ylabel('MAX', color='r')
		ax_7b.tick_params('y', colors='r')
		
		
		fig_8, ax_8 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_8, ax_8, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_8b = ax_8.twinx()
		ax_8b.plot(t_burst_corr, freq, '-r', marker='s')
		ax_8b.set_ylabel('FREQ', color='r')
		ax_8b.tick_params('y', colors='r')
		
		
		fig_9, ax_9 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_9, ax_9, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)	

		fig_10, ax_10 = plt.subplots(nrows=1, ncols=1)
		# ax_10.plot(t, xnew)
		ax_10.set_title(config['channel'] + ' WFM \n' + filename_signal, fontsize=10)
		ax_10.set_xlabel('Time (s)')		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax_10.set_ylabel('Amplitude (m$V_{in}$)')
		
		
		
		
		# fig_11, ax_11 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_11, ax_11, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_11b = ax_11.twinx()
		# ax_11b.plot(t_burst_corr, kurt, '-r', marker='s')
		# ax_11b.set_ylabel('kurt', color='r')
		# ax_11b.tick_params('y', colors='r')
		
		
		# fig_12, ax_12 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_12, ax_12, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_12b = ax_12.twinx()
		# ax_12b.plot(t_burst_corr, skew, '-r', marker='s')
		# ax_12b.set_ylabel('skew', color='r')
		# ax_12b.tick_params('y', colors='r')
		
		# fig_13, ax_13 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_13, ax_13, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_13b = ax_13.twinx()
		# ax_13b.plot(t_burst_corr, difp2, '-r', marker='s')
		# ax_13b.set_ylabel('difp2', color='r')
		# ax_13b.tick_params('y', colors='r')
		
		# fig_14, ax_14 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_14, ax_14, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_14b = ax_14.twinx()
		# ax_14b.plot(t_burst_corr, maxdif, '-r', marker='s')
		# ax_14b.set_ylabel('MaxDiff', color='r')
		# ax_14b.tick_params('y', colors='r')
		
		# fig_15, ax_15 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_15, ax_15, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_15b = ax_15.twinx()
		# ax_15b.plot(t_burst_corr, per25, '-r', marker='s')
		# ax_15b.set_ylabel('per25', color='r')
		# ax_15b.tick_params('y', colors='r')
		
		
		# fig_16, ax_16 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_16, ax_16, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_16b = ax_16.twinx()
		# ax_16b.plot(t_burst_corr, per50, '-r', marker='s')
		# ax_16b.set_ylabel('per50', color='r')
		# ax_16b.tick_params('y', colors='r')
		
		
		# fig_17, ax_17 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_17, ax_17, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_17b = ax_17.twinx()
		# ax_17b.plot(t_burst_corr, per75, '-r', marker='s')
		# ax_17b.set_ylabel('per75', color='r')
		# ax_17b.tick_params('y', colors='r')
		
		
		# fig_18, ax_18 = plt.subplots(nrows=1, ncols=1)
		# plot_burst_rev(fig_18, ax_18, 0, t, xnew, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		# ax_18b = ax_18.twinx()
		# ax_18b.plot(t_burst_corr, area, '-r', marker='s')
		# ax_18b.set_ylabel('area', color='r')
		# ax_18b.tick_params('y', colors='r')
		
		
		
		
		
		
		
		
		
		
		xnewraw = xnew
		
		
		xnew = np.absolute(xnew)
		reference = np.absolute(reference)
		
		# correlation = np.correlate(reference/(np.sum(reference**2))**0.5, xnew/(np.sum(xnew**2))**0.5, mode='same')		
		# correlation = correlation.tolist()
		# correlation = np.array(correlation[::-1])
		
		
		correlation = np.convolve(reference/(np.sum(reference**2))**0.5, xnew/(np.sum(xnew**2))**0.5, mode='same')		
		
		ax_10b = ax_10.twinx()
		ax_10b.plot(t, correlation, 'grey')
		ax_10.plot(t, xnewraw)
		ax_10b.set_ylabel('Cross-Correlation', color='grey')
		ax_10b.tick_params('y', colors='grey')
		
		
		fig_19, ax_19 = plt.subplots(nrows=1, ncols=1)
		ax_19.plot(t, correlation)
		ax_19.set_title(config['channel'] + ' Signal RB-CC \n' + filename_signal, fontsize=10)
		ax_19.set_xlabel('Time (s)')
		ax_19.set_ylabel('Amplitude (-)')
		
		
		
		
		fig_20, ax_20 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_20, ax_20, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_20b = ax_20.twinx()
		ax_20b.plot(t_burst_corr, n_id, '-r', marker='s')
		ax_20b.set_ylabel('INDEX', color='r')
		ax_20b.tick_params('y', colors='r')
		
		
		fig_21, ax_21 = plt.subplots(nrows=2, ncols=1, sharex=True)
		ax_21[1].plot(t, correlation)
		ax_21[0].plot(t, xnewraw)
		
		# ax_21[1].set_title(config['channel'] + ' WFM-RB Cross-Corr. with HSN-Source at A', fontsize=10)
		title = config['channel'] + ' Faltungsfunktion mit HSN-Burst an F'
		title = title.replace('_', '-')
		ax_21[1].set_title(title, fontsize=11)
		ax_21[1].set_xlabel('Dauer [s]', fontsize=11)
		ax_21[1].set_ylabel('Amplitude [-]', fontsize=11)
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax_21[0].set_ylabel('Amplitude [m$V_{in}$]', fontsize=11)
		
		
		title = config['channel'] + ' WFM Restsignal '
		title = title.replace('_', '-')
		ax_21[0].set_title(title, fontsize=11)
		# ax_21[0].set_xlabel('Time (s)')
		# ax_21[0].set_ylabel('Amplitude (-)')
		
		
		
		
		label = config['channel']
		label = label.replace('_', '-')
		
		fig_22, ax_22 = plt.subplots(nrows=2, ncols=1, sharex=True)
		ax_22[1].plot(t, xnewraw)
		ax_22[0].plot(t, xraw)
		
		ax_22[1].set_title(label + ' Erkennung der Bursts', fontsize=11)
		ax_22[1].set_xlabel('Dauer [s]', fontsize=12)
		ax_22[1].set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
		
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		ax_22[0].set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
		ax_22[0].tick_params(axis='both', labelsize=11)
		ax_22[1].tick_params(axis='both', labelsize=11)
		
		
		ax_22[0].set_title(label + ' mit Filter', fontsize=11)
		
		
		
		
		
		
		
		
		plt.tight_layout()
		plt.show()
		
		
		input_index = 0
		while input_index != -1:
			input_index = input('Select index burst:.... = ')
			print('rms = ', rms[int(input_index)])
			print('freq = ', freq[int(input_index)])
			print('max = ', max[int(input_index)])
			print('maxcorr = ', maxcorr[int(input_index)])
			print('rise = ', rise[int(input_index)])
			print('dura = ', dura[int(input_index)])
			print('crest = ', crest[int(input_index)])
			print('count = ', count[int(input_index)])

		
		
		
		
		
		

	
	elif config['mode'] == 'no_correlation_feature_2':
		print('Select signal')
		t, x, f, magX, xraw, filename_signal = invoke_signal(['--channel', 'AE_2', '--fs', str(config['fs']), '--power2', 'OFF', '--plot', 'ON'])
		
		print('Select burst for feature correlation')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		burst = read_pickle(filename)
		
		

		
		
		
		fig_1, ax_1 = plt.subplots(nrows=1, ncols=1)
		# ax.plot(t, correlation)
		ax_1.set_xlabel('Zeit (s)')
		ax_1.set_ylabel('Amplitude (-)')
		ax_1.set_title('AE_2' + ' ' + 'WFM' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		ax_1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		# plt.show()
		
		
		from THR_Burst_Detection import thr_burst_detector as thr_burst_detector
		from THR_Burst_Detection import read_threshold as read_threshold
		from THR_Burst_Detection import plot_burst_rev as plot_burst_rev
		
		xraw = x
		
		config = {'fs':1.e6, 'thr_value':1.0, 'thr_mode':'fixed_value', 'window_time':500, 'n_files':1, 'save_plot':'OFF'}
		config['window_time'] = config['window_time'] / 1000000.
		
		
		x, t_burst_corr, amp_burst_corr = thr_burst_detector(xraw, config, count=0)		
		xrev, t_burst_corr_rev, amp_burst_corr_rev = thr_burst_detector(xraw[::-1], config, count=0)
		
		tr = len(t)/config['fs']
		t_burst_corr_rev = t_burst_corr_rev[::-1]
		t_burst_corr_rev = tr*np.ones(len(t_burst_corr_rev)) - np.array(t_burst_corr_rev)
		t_burst_corr_rev = t_burst_corr_rev.tolist()
		amp_burst_corr_rev = amp_burst_corr_rev[::-1]
		
		
		
		# plot_burst_rev(fig_1, ax_1, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		
		
		count_lim_down = 3.
		count_lim_up = 15.
		
		maxcorr_lim_down = 0.4		
		
		crest_lim_down = 1.2
		crest_lim_up = 3.0
		
		dura_lim_up = 200. / 1000000.
		dura_lim_down = 5. / 1000000.
		
		rms_lim_up = 1.25
		
		rise_lim_up = 80.
		
		
		dura = []
		crest = []
		count = []
		rise = []
		maxcorr = []
		rms = []
		
		xnew = np.zeros(len(xraw))
	
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
		
			signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]			
				
			for i in range(len(signal)):
				xnew[i + int(t_ini*config['fs'])] = signal[i]
		
		ax_1.plot(t, xnew)		
		
		
		
		
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
			signal = xnew[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			
			try:
				dura_it = t_fin - t_ini
				dura.append(dura_it)
				rms_it = signal_rms(signal)
				
				rms.append(rms_it)
				
				rise_it = np.argmax(signal)/config['fs']
				rise.append(rise_it)
				
				contador = 0
				for u in range(len(signal)-1):			
					if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
						contador = contador + 1
				count.append(contador)
				
				
				maxcorr_it = max_norm_correlation(signal, burst)
				maxcorr.append(maxcorr_it)
				
				crest_it = np.max(np.absolute(signal))/rms_it
				crest.append(crest_it)
				
				
				if contador <= count_lim_down:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
						
				if contador >= count_lim_up:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				if maxcorr_it <= maxcorr_lim_down:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				if crest_it <= crest_lim_down:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				if crest_it >= crest_lim_up:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				if dura_it <= dura_lim_down:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				if dura_it >= dura_lim_up:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
						
				if rise_it >= rise_lim_up:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.
				
				if rms_it >= rms_lim_up:
					for i in range(len(signal)):
						xnew[i + int(t_ini*config['fs'])] = 0.

				
			except:
				dura.append(0)
				crest.append(0)
				rise.append(0)
				count.append(0)
				maxcorr.append(0)
				rms.append(0)
			
		dura = np.array(dura)*1000.*1000.
		dura = dura.tolist()
		
		rise = np.array(rise)*1000.*1000.
		rise = rise.tolist()
		
		fig_2, ax_2 = plt.subplots(nrows=1, ncols=1)
		ax_2.plot(t, xnew)
		
		
		
		xnew = np.absolute(xnew)
		burst = np.absolute(burst)
		
		correlation = np.correlate(burst/(np.sum(burst**2))**0.5, xnew/(np.sum(xnew**2))**0.5, mode='same')
		
		correlation = correlation.tolist()
		correlation = np.array(correlation[::-1])
		
		
		fig_3, ax_3 = plt.subplots(nrows=1, ncols=1)
		ax_3.plot(t, correlation)
		ax_3.set_xlabel('Zeit (s)')
		ax_3.set_ylabel('Amplitude (-)')
		ax_3.set_title('AE_2' + ' ' + 'Kreuzkorrelation' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		ax_3.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		plt.show()
		
		
		
		
		

		
		
		

		
		
		
		
		
		
		
		
		
		
		
		
	
	
	elif config['mode'] == 'no_correlation_feature_burst':
		print('Select signal')
		# t, x, f, magX, xraw, filename_signal = invoke_signal(['--channel', 'AE_2', '--fs', str(config['fs']), '--power2', 'OFF', '--plot', 'ON'])
		
		
		
		print('Select burst')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		
		x = read_pickle(filename)
		t = [i/config['fs'] for i in range(len(x))]

		
		
		
		fig_1, ax_1 = plt.subplots(nrows=1, ncols=1)
		# ax.plot(t, correlation)
		ax_1.set_xlabel('Zeit (s)')
		ax_1.set_ylabel('Amplitude (-)')
		# ax_1.set_title('AE_2' + ' ' + 'WFM' + '\n' + filename_signal, fontsize=10)
		# plt.title('Cross Correlation')
		ax_1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
		# plt.show()
		
		
		from THR_Burst_Detection import thr_burst_detector as thr_burst_detector
		from THR_Burst_Detection import read_threshold as read_threshold
		from THR_Burst_Detection import plot_burst_rev as plot_burst_rev
		
		xraw = x
		
		config = {'fs':1.e6, 'thr_value':2.5, 'thr_mode':'fixed_value', 'window_time':200., 'n_files':1, 'save_plot':'OFF'}
		
		config['window_time'] = config['window_time'] / 1000000.
		
		x, t_burst_corr, amp_burst_corr = thr_burst_detector(xraw, config, count=0)		
		xrev, t_burst_corr_rev, amp_burst_corr_rev = thr_burst_detector(xraw[::-1], config, count=0)
		
		tr = len(t)/config['fs']
		t_burst_corr_rev = t_burst_corr_rev[::-1]
		t_burst_corr_rev = tr*np.ones(len(t_burst_corr_rev)) - np.array(t_burst_corr_rev)
		t_burst_corr_rev = t_burst_corr_rev.tolist()
		amp_burst_corr_rev = amp_burst_corr_rev[::-1]
		
		
		
		plot_burst_rev(fig_1, ax_1, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)
		
		
		
		
		dura = []
		crest = []
		count = []
		rise = []
		for t_ini, t_fin in zip(t_burst_corr, t_burst_corr_rev):
			signal = xraw[int(t_ini*config['fs']) : int(t_fin*config['fs'])]
			

			
			dura.append(t_fin - t_ini)
			crest.append(np.max(np.absolute(signal))/signal_rms(signal))
			rise.append(np.argmax(signal)/config['fs'])
			
			contador = 0
			for u in range(len(signal)-1):			
				if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
					contador = contador + 1
			count.append(contador)
			
		dura = np.array(dura)*1000.*1000.
		dura = dura.tolist()
		
		rise = np.array(rise)*1000.*1000.
		rise = rise.tolist()
		
		ax_1b = ax_1.twinx()
		ax_1b.plot(t_burst_corr, dura, 's')
		ax_1b.set_ylabel('Dauer', color='r')
		ax_1b.tick_params('y', colors='r')
		
		
		
		fig_2, ax_2 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_2, ax_2, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_2b = ax_2.twinx()
		ax_2b.plot(t_burst_corr, crest, 's')
		ax_2b.set_ylabel('Crest', color='r')
		ax_2b.tick_params('y', colors='r')
		
		
		fig_3, ax_3 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_3, ax_3, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_3b = ax_3.twinx()
		ax_3b.plot(t_burst_corr, count, 's')
		ax_3b.set_ylabel('Count', color='r')
		ax_3b.tick_params('y', colors='r')
		
		
		fig_4, ax_4 = plt.subplots(nrows=1, ncols=1)
		plot_burst_rev(fig_4, ax_4, 0, t, xraw, config, t_burst_corr, amp_burst_corr, t_burst_corr_rev, amp_burst_corr_rev)		
		ax_4b = ax_4.twinx()
		ax_4b.plot(t_burst_corr, rise, 's')
		ax_4b.set_ylabel('Rise', color='r')
		ax_4b.tick_params('y', colors='r')
		
		
		
		
		
		
		
		plt.show()
	
	else:
		print('unknown mode')
		
		
		
	return

# plt.show()
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
	config['n_files'] = int(config['n_files'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config



if __name__ == '__main__':
	main(sys.argv)
