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
import scipy
import os.path
import sys
from os import chdir
from os.path import join, isdir, basename, dirname, isfile

import pandas as pd
sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))

plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser

# parser = argparse.ArgumentParser()

# parser.add_argument('--channel', nargs='?')
# # channel = 'AE_Signal'
# # channel = 'Koerperschall'
# # channel = 'Drehmoment'


# parser.add_argument('--power2', nargs='?')
# # n_points = 2**power2

# parser.add_argument('--type', nargs='?')
# args = parser.parse_args()

# if args.channel != None:
	# channel = args.channel
# else:
	# print('Required: Channel')
	# sys.exit()

# if args.power2 != None:
	# n_points = 2**int(args.power2)

# print(channel)
# print(n_points)

# print(args.pos)
# print(args.type)
# sys.exit()

#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {'power2':'OFF', 'channel_signal':'AE_0', 'fs_signal':1.e6, 'name':'auto', 'channel_tacho':'TAC_0', 'fs_tacho':50.e3}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	if config['mode'] == 'analysis':

		
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho pkl---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		tacho = read_pickle(filename)
		
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		

		point_index = filename.find('.')
		extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]

		if extension == 'mat':
			# x, channel = f_open_mat_2(filename)
			x = f_open_mat(filename, channel)
			x = np.ndarray.flatten(x)

		elif extension == 'tdm': #tdms
			x = f_open_tdms(filename, channel_signal)
			# x = f_open_tdms_2(filename)


		elif extension == 'txt': #tdms
			x = np.loadtxt(filename)
		filename = os.path.basename(filename) #changes from path to file
		print(filename)

		#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++

		# if power2 == None:
			# print('jiji')
			# n_points = 2**(max_2power(len(x)))
		# print(n_points)
		# x = x[0:n_points]	
		# nemo = 10.0
		# count = 0
		# while nemo > 1.0:
			# count = count + 1
			# nemo = n / 2.0**count
		# count = count - 1


		# x = x[0:n] #reduces number of points to a power of 2 to improve speed
			
		dt_tacho = 1.0/fs_tacho
		dt_x = 1.0/fs_signal
		
		# x = x[int(2.665/dt):int(2.7/dt)]
		# print(int((np.argmax(x)*dt - 0.1)*fs))
		# print(int((np.argmax(x)*dt + 0.01)*fs))
		
		# x = butter_bandpass(x=x, fs=config['fs'], freqs=[90.e3, 1000.e3], order=3)	
		# x = butter_highpass(x=x, fs=config['fs'], freq=90.e3, order=3)	
		# noise = x[0:int((np.argmax(x)*dt - 0.002)*fs)]
		# x = x[int((np.argmax(x)*dt - 0.002)*fs):int((np.argmax(x)*dt + 0.01)*fs)]	
		
		
		# print(signal_rms(x))
		# print(signal_rms(noise))
		# burst_rms = signal_rms(x) - signal_rms(noise)
		# print(burst_rms)
		
		if config['power2'] == 'auto':
			n_points = 2**(max_2power(len(x)))
		elif config['power2'] == 'OFF':
			n_points = len(x)
		else:
			n_points = 2**config['power2']
		
		x = x[0:n_points]
		# print(len(x))
		# x = x /
		# sys.exit()
		
		# x = (x /281.8) * 1000
		# x = (x /141.25) * 1000
		
		n_x = len(x)
		n_points_x = n_x
		tr_x = n_x*dt_x
		t_x = np.array([i*dt_x for i in range(n_x)])
		
		
		
		n_tacho = len(tacho)
		n_points_tacho = n_tacho
		tr_tacho = n_tacho*dt_tacho
		t_tacho = np.array([i*dt_tacho for i in range(n_tacho)])
		
		
		# for i in range(len(x)):
			# if x[i] <= 72.:
				# x[i] = 0.
			# else:
				# x[i] = 1.
		
		
		
		# x = to_dBAE(x, 43.)
		# x = x * 1000.
		#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

		config_analysis = {'WFM':True, 'FFT':False, 'PSD':False, 'STFT':False, 'STPSD':False, 'Cepstrum':False, 'CyclicSpectrum':False}

		config_demod = {'analysis':True, 'mode':'hilbert', 'prefilter':['highpass', 400.e3, 3], 
		'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3]}
		#When hilbert is selected, the other parameters are ignored

		config_diff = {'analysis':False, 'length':1, 'same':True}

		config_stft = {'segments':5000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}

		config_stPSD = {'segments':200, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}
		
		config_denois = {'analysis':True, 'mode':'butter_bandpass', 'freq':[95.e3, 140.e3]}
		
		config_CyclicSpectrum = {'segments':50, 'freq_range':[100.0e3, 500.0e3], 'window':'boxcar', 'mode':'magnitude', 'log':False, 'off_PSD':True, 'kHz':True, 'warm_points':None}

		#++++++++++++++++++++++ SIGNAL DEFINITION ++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if config_demod['analysis'] == True:
			print('with demod')
			if config_demod['mode'] == 'hilbert':
				x = hilbert_demodulation(x)
			elif config_demod['mode'] == 'butter':
				x = butter_demodulation(x=x, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
				type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
			else:
				print('Error assignment demodulation')

		if config_diff['analysis'] == True:
			print('with diffr')
			if config_diff['same'] == True:
				x = diff_signal_eq(x=x, length_diff=config_diff['length'])
			elif config_diff['same'] == False:
				x = diff_signal(x=x, length_diff=config_diff['length'])
			else:
				print('Error assignment diff')

		if config_denois['analysis'] == True:
			print('with filter')
			if config_denois['mode'] == 'butter_highpass':
				x = butter_highpass(x=x, fs=config['fs_signal'], freq=config_denois['freq'], order=3, warm_points=None)
			elif config_denois['mode'] == 'butter_bandpass':
				x = butter_bandpass(x=x, fs=config['fs'], freqs=config_denois['freq'], order=3, warm_points=None)
			elif config_denois['mode'] == 'butter_lowpass':
				x = butter_lowpass(x=x, fs=config['fs'], freq=config_denois['freq'], order=3, warm_points=None)

			else:
				print('Error assignment denois')
		
		# print('valor rms!!!!!!!!!!!!!!!!!!!!')
		# print(signal_rms(x))
		

		times = []
		for i in range(len(tacho)-1):
			if tacho[i] == 0 and tacho[i+1] == 1:
				times.append(i*dt_tacho)
		print(times)
		
		x = x[int(times[0]*fs_signal):int(times[1]*fs_signal)]	
		print(len(x))
		save_pickle('synchro_' + filename + '.pkl', x)
		
		
		n_x = len(x)
		n_points_x = n_x
		tr_x = n_x*dt_x
		t_x = np.array([i*dt_x for i in range(n_x)])
		
		#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config_analysis['FFT'] == True:
			magX, f, df = mag_fft(x, fs)
			print(len(magX))


		#++++++++++++++++++++++ STFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config_analysis['STFT'] == True:
			segments = config_stft['segments']
			window = config_stft['window']
			mode = config_stft['mode']
			
			stftX, f_stft, df_stft, t_stft = shortFFT(x, fs, segments, window, mode)
			if config_stft['log-scale'] == True:
				stftX = np.log(stftX)


		#++++++++++++++++++++++ PSD +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config_analysis['PSD'] == True:
			f_psd, psdX = signal.periodogram(x, fs, return_onesided=True, scaling='density')

		#++++++++++++++++++++++ STPSD +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config_analysis['STPSD'] == True:
			segments = config_stPSD['segments']
			window = config_stPSD['window']
			mode = config_stPSD['mode']
			
			stPSDX, f_stPSD, df, t_stPSD = shortPSD(x, fs, segments)	
			if config_stPSD['log-scale'] == True:
				stPSDX = np.log(stPSDX)


		#++++++++++++++++++++++ CEPSTRUM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if config_analysis['Cepstrum'] == True:
			cepstrumX, tc, dtc = cepstrum_real(x, fs)

			
		#++++++++++++++++++++++ CICLO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if config_analysis['CyclicSpectrum'] == True:
			segments = config_CyclicSpectrum['segments']
			window = config_CyclicSpectrum['window']
			mode = config_CyclicSpectrum['mode']
			freq_range = config_CyclicSpectrum['freq_range']
			
			CyclicSpectrum, a_CyclicSpectrum, f_CyclicSpectrum = Cyclic_Spectrum(x=x, fs=fs, segments=segments, freq_range=freq_range, warm_points=config_CyclicSpectrum['warm_points'])
			if config_CyclicSpectrum['off_PSD'] == True:
				for i in range(len(CyclicSpectrum)):
					CyclicSpectrum[i][0] = 0.

			if config_CyclicSpectrum['log'] == True:
				CyclicSpectrum = np.log(CyclicSpectrum)

		#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		count = 0
		fig = []
		for element in config_analysis:
			if config_analysis[element] == True:
				# count = count + 1
				fig.append(plt.figure(count))
				
				# plt.figure(count)
				# plt.title(channel + ' ' + element)
				# plt.suptitle(filename, fontsize=10)
				if element == 'WFM':
					ax_wfm = fig[count].add_subplot(1,1,1)
					ax_wfm.set_title(str(channel_signal) + ' ' + element + '\n' + filename, fontsize=10)
					ax_wfm.set_label('AE')
					ax_wfm.plot(t_x, x)
					ax_wfm.plot(t_tacho, tacho)
					ax_wfm.set_xlabel('Time s')
					params = {'mathtext.default': 'regular' }          
					plt.rcParams.update(params)
					ax_wfm.set_ylabel('Amplitude (m$V_{in}$)')
					# ax_wfm.text(0.008, 0.9*np.max(x), 'RMS ' + "{:.2E}".format(Decimal(str(burst_rms))), fontsize=12)
					# ax_wfm.set_xticks(np.arange(0, n_points*dt, 0.001))
					# plt.grid()
					# fig, ax = plt.subplots(nrows=1, ncols=1)
					# ax.set_xticks(np.arange(0, n_points*dt, window_points*dt))
					# ax.plot(t, x1)
					
					# plt.show()

				elif element == 'FFT':
					ax_fft = fig[count].add_subplot(1,1,1)
					ax_fft.set_title(channel + ' ' + element + '\n' + filename, fontsize=10)
					ax_fft.plot(f/1000., magX, 'r')
					ax_fft.set_xlabel('Frequency kHz')
					params = {'mathtext.default': 'regular' }          
					plt.rcParams.update(params)
					ax_fft.set_ylabel('Amplitude (m$V_{in}$)')
					ax_fft.ticklabel_format(style='sci', scilimits=(-4, 4))
				elif element == 'PSD':
					ax_psd = fig[count].add_subplot(1,1,1)
					ax_psd.set_title(channel + ' ' + element)
					ax_psd.plot(f_psd, psdX)
					ax_psd.set_xlabel('Frequency Hz')
					ax_psd.set_ylabel('Amplitude V')
					# ax.set_xticklabels(['{:.}'.format(int(x)) for x in ax.get_xticks().tolist()])
				elif element == 'STFT':
					ax_stft = fig[count].add_subplot(1,1,1)
					# ax_stft.set_title(channel + ' ' + element)
					ax_stft.set_title(str(channel) + ' ' + element + '\n' + filename, fontsize=10)
					
					ax_stft.pcolormesh(t_stft, f_stft/1000., stftX)
					ax_stft.set_xlabel('Time s')
					ax_stft.set_ylabel('Frequency kHz')
					
					map = []
					vmax = max_cspectrum(stftX)				
					map.append(ax_stft.pcolormesh(t_stft, f_stft/1000., stftX, cmap='plasma', vmax=vmax))
					
					# ax_stft.ticklabel_format(style='sci', scilimits=(-2, 2))
					
					
					# indmax = np.argmax([max_cspectrum(CyclicSpectrum1), max_cspectrum(CyclicSpectrum2)])
					fig[count].colorbar(map[0], ax=ax_stft, ticks=np.linspace(0, vmax, 5), format='%1.2e')
					
					
					# indmax = np.argmax([max_cspectrum(CyclicSpectrum1)])
					# fig[count].colorbar(ax=ax_stft)

				elif element == 'STPSD':
					ax_stpsd = fig[count].add_subplot(1,1,1)
					ax_stpsd.set_title(channel + ' ' + element)
					ax_stpsd.pcolormesh(t_stPSD, f_stPSD/1000., stPSDX)
					ax_stpsd.set_xlabel('Time s')
					ax_stpsd.set_ylabel('Frequency kHz')
				elif element == 'Cepstrum':
					ax_cepstrum = fig[count].add_subplot(1,1,1)
					ax_cepstrum.set_title(channel + ' ' + element)
					ax_cepstrum.plot(tc, cepstrumX)
					ax_cepstrum.set_xlabel('Quefrency s')
					ax_cepstrum.set_ylabel('Amplitude')
				
				
				elif element == 'CyclicSpectrum':
					ax_ciclic = fig[count].add_subplot(1,1,1)
					# ax_stft.set_title(channel + ' ' + element)
					ax_ciclic.set_title(str(channel) + ' ' + element + '\n' + filename, fontsize=10)
					
					ax_ciclic.pcolormesh(a_CyclicSpectrum, f_CyclicSpectrum/1000., CyclicSpectrum)
					ax_ciclic.set_xlabel('Cyclic Frequency Hz')
					ax_ciclic.set_ylabel('Frequency kHz')
					
					
					# map = []
					# vmax = max_cspectrum(CyclicSpectrum)				
					# map.append(ax_ciclic.pcolormesh(a_CyclicSpectrum, f_CyclicSpectrum/1000., CyclicSpectrum, cmap='plasma', vmax=vmax))
					
					# ax_stft.ticklabel_format(style='sci', scilimits=(-2, 2))
					
					
					# indmax = np.argmax([max_cspectrum(CyclicSpectrum1), max_cspectrum(CyclicSpectrum2)])
					
					# fig[count].colorbar(map[0], ax=ax_ciclic, ticks=np.linspace(0, vmax, 5), format='%1.2e')
					# fig[count].colorbar(ax=ax_ciclic)
					
					# indmax = np.argmax([max_cspectrum(CyclicSpectrum1)])
					# fig[count].colorbar(ax=ax_stft)
				
				
				

				count = count + 1

		plt.show()


	elif config['mode'] == 'plot_with_tacho':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho ---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		# tacho = 100*np.array(load_signal(filename, channel=channel_tacho))
		tacho = 1*np.array(load_signal(filename, channel=channel_tacho))
		t_tacho = np.array([i/fs_tacho for i in range(len(tacho))])
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		
		# signal = butter_highpass(x=load_signal(filename, channel=channel_signal), fs=config['fs'], order=3, freq=20.e3)
		
		# signal = butter_highpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3)
		signal = load_signal(filename, channel=channel_signal)
		
		# signal = hilbert_demodulation((butter_bandpass(x=load_signal(filename, channel=channel_signal), 
		# fs=fs_signal, freqs=[90.e3, 140.e3], order=3, warm_points=None))**2.0
		
		
		# signal = hilbert_demodulation((butter_highpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3, warm_points=None))**2.0)
		
		t_signal = np.array([i/fs_signal for i in range(len(signal))])
		
		fig, ax = plt.subplots()
		ax.plot(t_signal, signal)
		ax.plot(t_tacho, tacho, 'g')
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Amplitude [mV]')
		plt.show()
	
	elif config['mode'] == 'single_tacho_avg':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho ---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		tacho = load_signal(filename, channel=channel_tacho)
		t_tacho = np.array([i/fs_tacho for i in range(len(tacho))])
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		
		# signal = load_signal(filename, channel=channel_signal)
		
		# signal = hilbert_demodulation(butter_bandpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freqs=[90.e3, 144.e3], order=3, warm_points=None))
		
		# signal = hilbert_demodulation(butter_highpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3, warm_points=None))
		
		signal = hilbert_demodulation((butter_bandpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freqs=[95.e3, 140.e3], order=3, warm_points=None)))

		
		# signal = butter_bandpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freqs=[90.e3, 144.e3], order=3, warm_points=None)
		
		t_signal = np.array([i/fs_signal for i in range(len(signal))])

		
		times_pulses = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] != 0]
		
		print(times_pulses)
		print(tacho)
		print(signal)
		
		signal_avg = obtain_syn_avg(signal, tacho, times_pulses, config)
		
		
		t_avg = [i/fs_signal for i in range(len(signal_avg))]

		fig, ax = plt.subplots()
		print(type(signal_avg))
		print(signal_avg)
		ax.plot(list(signal_avg))
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Amplitude [mV]')
		plt.show()
	
	elif config['mode'] == 'avg_per_time':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']

		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
		
		# signal = load_signal(filename, channel=channel_signal)
		
		# signal = hilbert_demodulation(butter_bandpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freqs=[90.e3, 144.e3], order=3, warm_points=None))
		
		# signal = hilbert_demodulation(butter_highpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3, warm_points=None))
		
		signal = hilbert_demodulation((butter_bandpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freqs=[95.e3, 140.e3], order=3, warm_points=None)))

		
		# signal = butter_bandpass(x=load_signal(filename, channel=channel_signal), fs=fs_signal, freqs=[90.e3, 144.e3], order=3, warm_points=None)
		
		t_signal = np.array([i/fs_signal for i in range(len(signal))])

		
		times_pulses = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] != 0]
		
		print(times_pulses)
		print(tacho)
		print(signal)
		
		signal_avg = obtain_syn_avg(signal, tacho, times_pulses, config)
		
		
		t_avg = [i/fs_signal for i in range(len(signal_avg))]

		fig, ax = plt.subplots()
		print(type(signal_avg))
		print(signal_avg)
		ax.plot(list(signal_avg))
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Amplitude [mV]')
		plt.show()
	
	elif config['mode'] == 'multi_tacho_avg':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho ---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Tacho = filedialog.askopenfilenames()
		root.destroy()	
		
		Tachos = [load_signal(filepath, channel=channel_tacho) for filepath in Filepaths_Tacho]
		# tacho = 
		# t_tacho = np.array([i/fs_tacho for i in range(len(tacho))])
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Signal = filedialog.askopenfilenames()
		root.destroy()
		
		# Signals = [load_signal(filepath, channel=channel_signal) for filepath in Filepaths_Signal]
		# Signals = [ for filepath in Filepaths_Signal]
		
		# Signals = [hilbert_demodulation(butter_highpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3)) for filepath in Filepaths_Signal]
		
		# Signals = [hilbert_demodulation(butter_highpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freqs=[90.e3, 144.e3], order=3)) for filepath in Filepaths_Signal]
		
		Signals = [butter_demodulation(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, filter=['lowpass', 5000., 3], prefilter=['highpass', 20.e3, 3]) for filepath in Filepaths_Signal]
		# butter_demodulation(x, fs, filter, prefilter=None, type_rect=None, dc_value=None):
		
		# t_signal = np.array([i/fs_signal for i in range(len(signal))])

		
		Signals_Avg = []
		for tacho, signal in zip(Tachos, Signals):
			times_pulses = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] != 0]		
			signal_avg, Synchro_Segments_Eq = obtain_syn_avg(signal, tacho, times_pulses, config)
			
			Signals_Avg.append(signal_avg)
		
		min_it = np.min(np.array([len(element) for element in Signals_Avg]))
		
		# Synchro_Segments_Eq = []
		big_signal_avg = np.zeros(min_it)
		for k in range(len(Signals_Avg)):
			segment = Signals_Avg[k]
			new_x = np.linspace(0., 1., num=min_it)
			old_x = np.linspace(0., 1., num=len(segment))
			segment_eq = np.interp(new_x, old_x, segment)
			# Synchro_Segments_Eq.append(segment_eq)
			big_signal_avg += segment_eq
		big_signal_avg = signal_avg / len(Signals_Avg)


		
		t_avg = np.array([i/fs_signal for i in range(len(big_signal_avg))])
		fig, ax = plt.subplots()
		ax.plot(t_avg, big_signal_avg)
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Amplitude [mV]')
		plt.show()
		
		
		env = butter_lowpass(x=np.array(big_signal_avg), fs=fs_signal, freq=50., order=3)
		plt.plot(t_avg, env, 'r')
		plt.show()
	
	elif config['mode'] == 'multi_tacho_avg_2':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho ---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Tacho = filedialog.askopenfilenames()
		root.destroy()	
		
		Tachos = [load_signal(filepath, channel=channel_tacho) for filepath in Filepaths_Tacho]
		# tacho = 
		# t_tacho = np.array([i/fs_tacho for i in range(len(tacho))])
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Signal = filedialog.askopenfilenames()
		root.destroy()
		
		Signals = [load_signal(filepath, channel=channel_signal)*10. for filepath in Filepaths_Signal]
		# Signals = [ for filepath in Filepaths_Signal]
		
		# Signals = [hilbert_demodulation(butter_highpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3)) for filepath in Filepaths_Signal]
		
		# Signals = [butter_demodulation(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, filter=['lowpass', 5000., 3], prefilter=['highpass', 20.e3, 3]) for filepath in Filepaths_Signal]
		
		# Signals = [np.absolute(butter_highpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3)) for filepath in Filepaths_Signal]
		
		# Signals2 = [hilbert_demodulation(butter_bandpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freqs=[90.e3, 144.e3], order=3)) for filepath in Filepaths_Signal]
		
		
		Master_Segments = []
		for tacho, signal in zip(Tachos, Signals):
			times_pulses = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] != 0]		
			signal_avg, Synchro_Segments_Eq = obtain_syn_avg(signal, tacho, times_pulses, config)
			
			Master_Segments += Synchro_Segments_Eq
		
		min_it = np.min(np.array([len(element) for element in Master_Segments]))
		
		# Synchro_Segments_Eq = []
		big_signal_avg = np.zeros(min_it)
		for k in range(len(Master_Segments)):
			segment = Master_Segments[k]
			new_x = np.linspace(0., 1., num=min_it)
			old_x = np.linspace(0., 1., num=len(segment))
			segment_eq = np.interp(new_x, old_x, segment)
			# Synchro_Segments_Eq.append(segment_eq)
			big_signal_avg += segment_eq
		big_signal_avg = signal_avg / len(Master_Segments)


		
		t_avg = np.array([i/fs_signal for i in range(len(big_signal_avg))])
		fig, ax = plt.subplots()
		ax.plot(t_avg, big_signal_avg*1000)
		ax.set_xlabel('Time [s]', fontsize=12)
		ax.set_ylabel('Amplitude Envelope [mV]', fontsize=12)
		plt.show()
		
		
		env = butter_lowpass(x=np.array(big_signal_avg), fs=fs_signal, freq=50., order=3)
		plt.plot(t_avg, env, 'r')
		plt.show()
		
		magX, f, df = mag_fft(x=np.array(big_signal_avg), fs=fs_signal)
		plt.plot(f, magX)
		plt.show()
	
	elif config['mode'] == 'multi_tacho_avg_3':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho ---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Tacho = filedialog.askopenfilenames()
		root.destroy()

		# Raw_Signals = [np.array(list(x[it_lag:]) + list(x[0:it_lag])) for x in Raw_Signals2]		
		
		# Tachos = [load_signal(filepath, channel=channel_tacho) for filepath in Filepaths_Tacho]
		
		Tachos = []
		for filepath in Filepaths_Tacho:
			tac = load_signal(filepath, channel=channel_tacho)

			Tachos.append(tac)
		# tacho = 
		# t_tacho = np.array([i/fs_tacho for i in range(len(tacho))])
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Signal = filedialog.askopenfilenames()
		root.destroy()
		
		# Signals = [load_signal(filepath, channel=channel_signal)*10. for filepath in Filepaths_Signal]
		# Signals = [ for filepath in Filepaths_Signal]
		
		# Signals = [hilbert_demodulation(butter_highpass(x=load_signal(filepath, channel=channel_signal)*1000., fs=fs_signal, freq=20.e3, order=3)) for filepath in Filepaths_Signal]
		
		# Signals = [butter_demodulation(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, filter=['lowpass', 5000., 3], prefilter=['highpass', 20.e3, 3]) for filepath in Filepaths_Signal]
		
		# Signals = [np.absolute(butter_highpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freq=20.e3, order=3)) for filepath in Filepaths_Signal]
		
		# x = load_signal(Filepaths_Signal[0], channel=channel_signal)
		# print(type(x))
		# print(len(x))
		# sys.exit()
		
		# Signals = [hilbert_demodulation((butter_bandpass(x=load_signal(filepath, channel=channel_signal), fs=fs_signal, freqs=[95.e3, 140.e3], order=3))**2.0) for filepath in Filepaths_Signal]
		
		Signals = []
		for filepath, filepath_tacho in zip(Filepaths_Signal, Filepaths_Tacho):
			x = load_signal(filepath, channel=channel_signal)
			# x = butter_bandpass(x=x, fs=config['fs_signal'], freqs=[70.e3, 150.e3], order=3, warm_points=None)
			x = np.absolute(x)
			Signals.append(x)
			
			name_signal = os.path.basename(filepath)
			name_tacho = os.path.basename(filepath_tacho)
			if name_tacho.find(name_signal[3:-5]) == -1:
				
				print('fatal error 25834')
				# sys.exit()
		
		
		# Signals = [butter_demodulation((butter_bandpass(x=x, fs=fs_signal, freqs=[95.e3, 140.e3], order=3))**2.0, fs=config['fs_signal'], filter=['lowpass', 2000., 3]) for x in Raw_Signals]
		
		# Signals = [butter_demodulation(x=x, fs=config['fs_signal'], filter=['lowpass', 2000., 3]) for x in Raw_Signals]
		
		# Signals = [butter_demodulation((butter_bandpass(x, fs=fs_signal, freqs=[105.e3, 295.e3], order=3)), fs=config['fs_signal'], filter=['lowpass', 2000., 3]) for x in Raw_Signals]
		
		# Signals = [hilbert_demodulation(butter_bandpass(x=x, fs=fs_signal, freqs=[95.e3, 140.e3], order=3)) for x in Raw_Signals]
		
		# Signals = [x for x in Raw_Signals]
		
		# Signals = [butter_demodulation((butter_highpass(x, fs=fs_signal, freq=20.e3, order=3)), fs=config['fs_signal'], filter=['lowpass', 1000., 3]) for x in Raw_Signals]
		

		
		n_rotations = 20
		
		Master_Segments = []
		for tacho, signal in zip(Tachos, Signals):
			times_pulses = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] != 0]
			times_pulses = np.array(times_pulses[0::n_rotations])
			# times_pulses = np.array(times_pulses)/1.02307692 #FOR Q!!!

			
			signal_avg, Synchro_Segments_Eq = obtain_syn_avg(signal, tacho, times_pulses, config)
			
			Master_Segments += Synchro_Segments_Eq
		
		min_it = np.min(np.array([len(element) for element in Master_Segments]))
		
		# Synchro_Segments_Eq = []
		big_signal_avg = np.zeros(min_it)
		for k in range(len(Master_Segments)):
			segment = Master_Segments[k]
			new_x = np.linspace(0., 1., num=min_it)
			old_x = np.linspace(0., 1., num=len(segment))
			segment_eq = np.interp(new_x, old_x, segment)
			# Synchro_Segments_Eq.append(segment_eq)
			big_signal_avg += segment_eq
		big_signal_avg = signal_avg / len(Master_Segments)
		print('len proms = ', len(Master_Segments))

		
		t_avg = np.array([i/fs_signal for i in range(len(big_signal_avg))])
		fig, ax = plt.subplots()
		ax.plot(t_avg, big_signal_avg)#
		raw = Signals[0]
		t = np.array([i/fs_signal for i in range(len(raw))])
		ax.plot(t, raw, color='r')
		ax.set_xlabel('Time [s]', fontsize=13)
		ax.set_ylabel('Synchro', fontsize=13)
		plt.plot()
		plt.show()
		
		save_pickle('Synchro_AbsSignal_Mess_Q_20rev_noCorr_last.pkl', big_signal_avg)
		sys.exit()
		# it_lag = 105554
		# it_lag = 124746
		it_lag = 141200
		print(len(big_signal_avg))
		tacL = np.array(list(big_signal_avg[it_lag:]) + list(big_signal_avg[0:it_lag]))
		
		# ax.plot(t_avg*360./(np.max(t_avg)), tacL)
		ax.plot(t_avg*360./(np.max(t_avg)), big_signal_avg)
		ax.set_xlabel('Planet Carrier Angle [°]', fontsize=13)
		ax.set_ylabel('Amplitude Envelope [mV]', fontsize=13)
		# ax.set_title('Initial Faulty Condition', fontsize=13)
		ax.set_title('No Fault', fontsize=13)
		ax.tick_params(axis='both', labelsize=12)
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

		
		plt.show()
		
		
		env = butter_lowpass(x=np.array(big_signal_avg), fs=fs_signal, freq=50., order=3)
		plt.plot(t_avg, env, 'r')
		plt.show()
		
		magX, f, df = mag_fft(x=np.array(big_signal_avg), fs=fs_signal)
		plt.plot(f, magX)
		plt.show()
	
	
	elif config['mode'] == 'obtain_residual_signal':
		channel_tacho = config['channel_tacho']
		channel_signal = config['channel_signal']
		fs_tacho = config['fs_tacho']
		fs_signal = config['fs_signal']
		
		print('Select tacho ---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Tacho = filedialog.askopenfilenames()
		root.destroy()


		
		Tachos = []
		for filepath in Filepaths_Tacho:
			tac = load_signal(filepath, channel=channel_tacho)

			Tachos.append(tac)
		# tacho = 
		# t_tacho = np.array([i/fs_tacho for i in range(len(tacho))])
		
		print('Select signal---')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Signal = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select synchro---')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_syn = filedialog.askopenfilename()
		root.destroy()
		
		synchro = read_pickle(filepath_syn)
		

		
		Signals = []
		for filepath, filepath_tacho in zip(Filepaths_Signal, Filepaths_Tacho):
			x = load_signal(filepath, channel=channel_signal)

			x = np.absolute(x)
			Signals.append(x)
			
			name_signal = os.path.basename(filepath)
			name_tacho = os.path.basename(filepath_tacho)
			if name_tacho.find(name_signal[3:-5]) == -1:
				
				print('fatal error 25834')
				# sys.exit()
		
		

		

		
		n_rotations = 20
		
		Master_Segments = []
		for tacho, signal in zip(Tachos, Signals):
			times_pulses = [i/config['fs_tacho'] for i in range(len(tacho)) if tacho[i] != 0]
			times_pulses = np.array(times_pulses[0::n_rotations])
			# times_pulses = np.array(times_pulses)/1.02307692 #FOR Q!!!

			
			signal_avg, Synchro_Segments_Eq = obtain_syn_avg(signal, tacho, times_pulses, config)
			
			Master_Segments += Synchro_Segments_Eq
		
		# min_it = np.min(np.array([len(element) for element in Master_Segments]))
		min_it = len(synchro)

		# Synchro_Segments_Eq = []
		big_signal_avg = np.zeros(min_it)
		for k in range(len(Master_Segments)):
			segment = Master_Segments[k]
			new_x = np.linspace(0., 1., num=min_it)
			old_x = np.linspace(0., 1., num=len(segment))
			segment_eq = np.interp(new_x, old_x, segment)
			
			data = segment_eq - synchro
			# big_signal_avg += segment_eq
			
			name = os.path.basename(Filepaths_Signal[k])[:-5]
			save_pickle('diff_' + name + '.pkl', data)
			
		# big_signal_avg = signal_avg / len(Master_Segments)
		# print('len proms = ', len(Master_Segments))

		
		# t_avg = np.array([i/fs_signal for i in range(len(big_signal_avg))])
		# fig, ax = plt.subplots()
		# ax.plot(t_avg, big_signal_avg)#
		# raw = Signals[0]
		# t = np.array([i/fs_signal for i in range(len(raw))])
		# ax.plot(t, raw, color='r')
		# ax.set_xlabel('Time [s]', fontsize=13)
		# ax.set_ylabel('Synchro', fontsize=13)
		# plt.plot()
		# plt.show()
		
		# save_pickle('Synchro_AbsSignal_Mess_Q_20rev_noCorr_last.pkl', big_signal_avg)
		sys.exit()
		# it_lag = 105554
		# it_lag = 124746
		it_lag = 141200
		print(len(big_signal_avg))
		tacL = np.array(list(big_signal_avg[it_lag:]) + list(big_signal_avg[0:it_lag]))
		
		# ax.plot(t_avg*360./(np.max(t_avg)), tacL)
		ax.plot(t_avg*360./(np.max(t_avg)), big_signal_avg)
		ax.set_xlabel('Planet Carrier Angle [°]', fontsize=13)
		ax.set_ylabel('Amplitude Envelope [mV]', fontsize=13)
		# ax.set_title('Initial Faulty Condition', fontsize=13)
		ax.set_title('No Fault', fontsize=13)
		ax.tick_params(axis='both', labelsize=12)
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

		
		plt.show()
		
		
		env = butter_lowpass(x=np.array(big_signal_avg), fs=fs_signal, freq=50., order=3)
		plt.plot(t_avg, env, 'r')
		plt.show()
		
		magX, f, df = mag_fft(x=np.array(big_signal_avg), fs=fs_signal)
		plt.plot(f, magX)
		plt.show()
	
	
		
	elif config['mode'] == 'synchro_avg':
		print('Select signals')
		root = Tk()
		root.withdraw()
		root.update()
		Filenames = filedialog.askopenfilenames()
		root.destroy()	
		
		Signals = [np.array(read_pickle(filename)) for filename in Filenames]

		
		Lengths = [len(signal) for signal in Signals]
		T = []
		for i in range(len(Signals)):
			T.append([k for k in range(Lengths[i])])
		# T = [i]
		# for (i, signal) in zip()
		
		print(Lengths)
		# min_list = [np.min(signal) for signal in Signals]
		# print(min_list)
		min_point = np.min(np.array(Lengths))
		print(min_point)
		t_eq = [i for i in range(min_point)]
		
		# Signals_eq = [np.interp(x=t_eq, xp=t, fp=signal) for (t, signal) in zip(T,Signals)]
		
		Signals_eq = []
		for (t, signal) in zip(T, Signals):
			Signals_eq.append(np.interp(x=t_eq, xp=t, fp=signal))
		
		avg = np.zeros(min_point)
		
		for k in range(len(Signals_eq)):
			avg = avg + Signals_eq[k]
		avg = avg / len(Signals_eq)
		
		plt.plot(avg)
		plt.show()
		
		save_pickle('avg_synchro.pkl', avg)
	
	elif config['mode'] == 'asynchro_obtain':
	
		print('Select signal avg synchro')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		synchro = read_pickle(filename)
		
		print('Select signal Asynchro')		
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()		

		# point_index = filename.find('.')
		# extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
		
		
		# channel_signal = config['channel_signal']
		
		# if extension == 'mat':
			# # x, channel = f_open_mat_2(filename)
			# x = f_open_mat(filename, channel)
			# x = np.ndarray.flatten(x)

		# elif extension == 'tdm': #tdms
			# x = f_open_tdms(filename, channel_signal)
			# # x = f_open_tdms_2(filename)


		# elif extension == 'txt': #tdms
			# x = np.loadtxt(filename)
		# filename = os.path.basename(filename) #changes from path to file
		# print(filename)	
		
		# Asynchro = x
		Asynchro = read_pickle(filename)
		
		
		
		

		plt.figure(0)
		plt.plot(synchro)
		plt.title('synchro')
		
		
		plt.figure(1)
		plt.plot(Asynchro)
		plt.title('Asynchro')
		
		
		min = np.min([len(Asynchro), len(synchro)])
		
		if len(synchro) > len(Asynchro):
			synchro = np.interp(x=np.array([i for i in range(len(Asynchro))]), xp=np.array([i for i in range(len(synchro))]), fp=synchro)
			print('juuj')
		elif len(Asynchro) > len(synchro):
			print('aaaaa')
			Asynchro = np.interp(x=np.array([i for i in range(len(synchro))]), xp=np.array([i for i in range(len(Asynchro))]), fp=Asynchro)
		
		# dif = np.absolute(synchro) - np.absolute(Asynchro)
		
		signal1 = synchro
		signal2 = Asynchro
		correlation = np.correlate(signal1/(np.sum(signal1**2))**0.5, signal2/(np.sum(signal2**2))**0.5, mode='same')
		
		plt.figure(2)
		plt.plot(correlation)
		plt.title('correlation')
		
		
		plt.show()
	
	elif config['mode'] == 'single_to_pulse':
		print('Select signal tacho')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		
		
		tacho = load_signal(filename, channel=config['channel_tacho'])
		tacho = butter_lowpass(x=tacho, fs=config['fs_tacho'], freq=50., order=3)
		

		tacho_new = []
		for i in range(len(tacho)-1):
			if (tacho[i] >= -0.002 and tacho[i+1] < -0.002):
				tacho_new.append(0.01)
			else:
				tacho_new.append(0)
		tacho_new.append(0)
		plt.plot(tacho)
		plt.plot(tacho_new)
		plt.show()
	
	
	elif config['mode'] == 'single_to_pulse_SH':
		print('Select signal tacho')
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()	
		
		tacho = f_open_tdms(filename, channel=config['channel_tacho'])
		
		# tacho = tacho[int(61*config['fs_tacho']) : int(81*config['fs_tacho'])]
		
		# tacho_raw = load_signal(filename, channel=config['channel_tacho'])
		# tacho = butter_lowpass(x=tacho_raw, fs=config['fs_tacho'], freq=50., order=3)
		
		# plt.plot(tacho, 'r')
		# plt.show()
		

		tacho_new = []
		for i in range(len(tacho)-1):
			print('A... ', i/(len(tacho)-1))
			if (tacho[i] <= -9 and tacho[i+1] > -9):
				tacho_new.append(1)
			else:
				tacho_new.append(0)
		tacho_new.append(0)
		
		flag = 'ON'
		tacho_new_2 = []
		for i in range(len(tacho_new)-1):
			print('B... ', i/(len(tacho_new)-1))
			if tacho_new[i] == 1:
				if flag == 'ON':
					tacho_new_2.append(1)
					flag = 'OFF'
				else:
					tacho_new_2.append(0)
					flag = 'ON'
			else:
				tacho_new_2.append(0)
		tacho_new_2.append(0)

		
		# plt.plot(tacho, 'r')
		# plt.plot(tacho_new)
		# plt.plot(tacho_new_2, 'g')
		# plt.show()
		
		save_pickle('tacho_' + os.path.basename(filename)[:-5] + '.pkl', tacho_new_2)
		
		mytacho = tacho_new_2
		vec = []
		n_ones = 0
		for i in range(len(mytacho)):
			print('C... ', i/len(mytacho))
			if mytacho[i] == 1:
				vec.append(i)
		
		pulses = 20
		speed = []
		for k in range(len(vec)-1):
			print('D... ', k/(len(vec)-1))
			speed.append(1./(pulses*(vec[k+1] - vec[k])/config['fs_tacho']))
		

		# plt.plot(speed, 'ko-')
		# plt.show()
		
		save_pickle('speed_' + os.path.basename(filename)[:-5] + '.pkl', speed)
	
	elif config['mode'] == 'int_SH':
		print('Select signal tacho')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_tacho = filedialog.askopenfilename()
		root.destroy()	
		
		print('Select signal speed')
		root = Tk()
		root.withdraw()
		root.update()
		filepath_speed = filedialog.askopenfilename()
		root.destroy()	
		
		tacho = read_pickle(filepath_tacho)
		speed = read_pickle(filepath_speed)
		
		t_tacho = np.linspace(0, 1, len(tacho))
		t_speed = np.linspace(0, 1, len(speed))
		
		speed_eq = np.interp(t_tacho, t_speed, speed)
		
		# plt.plot(tacho)
		# plt.plot(speed_eq)
		# plt.show()
		
		save_pickle('eq_' + os.path.basename(filepath_speed)[:-5] + '.pkl', speed_eq)
		
		
	
	elif config['mode'] == 'multi_to_pulse':
		print('Select signals tachos')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
		
		Speeds = []
		Deviations = []
		Kurtosis = []
		
		Speeds_max = []
		Deviations_max = []
		Kurtosis_max = []
		
		Speeds_min = []
		Deviations_min = []
		Kurtosis_min = []
		
		for filepath in Filepaths:
			vec = []
			tacho = load_signal(filepath, channel=config['channel_tacho'])
			tacho = butter_lowpass(x=tacho, fs=config['fs_tacho'], freq=50., order=3)

			tacho_new = []
			#for a to n
			# lim = 0.002
			# for P and so on
			lim = 0.01
			for i in range(len(tacho)-1):
				if (tacho[i] >= -lim and tacho[i+1] < -lim):
					tacho_new.append(1)
					vec.append(i)
				else:
					tacho_new.append(0)
			tacho_new.append(0)
			tacho_new = np.array(tacho_new)
			save_pickle('pulse_' + os.path.basename(filepath)[:-5] + '.pkl', tacho_new)
			
			speed = []
			for k in range(len(vec)-1):
				speed.append(5*60./((vec[k+1] - vec[k])/config['fs_tacho']))
			Speeds.append(np.mean(np.array(speed)))
			Deviations.append(np.std(np.array(speed)))
			Kurtosis.append(scipy.stats.kurtosis(np.array(speed)))
			
		Speeds_max.append(np.max(np.array(Speeds)))
		Deviations_max.append(np.max(np.array(Deviations)))
		Kurtosis_max.append(np.max(np.array(Kurtosis)))
		
		Speeds_min.append(np.min(np.array(Speeds)))
		Deviations_min.append(np.min(np.array(Deviations)))
		Kurtosis_min.append(np.min(np.array(Kurtosis)))

		
		mydict = {}			
		row_names = [os.path.basename(filepath) for filepath in Filepaths]
		mydict['1_RPM'] = Speeds
		mydict['2_STD'] = Deviations
		mydict['3_KUR'] = Kurtosis
		
		mydict['4_RPM_max'] = Speeds_max
		mydict['6_STD_max'] = Deviations_max
		mydict['8_KUR_max'] = Kurtosis_max
		
		mydict['5_RPM_min'] = Speeds_min
		mydict['7_STD_min'] = Deviations_min
		mydict['9_KUR_min'] = Kurtosis_min
		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter('Tacho_Out_' + config['name'] + '.xlsx')		
		DataFr.to_excel(writer, sheet_name='Sheet1')	
		
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
	config['fs_tacho'] = float(config['fs_tacho'])
	config['fs_signal'] = float(config['fs_signal'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


if __name__ == '__main__':
	main(sys.argv)
