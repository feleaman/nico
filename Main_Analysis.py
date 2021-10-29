# Main_Analysis.py
# Last updated: 19.03.2019 by Felix Leaman

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
from Kurtogram3 import Fast_Kurtogram_filters

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
from M_Wavelet import *
# from os import chdir
# plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
plt.rcParams['savefig.dpi'] = 500
plt.rcParams['savefig.format'] = 'jpeg'
 
from argparse import ArgumentParser


#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs']
InputsOpt_Defaults = {'power2':'OFF', 'plot':'ON', 'title_plot':None, 'file':'OFF', 'range':None, 'savemat':'OFF', 'calculate_features':'OFF', 'savedict':'OFF'}

def main(argv):

	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	invoke_signal(config)
	
	return 

def invoke_signal(config):

	channel = config['channel']
	fs = config['fs']
	
	if config['file'] == 'OFF':
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()
	else:
		filename = config['file']
	
	
	try:
		x = load_signal(filename, channel=config['channel'])
	except:
		# x = f_open_mat(filename, channel=config['channel'])
		# x, channel = f_open_mat_2(filename, channel=config['channel'])
		# x = read_pickle(filename)
		x = f_open_tdms(filename, channel)
	
	# point_index = filename.find('.')
	# extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]

	# if extension == 'mat':
		# x, channel = f_open_mat_2(filename, channel=config['channel'])
		# # x = f_open_mat(filename, channel=config['channel'])
		# x = np.ndarray.flatten(x)

	# elif extension == 'tdm': #tdms
		# x = f_open_tdms(filename, channel)
		# # x = f_open_tdms_2(filename)
	
	# elif extension == 'pkl': #tdms
		# x = read_pickle(filename)
		# # x = f_open_tdms_2(filename)


	# elif extension == 'txt': #tdms
		# x = np.loadtxt(filename)
	
	# else :
		# # x = read_pickle(filename)
		# x = f_open_tdms(filename, channel)
		# # print('open as pkl')
		# # x = read_pickle(filename)
		# # x = f_open_tdms_2(filename)
	filename = os.path.basename(filename) #changes from path to file
	print(filename)

	#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++


		
	dt = 1.0/fs
	

	magX, f, df = mag_fft_test(x, fs)
	# plt.plot(f, magX)
	plt.plot(x)
	plt.show()
	sys.exit()
	
	if config['power2'] == 'auto':
		n_points = 2**(max_2power(len(x)))
	elif config['power2'] == 'OFF':
		n_points = len(x)
	else:
		n_points = 2**config['power2']	
	x = x[0:n_points]
	
	if config['range'] != None:
		x = x[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]

	n = len(x)
	n_points = n
	tr = n*dt
	t = np.array([i*dt for i in range(n)])
	
	# xraw = x
	
	# x = x*1000./70.8 #37 dB schottland
	# x = x*1000./70.8
	# x = x*1000./141.25 #43 dB bochum laststufen
	
	# x = x*1000.
	
	#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

	config_analysis = {'WFM':True, 'FFT':False, 'PSD':False, 'STFT':False, 'STPSD':False, 'Cepstrum':False, 'Kurtogram':False, 'Histogram':False, 'AutoMean':False, 'Wavelet':False}

	config_fft = {'window':'hanning', 'harmonics':None}
	
	config_demod = {'analysis':False, 'mode':'hilbert', 'prefilter':['bandpass', [1000., 3000.], 3], 'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 500., 3]}
	# config_demod = {'analysis':True, 'mode':'butter', 'prefilter':['highpass', 50., 3], 'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 500., 3]}
	#When hilbert is selected, the other parameters are ignored

	config_diff = {'analysis':False, 'length':1, 'same':True}
	
	config_automean = {'type':'fixed_time', 'length':1.1589}
	# 0.7748269 # group1
	# 0.7994805 #group2
	# 0.7880184 #group3
	# 1.1589 #bochum 75dreh
	
	config_wavelet = {'wv_crit':'kurt_sen', 'level':8, 'wv_mother':'db6', 'wv_approx':'OFF'}

	
	config_stft = {'segments':100000, 'window':'boxcar', 'mode':'magnitude', 'log-scale':False, 'SPK':False, 'surface':False}	
	
	config_kurtogram = {'levels':5}

	config_stPSD = {'segments':100, 'window':'hanning', 'mode':'magnitude', 'log-scale':True}
	
	# config_denois = {'analysis':False, 'mode':'kurtogram', 'freq':50., 'level':7}
	config_denois = {'analysis':False, 'mode':'butter_highpass', 'freq':5.e3, 'level':7}


	config_other = {'squared':False}	
	
	config_autocorr = {'analysis':False}


	#++++++++++++++++++++++ SIGNAL DEFINITION ++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config_denois['analysis'] == True:
		print('with filter')
		if config_denois['mode'] == 'butter_highpass':
			print(len(x))
			x = butter_highpass(x=x, fs=config['fs'], freq=config_denois['freq'], order=3, warm_points=None)
		elif config_denois['mode'] == 'butter_bandpass':
			x = butter_bandpass(x=x, fs=config['fs'], freqs=config_denois['freq'], order=3, warm_points=None)
		elif config_denois['mode'] == 'butter_lowpass':
			x = butter_lowpass(x=x, fs=config['fs'], freq=config_denois['freq'], order=3, warm_points=None)
		elif config_denois['mode'] == 'kurtogram':
			lp, hp, max_kurt = Fast_Kurtogram_filters(x, config_denois['level'], config['fs'])
			x = butter_bandpass(x=x, fs=config['fs'], freqs=[lp, hp], order=3, warm_points=None)

		else:
			print('Error assignment denois')
	
	if config_other['squared'] == True:
		x = x**2.0
	
	
	
	#++++++++++++++++++++++ WAVELET+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config_analysis['Wavelet'] == True:
		if config_wavelet['wv_crit'] == 'kurt_sen':					
			x, best_lvl, fs = return_best_wv_level_idx(x=x, fs=fs, sqr='OFF', levels=config_wavelet['level'], mother_wv=config_wavelet['wv_mother'], crit='kurt_sen', wv_approx=config_wavelet['wv_approx'])
			
			# w_signal, best_lvl, new_fs = return_best_wv_level_idx(x=x, fs=fs, sqr='OFF', levels=config_wavelet['level'], mother_wv=config_wavelet['wv_mother'], crit='kurt_sen', wv_approx=config_wavelet['wv_approx'])				
		else:
			print('not implemented here!')
			sys.exit()
	else:
		best_lvl = '0'
		n = len(x)
		dt = 1./fs
		t = np.array([i*dt for i in range(n)])
		
		# from scipy.signal import hilbert
		# hx = hilbert(x)
		# pha = np.angle(hx)
		# pha = np.unwrap(pha) 
		# pha = diff_signal(x=pha, length_diff=1)
		# pha = pha / dt
		# pha = pha / (2*np.pi)
		# pha = np.absolute(pha)
		# pha = np.array(list(pha) + [pha[len(pha)-1]])
		# # pha = butter_lowpass(x=pha, fs=fs, freq=500., order=3, warm_points=None)
		# # pha = butter_bandpass(x=pha, fs=fs, freqs=[250, 370.], order=3, warm_points=None)
		# amp = np.absolute(hx)
		
		# sf = 1
		
		# for i in range(len(pha)):
			# pha[i] = np.rint(pha[i])		

		# min_freq = np.min(pha)
		# max_freq = np.max(pha)
		# new_pha = np.arange(start=min_freq, stop=max_freq+sf, step=sf)
		
		# dict_freq = {}
		# for element in new_pha:
			# dict_freq[str(element)] = 0.
		
		# for i in range(len(pha)):
			# dict_freq[str(pha[i])] += amp[i]
		
		# new_amp = []
		# for key, value in dict_freq.items():
			# new_amp.append(value)

		
		
		
		# plt.plot(new_pha, new_amp, 'g')	
		
		# plt.plot(t, pha, 'r')
		
	
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
	
	if config_autocorr['analysis'] == True:
		print('with autocorr')
		x = autocorr_fft(x)
		
	if config['savemat'] == 'ON':
		x = list(x)
		x = np.array(x)
		mydict={config['channel']:x}
		scipy.io.savemat(filename[:-5] + '.mat', mydict)
		print('Signal was saved as .mat. Analysis finalizes')
		sys.exit()
	
	if config['savedict'] == 'ON':
		# x = list(x)
		# x = np.array(x)
		# mydict={config['channel']:x}
		# scipy.io.savemat(filename[:-5] + '.mat', mydict)
		print('Signal was saved pik Analysis finalizes')
		# sys.exit()		
		mydict = {'x':x, 'best_lvl':best_lvl, 'fs':fs}
		save_pickle(config['channel'] + '_' + filename[:-5] + '.pkl', mydict)
		sys.exit()
		
	if config['calculate_features'] == 'ON':
		print('RMS: ', signal_rms(x))
		
	
	#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config_analysis['FFT'] == True:
		if config_fft['window'] == 'boxcar':
			magX, f, df = mag_fft(x, fs)
		elif config_fft['window'] == 'hanning':
			magX, f, df = mag_fft_hanning(x, fs)
		else:
			print('error window 88416')
			sys.exit()
	else:
		magX = None
		f = None
		df = None

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
		
	#+++++++++++++++++++++AUTOMEAN
	#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config_analysis['AutoMean'] == True:
		if config_automean['type'] == 'fixed_time':
			n_avg_points = int(config_automean['length']*fs)
			n_signal = len(x)
			n_prom = int(n_signal/n_avg_points)
			print('N° Prom = ', n_prom)
			avg_signal = np.zeros(n_avg_points)
			for i in range(n_prom):
				avg_signal = avg_signal + np.array(x[i*n_avg_points : (i+1)*n_avg_points])
			avg_signal = avg_signal / n_prom
			t_avg_signal = np.arange(n_avg_points)/fs
		else:
			print('no other type avalaible for AutoMean')
			
		
	
	#++++++++++++++++++++++ KURTOGRAM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	if config_analysis['Kurtogram'] == True:
		kurtogramX, tc, dtc = kurtogram(x, fs, config_kurtogram['levels'])

	
	
		# segments = config_stft['segments']
		# window = config_stft['window']
		# mode = config_stft['mode']		
		# w_stftX, w_f_stft, w_df_stft, w_t_stft = shortFFT(w_signal, new_fs, segments, window, mode)
		# if config_stft['log-scale'] == True:
			# stftX = np.log(stftX)
		
		
	#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	ylabel = 'Amplitude [g]'
	# ylabel = 'Drehzahl [Hz]'
	ylabel_fft = 'Magnitude [g]'
	
	if channel.find('LS') != -1 or channel.find('DG') != -1:
		ylabel = 'Amplitude'
		# ylabel = 'Drehzahl [Hz]'
		ylabel_fft = 'Magnitude'
		
	if config['plot'] == 'ON':
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
					
					# figure, ax_wfm = plt.subplots(nrows=2, ncols=1, 
					
					
					ax_wfm = fig[count].add_subplot(1,1,1)
					# ax_wfm.set_title('Corte de Carbón', fontsize=11)
					ax_wfm.plot(t, x)		
					# ax_wfm.set_xlabel('Time [s]', fontsize=12)
					# ax_wfm.set_ylabel('Kraft [kN]', fontsize=12)
					# ax_wfm.set_ylabel(ylabel, fontsize=13.5)
					ax_wfm.set_ylabel('Amplitude [mV]', fontsize=13.5)

					# plt.tight_layout()
					
					title = str(channel) + ' WFM ' + filename[:-4]
					title = title.replace('_', '-')
					# title = 'AE: continuous type'
					title = 'AE: burst type'
					
					# title = 'AE-1'
					ax_wfm.set_title(title, fontsize=18)

					ax_wfm.set_xlabel('Time [ms]', fontsize=13.5)

					
					ax_wfm.tick_params(axis='both', labelsize=12.5)
					
					# ax_wfm.set_xlim(left=1623, right=1628)
					# ax_wfm.set_ylim(bottom=-8, top=9)
					
					# ax_wfm.set_xlim(left=1460, right=1465)
					# ax_wfm.set_ylim(bottom=-2, top=2)
					

					ax_wfm.grid(axis='both')
					# # plt.show()
	
				elif element == 'FFT':
					ax_fft = fig[count].add_subplot(1,1,1)
					channel = channel.replace('_', '-')
					# ax_fft.set_title(channel + ' ' + element + ' ' + filename, fontsize=12)
					# title = str(channel) + ' ' + element + ' ' + filename
					
					title = str(channel) + ' FFT ' + filename[:-5]
					title = title.replace('_', '-')
					
					# title = 'AE-1'
					ax_fft.set_title(title, fontsize=13)
					

					
					if config_demod['analysis'] == True:
						color_fft = 'g'
						color_marker = 'r'
					else:
						color_fft = 'r'
						color_marker = 'g'
					
					ax_fft.plot(f, magX, color=color_fft)					
					# ax_fft.set_xlabel('Frequenz [Hz]', fontsize=12)
					
					# ax_fft.plot(f/1000., magX, 'r')					
					ax_fft.set_xlabel('Frequenz [Hz]', fontsize=13.5)
					# ax_fft.set_xlim(left=0, right=40)
					# ax_fft.set_ylim(bottom=0, top=0.6)
					ax_fft.grid(axis='both')
					ax_fft.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
					params = {'mathtext.default': 'regular' }          
					plt.rcParams.update(params)
					# ax_fft.set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
					ax_fft.set_ylabel(ylabel_fft, fontsize=13.5)
					ax_fft.tick_params(axis='both', labelsize=12.5)
					
					if config_fft['harmonics'] != None:
						n_harmonics = int((fs/2.)/config_fft['harmonics'])
						harmonics = np.arange(0, n_harmonics)*config_fft['harmonics']
						harmonics = harmonics[1:]
						# print(type(magX[0]))
						# print(type(magX))
						# print(magX.shape[0])
						# print(harmonics)
						# print(int(harmonics/df))
						
						idx_harmonics = []
						for i in harmonics:
							idx_harmonics.append(int(np.rint(i/df)))
							
							
						
						ax_fft.plot(harmonics, magX[idx_harmonics], marker='s', color=color_marker)
					
					# plt.tight_layout()
					# ax_fft.ticklabel_format(style='sci', scilimits=(-4, 4))
				elif element == 'PSD':
					ax_psd = fig[count].add_subplot(1,1,1)
					ax_psd.set_title(channel + ' ' + element)
					ax_psd.plot(f_psd, psdX)
					ax_psd.set_xlabel('Frequency Hz')
					ax_psd.set_ylabel('Amplitude V')
					# ax.set_xticklabels(['{:.}'.format(int(x)) for x in ax.get_xticks().tolist()])
				elif element == 'AutoMean':
					ax_am = fig[count].add_subplot(1,1,1)
					ax_am.plot(t_avg_signal/np.max(t_avg_signal), avg_signal)
					
					title = 'AutoMean' + str(channel) + ' ' + filename[:-5]
					title = title.replace('_', '-')					
					ax_am.set_title(title, fontsize=13)					
					# ax_am.set_title('AutoMean', fontsize=13)
					ax_am.tick_params(axis='both', labelsize=12)
					ax_am.set_ylabel('Amplitude [mV$^{2}$]', fontsize=13)
					ax_am.set_xlabel('Revolutions [-]', fontsize=13)
				
				# elif element == 'Wavelet':
					# ax_w = fig[count].add_subplot(1,1,1)
					
					# new_dt = 1./new_fs
					# w_t = np.arange(len(w_signal))*dt
					
					# ax_w.plot(t, x)
					
					# title = 'Wavelet' + str(channel) + ' ' + filename[:-5]
					# title = title.replace('_', '-')					
					# ax_w.set_title(title, fontsize=13)					
					# # ax_am.set_title('AutoMean', fontsize=13)
					# ax_w.tick_params(axis='both', labelsize=12)
					# ax_w.set_ylabel('Amplitude', fontsize=13)
					# ax_w.set_xlabel('Time [s]', fontsize=13)
					
					
					
					# # ax_stft = fig[count+1].add_subplot(1,1,1)						
					# # title = str(channel) + ' STFT ' + filename[:-5]
					# # title = title.replace('_', '-')
					
					# # ax_stft.set_title(title, fontsize=13)
				
					# # ax_stft.pcolormesh(t_stft, f_stft, stftX)
					# # ax_stft.set_xlabel('Dauer [s]', fontsize=13.5)
					# # ax_stft.set_ylabel('Frequenz [Hz]', fontsize=13.5)
					

					
					# # map = []
					# # vmax = max_cspectrum(stftX)				
					# # map.append(ax_stft.pcolormesh(t_stft, f_stft/1000., stftX, cmap='plasma', vmax=None))

					# # cbar = fig[count].colorbar(map[0], ax=ax_stft)
					
					# # cbar.set_label('log' + ' ' + ylabel_fft, fontsize=13.5)
						
						

					
					
					
				elif element == 'STFT':
					
					
					
					if config_stft['surface'] == True:
						from mpl_toolkits.mplot3d import Axes3D
						ax_stft = fig[count].add_subplot(111, projection='3d')
						tgato, fgato = np.meshgrid(t_stft, f_stft/1000.)
						
						# caca = ax.pcolormesh(t, fgato, MAP, cmap='hsv', vmax=vmax)
						# caca = ax.plot_surface(t, fgato, MAP, cmap='plasma', vmax=vmax)
						# caca = ax.plot_surface(t, fgato, MAP, cmap='plasma')
						caca = ax_stft.plot_surface(tgato, fgato, stftX)
					else:


						segments = config_stft['segments']
						window = config_stft['window']
						mode = config_stft['mode']
						
						stftX, f_stft, df_stft, t_stft = shortFFT(x, fs, segments, window, mode)
						ax_stft = fig[count].add_subplot(1,1,1)
						# ax_stft.set_title(channel + ' ' + element)
						
						
						title = str(channel) + ' STFT ' + filename[:-5]
						title = title.replace('_', '-')
						
						ax_stft.set_title(title, fontsize=13)
					
						ax_stft.pcolormesh(t_stft, f_stft, stftX)
						ax_stft.set_xlabel('Dauer [s]', fontsize=13.5)
						ax_stft.set_ylabel('Frequenz [Hz]', fontsize=13.5)
						
						# print(len(t_stft))
						# print(len(f_stft))
						# print(len(stftX))
						# print(len(stftX[0]))
						# a = input('pause')
						
						map = []
						vmax = max_cspectrum(stftX)				
						map.append(ax_stft.pcolormesh(t_stft, f_stft/1000., stftX, cmap='plasma', vmax=None))
						
						# ax_stft.ticklabel_format(style='sci', scilimits=(-2, 2))
						
						
						# indmax = np.argmax([max_cspectrum(CyclicSpectrum1), max_cspectrum(CyclicSpectrum2)])
						# fig[count].colorbar(map[0], ax=ax_stft, ticks=np.linspace(0, vmax, 5), format='%1.2e')
						cbar = fig[count].colorbar(map[0], ax=ax_stft)
						
						cbar.set_label('log' + ' ' + ylabel_fft, fontsize=13.5)
						
						
						# indmax = np.argmax([max_cspectrum(CyclicSpectrum1)])
						# fig[count].colorbar(ax=ax_stft)
						extent_ = [0, np.max(t), 0, 500]						
						mydict = {'map':stftX**2., 'extent':extent_}
						save_pickle('STFT_SimulatedAE2.pkl', mydict)
						
		

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
				
				elif element == 'Histogram':
					ax_hist = fig[count].add_subplot(1,1,1)
					# ax_stft.set_title(channel + ' ' + element)
					ax_hist.set_title(str(channel) + ' ' + element + '\n' + filename, fontsize=10)
					
					ax_hist.hist(x, bins=50)
					ax_hist.set_xlabel('Bins')
					ax_hist.set_ylabel('Frequency')
				
					print(np.mean(x))
					print(np.std(x))
				

			
			
			

				count = count + 1
			if config['plot'] == 'ON':
				# plt.tight_layout()
				plt.show()


	dict_out = {'time':t, 'signal':x, 'freq':f, 'mag_fft':magX, 'filename':filename, 'signal_raw':xraw}
	return dict_out
	# return t, x, f, magX, filename
	# return t, x, f, magX, xraw, filename

# plt.show()
# def read_parser_old(argv, Inputs, InputsOpt_Defaults):
	# try:
		# Inputs_opt = [key for key in InputsOpt_Defaults]
		# Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		# parser = ArgumentParser()
		# for element in (Inputs + Inputs_opt):
			# print(element)
			# if element == 'no_element':
				# parser.add_argument('--' + element, nargs='+')
			# else:
				# parser.add_argument('--' + element, nargs='?')
		# print(parser.parse_args())
		# args = parser.parse_args()

		
	# except:
		# # args = argv
		# arguments = [element for element in argv if element[0:2] == '--']
		# values = [element for element in argv if element[0:2] != '--']

		# # from argparse import ArgumentParser
		# # from ArgumentParser import Namespace
		# parser = ArgumentParser()
		# for element in arguments:
			# parser.add_argument(element)

		# args = parser.parse_args(argv)

		# # print(test)
		# # sys.exit()
		
	# config = {}	
		
	# for element in Inputs:
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Required:', element)

	# for element, value in zip(Inputs_opt, Defaults):
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Default ' + element + ' = ', value)
			# config[element] = value
	
	# #Type conversion to float
	# if config['power2'] != 'auto' and config['power2'] != 'OFF':
		# config['power2'] = int(config['power2'])
	# config['fs'] = float(config['fs'])
	

	
	# # config['fscore_min'] = float(config['fscore_min'])
	# #Type conversion to int	
	# # Variable conversion
	# # print('caca')
	# return config

def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'range':
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
	config['fs'] = float(config['fs'])
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	if config['power2'] != 'auto' and config['power2'] != 'OFF':
		config['power2'] = int(config['power2'])
	
	
	
	return config

if __name__ == '__main__':
	main(sys.argv)
