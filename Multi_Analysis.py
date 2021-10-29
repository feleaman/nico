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
from m_plots import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


# from os import chdir
# plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'png'
 
#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs']
InputsOpt_Defaults = {'power2':'OFF', 'plot':'OFF', 'title_plot':None, 'file':None}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	if config['file'] == None:
		root = Tk()
		root.withdraw()
		root.update()
		print('+++Select Filepaths 1')
		Filepaths1 = filedialog.askopenfilenames()
		root.destroy()
	else:
		Filepaths1 = [config['file']]
	print(Filepaths1)
	
	Signals1 = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths1]
	n1 = len(Signals1)
	
	if config['file'] == None:
		root = Tk()
		root.withdraw()
		root.update()
		print('Select Filepaths 2')
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
	else:
		Filepaths2 = [config['file']]
	print(Filepaths2)
	
	Signals2 = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths2]
	n2 = len(Signals2)
	
	fig1, ax1 = plt.subplots(nrows=n1, ncols=1, sharex=True, sharey=True)
	ax1[0].set_title('Filepaths 1')
	for k in range(n1):
		ax1[k].plot(Signals1[k])
	
	
	fig2, ax2 = plt.subplots(nrows=n2, ncols=1, sharex=True, sharey=True)
	ax2[0].set_title('Filepaths 2')
	for k in range(n2):
		ax2[k].plot(Signals2[k])
	
	plt.show()
		
	
	
	return 

def invoke_signal(config):

	channel = config['channel']
	fs = config['fs']
	
	if config['file'] == None:
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
	else:
		Filepaths = [config['file']]
	
	
	#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++

	config_analysis = {'WFM':True, 'FFT':True, 'PSD':False, 'STFT':False, 'STPSD':False, 'Cepstrum':False, 'CyclicSpectrum':False, 'Kurtogram':False}

	# config_demod = {'analysis':True, 'mode':'butter', 'prefilter':['highpass', 20.e3, 3],
	# 'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 2000., 3]}
	
	config_demod = {'analysis':True, 'mode':'hilbert', 'prefilter':['bandpass', [100.e3, 150.e3], 3],
	'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 1000., 3]}


	config_diff = {'analysis':False, 'length':1, 'same':True}

	config_stft = {'segments':500, 'window':'boxcar', 'mode':'magnitude', 'log-scale':False, 'SPK':True}
	
	
	config_kurtogram = {'levels':5}

	config_stPSD = {'segments':500, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}
	
	# config_denois = {'analysis':True, 'mode':'butter_highpass', 'freq':70.e3}
	# config_denois = {'analysis':False, 'mode':'butter_highpass', 'freq':20.e3}
	config_denois = {'analysis':True, 'mode':'butter_bandpass', 'freq':[95.e3, 140.e3]}

	config_other = {'squared':True}
	
	config_denois2 = {'analysis':False, 'mode':'butter_bandstop', 'freq':[439.e3, 442.e3]}
	
	config_autocorr = {'analysis':False}
	
	config_CyclicSpectrum = {'segments':150, 'freq_range':[50.3, 200.e3], 'window':'boxcar', 'mode':'magnitude', 'log':False, 'off_PSD':True, 'kHz':False, 'warm_points':None}

	Signals_Dicts = []
	for filepath in Filepaths:
		x = load_signal(filepath, channel=config['channel'])

		
		filename = os.path.basename(filepath) #changes from path to file
		print(filename)

		#++++++++++++++++++++++ SAMPLING 
		dt = 1.0/fs


		n = len(x)
		n_points = n
		tr = n*dt
		t = np.array([i*dt for i in range(n)])
		
		
		# x = x*1000. / 141.3


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

			else:
				print('Error assignment denois')
		
		if config_other['squared'] == True:
			x = x**2.0
		
		
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
		
		
		# print(type(x))
		# sys.exit()
		# x = list(x)
		# x = x[0:int(len(x)/2)]
		# x = np.array(x)
		# mydict={'x':x}
		# scipy.io.savemat('x.mat', mydict)
		# sys.exit()
		
		
		if config_denois2['analysis'] == True:
			print('with filter 2')
			if config_denois2['mode'] == 'butter_bandstop':
				x = butter_bandstop(x=x, fs=config['fs'], freqs=config_denois2['freq'], order=3, warm_points=None)
			else:
				print('Error assignment denois')
				sys.exit()
		
		
		if config_autocorr['analysis'] == True:
			print('with autocorr')
			x = autocorr_fft(x)
		
		# print('!!!')
		# print('rms')
		# x = butter_highpass(x=x, fs=config['fs'], freq=280., order=3, warm_points=None)
		# x = butter_lowpass(x=x, fs=config['fs'], freq=340., order=3, warm_points=None)

		# print(signal_rms(x))
		# print('!!!')
		#++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config_analysis['FFT'] == True:
			magX, f, df = mag_fft(x, fs)
			print(len(magX))
		else:
			magX = None
			f = None
			df = None

		# test = fourier_filter(x=x, fs=config['fs'], type='bandpass', freqs=[300., 320.])
		# plt.plot(t, test)
		# plt.show()
		#++++++++++++++++++++++ STFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config_analysis['STFT'] == True:
			segments = config_stft['segments']
			window = config_stft['window']
			mode = config_stft['mode']
			
			stftX, f_stft, df_stft, t_stft = shortFFT(x, fs, segments, window, mode)
			if config_stft['log-scale'] == True:
				stftX = np.log(stftX)
			
			
			if config_stft['SPK'] == True:
				# pepo = [list(i) for i in zip(*stftX)]
				
				# plt.plot(f_stft, pepo[1])
				# plt.show()
				
				# print(len(stftX))
				
				# print(len(f_stft))
				# print(len(t_stft))
				# sys.exit()
				
				SPK = [scipy.stats.kurtosis(stftX[i], fisher=False) for i in range(len(stftX))]
				
				plt.plot(f_stft/1000., SPK, color='g')
				plt.show()
						


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
			
		
		#++++++++++++++++++++++ KURTOGRAM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if config_analysis['Kurtogram'] == True:
			kurtogramX, tc, dtc = kurtogram(x, fs, config_kurtogram['levels'])

			
		#++++++++++++++++++++++ CICLO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if config_analysis['CyclicSpectrum'] == True:
			segments = config_CyclicSpectrum['segments']
			window = config_CyclicSpectrum['window']
			mode = config_CyclicSpectrum['mode']
			freq_range = config_CyclicSpectrum['freq_range']
			
			CyclicSpectrum, a_CyclicSpectrum, f_CyclicSpectrum = Cyclic_Spectrum2(x=x, fs=fs, segments=segments, freq_range=freq_range, warm_points=config_CyclicSpectrum['warm_points'])
			if config_CyclicSpectrum['off_PSD'] == True:
				for i in range(len(CyclicSpectrum)):
					CyclicSpectrum[i][0] = 0.

			if config_CyclicSpectrum['log'] == True:
				CyclicSpectrum = np.log1p(CyclicSpectrum)
			
			CyclicSpectrum = list(np.nan_to_num(CyclicSpectrum))
			
			print(CyclicSpectrum)
		
		# analysis_dict = {'t':t, 'wfm':x, 'fft':magX, 'f':f, 'filename':filename, 'f_psd':f_psd, 'psd':psdX}
		analysis_dict = {'t':t, 'wfm':x, 'fft':magX, 'f':f, 'filename':filename, 'filepath':filepath}
		Signals_Dicts.append(analysis_dict)
	
	return Signals_Dicts

def analysis_plot(analysis_dict):
	
	#++++++++++++++++++++++ MULTI PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
					
					# figure, ax_wfm = plt.subplots(nrows=2, ncols=1, sharex=True)					
					# # ax_wfm = fig[count].add_subplot(2,1,2)					
					# # ax_raw = fig[count].add_subplot(2,1,1, sharex=ax_wfm)
					# # ax_wfm[0].set_title(str(channel) + ' ' + element + ' ' + filename, fontsize=11)
					# title = str(channel) + ' ' + element + ' ohne Filtern'
					# title = title.replace('_', '-')
					# ax_wfm[0].set_title(title, fontsize=11)
					# ax_wfm[0].plot(t, xraw)
					# params = {'mathtext.default': 'regular' }          
					# plt.rcParams.update(params)
					# ax_wfm[0].set_ylabel('Amplitude [m$V_{in}$]', fontsize=11)
					
					
					ax_wfm = fig[count].add_subplot(1,1,1)
					# ax_wfm.set_title('Corte de Carbón', fontsize=11)
					ax_wfm.plot(t, x)		
					# ax_wfm.set_xlabel('Time [s]', fontsize=12)
					# ax_wfm.set_ylabel('Kraft [kN]', fontsize=12)
					ax_wfm.set_ylabel('Amplitude [V]', fontsize=13)

					# plt.tight_layout()
					
					title = str(channel) + ' ' + filename[:-5]
					title = title.replace('_', '-')
					
					title = 'AE-1'
					ax_wfm.set_title(title, fontsize=13)
					
					# ax_wfm.set_title(str(channel) + ' ' + element + '\n' + filename, fontsize=10)
					# ax_wfm.set_title('Coal Cutting', fontsize=11)
					

					# ax_wfm[1].set_label('AE')
					# ax_wfm[1].plot(t, x)
					ax_wfm.set_xlabel('Time [s]', fontsize=13)
					# # ax_wfm.set_xlabel('Zeit (s)', fontsize=12)
					# # ax_wfm.set_xlabel('Tiempo (s)', fontsize=11)

					# params = {'mathtext.default': 'regular' }          
					# plt.rcParams.update(params)
					# ax_wfm[1].set_ylabel('Amplitude [m$V_{in}$]', fontsize=11)
					# # ax_wfm.set_ylabel('Amplitud (V)', fontsize=11)
					# # ax_wfm.set_ylabel('Amplitude (V)', fontsize=11)
					# plt.tight_layout()
					
					ax_wfm.tick_params(axis='both', labelsize=12)
					
					
					# ax_wfm.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
					# params = {'mathtext.default': 'regular' }          
					# plt.rcParams.update(params)
					
					# ax_wfm.tick_params(axis='both', labelsize=11)
					# ax_wfm.text(0.008, 0.9*np.max(x), 'RMS ' + "{:.2E}".format(Decimal(str(burst_rms))), fontsize=12)
					# ax_wfm.set_xticks(np.arange(0, n_points*dt, 0.001))
					# plt.grid()
					# fig, ax = plt.subplots(nrows=1, ncols=1)
					# ax.set_xticks(np.arange(0, n_points*dt, window_points*dt))
					# ax.plot(t, x1)
					
					# fg = 40.2
					# ax_wfm.set_xticks(np.arange(0, tr, 1./fg))
					# ax.plot(t, x1)
					# plt.grid()
					
					# plt.show()
					
					
					
					
					# ax_wfm = fig[count].add_subplot(1,1,1)
					
					# if config['title_plot'] != None:
						# title = config['title_plot']
						# title = title.replace('_', ' ')
						# ax_wfm.set_title(title, fontsize=12)
					# else:
						# ax_wfm.set_title(channel + ' ' + element + ' ' + filename, fontsize=10)
					
					# ax_wfm.plot(t, x*1000.)					
					# ax_wfm.set_xlabel('Time [s]', fontsize=12)

					
					# # ax_wfm.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
					# params = {'mathtext.default': 'regular' }          
					# plt.rcParams.update(params)
					# # ax_wfm.set_ylabel('Amplitude [mV]', fontsize=12)
					
					# fig[count].text(0.025, 0.5, 'Amplitude [mV]', ha='center', va='center', rotation='vertical', fontsize=12)
					
					# ax_wfm.tick_params(axis='both', labelsize=11)
					
					# # plt.tight_layout()
					# # ax_wfm.set_ylim(-400, 400)
					# # ax_wfm.set_ylim(-1800, 1800)
					# # fig[count].set_size_inches(20, 20)
					
					# # plt.show()

				elif element == 'FFT':
					ax_fft = fig[count].add_subplot(1,1,1)
					channel = channel.replace('_', '-')
					# ax_fft.set_title(channel + ' ' + element + ' ' + filename, fontsize=12)
					title = str(channel) + ' ' + element + ' ' + filename
					
					title = config['title_plot']
					# title = title.replace('_', ' ')
					title = 'AE-1 FFT-Envelope'
					ax_fft.set_title(title, fontsize=13)
					# ax_fft.set_title('Reference: 1500 [RPM] / 80% Load', fontsize=12)
					
					ax_fft.plot(f, magX, 'r')					
					# ax_fft.set_xlabel('Frequenz [Hz]', fontsize=12)
					
					# ax_fft.plot(f/1000., magX, 'r')					
					ax_fft.set_xlabel('Frequency [Hz]', fontsize=13)
					# ax_fft.set_xlim(left=0, right=50)
					
					ax_fft.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
					params = {'mathtext.default': 'regular' }          
					plt.rcParams.update(params)
					# ax_fft.set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
					ax_fft.set_ylabel('Amplitude [V]', fontsize=13)
					ax_fft.tick_params(axis='both', labelsize=12)
					# plt.tight_layout()
					# ax_fft.ticklabel_format(style='sci', scilimits=(-4, 4))
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
					
					# print(len(t_stft))
					# print(len(f_stft))
					# print(len(stftX))
					# print(len(stftX[0]))
					# a = input('pause')
					
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
				

			
			
			

				count = count + 1
			if config['plot'] == 'ON':
				plt.show()


	dict_out = {'time':t, 'signal':x, 'freq':f, 'mag_fft':magX, 'filename':filename, 'signal_raw':xraw}
	return dict_out
	# return t, x, f, magX, filename
	# return t, x, f, magX, xraw, filename

# plt.show()
def fft_prom_dict(analysis_dict_1):
	magX_1 = np.zeros(len(analysis_dict_1[0]['fft']))
	count = 0
	for element in analysis_dict_1:
		magX_1 += element['fft']
		count += 1
	magX_1 = magX_1/count
	return magX_1

def psd_prom_dict(analysis_dict_1):
	magX_1 = np.zeros(len(analysis_dict_1[0]['psd']))
	count = 0
	for element in analysis_dict_1:
		magX_1 += element['psd']
		count += 1
	magX_1 = magX_1/count
	return magX_1

def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'no_element':
				parser.add_argument('--' + element, nargs='+')
			else:
				parser.add_argument('--' + element, nargs='?')
		print(parser.parse_args())
		args = parser.parse_args()

		
	except:
		# args = argv
		arguments = [element for element in argv if element[0:2] == '--']
		values = [element for element in argv if element[0:2] != '--']

		# from argparse import ArgumentParser
		# from ArgumentParser import Namespace
		parser = ArgumentParser()
		for element in arguments:
			parser.add_argument(element)

		args = parser.parse_args(argv)

		# print(test)
		# sys.exit()
		
	config = {}	
		
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float
	if config['power2'] != 'auto' and config['power2'] != 'OFF':
		config['power2'] = int(config['power2'])
	config['fs'] = float(config['fs'])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	print('caca')
	return config


if __name__ == '__main__':
	main(sys.argv)
