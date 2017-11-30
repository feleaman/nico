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
Inputs = ['channel', 'fs']
InputsOpt_Defaults = {'power2':'OFF'}

def main(argv):



	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	channel = config['channel']
	fs = config['fs']

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
		x = f_open_tdms(filename, channel)
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
		
	dt = 1.0/fs
	
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
	
	# sys.exit()
	
	n = len(x)
	n_points = n
	tr = n*dt
	t = np.array([i*dt for i in range(n)])
	
	
	# x = to_dBAE(x, 43.)
	# x = x * 1000.
	#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++

	config_analysis = {'WFM':True, 'FFT':True, 'PSD':True, 'STFT':True, 'STPSD':False, 'Cepstrum':False, 'CyclicSpectrum':False}

	config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['highpass', 140.0e3, 3], 
	'rectification':'only_positives', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3]}
	#When hilbert is selected, the other parameters are ignored

	config_diff = {'analysis':False, 'length':1, 'same':True}

	config_stft = {'segments':1000, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}

	config_stPSD = {'segments':200, 'window':'hanning', 'mode':'magnitude', 'log-scale':False}
	
	config_denois = {'analysis':False, 'mode':'butter_bandpass', 'freq':[320.0e3, 380.e3]}
	
	config_CyclicSpectrum = {'segments':200, 'freq_range':[100.0e3, 400.0e3], 'window':'hanning', 'mode':'magnitude', 'log':False, 'off_PSD':True, 'kHz':True, 'warm_points':None}

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
			x = butter_highpass(x=x, fs=config['fs'], freq=config_denois['freq'], order=3, warm_points=None)
		elif config_denois['mode'] == 'butter_bandpass':
			x = butter_bandpass(x=x, fs=config['fs'], freqs=config_denois['freq'], order=3, warm_points=None)

		else:
			print('Error assignment denois')
	
	
	
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
				ax_wfm.set_title(str(channel) + ' ' + element + '\n' + filename, fontsize=10)
				ax_wfm.set_label('AE')
				ax_wfm.plot(t, x)
				ax_wfm.set_xlabel('Time s')
				ax_wfm.set_ylabel('Amplitude V')
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
				ax_fft.set_ylabel('Amplitude V')
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
				
				
				map = []
				vmax = max_cspectrum(CyclicSpectrum)				
				map.append(ax_ciclic.pcolormesh(a_CyclicSpectrum, f_CyclicSpectrum/1000., CyclicSpectrum, cmap='plasma', vmax=vmax))
				
				# ax_stft.ticklabel_format(style='sci', scilimits=(-2, 2))
				
				
				# indmax = np.argmax([max_cspectrum(CyclicSpectrum1), max_cspectrum(CyclicSpectrum2)])
				fig[count].colorbar(map[0], ax=ax_ciclic, ticks=np.linspace(0, vmax, 5), format='%1.2e')
				
				
				# indmax = np.argmax([max_cspectrum(CyclicSpectrum1)])
				# fig[count].colorbar(ax=ax_stft)
			
			
			

			count = count + 1

	plt.show()



	# count = 0
	# for element in config_analysis:
		# if config_analysis[element] == True:
			# count = count + 1
			# plt.figure(count)
			# plt.title(channel + ' ' + element)
			# plt.suptitle(filename, fontsize=10)
			# if element == 'WFM':
				# plt.xlabel('Time s')
				# plt.ylabel('Amplitude')
				# plt.plot(t, x)
			# elif element == 'FFT':
				# plt.xlabel('Frequency Hz')
				# plt.ylabel('Amplitude')
				# plt.plot(f, magX, 'r')
			# elif element == 'PSD':
				# plt.xlabel('Frequency Hz')
				# plt.ylabel('Amplitude')
				# plt.plot(f_psd, psdX, 'g')
			# elif element == 'STFT':
				# plt.xlabel('Time s')
				# plt.ylabel('Frequency Hz')
				# plt.pcolormesh(t_stft, f_stft, stftX)	
			# elif element == 'STPSD':
				# plt.xlabel('Time s')
				# plt.ylabel('Frequency Hz')
				# plt.pcolormesh(t_stPSD, f_stPSD, stPSDX)
			# elif element == 'Cepstrum':
				# plt.xlabel('Quefrency s')
				# plt.ylabel('Amplitude')
				# plt.plot(tc, cepstrumX)
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
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


if __name__ == '__main__':
	main(sys.argv)
