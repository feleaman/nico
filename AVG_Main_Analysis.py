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
from M_Wavelet import *

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
from m_plots import *

from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


# from os import chdir
# plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpeg'
 
#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'channel', 'fs']
InputsOpt_Defaults = {'mypaths':None, 'wv_deco':'OFF', 'wv_mother':'db1', 'wv_crit':'kurt_sen', 'wv_approx':'OFF', 'db_out':'OFF', 'units':'v', 'extension':'dms', 'filter':'OFF', 'freq_lp':'OFF', 'freq_hp':'OFF', 'sqr_envelope':'OFF', 'demodulation':'OFF', 'output':'plot', 'name':'auto', 'level':8, 'range':None, 'idx_levels':None}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == '6_plot':		
		analysis_dict_1 = invoke_signal(config)
		analysis_dict_2 = analysis_dict_1
		analysis_dict_3 = analysis_dict_1


		magX_1 = fft_prom_dict(analysis_dict_1)
		magX_2 = fft_prom_dict(analysis_dict_2)
		magX_3 = fft_prom_dict(analysis_dict_3)

		
		f_1 = analysis_dict_1[0]['f']
		f_2 = analysis_dict_2[0]['f']
		f_3 = analysis_dict_3[0]['f']/1.023077
		
		# inis = [100.e3, 200.e3, 300.e3, 400.e3]
		
		# style = {'xlabel':'Frequency [Hz]', 'ylabel':'Magnitude [mV]', 'legend':[None], 'title':'Relative counting', 'customxlabels':None, 'xlim':[0.0, 50.], 'ylim':[0, 1], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200]}
		# data = {'x':[f_1], 'y':[magX_1]}	
		# plot1_thesis(data, style)
		
		# from matplotlib import font_manager
		# font_manager.weight_dict['roman'] = 500
		
		# style = {'xlabel':'Frequency [Hz]', 'ylabel':'Magnitude [mV]', 'legend':[None], 'title':'Relative counting', 'customxlabels':None, 'xlim':[260, 360], 'ylim':[0, 1], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200]}
		# data = {'x':[f_1], 'y':[magX_1]}	
		# plot1_thesis(data, style)
		
		style = {'xlabel':'Frequency [Hz]', 'ylabel':'Magnitude [mV]', 'legend':['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10'], 'title':[None, None, None, None, None, None], 'customxlabels':None, 'xlim':[[0, 60], [0, 60], [0, 60], [280, 340], [280, 340], [280, 340]], 'ylim':[[0, 0.006], [0, 0.006], [0, 0.006], [0, 0.03], [0, 0.03], [0, 0.03]], 'color':[None, None, None, None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200]}
		data = {'x':[f_1, f_2, f_3, f_1, f_2, f_3], 'y':[magX_1, magX_2, magX_3, magX_1, magX_2, magX_3]}
		plot6_thesis_big(data, style)
		
	elif config['mode'] == '1_avg_spectrum':


		analysis_dict = invoke_signal(config)
		print(len(analysis_dict))
		# sys.exit()
		list_levels = [analysis_dict[i]['best_lvl'] for i in range(len(analysis_dict)) ]
		print(list_levels)
		# sys.exit()
		config['list_levels'] = list_levels

		
		magX = fft_prom_dict(analysis_dict)
		# magX = fft_prom_eq_dict(analysis_dict)
		
		f = analysis_dict[0]['f']
		
		
		
		plt.plot(f, magX)
		plt.show()
		# lens = []
		# for i in range(len(analysis_dict)):
			# lens.append(len(analysis_dict[i]['fft']))
		# idx = np.argmax(np.array(lens))		
		# f = analysis_dict[idx]['f']
		
		# name = 'AvgFFT_' + analysis_dict[0]['filename'][:-5] + '_' + config['name']
		name = config['name']

		mydic = {'fft':magX, 'f':f}
		save_pickle(name + '.pkl', mydic)
		save_pickle('config_' + name + '.pkl', config)
	
	elif config['mode'] == 'plot_avg_spectrum':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		mydict = read_pickle(filepath)
		magX = mydict['fft']
		# magX = 20*np.log10(magX*1000.)
		f = mydict['f']
		plt.plot(f, magX)
		plt.show()
		sys.exit()
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		# fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
		# i = 0
		# for filepath in Filepaths:
			# mydict = read_pickle(filepath)
			# magX = mydict['fft']
			# f = mydict['f']
			# ax[i].plot(f, magX)
			# print(os.path.basename(filepath))
			# i += 1
		# plt.show()
		
		
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		# cont = 0
		# fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
		# for i in range(3):
			# for j in range(4):
				# mydict = read_pickle(Filepaths[cont])
				# magX = mydict['fft']
				# f = mydict['f']
				# ax[i][j].plot(f, magX)
				# print(os.path.basename(Filepaths[cont]))
				# cont += 1
		# plt.show()
		
		print('select WITH fault')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_1 = filedialog.askopenfilenames()
		root.destroy()
		
		print('select NO fault')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_2 = filedialog.askopenfilenames()
		root.destroy()
		
		cont = 0
		fig, ax = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='row')
		for i in range(4):
			for j in range(3):
				mydict = read_pickle(Filepaths_1[cont])
				magX = mydict['fft']
				f = mydict['f']
				ax[i][j].plot(f, magX, '-or', markersize=2)
				print(os.path.basename(Filepaths_1[cont]))
				
				mydict = read_pickle(Filepaths_2[cont])
				magX = mydict['fft']
				f = mydict['f']
				ax[i][j].plot(f, magX, '-ob', markersize=2)
				print(os.path.basename(Filepaths_2[cont]))
				
				ax[i][j].set_xlim(left=0, right=750)
				
				cont += 1
		plt.show()
		


	
	return 

def invoke_signal(config):

	channel = config['channel']
	fs = config['fs']
	
	if config['mypaths'] == None:
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
	else:
		Filepaths = config['mypaths']
	

	# #++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++


	# config_demod = {'analysis':True, 'mode':'hilbert', 'prefilter':['bandpass', [2.e3, 3.e3], 3],
	# 'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 2000., 3]}
	
	# config_denois = {'analysis':False, 'mode':'butter_highpass', 'freq':20.e3}

	# config_other = {'squared':False}


	Signals_Dicts = []
	Filenames = []
	for filepath in Filepaths:
		# x = load_signal(filepath, channel=config['channel'])
		# print(filepath)
		# b = input('pause b ...')
		signal = load_signal(filepath, channel=config['channel'])
		
		if config['range'] != None:
			signal = signal[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		
		if config['channel'].find('AE') != -1:
		
			if config['db_out'] != 'OFF':
				if config['db_out'] == 37:
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
		else:
			print('Channel multiplied by 10 to obtain g')
			signal = signal*1.
		
		filename = os.path.basename(filepath) #changes from path to file
		Filenames.append(filename)
		print(filename)
		
		# print('++++++++++++++++++wwwwwwwwwwwww')
		# Wavelet
		if config['wv_deco'] == 'DWT':
			if config['wv_crit'] != 'avg_mpr':
				signal, best_lvl, new_fs = return_best_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'])
				print('best level = ', best_lvl)
			else:
				print('not implemented here!!')
				sys.exit()
		
		elif config['wv_deco'] == 'iDWT':
			if config['wv_crit'] != 'avg_mpr':
				signal, best_lvl, new_fs = return_best_inv_wv_level_idx(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_crit'], wv_approx=config['wv_approx'])
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
				signal, best_lvl, new_fs = return_best_wv_level_idx_PACKET(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit='kurt', wv_approx=config['wv_approx'])
				print('best level = ', best_lvl)
			elif config['wv_crit'] == 'kurt_sen':
				signal, best_lvl, new_fs = return_best_wv_level_idx_PACKET(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit='kurt_sen', wv_approx=config['wv_approx'])
				print('best level = ', best_lvl)
			elif config['wv_crit'] == 'mpr':
				# signal, best_lvl, new_fs = return_best_wv_level_idx_PACKET(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit=config['wv_mother'], wv_approx=config['wv_approx'], freq_values=[36.], freq_range=[26.,46.])
				
				signal, best_lvl, new_fs = return_best_wv_level_idx_PACKET(x=signal, fs=config['fs'], sqr=config['sqr_envelope'], levels=config['level'], mother_wv=config['wv_mother'], crit='mpr', wv_approx=config['wv_approx'], freq_values=[36.], freq_range=[31.,41.])
				# freq_values=[36.], freq_range=[31.,41.]
				# freq_values=[30.], freq_range=[25.,35.]
				print('best level = ', best_lvl)
		
		# plt.plot(signal)
		# plt.show()
		
		# mag, f, df = mag_fft(signal, new_fs)
		# env = hilbert_demodulation(mag)
		
		# plt.plot(f, env, color='green')
		# plt.show()
		# sys.exit()
		
		#Denois
		if config['filter'] != 'OFF':
			signal = multi_filter(signal, config, filename=filename)
		#Squared	
		if config['sqr_envelope'] == 'ON':
			signal = signal**2.0
			print('squared!')
		#Demodulation
		if config['demodulation'] == 'ON':
			signal = hilbert_demodulation(signal)
			print('demodulation!')			
		#FFT
		if config['wv_deco'] == 'DWT':
			print('info: wavelet affects FFT fs for DWT')
			magX, f, df = mag_fft(x=signal, fs=new_fs) # ONLY WAVELET!
			dt_ = 1./new_fs
		else:
			magX, f, df = mag_fft(x=signal, fs=config['fs'])
			dt_ = 1./config['fs']
		
		
		# #++++++++++++++++++++++ SAMPLING 
		# dt = 1.0/fs
		# fig, ax = plt.subplots(nrows=2, ncols=1)
		# t = np.array([i*dt_ for i in range(len(signal))])		
		# ax[0].plot(t, signal)
		# ax[1].plot(f, magX)
		# plt.show()
		# n = len(x)
		# n_points = n
		# tr = n*dt
		# t = np.array([i*dt for i in range(n)])
		
		
		# x = x*1000.
		# x = x/70.8
		
		

		# #++++++++++++++++++++++ SIGNAL DEFINITION ++++++++++++++++++++++++++++++++++++++++++++++++++++++
		# if config_denois['analysis'] == True:
			# print('with filter')
			# if config_denois['mode'] == 'butter_highpass':
				# print(len(x))
				# x = butter_highpass(x=x, fs=config['fs'], freq=config_denois['freq'], order=3, warm_points=None)
			# elif config_denois['mode'] == 'butter_bandpass':
				# x = butter_bandpass(x=x, fs=config['fs'], freqs=config_denois['freq'], order=3, warm_points=None)
			# elif config_denois['mode'] == 'butter_lowpass':
				# x = butter_lowpass(x=x, fs=config['fs'], freq=config_denois['freq'], order=3, warm_points=None)

			# else:
				# print('Error assignment denois')
		
		# if config_other['squared'] == True:
			# x = x**2.0
		
		
		# if config_demod['analysis'] == True:
			# print('with demod')
			# if config_demod['mode'] == 'hilbert':
				# x = hilbert_demodulation(x)
			# elif config_demod['mode'] == 'butter':
				# x = butter_demodulation(x=x, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
				# type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
			# else:
				# print('Error assignment demodulation')


		# #++++++++++++++++++++++ FFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		# if config_analysis['FFT'] == True:
			# magX, f, df = mag_fft(x, fs)
			# print(len(magX))
		# else:
			# magX = None
			# f = None
			# df = None

		# # test = fourier_filter(x=x, fs=config['fs'], type='bandpass', freqs=[300., 320.])
		# # plt.plot(t, test)
		# # plt.show()
		# #++++++++++++++++++++++ STFT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if config['wv_deco'] == 'OFF':
			best_lvl = 0
		
		analysis_dict = {'wfm':signal, 'fft':magX, 'f':f, 'filename':filename, 'best_lvl':best_lvl}
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

def fft_prom_eq_dict(analysis_dict_1):
	lens = []
	for i in range(len(analysis_dict_1)):
		lens.append(len(analysis_dict_1[i]['fft']))
	maxlens = np.max(np.array(lens))
	
	magX_1 = np.zeros(maxlens)
	count = 0
	for element in analysis_dict_1:
		
		if len(element['fft']) == maxlens:
			magX_1 += element['fft']
		elif len(element['fft']) < maxlens:
			data_old = np.array(element['fft'])
			x_data_old = np.linspace(start=0., stop=1., num=len(data_old))
			x_data_new = np.linspace(start=0., stop=1., num=maxlens)
			data_new = np.interp(x=x_data_new, xp=x_data_old, fp=data_old)
			magX_1 += data_new
		else:
			print('fatal error 78222')
			sys.exit()
		# magX_1 += element['fft']
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
			if element == 'no_element' or element == 'mypaths' or element == 'range' or element == 'idx_levels':
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
	# if config['power2'] != 'auto' and config['power2'] != 'OFF':
		# config['power2'] = int(config['power2'])
	config['fs'] = float(config['fs'])
	if config['db_out'] != 'OFF':
		config['db_out'] = int(config['db_out'])
	if config['freq_lp'] != 'OFF':
		config['freq_lp'] = float(config['freq_lp'])
	
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	if config['idx_levels'] != None:
		for i in range(len(config['idx_levels'])):
			config['idx_levels'][i] = int(config['idx_levels'][i])

	
	if config['freq_hp'] != 'OFF':
		config['freq_hp'] = float(config['freq_hp'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	config['level'] = int(config['level'])
	return config


if __name__ == '__main__':
	main(sys.argv)
