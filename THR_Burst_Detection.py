# Burst_Detection.py
# Last updated: 25.09.2017 by Felix Leaman
# Description:
# Code for opening N data files with single channel and detecting Bursts
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively
# Power2 option let the user to analyze only 2^Power2 points of each file

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from tkinter import filedialog
from tkinter import Tk
from skimage import img_as_uint
import skimage.filters
import os.path
import sys
sys.path.insert(0, './lib') #to open user-defined functions
import argparse
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
import io
import matplotlib.patches as mpatches
plt.rcParams['agg.path.chunksize'] = 20000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs', 'save', 'save_plot']
Opt_Input = {'interval':0, 'n_files':1, 'files':None, 'window_time':0.001, 'data_norm':None, 'plot':'ON', 'save_name':'NAME'}
Opt_Input_analysis = {'denois':'ON', 'processing':'OFF', 'demod_filter':['lowpass', 5000, 3], 'demod_prefilter':['highpass', 70.e3, 3], 'demod_rect':'only_positives', 'demod_dc':'without_dc', 'diff':'OFF'}
# Opt_Input_analysis = {'denois':'OFF', 'processing':'butter_demod', 'demod_filter':['lowpass', 5000, 3], 'demod_prefilter':['highpass', 70.e3, 3], 'demod_rect':'only_positives', 'demod_dc':'without_dc', 'diff':1}
Opt_Input_thr = {'thr_mode':'fixed_value', 'thr_value':1.0}


Opt_Input.update(Opt_Input_analysis)
Opt_Input.update(Opt_Input_thr)





def main(argv):
	config = read_parser(argv, Inputs, Opt_Input)
	print('jaja')
	X = [[] for j in range(config['n_files'])]
	T_Burst = [[] for j in range(config['n_files'])]
	A_Burst = [[] for j in range(config['n_files'])]
	T_Burst_rev = [[] for j in range(config['n_files'])]
	A_Burst_rev = [[] for j in range(config['n_files'])]
	Filenames = [[] for j in range(config['n_files'])]
	
	for k in range(config['n_files']):		
		if config['files'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			filename1 = filedialog.askopenfilename()
			root.destroy()
		else:
			filename1 = config['files'][k]
		
		# x = f_open_mat_2(filename1)
		# x = np.ndarray.flatten(x[0])
		x = load_signal(filename1, channel=config['channel'])
		
		
		n_points = int(len(x)/4)
		interval = config['interval']
		x = x[n_points*(interval):n_points*(interval+1)]
		
		# x = x[90000000:95000000]
		# x = (x /141.25) * 1000
		xraw = x
		
		
		# x = x[::-1]
		
		n_points = len(x)
		dt = 1.0/config['fs']
		tr = n_points*dt
		# t = np.array([(n_points*interval)*dt + i*dt for i in range(n_points)])
		t = np.array([i*dt for i in range(n_points)])
		traw = t

		filename1 = os.path.basename(filename1) #changes from path to file
		
		print(filename1[29:35])
		config['filename'] = filename1
		if filename1.find('1500') != -1:
			label = str(1500)
		else:
			label = str(1000)
		print(label)
		Filenames[k] = filename1
		

		x, t_burst_corr1, amp_burst_corr1 = thr_burst_detector(xraw, config, count=k)
		
		x1rev, t_burst_corr1rev, amp_burst_corr1rev = thr_burst_detector(xraw[::-1], config, count=k)
		
		
		if len(t_burst_corr1) != len(t_burst_corr1rev):
			print('fatal error')
			print(len(t_burst_corr1))
			print(len(t_burst_corr1rev))
			sys.exit()
		
		t_burst_corr1rev = t_burst_corr1rev[::-1]
		t_burst_corr1rev = tr*np.ones(len(t_burst_corr1rev)) - np.array(t_burst_corr1rev)
		t_burst_corr1rev = t_burst_corr1rev.tolist()
		amp_burst_corr1rev = amp_burst_corr1rev[::-1]
		
		print('t burst11111')
		print(t_burst_corr1)
		X[k] = x
		T_Burst[k] = t_burst_corr1
		A_Burst[k] = amp_burst_corr1
		
		T_Burst_rev[k] = t_burst_corr1rev
		A_Burst_rev[k] = amp_burst_corr1rev
	
	print('Detected Burst')
	for i in range(config['n_files']):		
		print(len(T_Burst[i]))
	print(config)
	
	
	tburst = t_burst_corr1
	feat_rms = []
	feat_max = []
	feat_p2p = []
	feat_freq = []
	feat_rise = []
	feat_crest = []
	
	
	for j in range(len(t_burst_corr1)):
		signal = x[int(tburst[j]*config['fs']) : int(tburst[j]*config['fs'] + config['window_time']*config['fs'])]
		feat_max.append(np.max(signal))
		rms_it = signal_rms(signal)
		feat_rms.append(rms_it)
		feat_p2p.append(np.max(signal) - np.min(signal))
		feat_crest.append(np.max(np.absolute(signal))/rms_it)
		magX, f, df = mag_fft(signal, config['fs'])
		# plt.plot(f, magX)
		# plt.show()
		feat_freq.append(np.argmax(magX)/config['window_time'])
		feat_rise.append(np.argmax(signal)/config['fs'])

	feat_dur = []
	feat_count = []
	print('+++++++++N Burst')
	print(len(t_burst_corr1))
	print(len(t_burst_corr1rev))
	print(t_burst_corr1)
	print(t_burst_corr1rev)
	# aaa = input('pause')
	for a, b in zip(t_burst_corr1, t_burst_corr1rev):
		feat_dur.append(b-a)
		ap = a*config['fs']
		bp = b*config['fs']
		signal = x[int(ap) : int(bp)]
		contador = 0
		# plt.plot(signal)
		# plt.show()
		for u in range(len(signal)-1):			
			if (signal[u] < config['thr_value'] and signal[u+1] >= config['thr_value']):
				contador = contador + 1
		feat_count.append(contador)
		# signal = x[int(tburst[j]*config['fs']) : int(tburst[j]*config['fs'] + config['window_time']*config['fs'])]
	
	feat_dur = np.array(feat_dur)*1000.
	feat_dur = feat_dur.tolist()
	
	feat_rise = np.array(feat_rise)*1000.
	feat_rise = feat_rise.tolist()
	
	print('++++++++++++duracion')
	print(feat_dur)
	print('++++++++++++rise time')
	print(feat_rise)
	print('++++++++++++count')
	print(feat_count)
	print('++++++++++++freq')
	print(feat_freq)
	print('++++++++++++p2p')
	print(feat_p2p)
	print('++++++++++++max')
	print(feat_max)
	print('++++++++++++rms')
	print(feat_rms)
	print('++++++++++++crest')
	print(feat_crest)
	
	if config['save'] == 'ON':
	
		# mylist = [filename1, interval, '4 intervals', config['thr_value'], 'no_filter', t_burst_corr1, ]
		# mylist = [Filenames, t_burst_corr1, config]
		mydict = {'file': filename1, 'intervalXof4':config['interval'], 'thr_value':config['thr_value'], 'wtime':config['window_time'], 'filter':'no_filter', 't_burst':t_burst_corr1, 'feat_max':feat_max, 'feat_p2p':feat_p2p, 'feat_rms':feat_rms, 'feat_freq':feat_freq, 'feat_dur':feat_dur, 'feat_count':feat_count, 'feat_rise':feat_rise, 'feat_crest':feat_crest}
		
		# out_file = 'features_' + filename1[29:35] + '_interval_' + str(config['interval']) + '_thr_' + str(config['thr_value']) + '_wtime_' + str(config['window_time']) + '_80khzHP_' +'.pkl'
		
		out_file = 'features_' + filename1[0:len(filename1)-5] + '_thr_' + str(config['thr_value']) + '_wtime_' + str(config['window_time']) + '_300khzHP_441khzBS_' +'.pkl'
		

		save_pickle(out_file, mydict)

	zoom_list = [None]
	
	print('n burst')
	print(len(t_burst_corr1))
	
	if config['plot'] == 'ON':		
		

		
		# fig, ax = plt.subplots(nrows=config['n_files'], ncols=1, sharex=True, sharey=True)			
		# plot_burst(fig, ax, i, traw, X[i], config, T_Burst[i], A_Burst[i], name='aRAW', thr=True)
		
		fig2, ax2 = plt.subplots(nrows=config['n_files'], ncols=1, sharex=True, sharey=True)			
		plot_burst_rev(fig2, ax2, i, traw, X[i], config, T_Burst[i], A_Burst[i], T_Burst_rev[i], A_Burst_rev[i])
		
		fig0, ax0 = plt.subplots(nrows=config['n_files'], ncols=1, sharex=True, sharey=True)			
		plot_burst(fig0, ax0, i, traw, xraw, config, T_Burst[i], A_Burst[i], name='bRAW', thr=True)
	
	
			
		if config['save_plot'] == 'OFF':
			plt.show()
	else:
		print('Plot Off')
	plt.close('all')
	
	return


def thr_burst_detector(x1, config, count=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t
	#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++
	

	#++++++++++++++++++++++CHECKS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	

	#++++++++++++++++++++++SIGNAL PROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# if config['data_norm'] == 'per_signal':
		# x1 = x1 / np.max(np.absolute(x1))
		# print('Normalization per signal')
	# elif config['data_norm'] == 'per_rms':
		# x1 = x1 / signal_rms(x1)
		# print('Normalization per rms')
	# else:
		# print('No normalization')
	
	
	
	
	
	# if config['denois'] != 'OFF':
		# print('with denois HP+++++++++++++++++++++++++++')
		# # x1 = signal_denois(x=x1, denois=config['denois'], med_kernel=config['med_kernel'])
		
		# x1 = butter_highpass(x=x1, fs=config['fs'], freq=300.e3, order=3, warm_points=None)
		
		# print('with denois BS+++++++++++++++++++++++++++')
		# x1 = butter_bandstop(x=x1, fs=config['fs'], freqs=[439.e3, 442.e3], order=3, warm_points=None)
		
	# else:
		# print('without denois')
		
	# if config['processing'] != 'OFF':
		# print('with processing')
		# x1 = signal_processing(x1, config)
	# else:
		# print('without processing')
	
	# if config['diff'] != 'OFF':
		# print('with diff')
		# x1 = diff_signal_eq(x1, config['diff'])
	
	
	#++++++++++++++++++++++ BURST DETECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	

	
	threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold1, t_window=config['window_time'])
	print('time TP111111111111111111111')


	

	return t_burst_corr1, amp_burst_corr1

def thr_burst_detector_rev(x, config, count=None):

	x = x[::-1]

	dt = 1.0/config['fs']
	n_points = len(x)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t

	threshold = read_threshold(config['thr_mode'], config['thr_value'], x)
	
	n_burst, t_burst_corr, amp_burst_corr, t_burst, amp_burst = id_burst_threshold(x=x, fs=config['fs'], threshold=threshold, t_window=config['window_time'])
	
	
	# tr = len(t)/config['fs']
	t_burst_corr = t_burst_corr[::-1]
	t_burst_corr = tr*np.ones(len(t_burst_corr)) - np.array(t_burst_corr)
	t_burst_corr = t_burst_corr.tolist()
	amp_burst_corr = amp_burst_corr[::-1]


	

	return t_burst_corr, amp_burst_corr

# def full_thr_burst_detector(x1, config, count=None):
	# dt = 1.0/config['fs']
	# n_points = len(x1)
	# tr = n_points*dt
	# t = np.array([i*dt for i in range(n_points)])
	# traw = t


	


	# threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	# n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold1, t_window=config['window_time'])
	# print('time TP111111111111111111111')
	

	# t_burst_corr_rev = []
	
	# for t_ini in t_burst_corr1:
		# signal = x1[t_ini*config['fs'] : (t_ini + config['window_time'])*config['fs']]
		# signal = signal[::-1]
		# for k in range(len(signal)):
			# if signal[k] >= threshold1:
				# t_end = t_ini + config['window_time'] - (k+1)/config['fs']
				# t_burst_corr_rev.append(t_end)
				# break
	

	# # amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]
	# amp_burst_corr_rev = amp_burst_corr1

	# return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev


def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]

	parser = argparse.ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if (element == 'files' or element == 'demod_prefilter' or element == 'demod_filter' or element == 'clf_files' or element == 'pv_removal'):
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config_input = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config_input[element] = value
	
	#Type conversion to float
	# print(config_input['fs'])
	config_input['fs'] = float(config_input['fs'])
	config_input['window_time'] = float(config_input['window_time'])
	config_input['thr_value'] = float(config_input['thr_value'])

	
	#Type conversion to int
	config_input['n_files'] = int(config_input['n_files'])
	config_input['interval'] = int(config_input['interval'])
	# config_input['power2'] = int(config_input['power2'])
	# config_input['diff_points'] = int(config_input['diff_points'])

	
	# Variable conversion

	return config_input
	
#Signal RAW
def plot_burst(fig, ax, nax, t, x1, config, t_burst_corr1, amp_burst_corr1, thr=None, name=None, color=None):
	# print(signal_rms(x1))
	# a = input('oooooo')
	# x1 = x1/signal_rms(x1)
	if name != None:
		name = name + ' '
	else:
		name = ''
	if color == None:
		color = None
	if config['n_files'] == 1:
		ax = [ax]

	ax[nax].plot(t, x1, color=color)
	# ax[nax].plot(t, x1, color=color)

	# print(signal_rms(x1))
	
	
	if thr == True:
		threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
		ax[nax].axhline(threshold1, color='k')		
	ax[nax].plot(t_burst_corr1, amp_burst_corr1, 'ro') 
	# print(amp_burst_corr1/signal_rms(x1))
	# a = input('aaa')
	
	
	
	if nax == 0:
		name = 'Faulty Case Test Signal: '
	else:
		name = 'Healthy Case Test Signal: '
		ax[nax].set_xlabel('Time (s)')
	flag = config['filename'].find('1500')
	if flag != -1:
		flag2 = config['filename'].find('80')
		if flag2 != -1:
			name = name + '1500RPM / 80% Load'
			plotname = '_RAW_1500_80_'
		else:
			name = name + '1500RPM / 40% Load'
			plotname = '_RAW_1500_40_'
	else:		
		flag2 = config['filename'].find('80')
		if flag2 != -1:
			name = name + '1000RPM / 80% Load'
			plotname = '_RAW_1000_80_'
		else:
			name = name + '1000RPM / 40% Load'
			plotname = '_RAW_1000_40_'
		
		
	ax[nax].set_title(name, fontsize=10)
	ax[nax].set_ylabel('Norm. Amplitude')
	
	if config['save_plot'] == 'ON':
		if zoom == None:
			zoom_ini = 'All'
		else:
			zoom_ini = str(zoom[0])
		plt.savefig(config['method'] + plotname + '_' + zoom_ini + '.png')
	
	
	
	
	# ax[nax].set_title(name + config['channel'] + ' ' + config['method'] + '\n' + config['filename'], fontsize=10)
	# ax[nax].set_ylabel('Amplitude')
	return

def plot_burst_rev(fig, ax, nax, t, x1, config, t_burst_corr1, amp_burst_corr1, t_burst_corr1rev, amp_burst_corr1rev):
	# print(signal_rms(x1))
	# a = input('oooooo')
	# x1 = x1/signal_rms(x1)
	# if name != None:
		# name = name + ' '
	# else:
		# name = ''
	# if color == None:
		# color = None
	if config['n_files'] == 1:
		ax = [ax]

	ax[nax].plot(t, x1)
	# ax[nax].plot(t, x1, color=color)

	# print(signal_rms(x1))
	
	thr = True
	if thr == True:
		threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
		ax[nax].axhline(threshold1, color='k')		
	ax[nax].plot(t_burst_corr1, amp_burst_corr1, 'ro')
	ax[nax].plot(t_burst_corr1rev, amp_burst_corr1rev, 'mo') 	
	# print(amp_burst_corr1/signal_rms(x1))
	# a = input('aaa')
	

	# ax[nax].set_title(name, fontsize=10)
	# ax[nax].set_ylabel('Norm. Amplitude')
	
	params = {'mathtext.default': 'regular' }          
	plt.rcParams.update(params)
	ax[nax].set_ylabel('Amplitude [mV]', fontsize=12)
	ax[nax].set_xlabel('Dauer [s]', fontsize=12)
	ax[nax].tick_params(axis='both', labelsize=11)
	
	if config['save_plot'] == 'ON':
		if zoom == None:
			zoom_ini = 'All'
		else:
			zoom_ini = str(zoom[0])
		plt.savefig(config['method'] + plotname + '_' + zoom_ini + '.png')
	
	
	
	
	# ax[nax].set_title(name + config['channel'] + ' ' + config['method'] + '\n' + config['filename'], fontsize=10)
	# ax[nax].set_ylabel('Amplitude')
	return

# def read_threshold(mode, value, x1=None):
	# if mode == 'factor_rms':
		# threshold1 = value*signal_rms(x1)
	# elif mode == 'fixed_value':
		# threshold1 = value
	# else:
		# print('error threshold mode')
		# sys.exit()
	# return threshold1

def prepare_windows(x, config):
	n_points = len(x)
	Windows1 = []
	window_points = int(config['window_time']*config['fs'])
	window_advance = int(window_points*config['overlap'])
	if config['window_delay'] != 0:
		print('With window delay')
		window_delay = int(config['window_delay']*config['fs'])
	else:
		window_delay = 0
	
	if config['overlap'] != 0:
		print('Windows with overlap')
		n_windows = int((n_points - window_points)/window_advance) + 1
	else:
		n_windows = int(n_points/window_points)
	print('Number of windows: ', n_windows)
	
	
	for count in range(n_windows):
		if config['overlap'] != 0: #with overlap not working?
			Windows1.append(x[count*window_advance:window_points+window_advance*count])
		else:
			if count != 0:
				Windows1.append(x[(count*window_points - window_delay):((count+1)*window_points - window_delay)])
			else:
				Windows1.append(x[(count*window_points):(count+1)*window_points])
	return Windows1



	

if __name__ == '__main__':
	main(sys.argv)