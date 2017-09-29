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

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs', 'power2', 'method', 'n_files', 'save']
Opt_Input = {'files':None, 'window_time':0.001, 'overlap':0, 'data_norm':None, 'clf_files':None, 'clf_check':'OFF', 'plot':'ON'}
Opt_Input_analysis = {'EMD':'OFF', 'denois':'OFF', 'med_kernel':3, 'processing':'OFF'}
Opt_Input_thr = {'thr_mode':'factor_rms', 'thr_value':1}
Opt_Input_cant = {'demod':None, 'prefilter':None, 'postfilter':None, 'rectification':None, 'dc_value':None, 'warm_points':100, 'diff_points':1}
Opt_Input_nn = {'NN_model':None, 'features':None, 'feat_norm':'standard', 'class2':0}
Opt_Input.update(Opt_Input_analysis)
Opt_Input.update(Opt_Input_thr)
Opt_Input.update(Opt_Input_cant)
Opt_Input.update(Opt_Input_nn)



def main(argv):
	config = read_parser(argv, Inputs, Opt_Input)

	X = [[] for j in range(config['n_files'])]
	XRAW = [[] for j in range(config['n_files'])]
	T_Burst = [[] for j in range(config['n_files'])]
	A_Burst = [[] for j in range(config['n_files'])]
	ARAW_Burst = [[] for j in range(config['n_files'])]
	Results = [[] for j in range(config['n_files'])]
	CLFs = [[] for j in range(config['n_files'])]
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
		
		if config['EMD'] == 'ON':
			print('with MED component h1')
			filename1raw = filename1
			filename1 = filename1.replace(os.path.basename(filename1), 'h1_' + os.path.basename(filename1))
			filename1 = filename1.replace('.mat', '.txt')
		x1raw = load_signal(filename1, channel=config['channel'])
		
		if config['EMD'] == 'ON':
			filename1 = filename1raw
		
		if config['power2'] == None:
			n_points = 2**(max_2power(len(x1)))
		else:
			n_points = 2**config['power2']
		
		x1raw = x1raw[0:n_points]
		dt = 1.0/config['fs']
		tr = n_points*dt
		t = np.array([i*dt for i in range(n_points)])
		traw = t

		filename1 = os.path.basename(filename1) #changes from path to file
		config['filename'] = filename1
		if filename1.find('1500') != -1:
			label = str(1500)
		else:
			label = str(1000)
		print(label)
		Filenames[k] = filename1
		
		x1, t_burst_corr1, amp_burst_corr1, results, clf_1 = burst_detector(x1raw, config, count=k)
		X[k] = x1
		XRAW[k] = x1raw		
		T_Burst[k] = t_burst_corr1
		A_Burst[k] = amp_burst_corr1
		ARAW_Burst[k] = np.array([x1raw[int(time*config['fs'])] for time in t_burst_corr1])
		Results[k] = results
		CLFs[k] = clf_1
	Name = config['method']
	
	print('Detected Burst')
	for i in range(config['n_files']):		
		print(len(T_Burst[i]))
		print(Results[i])
	print(config)
	if config['save'] == 'ON':
		mylist = [Filenames, Results, config]
		out_file = 'result_' + os.path.basename(config['NN_model'])
		out_file = label + out_file
		save_pickle(out_file, mylist)
		# with io.open(out_file, 'w', encoding='unicode-escape') as f:
			# f.writelines(line + u'\n' for line in mylist)
		# np.savetxt('result_' + os.path.basename(config['NN_model']).replace('pkl', 'txt'), )
	
	if config['plot'] == 'ON':
		fig = [[], []]
		fig[0], ax = plt.subplots(nrows=config['n_files'], ncols=1, sharex=True, sharey=True)
		for i in range(config['n_files']):
			plot_burst_paper(fig[0], ax, i, t, X[i], config, T_Burst[i], A_Burst[i], thr=True)	

		#++++++++++++++++++++++ BURSTS BACK IN RAW SIGNALS ++++++++
		fig[1], ax = plt.subplots(nrows=config['n_files'], ncols=1, sharex=True, sharey=True)
		for i in range(config['n_files']):
			plot_burst(fig[1], ax, i, traw, XRAW[i], config, T_Burst[i], ARAW_Burst[i], name='RAW', color='darkblue', clf=CLFs[i])	
		plt.show()
	else:
		print('Plot Off')
	
	return


def burst_detector(x1, config, count=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t
	#++++++++++++++++++++++ ANALYSIS CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++
	if config['clf_check'] == 'ON':
		if config['clf_files'] == None:
			print('Select Classifications file in order')
			root = Tk()
			root.withdraw()
			root.update()
			clf_pickle1 = filedialog.askopenfilename()
		# clf_file2 = filedialog.askopenfilename()
			root.destroy()
		else:
		# print(config['clf_files'][0])
			clf_pickle1 = read_pickle(config['clf_files'][count])
		# print(clf_pickle1)
		# sys.exit()
		clf_1 = clf_pickle1['classification']
		# if clf_pickle1['config_analysis']['WindowTime'] != config_neuronal['WindowTime']:
			# print('error window time')
			# sys.exit()
		print(clf_pickle1['filename'])
		print(config['filename'])
		if clf_pickle1['filename'] != config['filename']:
			print('error filename 1')
			sys.exit()


	if config['method'] == 'NN':
		if config['NN_model'] == None:
			print('Select NN Model:')
			root = Tk()
			root.withdraw()
			root.update()
			path_info_model = filedialog.askopenfilename()
			info_model = read_pickle(path_info_model)
			root.destroy()
		else:
			info_model = read_pickle(config['NN_model'])		
		clf = info_model[1]
		config_model = info_model[0]
		plt.show()
	else:
		clf = None
	

	if (config['feat_norm'] == 'standard' and config['method'] == 'NN'):
		print('Standard Scale:')
		scaler = info_model[2]
	else:
		scaler = 0
	
	if config['features'] == 'pca_50' or config['features'] == 'pca_10' or config['features'] == 'pca_5':
		print('PCA:')
		pca = info_model[3]
	else:
		pca = 0

	#Pre-processing
	# config_filter = {'analysis':False, 'type':'median', 'mode':'bandpass', 'params':[[70.0e3, 350.0e3], 3]}

	# config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

	# config_diff = {'analysis':False, 'length':1, 'same':True}

	# config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [70.0e3, 170.0e3] , 3], 
	# 'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warm_points':20000}
	#When hilbert is selected, the other parameters are ignored


	#++++++++++++++++++++++CHECKS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config['method'] == 'NN':
		if config['data_norm'] != config_model['data_norm']:
			print('error normalization model NN')
			sys.exit()
		if config['denois'] != config_model['denois']:
			print('error denois model NN')
			sys.exit()
		if config['EMD'] != config_model['EMD']:
			print('error EMD model NN')
			sys.exit()
		if config['processing'] != config_model['processing']:
			print('error processing model NN')
			sys.exit()
		# print(config_model['denois'])
		# a = input('pause....')
		if config_model['denois'] == 'median':
			if config['med_kernel'] != config_model['med_kernel']:
				print('error med_kernel model NN')
				sys.exit()
		if config['features'] != config_model['features']:
			print('error features model NN')
			sys.exit()

	#++++++++++++++++++++++SIGNAL PROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config['denois'] != 'OFF':
		print('with denois')
		x1 = signal_denois(x=x1, denois=config['denois'], med_kernel=config['med_kernel'])
	else:
		print('without denois')
		
	if config['processing'] != 'OFF':
		print('with processing')
		x1 = signal_processing(x1, config['processing'])
	else:
		print('without processing')
	#Filter
	# if config_filter['analysis'] == True:
		# print('+++Filter:')
		# if config_filter['type'] == 'bandpass':
			# print('Bandpass')
			# f_nyq = 0.5*fs
			# order = config_filter['params'][1]
			# freqs_bandpass = [config_filter['params'][0][0]/f_nyq, config_filter['params'][0][1]/f_nyq]
			# b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
			# x1 = signal.filtfilt(b, a, x1)
			# x2 = signal.filtfilt(b, a, x2)
		# elif config_filter['type'] == 'median':
			# print('Median')
			# x1 = scipy.signal.medfilt(x1)
			# x2 = scipy.signal.medfilt(x2)

	# #Autocorrelation
	# if config_autocorr['analysis'] == True:
		# print('+++Filter:')
		# if config_autocorr['type'] == 'definition':
			# x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
			# x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
		
		# elif config_autocorr['type'] == 'wiener':	
			# fftx1 = np.fft.fft(x1)
			# x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))		
			# fftx2 = np.fft.fft(x2)
			# x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))
		# else:
			# print('Error assignment autocorrelation')

	# #Demodulation
	# if config_demod['analysis'] == True:
		# print('+++Demodulation:')
		# if config_demod['mode'] == 'hilbert':
			# x1 = hilbert_demodulation(x1)
			# x2 = hilbert_demodulation(x2)
		# elif config_demod['mode'] == 'butter':
			# x1 = butter_demodulation(x=x1, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
			# type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
			# x2 = butter_demodulation(x=x2, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
			# type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
		# else:
			# print('Error assignment demodulation')


	# #Differentiation
	# if config_diff['analysis'] == True:
		# print('+++Differentiation:')
		# if config_diff['same'] == True:
			# x1 = diff_signal_eq(x=x1, length_diff=config_diff['length'])
			# x2 = diff_signal_eq(x=x2, length_diff=config_diff['length'])
		# elif config_diff['same'] == False:
			# x1 = diff_signal(x=x1, length_diff=config_diff['length'])
			# x2 = diff_signal(x=x2, length_diff=config_diff['length'])
		# else:
			# print('Error assignment diff')	

	# warm = 0.0
	# if (config_demod['warm_points'] != 0 and config_demod['mode'] == 'butter' and config_demod['analysis'] == True):
		# x1 = x1[config_demod['warm_points']:]
		# x2 = x2[config_demod['warm_points']:]
		# t = t[config_demod['warm_points']:]
		# warm = float(config_demod['warm_points'])



	#++++++++++++++++++++++ BURST DETECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	
	Results = {'POS':0, 'NEG':0, 'FP':0, 'TP':0, 'TN':0, 'FN':0}

	if config['method'] != None:
		if config['data_norm'] == 'per_signal':
			x1 = x1 / np.max(np.absolute(x1))
			print('Normalization per signal')
		elif config['data_norm'] == 'per_rms':
			x1 = x1 / signal_rms(x1)
			print('Normalization per rms')
		
		
		if config['method'] == 'THR':
		
			
		
		
			threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
			
			n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold1, t_window=config['window_time'])
			
			tWindows = []
			window_points = int(config['window_time']*config['fs'])
			n_windows = int(n_points/window_points)
			for count in range(n_windows):
				tWindows.append(t[count*window_points:(count+1)*window_points])
			count = 0
			for twindow in tWindows:
				flag = 'OFF'
				if clf_1[count] == 2:
					clf_1[count] = config['class2']
				if clf_1[count] == 0:
					Results['NEG'] = Results['NEG'] + 1
					for time_burst in t_burst_corr1:
						if time_burst in twindow:
							# print('++++++++++++FP in ', count*config['window_time'])
							Results['FP'] = Results['FP'] + 1
							flag = 'ON'
					if flag == 'OFF':
						Results['TN'] = Results['TN'] + 1
				elif clf_1[count] == 1:
					Results['POS'] = Results['POS'] + 1
					for time_burst in t_burst_corr1:
						if time_burst in twindow:
							Results['TP'] = Results['TP'] + 1
							flag = 'ON'
					if flag == 'OFF':
						Results['FN'] = Results['FN'] + 1
				else:
					print('error file clf')
					sys.exit()
				count = count + 1

			# t_burst_corr1 = t_burst_corr1 + np.ones(len(t_burst_corr1))*warm*dt
		
		elif config['method'] == 'NN':
		
			Windows1 = []
			window_points = int(config['window_time']*config['fs'])
			window_advance = int(window_points*config['overlap'])
			if config['overlap'] != 0:
				print('Windows with overlap')
				n_windows = int((n_points - window_points)/window_advance) + 1
			else:
				n_windows = int(n_points/window_points)
			print('Number of windows: ', n_windows)
			
			if config['data_norm'] == 'per_signal':
				x1 = x1 / np.max(np.absolute(x1))
				print('Normalization per signal')		
			
			for count in range(n_windows):
				if config['overlap'] != 0:
					Windows1.append(x1[count*window_advance:window_points+window_advance*count])
				else:
					Windows1.append(x1[count*window_points:(count+1)*window_points])
			
			Predictions1 = []
			features_fault = []
			features_ok = []
			numero = 0
			for window1 in Windows1:
				if config['data_norm'] == 'per_window':
					print('Normalization per window')
					window1 = window1 / np.max(np.absolute(window1))
				
				if config['features'] == 'interval10_stats_nomean':
					values = interval10_stats_nomean(window1)
				elif config['features'] == 'interval5_stats_nomean':
					values = interval5_stats_nomean(window1)
				elif config['features'] == 'leftright_stats_nomean':
					values = leftright_stats_nomean(window1)
				elif config['features'] == 'leftright_std':
					values = leftright_std(window1)
				elif config['features'] == 'i10statsnm_lrstd':
					values = i10statsnm_lrstd(window1)
				elif config['features'] == 'i10statsnm_dif_lrstd':
					values = i10statsnm_dif_lrstd(window1)
				elif config['features'] == 'i10statsnm_lrstatsnm':
					values = i10statsnm_lrstatsnm(window1)
				elif config['features'] == 'means10':
					values = means10(window1)
				elif config['features'] == 'pca_50':
					values = window1
				elif config['features'] == 'pca_10':
					values = window1
				elif config['features'] == 'pca_5':
					values = window1
				elif config['features'] == 'i10statsnmnsnk_lrstd':
					values = i10statsnmnsnk_lrstd(window1)
				else:
					print('error features')
					sys.exit()
				# points_intervals = n_per_intervals_left_right(window1, [-1., 1.], 5)
				# values = basic_stats_sides
				if config['features'] == 'pca_50' or config['features'] == 'pca_10' or config['features'] == 'pca_5':
					values = pca.transform(values)
				
				
				values = scaler.transform(values)
				
				# if numero == 5:
					# print(values)
					# sys.exit()
				
				features_fault.append(values)
				prediction = clf.predict(values)
				
				if prediction[0] == 2:
					prediction[0] = config['class2']
				if clf_1[numero] == 2:
					clf_1[numero] = config['class2']
				
				if clf_1[numero] == 0:
					Results['NEG'] = Results['NEG'] + 1
					if prediction[0] == clf_1[numero]:
						Results['TN'] = Results['TN'] + 1
					else:
						Results['FP'] = Results['FP'] + 1
				elif clf_1[numero] == 1:
					Results['POS'] = Results['POS'] + 1
					if prediction[0] == clf_1[numero]:
						Results['TP'] = Results['TP'] + 1
					else:
						Results['FN'] = Results['FN'] + 1

				Predictions1.append(prediction[0])

				numero = numero + 1
				
			t_burst_corr1 = []
			amp_burst_corr1 = []
			for i in range(len(Predictions1)):
				if Predictions1[i] == 1:
					t_burst_corr1.append(i*config['window_time'])
					amp_burst_corr1.append(x1[int(i*config['window_time']*config['fs'])])

	#++++++++++++++++++++++ PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# if config_filter['analysis'] == True:
		# Name = Name + ' FIL'
	# if config_autocorr['analysis'] == True:
		# Name = Name + ' ACR'
	# if config_demod['analysis'] == True:
		# Name = Name + ' ENV'		
	# if config_diff['analysis'] == True:
		# Name = Name + ' DIF'
	
	Results['Accu'] = 100. * (Results['TN'] + Results['TP']) / (Results['POS'] + Results['NEG'])
	if Results['POS'] != 0:
		Results['Recall'] = 100. * (Results['TP']) / (Results['POS'])
	else:
		Results['Recall'] = -1
	if (Results['TP'] + Results['FP']) != 0:
		Results['Precision'] = 100. * (Results['TP']) / (Results['TP'] + Results['FP'])
	else:
		Results['Precision'] = -1
	Results['FScore'] = 2 * Results['Recall'] * Results['Precision'] / (Results['Recall'] + Results['Precision'])
	Results['FPR'] = 100. * (Results['FP']) / (Results['NEG'])

	return x1, t_burst_corr1, amp_burst_corr1, Results, clf_1


def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]

	parser = argparse.ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'files' or element == 'prefilter' or element == 'postfilter' or element == 'clf_files':
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
	config_input['fs'] = float(config_input['fs'])
	config_input['overlap'] = float(config_input['overlap'])
	config_input['window_time'] = float(config_input['window_time'])
	config_input['thr_value'] = float(config_input['thr_value'])
	
	#Type conversion to int
	config_input['n_files'] = int(config_input['n_files'])
	config_input['power2'] = int(config_input['power2'])
	config_input['warm_points'] = int(config_input['warm_points'])
	config_input['diff_points'] = int(config_input['diff_points'])
	config_input['class2'] = int(config_input['class2'])
	config_input['med_kernel'] = int(config_input['med_kernel'])
	
	# Variable conversion
	if config_input['prefilter'] != None:
		config_input['prefilter'] = [config_input['prefilter'][0], float(config_input['prefilter'][1]), float(config_input['prefilter'][2]), int(config_input['prefilter'][3])]
	if config_input['postfilter'] != None:
		config_input['postfilter'] = [config_input['postfilter'][0], float(config_input['postfilter'][1]), int(config_input['postfilter'][2])]


	return config_input

def plot_burst(fig, ax, nax, t, x1, config, t_burst_corr1, amp_burst_corr1, thr=None, name=None, color=None, clf=None):
	if name != None:
		name = name + ' '
	else:
		name = ''
	if color == None:
		color = None
	if config['n_files'] == 1:
		ax = [ax]

	ax[nax].plot(t, x1, color=color)
	if config['method'] == 'THR':
		if thr == True:
			threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
			ax[nax].axhline(threshold1, color='k')		
		ax[nax].plot(t_burst_corr1, amp_burst_corr1, 'ro')
	elif config['method'] == 'NN':
		for i in range(len(t_burst_corr1)):
			ax[nax].axvspan(xmin=t_burst_corr1[i], xmax=t_burst_corr1[i]+config['window_time'], facecolor='y', alpha=0.5)
	if clf != None:
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		for k in range(len(clf)):
			if clf[k] == 2:
				print('error clf')
				sys.exit()
		ind_w_positives = [k for k in range(len(clf)) if clf[k] == 1]
		print(len(ind_w_positives))
		print(len(clf))
		for i in range(len(ind_w_positives)):
			ax[nax].axvspan(xmin=config['window_time']*ind_w_positives[i], xmax=config['window_time']*(ind_w_positives[i] + 1), facecolor='r', alpha=0.5)
		
		
	ax[nax].set_title(name + config['channel'] + ' ' + config['method'] + '\n' + config['filename'], fontsize=10)
	ax[nax].set_ylabel('Amplitude')
	return

def plot_burst_paper(fig, ax, nax, t, x1, config, t_burst_corr1, amp_burst_corr1, thr=None, name=None, color=None, clf=None):
	if name != None:
		name = name + ' '
	else:
		name = ''
	if color == None:
		color = None
	if config['n_files'] == 1:
		ax = [ax]

	ax[nax].plot(t, x1, color=color)
	if config['method'] == 'THR':
		if thr == True:
			threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
			ax[nax].axhline(threshold1, color='k')		
		ax[nax].plot(t_burst_corr1, amp_burst_corr1, 'ro')
	elif config['method'] == 'NN':
		for i in range(len(t_burst_corr1)):
			ax[nax].axvspan(xmin=t_burst_corr1[i], xmax=t_burst_corr1[i]+config['window_time'], facecolor='y', alpha=0.5)
	if clf != None:
		# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		for k in range(len(clf)):
			if clf[k] == 2:
				print('error clf')
				sys.exit()
		ind_w_positives = [k for k in range(len(clf)) if clf[k] == 1]
		# print(len(ind_w_positives))
		# print(len(clf))
		for i in range(len(ind_w_positives)):
			ax[nax].axvspan(xmin=config['window_time']*ind_w_positives[i], xmax=config['window_time']*(ind_w_positives[i] + 1), facecolor='r', alpha=0.5)
		
	if nax == 0:
		name = 'Faulty Validation Signal: '
	else:
		name = 'Healthy Validation Signal: '
		ax[nax].set_xlabel('Time (s)')
	flag = config['filename'].find('1500')
	if flag != -1:
		flag2 = config['filename'].find('80')
		if flag2 != -1:
			name = name + '1500RPM / 80% Load'
		else:
			name = name + '1500RPM / 40% Load'
	else:
		name = name + '1000RPM / 80% Load'
	ax[nax].set_title(name, fontsize=10)
	ax[nax].set_ylabel('Amplitude')
	return

def read_threshold(mode, value, x1=None):
	if mode == 'factor_rms':
		threshold1 = value*signal_rms(x1)
	elif mode == 'fixed_value':
		threshold1 = value
	else:
		print('error threshold mode')
		sys.exit()
	return threshold1


if __name__ == '__main__':
	main(sys.argv)