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

plt.rcParams['agg.path.chunksize'] = 20000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs', 'power2', 'method', 'n_files', 'save']
Opt_Input = {'files':None, 'window_time':0.001, 'overlap':0, 'data_norm':None, 'clf_files':None, 'clf_check':'OFF', 'plot':'ON'}
Opt_Input_analysis = {'EMD':'OFF', 'denois':'OFF', 'med_kernel':3, 'processing':'OFF', 'demod_filter':None, 'demod_prefilter':None, 'demod_rect':None, 'demod_dc':None, 'diff':'OFF'}
Opt_Input_thr = {'thr_mode':'factor_rms', 'thr_value':1}
Opt_Input_cant = {'demod':None, 'prefilter':None, 'postfilter':None, 'rectification':None, 'dc_value':None, 'warm_points':100, 'diff_points':1, 'window_extend':0}
Opt_Input_nn = {'NN_model':None, 'features':None, 'feat_norm':'standard', 'class2':0}
Opt_Input_dfp = {'pv_removal':[0.01, 1.0, 4]}


Opt_Input_win = {'rms_change':0.5}

Opt_Input.update(Opt_Input_analysis)
Opt_Input.update(Opt_Input_thr)
Opt_Input.update(Opt_Input_cant)
Opt_Input.update(Opt_Input_nn)
Opt_Input.update(Opt_Input_win)
Opt_Input.update(Opt_Input_dfp)




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
		print('t burst11111')
		print(t_burst_corr1)
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
			clf_pickle1 = read_pickle(clf_pickle1)
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
	else:
		print('without clf check')
		clf_1 = None


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
	if config['method'] == 'ENVTHR':
		if config['processing'] == 'OFF':
			sys.exit()
			print('env thr must have processing')

	#++++++++++++++++++++++SIGNAL PROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	if config['denois'] != 'OFF':
		print('with denois')
		x1 = signal_denois(x=x1, denois=config['denois'], med_kernel=config['med_kernel'])
	else:
		print('without denois')
		
	if config['processing'] != 'OFF':
		print('with processing')
		x1 = signal_processing(x1, config)
	else:
		print('without processing')
	
	if config['diff'] != 'OFF':
		print('with diff')
		x1 = diff_signal_eq(x1, config['diff'])
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
		
		
		if (config['method'] == 'THR' or config['method'] == 'ENVTHR'):
		
			
		
		
			threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
			
			n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold1, t_window=config['window_time'])
			print('time TP111111111111111111111')
			
			if config['clf_check'] == 'ON':
				tWindows = prepare_windows(t, config)
				# tWindows = []
				# window_points = int(config['window_time']*config['fs'])
				# n_windows = int(n_points/window_points)
				# for count in range(n_windows):
					# tWindows.append(t[count*window_points:(count+1)*window_points])
				
				count = 0
				for twindow in tWindows:
					flag = 'OFF'
					if clf_1[count] == 2:
						clf_1[count] = config['class2']
					if clf_1[count] == 0:
						Results['NEG'] = Results['NEG'] + 1
						for time_burst in t_burst_corr1:
							if time_burst in twindow:
								print('++++++++++++FP in ', count*config['window_time'])
								Results['FP'] = Results['FP'] + 1
								flag = 'ON'
						if flag == 'OFF':
							Results['TN'] = Results['TN'] + 1
					elif clf_1[count] == 1:
						Results['POS'] = Results['POS'] + 1
						for time_burst in t_burst_corr1:
							if time_burst in twindow:
								# print(time_burst)
								Results['TP'] = Results['TP'] + 1
								flag = 'ON'
						if flag == 'OFF':
							Results['FN'] = Results['FN'] + 1
					else:
						print('error file clf')
						sys.exit()
					count = count + 1
			else:
				print('without clf check')

			# t_burst_corr1 = t_burst_corr1 + np.ones(len(t_burst_corr1))*warm*dt
		
		elif config['method'] == 'NN':		
			# Windows1 = []
			# window_points = int(config['window_time']*config['fs'])
			# window_advance = int(window_points*config['overlap'])
			# if config['overlap'] != 0:
				# print('Windows with overlap')
				# n_windows = int((n_points - window_points)/window_advance) + 1
			# else:
				# n_windows = int(n_points/window_points)
			# print('Number of windows: ', n_windows)
			
			
			# for count in range(n_windows):
				# if config['overlap'] != 0:
					# Windows1.append(x1[count*window_advance:window_points+window_advance*count])
				# else:
					# Windows1.append(x1[count*window_points:(count+1)*window_points])
			
			Windows1 = prepare_windows(x1, config)
			
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
				elif config['features'] == 'i10statsnmnsnknmin_lrstd':
					values = i10statsnmnsnknmin_lrstd(window1)
				elif config['features'] == 'i10maxminrms_lrrms':
					values = i10maxminrms_lrrms(window1)
				elif config['features'] == 'i10maxminstd_lrrmsstd':
					values = i10maxminstd_lrrmsstd(window1)
				elif config['features'] == 'i10statsnmnsnknmin_lrstd_lrnper5':
					values = i10statsnmnsnknmin_lrstd_lrnper5(window1)
				elif config['features'] == 'i10statsnmnsnk_lrstd_lrmeanper5':
					values = i10statsnmnsnk_lrstd_lrmeanper5(window1)
				elif config['features'] == 'i10statsnmnsnk_lrstd_lrnper5':
					values = i10statsnmnsnk_lrstd_lrnper5(window1)
					
					
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
				
				if config['clf_check'] == 'ON':
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
				else:
					print('without check')

				Predictions1.append(prediction[0])

				numero = numero + 1
				
			t_burst_corr1 = []
			amp_burst_corr1 = []
			for i in range(len(Predictions1)):
				if Predictions1[i] == 1:
					t_burst_corr1.append(i*config['window_time'])
					amp_burst_corr1.append(x1[int(i*config['window_time']*config['fs'])])
		
		elif config['method'] == 'WIN':
			print('not yet')
			Windows = prepare_windows(x1, config)
			RMSs = []
			for k in range(len(Windows)):
				RMSs.append(signal_rms(Windows[k]))
			
			Predictions = []
			Predictions.append(0)
			for k in range(len(RMSs)-1):
				if (RMSs[k+1] - RMSs[k] > config['rms_change']):
					Predictions.append(1)
				else:
					Predictions.append(0)
			t_burst_corr1 = []
			amp_burst_corr1 = []
			for i in range(len(Predictions)):
				if Predictions[i] == 1:
					if config['overlap'] == 0:
						t_burst_corr1.append(i*config['window_time'])
						amp_burst_corr1.append(x1[int((i)*config['window_time']*config['fs'])])
						
						# ventana = x1[int(i*config['window_time']*config['fs']) : int((i+1)*config['window_time']*config['fs'])]
						# amp_burst_corr1.append(np.max(ventana))
						# t_burst_corr1.append(i*config['window_time'] + np.argmax(ventana)/config['fs'])
						# print(amp_burst_corr1)
						# q = input('pause')
					else:
						#detecciones dobles
						t_burst_corr1.append(i*config['window_time']*config['overlap'])
						amp_burst_corr1.append(x1[int(i*config['window_time']*config['overlap']*config['fs'])])
			
			
			if config['clf_check'] == 'ON':

				tWindows = []
				window_points = int(config['window_time']*config['fs'])
				n_windows = int(n_points/window_points)
				for count in range(n_windows):
					tWindows.append(t[count*window_points:(count+1)*window_points])
				
				if config['overlap'] == 0:
					count = 0
					for twindow in tWindows:
						flag = 'OFF'
						if clf_1[count] == 2:
							clf_1[count] = config['class2']
						if clf_1[count] == 0:
							Results['NEG'] = Results['NEG'] + 1
							for time_burst in t_burst_corr1:
								if time_burst in twindow:
									print('++++++++++++FP in ', count*config['window_time'])
									print('mmmmmmmmmmm FP in ', time_burst)
									Results['FP'] = Results['FP'] + 1
									flag = 'ON'
							if flag == 'OFF':
								Results['TN'] = Results['TN'] + 1
						elif clf_1[count] == 1:
							Results['POS'] = Results['POS'] + 1
							for time_burst in t_burst_corr1:
								if time_burst in twindow:
									# print(time_burst)
									Results['TP'] = Results['TP'] + 1
									flag = 'ON'
							if flag == 'OFF':
								Results['FN'] = Results['FN'] + 1
						else:
							print('error file clf')
							sys.exit()
						count = count + 1
				else:
					config_overlap = config
					config_overlap['overlap'] = 0
					tWindows = prepare_windows(t, config_overlap)
				
					contador = 0
					Results['NEG'] = Results['NEG'] + 1
					Results['TN'] = Results['TN'] + 1
					for k in range(len(tWindows)-1):

						contador = contador + 1
						flag = 'OFF'
						if clf_1[k] == 2:
							clf_1[k] = config['class2']
						if clf_1[k+1] == 2:
							clf_1[k+1] = config['class2']
						
						if clf_1[k] == 0:
							Results['NEG'] = Results['NEG'] + 1
							for time_burst in t_burst_corr1:
								if (time_burst in tWindows[k]):
									print(time_burst)
									a = input('pause11119438576')
									
									if clf_1[k+1] == 0:
										Results['FP'] = Results['FP'] + 1
										# print('++++++++++++FP in ', time_burst)
										flag = 'ON'
									elif clf_1[k+1] == 1:
										Results['TP'] = Results['TP'] + 1
										print('++++++++++++TP in ', time_burst)
										flag = 'ON'
									else:
										print(clf_1[k+1])
										print('error clf 125')
									
							if flag == 'OFF':
								Results['TN'] = Results['TN'] + 1
								
								
						elif clf_1[k] == 1:
							Results['POS'] = Results['POS'] + 1
							for time_burst in t_burst_corr1:
								if time_burst in tWindows[k]:
									print(time_burst)
									a = input('jajajajaaja pausa')
									# print(time_burst)
									Results['TP'] = Results['TP'] + 1
									print('++++++++++++TP in ', time_burst)
									flag = 'ON'
							if flag == 'OFF':
								Results['FN'] = Results['FN'] + 1
						else:
							print('error file clf')
							sys.exit()
						count = count + 1
						
					# x1 = RMSs
					if clf_1[contador] == 2:
						clf_1[contador] = config['class2']
				
				
				
			else:
				print('without clf check')
			
			
			
			
			
			
			
		
		elif config['method'] == 'DFP':
			x1 = np.log10(1. + np.absolute(x1))
			# # df_x = x1
			# plt.figure(0)
			# plt.plot(x1)
			

			
			
			locs = dfp_alg2(x1)

			
			# plt.figure(1)
			# plt.plot(df_x)
			level_ini = config['pv_removal'][0]
			level_fin = config['pv_removal'][1]
			steps = config['pv_removal'][2]
			locs = dfp_alg3(x1, locs, level_ini, level_fin, steps)
			# level = 0.1
			# for i in range(len(locs)):
				# if (locs[i] == 1 or locs[i] == -1):
					# flag = False
					# count = 0
					# while flag == False:
						# count = count + 1
						# if i+count >= len(locs):
							# break
						# if (locs[i+count] == 1 or locs[i+count] == -1):
							# flag = True
					# if flag == True:
						# dif  = np.absolute(df_x[i+count] - df_x[i])
						# if dif < level:
							# locs[i] = 0
							# locs[i+count] = 0
			
			
			for i in range(len(x1)):
				if locs[i] == 0:
					x1[i] = 0
			
			
			# plt.figure(1)
			# plt.plot(x1)
					
					
			
			
			# plt.show()
			
			threshold1 = config['thr_value']
			n_burst_corr1, t_burst_corr1, amp_burst_corr1, t_burst1, amp_burst1 = id_burst_threshold(x=x1, fs=config['fs'], threshold=threshold1, t_window=config['window_time'])
			
			
			if config['clf_check'] == 'ON':
				tWindows = prepare_windows(t, config)
				# tWindows = []
				# window_points = int(config['window_time']*config['fs'])
				# n_windows = int(n_points/window_points)
				# for count in range(n_windows):
					# tWindows.append(t[count*window_points:(count+1)*window_points])
				
				count = 0
				for twindow in tWindows:
					flag = 'OFF'
					if clf_1[count] == 2:
						clf_1[count] = config['class2']
					if clf_1[count] == 0:
						Results['NEG'] = Results['NEG'] + 1
						for time_burst in t_burst_corr1:
							if time_burst in twindow:
								print('++++++++++++FP in ', count*config['window_time'])
								Results['FP'] = Results['FP'] + 1
								flag = 'ON'
						if flag == 'OFF':
							Results['TN'] = Results['TN'] + 1
					elif clf_1[count] == 1:
						Results['POS'] = Results['POS'] + 1
						for time_burst in t_burst_corr1:
							if time_burst in twindow:
								# print(time_burst)
								Results['TP'] = Results['TP'] + 1
								flag = 'ON'
						if flag == 'OFF':
							Results['FN'] = Results['FN'] + 1
					else:
						print('error file clf')
						sys.exit()
					count = count + 1
			else:
				print('without clf check')
			
			
			

			# sys.exit()
			
		else:
			print('Unknown Method')
			sys.exit()
			
			
			
	#++++++++++++++++++++++ PLOT +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# if config_filter['analysis'] == True:
		# Name = Name + ' FIL'
	# if config_autocorr['analysis'] == True:
		# Name = Name + ' ACR'
	# if config_demod['analysis'] == True:
		# Name = Name + ' ENV'		
	# if config_diff['analysis'] == True:
		# Name = Name + ' DIF'
	if config['clf_check'] == 'ON':
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
	config_input['overlap'] = float(config_input['overlap'])
	config_input['window_time'] = float(config_input['window_time'])
	config_input['thr_value'] = float(config_input['thr_value'])
	config_input['rms_change'] = float(config_input['rms_change'])
	config_input['window_extend'] = float(config_input['window_extend'])
	
	# config_input['demod_prefilter'][3] = float(config_input['demod_prefilter'][3]) #order
	# config_input['demod_filter'][2] = float(config_input['demod_filter'][2]) #order
	# config_input['demod_filter'][1] = float(config_input['demod_filter'][1]) #lowpass freq
	
	# config_input['demod_prefilter'][2] = float(config_input['demod_prefilter'][2]) #high freq bandpass
	# config_input['demod_prefilter'][1] = float(config_input['demod_prefilter'][1]) #low freq bandpass
	
	#Type conversion to int
	config_input['n_files'] = int(config_input['n_files'])
	config_input['power2'] = int(config_input['power2'])
	config_input['warm_points'] = int(config_input['warm_points'])
	config_input['diff_points'] = int(config_input['diff_points'])
	config_input['class2'] = int(config_input['class2'])
	config_input['med_kernel'] = int(config_input['med_kernel'])
	
	# Variable conversion
	config_input['pv_removal'] = [float(config_input['pv_removal'][0]), float(config_input['pv_removal'][1]), int(config_input['pv_removal'][2])]
	if config_input['demod_prefilter'] != None:
		if config_input['demod_prefilter'][0] == 'bandpass':
			config_input['demod_prefilter'] = [config_input['demod_prefilter'][0], [float(config_input['demod_prefilter'][1]), float(config_input['demod_prefilter'][2])], float(config_input['demod_prefilter'][3])]
		elif config_input['demod_prefilter'][0] == 'highpass':
			config_input['demod_prefilter'] = [config_input['demod_prefilter'][0], float(config_input['demod_prefilter'][1]), float(config_input['demod_prefilter'][2])]
		else:
			print('error prefilter')
			sys.exit()
	if config_input['demod_filter'] != None:
		config_input['demod_filter'] = [config_input['demod_filter'][0], float(config_input['demod_filter'][1]), float(config_input['demod_filter'][2])]
	if config_input['diff'] != 'OFF':
		config_input['diff'] = float(config_input['diff'])


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

	ax[nax].plot(t, x1/signal_rms(x1), color=color)
	if (config['method'] == 'THR' or config['method'] == 'DFP' or config['method'] == 'ENVTHR'):
		if thr == True:
			threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
			ax[nax].axhline(threshold1, color='k')		
		ax[nax].plot(t_burst_corr1, amp_burst_corr1/signal_rms(x1), 'ro')
	elif (config['method'] == 'NN' or config['method'] == 'WIN'):
		for i in range(len(t_burst_corr1)):
			# ax[nax].axvspan(xmin=t_burst_corr1[i], xmax=t_burst_corr1[i]+config['window_time'], facecolor='r', alpha=0.5)
			ax[nax].plot(t_burst_corr1[i] + config['window_time']/2.0, 0., 'ro')
	if clf != None:
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		for k in range(len(clf)):
			if clf[k] == 2:
				print('error clf 895')
				sys.exit()
		ind_w_positives = [k for k in range(len(clf)) if clf[k] == 1]
		print(len(ind_w_positives))
		print(len(clf))
		for i in range(len(ind_w_positives)):
			ax[nax].axvspan(xmin=config['window_time']*ind_w_positives[i], xmax=config['window_time']*(ind_w_positives[i] + 1), facecolor='y', alpha=0.5)
	
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
		else:
			name = name + '1500RPM / 40% Load'
	else:
		name = name + '1000RPM / 80% Load'
	ax[nax].set_title(name, fontsize=10)
	ax[nax].set_ylabel('Norm. Amplitude')
	
	
	
	
	# ax[nax].set_title(name + config['channel'] + ' ' + config['method'] + '\n' + config['filename'], fontsize=10)
	# ax[nax].set_ylabel('Amplitude')
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
	if config['method'] == 'WIN':
		Windows = prepare_windows(x1, config)
		RMSs = []
		t2 = []
		for k in range(len(Windows)):
			RMSs.append(signal_rms(Windows[k]))
		for u in range(len(RMSs)):
			t2.append((u+1)*config['window_time'] - config['window_time']/2.0)
		ax[nax].plot(t2, RMSs, '-s')
	
	if (config['method'] == 'THR' or config['method'] == 'DFP' or config['method'] == 'ENVTHR'):
		if thr == True:
			threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
			ax[nax].axhline(threshold1, color='k')		
		ax[nax].plot(t_burst_corr1, amp_burst_corr1, 'ro')
	elif config['method'] == 'NN' or config['method'] == 'WIN':
		for i in range(len(t_burst_corr1)):
			ax[nax].axvspan(xmin=t_burst_corr1[i], xmax=(t_burst_corr1[i]+config['window_time']), facecolor='r', alpha=0.5)
			# ax[nax].plot(t_burst_corr1[i], amp_burst_corr1[i], 'ro')
	if clf != None:
		for k in range(len(clf)):
			if clf[k] == 2:
				print('error clf')
				sys.exit()
		ind_w_positives = [k for k in range(len(clf)) if clf[k] == 1]
		# print(len(ind_w_positives))
		# print(len(clf))
		for i in range(len(ind_w_positives)):
			ax[nax].axvspan(xmin=config['window_time']*ind_w_positives[i], xmax=config['window_time']*(ind_w_positives[i] + 1), facecolor='b', alpha=0.5)
		
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
		else:
			name = name + '1500RPM / 40% Load'
	else:
		name = name + '1000RPM / 80% Load'
	ax[nax].set_title(name, fontsize=10)
	# ax[nax].set_ylabel('Amplitude')
	ax[nax].set_ylabel('DF Amplitude')
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

def prepare_windows(x, config):
	n_points = len(x)
	Windows1 = []
	window_points = int(config['window_time']*config['fs'])
	window_advance = int(window_points*config['overlap'])
	if config['window_extend'] != 0:
		print('With window extend')
		window_extend = int(config['window_extend']*config['fs'])
	else:
		window_extend = 0
	
	if config['overlap'] != 0:
		print('Windows with overlap')
		n_windows = int((n_points - window_points)/window_advance) + 1
	else:
		n_windows = int(n_points/window_points)
	print('Number of windows: ', n_windows)
	
	
	for count in range(n_windows):
		if config['overlap'] != 0: #with overlap not working
			Windows1.append(x[count*window_advance:window_points+window_advance*count])
		else:
			if count != 0:
				Windows1.append(x[(count*window_points - window_extend):(count+1)*window_points])
			else:
				Windows1.append(x[(count*window_points):(count+1)*window_points])
	return Windows1

def dfp_alg2(df_x):
	peaks_ind = []
	valleys_ind = []
	locs = np.zeros(len(df_x))
	for i in range(len(df_x)-2):
		if (df_x[i+1] > df_x[i] and df_x[i+1] > df_x[i+2]):
			peaks_ind.append(i)
			locs[i+1] = 1
		elif (df_x[i+1] < df_x[i] and df_x[i+1] < df_x[i+2]):
			valleys_ind.append(i)
			locs[i+1] = -1
	return locs

def dfp_alg3(df_x, locs, level_ini, level_fin, steps):
	Levels = np.linspace(level_ini, level_fin, steps)
	for level in Levels:
		for i in range(len(locs)):
			if (locs[i] == 1 or locs[i] == -1):
				flag = False
				count = 0
				while flag == False:
					count = count + 1
					if i+count >= len(locs):
						break
					if (locs[i+count] == 1 or locs[i+count] == -1):
						flag = True
				if flag == True:
					dif  = np.absolute(df_x[i+count] - df_x[i])
					if dif < level:
						locs[i] = 0
						locs[i+count] = 0
		for i in range(len(df_x)):
			if locs[i] == 0:
				df_x[i] = 0
		locs = dfp_alg2(df_x)
	return locs
	
	
	

if __name__ == '__main__':
	main(sys.argv)