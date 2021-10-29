# XValidation.py
# Last updated: 14.11.2017 by Felix Leaman
# Description:


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
Inputs = ['channel', 'fs', 'power2', 'method', 'n_files', 'save', 'save_plot']
Opt_Input = {'files':None, 'window_time':0.001, 'overlap':0, 'data_norm':None, 'clf_files':None, 'clf_check':'OFF', 'plot':'ON', 'save_name':'NAME'}
Opt_Input_analysis = {'EMD':'OFF', 'denois':'OFF', 'med_kernel':3, 'processing':'OFF', 'demod_filter':None, 'demod_prefilter':None, 'demod_rect':None, 'demod_dc':None, 'diff':'OFF'}
Opt_Input_thr = {'thr_mode':'factor_rms', 'thr_value':1}
Opt_Input_cant = {'demod':None, 'prefilter':None, 'postfilter':None, 'rectification':None, 'dc_value':None, 'warm_points':0, 'window_delay':0}
Opt_Input_nn = {'NN_model':None, 'features':None, 'feat_norm':'standard', 'class2':None, 'classes':None}
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
	Filepaths
	Classifications
	
	Signals = []
	for filepath in Filepaths:
		Signals.append(load_signal(filepath, channel=config['channel'])[0:2**(config['power2'])])
	
	Processed_Signals = []
	for x in Signals:
		#Normalization
		if config['data_norm'] == 'per_signal':
			x = x / np.max(np.absolute(x))
			x = x.tolist()
			print('normalization per signal!!!!!!')
		elif config['data_norm'] == 'per_rms':
			x = x / signal_rms(x)
			x = x.tolist()
			print('normalization per RMS!!!!!!')
		
		#Denoising
		if config['denois'] != 'OFF':
			print('with denois')
			x = signal_denois(x=x, denois=config['denois'], med_kernel=config['med_kernel'])
		else:
			print('without denois')
			
		#Processing
		if config['processing'] != 'OFF':
			print('with processing')
			# print(config['processing'])
			x = signal_processing(x, config)
		else:
			print('without processing')
			
		#Differentiate
		if config['diff'] != 'OFF':
			print('with diff')
			x = diff_signal_eq(x, config['diff'])
		
		Processed_Signals.append(x)
	
	
	return





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
	config_input['window_delay'] = float(config_input['window_delay'])
	
	# config_input['demod_prefilter'][3] = float(config_input['demod_prefilter'][3]) #order
	# config_input['demod_filter'][2] = float(config_input['demod_filter'][2]) #order
	# config_input['demod_filter'][1] = float(config_input['demod_filter'][1]) #lowpass freq
	
	# config_input['demod_prefilter'][2] = float(config_input['demod_prefilter'][2]) #high freq bandpass
	# config_input['demod_prefilter'][1] = float(config_input['demod_prefilter'][1]) #low freq bandpass
	
	#Type conversion to int
	config_input['n_files'] = int(config_input['n_files'])
	config_input['power2'] = int(config_input['power2'])
	config_input['warm_points'] = int(config_input['warm_points'])
	# config_input['diff_points'] = int(config_input['diff_points'])
	config_input['class2'] = int(config_input['class2'])
	config_input['med_kernel'] = int(config_input['med_kernel'])
	if config_input['diff'] != 'OFF':
		config_input['diff'] = int(config_input['diff'])
	
	# Variable conversion
	config_input['pv_removal'] = [float(config_input['pv_removal'][0]), float(config_input['pv_removal'][1]), int(config_input['pv_removal'][2])]
	if config_input['demod_prefilter'][0] != 'OFF':
		if config_input['demod_prefilter'][0] == 'bandpass':
			config_input['demod_prefilter'] = [config_input['demod_prefilter'][0], [float(config_input['demod_prefilter'][1]), float(config_input['demod_prefilter'][2])], float(config_input['demod_prefilter'][3])]
		elif config_input['demod_prefilter'][0] == 'highpass':
			config_input['demod_prefilter'] = [config_input['demod_prefilter'][0], float(config_input['demod_prefilter'][1]), float(config_input['demod_prefilter'][2])]
		else:
			print('error prefilter')
			sys.exit()
	if config_input['demod_filter'] != None:
		config_input['demod_filter'] = [config_input['demod_filter'][0], float(config_input['demod_filter'][1]), float(config_input['demod_filter'][2])]
	


	return config_input
	
#Signal RAW

	

if __name__ == '__main__':
	main(sys.argv)