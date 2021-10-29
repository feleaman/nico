# Generate_Features.py
# Last updated: 15.08.2017 by Felix Leaman
# Description:
# Code for generate features
#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import time
from datetime import datetime
# import os.path
import pandas as pd
import os
import csv
from scipy import stats

import sys
sys.path.insert(0, './lib')

from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *

from os.path import isfile, join

# plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
start_time = time.time()

#++++++++++++++++++++++ CONFIGURATION ++++++++++++++++++++++++++++++++++++++++++++++
channel = 'AE_Signal'
# channel = 'Koerperschall'
# channel = 'Drehmoment'
# mypath = 'C:\Felix\Daten\CNs_Getriebe\Faulty\AE' #Path where files are
# mypath = 'M:\Betriebsmessungen\Getriebeprüfstand CNs\Getriebe_i_O_20160506\AE'
# mypath = 'M:\Betriebsmessungen\Getriebeprüfstand CNs\Getriebe_i_O_20160506\AE'
mypath = 'M:\Betriebsmessungen\Getriebeprüfstand CNs\Getriebe_Hohlradschaden\AE'
mypath = 'M:\Betriebsmessungen\Getriebeprüfstand CNs\Getriebe_Hohlradschaden\AE'
# n_points = 2**26 #Number of points of the signal, if None, it uses the total length in power of 2
n_points = 2**20

rpms = ['n500', 'n1000', 'n1500']
loads = ['M40', 'M80']
n_tests = 3 

#++++++++++++++++++++++ MAIN ++++++++++++++++++++++++++++++++++++++++++++++
#++++++Sampling Frequency Assignment
if channel == 'Koerperschall':
	fs = 1000.0
elif channel == 'Drehmoment':
	fs = 1000.0
elif channel == 'AE_Signal':
	fs = 1000000.0
else:
	print('Error fs assignment')
dt = 1.0/fs


#++++++List of Files in Directory

filenames = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

features = [
# 'KURT_WFM_0',
# 'RMS_WFM_0',
# 'LP6_WFM_0',
# 'LP7_WFM_0',
# 'LP16_WFM_0',
# 'LP17_WFM_0',
# 'LP21_WFM_0',
# 'LP24_WFM_0',
# 'LP16_FFT_0',
# 'LP17_FFT_0',
# 'LP21_FFT_0',
# 'LP24_FFT_0',
'NBU_WFM_0',
# 'NBU_WFM_1',
'NBU_DIF_0',
# 'NBU_DIF_1',
'NBU_DIF_ENV_0',
# 'NBU_DIF_ENV_1',
# 'NBU_ENV_STF_BIN_0',
# 'NBU_DIF_ENV_STF_BIN_0'
]






features_dicts = [{} for feature in features]

writers = [pd.ExcelWriter(feature + '.xlsx', engine='openpyxl') for feature in features]

for test in range(n_tests):
	print('Working on: V' + str(test+1))
	test = test + 1
	ctimer = 0
	for rpm in rpms:
		ctimer = ctimer + 1
		print('...' + str(ctimer) + '/3' + '...')
		features_lists = [[] for feature in features]

		for load in loads:
			filename = [f for f in filenames if f.find(rpm) != -1 if f.find(load) != -1 if f[1] == str(test)]
			if len(filename) != 1:
				print('error file finding')
				# print(len(filename))
				sys.exit()
			filename = filename[0]
			filepath = join(mypath, filename)
			x = f_open_mat(filepath, channel)
			x = np.ndarray.flatten(x)
			
			x = x[0:n_points]	
			magX, f, df = mag_fft(x, fs)
			difX = diff_signal_eq(x=x, length_diff=1)
			envdifX = butter_demodulation(x=difX, fs=fs, filter=['lowpass', 50.0, 3], prefilter=['bandpass',[180.0e3, 350.0e3], 3], 
			type_rect='only_positives', dc_value='without_dc')
			threshold = 8*signal_rms(x)
			t_window = 0.002
			
			# print(len(x))
			# print(max_2power(len(x)))
			# sys.exit()
			for i in range(len(features)):
				print(i)
				value = features_master(name=features[i], x=x, dt=dt, magX=magX, difX=difX, envdifX=envdifX, df=df, fs=fs, threshold=threshold, t_window=t_window)
				features_lists[i].append(value)
				print('Ready Feature ' + str(i) + ' from ' + str(len(features)))
		
		for i in range(len(features_dicts)):		
			features_dicts[i][rpm] = pd.Series(features_lists[i], loads)

	for i in range(len(features_dicts)):
		features_dicts[i] = pd.DataFrame(features_dicts[i], columns=rpms)
		features_dicts[i].to_excel(writers[i], sheet_name='V' + str(test), header=True, index=True, engine='openpyxl')

for i in range(len(writers)):
	writers[i].save()

#############
for feature in features:
	dic = pd.read_excel(feature + '.xlsx', sheetname=None)

	dic['VM'] = dic['V1']
	dic['Info'] = {'Datetime':datetime.now(), 'N_Points':n_points, 'F_Sampling':fs, 'Path':mypath}
	
	# print(type(pd.DataFrame(data=dic['Info'], index=[i for i in range(4)] )))
	
	dic['Info'] = pd.DataFrame(data=dic['Info'], index=[i for i in range(1)] )
	
	# sys.exit()
	for i in range(n_tests-1):
		i = i + 2
		dic['VM'] = dic['VM'] + dic['V' + str(i)]
	dic['VM'] = dic['VM'] / n_tests
	

	
	dic['VS'] = (dic['V1'] - dic['VM'])**2
	for i in range(n_tests-1):
		i = i + 2
		dic['VS'] = dic['VS'] + (dic['V' + str(i)] - dic['VM'])**2
	dic['VS'] = (dic['VS'] / n_tests)**0.5
	
	# dic['Info'] = {}


	writer = pd.ExcelWriter(feature + '.xlsx')
	for key in sorted(dic):
		pd.DataFrame(dic[key]).to_excel(writer, sheet_name=key)

	writer.save()







print("--- %s seconds ---" % (time.time() - start_time))
sys.exit()



