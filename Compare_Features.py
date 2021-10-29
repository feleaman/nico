# Compare_Features.py
# Last updated: 15.08.2017 by Felix Leaman
# Description:
# Code for compare features between different states

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
import time
# import os.path
import os
import csv
import collections
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, './lib')

from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
import inspect
# from os import listdir
from os.path import isfile, join


plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
start_time = time.time()
###########################################################################
path_faulty = 'C:\Felix\Code\Working\Faulty'
path_ok = 'C:\Felix\Code\Working\OK'

# print(sys.argv[1:])

# print(m)






features = [
'KURT_WFM_0',
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
]
# for feature in features:
	# features = join(mypath, feature)
# print(features)

features_faulty = [join(path_faulty, feature) for feature in features]
features_ok = [join(path_ok, feature) for feature in features]	

# print(features)
# sys.exit()

for (feature_faulty, feature_ok) in zip(features_faulty, features_ok):

	#Load Files:Mean
	bookfile_faulty = pd.read_excel(feature_faulty + '.xlsx', sheetname='VM', header=0, index_col=0)
	bookfeat_faulty = bookfile_faulty.transpose()
	
	bookfile_ok = pd.read_excel(feature_ok + '.xlsx', sheetname='VM', header=0, index_col=0)
	bookfeat_ok = bookfile_ok.transpose()	
	
	
	#Load Files:STD
	std_per_rpm_faulty = pd.read_excel(feature_faulty + '.xlsx', sheetname='VS', header=0, index_col=0)
	n_loads = len(std_per_rpm_faulty)	
	std_per_load_faulty = std_per_rpm_faulty.transpose()
	xerr_faulty = []
	i = 0
	for load in std_per_rpm_faulty.index.values:
		xerr_faulty.append(std_per_load_faulty[load].tolist())
		i = i + 1
	
	std_per_rpm_ok = pd.read_excel(feature_ok + '.xlsx', sheetname='VS', header=0, index_col=0)
	n_loads = len(std_per_rpm_ok)	
	std_per_load_ok = std_per_rpm_ok.transpose()
	xerr_ok = []
	i = 0
	for load in std_per_rpm_ok.index.values:
		xerr_ok.append(std_per_load_ok[load].tolist())
		i = i + 1
	
	

	
	#Plots
	fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
	
	bookfeat_faulty.plot(use_index=True, title=os.path.basename(feature_faulty)+'_Faulty', kind='barh', xerr=xerr_faulty
	, capsize=4, rot=0, ax=axes[0], legend=False)
	lims_x_faulty = axes[0].get_xlim()
	# print(lims_x_faulty)
	range_x_faulty = np.linalg.norm(np.asarray(lims_x_faulty))
	
	bookfeat_ok.plot(use_index=True, title=os.path.basename(feature_ok)+'_OK', kind='barh', xerr=xerr_ok
	, capsize=4, rot=0, ax=axes[1])	
	lims_x_ok = axes[1].get_xlim()
	# print(lims_x_ok)
	range_x_ok = np.linalg.norm(np.asarray(lims_x_ok))
	
	
	

	
	if range_x_faulty >= range_x_ok: #no hay manejo de features con escalas negativas y positivas
		if os.path.basename(feature_faulty) == 'LP6_WFM_0':
			axes[1].set_xlim((-lims_x_faulty[0], lims_x_faulty[1]))
		else:
			axes[1].set_xlim(lims_x_faulty)
	else:
		axes[0].set_xlim(lims_x_ok)
	
	axes[0].invert_xaxis()
	

	plt.subplots_adjust(wspace=0, hspace=0)
	# plt.savefig(os.path.basename(feature_faulty)+'.png')


plt.show()



sys.exit()
