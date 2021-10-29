# Plot_Features.py
# Last updated: 15.08.2017 by Felix Leaman
# Description:
# Code for plotting features

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

# from os import listdir
from os.path import isfile, join



plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
start_time = time.time()
###########################################################################

features = [
'KURT_WFM_0',
# 'RMS_WFM_0',
# 'LP6_WFM_0',
# 'LP7_WFM_0',
# 'LP16_WFM_0',
# 'LP17_WFM_0',
# 'LP21_WFM_0',
# 'LP24_WFM_0',
]


for feature in features:
	bookfile = pd.read_excel(feature + '.xlsx', sheetname='VM', header=0, index_col=0)
	bookfeat = bookfile.transpose()
	
	std_per_rpm = pd.read_excel(feature + '.xlsx', sheetname='VS', header=0, index_col=0)
	n_loads = len(std_per_rpm)
	
	std_per_load = std_per_rpm.transpose()

	yerr = []

	i = 0
	for load in std_per_rpm.index.values:
		yerr.append(std_per_load[load].tolist())
		i = i + 1

	ax = bookfeat.plot(use_index=True, title=feature, kind='bar', yerr=yerr
	, capsize=4, rot=0)
	ax.plot()

plt.show()



sys.exit()
