# import os
from os import listdir
import matplotlib.pyplot as plt

from Kurtogram3 import Fast_Kurtogram_filters
from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))
# from sys import exit
# from sys.path import path.insert
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import numpy as np
# import pandas as pd
from m_open_extension import *
from m_det_features import *
from Genetic_Filter import *
from M_Wavelet import *

# from THR_Burst_Detection import full_thr_burst_detector
# from THR_Burst_Detection import read_threshold
# from THR_Burst_Detection import plot_burst_rev


from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime


print('hello world!')

root = Tk()
root.withdraw()
root.update()
Filepaths = filedialog.askopenfilenames()
root.destroy()


for filepath in Filepaths:
	mydict = pd.read_excel(filepath, sheetname='OV_Features')
	row_names = list(mydict.index)
	
	
	mydict = mydict.to_dict(orient='list')
	
	mydict2 = {}

	
	row_names.append('Mean')
	row_names.append('Std')
	for key in mydict.keys():
		mydict2[key] = mydict[key]
		
		mean = np.mean(mydict[key])
		std = np.std(mydict[key])
		mydict2[key].append(mean)
		mydict2[key].append(std)
	
	DataFr = pd.DataFrame(data=mydict2, index=row_names)
	writer = pd.ExcelWriter('_' + basename(filepath))

	
	DataFr.to_excel(writer, sheet_name='OV_Features')	
	print('Result in Excel table')
	writer.close()
