# Reco_Signal_Training.py
# Last updated: 24.08.2017 by Felix Leaman
# Description:
# 

#++++++++++++++++++++++ IMPORT MODULES +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cm
# from tkinter import filedialog
# from skimage import img_as_uint
from tkinter import Tk
from tkinter import Button
# import skimage.filters
from tkinter import filedialog
from tkinter import Tk
import os.path
import sys
sys.path.insert(0, './lib') #to open user-defined functions

from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from os.path import isfile, join
import pickle
import argparse
from sklearn.neural_network import MLPClassifier

import datetime
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes




	
#+++++++++++++++++++++++++++FUNCTIONS++++++++++++++++++++++++++++++++++++++++++
def save_pickle(pickle_name, pickle_data):
	pik = open(pickle_name, 'wb')
	pickle.dump(pickle_data, pik)
	pik.close()

def read_pickle(pickle_name):
	pik = open(pickle_name, 'rb')
	pickle_data = pickle.load(pik)
	return pickle_data




classification_pickle = filedialog.askopenfilename()

info_classification_pickle = read_pickle(classification_pickle)
classification_pickle = os.path.basename(classification_pickle)


info_classification_pickle['config_analysis']['start_in'] = 0

save_pickle(classification_pickle, info_classification_pickle)






