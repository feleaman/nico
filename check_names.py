import os
import sys
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from m_open_extension import *
from m_fft import *

from openpyxl import load_workbook

import matplotlib
import cmath
from scipy import signal
from Burst_Detection import burst_detector



print('++++++++Select Signal ')
root = Tk()
root.withdraw()
root.update()
filepath = filedialog.askopenfilename()			
root.destroy()
# if config['channel'] == 'all':
	# Channels = ['0', '1', '2']		

channel = 'AE_2'

signal = load_signal(filepath, channel=channel)
plt.plot(signal)
plt.show()

# Inputs = ['mode', 'fs', 'channel', 'method', 'plot']

# InputsOpt_Defaults = {'demod_filter':['lowpass', 2000., 3], 'demod_prefilter':['highpass', 70.e3, 3], 'diff':1, 'thr_value':0.0004, 'processing':'butter_demod', 'thr_mode':'fixed_value', 'demod_dc':'without_dc', 'demod_rect':'only_positives', 'clf_check':'OFF', 'data_norm':'per_rms', 'denois':'OFF', 'window_time':0.001}


# def main(argv):
	# if
		
	# else:
		# print('unknown mode')
		# sys.exit()

		
		

		
		
	# return



# def read_parser(argv, Inputs, InputsOpt_Defaults):
	# Inputs_opt = [key for key in InputsOpt_Defaults]
	# Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	# parser = ArgumentParser()
	# for element in (Inputs + Inputs_opt):
		# print(element)
		# if element == 'no_element':
			# parser.add_argument('--' + element, nargs='+')
		# else:
			# parser.add_argument('--' + element, nargs='?')
	
	# args = parser.parse_args()
	# config = {}
	# for element in Inputs:
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Required:', element)
			# sys.exit()

	# for element, value in zip(Inputs_opt, Defaults):
		# if getattr(args, element) != None:
			# config[element] = getattr(args, element)
		# else:
			# print('Default ' + element + ' = ', value)
			# config[element] = value
	
	# #Type conversion to float
	# # config['value'] = float(config['value'])
	# config['fs'] = float(config['fs'])
	# # config['fscore_min'] = float(config['fscore_min'])
	# #Type conversion to int	
	# # Variable conversion
	# return config




# if __name__ == '__main__':
	# main(sys.argv)
