# Main_Analysis.py
# Last updated: 15.08.2017 13:13 by Felix Leaman
# Description:
# Code for opening a .mat or .tdms data file with single channel and plotting different types of analysis
# The file and channel is selected by the user
# Channel must be 'AE_Signal', 'Koerperschall', or 'Drehmoment'. Defaults sampling rates are 1000kHz, 1kHz and 1kHz, respectively

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

import os.path
import sys


sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'save']
InputsOpt_Defaults = {'power2':'OFF', 'name':'auto'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	from Main_Analysis import main as invoke_signal
	
	if config['mode'] == 'spectrum':
	
	
		t, x, f, magX = invoke_signal(['--channel', 'AE_3', '--fs', '1.e6', '--power2', 'OFF', '--plot', 'OFF'])
		plt.show()
		
		df= f[2] - f[1]
		
		f_fund = 40.2093
		n_harm = 5		
		F_harm = [(i+1)*f_fund for i in range(n_harm)]
		
		NL_harm = [f_harm/df for f_harm in F_harm]
		
		AMP_harm = [magX[int(nl_harm)] for nl_harm in NL_harm]
		
		plt.plot(f, magX, 'r', F_harm, AMP_harm, 'o')
		
		plt.show()
		
		

	
	else:
		print('unknown mode')
		
		
		
	return

# plt.show()
def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	
	#Type conversion to float
	if config['power2'] != 'auto' and config['power2'] != 'OFF':
		config['power2'] = int(config['power2'])
	# config['mode'] = float(config['fs_tacho'])
	# config['fs_signal'] = float(config['fs_signal'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


if __name__ == '__main__':
	main(sys.argv)
