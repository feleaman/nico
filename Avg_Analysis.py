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
	
	if config['mode'] == 'avg_spectrum':
		n_avg = 3
		magX_list =  []
		for i in range(n_avg):
			t, x, f, magX = invoke_signal(['--channel', 'AE_3', '--fs', '10.e6', '--power2', 'OFF', '--plot', 'OFF'])
			if i == 0:
				magX_avg = np.zeros(len(magX))
			magX_avg = magX_avg + magX
		magX_avg = magX_avg / float(n_avg)
		
		if config['save'] == 'ON':
			mydict = {}
			mydict['freq_array'] = f
			mydict['avg_spectrum'] = magX_avg		
			
			if config['name'] == 'auto':
				import datetime
				stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
			else:
				stamp = config['name']
			save_pickle('avg_spectrum_' + stamp + '.pkl', mydict)
		else:
			print('no save')
		

	elif config['mode'] == 'plot_spectrum':
		
		magX_avg_list = []
		f_list = []
		
		root = Tk()
		root.withdraw()
		root.update()
		Filenames = filedialog.askopenfilenames()
		root.destroy()	
		
		for filename in Filenames:
			print(os.path.basename(filename))
			mydict = read_pickle(filename)		
			f_list.append(mydict['freq_array'])
			magX_avg_list.append(mydict['avg_spectrum'])
			
		

		channel = 'AE_1'
		element = 'Mag-FFT'
		fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
		# ax[].set_title(channel + ' ' + element + '\n' + os.path.basename(filename), fontsize=10)
		
		ax[0].set_title('BMB am Punkt D', fontsize=10, fontweight='bold')
		ax[1].set_title('Drahtbuerste am Punkt F', fontsize=10, fontweight='bold')
		ax[2].set_title('Kugelfall Maximale Hoehe', fontsize=10, fontweight='bold')
		ax[0].title.set_position([0.5, 1.0])
		ax[1].title.set_position([0.5, 0.8])
		ax[2].title.set_position([0.5, 0.8])

		
		
		ax[0].plot(f_list[0]/1000., magX_avg_list[0], 'r')
		ax[1].plot(f_list[1]/1000., magX_avg_list[1], 'r')
		ax[2].plot(f_list[2]/1000., magX_avg_list[2], 'r')
		
	
		ax[2].set_xlabel('Frequenz kHz')
		ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax[0].set_ylabel('Amplitude (m$V_{in}$)')
		ax[1].set_ylabel('Amplitude (m$V_{in}$)')
		ax[2].set_ylabel('Amplitude (m$V_{in}$)')
		
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
