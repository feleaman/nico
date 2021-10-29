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
from os import chdir
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))

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
Inputs = ['channel', 'mode', 'save']
InputsOpt_Defaults = {'power2':'OFF', 'name':'auto', 'fs':1.e6, 'plot':'ON'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	from Main_Analysis import  invoke_signal
	
	if config['mode'] == 'plot_envelope':

		# t, x, f, magX, filename = invoke_signal(config)
		dict_int = invoke_signal(config)
		xraw = dict_int['signal_raw']
		x = dict_int['signal']
		t = dict_int['time']
		filename = dict_int['filename']
		
		v_max = np.max(xraw)
		xraw = xraw / v_max
		x = x / np.max(x)		
		x = x - np.min(x)
		
		xraw = xraw * v_max
		x = x * v_max
		
		


		
		
		fig1, ax1 = plt.subplots()
		ax1.plot(t, xraw, label='Signal')
		ax1.plot(t, x, label='Envelope')
		ax1.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)
		ax1.set_xlabel('Time (s)')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax1.set_ylabel('Amplitude (m$V_{in}$)')
		plt.legend()
		plt.tight_layout()
		plt.savefig('AE_3_demod_bp_100_450_lp_36_source_gearbox_n_1.png')
		ax1.set_xlim(left=2.74, right=4.76)
		plt.savefig('AE_3_demod_bp_100_450_lp_36_source_gearbox_n_2.png')
		

		
		fig2, ax2 = plt.subplots()
		ax2.plot(t, xraw, label='Signal')
		ax2.plot(t, x, label='Huellkurve')
		ax2.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)
		ax2.set_xlabel('Zeit (s)')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax2.set_ylabel('Amplitude (m$V_{in}$)')
		plt.legend()
		plt.tight_layout()
		plt.savefig('DE_AE_3_demod_bp_100_450_lp_36_source_gearbox_n_1.png')
		ax2.set_xlim(left=2.74, right=4.76)
		plt.savefig('DE_AE_3_demod_bp_100_450_lp_36_source_gearbox_n_2.png')
		
		
		
		
		
		
		# fig_22, ax_22 = plt.subplots(nrows=2, ncols=1, sharex=True)
		# ax_22[1].plot(t, xnewraw)
		# ax_22[0].plot(t, xraw)
		
		# ax_22[1].set_title(config['channel'] + ' WFM Erkennung der Bursts', fontsize=11)
		# ax_22[1].set_xlabel('Zeit (s)', fontsize=12)
		# ax_22[1].set_ylabel('Amplitude (m$V_{in}$)', fontsize=12)
		
		# # params = {'mathtext.default': 'regular' }          
		# # plt.rcParams.update(params)
		# ax_22[0].set_ylabel('Amplitude (m$V_{in}$)', fontsize=12)
		# ax_22[0].tick_params(axis='both', labelsize=11)
		# ax_22[1].tick_params(axis='both', labelsize=11)
		
		
		# ax_22[0].set_title(config['channel'] + ' WFM mit Filter', fontsize=11)
		
		
		
		
		
		
		
		plt.show()
		
		
		

		


	elif config['mode'] == 'compare_envelope':
		print('invoke NO drahtburst')
		dict_int = invoke_signal(config)
		xraw = dict_int['signal_raw']
		x = dict_int['signal']
		t = dict_int['time']
		filename = dict_int['filename']
		
		v_max = np.max(xraw)
		xraw = xraw / v_max
		x = x / np.max(x)		
		x = x - np.min(x)
		
		xraw = xraw * v_max
		x = x * v_max
		
		
		print('invoke WITH drahtburst')
		dict_int = invoke_signal(config)
		yraw = dict_int['signal_raw']
		y = dict_int['signal']
		ty = dict_int['time']
		filename_y = dict_int['filename']
		
		m_max = np.max(yraw)
		yraw = yraw / m_max
		y = y / np.max(y)		
		y = y - np.min(y)
		
		yraw = yraw * m_max
		y = y * m_max
		
		


		
		
		# fig1, ax1 = plt.subplots()
		# ax1.plot(t, xraw, label='Signal')
		# ax1.plot(t, x, label='Envelope')
		# ax1.set_title(config['channel'] + ' WFM ' + filename, fontsize=10)
		# ax1.set_xlabel('Time (s)')
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		# ax1.set_ylabel('Amplitude (m$V_{in}$)')
		# plt.legend()
		# plt.tight_layout()
		# plt.savefig('AE_3_demod_bp_100_450_lp_36_source_gearbox_n_1.png')
		# ax1.set_xlim(left=2.74, right=4.76)
		# plt.savefig('AE_3_demod_bp_100_450_lp_36_source_gearbox_n_2.png')
		

		label = config['channel']
		label = label.replace('_', '-')
		
		fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
		ax2[0].plot(t, xraw, label='Signal')
		ax2[0].plot(t, x, label='Huellkurve')
		ax2[0].set_title(label + ' ohne Drahtbuerste', fontsize=11)
		# ax2[0].set_xlabel('Zeit (s)', fontsize=12)
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax2[0].set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
		# plt.legend(fontsize=10)		
		# ax2[0].set_xlim(left=2.74, right=4.76)
		ax2[0].tick_params(axis='both', labelsize=11)
		
		ax2[1].plot(t, yraw, label='Signal')
		ax2[1].plot(ty, y, label='Huellkurve')
		ax2[1].set_title(label + ' mit Drahtbuerste', fontsize=11)
		ax2[1].set_xlabel('Dauer [s]', fontsize=12)
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax2[1].set_ylabel('Amplitude [m$V_{in}$]', fontsize=12)
		
		# ax2[1].set_xlim(left=2.74, right=4.76)
		ax2[1].tick_params(axis='both', labelsize=11)
		
		plt.legend(fontsize=10)		
		plt.tight_layout()
		
		
		# fig_22, ax_22 = plt.subplots(nrows=2, ncols=1, sharex=True)
		# ax_22[1].plot(t, xnewraw)
		# ax_22[0].plot(t, xraw)
		
		# ax_22[1].set_title(config['channel'] + ' WFM Erkennung der Bursts', fontsize=11)
		# ax_22[1].set_xlabel('Zeit (s)', fontsize=12)
		# ax_22[1].set_ylabel('Amplitude (m$V_{in}$)', fontsize=12)
		
		# # params = {'mathtext.default': 'regular' }          
		# # plt.rcParams.update(params)
		# ax_22[0].set_ylabel('Amplitude (m$V_{in}$)', fontsize=12)
		# ax_22[0].tick_params(axis='both', labelsize=11)
		# ax_22[1].tick_params(axis='both', labelsize=11)
		
		
		# ax_22[0].set_title(config['channel'] + ' WFM mit Filter', fontsize=11)
		
		
		
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
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


if __name__ == '__main__':
	main(sys.argv)
