

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


# from os import chdir
# plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'png'
 
#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
# Inputs = ['channel', 'fs']
# InputsOpt_Defaults = {'power2':'OFF', 'plot':'OFF', 'title_plot':None, 'file':'OFF'}

# class Plot3V:
	# def __init__(self, three_signals):
		# fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)
		# keys = list(three_signals)
		# ax[0].plot(three_signals['dom'], three_signals[keys[0]], label=keys[0])
		# ax[1].plot(three_signals['dom'], three_signals[keys[1]], label=keys[1])
		# ax[2].plot(three_signals['dom'], three_signals[keys[2]], label=keys[2])
		
		# # if info_3signals['dom'] == 'time':	
			# # ax[2].set_xlabel('Time [s]', fontsize=font_big)	


	# def style(self, type):
		# # del font_manager.weight_dict['roman']
		# # font_manager._rebuild()
		# # plt.rcParams['font.family'] = 'Times New Roman'
		# font_big = 17
		# font_little = 14
		# font_label = 12
			
	
	# def show(self):
		# plt.show()
		# return

def plot3v(three_signals, style):
	#Modules and global properties
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.subplots_adjust(hspace=0.175, left=0.15, right=0.9, bottom=0.125, top=0.95)
	
	#Values Fixed
	font_big = 17
	font_little = 14
	font_label = 12
	
	#Plot
	fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)
	keys = list(three_signals)
	ax[0].plot(three_signals['dom'], three_signals[keys[0]], label=keys[0])
	ax[1].plot(three_signals['dom'], three_signals[keys[1]], label=keys[1])
	ax[2].plot(three_signals['dom'], three_signals[keys[2]], label=keys[2])
	
	#Axis X
	if style['dom'] == 'time':	
		ax[2].set_xlabel('Time [s]', fontsize=font_big)
	elif style['dom'] == 'frequency':
		if style['kHz'] == 'ON':
			ax[2].set_xlabel('Frequency [kHz]', fontsize=font_big)
		else:
			ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)
	
	#Axis Y
	if style['mV'] == 'ON':
		ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
	
	#Size labels from axis
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
		
	#Scientific notation
	lim = 2
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)
	
	
	

	#Figure Size
	fig.set_size_inches(6, 6)
	
	#Eliminate line from label
	ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)	
	ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	#Size from offset text
	ax[0].yaxis.offsetText.set_fontsize(font_little-2)
	ax[1].yaxis.offsetText.set_fontsize(font_little-2)
	ax[2].yaxis.offsetText.set_fontsize(font_little-2)
	
	#Hidde labels from Axis X in upper plots
	ax[0].tick_params(labelbottom=False) 
	ax[1].tick_params(labelbottom=False) 
	
	# #Set Ticks in Axis Y
	# ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[1].set_yticks([-20, -10, 0, 10, 20]) 
	# ax[2].set_yticks([-20, -10, 0, 10, 20]) 
	
	
	# #Set Limits in Axis X
	# ax[0].set_xlim(left=0, right=2)

	
	plt.tight_layout()
	plt.show()
	return


def main(argv):
	time = np.arange(0, 100, 0.1)
	signal1 = np.cos(2*np.pi*2*time+1)
	signal2 = np.sin(2*np.pi*7*time+1)
	signal3 = np.cos(2*np.pi*8*time+2)
	three_signals = {'AE-1':signal1, 'AE-2':signal2, 'AE-3':signal3, 'dom':time}
	style = {'dom':'time', 'kHz':'ON', 'mV':'ON'}
	# print(list(three_signals	))
	Plot3AeWfms = Plot3V(three_signals)
	Plot3AeWfms.style('hola')
	Plot3AeWfms.show()
	
	
	
	
	
	return


def main2(argv):
	from matplotlib import font_manager
	del font_manager.weight_dict['roman']
	font_manager._rebuild()
	plt.rcParams['font.family'] = 'Times New Roman'
	font_big = 17
	font_little = 14
	font_label = 12
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	config['channel'] = 'AE_1'
	analysis_dict_1 = invoke_signal(config)
	config['channel'] = 'AE_2'
	analysis_dict_2 = invoke_signal(config)
	config['channel'] = 'AE_3'
	analysis_dict_3 = invoke_signal(config)
	
	fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)
	# fig, ax = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=True)
	
	# magX_1 = fft_prom_dict(analysis_dict_1)
	# magX_2 = fft_prom_dict(analysis_dict_2)
	# magX_3 = fft_prom_dict(analysis_dict_3)	
	# f_1 = analysis_dict_1[0]['f']
	# f_2 = analysis_dict_2[0]['f']
	# f_3 = analysis_dict_3[0]['f']	
	# ax[0].plot(f_1/1000., magX_1, label='AE-1')
	# ax[1].plot(f_2/1000., magX_2, label='AE-2')
	# ax[2].plot(f_3/1000., magX_3, label='AE-3')
	# ax[0].set_xlabel('Frequency [Hz]', fontsize=font_big)	
	# ax[1].set_xlabel('Frequency [Hz]', fontsize=font_big)
	# ax[2].set_xlabel('Frequency [Hz]', fontsize=font_big)	
	
	
	# magX_1 = psd_prom_dict(analysis_dict_1)
	# magX_2 = psd_prom_dict(analysis_dict_2)
	# magX_3 = psd_prom_dict(analysis_dict_3)	
	# f_1 = analysis_dict_1[0]['f_psd']
	# f_2 = analysis_dict_2[0]['f_psd']
	# f_3 = analysis_dict_3[0]['f_psd']	
	# ax[0].plot(f_1/1000., magX_1, 'g')
	# ax[1].plot(f_2/1000., magX_2, 'g')
	# ax[2].plot(f_3/1000., magX_3, 'g')

	
	x1 = analysis_dict_1[0]['wfm']
	largo = int(len(x1)/5)
	x1 = x1[0:largo]
	x2 = analysis_dict_2[0]['wfm'][0:largo]
	x3 = analysis_dict_3[0]['wfm'][0:largo]
	t1 = analysis_dict_1[0]['t'][0:largo]
	t2 = analysis_dict_2[0]['t'][0:largo]
	t3 = analysis_dict_3[0]['t'][0:largo]
	ax[0].plot(t1, x1, label='AE-1')
	ax[1].plot(t2, x2, label='AE-2')
	ax[2].plot(t3, x3, label='AE-3')
	# ax[0].set_xlabel('Time [s]', fontsize=font_big)	
	# ax[1].set_xlabel('Time [s]', fontsize=font_big)
	ax[2].set_xlabel('Time [s]', fontsize=font_big)	
	
	# ax[0].set_title('No Fault', fontsize=13)	
	# ax[1].set_title('Initial Faulty Condition', fontsize=13)	
	# ax[2].set_title('Developed Faulty Condition', fontsize=13)
	
	ax[0].tick_params(axis='both', labelsize=font_little)
	ax[1].tick_params(axis='both', labelsize=font_little)	
	ax[2].tick_params(axis='both', labelsize=font_little)	
	
	
	
	# ax[0].plot(f_1, magX_1)
	# ax[1].plot(f_2, magX_2)
	# ax[2].plot(f_3, magX_3)
	
	# ax[0].set_title(analysis_dict_1[0]['filename'], fontsize=13)
	# ax[1].set_title(analysis_dict_2[0]['filename'], fontsize=13)
	# ax[2].set_title(analysis_dict_3[0]['filename'], fontsize=13)
	
	# ax[0].set_xlabel('Frequency [kHz]', fontsize=13)	
	# ax[0].set_title('No Fault', fontsize=13)	
	# ax[0].tick_params(axis='both', labelsize=12)	
	# ax[1].set_xlabel('Frequency [kHz]', fontsize=13)
	# ax[1].set_title('Initial Faulty Condition', fontsize=13)	
	# ax[1].tick_params(axis='both', labelsize=12)	
	# ax[2].set_xlabel('Frequency [kHz]', fontsize=13)	
	# ax[2].set_title('Developed Faulty Condition', fontsize=13)	
	# ax[2].tick_params(axis='both', labelsize=12)
	
	
	# ax[0].set_ylabel('Amplitude [m$V^{2}$]', fontsize=13)
	# ax[1].set_ylabel('Amplitude [m$V^{2}$]', fontsize=13)
	# ax[2].set_ylabel('Amplitude [m$V^{2}$]', fontsize=13)	
	# ax[0].set_ylabel('Amplitude [mV]', fontsize=font_big)
	ax[1].set_ylabel('Amplitude [mV]', fontsize=font_big)
	# ax[2].set_ylabel('Amplitude [mV]', fontsize=font_big)
	
	lim = 2
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	params = {'mathtext.default': 'regular' }          
	plt.rcParams.update(params)
	
	# ax.get_xaxis().get_major_formatter().set_useOffset(False)
	# ax[1].yaxis.offsetText.set_visible(True)
	
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)
		# pelo = ax_it.get_xaxis().get_major_formatter()
		# pelo.set_visible(True)
		# for tk in ax_it.get_xaxis().get_major_formatter():
			# tk.set_visible(True)
	
	# plt.subplots_adjust(wspace=0.275, left=0.06, right=0.98, bottom=0.175, top=0.94)
	plt.subplots_adjust(hspace=0.175, left=0.15, right=0.9, bottom=0.125, top=0.95)


	# fig.set_size_inches(14, 4.0)
	fig.set_size_inches(6, 6)
	# fig.set_size_inches(20.4, 4.8)
	ax[0].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	# for item in leg.legendHandles:
		# item.set_visible(False)
	ax[1].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	ax[2].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
	
	
	ax[0].yaxis.offsetText.set_fontsize(font_little-2)
	ax[1].yaxis.offsetText.set_fontsize(font_little-2)
	ax[2].yaxis.offsetText.set_fontsize(font_little-2)
	
	
	ax[0].tick_params(labelbottom=False) 
	ax[1].tick_params(labelbottom=False) 
	
	ax[0].set_yticks([-20, -10, 0, 10, 20]) 
	ax[1].set_yticks([-20, -10, 0, 10, 20]) 
	ax[2].set_yticks([-20, -10, 0, 10, 20]) 
	
	
	
	# ax[0].set_xlim(left=0, right=2)
	# ax[2].set_xlim(left=1.01, right=1.06)
	# ax[1].set_xlim(left=1.29, right=1.34)
	# ax[0].set_xlim(left=1.58, right=1.63)
	
	# plt.tight_layout()
	plt.show()
	
	# plot_analysis(analysis_dict)
	
	return 




if __name__ == '__main__':
	main(sys.argv)
