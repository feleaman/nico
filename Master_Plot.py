# import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import isnan
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
from m_plots import *

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import RobustScaler  
import xlrd
plt.rcParams['savefig.dpi'] = 800
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode', 'calculation', 'n_data', 'output']
InputsOpt_Defaults = {'legend':None, 'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.7, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature_cond':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature5':'RMS', 'feature_array':['RMS'], 'channel':'AE'}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == '1x4_two_scale_plot':
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'
		print('Select Features files')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		print('Select Feature Condition...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath2 = filedialog.askopenfilename()			
		root.destroy()
		
		
		
		mydict2 = pd.read_excel(filepath2, sheetname=config['sheet'])
		mydict2 = mydict2.to_dict(orient='list')
		Feature2 = np.array(mydict2[config['feature_cond']])
		
		index_nonan = []
		count = 0
		for element in Feature2:
			if not isnan(element):
				index_nonan.append(count)
			count +=1
		Feature2 = Feature2[index_nonan]
		if config['feature_cond'] == 'n':
			Feature2 = Feature2/117.18
		
		fig, ax = plt.subplots(nrows=1, ncols=2)
		# ax2 = ax.twinx()
		
		# count = 0
		titles = ['AE-1', 'AE-2', 'AE-3', 'AE-4']
		features = ['MAX', 'RMS']
		fontsize_big = 18
		fontsize_little = 17
		fontsize_legend = 14
		for k in range(2):
			count = 0
			for filepath in Filepaths:
			
				mydict = pd.read_excel(filepath, sheetname=config['sheet'])
				mydict = mydict.to_dict(orient='list')
				Feature = mydict[features[k]]
				
				Feature = 1000*np.array(Feature)/141.3
				
				if config['n_mov_avg'] != 0:
					Feature = movil_avg(Feature, config['n_mov_avg'])			
				
				
				time = np.array([i*10. for i in range(len(Feature))])/60.
				time2 = time[index_nonan]
				
				ax[k].plot(time, Feature, label=titles[count])
				
				ax[k].set_xlabel('Time [min]', fontsize=fontsize_big)
				
				if features[k] == 'RMS':
					ax[k].set_ylabel('RMS value [mV]', color='k', fontsize=fontsize_big)
					ax[k].set_ylim(bottom=0, top=0.8)
				else:
					ax[k].set_ylabel('Maximum value [mV]', color='k', fontsize=fontsize_big)
					ax[k].set_ylim(bottom=0, top=40)
				
				ax[k].tick_params('y', colors='k')				
				
				ax[k].legend(fontsize=fontsize_legend, loc='best')
				ax[k].tick_params(axis='both', labelsize=fontsize_little)
				
				ax[k].set_xlim(left=0, right=60)
				count += 1	
				
			ax2 = ax[k].twinx()			
			ax2.set_xlim(left=0, right=60)
			# ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
			
			if config['feature_cond'] == 'T':
				ax2.set_ylim(bottom=25, top=55)
				ax2.set_ylabel('Temperature [°C]', color='m', fontsize=fontsize_big)
			elif config['feature_cond'] == 'n':
				
				ax2.set_ylabel('Rotational speed [CPM]', color='m', fontsize=fontsize_big)
				# Feature2 = Feature2/117.18
				ax2.set_ylim(bottom=0, top=15)
				ax2.set_yticks([0, 3, 6, 9, 12, 15])
			
			elif config['feature_cond'] == 'M':
				
				ax2.set_ylabel('Torque [Nm]', color='m', fontsize=fontsize_big)
				# Feature2 = Feature2/117.18
				ax2.set_ylim(bottom=0, top=18)
				ax2.set_yticks([0, 3, 6, 9, 12, 15, 18])
				# ax2.set_yticks([0, 3, 6, 9, 12, 15])
			# ax2.set_ylabel('Rotational speed [CPM]', color='r', fontsize=15)
			
			ax2.plot(time2, Feature2, 'sm')			
			
			ax2.tick_params('y', colors='m')
			ax2.tick_params(axis='both', labelsize=fontsize_little)
			


		fig.set_size_inches(12, 4.5)


		plt.subplots_adjust(wspace=0.4, left=0.065, right=0.935, bottom=0.15, top=0.94)
		# plt.tight_layout()
		plt.show()
	
	elif config['mode'] == '3_plot':
		
		index = np.array([0., 1., 2., 3., 4., 5.])
		index_labels = [1, 2, 4, 7, 9, 10]	
		
		
		#MPR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#MPR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#MPR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#MPR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#MPR++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()			
		# root.destroy()
		
		# workbook = xlrd.open_workbook(filepath)
		# worksheet = workbook.sheet_by_name('MPR')
		
		# # #MPR
		# # ffr = np.array(worksheet.col_values(colx=8-1, start_rowx=57-1, end_rowx=63-1))
		# # fg = np.array(worksheet.col_values(colx=9-1, start_rowx=57-1, end_rowx=63-1))
		# # sb = np.array(worksheet.col_values(colx=10-1, start_rowx=57-1, end_rowx=63-1))
		
		# #MAG
		# ffr = np.array(worksheet.col_values(colx=2-1, start_rowx=57-1, end_rowx=63-1))
		# fg = np.array(worksheet.col_values(colx=3-1, start_rowx=57-1, end_rowx=63-1))
		# sb = np.array(worksheet.col_values(colx=4-1, start_rowx=57-1, end_rowx=63-1))
		
		# name = basename(filepath)[:-5]
		# name = name.replace('Meaning', '\\TestBench_MPR')
		
		# path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures'
		# path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench'

		
		# path_1b = path_1 + name + '.svg'
		# path_2b = path_2 + name + '.pdf'		

		
		
		# plt.rcParams['mathtext.fontset'] = 'cm'	
		
		# three_signals = {'Harmonics $n f_{f,r}$':ffr, 'Harmonics $p f_{m}$':fg, 'Sidebands $p f_{m} \pm s f_{c}$':sb, 'dom':index}
		# style = {'dom':'other', 'sharex':True, 'sharey':False, 'type':'bar', 'ytitle':['Magnitude [mV]', 'Magnitude [mV]', 'Magnitude [mV]'], 'xtitle':['MC', 'MC', 'MC'], 'xticklabels':index_labels, 'legend':False, 'title':True, 'xmax':None, 'ymax':[60.e-3, 15.e-2, 15.e-2], 'autolabel':False, 'caption':'lower left', 'path_1':path_1b, 'path_2':path_2b, 'output':'plot'}
		# plot3h_thesis_big(three_signals, style)
		# # [r'MPR [dB]', 'MPR [dB]', 'MPR [dB]']
		# # ['Magnitude [mV$^{2}$]', 'Magnitude [mV$^{2}$]', 'Magnitude [mV$^{2}$]']
		# # ['Magnitude [$\mu$V$^{2}$]', 'Magnitude [$\mu$V$^{2}$]', 'Magnitude [$\mu$V$^{2}$]']
		# # ['Magnitude [mV]', 'Magnitude [mV]', 'Magnitude [mV]']
		# # [60, 60, 60]
		# # [60.e-3, 15.e-2, 15.e-2]
		# sys.exit()	

		
		
		ffr = []
		fg = []
		sb = []
		for i in range(config['n_data']):
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			
			workbook = xlrd.open_workbook(filepath)
			worksheet = workbook.sheet_by_name('MPR')
			
			name = basename(filepath)[:-5]
			name = name.replace('Meaning', '\\TestBench_' + config['calculation'])

			
			if name.find('TestBench_MAG') != -1:
				print('------- Calculation of MAG -------')
				aa, bb, cc = 2, 3, 4
			else:
				print('++++++ Calculation of MPR ++++++')
				aa, bb, cc = 8, 9, 10			
			
			ffr.append(np.array(worksheet.col_values(colx=aa-1, start_rowx=57-1, end_rowx=63-1)))
			fg.append(np.array(worksheet.col_values(colx=bb-1, start_rowx=57-1, end_rowx=63-1)))
			sb.append(np.array(worksheet.col_values(colx=cc-1, start_rowx=57-1, end_rowx=63-1)))	
			
			
			path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures'
			path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench'
			
			path_1b = path_1 + name + '.svg'
			path_2b = path_2 + name + '.pdf'	
		
		if config['n_data'] == 1:
			ffr = ffr[0]
			fg = fg[0]
			sb = sb[0]
			type_ = 'bar'
		else:
			type_ = 'plot'
		
		
		#Asignation Y Title
		if config['calculation'] == 'MPR':
			ytitle_ = [r'MPR [dB]', 'MPR [dB]', 'MPR [dB]']
		elif config['calculation'] == 'MAG':
			if name.find('SQR') != -1:
				ytitle_ = ['Magnitude [mV$^{2}$]', 'Magnitude [mV$^{2}$]', 'Magnitude [mV$^{2}$]']
			else:
				ytitle_ = ['Magnitude [mV]', 'Magnitude [mV]', 'Magnitude [mV]']
		
		#Asignation Legend	
		if config['legend'] == 'db':
			root_ = ['db6', 'db12', 'db18']
			legend_ = {'first':root_, 'second':root_, 'third':root_}
		elif config['legend'] == 'sym':
			root_ = ['sym6', 'sym12', 'sym18']
			legend_ = {'first':root_, 'second':root_, 'third':root_}
		elif config['legend'] == 'levels':
			root_ = ['3 levels', '5 levels', '7 levels', '9 levels']
			legend_ = {'first':root_, 'second':root_, 'third':root_}
		else:
			legend_ = None
			
		
		#Config Plot
		plt.rcParams['mathtext.fontset'] = 'cm'
		three_signals = {'Harmonics $n f_{f,r}$':ffr, 'Harmonics $p f_{m}$':fg, 'Sidebands $p f_{m} \pm s f_{c}$':sb, 'dom':index}
		style = {'dom':'other', 'sharex':True, 'sharey':False, 'type':type_, 'ytitle':ytitle_, 'xtitle':['MC', 'MC', 'MC'], 'xticklabels':index_labels, 'legend':legend_, 'title':True, 'xmax':None, 'ymax':[50., 50., 50.], 'ymin':[20, 20, 20], 'autolabel':False, 'caption':'lower left', 'path_1':path_1b, 'path_2':path_2b, 'output':config['output']}
		
		#PLOT function
		if config['n_data'] == 1:
			plot3h_thesis_big(three_signals, style)
		else:
			plot3h_thesis_big_multi(three_signals, style)
	
	
	elif config['mode'] == '6B_plot':
		
		index = np.array([0., 1., 2., 3., 4., 5.])
		index_labels = [1, 2, 4, 7, 9, 10]	
		
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		
		ffr_mag = []
		fg_mag = []
		sb_mag = []
		
		ffr_mpr = []
		fg_mpr = []
		sb_mpr = []
		
		for filepath in Filepaths:
			print(basename(filepath))
			# root = Tk()
			# root.withdraw()
			# root.update()
			# filepath = filedialog.askopenfilename()			
			# root.destroy()
			
			workbook = xlrd.open_workbook(filepath)
			worksheet = workbook.sheet_by_name('MPR')
			
			name = basename(filepath)[:-5]
			name = name.replace('Meaning', '\\TestBench_' + config['calculation'])

			
			ffr_mpr.append(np.array(worksheet.col_values(colx=9-1, start_rowx=2-1, end_rowx=8-1)))
			fg_mpr.append(np.array(worksheet.col_values(colx=10-1, start_rowx=2-1, end_rowx=8-1)))
			sb_mpr.append(np.array(worksheet.col_values(colx=8-1, start_rowx=2-1, end_rowx=8-1)))
			
			ffr_mag.append(np.array(worksheet.col_values(colx=3-1, start_rowx=2-1, end_rowx=8-1))*1000.)
			fg_mag.append(np.array(worksheet.col_values(colx=4-1, start_rowx=2-1, end_rowx=8-1))*1000.)
			sb_mag.append(np.array(worksheet.col_values(colx=2-1, start_rowx=2-1, end_rowx=8-1))*1000.)
			
			
			path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures'
			path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench'
			
			path_1b = path_1 + name + '.svg'
			path_2b = path_2 + name + '.pdf'	
		
		if config['n_data'] == 1:
			ffr_mpr = ffr_mpr[0]
			fg_mpr = fg_mpr[0]
			sb_mpr = sb_mpr[0]
			
			ffr_mag = ffr_mag[0]
			fg_mag = fg_mag[0]
			sb_mag = sb_mag[0]
			
			
			type_ = 'bar'
		else:
			type_ = 'plot'
		
		
		#Asignation Y Title
		ytitle_ = [r'Magnitude [$\mu$V]', 'Magnitude [$\mu$V]', 'Magnitude [$\mu$V]', 'MPR [dB]', 'MPR [dB]', 'MPR [dB]']
		
		#Asignation Legend	
		if config['legend'] == 'db':
			# root_ = ['db6', 'db12', 'db18']
			# legend_ = {'first':root_, 'second':root_, 'third':root_}
			legend_ = ['db6', 'db12', 'db18']
		elif config['legend'] == 'sym':
			# root_ = ['sym6', 'sym12', 'sym18']
			# legend_ = {'first':root_, 'second':root_, 'third':root_}
			legend_ = ['sym6', 'sym12', 'sym18']
		elif config['legend'] == 'levels':
			# root_ = ['3 levels', '5 levels', '7 levels', '9 levels']
			# legend_ = {'first':root_, 'second':root_, 'third':root_}
			legend_ = ['3 levels', '5 levels', '7 levels', '9 levels']
		elif config['legend'] == 'emd':
			legend_ = ['No filter', 'BP filter']
		else:
			legend_ = [None, None, None, None, None, None]
			
		
		#Config Plot
		plt.rcParams['mathtext.fontset'] = 'cm'
		
		# six_signals = {'ffr_mpr':ffr_mpr, 'fg_mpr':fg_mpr, 'sb_mpr':sb_mpr, 'ffr_mag':ffr_mag, 'fg_mag':fg_mag, 'sb_mag':sb_mag, 'dom':index}
		
		# title_ = ['Harmonics $n f_{f,r}$', 'Harmonics $p f_{m}$', 'Sidebands $p f_{m} \pm s f_{c}$', 'Harmonics $n f_{f,r}$', 'Harmonics $p f_{m}$', 'Sidebands $p f_{m} \pm s f_{c}$']
		
		title_ = ['Harmonics $n f_{c}$', 'Harmonics $p f_{m}$', 'Sidebands $p f_{m} \pm s f_{c}$', 'Harmonics $n f_{c}$', 'Harmonics $p f_{m}$', 'Sidebands $p f_{m} \pm s f_{c}$']
		
		
		
		data_y = [ffr_mag, fg_mag, sb_mag, ffr_mpr, fg_mpr, sb_mpr]
		data_x = index
		data = {'data_x':data_x, 'data_y':data_y}
		
		style = {'dom':'other', 'sharex':True, 'sharey':False, 'type':type_, 'ytitle':ytitle_, 'xtitle':'MC', 'xticklabels':index_labels, 'title':title_, 'legend':legend_, 'xlim':None, 'ylim':[[0, 20], [0, 20], [0, 110], [20, 40], [10, 50], [40, 60]], 'path_1':path_1b, 'path_2':path_2b, 'output':config['output'], 'n_data':config['n_data']}
		
		plot6B_thesis_big(data, style)
		sys.exit()
		#PLOT function
		if config['n_data'] == 1:
			plot6B_thesis_big(data, style)
		else:
			plot6B_thesis_big_multi(six_signals, style)
		

	elif config['mode'] == '1_plot_scale':
		
		name = 'WindTurbine_Scgr_S1pai_AE3p_Bu13eq'
		# name = 'WPDLOW_lvl5_sym6_SimulatedAE2'
		# name = 'CycloDensity_2_SimulatedAE2'
		# name = 'STFT_100k_SimulatedAE2'

		
		# path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		# path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\'
		
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\ChapterX_SigProcessing\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_SigProcessing\\'			
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		
		mydict = read_pickle(filepath)
		map = mydict['map']
		extent_ = mydict['extent']
		extent_[1] = extent_[1]*1000
		print(extent_)
		# sys.exit()
		xx = 0
		# 
		style = {'xlabel':'Cyclic frequency [Hz]', 'ylabel':'Frequency [kHz]', 'legend':[None], 'title':'Spectral correlation density', 'xticklabels':None, 'xlim':[1.00, 1.25], 'ylim':[0, 500], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'extent':extent_, 'colormap':'YlGnBu', 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}
		# data = {'x':[t], 'y':[Feature]}
		# 'STFT Spectrogram $\it{\Delta}$t = 0.01 ms'
		# 'YlOrBr'
		# 'plasma'
		# 'YlOrRd'
		# 'BuPu'
		plot1_scalo_thesis(map, style)
		
		
		
		
		
		#wavelet+++++++++++++++++++++++++++++++
		extent_[1] = extent_[1]*1000.
		
		# 'AE-3: 2-frequency burst scalogram'
		
		
		# [0.00105*1000, 0.00125*1000]
		 # [0.00105*1000, 0.00125*1000]
		style = {'xlabel':'Time [ms]', 'ylabel':'Frequency [kHz]', 'legend':[None], 'title':'2-frequency burst scalogram on AE-3', 'customxlabels':None, 'xlim':[1.00, 1.25], 'ylim':None, 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'extent':extent_, 'colormap':'YlGnBu', 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}
		# data = {'x':[t], 'y':[Feature]}
		# 'YlGnBu'
		# 'YlOrBr'
		# 'plasma'
		# 'YlOrRd'
		# 'BuPu'
		plot1_scalo_thesis(map, style)
	
	elif config['mode'] == '1_plot':
		
		name = 'WindTurbine_Wfm_S1pai_AE3p_Bu13eq'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		
		
		#+++++++++++ABRIR pik++++++++++++++++
		mydict = read_pickle(filepath)		
		
		energy = mydict['energy']
		freq = mydict['freq']
		fig, ax = plt.subplots()
		ax.semilogy(freq/1000., energy)
		ax.set_xlabel('Frequency [kHz]')
		ax.set_ylabel('Wavelet Energy [V]')
		plt.show()
		sys.exit()
		# #++++++++++++++++++++++++++++++++++++++++
		
		
		#+++++++++++ABRIR SENAL++++++++++++++++
		# factor = 1000/70.8 #schottland
		factor = 1000/281.8
		signal = load_signal(filepath, channel=config['channel'])		
		
		# tb = 2.688736 #s1 ae3 b13
		# tb = 6.232396 #s1 ae2 b78
		tb = 4.76218 #s1pai ae3 b13eq
		wid = 0.002
		tini = tb - wid/2 
		tfin = tb + wid
		signal = signal[int(tini*config['fs']) : int(tfin*config['fs'])]*factor
		
		# signal = np.array(list(np.zeros(75)) + list(signal))
		
		dt = 1./config['fs']		
		
		t = np.arange(len(signal))*dt*1000.
		n = len(signal)
		#++++++++++++++++++++++++++++++++++++++++
		
		
		
		# [0.8+delta, 1.6+delta]
		# [0.00105*1000, 0.00125*1000]
		# [(0.8+2.1)/1000, (1.6+2.1)/1000]
		# [-20, 20]
		style = {'xlabel':'Time [ms]', 'ylabel':'Amplitude [mV]', 'legend':[None], 'title':'2-frequency burst on AE-3p', 'customxlabels':None, 'xlim':[0.8, 1.6], 'ylim':[-10,10], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}
		# data = {'x':[t], 'y':[Feature]}
		# 'YlGnBu'
		# 'YlOrBr'
		# 'plasma'
		# 'YlOrRd'
		# 'BuPu'
		data = {'x':[t], 'y':[signal]}
		# data = {'x':[freq], 'y':[energy]}
		plot1_thesis(data, style)
		
		
	elif config['mode'] == '6_plot':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
		for filepath in Filepaths:
			print(basename(filepath))
		dict1 = read_pickle(Filepaths[0])
		dict2 = read_pickle(Filepaths[1])
		dict3 = read_pickle(Filepaths[2])
		f_1 = dict1['f']
		f_2 = dict2['f']
		f_3 = dict3['f']/1.023077
		magX_1 = dict1['fft']
		magX_2 = dict2['fft']
		magX_3 = dict3['fft']
		
		# name = basename(filepath)[:-5]
		# name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
		
		idxidx = basename(filepath).find('Idx')
		idx = basename(filepath)[idxidx+4:idxidx+6]
		if idx[1] == '.':
			idx = idx[0]
		name = 'TestBench_Avg_EnvSpectrum_Idx_' + idx
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		min_a = -40
		max_a = 20
		min_b = -40
		max_b = 20
		
		# min_a = 0
		# max_a = 0.005
		# min_b = 0
		# max_b = 0.02
		
		['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10']
		[None, None, None, None, None, None]
		
		style = {'xlabel':'Frequency [Hz]', 'ylabel':r'Magnitude [dB$_{AE}$]', 'legend':[None, None, None, None, None, None], 'title':['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10'], 'customxlabels':[210, 260, 310, 360, 410], 'xlim':[[0, 100], [0, 100], [0, 100], [210, 410], [210, 410], [210, 410]], 'ylim':[[min_a, max_a], [min_a, max_a], [min_a, max_a], [min_b, max_b], [min_b, max_b], [min_b, max_b]], 'color':[None, None, None, None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'ON'}
		data = {'x':[f_1, f_2, f_3, f_1, f_2, f_3], 'y':[magX_1, magX_2, magX_3, magX_1, magX_2, magX_3]}
		plot6_thesis_big(data, style)
	
	
	elif config['mode'] == '9_plot_imf':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
		for filepath in Filepaths:
			print(basename(filepath))
		dict1 = read_pickle(Filepaths[0])
		dict2 = read_pickle(Filepaths[1])
		dict3 = read_pickle(Filepaths[2])
		dict4 = read_pickle(Filepaths[3])
		dict5 = read_pickle(Filepaths[4])
		dict6 = read_pickle(Filepaths[5])
		dict7 = read_pickle(Filepaths[6])
		dict8 = read_pickle(Filepaths[7])
		dict9 = read_pickle(Filepaths[8])
		f_1 = dict1['f']/1000.
		f_2 = dict2['f']/1000.
		f_3 = dict3['f']/1000.
		f_4 = dict4['f']/1000.
		f_5 = dict5['f']/1000.
		f_6 = dict6['f']/1000.
		f_7 = dict7['f']/1000.
		f_8 = dict8['f']/1000.
		f_9 = dict9['f']/1000.
		
		magX_1 = dict1['fft']
		magX_2 = dict2['fft']
		magX_3 = dict3['fft']		
		magX_4 = dict4['fft']
		magX_5 = dict5['fft']
		magX_6 = dict6['fft']		
		magX_7 = dict7['fft']
		magX_8 = dict8['fft']
		magX_9 = dict9['fft']
		
		# name = basename(filepath)[:-5]
		# name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
		
		# idxidx = basename(filepath).find('Idx')
		# idx = basename(filepath)[idxidx+4:idxidx+6]
		# if idx[1] == '.':
			# idx = idx[0]
		name = 'TestBench_AvgSpectrum_IMF'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		# min_a = -40
		# max_a = 20
		# min_b = -40
		# max_b = 20
		

		
		# title_ = ['MC 1: First IMF', 'MC 7: First IMF', 'MC 10: First IMF', 'MC 1: Second IMF', 'MC 7: Second IMF', 'MC 10: Second IMF', 'MC 1: Third IMF', 'MC 7: Third IMF', 'MC 10: Third IMF']
		
		title_ = ['First IMF for MC 1', 'First IMF for MC 4', 'First IMF for MC 10', 'Second IMF for MC 1', 'Second IMF for MC 4', 'Second IMF for MC 10', 'Third IMF for MC 1', 'Third IMF for MC 4', 'Third IMF for MC 10']
		[None, None, None, None, None, None, None, None, None]
		
		style = {'xlabel':'Frequency [kHz]', 'ylabel':r'Magnitude [mV]', 'legend':[None, None, None, None, None, None, None, None, None], 'title':title_, 'customxlabels':[0, 100, 200, 300, 400, 500], 'xlim':[[0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500], [0, 500]], 'ylim':[[0, 3e-4], [0, 3e-4], [0, 3e-4], [0, 6e-4], [0, 6e-4], [0, 6e-4], [0, 6e-4], [0, 6e-4], [0, 6e-4]], 'color':[None, None, None, None, None, None, None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF'}
		data = {'x':[f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9], 'y':[magX_1, magX_2, magX_3, magX_4, magX_5, magX_6, magX_7, magX_8, magX_9]}
		plot9_thesis_big(data, style)
	
		
		
	elif config['mode'] == '3_plot_avg_spectra':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
		for filepath in Filepaths:
			print(basename(filepath))
		dict1 = read_pickle(Filepaths[0])
		dict2 = read_pickle(Filepaths[1])
		dict3 = read_pickle(Filepaths[2])
		f_1 = dict1['f']
		f_2 = dict2['f']
		f_3 = dict3['f']
		magX_1 = dict1['fft']
		magX_2 = dict2['fft']
		magX_3 = dict3['fft']
		
		# name = basename(filepath)[:-5]
		# name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
		
		idxidx = basename(filepath).find('Idx')
		idx = basename(filepath)[idxidx+4:idxidx+6]
		if idx[1] == '.':
			idx = idx[0]
		# name = 'TestBench_Avg_EnvSpectrum_Idx_' + idx
		name = 'WindTurbine_AvgSpectrum_Idx_' + idx
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		# name = 'WindTurbine_EnvFft_S1pai'
		# path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		# path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\'		
		# path_1b = path_1 + name + '.svg'
		# path_2b = path_2 + name + '.pdf'
		
		
		
		min_a = -50
		max_a = 30
		min_b = -20
		max_b = 40
		
		# min_a = 0
		# max_a = 0.005
		# min_b = 0
		# max_b = 0.02
		
		# ['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10']
		# [None, None, None, None, None, None]
		
		# 140, 220
		# 120, 200
		
		# ['AE-1p: harmonics $N_{p} f_{c}$', 'AE-2p: harmonics $N_{p} f_{c}$', 'AE-3p: harmonics $N_{p} f_{c}$', 'AE-1p: $1^{st}$ harmonic $f_{m}$', 'AE-2p: $1^{st}$ harmonic $f_{m}$', 'AE-3p: $1^{st}$ harmonic $f_{m}$']
		
		
		# [[min_a, max_a], [min_a, max_a], [min_a, max_a]]
		style = {'xlabel':'Frequency [kHz]', 'ylabel':r'Magnitude [dB$_{AE}$]', 'legend':[None, None, None], 'title':['AE-1', 'AE-2', 'AE-3'], 'customylabels':None, 'ylim':None, 'color':[None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'ON', 'ylog':'OFF', 'xlim':[[0, 500], [0, 500], [0, 500]], 'customxlabels':None}		
		data = {'x':[f_1/1000., f_2/1000., f_3/1000.], 'y':[magX_1, magX_2, magX_3]}
		plot3_thesis_new(data, style)
		print(path_2b)
		
		sys.exit()
		
		style = {'xlabel':'Frequency [kHz]', 'ylabel':r'Magnitude [dB$_{AE}$]', 'legend':[None, None, None, None, None, None], 'title':['MC 1: harmonics of $f_{c}$', 'MC 7: harmonics of $f_{c}$', 'MC 10: harmonics of $f_{c}$', 'MC 1: $1^{st}$ harmonic $f_{m}$', 'MC 7: $1^{st}$ harmonic $f_{m}$', 'MC 10: $1^{st}$ harmonic $f_{m}$'], 'customxlabels':[210, 260, 310, 360, 410], 'xlim':[[0, 100], [0, 100], [0, 100], [210, 410], [210, 410], [210, 410]], 'ylim':[[min_a, max_a], [min_a, max_a], [min_a, max_a], [min_b, max_b], [min_b, max_b], [min_b, max_b]], 'color':[None, None, None, None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'ON'}		
		data = {'x':[f_1, f_2, f_3, f_1, f_2, f_3], 'y':[magX_1, magX_2, magX_3, magX_1, magX_2, magX_3]}
		plot6_thesis_big(data, style)
		print(path_2b)
	
	elif config['mode'] == '3_plot_wfm':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
		for filepath in Filepaths:
			print(basename(filepath))
		
		
		#++++++++++++++ WAVEFORMS ++++++++++++++++++++++
		# factor = 1000/70.8 #schottland/ AMT
		# factor = 1000/281.8 #schottland PAI
		# factor = 1000/141.25 #bochum
		
		# tini = 3 + 2.1
		# tfin = 7 + 2.1
		
		# tini = 3 #for S1-1
		# tfin = 7 #for S1-1
		
		# tini = 0.09
		# tfin = 0.51
		
		# tini = 0.
		# tfin = 4.
		
		# t = np.arange((tfin-tini)*config['fs'])/config['fs']		
		# x1 = load_signal(Filepaths[0], channel='AE_1')*factor		
		# x1 = x1[int(tini*config['fs']) : int(tfin*config['fs'])]		
		# x2 = load_signal(Filepaths[0], channel='AE_2')*factor		
		# x2 = x2[int(tini*config['fs']) : int(tfin*config['fs'])]		
		# x3 = load_signal(Filepaths[0], channel='AE_3')*factor		
		# x3 = x3[int(tini*config['fs']) : int(tfin*config['fs'])]
		# t = t[0:-1]
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		# x1 = hilbert_demodulation(x1)
		# x2 = hilbert_demodulation(x2)
		# x3 = hilbert_demodulation(x3)
		
		# x1 = butter_demodulation(x=x1, fs=config['fs'], filter=['lowpass', 40., 3], prefilter=['bandpass', [20.e3, 180.e3], 3], type_rect='absolute_value', dc_value='without_dc')
		# x2 = butter_demodulation(x=x2, fs=config['fs'], filter=['lowpass', 40., 3], prefilter=['bandpass', [20.e3, 180.e3], 3], type_rect='absolute_value', dc_value='without_dc')
		# x3 = butter_demodulation(x=x3, fs=config['fs'], filter=['lowpass', 40., 3], prefilter=['bandpass', [20.e3, 180.e3], 3], type_rect='absolute_value', dc_value='without_dc')
		
		# 'only_positives', 'dc_value':
		# #++++++++++++++ ENERGY FREQ ++++++++++++++++++++++
		dict1 = read_pickle(Filepaths[0])
		x1 = dict1['energy']
		t1 = dict1['freq']/1000.
		dict2 = read_pickle(Filepaths[1])
		x2 = dict2['energy']
		t2 = dict2['freq']/1000.
		dict3 = read_pickle(Filepaths[2])
		x3 = dict3['energy']
		t3 = dict3['freq']/1000.
		# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		# # #++++++++++++++ BARPLOT ++++++++++++++++++++++
		# filepath = None
		# x1 = [24, 71, 79, 178, 184, 428] #07nf
		# t1 = [1, 2, 3, 4, 5, 6]
		
		# x2 = [5, 37, 29, 67, 71, 83] #08nf
		# t2 = [1, 2, 3, 4, 5, 6]
		
		# x3 = [2, 11, 15, 25, 35, 16] #09nf
		# t3 = [1, 2, 3, 4, 5, 6]
		# # #++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		
		
		# # #++++++++++++++ EXCEL FEAT ++++++++++++++++++++++
		# # mydict = pd.read_excel(Filepaths[0], sheetname=config['sheet'])
		# mydict = pd.read_excel(Filepaths[0])
		# mydict = mydict.to_dict(orient='list')
		# x1 = np.array(mydict['fmax'])/1000.
		# t1 = np.array(mydict['crest'])
		# # x1 = np.array(mydict['fp1'])/1000.
		# # t1 = np.array(mydict['fp2'])/1000.
		
		# # t1 = np.arange(len(x1))
		
		# mydict = pd.read_excel(Filepaths[1])
		# mydict = mydict.to_dict(orient='list')
		# x2 = np.array(mydict['fmax'])/1000.
		# t2 = np.array(mydict['crest'])
		# # x2 = np.array(mydict['fp1'])/1000.
		# # t2 = np.array(mydict['fp2'])/1000.
		# # t2 = np.arange(len(x2))
		
		# mydict = pd.read_excel(Filepaths[2])
		# mydict = mydict.to_dict(orient='list')
		# x3 = np.array(mydict['fmax'])/1000.
		# t3 = np.array(mydict['crest'])
		# # t3 = np.arange(len(x3))
		# # x3 = np.array(mydict['fp1'])/1000.
		# # t3 = np.array(mydict['fp2'])/1000.
		
		
		
		
		
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		
		# name = basename(filepath)[:-5]
		# name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
		
		# if filepath != None:
			# idxidx = basename(filepath).find('Idx')
			# idx = basename(filepath)[idxidx+4:idxidx+6]
			# if idx[1] == '.':
				# idx = idx[0]
		
		# # name = 'TestBench_Avg_EnvSpectrum_Idx_' + idx
		# name = 'TestBench_Bursts_Fmax_Crest'
		# path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		# path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		# path_1b = path_1 + name + '.svg'
		# path_2b = path_2 + name + '.pdf'
		
		name = 'WindTurbine_WT_Envelope_S1'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		min_a = -1
		max_a = 1
		# min_b = -20
		# max_b = 40
		
		# min_a = 0
		# max_a = 0.005
		# min_b = 0
		# max_b = 0.02
		
		# ['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10']
		# [None, None, None, None, None, None]
		
		# 'xlim':[[0, 4], [0, 4], [0, 4]]
		# 'xlim':[[0, 500], [0, 500], [0, 500]]
		# 'ylim':[[min_a, max_a], [min_a, max_a], [min_a, max_a]]
		# 'ylabel':r'Amplitude [mV]'
		# 'xlabel':'Time [s]'
		# 'ylabel':'Energy [mV$^{2}$]'
		# 'xlabel':'Frequency [kHz]'
		# 'xlabel':'Crest factor [-]'
		# 'ylabel':'Main frequency [kHz]'
		# 'xlim':[[0, 20], [0,20], [0,20]]
		# 'Threshold 0.7 [mV]', 'Threshold 0.8 [mV]', 'Threshold 0.9 [mV]'
		# 'Number of bursts'
		# 'title':['MC 1', 'MC 7', 'MC 10']
		# 'title':['AE-1', 'AE-2', 'AE-3']
		# 'title':['AE-1: envelope', 'AE-2: envelope', 'AE-3: envelope']
		
		style = {'xlabel':'Frequency [kHz]', 'ylabel':r'Energy [V$^{2}$]', 'legend':[None, None, None, None, None, None], 'title':['Cold', 'Normal', 'Parameters'], 'customxlabels':None, 'xlim':[[0, 1000], [0,1000], [0,1000]], 'ylim':None, 'color':[None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF', 'ylog':'ON'}
		data = {'x':[t1, t2, t3], 'y':[x1, x2, x3]}
		# data = {'x':[t, t, t], 'y':[x1, x2, x3]}
		
		# plot6_thesis_big(data, style)
		plot3_thesis_new(data, style)
	
	
	elif config['mode'] == '3_plot_wfm_varb':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths1 = filedialog.askopenfilenames()
		root.destroy()	
		for filepath in Filepaths1:
			print(basename(filepath))
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths2 = filedialog.askopenfilenames()
		# root.destroy()	
		# for filepath in Filepaths2:
			# print(basename(filepath))
		
		
		
		
		
		# #++++++++++++++ EXCEL FEAT ++++++++++++++++++++++
		# mydict = pd.read_excel(Filepaths[0], sheetname=config['sheet'])
		mydict = pd.read_excel(Filepaths1[0])
		mydict = mydict.to_dict(orient='list')
		# x1 = np.array(mydict['fmax'])/1000.
		x1 = np.array(mydict['freq2'])/1000.
		tx1 = np.array(mydict['crest'])
		
		mydict = pd.read_excel(Filepaths1[1])
		mydict = mydict.to_dict(orient='list')
		# x2 = np.array(mydict['fmax'])/1000.
		x2 = np.array(mydict['freq2'])/1000.
		tx2 = np.array(mydict['crest'])
		
		mydict = pd.read_excel(Filepaths1[2])
		mydict = mydict.to_dict(orient='list')
		# x3 = np.array(mydict['fmax'])/1000.
		x3 = np.array(mydict['freq2'])/1000.
		tx3 = np.array(mydict['crest'])
		
		# #filepathsB
		
		# mydict = pd.read_excel(Filepaths2[0])
		# mydict = mydict.to_dict(orient='list')
		# y1 = np.array(mydict['fmax'])/1000.
		# ty1 = np.array(mydict['crest'])
		
		# mydict = pd.read_excel(Filepaths2[1])
		# mydict = mydict.to_dict(orient='list')
		# y2 = np.array(mydict['fmax'])/1000.
		# ty2 = np.array(mydict['crest'])
		
		# mydict = pd.read_excel(Filepaths2[2])
		# mydict = mydict.to_dict(orient='list')
		# y3 = np.array(mydict['fmax'])/1000.
		# ty3 = np.array(mydict['crest'])

		
		
		
		
		# #++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		
		# name = basename(filepath)[:-5]
		# name = name.replace('Meaning', '\\TestBench_' + config['calculation'])
		
		idxidx = basename(filepath).find('Idx')
		idx = basename(filepath)[idxidx+4:idxidx+6]
		if idx[1] == '.':
			idx = idx[0]
			
		# # name = 'TestBench_Avg_EnvSpectrum_Idx_' + idx
		# name = 'TestBench_Bursts_Fmax_Crest_2'
		# path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		# path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		# path_1b = path_1 + name + '.svg'
		# path_2b = path_2 + name + '.pdf'
		
		name = 'WindTurbine_FmaxFourier_Crest_S1'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		min_a = -0
		max_a = 500
		# min_b = -20
		# max_b = 40
		
		# min_a = 0
		# max_a = 0.005
		# min_b = 0
		# max_b = 0.02
		
		# ['MC 1', 'MC 7', 'MC 10', 'MC 1', 'MC 7', 'MC 10']
		# [None, None, None, None, None, None]
		
		# 'xlim':[[0, 4], [0, 4], [0, 4]]
		# 'ylim':[[min_a, max_a], [min_a, max_a], [min_a, max_a]]
		# 'ylabel':r'Amplitude [mV]'
		# 'xlabel':'Time [s]'
		# 'ylabel':'Energy [mV$^{2}$]'
		# 'xlabel':'Frequency [kHz]'
		# [True, None, None]
		# [True, True, True]
		
		style = {'xlabel':'Crest factor [-]', 'ylabel':'Main frequency [kHz]', 'legend':[None, None, None], 'title':['AE-1', 'AE-2', 'AE-3'], 'customxlabels':None, 'xlim':[[0, 20], [0,20], [0,20]], 'ylim':[[min_a, max_a], [min_a, max_a], [min_a, max_a]], 'color':[None, None, None], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF', 'ylog':'OFF'}
		data1 = {'x':[tx1, tx2, tx3], 'y':[x1, x2, x3]}
		# data2 = {'x':[ty1, ty2, ty3], 'y':[y1, y2, y3]}
		
		# plot6_thesis_big(data, style)
		# plot3_thesis_new_varb(data1, data2, style)
		plot3_thesis_new(data1, style)
	
	
	elif config['mode'] == '15_plot':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		
		root.destroy()

		kur = []
		sen = []
		max = []
		crf = []
		p21 = []
		
		rms = []
		std = []
		sts = []
		var = []
		vas = []
		
		avg = []
		avs = []
		cef = []
		p17 = []
		p24 = []
		
		for filepath in Filepaths:
			print(basename(filepath))
			workbook = xlrd.open_workbook(filepath)
			worksheet = workbook.sheet_by_name('Analysis')
			
			#group 1
			kur.append(np.array(worksheet.col_values(colx=8-1, start_rowx=2-1, end_rowx=8-1)))
			
			sen.append(np.array(worksheet.col_values(colx=24-1, start_rowx=2-1, end_rowx=8-1)))
			
			max.append(np.array(worksheet.col_values(colx=21-1, start_rowx=2-1, end_rowx=8-1)))
			
			crf.append(np.array(worksheet.col_values(colx=5-1, start_rowx=2-1, end_rowx=8-1)))
			
			p21.append(np.array(worksheet.col_values(colx=18-1, start_rowx=2-1, end_rowx=8-1)))
			
			#group 2
			
			rms.append(np.array(worksheet.col_values(colx=23-1, start_rowx=2-1, end_rowx=8-1)))

			std.append(np.array(worksheet.col_values(colx=25-1, start_rowx=2-1, end_rowx=8-1)))
			
			sts.append(np.array(worksheet.col_values(colx=26-1, start_rowx=2-1, end_rowx=8-1)))
			
			var.append(np.array(worksheet.col_values(colx=11-1, start_rowx=2-1, end_rowx=8-1)))
			
			vas.append(np.array(worksheet.col_values(colx=12-1, start_rowx=2-1, end_rowx=8-1)))
			
			#group 3
			avg.append(np.array(worksheet.col_values(colx=9-1, start_rowx=2-1, end_rowx=8-1)))
			
			avs.append(np.array(worksheet.col_values(colx=10-1, start_rowx=2-1, end_rowx=8-1)))
			
			cef.append(np.array(worksheet.col_values(colx=3-1, start_rowx=2-1, end_rowx=8-1)))
			
			p17.append(np.array(worksheet.col_values(colx=16-1, start_rowx=2-1, end_rowx=8-1)))
			
			p24.append(np.array(worksheet.col_values(colx=20-1, start_rowx=2-1, end_rowx=8-1)))

		if config['n_data'] == 1:
			kur = kur[0]
			sen = sen[0]
			max = max[0]
			crf = crf[0]
			p21 = p21[0]
			
			rms = rms[0]
			std = std[0]
			sts = sts[0]
			var = var[0]
			vas = vas[0]
			
			avg = avg[0]
			avs = avs[0]
			cef = cef[0]
			p17 = p17[0]
			p24 = p24[0]
			

		
		idxidx = basename(filepath).find('Idx')
		idx = basename(filepath)[idxidx+4:idxidx+6]
		if idx[1] == '.':
			idx = idx[0]
		name = 'aaaaTestBench_OV_Features_Idx_' + idx
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'

		ylabel_ = []
		color_ = []
		for i in range(15):			
			ylabel_.append(None)
			color_.append(None)
		# Data_Y = [max, rms, avg, crf, std, avs, kur, var, cef, sen, vas, p17, p21, sts, p24]
		
		if config['n_data'] == 1:
			Data_Y = [max*1000, rms*1000, avg*1000, crf, std*1000, avs*1000, kur, var*(1000)**2., cef/1000., sen, vas*(1000)**2., p17/1000*(1000)**0.5, p21*(1000)**0.5, sts/1000., p24*(1000)**0.75]
			legend_ = []
			for i in range(15):
				legend_.append(None)
		else:
			Data_Y = []
			for i in range(config['n_data']):
				Data_Y.append([max[i]*1000, rms[i]*1000, avg[i]*1000, crf[i], std[i]*1000, avs[i]*1000, kur[i], var[i]*(1000)**2., cef[i]/1000., sen[i], vas[i]*(1000)**2., p17[i]/1000*(1000)**0.5, p21[i]*(1000)**0.5, sts[i]/1000., p24[i]*(1000)**0.75])
			legend_ = ['db6', 'db12', 'db18']
			# legend_ = ['sym6', 'sym12', 'sym18']
			legend_ = ['level 3', 'level 5', 'level 7', 'level 9']
			# legend_ = ['No filter', 'BP Filter']
		

		title_ = ['MAX [$\mu$V]', 'RMS [$\mu$V]', 'AVG [$\mu$V]', 'CRF [-]', 'STD [$\mu$V]', 'AVS [$\mu$V]', 'KUR [-]', 'VAR [$\mu$V$^{2}$]', 'CEF [kHz]', 'SEN [-]', 'VAS [$\mu$V$^{2}$]', 'P17 [kHz $\mu$V$^{0.5}$]', 'P21 [$\mu$V$^{0.5}$]', 'STS [kHz]', 'P24 [$\mu$V$^{0.75}$]']

		
		style = {'xlabel':'MC', 'ylabel':ylabel_, 'legend':legend_, 'title':title_, 'customxlabels':[1, 2, 4, 7, 9, 10], 'xlim':None, 'ylim':None, 'color':color_, 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'n_data':config['n_data']}
		
		data_x = np.array([1, 2, 3, 4, 5, 6])
		Data_X = []
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		
		for i in range(15):
			Data_X.append(data_x)
		
		data = {'x':Data_X, 'y':Data_Y}
		plot15_thesis_big(data, style)
	
	
	elif config['mode'] == '12_plot_testbench':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		
		root.destroy()

		kur = []
		sen = []
		max = []
		crf = []
		p21 = []
		
		rms = []
		std = []
		sts = []
		var = []
		vas = []
		
		avg = []
		avs = []
		cef = []
		p17 = []
		p24 = []
		
		for filepath in Filepaths:
			print(basename(filepath))
			workbook = xlrd.open_workbook(filepath)
			worksheet = workbook.sheet_by_name('Analysis')
			
			#group 1
			kur.append(np.array(worksheet.col_values(colx=8-1, start_rowx=2-1, end_rowx=8-1)))
			
			sen.append(np.array(worksheet.col_values(colx=24-1, start_rowx=2-1, end_rowx=8-1)))
			
			max.append(np.array(worksheet.col_values(colx=21-1, start_rowx=2-1, end_rowx=8-1)))
			
			crf.append(np.array(worksheet.col_values(colx=5-1, start_rowx=2-1, end_rowx=8-1)))
			
			p21.append(np.array(worksheet.col_values(colx=18-1, start_rowx=2-1, end_rowx=8-1)))
			
			#group 2
			
			rms.append(np.array(worksheet.col_values(colx=23-1, start_rowx=2-1, end_rowx=8-1)))

			std.append(np.array(worksheet.col_values(colx=25-1, start_rowx=2-1, end_rowx=8-1)))
			
			sts.append(np.array(worksheet.col_values(colx=26-1, start_rowx=2-1, end_rowx=8-1)))
			
			var.append(np.array(worksheet.col_values(colx=11-1, start_rowx=2-1, end_rowx=8-1)))
			
			vas.append(np.array(worksheet.col_values(colx=12-1, start_rowx=2-1, end_rowx=8-1)))
			
			#group 3
			avg.append(np.array(worksheet.col_values(colx=9-1, start_rowx=2-1, end_rowx=8-1)))
			
			avs.append(np.array(worksheet.col_values(colx=10-1, start_rowx=2-1, end_rowx=8-1)))
			
			cef.append(np.array(worksheet.col_values(colx=3-1, start_rowx=2-1, end_rowx=8-1)))
			
			p17.append(np.array(worksheet.col_values(colx=16-1, start_rowx=2-1, end_rowx=8-1)))
			
			p24.append(np.array(worksheet.col_values(colx=20-1, start_rowx=2-1, end_rowx=8-1)))

		if config['n_data'] == 1:
			kur = kur[0]
			sen = sen[0]
			max = max[0]
			crf = crf[0]
			p21 = p21[0]
			
			rms = rms[0]
			std = std[0]
			sts = sts[0]
			var = var[0]
			vas = vas[0]
			
			avg = avg[0]
			avs = avs[0]
			cef = cef[0]
			p17 = p17[0]
			p24 = p24[0]
			

		
		idxidx = basename(filepath).find('Idx')
		idx = basename(filepath)[idxidx+4:idxidx+6]
		if idx[1] == '.':
			idx = idx[0]
		name = 'TestBench_OV_Features_CORR_Idx_' + idx
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'

		ylabel_ = []
		color_ = []
		for i in range(12):			
			ylabel_.append(None)
			color_.append(None)
		# Data_Y = [max, rms, avg, crf, std, avs, kur, var, cef, sen, vas, p17, p21, sts, p24]
		
		if config['n_data'] == 1:
			Data_Y = [max*1000, rms*1000, crf, std*1000, avs*1000, kur, cef/1000., sen, p17/1000*(1000)**0.5, p21*(1000)**0.5, sts/1000., p24*(1000)**0.75]
			legend_ = []
			for i in range(15):
				legend_.append(None)
		else:
			Data_Y = []
			for i in range(config['n_data']):
				Data_Y.append([max[i]*1000, rms[i]*1000, crf[i], std[i]*1000, avs[i]*1000, kur[i], cef[i]/1000., sen[i], p17[i]/1000*(1000)**0.5, p21[i]*(1000)**0.5, sts[i]/1000., p24[i]*(1000)**0.75])
			# legend_ = ['db6', 'db12', 'db18']
			legend_ = ['sym6', 'sym12', 'sym18']
			# legend_ = ['level 3', 'level 5', 'level 7', 'level 9']
			# legend_ = ['No filter', 'BP filter']
		

		title_ = ['MAX [$\mu$V]', 'RMS [$\mu$V]', 'CRF [-]', 'STD [$\mu$V]', 'AVS [$\mu$V]', 'KUR [-]', 'CEF [kHz]', 'SEN [-]', 'P17 [kHz $\mu$V$^{0.5}$]', 'P21 [$\mu$V$^{0.5}$]', 'STS [kHz]', 'P24 [$\mu$V$^{0.75}$]']

		
		style = {'xlabel':'MC', 'ylabel':ylabel_, 'legend':legend_, 'title':title_, 'customxlabels':[1, 2, 4, 7, 9, 10], 'xlim':None, 'ylim':None, 'color':color_, 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'n_data':config['n_data']}
		
		data_x = np.array([1, 2, 3, 4, 5, 6])
		Data_X = []
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		
		for i in range(15):
			Data_X.append(data_x)
		
		data = {'x':Data_X, 'y':Data_Y}
		plot12_thesis_testbench(data, style)
	
	
	elif config['mode'] == '12_plot_ch4':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		
		root.destroy()

		kur = []
		sen = []
		max = []
		crf = []
		p21 = []
		
		rms = []
		std = []
		sts = []
		# var = []
		# vas = []
		
		# avg = []
		avs = []
		cef = []
		p17 = []
		p24 = []
		
		for filepath in Filepaths:
			print(basename(filepath))
			workbook = xlrd.open_workbook(filepath)
			worksheet = workbook.sheet_by_name('Analysis')
			
			#group 1
			kur.append(np.array(worksheet.col_values(colx=8-1, start_rowx=2-1, end_rowx=5-1)))
			
			sen.append(np.array(worksheet.col_values(colx=24-1, start_rowx=2-1, end_rowx=5-1)))
			
			max.append(np.array(worksheet.col_values(colx=21-1, start_rowx=2-1, end_rowx=5-1)))
			
			crf.append(np.array(worksheet.col_values(colx=5-1, start_rowx=2-1, end_rowx=5-1)))
			
			p21.append(np.array(worksheet.col_values(colx=18-1, start_rowx=2-1, end_rowx=5-1)))
			
			#group 2
			
			rms.append(np.array(worksheet.col_values(colx=23-1, start_rowx=2-1, end_rowx=5-1)))

			std.append(np.array(worksheet.col_values(colx=25-1, start_rowx=2-1, end_rowx=5-1)))
			
			sts.append(np.array(worksheet.col_values(colx=26-1, start_rowx=2-1, end_rowx=5-1)))
			
			# var.append(np.array(worksheet.col_values(colx=11-1, start_rowx=2-1, end_rowx=8-1)))
			
			# vas.append(np.array(worksheet.col_values(colx=12-1, start_rowx=2-1, end_rowx=8-1)))
			
			#group 3
			# avg.append(np.array(worksheet.col_values(colx=9-1, start_rowx=2-1, end_rowx=8-1)))
			
			avs.append(np.array(worksheet.col_values(colx=10-1, start_rowx=2-1, end_rowx=5-1)))
			
			cef.append(np.array(worksheet.col_values(colx=3-1, start_rowx=2-1, end_rowx=5-1)))
			
			p17.append(np.array(worksheet.col_values(colx=16-1, start_rowx=2-1, end_rowx=5-1)))
			
			p24.append(np.array(worksheet.col_values(colx=20-1, start_rowx=2-1, end_rowx=5-1)))

		if config['n_data'] == 1:
			kur = kur[0]
			sen = sen[0]
			max = max[0]
			crf = crf[0]
			p21 = p21[0]
			
			rms = rms[0]
			std = std[0]
			sts = sts[0]
			# var = var[0]
			# vas = vas[0]
			
			# avg = avg[0]
			avs = avs[0]
			cef = cef[0]
			p17 = p17[0]
			p24 = p24[0]
			

		
		idxidx = basename(filepath).find('Idx')
		idx = basename(filepath)[idxidx+4:idxidx+6]
		if idx[1] == '.':
			idx = idx[0]
		name = 'WindTurbine_OVFeat_S1'
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_WT_Gearboxes\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'

		ylabel_ = []
		color_ = []
		for i in range(12):			
			ylabel_.append(None)
			color_.append(None)
		# Data_Y = [max, rms, avg, crf, std, avs, kur, var, cef, sen, vas, p17, p21, sts, p24]
		fac_uv = 1.
		if config['n_data'] == 1:
			Data_Y = [max*fac_uv, rms*fac_uv, crf, std*fac_uv, avs*fac_uv, kur, cef/1000., sen, p17/1000*(fac_uv)**0.5, p21*(fac_uv)**0.5, sts/1000., p24*(fac_uv)**0.75]
			legend_ = []
			for i in range(12):
				legend_.append(None)
		else:
			Data_Y = []
			for i in range(config['n_data']):
				Data_Y.append([max[i]*fac_uv, rms[i]*fac_uv, crf[i], std[i]*fac_uv, avs[i]*fac_uv, kur[i], cef[i]/1000., sen[i], p17[i]/1000*(fac_uv)**0.5, p21[i]*(fac_uv)**0.5, sts[i]/1000., p24[i]*(fac_uv)**0.75])
			legend_ = ['db6', 'db12', 'db18']
			# legend_ = ['sym6', 'sym12', 'sym18']
			legend_ = ['level 3', 'level 5', 'level 7', 'level 9']
			# legend_ = ['No filter', 'BP Filter']
		
		if fac_uv == 1000.:
			title_ = ['MAX [$\mu$V]', 'RMS [$\mu$V]', 'CRF [-]', 'STD [$\mu$V]', 'AVS [$\mu$V]', 'KUR [-]', 'CEF [kHz]', 'SEN [-]', 'P17 [kHz $\mu$V$^{0.5}$]', 'P21 [$\mu$V$^{0.5}$]', 'STS [kHz]', 'P24 [$\mu$V$^{0.75}$]']
		else:
			title_ = ['MAX [mV]', 'RMS [mV]', 'CRF [-]', 'STD [mV]', 'AVS [mV]', 'KUR [-]', 'CEF [kHz]', 'SEN [-]', 'P17 [kHz mV$^{0.5}$]', 'P21 [mV$^{0.5}$]', 'STS [kHz]', 'P24 [mV$^{0.75}$]']


		
		style = {'xlabel':None, 'ylabel':ylabel_, 'legend':legend_, 'title':title_, 'customxlabels':['AE-1p', 'AE-2p', 'AE-3p'], 'xlim':None, 'ylim':None, 'color':color_, 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'n_data':config['n_data']}
		
		data_x = np.array([1, 2, 3])
		Data_X = []
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		
		for i in range(12):
			Data_X.append(data_x)
		
		data = {'x':Data_X, 'y':Data_Y}
		plot12_thesis_new(data, style)
	

	
	elif config['mode'] == 'calculate_correlation':

		print('Select Features files')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		print('Select Feature Condition...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath2 = filedialog.askopenfilename()			
		root.destroy()
		
		
		
		mydict2 = pd.read_excel(filepath2, sheetname=config['sheet'])
		mydict2 = mydict2.to_dict(orient='list')
		Feature2 = np.array(mydict2[config['feature_cond']])
		
		index_nonan = []
		count = 0
		for element in Feature2:
			if not isnan(element):
				index_nonan.append(count)
			count +=1
		Feature2 = Feature2[index_nonan]
		if config['feature_cond'] == 'n':
			Feature2 = Feature2/117.18
		
		# ax2 = ax.twinx()
		mydict2 = {}
		count = 0
		titles = ['AE-1', 'AE-2', 'AE-3', 'AE-4']
		features = ['MAX', 'RMS']
		for k in range(2):
			vec = []
			for filepath in Filepaths:
				
				mydict = pd.read_excel(filepath, sheetname=config['sheet'])
				mydict = mydict.to_dict(orient='list')
				Feature = mydict[features[k]]
				
				Feature = 1000*np.array(Feature)/141.3
				
				if config['n_mov_avg'] != 0:
					Feature = movil_avg(Feature, config['n_mov_avg'])			
				
				
				time = np.array([i*10. for i in range(len(Feature))])/60.
				time2 = time[index_nonan]
				
				Feature2int = np.interp(time, time2, Feature2)
				
				# plt.plot(time, Feature, 'b', time, Feature2int, 'or')
				# plt.show()
				print(titles[count], ' = ' , np.corrcoef(Feature, Feature2int)[0][1])
				vec.append(np.corrcoef(Feature, Feature2int)[0][1])
			mydict2[features[k]] = vec
			



		print(mydict2)
		DataFr = pd.DataFrame(data=mydict2, index=titles)
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Correlations')	
		print('Result in Excel table')
			


	else:
		print('unknown mode')
		sys.exit()

		
	return






def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'layers' or element == 'feature_array':
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
	
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# config['n_batches'] = int(config['n_batches'])
	# config['db'] = int(config['db'])
	# config['divisions'] = int(config['divisions'])
	config['n_mov_avg'] = int(config['n_mov_avg'])
	config['train'] = float(config['train'])
	
	config['n_pre'] = float(config['n_pre'])
	config['m_post'] = float(config['m_post'])
	
	config['alpha'] = float(config['alpha'])
	config['tol'] = float(config['tol'])	
	config['learning_rate_init'] = float(config['learning_rate_init'])	
	#Type conversion to int	
	config['max_iter'] = int(config['max_iter'])
	config['rs'] = int(config['rs'])
	config['n_data'] = int(config['n_data'])

	# Variable conversion	
	correct_layers = tuple([int(element) for element in (config['layers'])])
	config['layers'] = correct_layers
	
	config['ylabel'] = config['ylabel'].replace('_', ' ')
	config['zlabel'] = config['zlabel'].replace('_', ' ')
	config['title'] = config['title'].replace('_', ' ')

	
	# Variable conversion
	
	# Variable conversion
	if config['sheet'] == 'OFF':
		config['sheet'] = 0
	
	return config


	
if __name__ == '__main__':
	main(sys.argv)
