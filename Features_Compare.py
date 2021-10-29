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
plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'legend':None, 'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.7, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature_cond':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature3':'RMS', 'feature4':'RMS', 'feature5':'RMS', 'feature_array':['RMS'], 'type_feature':'overall'}

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
		
		title_ = ['Harmonics $n f_{f,r}$', 'Harmonics $p f_{m}$', 'Sidebands $p f_{m} \pm s f_{c}$', 'Harmonics $n f_{f,r}$', 'Harmonics $p f_{m}$', 'Sidebands $p f_{m} \pm s f_{c}$']
		
		
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
	
	

	
	elif config['mode'] == 'eval_feature':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		
		root.destroy()


		for filepath in Filepaths:
			name = 'Scores_' + basename(filepath)
			print(basename(filepath))
			workbook = xlrd.open_workbook(filepath)
			
			if config['type_feature'] == 'overall':	
				Features = {}
				worksheet = workbook.sheet_by_name('Analysis')			
				#group 1
				Features['kur'] = np.array(worksheet.col_values(colx=8-1, start_rowx=2-1, end_rowx=8-1))			
				Features['sen'] = np.array(worksheet.col_values(colx=24-1, start_rowx=2-1, end_rowx=8-1))			
				Features['max'] = np.array(worksheet.col_values(colx=21-1, start_rowx=2-1, end_rowx=8-1))			
				Features['crf'] = np.array(worksheet.col_values(colx=5-1, start_rowx=2-1, end_rowx=8-1))			
				Features['p21'] = np.array(worksheet.col_values(colx=18-1, start_rowx=2-1, end_rowx=8-1))			
				#group 2				
				Features['rms'] = np.array(worksheet.col_values(colx=23-1, start_rowx=2-1, end_rowx=8-1))
				Features['std'] = np.array(worksheet.col_values(colx=25-1, start_rowx=2-1, end_rowx=8-1))			
				Features['sts'] = np.array(worksheet.col_values(colx=26-1, start_rowx=2-1, end_rowx=8-1))			
				Features['var'] = np.array(worksheet.col_values(colx=11-1, start_rowx=2-1, end_rowx=8-1))			
				Features['vas'] = np.array(worksheet.col_values(colx=12-1, start_rowx=2-1, end_rowx=8-1))			
				#group 3
				Features['avg'] = np.array(worksheet.col_values(colx=9-1, start_rowx=2-1, end_rowx=8-1))			
				Features['avs'] = np.array(worksheet.col_values(colx=10-1, start_rowx=2-1, end_rowx=8-1))			
				Features['cef'] = np.array(worksheet.col_values(colx=3-1, start_rowx=2-1, end_rowx=8-1))			
				Features['p17'] = np.array(worksheet.col_values(colx=16-1, start_rowx=2-1, end_rowx=8-1))			
				Features['p24'] = np.array(worksheet.col_values(colx=20-1, start_rowx=2-1, end_rowx=8-1))
				
				# diff_end_thesis(kur)
				# diff_end_thesis(kur)
				MyDict = {'Mono':[], 'DiffStart':[], 'DiffEnd':[], 'Combi':[]}
				titles = ['rms', 'avg', 'crf', 'std', 'avs', 'var', 'cef', 'vas', 'p17', 'p21', 'p24']
				
				for feat in titles:
					v_mono = monotonicity(Features[feat])
					v_start = diff_start_thesis(Features[feat])
					v_end = diff_end_thesis(Features[feat])
					v_combi = (v_start+v_end)*v_mono
					
					MyDict['Mono'].append(v_mono)				
					MyDict['DiffStart'].append(v_start)			
					MyDict['DiffEnd'].append(v_end)
					MyDict['Combi'].append(v_combi)

					
				
				# print('+++++')
				# print(MyDict)
				
				DataFr = pd.DataFrame(data=MyDict, index=titles)
				writer = pd.ExcelWriter(name)
				DataFr.to_excel(writer, sheet_name='Scores')	
				
		sys.exit()		
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
		name = 'TestBench_OV_Features_Idx_' + idx
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
