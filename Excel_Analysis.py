# import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import isnan
from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('D:'))
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

plt.rcParams['savefig.dpi'] = 1500
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode']
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.7, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-1, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity', 'ylabel':'Amplitude_[mV]', 'title':'_', 'color':'#1f77b4', 'feature_cond':'RMS', 'zlabel':'None', 'plot':'OFF', 'interp':'OFF', 'feature2':'RMS', 'feature3':'RMS', 'feature4':'RMS', 'feature_array':['RMS']}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'modify':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			row_names = list(mydict.index)
			mydict = mydict.to_dict(orient='list')
			# Feature += mydict[config['feature']][:-2]
			# print(type(mydict))
			# print(type({}))
			# print(len(multi_sum_values_dict(mydict)))
			# print(len(mydict['aby_T50']))
			mydict['all'] = multi_sum_values_dict(mydict)
			# sys.exit()
			
			config['name'] = basename(filepath)
			
			DataFr = pd.DataFrame(data=mydict, index=row_names)
			writer = pd.ExcelWriter('all_' + config['name'])		
			DataFr.to_excel(writer, sheet_name='AE_Fft')
	
	elif config['mode'] == 'plot_from_xls':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []
		inis = [0]
		acu = 0
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			
			# acu += len(mydict[config['feature']][:-2])
			acu += len(mydict[config['feature']])
			
			if filepath.find('_Last_') != -1 and filepath.find('K1') == -1:
				inis.append(acu)
			# Feature += mydict[config['feature']][:-2]
			Feature += mydict[config['feature']]
		
		# Feature = list(np.nan_to_num(Feature)/1000*(1000)**0.5) #17
		Feature = list(np.nan_to_num(Feature))
		# Feature = list(np.nan_to_num(Feature)*(1000)**0.5) #21
		
		same_length_ = True
		# Feature = median_filter(Feature, 19, same_length_)
		# Feature = median_filter(Feature, 11, same_length_)
		# Feature = median_filter(Feature, 9, same_length_)
		# Feature = median_filter(Feature, 7, same_length_)
		# Feature = median_filter(Feature, 3, same_length_)
		# Feature = median_filter(Feature, 3, same_length_)
		# Feature = median_filter(Feature, 3, same_length_)
		# Feature = median_filter(Feature, 3, same_length_)
		# Feature = median_filter(Feature, 5, same_length_)
		# Feature = median_filter(Feature, 7, same_length_)
		# Feature = median_filter(Feature, 9, same_length_)
		# Feature = median_filter(Feature, 11, same_length_)
		
		Feature = movil_avg(Feature, config['n_mov_avg'])
		# Feature = np.array(Feature)/70.8
		
		print(inis)
		print(len(inis))
		print('Length!!...', len(Feature))
		
		t = np.array([i for i in range(len(Feature))])*5/60/60.
		inis = list(np.array(inis)*5/60/60.)
		# plt.plot(t, Feature)
		# plt.show()
		n_files = len(Filepaths)
		# 'Max. cross-correlation'
		# 0.05, 0.25
		# 'Rate of AE bursts'
		
		# name = 'Long_STD'
		# range_ = [110,160]
		# ylabel_ = 'STD [$\mu$V]'
		
		# name = 'Long_Temp'
		# range_ = [20,60]
		# ylabel_ = 'Temperature [°C]'
		
		# name = 'Long_P17'
		# range_ = [5,15]
		# ylabel_ = 'P17 []'
		
		# name = 'Long_P21'
		# range_ = [0.25,0.45]
		# ylabel_ = r'P21 [$\it{\mu}$V$^{0.5}$]'
		
		# name = 'TestBench_Long_Bursts_Fixed_Counting'
		# range_ = [0,140]
		# ylabel_ = r'Rate of AE bursts'
		
		name = 'TestBench_Long_P21_corr'
		range_ = [110,160]
		range_ = [0.2,0.35]
		ylabel_ = r'Max. cross-correlation'
		factor = (1000.)**0.5
		# name = 'Long_Fc'
		# range_ = [40,80]
		# # range_ = [20,150]
		# ylabel_ = r'Harmonics $\it{n}\ f_{c}$ [$\it{\mu}$V]'
		
		# name = 'Long_Fm'
		# range_ = [00,250]
		# ylabel_ = r'Harmonics $\it{p}\ f_{m}$ [$\it{\mu}$V]'
		
		# name = 'Long_SB'
		# range_ = [150,400]
		# ylabel_ = r'Sidebands $\it{p}\ f_{m} \pm \it{s}\ f_{c}$ [$\it{\mu}$V]'
		
		
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		
		style = {'xlabel':'Cumulative operating hours', 'ylabel':ylabel_, 'legend':[None], 'title':None, 'customxlabels':None, 'xlim':[0.0, 10.5], 'ylim':range_, 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':inis, 'range_lines':range_, 'output':'save', 'path_1':path_1b, 'path_2':path_2b}
		
		Feature = np.array(Feature)*factor
		data = {'x':[t], 'y':[Feature]}
		
		plot1_thesis(data, style)
		
		
		mono = monotonicity_poly(Feature, 51)
		print('monotonicity = ', mono)
		print('len = ', len(Feature))
		
		
		
		sys.exit()
		
		
		
		
		fig, ax = plt.subplots()
		
		
		# n_files = 9
		
		# ax.set_xticks([i/720 for i in range(n_files)])
		# ax.set_xticklabels([i for i in range(n_files)])
		
		ax.set_xlabel('Time', fontsize=13)
		
		
		
		# ax.plot(np.array(t)/720., np.array(Feature), color=config['color'])
		ax.plot(np.array(t), np.array(Feature), '-')
		# ax.set_xlabel('Time [h]', fontsize=12)
		ax.set_ylabel('Magnitude', fontsize=13)
		# ax.set_ylabel(config['ylabel'], fontsize=13)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier (4 min mov. avg.)', fontsize=12)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		filename = basename(Filepaths[0])
		# config['title'] = filename[filename.find('Idx') : filename.find('Idx') + 5] + ' ' + config['feature']
		ax.set_title(config['title'], fontsize=13)

		ax.tick_params(axis='both', labelsize=12)
		
		# plt.savefig(config['title'] + '_' + config['feature'] + '.png')
		plt.show()
		
		# noise = np.random.normal(loc=0., scale=1., size=len(Feature))
		# print('Feature - Ones')
		# print(np.corrcoef(np.array(Feature), noise+np.ones(len(Feature)))[0][1])
		# print('Feature - Zeros')
		# print(np.corrcoef(np.array(Feature), noise+np.zeros(len(Feature)))[0][1])
		# print('Feature - 45deg')
		# print(np.corrcoef(np.array(Feature), np.arange(len(Feature)))[0][1])
		# print('45deg - 45deg')
		# print(np.corrcoef(np.arange(len(Feature)), np.arange(len(Feature)))[0][1])
		# print('Ones - Ones')
		# print(np.corrcoef(noise+np.ones(len(Feature)), noise+np.ones(len(Feature)))[0][1])
		# print('Zeros - Zeros')
		# print(np.corrcoef(noise+np.zeros(len(Feature)), noise+np.zeros(len(Feature)))[0][1])
		# print('Zeros - Ones')
		# print(np.corrcoef(noise+np.zeros(len(Feature)), noise+np.ones(len(Feature)))[0][1])
		# print('Zeros - 45deg')
		# print(np.corrcoef(noise+np.zeros(len(Feature)), noise+np.arange(len(Feature)))[0][1])
		# print('Ones - 45deg')
		# print(np.corrcoef(noise+np.ones(len(Feature)), noise+np.arange(len(Feature)))[0][1])
		
		# print(Feature)
	
	elif config['mode'] == 'plot_from_xls_screw':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature_100 = []
		Feature_140 = []
		Feature_180 = []
		Feature_220 = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature_100 += mydict['100_140']
			Feature_140 += mydict['140_180']
			Feature_180 += mydict['180_220']
			Feature_220 += mydict['220_260']
		
		# Feature = list(np.nan_to_num(Feature))

		t = np.array([i for i in range(len(Feature_100))])
		
		
		
		
		fig, ax = plt.subplots()
		perro = t[0::3]
		xticklabels = (perro + np.ones(len(perro)))*1260
		xlabels = t[0::3]
		print()
		ax.set_xticklabels(xticklabels)
		ax.set_xticks(xlabels)
		ax.axvline(x=(25200+1260/2.)/1260)
		
		ax.set_xlabel('Time', fontsize=13)
		
		Feature_100 = np.log(np.ones(len(Feature_100)) + np.array(Feature_100))
		Feature_140 = np.log(np.ones(len(Feature_140)) + np.array(Feature_140))
		Feature_180 = np.log(np.ones(len(Feature_180)) + np.array(Feature_180))
		Feature_220 = np.log(np.ones(len(Feature_220)) + np.array(Feature_220))

		
		ax.bar(np.array(t), Feature_100)
		ax.bar(np.array(t), Feature_140, bottom=Feature_100)
		ax.bar(np.array(t), Feature_180, bottom=Feature_100+Feature_140)
		ax.bar(np.array(t), Feature_220, bottom=Feature_100+Feature_140+Feature_180)
		
		ax.set_ylabel('Magnitude', fontsize=13)
		

		ax.set_title(config['title'], fontsize=13)

		ax.tick_params(axis='both', labelsize=12)
		
		plt.show()
	
	elif config['mode'] == 'plot_from_xls_screw_2':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature_100 = []
		Feature_120 = []
		Feature_140 = []
		Feature_160 = []
		Feature_180 = []
		Feature_200 = []
		Feature_220 = []
		Feature_240 = []
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature_100 += mydict['100']
			Feature_120 += mydict['120']
			Feature_140 += mydict['140']
			Feature_160 += mydict['160']
			Feature_180 += mydict['180']
			Feature_200 += mydict['200']
			Feature_220 += mydict['220']
			Feature_240 += mydict['240']
		
		# Feature = list(np.nan_to_num(Feature))

		t = np.array([i for i in range(len(Feature_100))])
		
		
		
		
		fig, ax = plt.subplots()
		perro = t[0::3]
		xticklabels = (perro + np.ones(len(perro)))*1260
		xlabels = t[0::3]
		print()
		ax.set_xticklabels(xticklabels)
		ax.set_xticks(xlabels)
		ax.axvline(x=(25200+1260/2.)/1260)
		
		ax.set_xlabel('Time', fontsize=13)
		
		Feature_100 = np.log(np.ones(len(Feature_100)) + np.array(Feature_100))
		Feature_120 = np.log(np.ones(len(Feature_120)) + np.array(Feature_120))
		Feature_140 = np.log(np.ones(len(Feature_140)) + np.array(Feature_140))
		Feature_160 = np.log(np.ones(len(Feature_160)) + np.array(Feature_160))
		Feature_180 = np.log(np.ones(len(Feature_180)) + np.array(Feature_180))
		Feature_200 = np.log(np.ones(len(Feature_200)) + np.array(Feature_200))
		Feature_220 = np.log(np.ones(len(Feature_220)) + np.array(Feature_220))
		Feature_240 = np.log(np.ones(len(Feature_240)) + np.array(Feature_240))

		
		ax.bar(np.array(t), Feature_100)
		ax.bar(np.array(t), Feature_120, bottom=Feature_100)
		ax.bar(np.array(t), Feature_140, bottom=Feature_100+Feature_120)
		ax.bar(np.array(t), Feature_160, bottom=Feature_100+Feature_120+Feature_140)
		ax.bar(np.array(t), Feature_180, bottom=Feature_100+Feature_120+Feature_140+Feature_160)
		ax.bar(np.array(t), Feature_200, bottom=Feature_100+Feature_120+Feature_140+Feature_160+Feature_180)
		ax.bar(np.array(t), Feature_220, bottom=Feature_100+Feature_120+Feature_140+Feature_160+Feature_180+Feature_200)
		ax.bar(np.array(t), Feature_240, bottom=Feature_100+Feature_120+Feature_140+Feature_160+Feature_180+Feature_200+Feature_220)
		
		
		
		
		ax.set_ylabel('Magnitude', fontsize=13)
		

		ax.set_title(config['title'], fontsize=13)

		ax.tick_params(axis='both', labelsize=12)
		
		plt.show()
	
	
	elif config['mode'] == 'eliminate_nan':
		# print('Select xls')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		# Feature = []		
		# for filepath in Filepaths:
			# mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			# mydict = mydict.to_dict(orient='list')
			# Feature += mydict[config['feature']][:-2]
		# Feature = np.array(Feature)
		# print(np.isnan(Feature))
		# vec = np.isnan(Feature)
		# for i in range(len(vec)):
			# vec[i] = not vec[i]
		# # print(np.argwhere(np.isnan(Feature)))
		# print(Feature[np.argwhere(vec)])
		# sys.exit()
		
		
		
		
		
		
		
		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
		
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index.values)
			
			mydict = mydict.to_dict(orient='list')
			mydict2 = {}
			
			for key in mydict.keys():
				# data = mydict[key]
				# print(data)
				# sys.exit()
				data = np.array(mydict[key])
				vec = np.isnan(data)
				# print(vec)

				
				
				for i in range(len(vec)):
					# if vec[i] == True:
						# print('!!!!!!')
					vec[i] = not vec[i]
				data2 = np.ravel(data[np.argwhere(vec)])
				# print(np.ravel(data2))
				# sys.exit()
				mydict2[key] = data2
			
			# rownames2 = rownames[np.argwhere(vec)]
			# print(np.argwhere(vec))
			# print(np.argwhere(vec)[0])
			
			rownames2 = [rownames[index] for index in np.ravel(np.argwhere(vec))]
			# print(rownames2)
			# sys.exit()
			# writer = pd.ExcelWriter(basename(filepath)[-6:-5] + '_' + basename(filepath)[:-5] + '.xlsx')
			# writer = pd.ExcelWriter(basename(filepath)[-11:-10] + '_' + basename(filepath)[:-5] + '.xlsx')	
			
			# writer = pd.ExcelWriter(basename(filepath)[11] + '_' + basename(filepath)[:-5] + '.xlsx')

			writer = pd.ExcelWriter(basename(filepath))				
			# print(mydict2)
			
			DataFr = pd.DataFrame(data=mydict2, index=rownames2)		
			DataFr.to_excel(writer, sheet_name='Bursts')
			writer.close()
		
		
		
		
		
		
	
	
	elif config['mode'] == 'hist_from_xls':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		
		Feature = list(np.nan_to_num(Feature))
		# Feature = movil_avg(Feature, config['n_mov_avg'])
		
		
		
		# print(Feature)
		
		t = [i for i in range(len(Feature))]
		n_files = len(Filepaths)
		
		
		
		print('Mean: ', np.mean(np.array(Feature)))
		print('Std: ', np.std(np.array(Feature)))
		
		
		# n_files = 9
		
		# ax.set_xticks([(i)*180 for i in range(n_files+1)])
		# ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		# ax.set_xlabel('N° Campaign', fontsize=13)
		
		
		# plt.plot(np.array(Feature))
		# plt.show()
		fig, ax = plt.subplots()
		ax.hist(np.array(Feature), bins=25)
		# ax.set_xlabel('Time [h]', fontsize=12)
		# ax.set_ylabel('RMS Value [mV]', fontsize=12)
		ax.set_ylabel(config['ylabel'], fontsize=13)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier (4 min mov. avg.)', fontsize=12)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		ax.set_title(config['title'], fontsize=13)

		ax.tick_params(axis='both', labelsize=12)
		
		# plt.savefig(config['title'] + '_' + config['feature'] + '.png')
		# plt.show()
		# print(Feature)
		
		
		
		from scipy.stats import norm, powerlognorm
		
		mu = 0.11677
		std = 0.05278
		
		x = np.linspace(-0.1, 1, 1000)
		y = norm.pdf(x, mu, std)
		# c, s = 50., 1.
		# y = powerlognorm.pdf(x, c, s)
		
		ax2 = ax.twinx()
		ax2.plot(x, y, 'r-', lw=5, alpha=0.6, label='norm pdf')
		# print(norm.cdf(mu+std, mu, std) - norm.cdf(mu-std, mu, std))
		plt.show()
		
		
		vec = []
		for element in Feature:
			value = norm.cdf(element, mu, std)
			# value = norm.sf(element, c, s)
			vec.append(value)
		
		
		vec = movil_avg(vec, config['n_mov_avg'])
		plt.plot(vec, color='g')
		plt.show()
		
		
		mydict = {}
		
		row_names = [i for i in range(len(vec))]
		mydict['SP_nma10'] = vec


		
		DataFr = pd.DataFrame(data=mydict, index=row_names)
		writer = pd.ExcelWriter('to_use_batch_' + '.xlsx')

	
		DataFr.to_excel(writer, sheet_name='Sheet1')	
		print('Result in Excel table')
		
		
		
		
		
		# x = np.linspace(-0.1, 1, 1000)
		# y = powerlognorm.pdf(x, mu, std)
		# ax2 = ax.twinx()
		# ax2.plot(x, y, 'r-', lw=5, alpha=0.6, label='powerlognorm pdf')
		# # print(norm.cdf(mu+std, mu, std) - norm.cdf(mu-std, mu, std))
		# plt.show()
		
		# vec = []
		# for element in Feature:
			# value = powerlognorm.sf(element, mu, std)
			# vec.append(value)
		
		
		# vec = movil_avg(vec, config['n_mov_avg'])
		# plt.plot(vec)
		# plt.show()
	
	elif config['mode'] == 'two_scale_plot':
		print('Select Feature')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		
		mydict = pd.read_excel(filepath, sheetname=config['sheet'])
		mydict = mydict.to_dict(orient='list')
		Feature = mydict[config['feature']]
		
		print('Select Feature Condition...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath2 = filedialog.askopenfilename()			
		root.destroy()
		
		mydict2 = pd.read_excel(filepath2, sheetname=config['sheet'])
		mydict2 = mydict2.to_dict(orient='list')
		Feature2 = mydict2[config['feature_cond']]
		
		
		time = [i*10. for i in range(len(Feature))]
		
		fig, ax = plt.subplots()
		ax.set_xlabel('Time s')
		
		ax.plot(time, Feature, '-b')
		ax.set_ylabel('RMS Value', color='b')
		ax.tick_params('y', colors='b')				
		# data = data * 1000
		ax2 = ax.twinx()
		ax2.plot(time, 	Feature2, 'or')
		ax2.set_ylabel('Temperature', color='r')
		ax2.tick_params('y', colors='r')				


		plt.show()
	
	elif config['mode'] == '4x_two_scale_plot':
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
		
		
		fig, ax = plt.subplots(nrows=2, ncols=2)
		# ax2 = ax.twinx()
		
		count = 0
		mylist = [[0,0], [0,1], [1,0], [1,1]]
		titles = ['AE-1', 'AE-2', 'AE-3', 'AE-4']
		for filepath in Filepaths:
		
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature = mydict[config['feature']]
			
			Feature = 1000*np.array(Feature)/141.3
			
			if config['n_mov_avg'] != 0:
				Feature = movil_avg(Feature, config['n_mov_avg'])			
			
			
			time = np.array([i*10. for i in range(len(Feature))])/60.
			time2 = time[index_nonan]
			
			ax[mylist[count][0]][mylist[count][1]].plot(time, Feature, '-b', label=titles[count])
			
			ax[mylist[count][0]][mylist[count][1]].set_xlabel('Time [min]', fontsize=15)
			
			if config['feature'] == 'RMS':
				ax[mylist[count][0]][mylist[count][1]].set_ylabel('RMS value [mV]', color='b', fontsize=15)
				ax[mylist[count][0]][mylist[count][1]].set_ylim(bottom=0, top=0.8)
			else:
				ax[mylist[count][0]][mylist[count][1]].set_ylabel('Maximum value [mV]', color='b', fontsize=15)
				ax[mylist[count][0]][mylist[count][1]].set_ylim(bottom=0, top=40)
			
			ax[mylist[count][0]][mylist[count][1]].tick_params('y', colors='b')				
			
			ax[mylist[count][0]][mylist[count][1]].legend(fontsize=14, loc='upper left')
			ax[mylist[count][0]][mylist[count][1]].tick_params(axis='both', labelsize=14)
			
			ax[mylist[count][0]][mylist[count][1]].set_xlim(left=0, right=14)
			
			ax2 = ax[mylist[count][0]][mylist[count][1]].twinx()
			
			ax2.set_xlim(left=0, right=14)
			
			if config['feature_cond'] == 'T':
				ax2.set_ylim(bottom=25, top=55)
				ax2.set_ylabel('Temperature [°C]', color='r', fontsize=15)
			elif config['feature_cond'] == 'n':
				ax2.set_ylim(bottom=250, top=1750)
				ax2.set_ylabel('Rotational speed [CPM]', color='r', fontsize=15)
			
			# ax2.set_ylabel('Rotational speed [CPM]', color='r', fontsize=15)
			
			ax2.plot(time2, Feature2, 'or')			
			
			ax2.tick_params('y', colors='r')
			ax2.tick_params(axis='both', labelsize=14)
			
			
			
			
			
			count += 1

		fig.set_size_inches(12, 6.5)
		# plt.tight_layout()
		plt.show()
	
	elif config['mode'] == '1x4_two_scale_plot':
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
	
	
	
	
	
	
	elif config['mode'] == 'progno_from_xls':

		
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()		
		n_files = len(Filepaths)		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']][:-2]
		Feature = list(np.nan_to_num(Feature))		
		n = len(Feature)		
		if config['feature'] == 'Avg':
			for i in range(len(Feature)):
				if Feature[i] == 0:
					Feature[i] = Feature[i+1]
					print('!!!!!!!!!!!')		
		feature_raw = movil_avg(Feature, 1)

		
		
		
		
		Atributo = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Atributo += mydict[config['feature']][:-2]
		Atributo = list(np.nan_to_num(Atributo))		
		if config['feature'] == 'Avg':
			for i in range(len(Atributo)):
				if Atributo[i] == 0:
					Atributo[i] = Atributo[i+1]
					print('!!!!!!!!!!!')
		
		trend = movil_avg(Atributo, config['n_mov_avg'])		
		trend = list(butter_lowpass(x=trend, order=3, freq=0.01, fs=1.))
		t = np.array([i for i in range(len(feature_raw))])		
		
		if config['plot'] == 'ON':
			fig, ax = plt.subplots()
			
			ax.set_xticks([(i)*180 for i in range(n_files+1)])
			ax.set_xticklabels([i+1 for i in range(n_files+1)])
			
			ax.set_xlabel('N° Campaign', fontsize=13)
			# ax.set_ylabel(config['ylabel'] + ' [m$V^{2}$]', fontsize=13)
			ax.set_ylabel(config['ylabel'] + ' [-]', fontsize=13)
			ax.set_title(config['title'], fontsize=13)				
			ax.tick_params(axis='both', labelsize=12)

			ax.plot(t, feature_raw, 'b')		
			ax.plot(t, trend, 'r', linewidth=2.5)
			
			params = {'mathtext.default': 'regular' }          
			plt.rcParams.update(params)
			
			plt.show()

		
		
		
		
		
		
		
		
		#Prognosability
		final_value = trend[-1]
		initial_value = trend[0]
		# progno = np.exp(np.std(np.array([final_value])) / np.absolute(final_value - initial_value))
		# progno = np.exp(-1. / np.absolute(final_value - initial_value))
		progno = np.exp(-initial_value / np.absolute(final_value - initial_value))
		# progno = initial_value / np.absolute(final_value - initial_value)
	
		
		
		#Monotonicity
		diff_len = 1
		d_trend = diff_signal(trend, diff_len)
		pos = 0
		neg = 0
		for element in d_trend:
			if element >= 0:
				pos += 1
			else:
				neg += 1
		mono = pos/(n-diff_len) - neg/(n-diff_len)
		
		
		#Trendability1
		dd_trend = diff_signal(d_trend, diff_len)
		pos2 = 0
		for element in dd_trend:
			if element >= 0:
				pos2 += 1
		trend1 = pos/(n-diff_len) + pos2/(n-diff_len-1)
		trend1 = 1 - trend1
		
		
		#Trendability2
		# Feature_smooth = np.array(Feature_smooth)
		# t = np.array(t)
		trend = np.array(trend)
		# trend2 = np.absolute(n*np.sum(trend*t) - np.sum(trend)*np.sum(t)) / ((n*np.sum(trend**2.0) - np.sum(trend)**2.0) * (n*np.sum(t**2.0) - np.sum(t)**2.0))**0.5
		trend2b = np.corrcoef(trend, t)
		# if trend2 != trend2b[0][1]:
			# print('fatal error 555')
			# print(trend2)
			# print(trend2b[0][1])
			# sys.exit()
		
		#Robustness		
		# Feature_raw = np.array(Feature_raw)
		residual = np.array(feature_raw) - np.array(trend)
		robust = np.sum(np.exp(-np.absolute(residual/feature_raw)))/n
		
		
		print('Progno', progno)
		print('Mono', mono)
		# print('Trend1', trend1)
		print('Trend2', trend2b[0][1])
		print('Robust', robust)
		
		
		
		mydict = {}
	
		# row_names = [basename(filepath) for filepath in Filepaths]
		# row_names.append('Mean')
		# row_names.append('Std')

		mydict['Progno'] = [progno]
		mydict['Mono'] = [mono]
		mydict['Trend'] = [trend2b[0][1]]
		mydict['Robust'] = [robust]
		


		DataFr = pd.DataFrame(data=mydict, index=[0])
		writer = pd.ExcelWriter(config['name'] + '.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Progno_Param')	
		print('Result in Excel table')
		
		
		
	
	elif config['mode'] == 'plot_from_xls_trend':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		n_files = len(Filepaths)
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		# print(Feature)
		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				count += 1
				Feature[i] = Feature[i + count]				
		# Feature = np.nan_to_num(Feature)
		
		# Feature = list(np.nan_to_num(Feature))
		
		simple = movil_avg(Feature, 0)
		# simple = diff_signal(simple, 1)
		# print(simple)
		# plt.plot(caca1)
		# plt.show()
		
		
		Atributo = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Atributo += mydict[config['feature']]
		for i in range(len(Atributo)):
			count = 0
			while np.isnan(Atributo[i]) == True:
				count += 1
				Atributo[i] = Atributo[i + count]			
		# Atributo = list(np.nan_to_num(Atributo))
		trend = movil_avg(Atributo, config['n_mov_avg'])
		# plt.plot(peoN)
		# plt.show()
		# trend = diff_signal(trend, 1)
		
		
		# plt.plot(caca1)
		# plt.show()
		fig, ax = plt.subplots(ncols=2, nrows=1)
		t = np.array([i for i in range(len(simple))])		
		ax[0].set_xticks([(i)*180 for i in range(n_files+1)])
		ax[0].set_xticklabels([i+1 for i in range(n_files+1)])
		
		ax[0].set_xlabel('N° Campaign', fontsize=13)
		ax[0].set_ylabel(config['ylabel'], fontsize=13)
		ax[0].set_title(config['title'], fontsize=13)				
		ax[0].tick_params(axis='both', labelsize=12)
		
		
		# ax[1].set_xticks([(i)*180 for i in range(n_files+1)])
		# ax[1].set_xticklabels([i+1 for i in range(n_files+1)])
		
		
		ax[1].set_xlabel('N° Campaign', fontsize=13)
		ax[1].set_ylabel(config['ylabel'], fontsize=13)	
		ax[1].set_title(config['title'] + ' (Mov. Average)', fontsize=13)				
		ax[1].tick_params(axis='both', labelsize=12)
		
		
		
		
		ax[0].plot(t, simple, 'b')
		
		ax[1].plot(t, trend, 'r')
		
		fig.set_size_inches(10, 4)
		fig.tight_layout()
		
		plt.show()
		
		
		
		
		# data = {'x':5*t/3600., 'y':trend}
		# style = {'ylabel':'AE bursts [-]', 'xlabel':'Time [h]', 'customxlabels':None, 'legend':False, 'title':'Fixed counting', 'ylim':[0, 125], 'xlim':None}
		# plot1_thesis(data, style)
	
	
	elif config['mode'] == 'plot_from_xls_trend_2':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		n_files = len(Filepaths)
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']][:-2]
		Feature = list(np.nan_to_num(Feature))
		
		if config['feature'] == 'Avg':
			for i in range(len(Feature)):
				if Feature[i] == 0:
					Feature[i] = Feature[i+1]
					print('!!!!!!!!!!!')
		
		feature_raw = movil_avg(Feature, 1)

		
		
		
		
		Atributo = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Atributo += mydict[config['feature']][:-2]
		Atributo = list(np.nan_to_num(Atributo))
		
		
		if config['feature'] == 'Avg':
			for i in range(len(Atributo)):
				if Atributo[i] == 0:
					Atributo[i] = Atributo[i+1]
					print('!!!!!!!!!!!')
		
		trend = movil_avg(Atributo, config['n_mov_avg'])

		
		
		trend = list(butter_lowpass(x=trend, order=3, freq=0.01, fs=1.))
		
		
		# plt.plot(caca1)
		# plt.show()
		fig, ax = plt.subplots()
		t = np.array([i for i in range(len(feature_raw))])		
		ax.set_xticks([(i)*180 for i in range(n_files+1)])
		ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		ax.set_xlabel('N° Campaign', fontsize=13)
		# ax.set_ylabel(config['ylabel'] + ' [m$V^{2}$]', fontsize=13)
		ax.set_ylabel(config['ylabel'] + ' [mV]', fontsize=13)

		ax.set_title(config['title'], fontsize=13)				
		ax.tick_params(axis='both', labelsize=12)
		
		
		# ax.set_xticks([(i)*180 for i in range(n_files+1)])
		# ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		
		# ax.set_xlabel('N° Campaign', fontsize=13)
		# ax.set_ylabel(config['ylabel'], fontsize=13)	
		# ax.set_title(config['title'] + ' (Mov. Average)', fontsize=13)				
		# ax.tick_params(axis='both', labelsize=12)
		
		
		
		
		ax.plot(t, feature_raw, 'b')
		
		ax.plot(t, trend, 'r', linewidth=2.5)
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		
		# fig.set_size_inches(10, 4)
		# fig.tight_layout()
		
		plt.show()
	
	elif config['mode'] == 'plot_from_xls_trend_3_cost':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		n_files = len(Filepaths)
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']][:-2]
		Feature = list(np.nan_to_num(Feature))
		
		if config['feature'] == 'Avg':
			for i in range(len(Feature)):
				if Feature[i] == 0:
					Feature[i] = Feature[i+1]
					print('!!!!!!!!!!!')
		
		feature_raw = movil_avg(Feature, 1)

		
		
		
		
		Atributo = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Atributo += mydict[config['feature']][:-2]
		Atributo = list(np.nan_to_num(Atributo))
		
		
		if config['feature'] == 'Avg':
			for i in range(len(Atributo)):
				if Atributo[i] == 0:
					Atributo[i] = Atributo[i+1]
					print('!!!!!!!!!!!')
		
		N_MOV_AVG = [1, 9, 18, 36, 50, 74, 100, 130, 150, 180, 200, 250, 270, 320, 450, 700]
		Targets = []
		for n_mov_avg in N_MOV_AVG:
		
			trend = movil_avg(Atributo, n_mov_avg)

			trend = list(butter_lowpass(x=trend, order=3, freq=0.01, fs=1.))
		
			target = np.corrcoef(trend, feature_raw)[0][1] / np.log(np.std(trend))
			# target = np.corrcoef(trend, feature_raw)[0][1] / (np.std(trend))**0.5


			Targets.append(target)
		
		plt.plot(N_MOV_AVG, Targets)
		plt.show()
		# plt.plot(caca1)
		# plt.show()
		fig, ax = plt.subplots()
		t = np.array([i for i in range(len(feature_raw))])		
		ax.set_xticks([(i)*180 for i in range(n_files+1)])
		ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		ax.set_xlabel('N° Campaign', fontsize=13)
		ax.set_xlabel('N° Campaign', fontsize=13)
		# ax.set_ylabel(config['ylabel'] + ' [m$V^{2}$]', fontsize=13)
		ax.set_ylabel(config['ylabel'] + ' [mV]', fontsize=13)

		ax.set_title(config['title'], fontsize=13)				
		ax.tick_params(axis='both', labelsize=12)
		
		
		# ax.set_xticks([(i)*180 for i in range(n_files+1)])
		# ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		
		# ax.set_xlabel('N° Campaign', fontsize=13)
		# ax.set_ylabel(config['ylabel'], fontsize=13)	
		# ax.set_title(config['title'] + ' (Mov. Average)', fontsize=13)				
		# ax.tick_params(axis='both', labelsize=12)
		
		
		
		
		ax.plot(t, feature_raw, 'b')
		
		ax.plot(t, trend, 'r', linewidth=2.5)
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		
		# fig.set_size_inches(10, 4)
		# fig.tight_layout()
		
		plt.show()
	
	elif config['mode'] == 'plot_from_xls_teeth':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# Feature = []
		MasterDict = {}
		count = 0
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			
			if count == 0:
				Names = []
				for key in mydict:
					if key != 'Sum' and key != 'N_Burst' and key != 'Avg':
						Names.append(key)
			
			
			mydict = mydict.to_dict(orient='list')
			
			for key, value in mydict.items():
				perro = len(mydict[key])
				newvalue = value				
				newvalue = list(np.nan_to_num(newvalue))
				newvalue = movil_avg(newvalue, config['n_mov_avg'])
				
				
				# newdict = {}
				# for key, values in mydict.items():
				# newdict[key] = movil_avg(mydict[key], config['n_mov_avg'])
				
				
				
				if len(value) != len(newvalue):
					print('fatal error 79')
					sys.exit()
				mydict[key] = newvalue				
				loro = len(mydict[key])
				if perro != loro:
					print('fatal error 78')
					sys.exit()
			

			
			if count == 0:
				for name in Names:
					# MasterDict[name] = mydict[name][:-1]
					MasterDict[name] = mydict[name]
			else:
				for name in Names:
					# MasterDict[name] += mydict[name][:-1]
					MasterDict[name] += mydict[name]
			count += 1
			
			
		# print(MasterDict['aaa_T0'])
		# sys.exit()
			
		# Feature = movil_avg(Feature, config['n_mov_avg'])
		# sys.exit()
		# t = np.array([i*5./60/60 for i in range(len(MasterDict['aar_T17']))])
		
		
		
		# perro = np.array(np.array(MasterDict['aaa_T0']), np.array(MasterDict['aab_T1']), np.array(MasterDict['aac_T2']))
		
		# perro = np.array([MasterDict['aaa_T0'], MasterDict['aab_T1'], MasterDict['aac_T2']])
		
		MAP = []
		for key in MasterDict:
			if key != 'Sum' and key != 'N_Burst' and key != 'Avg':
				# print(key)
				# MAP.append(list(np.log10(np.ones(len(MasterDict[key]))+np.array(MasterDict[key]))))
				
				MAP.append(MasterDict[key])
		map_array = MAP		
		MAP = np.array(MAP)
		print(np.max(MAP))
		fgato = np.arange(72)
		
		
		
		# print(len(t))
		# print(len(fgato))
		# print(len(MAP))
		# print(len(MAP[0]))
		
		vmax = None		
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		
		t = np.array([i for i in range(len(MasterDict['aar_T17']))])*5/3600.
		t_array = t
		# fig, ax = plt.subplots()
		print('!!!!!!!!!!!')
		print(t.shape)
		print(fgato.shape)
		print(MAP.shape)
		
		print(t.shape)
		print(fgato.shape)
		print(MAP.shape)
		print('!!!!!!!!!!!')
		t, fgato = np.meshgrid(t, fgato)
		
		# caca = ax.pcolormesh(t, fgato, MAP, cmap='hsv', vmax=vmax)
		# caca = ax.plot_surface(t, fgato, MAP, cmap='plasma', vmax=vmax)
		caca = ax.plot_surface(t, fgato, MAP, cmap='plasma')
		# caca.se
		# fig.colorbar(caca, ax=ax)
		# fig.colorbar(im, ax=ax0)
		# fig.colorbar(im, ax=ax0)
		
		# print(type(caca.cm))
		
		# caca.set_label('# of contacts')
		# map = []
		# vmax = max_cspectrum(perro)
		
		# print(vmax)
		# map.append(ax.pcolormesh(t, fgato, perro, cmap='plasma', vmax=vmax))
		# norm = colors.Normalize(vmin=1, vmax=500)
		# cbar = fig.colorbar(caca, ax=ax, norm=norm)
		
		# cbar = fig.colorbar(caca, ax=ax)
		# cbar.set_label(config['zlabel'], fontsize=13)
		
		n_files = len(Filepaths)
		# ax.set_xticks([(i)*180 for i in range(n_files+1)])
		# ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		
		ax.set_yticks([(i)*14.2 for i in range(6)])
		ax.set_yticklabels([(i)*72 for i in range(6)])
		
		
		ax.set_xlabel('Accumulated Operating Hours', fontsize=13)
		# ax.set_ylabel('RMS Value [mV]', fontsize=12)
		ax.set_ylabel('Planet Carrier Angle [°]', fontsize=13)
		ax.set_zlabel('Maximal Amplitude [mV]', fontsize=13)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier (4 min mov. avg.)', fontsize=12)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		ax.set_title(config['title'], fontsize=12)
		ax.tick_params(axis='both', labelsize=12)
		
		# plt.savefig(config['title'] + '_' + config['feature'] + '.png')
		plt.show()
		# print(Feature)
		
		map_array = np.array(map_array)
		map_array = np.transpose(map_array)
		print(type(map_array))
		print('shape map = ', map_array.shape)
		print('shape map = ', map_array[0,:].shape)
		print('shape map = ', map_array[0].shape)
		print('shape t = ', t_array.shape)
		
		# sys.exit()
		
		from matplotlib import font_manager		
		params = {'font.family':'Times New Roman'}
		plt.rcParams.update(params)	
		params = {'mathtext.fontset': 'stix' }	
		plt.rcParams.update(params)
		
		
		fig2, ax2 = plt.subplots()
		import matplotlib.cm as cm
		colors = cm.rainbow_r(np.linspace(0, 1, 72))
		# colors = cm.gist_ncar(np.linspace(0, 1, 72))
		
		# ax2.plot(t_array, map_array)
		for i in range(72):
			ax2.plot(t_array, map_array[:,i], label=str(i+1), color=colors[i])
		ax2.legend(ncol=4)
		ax2.set_ylabel('Rate of AE bursts', fontsize=20)
		ax2.set_xlabel('Cumulative operating hours', fontsize=20)
		# ax2.set_xlabel('Minutes', fontsize=20)
		ax2.set_title('Distribution per teeth', fontsize=20)
		ax2.tick_params(axis='both', labelsize=16)
		plt.show()
		
	
	
	elif config['mode'] == 'plot_from_xls_teeth2':
		print('Select xls 1')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		MasterDict = {}
		count = 0
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			
			if count == 0:
				Names = []
				for key in mydict:
					if key != 'Sum' and key != 'N_Burst' and key != 'Avg':
						Names.append(key)

			mydict = mydict.to_dict(orient='list')
			
			for key, value in mydict.items():
				perro = len(mydict[key])
				newvalue = value				
				newvalue = list(np.nan_to_num(newvalue))
				if len(value) != len(newvalue):
					print('fatal error 79')
					sys.exit()
				mydict[key] = newvalue				
				loro = len(mydict[key])
				if perro != loro:
					print('fatal error 78')
					sys.exit()
			if count == 0:
				for name in Names:
					MasterDict[name] = mydict[name][:-1]
			else:
				for name in Names:
					MasterDict[name] += mydict[name][:-1]
			count += 1
		
		
		print('Select xls 2')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
		MasterDict2 = {}
		count = 0
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			
			if count == 0:
				Names = []
				for key in mydict:
					if key != 'Sum' and key != 'N_Burst' and key != 'Avg':
						Names.append(key)

			mydict = mydict.to_dict(orient='list')
			
			for key, value in mydict.items():
				perro = len(mydict[key])
				newvalue = value				
				newvalue = list(np.nan_to_num(newvalue))
				if len(value) != len(newvalue):
					print('fatal error 79')
					sys.exit()
				mydict[key] = newvalue				
				loro = len(mydict[key])
				if perro != loro:
					print('fatal error 78')
					sys.exit()
			if count == 0:
				for name in Names:
					MasterDict2[name] = mydict[name][:-1]
			else:
				for name in Names:
					MasterDict2[name] += mydict[name][:-1]
			count += 1
			

		
		
		
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		t = np.array([i for i in range(len(MasterDict['aar_T17']))])
		fig, ax = plt.subplots()
		
		MAP = []
		for key in MasterDict:
			if key != 'Sum' and key != 'N_Burst' and key != 'Avg':
				print(key)
				# MAP.append(list(np.log10(np.ones(len(MasterDict[key]))+np.array(MasterDict[key]))))
				# MAP.append(list(np.log10(np.ones(len(MasterDict[key])) + np.array(MasterDict[key]) / np.array(MasterDict[key]))))
				# MAP.append(MasterDict[key])
				# MAP.append(list(np.log10(np.ones(len(MasterDict2[key]))+np.nan_to_num(np.array(MasterDict2[key])*np.array(MasterDict[key])))))
				
		MAP = np.array(MAP)
		# sys.exit()
		fgato = np.arange(72)
		
		
		
		# print(len(t))
		# print(len(fgato))
		# print(len(MAP))
		# print(len(MAP[0]))
		
		vmax = None		
		caca = ax.pcolormesh(t, fgato, MAP, cmap='plasma', vmax=vmax)
		
		# caca.se
		# fig.colorbar(caca, ax=ax)
		# fig.colorbar(im, ax=ax0)
		# fig.colorbar(im, ax=ax0)
		
		# print(type(caca.cm))
		
		# caca.set_label('# of contacts')
		# map = []
		# vmax = max_cspectrum(perro)
		
		# print(vmax)
		# map.append(ax.pcolormesh(t, fgato, perro, cmap='plasma', vmax=vmax))
		cbar = fig.colorbar(caca, ax=ax)
		cbar.set_label(config['zlabel'], fontsize=13)
		
		n_files = len(Filepaths)
		ax.set_xticks([(i)*180 for i in range(n_files+1)])
		ax.set_xticklabels([i+1 for i in range(n_files+1)])
		
		
		ax.set_yticks([(i)*14.2 for i in range(6)])
		ax.set_yticklabels([(i)*72 for i in range(6)])
		
		
		ax.set_xlabel('N° Campaign', fontsize=13)
		# ax.set_ylabel('RMS Value [mV]', fontsize=12)
		ax.set_ylabel('Planet Carrier Angular Position', fontsize=13)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier (4 min mov. avg.)', fontsize=12)
		# ax.set_title('Sum 5 harmonics SB gear frequency - carrier', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		# ax.set_title('Sum 5 harmonics gear frequency', fontsize=12)
		ax.set_title(config['title'], fontsize=12)
		ax.tick_params(axis='both', labelsize=12)
		
		# plt.savefig(config['title'] + '_' + config['feature'] + '.png')
		plt.show()
		# print(Feature)


	
	elif config['mode'] == '2plot_from_xls_2':
		print('Select xls 1')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select xls 2----')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		

		
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		Feature = list(np.nan_to_num(Feature))
		
		
		Feature2 = []		
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature2 += mydict[config['feature2']]
		Feature2 = list(np.nan_to_num(Feature2))
		
		
		
		if config['interp'] == 'ON':
			n_f = len(Feature)
			n_f2 = len(Feature2)
			if n_f2 > n_f:
				xold = np.linspace(0., 1., n_f)
				xnew = np.linspace(0., 1., n_f2)				
				Feature = list(np.interp(x=xnew, xp=xold, fp=np.array(Feature)))
			elif n_f > n_f2:
				xold = np.linspace(0., 1., n_f2)
				xnew = np.linspace(0., 1., n_f)				
				Feature2 = list(np.interp(x=xnew, xp=xold, fp=np.array(Feature2)))
			
			
			
		
		
		n_2 = len(Feature2)
		
		
		
		
		plt.plot(Feature2, Feature)
		plt.show()
		
		
		t = [i*5./60/60 for i in range(len(Feature))]
		t2 = [i*5./60/60 for i in range(len(Feature2))]
		
		fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True)
		ax[0].plot(t, Feature, color=config['color'])
		ax[0].set_xlabel('Time [h]', fontsize=12)
		ax[0].set_ylabel(config['feature'], fontsize=12)
		ax[0].set_title(config['title'], fontsize=12)
		ax[0].tick_params(axis='both', labelsize=11)
		
		ax[1].plot(t2, Feature2, color='red')
		ax[1].set_xlabel('Time [h]', fontsize=12)
		ax[1].set_ylabel(config['feature2'], fontsize=12)
		ax[1].set_title(config['title'], fontsize=12)
		ax[1].tick_params(axis='both', labelsize=11)
		
	
		plt.show()
	
	elif config['mode'] == '2plot_from_xls_norm':
		print('Select xls 1')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select xls 2 ----')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		

		
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]
		# Feature = list(np.nan_to_num(Feature))
		
		# for i in range(len(Atributo)):
			# count = 0
			# while np.isnan(Atributo[i]) == True:
				# count += 1
				# Atributo[i] = Atributo[i + count]			
		# Atributo = list(np.nan_to_num(Atributo))
		Feature = np.array(Feature)
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		
		
		Feature2 = []		
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature2 += mydict[config['feature2']]
		# Feature2 = list(np.nan_to_num(Feature2))
		Feature2 = np.array(Feature2)
		Feature2 = movil_avg(Feature2, config['n_mov_avg'])
		# n_f = len(Feature)
		# n_f2 = len(Feature2)
		# xold = np.linspace(0., 1., n_f2)
		# xnew = np.linspace(0., 1., n_f)				
		# Feature2 = list(np.interp(x=xnew, xp=xold, fp=np.array(Feature2)))

		# FeatureX = [Feature[i]*np.exp(-0.21*Feature2[i]) for i in range(n_f)]
		FeatureX = [Feature[i]/Feature2[i] for i in range(len(Feature))]
		
		t = [i for i in range(len(FeatureX))]
		
		
		fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True)
		ax.plot(t, FeatureX, color=config['color'])
		ax.set_xlabel('Time [h]', fontsize=12)
		ax.set_ylabel(config['feature'], fontsize=12)
		ax.set_title(config['title'], fontsize=12)
		ax.tick_params(axis='both', labelsize=11)
		

		
		
		
		
		plt.show()
		# print(Feature)
	
	
	elif config['mode'] == 'sort_teeth':
	
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
		
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index.values)
			
			
			mydict_mod = {}
			count = 0
			numbers = [str(i) for i in range(72)]
			letters = ['aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aaj', 'aak', 'aal', 'aam', 'aan', 'aao', 'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aav', 'aaw', 'aax', 'aay', 'aaz', 'aba', 'abb', 'abc', 'abd', 'abe', 'abf', 'abg', 'abh', 'abi', 'abj', 'abk', 'abl', 'abm', 'abn', 'abo', 'abp', 'abq', 'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acb', 'acc', 'acd', 'ace', 'acf', 'acg', 'ach', 'aci', 'acj', 'ack', 'acl', 'acm', 'acn', 'aco', 'acp', 'acq', 'acr', 'acs', 'act']
			dog = 0
			for key, values in mydict.items():			
				if count+33 >= len(numbers):
					dog = 72			
				mydict_mod[letters[count+33-dog] + '_T' + numbers[count+33-dog]] = values
				count += 1
		

		
			writer = pd.ExcelWriter(basename(filepath)[:-5] + '_Corr' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Teeth_Burst_Corr')
			writer.close()
	
	elif config['mode'] == 'sort_campaign':
	
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
		
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index.values)
			
			
			
			# writer = pd.ExcelWriter(basename(filepath)[-6:-5] + '_' + basename(filepath)[:-5] + '.xlsx')
			# writer = pd.ExcelWriter(basename(filepath)[-11:-10] + '_' + basename(filepath)[:-5] + '.xlsx')	
			
			# writer = pd.ExcelWriter(basename(filepath)[11] + '_' + basename(filepath)[:-5] + '.xlsx')

			writer = pd.ExcelWriter(basename(filepath)[19] + '_' + basename(filepath)[:-5] + '.xlsx')				

			
			DataFr = pd.DataFrame(data=mydict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')
			writer.close()
	
	elif config['mode'] == 'multiplicate_table':
	
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
		
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index.values)
			
			mydict = mydict.to_dict(orient='list')	
			
			mydict2 = {}
			for key in mydict.keys():
				mydict2[key] = np.array(mydict[key])/1000.
			
			# writer = pd.ExcelWriter(basename(filepath)[-6:-5] + '_' + basename(filepath)[:-5] + '.xlsx')
			# writer = pd.ExcelWriter(basename(filepath)[-11:-10] + '_' + basename(filepath)[:-5] + '.xlsx')	
			
			# writer = pd.ExcelWriter(basename(filepath)[11] + '_' + basename(filepath)[:-5] + '.xlsx')

			writer = pd.ExcelWriter(basename(filepath))				

			
			DataFr = pd.DataFrame(data=mydict2, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Wfm')
			writer.close()
	
	
	elif config['mode'] == 'sort_long_features':
	
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		
		# for filepath in Filepaths:				
			# mydict = pd.read_excel(filepath)
			# rownames = list(mydict.index.values)
		
		
		mydictloc = pd.read_excel(Filepaths[0])
		# rownames = list(mydict.index.values)

		mydictloc = mydictloc.to_dict(orient='list')
		feature_array = mydictloc.keys()
		
		mydict_mod = {}		
		for name_feature in feature_array:
			Feature = []
			Rownames = []
			for filepath in Filepaths:
				
				mydict = pd.read_excel(filepath)
				rownames = list(mydict.index.values)

				mydict = mydict.to_dict(orient='list')
				
				# Feature += mydict[name_feature][:-2]
				# Rownames += rownames[:-2]
				
				# Feature += mydict[name_feature][:-1]
				# Rownames += rownames[:-1]
				
				Feature += mydict[name_feature]
				Rownames += rownames
				
				# Feature = list(np.nan_to_num(Feature))
				# Feature = movil_avg(Feature, config['n_mov_avg'])
			
			mydict_mod[name_feature] = Feature
				
				
				
				# writer = pd.ExcelWriter(basename(filepath)[-6:-5] + '_' + basename(filepath)[:-5] + '.xlsx')

		writer = pd.ExcelWriter('MASTER_Features.xlsx')			
		DataFr = pd.DataFrame(data=mydict_mod, index=Rownames)
		DataFr.to_excel(writer, sheet_name='Features')
		writer.close()
	
	elif config['mode'] == 'add_column_sum':
	
		# print('Add prom to n° burst too?')
		# flag = input('Add prom to n° burst too? y/n... ')
		flag = 'y'

	
		print('Select xls feature')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		if flag == 'y':
			print('Now... Select xls N burst....')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths_Burst = filedialog.askopenfilenames()
			root.destroy()
		else:
			Filepaths_Burst = Filepaths
		
		
		for filepath, filepath_burst in zip(Filepaths, Filepaths_Burst):
		
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index.values)
			
			Sums = []
			
			
			
			for i in range(mydict.shape[0]):
				row = mydict.loc[rownames[i], :]
				row = row.tolist()

				
				Sums.append(np.nansum(row))
			
			mydict_mod = mydict
			
			mydict_mod['Sum'] = Sums
			
			if flag == 'y':
				mydict_burst = pd.read_excel(filepath_burst)
				N_burst = mydict_burst['Sum'].tolist()
				
				mydict_mod['N_Burst'] = N_burst
				
				Avgs = [Sums[k]/N_burst[k] for k in range(len(N_burst))]
				mydict_mod['Avg'] = Avgs
				
			
			
			
			
		

		
			writer = pd.ExcelWriter(basename(filepath)[:-5] + '_' + '.xlsx')			
			DataFr = pd.DataFrame(data=mydict_mod, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')
			writer.close()
	
	elif config['mode'] == 'result_regre':

		# print('Select xls results regres')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()
		# root.destroy()
		
		# print('Select xls results configs')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# FilepathsC = filedialog.askopenfilenames()
		# root.destroy()

		# for filepath, filepathC in zip(Filepaths, FilepathsC):
		
			# mydict = pd.read_excel(filepath)
			# mydictC = pd.read_excel(filepathC)
			
			# rownames = list(mydict.index.values)
			# rownamesC = list(mydictC.index.values)
			
			# mydict = mydict.to_dict(orient='list')
			# mydictC = mydictC.to_dict(orient='list')
			
			# train = str(mydictC['train'][0])

			# writer = pd.ExcelWriter('train_' + train + '_' + basename(filepath)[:-5] + '.xlsx')			
			# DataFr = pd.DataFrame(data=mydict, index=rownames)		
			# DataFr.to_excel(writer, sheet_name='Regre')
			# writer.close()
			
			# writerC = pd.ExcelWriter('train_' + train + '_' + basename(filepathC)[:-5] + '.xlsx')			
			# DataFr = pd.DataFrame(data=mydictC, index=rownamesC)		
			# DataFr.to_excel(writerC, sheet_name='Regre')
			# writer.close()
		
		
		print('Select xls results regres')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Error = []
		Error_Final = []
		Numbers = []
		Layers = []

		for filepath in Filepaths:
		
			mydict = pd.read_excel(filepath)
			
			rownames = list(mydict.index.values)
			
			mydict = mydict.to_dict(orient='list')
			
			
			mydict['Error_Final']
			index = basename(filepath).find('regre_')
			num = basename(filepath)[:-5][index+6:]
			List_Num = [num for i in range(len(rownames))]
			
			# Error.append(mydict['Error'])
			# Error_Final.append(mydict['Error_Final'])
			# Numbers.append(List_Num)
			# Layers.append(rownames)
			
			Error += mydict['Error']
			Error_Final += mydict['Error_Final']
			Numbers += List_Num
			Layers += rownames
		
		# print(Error)
		# print(Error_Final)
		# print(Numbers)
		# print(Layers)
		Error = np.array(Error)
		Error_Final = np.array(Error_Final)
		Numbers = np.array(Numbers)
		Layers = np.array(Layers)
		
		print('++++Error Min')
		argmin = np.argmin(Error)
		print('error = ', Error[argmin])
		print('error_final = ',Error_Final[argmin])
		print('number = ',Numbers[argmin])
		print('layers = ',Layers[argmin])
		
		
		print('++++Error final Min')
		argmin_Final = np.argmin(Error_Final)
		print('error = ', Error[argmin_Final])
		print('error_final = ',Error_Final[argmin_Final])
		print('number = ',Numbers[argmin_Final])
		print('layers = ',Layers[argmin_Final])
		
		result_dict = {'Min_Error':[Error[argmin], Error_Final[argmin], Numbers[argmin], Layers[argmin]], 'Min_Error_Final':[Error[argmin_Final], Error_Final[argmin_Final], Numbers[argmin_Final], Layers[argmin_Final]]}
		names = ['Error', 'Error_Final', 'Number', 'Layers']
		
		writer = pd.ExcelWriter('Best_Results_train_02' + '.xlsx')			
		DataFr = pd.DataFrame(data=result_dict, index=names)		
		DataFr.to_excel(writer, sheet_name='Regre')
		writer.close()
			

	
	elif config['mode'] == 'generate_prom':
	
		# print('Add prom to n° burst too?')
	
		print('Select xls feature')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		print('Now... Select xls N burst....')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Burst = filedialog.askopenfilenames()
		root.destroy()

				
		for filepath, filepath_burst in zip(Filepaths, Filepaths_Burst):
		
			mydict = pd.read_excel(filepath)
			rownames = list(mydict.index.values)
			colnames = list(mydict.columns)
			
			mydict_burst = pd.read_excel(filepath_burst)

			newdict = {}
			
			for i in range(mydict.shape[1]):
				col = mydict.loc[:, colnames[i]]
				col = col.tolist()
				
				col_burst = mydict_burst.loc[:, colnames[i]]
				col_burst = col_burst.tolist()
				
				vec = []
				for k in range(len(col)):
					if col_burst[k] != 0:
						vec.append(col[k]/col_burst[k])
					else:
						vec.append(0)
				
				newdict[colnames[i]] = vec
				
				# Sums.append(np.nansum(row))
			
			
			
		

		
			writer = pd.ExcelWriter(basename(filepath)[:-5] + '_Prom' + '.xlsx')			
			DataFr = pd.DataFrame(data=newdict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')
			writer.close()
	
	elif config['mode'] == 'generate_ra':
	
		# print('Add prom to n° burst too?')
	
		print('Select xls MAX')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_max = filedialog.askopenfilenames()
		root.destroy()
		
		
		print('Now... Select xls RISE....')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_rise = filedialog.askopenfilenames()
		root.destroy()

	
		for filepath_max, filepath_rise in zip(Filepaths_max, Filepaths_rise):
		
			mydict = pd.read_excel(filepath_max)
			rownames = list(mydict.index.values)
			colnames = list(mydict.columns)
			
			mydict_burst = pd.read_excel(filepath_rise)

			newdict = {}
			
			for i in range(mydict.shape[1]):
				col = mydict.loc[:, colnames[i]]
				col = col.tolist()
				
				col_burst = mydict_burst.loc[:, colnames[i]]
				col_burst = col_burst.tolist()
				
				vec = []
				for k in range(len(col)):
					if col_burst[k] != 0:
						vec.append(col[k]/col_burst[k])
					else:
						vec.append(0)
				
				newdict[colnames[i]] = vec
				
				# Sums.append(np.nansum(row))
			
			
			
		

		
			writer = pd.ExcelWriter(basename(filepath_max)[0:12] + '_SumRA' + '.xlsx')			
			DataFr = pd.DataFrame(data=newdict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')
			writer.close()
	
	elif config['mode'] == 'generate_fcount':
	
		# print('Add prom to n° burst too?')
	
		print('Select xls COUNTS')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_max = filedialog.askopenfilenames()
		root.destroy()
		
		
		print('Now... Select xls DURA....')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_rise = filedialog.askopenfilenames()
		root.destroy()

	
		for filepath_max, filepath_rise in zip(Filepaths_max, Filepaths_rise):
		
			mydict = pd.read_excel(filepath_max)
			rownames = list(mydict.index.values)
			colnames = list(mydict.columns)
			
			mydict_burst = pd.read_excel(filepath_rise)

			newdict = {}
			
			for i in range(mydict.shape[1]):
				col = mydict.loc[:, colnames[i]]
				col = col.tolist()
				
				col_burst = mydict_burst.loc[:, colnames[i]]
				col_burst = col_burst.tolist()
				
				vec = []
				for k in range(len(col)):
					if col_burst[k] != 0:
						vec.append(col[k]/col_burst[k])
					else:
						vec.append(0)
				
				newdict[colnames[i]] = vec
				
				# Sums.append(np.nansum(row))
			
			
			
		

		
			writer = pd.ExcelWriter(basename(filepath_max)[0:12] + '_SumFcount' + '.xlsx')			
			DataFr = pd.DataFrame(data=newdict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Teeth_Burst')
			writer.close()
		

	
	
	
	elif config['mode'] == 'interp_temp':
		print('Select xls 1 train')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select xls 2 TEMPERATURE train----')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
			
		for filepath, filepath2 in zip(Filepaths, Filepaths2):
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			rownames = list(mydict.index.values)
			
			mydict = mydict.to_dict(orient='list')
			Feature = list(mydict[config['feature']])
		# Feature = list(np.nan_to_num(Feature))
				
			
			mydict2 = pd.read_excel(filepath2, sheetname=config['sheet'])
			mydict2 = mydict2.to_dict(orient='list')
			Feature2 = list(mydict2[config['feature2']])
		# Feature2 = list(np.nan_to_num(Feature2))
		
		
			n_f = len(Feature)
			n_f2 = len(Feature2)
			xold = np.linspace(0., 1., n_f2)
			xnew = np.linspace(0., 1., n_f)				
			Feature2 = list(np.interp(x=xnew, xp=xold, fp=np.array(Feature2)))

		
			mydict2 = {}
			mydict2[config['feature2']] = Feature2
			
		
		
		
		# writer = pd.ExcelWriter(basename(filepath)[-6:-5] + '_' + basename(filepath)[:-5] + '.xlsx')
		# writer = pd.ExcelWriter(basename(filepath)[-11:-10] + '_' + basename(filepath)[:-5] + '.xlsx')	
		
		# writer = pd.ExcelWriter(basename(filepath)[11] + '_' + basename(filepath)[:-5] + '.xlsx')

			writer = pd.ExcelWriter(basename(filepath2) + '_int' + '.xlsx')				

			
			DataFr = pd.DataFrame(data=mydict2, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Temperature')
			writer.close()
		
		# Train = Feature[0:int(config['train']*len(Feature))]
		# x_Train = np.arange(float(len(Train)))				
		
		# x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	
	
	
	elif config['mode'] == 'feature_temp':
		print('Select xls ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		
			
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			rownames = list(mydict.index.values)
			
			name_feat = list(mydict.columns)[0]
			
			mydict = mydict.to_dict(orient='list')
			Feature = list(mydict[name_feat])


		
			mydict2 = {}
			mydict2['Temp'] = Feature
			
		
		
		
		# writer = pd.ExcelWriter(basename(filepath)[-6:-5] + '_' + basename(filepath)[:-5] + '.xlsx')
		# writer = pd.ExcelWriter(basename(filepath)[-11:-10] + '_' + basename(filepath)[:-5] + '.xlsx')	
		
		# writer = pd.ExcelWriter(basename(filepath)[11] + '_' + basename(filepath)[:-5] + '.xlsx')

			writer = pd.ExcelWriter(basename(filepath)[:-14] + '_int' + '.xlsx')				

			
			DataFr = pd.DataFrame(data=mydict2, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Temperature')
			writer.close()
		
		# Train = Feature[0:int(config['train']*len(Feature))]
		# x_Train = np.arange(float(len(Train)))				
		
		# x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	

		
	
	
	
	elif config['mode'] == 'plot_from_xls_norm':
		print('Select xls 1 feature 1')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select xls 2 feature 2 ----')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()	

		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']]		
		for i in range(len(Feature)):
			count = 0
			while np.isnan(Feature[i]) == True:
				count += 1
				Feature[i] = Feature[i + count]		
		Feature = movil_avg(Feature, config['n_mov_avg'])


		

		Feature2 = []		
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			Feature2 += mydict[config['feature2']]		
		for i in range(len(Feature2)):
			count = 0
			while np.isnan(Feature2[i]) == True:
				count += 1
				Feature2[i] = Feature2[i + count]		
		Feature2 = movil_avg(Feature2, config['n_mov_avg'])

		print(len(Feature))
		print(len(Feature2))
		
		# data = {'x':5*t/3600., 'y':trend}
		# style = {'ylabel':'AE bursts [-]', 'xlabel':'Time [h]', 'customxlabels':None, 'legend':False, 'title':'Fixed counting', 'ylim':[0, 125], 'xlim':None}
		# plot1_thesis(data, style)
		

		
		fig, ax = plt.subplots()
		
		t = np.array([5*i/3600. for i in range(len(Feature))])
		
		lns1 = ax.plot(t, Feature, color='blue', label='Relative counting')		
		ax.set_ylabel('Rate AE bursts', color='b', fontsize=13)
		ax.set_xlabel('Accumulated operating hours', fontsize=13)
		ax.tick_params('y', colors='b')
		ax.tick_params(axis='both', labelsize=12)
		
		
		ax2 = ax.twinx()
		lns2 = ax2.plot(t, Feature2, color='red', label='Fixed counting')
		ax2.set_ylabel('Rate AE bursts', color='r', fontsize=13)
		ax2.tick_params('y', colors='r')	
		ax2.tick_params(axis='both', labelsize=12)
		
		
		lns = lns1 + lns2
		labs = [l.get_label() for l in lns]
		ax.legend(lns, labs, loc=0, fontsize=11.5)
		
		plt.show()
		
		
		
		
		# n_f = len(Feature)
		# n_f2 = len(Feature2)
		# xold = np.linspace(0., 1., n_f2)
		# xnew = np.linspace(0., 1., n_f)				
		# Feature2 = np.interp(x=xnew, xp=xold, fp=np.array(Feature2))

		
	elif config['mode'] == 'plot_from_xls_norm3':
		print('Select files')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')			
			signal = np.array(mydict[config['feature']])
			Feature += list(signal)
		Feature = np.array(Feature)

		Feature2 = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')			
			signal = np.array(mydict[config['feature2']])
			Feature2 += list(signal)		
		Feature2 = np.array(Feature2)
		
		Feature3 = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')			
			signal = np.array(mydict[config['feature3']])
			Feature3 += list(signal)		
		Feature3 = np.array(Feature2)
		
		
		
		
		Feature4 = Feature*Feature2**0.1
		fig, ax = plt.subplots(ncols=1, nrows=4)
		ax[0].plot(Feature)
		ax[1].plot(Feature2)
		# Feature3 = Feature*(Feature2)**0.15
		Feature3 = Feature*Feature2
		ax[2].plot(Feature3)
		ax[3].plot(Feature4, 'r')
		plt.show()
		sys.exit()
		
	
	
	elif config['mode'] == 'norm_with_minsq':
		print('Select xls 1 train')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select xls 2 TEMPERATURE train----')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2 = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			
			signal = np.array(mydict[config['feature']])
			signal = butter_lowpass(x=signal, order=3, freq=0.01, fs=1.)			
			Feature += list(signal)
			
			
		# Feature = np.nan_to_num(Feature)
		# Feature = movil_avg(Feature, config['n_mov_avg'])		
		# Feature = butter_lowpass(x=Feature, order=3, freq=0.01, fs=1.)
		
		
		
				
		Feature2 = []		
		for filepath in Filepaths2:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			
			signal = np.array(mydict[config['feature2']])
			signal = butter_lowpass(x=signal, order=3, freq=0.01, fs=1.)			
			Feature2 += list(signal)
			
		# Feature2 = np.nan_to_num(Feature2)
		# Feature2 = movil_avg(Feature2, config['n_mov_avg'])		
		# Feature2 = butter_lowpass(x=Feature2, order=3, freq=0.01, fs=1.)
		
		
		n_f = len(Feature)
		n_f2 = len(Feature2)
		xold = np.linspace(0., 1., n_f2)
		xnew = np.linspace(0., 1., n_f)				
		Feature2 = np.interp(x=xnew, xp=xold, fp=np.array(Feature2))

		
		n_f = len(Feature)
		n_f2 = len(Feature2)
		xold = np.linspace(np.min(Feature2), np.max(Feature2), n_f)
		# FeatureT = np.interp(x=np.array(Feature2), xp=xold, fp=np.array(Feature))
		FeatureT = np.interp(x=xold, xp=np.array(Feature2), fp=np.array(Feature))
		
		
		
		A = np.vstack([Feature2, np.ones(len(Feature2))]).T
		m, c = np.linalg.lstsq(A, FeatureT)[0]
		
		# plt.plot(Feature2, FeatureT, 'o', label='Original data', markersize=10)
		# plt.plot(Feature2, m*Feature2 + c, 'r', label='Fitted line')
		# plt.legend()
		# plt.show()
		
		
		
		# #############
		
		
		
		
		# fig, ax = plt.subplots(nrows=4, ncols=1)
		
		# ax[0].plot(Feature, 'b')
		# ax[0].set_title('feat_t')
		
		# ax[1].plot(Feature2, 'r')
		# ax[1].set_title('temp_t')
		
		# ax[2].plot(FeatureT, 'k')
		# ax[2].set_title('feat_temp')
		
		# ax[3].plot(m*Feature2 + c, 'g')
		# ax[3].set_title('minsq')
		# plt.show()
		

		# sys.exit()
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		
		
		
		
		
		print('Select xls 1 test.............')
		root = Tk()
		root.withdraw()
		root.update()
		FilepathsX = filedialog.askopenfilenames()
		root.destroy()
		
		print('Select xls 2 TEMPERATURE test..............')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths2X = filedialog.askopenfilenames()
		root.destroy()
		
		FeatureX = []		
		for filepath in FilepathsX:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			signal = np.array(mydict[config['feature']])
			signal = butter_lowpass(x=signal, order=3, freq=0.01, fs=1.)			
			FeatureX += list(signal)
			
			
		# FeatureX = np.nan_to_num(FeatureX)
		# FeatureX = movil_avg(FeatureX, config['n_mov_avg'])		
		# FeatureX = butter_lowpass(x=FeatureX, order=3, freq=0.01, fs=1.)
		
		
				
		Feature2X = []
		count = 0		
		for filepath in Filepaths2X:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])
			mydict = mydict.to_dict(orient='list')
			if count == 0:
				signal = np.array(mydict[config['feature3']])
				signal = butter_lowpass(x=signal, order=3, freq=0.01, fs=1.)			
				Feature2X += list(signal)
			else:
				signal = np.array(mydict[config['feature4']])
				signal = butter_lowpass(x=signal, order=3, freq=0.01, fs=1.)			
				Feature2X += list(signal)
				
		# Feature2X = np.nan_to_num(Feature2X)
		# Feature2X = movil_avg(Feature2X, config['n_mov_avg'])		
		# Feature2X = butter_lowpass(x=Feature2X, order=3, freq=0.01, fs=1.)
		
		
		
		n_f = len(FeatureX)
		n_f2 = len(Feature2X)
		xold = np.linspace(0., 1., n_f2)
		xnew = np.linspace(0., 1., n_f)				
		Feature2X = np.interp(x=xnew, xp=xold, fp=np.array(Feature2X))
		
		
		FT = m*Feature2X + c
		
		
		fig, ax = plt.subplots(nrows=4, ncols=1)
		
		ax[0].plot(FeatureX, 'b')
		ax[0].set_title('Feat original')
		
		ax[1].plot(Feature2X, 'r')
		ax[1].set_title('Temperature ')
		
		ax[2].plot(FT , 'k')
		# ax[2].plot(Feature2X, 'r')
		ax[2].set_title('Aporte T')
		
		ax[3].plot(FeatureX - FT , 'k')
		# ax[2].plot(Feature2X, 'r')
		ax[3].set_title('Diff.')
		
		
		# plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		plt.show()
	
	
	
	
	
	
	
	
	elif config['mode'] == 'predict_from_xls_kal':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		Feature = []		
		for filepath in Filepaths:
			mydict = pd.read_excel(filepath, sheetname=config['sheet'])

			mydict = mydict.to_dict(orient='list')
			Feature += mydict[config['feature']][:-2]
		Feature = list(np.nan_to_num(Feature))
		
		Feature = movil_avg(Feature, config['n_mov_avg'])
		
		Feature = np.array(Feature)
		x_Feature = np.arange(len(Feature))
		
		Train = Feature[0:int(config['train']*len(Feature))]
		x_Train = np.arange(float(len(Train)))				
		
		x_Predict = np.linspace(len(Train), len(Feature), num=len(Feature) - len(Train), endpoint=False)
		
		from pykalman import KalmanFilter
		# scaler = StandardScaler()
		# scaler = RobustScaler()
		# scaler.fit(Train)
		# Train = scaler.transform(Train)	

		kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]), transition_covariance=0.01 * np.eye(2))

		# You can use the Kalman Filter immediately without fitting, but its estimates
		# may not be as good as if you fit first.
		kf = kf.em(Train)
		states_pred = kf.smooth(x_Predict)[0]
		
		
		n_pre = int(config['n_pre']*len(Train))
		m_post = 1
		n_ex = len(Train) - n_pre - m_post
		print('+++++++++++++Info: Input points n = ', n_pre)
		print('+++++++++++++Info: Output points m = ', m_post)
		print('+++++++++++++Info: Training examples = ', n_ex)
		a = input('enter to continue...')
		T_Inputs = []
		T_Outputs = []
		for k in range(n_ex + 1):

			T_Inputs.append(Train[k : k + n_pre])
			T_Outputs.append(Train[k + n_pre : k + n_pre + m_post])
		
		clf.fit(T_Inputs, T_Outputs)
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
		Predict = []
		It_Train = list(Train)

		for k in range(len(x_Predict) + m_post - 1):
			P_Input = It_Train[n_ex + k + 1 : n_ex + n_pre + k + 1]

			P_Output = clf.predict(P_Input)
			P_Output = P_Output[0]
			
			
			Predict.append(P_Output)
			It_Train.append(P_Output)

		# Predict = Predict[:-(m_post-1)]
	
		plt.plot(x_Feature, Feature, 'b', x_Predict, Predict, 'r', x_Train, Train, 'k')
		plt.show()
	
	elif config['mode'] == 'sum_from_xls':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
			newdict = {}
			olddict = pd.read_excel(filepath)
			rownames = list(olddict.index.values)
			n_files = len(rownames)
			olddict = olddict.to_dict(orient='list')
			# for key in olddict:
				# print(key)
			# print(olddict)
			# sys.exit()
			
			# newdict['f_g'] = olddict['312.0'][:-2]
			
			newdict['10h_f_f_r'] = np.zeros(n_files)
			#1300
			freq_list = ['4.33', '8.66', '12.99', '17.32', '21.65', '26.0', '30.33', '34.66', '38.85', '43.33']
			
			# #1000
			# freq_list = ['3.33', '6.66', '9.99', '13.32', '16.65', '19.98', '23.32', '26.65', '29.98', '33.21']
			
			# #700
			# freq_list = ['2.33', '4.66', '6.99', '9.33', '11.66', '13.99', '16.32', '18.65', '20.98', '23.32']
			
			
			# #400
			# freq_list = ['1.33', '2.66', '4.0', '5.33', '6.66', '7.99', '9.33', '10.66', '11.99', '13.32']
			
			for freq in freq_list:
				newdict['10h_f_f_r'] += np.array(olddict[freq])
			newdict['10h_f_f_r'] = list(newdict['10h_f_f_r'])
			
			
			newdict['5h_f_g'] = np.zeros(n_files)
			freq_list = ['312.0', '624.0', '936.0', '1248.0', '1560.0']
			for freq in freq_list:
				newdict['5h_f_g'] += np.array(olddict[freq])
			newdict['5h_f_g'] = list(newdict['5h_f_g'])
			
			
			newdict['5h_1sbi_f_g_c'] = np.zeros(n_files)
			freq_list = ['307.67', '619.67', '931.67', '1243.67', '1555.67']
			for freq in freq_list:
				newdict['5h_1sbi_f_g_c'] += np.array(olddict[freq])
			newdict['5h_1sbi_f_g_c'] = list(newdict['5h_1sbi_f_g_c'])
			
			newdict['5h_1sbd_f_g_c'] = np.zeros(n_files)
			freq_list = ['316.33', '628.33', '940.33', '1252.33', '1564.33']
			for freq in freq_list:
				newdict['5h_1sbd_f_g_c'] += np.array(olddict[freq])
			newdict['5h_1sbd_f_g_c'] = list(newdict['5h_1sbd_f_g_c'])
			
			
			newdict['5h_2sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [303.34, 615.34, 927.34, 1239.34, 1551.34]
			for freq in freq_list:
				newdict['5h_2sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_2sbi_f_g_c'] = list(newdict['5h_2sbi_f_g_c'])
			
			newdict['5h_2sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [320.66, 632.66, 944.66, 1256.66, 1568.66]
			for freq in freq_list:
				newdict['5h_2sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_2sbd_f_g_c'] = list(newdict['5h_2sbd_f_g_c'])
			
			newdict['5h_3sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [299.01, 611.01, 923.01, 1235.01, 1547.01]
			for freq in freq_list:
				newdict['5h_3sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_3sbi_f_g_c'] = list(newdict['5h_3sbi_f_g_c'])
			
			newdict['5h_3sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [324.99, 636.99, 948.99, 1260.99, 1572.99]
			for freq in freq_list:
				newdict['5h_3sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_3sbd_f_g_c'] = list(newdict['5h_3sbd_f_g_c'])
			
			newdict['5h_4sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [294.68, 606.68, 918.68, 1230.68, 1542.68]
			for freq in freq_list:
				newdict['5h_4sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_4sbi_f_g_c'] = list(newdict['5h_4sbi_f_g_c'])
			
			newdict['5h_4sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [329.32, 641.32, 953.32, 1265.32, 1577.32]
			for freq in freq_list:
				newdict['5h_4sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_4sbd_f_g_c'] = list(newdict['5h_4sbd_f_g_c'])
			
			newdict['5h_5sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [290.35, 602.35, 914.35, 1226.35, 1538.35]
			for freq in freq_list:
				newdict['5h_5sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_5sbi_f_g_c'] = list(newdict['5h_5sbi_f_g_c'])
			
			newdict['5h_5sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [333.65, 645.65, 957.65, 1269.35, 1581.65]
			for freq in freq_list:
				newdict['5h_5sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_5sbd_f_g_c'] = list(newdict['5h_5sbd_f_g_c'])
			
			newdict['5h_6sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [286.02, 598.02, 910.02, 1222.02, 1534.02]
			for freq in freq_list:
				newdict['5h_6sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_6sbi_f_g_c'] = list(newdict['5h_6sbi_f_g_c'])
			
			newdict['5h_6sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [337.98, 649.98, 961.98, 1273.98, 1585.98]
			for freq in freq_list:
				newdict['5h_6sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_6sbd_f_g_c'] = list(newdict['5h_6sbd_f_g_c'])
			
			
			newdict['5h_sbd_f_g_c'] = list(np.array(newdict['5h_6sbd_f_g_c']) + np.array(newdict['5h_5sbd_f_g_c']) + np.array(newdict['5h_4sbd_f_g_c']) + np.array(newdict['5h_3sbd_f_g_c']) + np.array(newdict['5h_2sbd_f_g_c']) + np.array(newdict['5h_1sbd_f_g_c']))
			
			newdict['5h_sbi_f_g_c'] = list(np.array(newdict['5h_6sbi_f_g_c']) + np.array(newdict['5h_5sbi_f_g_c']) + np.array(newdict['5h_4sbi_f_g_c']) + np.array(newdict['5h_3sbi_f_g_c']) + np.array(newdict['5h_2sbi_f_g_c']) + np.array(newdict['5h_1sbi_f_g_c']))
			
			newdict['5h_sb_f_g_c'] = list(np.array(newdict['5h_sbd_f_g_c']) + np.array(newdict['5h_sbi_f_g_c']))

			
			newdict['5h_sb_f_g_c_1'] = list( np.array(newdict['5h_1sbd_f_g_c']) + np.array(newdict['5h_1sbi_f_g_c']))
			newdict['5h_sb_f_g_c_2'] = list( np.array(newdict['5h_2sbd_f_g_c']) + np.array(newdict['5h_2sbi_f_g_c']))
			newdict['5h_sb_f_g_c_3'] = list( np.array(newdict['5h_3sbd_f_g_c']) + np.array(newdict['5h_3sbi_f_g_c']))
			newdict['5h_sb_f_g_c_4'] = list( np.array(newdict['5h_4sbd_f_g_c']) + np.array(newdict['5h_4sbi_f_g_c']))
			newdict['5h_sb_f_g_c_5'] = list( np.array(newdict['5h_5sbd_f_g_c']) + np.array(newdict['5h_5sbi_f_g_c']))
			newdict['5h_sb_f_g_c_6'] = list( np.array(newdict['5h_6sbd_f_g_c']) + np.array(newdict['5h_6sbi_f_g_c']))
			
			
			writer = pd.ExcelWriter('Meaning_' + basename(filepath))			
			DataFr = pd.DataFrame(data=newdict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Env_Fft')
			writer.close()
	
	elif config['mode'] == 'sum_from_xls_more':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
			newdict = {}
			olddict = pd.read_excel(filepath)
			rownames = list(olddict.index.values)
			n_files = len(rownames)
			olddict = olddict.to_dict(orient='list')

			
			newdict['23h_f_f_r'] = np.zeros(n_files)
			#1300			
			frequencies = [4.33, 8.66, 13.00, 17.33, 21.67, 26.00, 30.33, 34.67, 39.00, 43.33, 47.67, 52.00, 56.33, 60.67, 65.00, 69.33, 73.67, 78.00, 82.33, 86.67, 91.00, 95.33, 99.67]
			freq_list = [str(f) for f in frequencies]
			
			
			for freq in freq_list:
				newdict['23h_f_f_r'] += np.array(olddict[freq])
			newdict['23h_f_f_r'] = list(newdict['23h_f_f_r'])
			
			
			newdict['5h_f_g'] = np.zeros(n_files)
			frequencies = [312., 624., 936., 1248., 1560.]
			freq_list = [str(f) for f in frequencies]
			for freq in freq_list:
				newdict['5h_f_g'] += np.array(olddict[freq])
			newdict['5h_f_g'] = list(newdict['5h_f_g'])
			
			
			
			newdict['1h_20sb_f_g_c'] = np.zeros(n_files)			
			frequencies = [316.33, 320.67, 325.00, 329.33, 333.67, 338.00, 342.33, 346.67, 351.00, 355.33, 359.67, 364.00, 368.33, 372.67, 377.00, 381.33, 385.67, 390.00, 394.33, 398.67] + [225.33, 229.67, 234.00, 238.33, 242.67, 247.00, 251.33, 255.67, 260.00, 264.33, 268.67, 273.00, 277.33, 281.67, 286.00, 290.33, 294.67, 299.00, 303.33, 307.67]
			freq_list = [str(f) for f in frequencies]			
			for freq in freq_list:
				newdict['1h_20sb_f_g_c'] += np.array(olddict[freq])
			newdict['1h_20sb_f_g_c'] = list(newdict['1h_20sb_f_g_c'])
			
			newdict['2h_20sb_f_g_c'] = np.zeros(n_files)			
			frequencies = [628.33, 632.67, 637.00, 641.33, 645.67, 650.00, 654.33, 658.67, 663.00, 667.33, 671.67, 676.00, 680.33, 684.67, 689.00, 693.33, 697.67, 702.00, 706.33, 710.67] + [537.33, 541.67, 546.00, 550.33, 554.67, 559.00, 563.33, 567.67, 572.00, 576.33, 580.67, 585.00, 589.33, 593.67, 598.00, 602.33, 606.67, 611.00, 615.33, 619.67]
			freq_list = [str(f) for f in frequencies]			
			for freq in freq_list:
				newdict['2h_20sb_f_g_c'] += np.array(olddict[freq])
			newdict['2h_20sb_f_g_c'] = list(newdict['2h_20sb_f_g_c'])
			
			newdict['3h_20sb_f_g_c'] = np.zeros(n_files)			
			frequencies = [940.33, 944.67, 949.00, 953.33, 957.67, 962.00, 966.33, 970.67, 975.00, 979.33, 983.67, 988.00, 992.33, 996.67, 1001.00, 1005.33, 1009.67, 1014.00, 1018.33, 1022.67] + [849.33, 853.67, 858.00, 862.33, 866.67, 871.00, 875.33, 879.67, 884.00, 888.33, 892.67, 897.00, 901.33, 905.67, 910.00, 914.33, 918.67, 923.00, 927.33, 931.67]
			freq_list = [str(f) for f in frequencies]			
			for freq in freq_list:
				newdict['3h_20sb_f_g_c'] += np.array(olddict[freq])
			newdict['3h_20sb_f_g_c'] = list(newdict['3h_20sb_f_g_c'])
			
			newdict['4h_20sb_f_g_c'] = np.zeros(n_files)			
			frequencies = [1252.33, 1256.67, 1261.00, 1265.33, 1269.67, 1274.00, 1278.33, 1282.67, 1287.00, 1291.33, 1295.67, 1300.00, 1304.33, 1308.67, 1313.00, 1317.33, 1321.67, 1326.00, 1330.33, 1334.67] + [1161.33, 1165.67, 1170.00, 1174.33, 1178.67, 1183.00, 1187.33, 1191.67, 1196.00, 1200.33, 1204.67, 1209.00, 1213.33, 1217.67, 1222.00, 1226.33, 1230.67, 1235.00, 1239.33, 1243.67]
			freq_list = [str(f) for f in frequencies]			
			for freq in freq_list:
				newdict['4h_20sb_f_g_c'] += np.array(olddict[freq])
			newdict['4h_20sb_f_g_c'] = list(newdict['4h_20sb_f_g_c'])
			
			newdict['5h_20sb_f_g_c'] = np.zeros(n_files)			
			frequencies = [1564.33, 1568.67, 1573.00, 1577.33, 1581.67, 1586.00, 1590.33, 1594.67, 1599.00, 1603.33, 1607.67, 1612.00, 1616.33, 1620.67, 1625.00, 1629.33, 1633.67, 1638.00, 1642.33, 1646.67] + [1473.33, 1477.67, 1482.00, 1486.33, 1490.67, 1495.00, 1499.33, 1503.67, 1508.00, 1512.33, 1516.67, 1521.00, 1525.33, 1529.67, 1534.00, 1538.33, 1542.67, 1547.00, 1551.33, 1555.67]
			freq_list = [str(f) for f in frequencies]			
			for freq in freq_list:
				newdict['5h_20sb_f_g_c'] += np.array(olddict[freq])
			newdict['5h_20sb_f_g_c'] = list(newdict['5h_20sb_f_g_c'])

			
			newdict_versionB = {}
			# newdict['20sb_5h_f_g_c'] = list(np.array(newdict['5h_20sb_f_g_c']) + np.array(newdict['4h_20sb_f_g_c']) + np.array(newdict['3h_20sb_f_g_c']) + np.array(newdict['2h_20sb_f_g_c']) + np.array(newdict['1h_20sb_f_g_c']))
			newdict_versionB['20sb_5h_f_g_c'] = list(np.array(newdict['5h_20sb_f_g_c']) + np.array(newdict['4h_20sb_f_g_c']) + np.array(newdict['3h_20sb_f_g_c']) + np.array(newdict['2h_20sb_f_g_c']) + np.array(newdict['1h_20sb_f_g_c']))
			newdict_versionB['23h_f_f_r'] = newdict['23h_f_f_r']
			newdict_versionB['5h_f_g'] = newdict['5h_f_g']
			
			writer = pd.ExcelWriter('Meaning_' + basename(filepath))			
			DataFr = pd.DataFrame(data=newdict_versionB, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Env_Fft')
			writer.close()
		
	elif config['mode'] == 'sum_from_xls_cwd_damage':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
			newdict = {}
			olddict = pd.read_excel(filepath)
			rownames = list(olddict.index.values)
			n_files = len(rownames)
			olddict = olddict.to_dict(orient='list')
			# for key in olddict:
				# print(key)
			# print(olddict)
			# sys.exit()
			
			# newdict['f_g'] = olddict['312.0'][:-2]
			
			newdict['10h_f_f_r'] = np.zeros(n_files)
			#1300
			freq_list = ['0.3625', '0.725', '1.0875', '1.45', '1.8125', '2.175', '2.5375', '2.9', '3.2625', '3.625']

			
			# #1000
			# freq_list = ['3.33', '6.66', '9.99', '13.32', '16.65', '19.98', '23.32', '26.65', '29.98', '33.21']
			
			# #700
			# freq_list = ['2.33', '4.66', '6.99', '9.33', '11.66', '13.99', '16.32', '18.65', '20.98', '23.32']
			
			
			# #400
			# freq_list = ['1.33', '2.66', '4.0', '5.33', '6.66', '7.99', '9.33', '10.66', '11.99', '13.32']
			
			for freq in freq_list:
				newdict['10h_f_f_r'] += np.array(olddict[freq])
			newdict['10h_f_f_r'] = list(newdict['10h_f_f_r'])
			
			
			newdict['5h_f_g'] = np.zeros(n_files)
			freq_list = ['35.8875', '71.775', '107.6625', '143.55', '179.4375']

			
			for freq in freq_list:
				newdict['5h_f_g'] += np.array(olddict[freq])
			newdict['5h_f_g'] = list(newdict['5h_f_g'])
			
			
			newdict['5h_1sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [35.525, 71.4125, 107.3, 143.1875, 179.075]
			for freq in freq_list:
				newdict['5h_1sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_1sbi_f_g_c'] = list(newdict['5h_1sbi_f_g_c'])
			
			
			newdict['5h_1sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [36.25, 72.1375, 108.025, 143.9125, 179.8]
			for freq in freq_list:
				newdict['5h_1sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_1sbd_f_g_c'] = list(newdict['5h_1sbd_f_g_c'])
			
			
			newdict['5h_2sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [35.1625, 71.05, 106.9375, 142.825, 178.7125]
			for freq in freq_list:
				newdict['5h_2sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_2sbi_f_g_c'] = list(newdict['5h_2sbi_f_g_c'])
			
			
			newdict['5h_2sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [36.6125, 72.5, 108.3875, 144.275, 180.1625]
			for freq in freq_list:
				newdict['5h_2sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_2sbd_f_g_c'] = list(newdict['5h_2sbd_f_g_c'])
			
			
			newdict['5h_3sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [34.8, 70.6875, 106.575, 142.4625, 178.35]
			for freq in freq_list:
				newdict['5h_3sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_3sbi_f_g_c'] = list(newdict['5h_3sbi_f_g_c'])
			
			
			newdict['5h_3sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [36.975, 72.8625, 108.75, 144.6375, 180.525]
			for freq in freq_list:
				newdict['5h_3sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_3sbd_f_g_c'] = list(newdict['5h_3sbd_f_g_c'])
			
			
			newdict['5h_4sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [34.4375, 70.325, 106.2125, 142.1, 177.9875]
			for freq in freq_list:
				newdict['5h_4sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_4sbi_f_g_c'] = list(newdict['5h_4sbi_f_g_c'])
			
			
			newdict['5h_4sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [37.3375, 73.225, 109.1125, 145, 180.8875]
			for freq in freq_list:
				newdict['5h_4sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_4sbd_f_g_c'] = list(newdict['5h_4sbd_f_g_c'])
			
			
			newdict['5h_5sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [34.075, 69.9625, 105.85, 141.7375,	177.625]
			for freq in freq_list:
				newdict['5h_5sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_5sbi_f_g_c'] = list(newdict['5h_5sbi_f_g_c'])
			
			
			newdict['5h_5sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [37.7, 73.5875,	109.475, 145.3625, 181.25]
			for freq in freq_list:
				newdict['5h_5sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_5sbd_f_g_c'] = list(newdict['5h_5sbd_f_g_c'])
			
			
			newdict['5h_6sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [33.7125, 69.6,	105.4875, 141.375, 177.2625]
			for freq in freq_list:
				newdict['5h_6sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_6sbi_f_g_c'] = list(newdict['5h_6sbi_f_g_c'])
			
			
			newdict['5h_6sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [38.0625, 73.95, 109.8375, 145.725, 181.6125]
			for freq in freq_list:
				newdict['5h_6sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_6sbd_f_g_c'] = list(newdict['5h_6sbd_f_g_c'])
			
			
			newdict['5h_sbd_f_g_c'] = list(np.array(newdict['5h_6sbd_f_g_c']) + np.array(newdict['5h_5sbd_f_g_c']) + np.array(newdict['5h_4sbd_f_g_c']) + np.array(newdict['5h_3sbd_f_g_c']) + np.array(newdict['5h_2sbd_f_g_c']) + np.array(newdict['5h_1sbd_f_g_c']))
			
			newdict['5h_sbi_f_g_c'] = list(np.array(newdict['5h_6sbi_f_g_c']) + np.array(newdict['5h_5sbi_f_g_c']) + np.array(newdict['5h_4sbi_f_g_c']) + np.array(newdict['5h_3sbi_f_g_c']) + np.array(newdict['5h_2sbi_f_g_c']) + np.array(newdict['5h_1sbi_f_g_c']))
			
			newdict['5h_sb_f_g_c'] = list(np.array(newdict['5h_sbd_f_g_c']) + np.array(newdict['5h_sbi_f_g_c']))

			
			newdict['5h_sb_f_g_c_1'] = list( np.array(newdict['5h_1sbd_f_g_c']) + np.array(newdict['5h_1sbi_f_g_c']))
			newdict['5h_sb_f_g_c_2'] = list( np.array(newdict['5h_2sbd_f_g_c']) + np.array(newdict['5h_2sbi_f_g_c']))
			newdict['5h_sb_f_g_c_3'] = list( np.array(newdict['5h_3sbd_f_g_c']) + np.array(newdict['5h_3sbi_f_g_c']))
			newdict['5h_sb_f_g_c_4'] = list( np.array(newdict['5h_4sbd_f_g_c']) + np.array(newdict['5h_4sbi_f_g_c']))
			newdict['5h_sb_f_g_c_5'] = list( np.array(newdict['5h_5sbd_f_g_c']) + np.array(newdict['5h_5sbi_f_g_c']))
			newdict['5h_sb_f_g_c_6'] = list( np.array(newdict['5h_6sbd_f_g_c']) + np.array(newdict['5h_6sbi_f_g_c']))
			
			
			writer = pd.ExcelWriter('Meaning_' + basename(filepath))			
			DataFr = pd.DataFrame(data=newdict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Env_Fft')
			writer.close()
	
	elif config['mode'] == 'sum_from_xls_cwd_NOdamage':
		print('Select xls')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath in Filepaths:
			newdict = {}
			olddict = pd.read_excel(filepath)
			rownames = list(olddict.index.values)
			n_files = len(rownames)
			olddict = olddict.to_dict(orient='list')
			# for key in olddict:
				# print(key)
			# print(olddict)
			# sys.exit()
			
			# newdict['f_g'] = olddict['312.0'][:-2]
			
			newdict['10h_f_f_r'] = np.zeros(n_files)
			#1300
			freq_list = [0.3091667, 0.6183334, 0.9275001, 1.2366668, 1.5458335, 1.8550002, 2.1641669, 2.4733336, 2.7825003, 3.091667]

			
			# #1000
			# freq_list = ['3.33', '6.66', '9.99', '13.32', '16.65', '19.98', '23.32', '26.65', '29.98', '33.21']
			
			# #700
			# freq_list = ['2.33', '4.66', '6.99', '9.33', '11.66', '13.99', '16.32', '18.65', '20.98', '23.32']
			
			
			# #400
			# freq_list = ['1.33', '2.66', '4.0', '5.33', '6.66', '7.99', '9.33', '10.66', '11.99', '13.32']
			
			for freq in freq_list:
				newdict['10h_f_f_r'] += np.array(olddict[str(freq)])
			newdict['10h_f_f_r'] = list(newdict['10h_f_f_r'])
			
			
			newdict['5h_f_g'] = np.zeros(n_files)
			freq_list = [30.6075, 61.215, 91.8225, 122.43, 153.0375]

			
			for freq in freq_list:
				newdict['5h_f_g'] += np.array(olddict[str(freq)])
			newdict['5h_f_g'] = list(newdict['5h_f_g'])
			
			
			newdict['5h_1sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [30.2983333, 60.9058333, 91.5133333, 122.1208333, 152.7283333]
			for freq in freq_list:
				newdict['5h_1sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_1sbi_f_g_c'] = list(newdict['5h_1sbi_f_g_c'])
			
			
			newdict['5h_1sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [30.9166667, 61.5241667, 92.1316667, 122.7391667, 153.3466667]
			for freq in freq_list:
				newdict['5h_1sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_1sbd_f_g_c'] = list(newdict['5h_1sbd_f_g_c'])
			
			
			newdict['5h_2sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [29.9891666, 60.5966666, 91.2041666, 121.8116666, 152.4191666]
			for freq in freq_list:
				newdict['5h_2sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_2sbi_f_g_c'] = list(newdict['5h_2sbi_f_g_c'])
			
			
			newdict['5h_2sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [31.2258334, 61.8333334, 92.4408334, 123.0483334, 153.6558334]
			for freq in freq_list:
				newdict['5h_2sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_2sbd_f_g_c'] = list(newdict['5h_2sbd_f_g_c'])
			
			
			newdict['5h_3sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [29.6799999, 60.2874999, 90.8949999, 121.5024999, 152.1099999]
			for freq in freq_list:
				newdict['5h_3sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_3sbi_f_g_c'] = list(newdict['5h_3sbi_f_g_c'])
			
			
			newdict['5h_3sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [31.5350001, 62.1425001, 92.7500001, 123.3575001, 153.9650001]
			for freq in freq_list:
				newdict['5h_3sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_3sbd_f_g_c'] = list(newdict['5h_3sbd_f_g_c'])
			
			
			newdict['5h_4sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [29.3708332, 59.9783332, 90.5858332, 121.1933332, 151.8008332]
			for freq in freq_list:
				newdict['5h_4sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_4sbi_f_g_c'] = list(newdict['5h_4sbi_f_g_c'])
			
			
			newdict['5h_4sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [31.8441668, 62.4516668, 93.0591668, 123.6666668, 154.2741668]
			for freq in freq_list:
				newdict['5h_4sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_4sbd_f_g_c'] = list(newdict['5h_4sbd_f_g_c'])
			
			
			newdict['5h_5sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [29.0616665, 59.6691665, 90.2766665, 120.8841665, 151.4916665]
			for freq in freq_list:
				newdict['5h_5sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_5sbi_f_g_c'] = list(newdict['5h_5sbi_f_g_c'])
			
			
			newdict['5h_5sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [32.1533335, 62.7608335, 93.3683335, 123.9758335, 154.5833335]
			for freq in freq_list:
				newdict['5h_5sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_5sbd_f_g_c'] = list(newdict['5h_5sbd_f_g_c'])
			
			
			newdict['5h_6sbi_f_g_c'] = np.zeros(n_files)
			freq_list = [28.7524998, 59.3599998, 89.9674998, 120.5749998, 151.1824998]
			for freq in freq_list:
				newdict['5h_6sbi_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_6sbi_f_g_c'] = list(newdict['5h_6sbi_f_g_c'])
			
			
			newdict['5h_6sbd_f_g_c'] = np.zeros(n_files)
			freq_list = [32.4625002, 63.0700002, 93.6775002, 124.2850002, 154.8925002]
			for freq in freq_list:
				newdict['5h_6sbd_f_g_c'] += np.array(olddict[str(freq)])
			newdict['5h_6sbd_f_g_c'] = list(newdict['5h_6sbd_f_g_c'])
			
			
			newdict['5h_sbd_f_g_c'] = list(np.array(newdict['5h_6sbd_f_g_c']) + np.array(newdict['5h_5sbd_f_g_c']) + np.array(newdict['5h_4sbd_f_g_c']) + np.array(newdict['5h_3sbd_f_g_c']) + np.array(newdict['5h_2sbd_f_g_c']) + np.array(newdict['5h_1sbd_f_g_c']))
			
			newdict['5h_sbi_f_g_c'] = list(np.array(newdict['5h_6sbi_f_g_c']) + np.array(newdict['5h_5sbi_f_g_c']) + np.array(newdict['5h_4sbi_f_g_c']) + np.array(newdict['5h_3sbi_f_g_c']) + np.array(newdict['5h_2sbi_f_g_c']) + np.array(newdict['5h_1sbi_f_g_c']))
			
			newdict['5h_sb_f_g_c'] = list(np.array(newdict['5h_sbd_f_g_c']) + np.array(newdict['5h_sbi_f_g_c']))

			
			newdict['5h_sb_f_g_c_1'] = list( np.array(newdict['5h_1sbd_f_g_c']) + np.array(newdict['5h_1sbi_f_g_c']))
			newdict['5h_sb_f_g_c_2'] = list( np.array(newdict['5h_2sbd_f_g_c']) + np.array(newdict['5h_2sbi_f_g_c']))
			newdict['5h_sb_f_g_c_3'] = list( np.array(newdict['5h_3sbd_f_g_c']) + np.array(newdict['5h_3sbi_f_g_c']))
			newdict['5h_sb_f_g_c_4'] = list( np.array(newdict['5h_4sbd_f_g_c']) + np.array(newdict['5h_4sbi_f_g_c']))
			newdict['5h_sb_f_g_c_5'] = list( np.array(newdict['5h_5sbd_f_g_c']) + np.array(newdict['5h_5sbi_f_g_c']))
			newdict['5h_sb_f_g_c_6'] = list( np.array(newdict['5h_6sbd_f_g_c']) + np.array(newdict['5h_6sbi_f_g_c']))
			
			
			writer = pd.ExcelWriter('Meaning_' + basename(filepath))			
			DataFr = pd.DataFrame(data=newdict, index=rownames)		
			DataFr.to_excel(writer, sheet_name='Env_Fft')
			writer.close()

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
