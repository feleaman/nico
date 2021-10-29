#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
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
from m_plots import *
from decimal import Decimal

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser


#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {'channel':'OFF', 'fs':'OFF', 'output':'plot', 'color':'red', 'marker':'D', 'spectrum':'fourier'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	if config['mode'] == 'scatter':		
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
	
		Filepaths_a = Filepaths[0::2]
		Filepaths_b = Filepaths[1::2]
				
		fmax_a = []
		crest_a = []
		for filepath_a in Filepaths_a:
			print(basename(filepath_a))
			mydict = pd.read_excel(filepath_a)
			mydict = mydict.to_dict(orient='list')
			fmax_a += list(np.array(mydict['fmax'])/1000.)
			crest_a += mydict['crest']
		fmax_a = np.array(fmax_a)
		crest_a = np.array(crest_a)
		print('len first channel = ', len(fmax_a))
		
		fmax_b = []
		crest_b = []
		for filepath_b in Filepaths_b:
			print(basename(filepath_b))
			mydict = pd.read_excel(filepath_b)
			mydict = mydict.to_dict(orient='list')
			fmax_b += list(np.array(mydict['fmax'])/1000.)
			crest_b += mydict['crest']
		fmax_b = np.array(fmax_b)
		crest_b = np.array(crest_b)
		print('len second channel = ', len(fmax_b))
		
		name = 'Fwdom_Crest_S1_Rest'
		path_1 = 'C:\\Felix\\58_Crack_Detection_WTG\\03_Figures\\'
		path_2 = 'C:\\Felix\\58_Crack_Detection_WTG\\12_Latex\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		print(np.max(crest_a))
		print(np.max(crest_b))
		
		style = {'xlabel':'$x_{crf}$ [-]', 'ylabel':'$f_{dom}^{\psi}$ [kHz]', 'legend':[None, None, None, None, None, None], 'title':['AE-1', 'AE-2', 'AE-3: envelope'], 'customxlabels':None, 'xlim':[0, 25], 'ylim':[0, 500], 'color':config['color'], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'OFF', 'ylog':'OFF', 'marker':config['marker'], 'scatter':'ON', 'customylabels':None}
		
		data = {'x':[crest_a, crest_b], 'y':[fmax_a, fmax_b]}

		# plt.rcParams['mathtext.fontset'] = 'cm'
		myplot_scatter_2h(data, style)
	
	# elif config['mode'] == 'cyclic_spectrum':		
		
		
		# print('Select MAT from DATA') 
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()
		# root.destroy()		
		# data = load_signal(filepath, channel=config['channel'])
		
		
		# print('Select MAT from ALPHA')
		# root = Tk()
		# root.withdraw()
		# root.update()  
		# filepath = filedialog.askopenfilename()
		# root.destroy()		
		# alpha = load_signal(filepath, channel='alpha')
		
		# print('Select MAT from FREQ')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()
		# root.destroy()		
		# f = load_signal(filepath, channel='f')
		
		# data = np.reshape(data, (int(len(f)*2), len(alpha)))
		# data = data[0:int(data.shape[0]/2) , :]
		# data = np.absolute(data)		
		
		# ##frequencies
		# # sum = np.zeros(len(data[:,0])) #sum len 100
		# # for i in range(len(data[0,:])): #i itera de 1 a 100
			# # sum += data[:,i] #data suma a len 100
		# # plt.plot(f/1000, sum)
		# # plt.show()
		
		# sum = np.zeros(len(data[0,:])) #sum len 100
		# n = 0
		# for i in range(len(data[:,0])): #i itera de 1 a 100
			# sum += data[i,:] #data suma a len 100
			# n += 1
		# spectrum = sum/n
		# plt.plot(alpha, spectrum)
		# plt.show()
		
		# mydict = {'coherence':spectrum, 'alpha':alpha}
		# name = 'B4c_AE3_' + config['channel']
		# save_pickle(name + '.pkl', mydict)
	
	
	elif config['mode'] == 'cyclic_map_single':		
		
		
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data = load_signal(filepath, channel=config['channel'])
		
		
		print('Select MAT from ALPHA')
		root = Tk()
		root.withdraw()
		root.update()  
		filepath = filedialog.askopenfilename()
		root.destroy()		
		alpha = load_signal(filepath, channel='alpha')
		
		print('Select MAT from FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f = load_signal(filepath, channel='f')
		
		data = np.reshape(data, (int(len(f)*2), len(alpha)))
		data = data[0:int(data.shape[0]/2) , :]
		data = np.absolute(data)
		# data = np.log(data)
		data[:,0] = 0.
		data[:,2000:] = 0.
		# data = data[:,2000:]
		
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')			
		# alpha, f = np.meshgrid(alpha, f)			
		# ax.plot_surface(alpha, f/1000, data, cmap='plasma')	
		
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# dog = ax.pcolormesh(alpha, f/1000, data, cmap='inferno', vmax=None)
		# cbar = fig.colorbar(dog, ax=ax)			
		# cbar.set_label('SCoh [-]', fontsize=13)
		# cbar.ax.tick_params(axis='both', labelsize=12)		
		
		# fig = plt.figure()
		# extent_ = [0, np.max(alpha), 0, 500]			
		# mydict = {'map':data, 'extent':extent_}
		# save_pickle('CycloDensity_simulated_AE_mod_fault_frec.pkl', mydict)
		
		# fig, ax = plt.subplots()
		# extent_ = [0, np.max(alpha), 0, 500]
		# contour_ = ax.contourf(data, extent=extent_, cmap='plasma')	
		# cbar = fig.colorbar(contour_, ax=ax)
		
			
		# ax.set_xlabel('Cyclic Frequency [Hz]', fontsize=13)
		# ax.set_ylabel('Frequency [kHz]', fontsize=13)
		# ax.tick_params(axis='both', labelsize=12)
		# ax.set_title('S-1: AE-3', fontsize=12)
		# ax.set_xlim(left=0, right=200)
		
		# plt.show()
		
		
		name = 'SpectralCoherence_S1_AE1'
		path_1 = 'C:\\Felix\\58_Crack_Detection_WTG\\03_Figures\\'
		path_2 = 'C:\\Felix\\58_Crack_Detection_WTG\\12_Latex\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		
		extent_ = [0, np.max(alpha), 0, 500]
		style = {'xlabel':r'$\alpha$ [Hz]', 'ylabel':r'$f$ [kHz]', 'legend':[None], 'title':'AE-1', 'xticklabels':None, 'xlim':[0, 200], 'ylim':[0, 500], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'extent':extent_, 'colormap':'YlGnBu', 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b}

		myplot_map(data, style)
		
		# 'YlGnBu'
		# 'YlOrBr'
	
	
	elif config['mode'] == 'cyclic_map_double':		
		
		#data 1+++++++++++++++
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data1 = load_signal(filepath, channel=config['channel'])
		
		
		print('Select MAT from ALPHA')
		root = Tk()
		root.withdraw()
		root.update()  
		filepath = filedialog.askopenfilename()
		root.destroy()		
		alpha1 = load_signal(filepath, channel='alpha')
		
		print('Select MAT from FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f1 = load_signal(filepath, channel='f')
		
		data1 = np.reshape(data1, (int(len(f1)*2), len(alpha1)))
		data1 = data1[0:int(data1.shape[0]/2) , :]
		data1 = np.absolute(data1)
		# data = np.log(data)
		data1[:,0] = 0.
		data1[:,2000:] = 0.
		
		
		
		#data 2+++++++++++++++
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data2 = load_signal(filepath, channel=config['channel'])
		
		
		print('Select MAT from ALPHA')
		root = Tk()
		root.withdraw()
		root.update()  
		filepath = filedialog.askopenfilename()
		root.destroy()		
		alpha2 = load_signal(filepath, channel='alpha')
		
		print('Select MAT from FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f2 = load_signal(filepath, channel='f')
		
		data2 = np.reshape(data2, (int(len(f2)*2), len(alpha2)))
		data2 = data2[0:int(data2.shape[0]/2) , :]
		data2 = np.absolute(data2)
		# data = np.log(data)
		data2[:,0] = 0.
		data2[:,2000:] = 0.
		
		
		
		name = 'Coherence_S1'
		path_1 = 'C:\\Felix\\58_Crack_Detection_WTG\\03_Figures\\'
		path_2 = 'C:\\Felix\\58_Crack_Detection_WTG\\12_Latex\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		
		extent_ = [0, np.max(alpha2), 0, 500]
		style = {'xlabel':r'$\alpha$ [Hz]', 'ylabel':r'$f$ [kHz]', 'legend':[None], 'title':['AE-1', 'AE-2'], 'xticklabels':None, 'xlim':[0, 200], 'ylim':[0, 500], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'extent':extent_, 'colormap':'OrRd', 'output':config['output'], 'cbar_title':r'$\gamma_{xx}(f,\alpha)$ [-]', 'path_1':path_1b, 'path_2':path_2b}	
		
		myplot_map_double_join(data1, data2, style)
		
		# 'PuBu'
		# 'OrRd'
		# 'YlGnBu'
		# 'YlGnBu'
		# 'YlOrBr'
	
	elif config['mode'] == 'cyclic_map_four':		
		
		#data 1+++++++++++++++
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data1 = load_signal(filepath, channel=config['channel'])
		
		
		print('Select MAT from ALPHA')
		root = Tk()
		root.withdraw()
		root.update()  
		filepath = filedialog.askopenfilename()
		root.destroy()		
		alpha1 = load_signal(filepath, channel='alpha')
		
		print('Select MAT from FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f1 = load_signal(filepath, channel='f')
		
		data1 = np.reshape(data1, (int(len(f1)*2), len(alpha1)))
		data1 = data1[0:int(data1.shape[0]/2) , :]
		data1 = np.absolute(data1)
		# data = np.log(data)
		data1[:,0] = 0.
		data1[:,2000:] = 0.
		
		
		
		#data 2+++++++++++++++
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data2 = load_signal(filepath, channel=config['channel'])
		
		
		print('Select MAT from ALPHA')
		root = Tk()
		root.withdraw()
		root.update()  
		filepath = filedialog.askopenfilename()
		root.destroy()		
		alpha2 = load_signal(filepath, channel='alpha')
		
		print('Select MAT from FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f2 = load_signal(filepath, channel='f')
		
		data2 = np.reshape(data2, (int(len(f2)*2), len(alpha2)))
		data2 = data2[0:int(data2.shape[0]/2) , :]
		data2 = np.absolute(data2)
		# data = np.log(data)
		data2[:,0] = 0.
		data2[:,2000:] = 0.
		
		
		
		name = 'Density_S1'
		path_1 = 'C:\\Felix\\58_Crack_Detection_WTG\\03_Figures\\'
		path_2 = 'C:\\Felix\\58_Crack_Detection_WTG\\12_Latex\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		# foo = ' '
		# for p in range(30):
			# foo += foo
		# 'title':['                                                                      (a) AE-1', '                                                                      (b) AE-2']
		# r'$\gamma_{xx}(f,\alpha)$ [-]'
		extent_ = [0, np.max(alpha2), 0, 500]
		style = {'xlabel':r'$\alpha$ [Hz]', 'ylabel':r'$f$ [kHz]', 'legend':[None], 'title':['AE-1', 'AE-2'], 'xticklabels':None, 'xlim':[0, 20, 160, 200], 'ylim':[0, 500], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'extent':extent_, 'colormap':'OrRd', 'output':config['output'], 'cbar_title':r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', 'yticklabels':[0, 100, 200, 300, 400, 500], 'path_1':path_1b, 'path_2':path_2b}
		
		myplot_map_four_join(data1, data2, style)
		
		# 'PuBu'
		# 'OrRd'
		# 'YlGnBu'
		# 'YlGnBu'
		# 'YlOrBr'
	
	elif config['mode'] == 'avg_spectra':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()	
		for filepath in Filepaths:
			print(basename(filepath))
		dict1 = read_pickle(Filepaths[0])
		dict2 = read_pickle(Filepaths[1])

		factor_f = 1.
		if config['spectrum'] == 'wavelet':
			f_1 = dict1['freq']/factor_f
			f_2 = dict2['freq']/factor_f
			magX_1 = dict1['energy']
			magX_2 = dict2['energy']
		elif config['spectrum'] == 'coherence':
			f_1 = dict1['alpha']/factor_f
			f_2 = dict2['alpha']/factor_f
			magX_1 = dict1['coherence']
			magX_2 = dict2['coherence']
		elif config['spectrum'] == 'density':
			f_1 = dict1['alpha']/factor_f
			f_2 = dict2['alpha']/factor_f
			magX_1 = dict1['density']
			magX_2 = dict2['density']
		else:
			f_1 = dict1['f']/factor_f
			f_2 = dict2['f']/factor_f
			magX_1 = dict1['fft']
			magX_2 = dict2['fft']
			

		

		# name = 'WaveletSpectrum_B4c'
		# name = 'FourierSpectrum_S1'
		# name = 'EnvSpectrum_S1'
		name = 'BPEnvSpectrum_S1'

		path_1 = 'D:\\2017_2019\\58_Crack_Detection_WTG\\03_Figures\\'
		path_2 = 'D:\\2017_2019\\58_Crack_Detection_WTG\\12_Latex\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		

		# 'Magnitude [dB$_{AE}$]'
		# '$X_{\psi}^{2}$ [mV$^{2}$]'
		
		# [1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9]
		# [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e7, 1.e8, 1.e9]
		
		style = {'xlabel':r'$f$ [Hz]', 'ylabel':r'A [dB$_{AE}$]', 'legend':[None, None, None, None, None, None], 'title':['AE-1', 'AE-2', 'AE-3: envelope'], 'xlim':[0,200], 'ylim':[-20, 30], 'color':config['color'], 'loc_legend':'upper right', 'legend_line':'OFF', 'vlines':None, 'range_lines':[0,200], 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'dbae':'ON', 'ylog':'OFF', 'marker':config['marker'], 'scatter':'OFF', 'customxlabels':None, 'customylabels':None}
		
		data = {'x':[f_1, f_2], 'y':[magX_1, magX_2]}

		# plt.rcParams['mathtext.fontset'] = 'cm'
		myplot_scatter_2h(data, style)		
	
	
	elif config['mode'] == 'scalogram':
		
		name = 'Scgr_Bursts_B4c'
			
		path_1 = 'C:\\Felix\\58_Crack_Detection_WTG\\03_Figures\\'
		path_2 = 'C:\\Felix\\58_Crack_Detection_WTG\\12_Latex\\'			
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		
		print('++++++++select scgr')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		# sss = []
		
		map = []
		for filepath in Filepaths:
			mydict = read_pickle(filepath)
			
			# mapa = mydict['map']
			# print(type(mapa))
			# n = len(mapa)
			# sum = np.zeros(n)
			# for i in range(n):
				# sum[i] = np.sum(mapa[i])
			# # plt.plot(sum)
			# # plt.show()			
			# # sys.exit()
			# sss.append(sum)
			
			map.append(mydict['map'])
			extent_ = mydict['extent']
			extent_[1] = extent_[1]*1000
		
		# fig1, ax1 = plt.subplots(ncols=3)
		# ax1[0].plot(sss[0])
		# ax1[1].plot(sss[1])
		# ax1[2].plot(sss[2])
		# plt.show()
		
		print('++++++++select wfm')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		
		x = []
		t = []
		for filepath in Filepaths:
			mydict = read_pickle(filepath)
			mydict = read_pickle(filepath)
			t.append(mydict['t']*1000.)
			x.append(mydict['x'])	
		
		# [[-40, 40], [-10, 10], [-10, 10], [0, 500], [0, 500], [0, 500]]
		# ['$t$ [ms]', '$t$ [ms]', '$t$ [ms]', None, None, None]
		# ['$x$ [mV]', '$x$ [mV]', '$x$ [mV]', '$f$ [kHz]', '$f$ [kHz]', '$f$ [kHz]']
		
		style = {'xlabel':['$t$ [ms]', '$t$ [ms]', '$t$ [ms]', None, None, None], 'ylabel':['$x$ [mV]', '$x$ [mV]', '$x$ [mV]', '$f$ [kHz]', '$f$ [kHz]', '$f$ [kHz]'], 'legend':[None], 'title':None, 'xticklabels':None, 'xlim':[0.95, 1.55], 'ylim':[[-10, 10], [-10, 10], [-40, 40], [0, 500], [0, 500], [0, 500]], 'color':[None], 'loc_legend':'upper left', 'legend_line':'OFF', 'vlines':None, 'range_lines':None, 'extent':extent_, 'colormap':'PuBu', 'cbar_title':r'$x_{\psi}^{2}$ [mV$^{2}$]', 'output':config['output'], 'path_1':path_1b, 'path_2':path_2b, 'color':None}
		# data = {'x':[t], 'y':[Feature]}
		# 'STFT Spectrogram $\it{\Delta}$t = 0.01 ms'
		# 'YlOrBr'
		# 'plasma'
		# 'YlOrRd'
		# 'OrRd'
		# 'PuBu'
		#'darkred'
		myplot_burst_scgr_6(map, t, x, style)
		# myplot_scgr(map, style)
	
	else:
		print('error mode')
		
		
		
		
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
	# if config['power2'] != 'auto' and config['power2'] != 'OFF':
		# config['power2'] = int(config['power2'])
	# config['fs_tacho'] = float(config['fs_tacho'])
	# config['fs_signal'] = float(config['fs_signal'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config

def myplot_burst_scgr_6(map, t, x, style):
	#Modules and global properties
	# from matplotlib import font_manager
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=3, nrows=2, squeeze=True, gridspec_kw = {'height_ratios':[1,1.4]})
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	font_caption = 23+3
	lim = 3
	
	fig.text(0.03, 0.025, '(a)', fontsize=font_caption)
	fig.text(0.355, 0.025, '(b)', fontsize=font_caption)
	fig.text(0.68, 0.025, '(c)', fontsize=font_caption)
	
	# plt.subplots_adjust(left=0.15, right=0.93, bottom=0.165, top=0.92)
	
	
	plt.subplots_adjust(hspace=0.1, wspace=0.295, left=0.07, right=0.97, bottom=0.08, top=0.91)
	fig.set_size_inches(14.2, 7.5)
	
	count = 0
	for i in range(2):
		for j in range(3):
			
			
			if i == 1:
				extent_ = style['extent']
				# mymax = np.max(map[j])
				# mymin = np.min(map[j])
				# levels_ = np.linspace(mymin, mymax, 5)
				contour_ = ax[i][j].contourf(map[j], extent=extent_, cmap=style['colormap'])	
				cbar = fig.colorbar(contour_, ax=ax[i][j], orientation='horizontal', pad=0.05)			
				cbar.set_label(style['cbar_title'], fontsize=font_big)	
				cbar.ax.tick_params(labelsize=font_little) 			
				cbar.ax.yaxis.get_offset_text().set(size=font_little)
				
				if j == 2:
					foo = cbar.ax.get_xticklabels()[0::2]
					fuu = []
					for fo in foo:
						fuu.append(fo)
						fuu.append('')
					cbar.ax.set_xticklabels(fuu, rotation=0)
					
				# if j == 1:
					# # ax[i][j].xaxis.set_tick_params(rotation=45)
					# cbar.ax.yaxis.set_tick_params(rotation=45)
				
				# for ax_it in ax[i][j].flatten():
					# for tk in ax_it.get_yticklabels():
						# tk.set_visible(True)
				for tk in ax[i][j].get_xticklabels():
					tk.set_visible(False)
				# ax[i][j].yaxis.offsetText.set_visible(True)
				# ax[i][j].xaxis.offsetText.set_visible(False)
				# ax[i][j].axes.get_xaxis().set_visible(False)
				
			elif i == 0:
				ax[i][j].plot(t[j], x[j], color=style['color'])
				# ax[i][j].xaxis.tick_top()
				ax[i][j].xaxis.set_ticks_position('top')
				ax[i][j].xaxis.set_label_position('top')
			

			if style['xlabel'][count] != None:
				ax[i][j].set_xlabel(style['xlabel'][count], fontsize=font_big)
			ax[i][j].set_ylabel(style['ylabel'][count], fontsize=font_big)

			
			#Size labels from axis
			ax[i][j].tick_params(axis='both', labelsize=font_little)
				
			#Scientific notation	
			ax[i][j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
			
			# #Visibility
			# for ax_it in ax.flatten():
				# for tk in ax_it.get_yticklabels():
					# tk.set_visible(True)
				# for tk in ax_it.get_xticklabels():
					# tk.set_visible(True)
				# ax_it.yaxis.offsetText.set_visible(True)

			#Eliminate line from label
			if style['legend'] != None:
				if style['legend_line'] == 'OFF':
					ax[i][j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
				else:
					ax[i][j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
			
			#Title
			if style['title'] != None:
				# plt.rcParams['mathtext.fontset'] = 'cm'
				# ax.set_title(keys[0], fontsize=font_big)	
				ax[i][j].set_title(style['title'], fontsize=font_big)
			#Size from offset text
			ax[i][j].yaxis.offsetText.set_fontsize(font_offset)
			

			
			
				
			
			
			#Set Limits in Axis X
			if style['ylim'] != None:
				ax[i][j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])
			
			if style['xlim'] != None:
				ax[i][j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
				
			#Set Vertical Lines

			#Set Ticks in Axis X
			if style['xticklabels'] != None:
				print('+++++++++++++++')
				ax[i][j].set_xticks(style['xticklabels']) 		
				ax[i][j].set_xticklabels(style['xticklabels']) 
			
			
			ax[i][j].grid(axis='both')
			count += 1
		
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return	

def myplot_scgr(map, style):
	#Modules and global properties
	# from matplotlib import font_manager
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	fig, ax = plt.subplots(ncols=1, nrows=1)
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	lim = 3

	plt.subplots_adjust(left=0.15, right=0.93, bottom=0.165, top=0.92)
	
	
	
	ax.set_xlabel(style['xlabel'], fontsize=font_big)
	

	extent_ = style['extent']
	contour_ = ax.contourf(map, extent=extent_, cmap=style['colormap'])
	
	
	cbar = fig.colorbar(contour_, ax=ax)
	
	# cbar.set_label('SCD [V$^{2}$]', fontsize=font_big)
	cbar.set_label(style['cbar_title'], fontsize=font_big)
	
	# cbar.set_label('SC [-]', fontsize=font_big)
	# cbar.set_label('Energy [V$^{2}$]', fontsize=font_big)
	# cbar.set_label('Magnitude [V]', fontsize=font_big)
	# cbar.set_ticks([0, 200, 400, 600, 800])
	# cbar.set_ticks([0, 1200, 2400, 3600, 4800])
	cbar.ax.tick_params(labelsize=font_little) 
	
	cbar.ax.yaxis.get_offset_text().set(size=font_little)
	
	
	ax.set_ylabel(style['ylabel'], fontsize=font_big)

	
	#Size labels from axis
	ax.tick_params(axis='both', labelsize=font_little)
		
	#Scientific notation	
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	# #Visibility
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
		# ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] != None:
		if style['legend_line'] == 'OFF':
			ax.legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		else:
			ax.legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
	
	#Title
	if style['title'] != None:
		# plt.rcParams['mathtext.fontset'] = 'cm'
		# ax.set_title(keys[0], fontsize=font_big)	
		ax.set_title(style['title'], fontsize=font_big)
	#Size from offset text
	ax.yaxis.offsetText.set_fontsize(font_offset)
	

	
	
		
	
	
	#Set Limits in Axis X
	if style['ylim'] != None:
		ax.set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
	
	if style['xlim'] != None:
		ax.set_xlim(left=style['xlim'][0], right=style['xlim'][1])
		
	#Set Vertical Lines

	#Set Ticks in Axis X
	if style['xticklabels'] != None:
		print('+++++++++++++++')
		ax.set_xticks(style['xticklabels']) 		
		ax.set_xticklabels(style['xticklabels']) 
	
	ax.grid(axis='both')
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return
	
def myplot_scatter_2h(data, style):
	# from matplotlib import font_manager	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	# fig, ax = plt.subplots(ncols=3, nrows=1, sharey='row')
	fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
	lim = 2
	
	
	font_big = 17+3
	font_little = 15+3
	font_label = 13+3
	font_offset = 15+3
	font_autolabel = 15+3
	font_caption = 23+3
	# plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.15, top=0.95, hspace=0.52)
	# # hspace=0.47
	# fig.set_size_inches(14.2, 3.6)
	# # 6.5
	# fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
	
	# fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
	# fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
	# fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)
	# # 0.522
	
	
	# # plt.subplots_adjust(wspace=0.32, left=0.065, right=0.98, bottom=0.213, top=0.89)
	# plt.subplots_adjust(wspace=0.32, left=0.066, right=0.98, bottom=0.213, top=0.81)
	# # fig.set_size_inches(14.2, 4.0)
	# fig.set_size_inches(14.2, 4.6)

	# fig.text(0.059, 0.05, '(a)', fontsize=font_caption)
	# fig.text(0.387, 0.05, '(b)', fontsize=font_caption)
	# fig.text(0.717, 0.05, '(c)', fontsize=font_caption)

	
	
	# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.21, top=0.89)
	plt.subplots_adjust(wspace=0.33, left=0.095, right=0.975, bottom=0.213, top=0.89)
	fig.set_size_inches(10.2, 4.0)
	fig.text(0.05, 0.03, '(a)', fontsize=font_caption)
	fig.text(0.55, 0.03, '(b)', fontsize=font_caption)

	

	count = 0
	
	for j in range(2):
		if style['dbae'] == 'ON':
			data_y = 20*np.log10(1000*data['y'][count])
		else:
			data_y = data['y'][count]
		# ax[j].set_yticks([-30, -15, 0, 15, 30])
		
		if style['ylog'] == 'ON':
			ax[j].semilogy(data['x'][count], data_y, label=style['legend'][count], color=style['color'])
		else:
			if style['scatter'] == 'ON':
				ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker=style['marker'], ls='', color=style['color'])
			# ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker='o', ls='')
			# ax[j].plot(data['x'][count], data_y, label=style['legend'][count])
			# ax[j].bar(data['x'][count], data_y, label=style['legend'][count])
			else:
				ax[j].plot(data['x'][count], data_y, label=style['legend'][count], color=style['color'])
		ax[j].set_xlabel(style['xlabel'], fontsize=font_big)
		ax[j].set_ylabel(style['ylabel'], fontsize=font_big)
		ax[j].tick_params(axis='both', labelsize=font_little)
		# ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		if style['legend'][count] != None:
			if style['legend_line'] == 'OFF':
				ax[j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax[j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		if style['title'][count] != None:
			ax[j].set_title(style['title'][count], fontsize=font_big)
		ax[j].yaxis.offsetText.set_fontsize(font_offset)
		if j >= 0:
			if style['customxlabels'] != None:
				# ax[i][j].set_xticklabels(style['xticklabels'])
				# ax[i][j].set_xticks(style['xticklabels'])
				ax[j].set_xticklabels(style['customxlabels'])
				ax[j].set_xticks(style['customxlabels'])
		if style['ylim'] != None:
			# ax[j].set_ylim(bottom=style['ylim'][count][0], top=style['ylim'][count][1])
			ax[j].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])				
		if style['xlim'] != None:
			# ax[j].set_xlim(left=style['xlim'][count][0], right=style['xlim'][count][1])
			ax[j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
		ax[j].grid(axis='both')
		count += 1
		# ax[2].set_xticks(three_signals['dom'])
	
		# ax[0].set_xticklabels(style['xticklabels']) 
	
	#Visibility
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
		ax_it.yaxis.offsetText.set_visible(True)

	ax[1].tick_params(labelleft=True)

	# ax[0][1].set_yticklabels([-15, 0, 15, 30])
	# ax[0][1].set_yticks([-15, 0, 15, 30])
	
	
	#Set Limits in Axis X
	
		
	
		
	# ax[0].set_yticks([1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
	# ax[1].set_yticks([1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9])
	# ax[2].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	
	# ax[0].set_yticks([-10, 10, 30, 50])	
	# ax[1].set_yticks([-10, 10, 30, 50])
	
	# ax[0].set_yticks([-10 ,0, 10, 20, 30, 40, 50])
	# ax[1].set_yticks([-10 ,0, 10, 20, 30, 40, 50])
	
	
	# ax[2].set_yticks([-50, -30, -10, 10, 30])	
	if style['customylabels'] != None:
		ax[0].set_yticks(style['customylabels'])
		ax[1].set_yticks(style['customylabels'])
	# ax[1].set_xticks([1, 2, 3, 4, 5, 6])
	# ax[2].set_xticks([1, 2, 3, 4, 5, 6])
	
	# ax[0].set_xticks([0, 100, 200, 300, 400, 500])
	# ax[1].set_xticks([0, 100, 200, 300, 400, 500])
	# ax[2].set_xticks([0, 100, 200, 300, 400, 500])
	
	# ax[0].set_xticklabels([1, 2, 4, 7, 9, 10])
	# ax[1].set_xticklabels([1, 2, 4, 7, 9, 10])
	# ax[2].set_xticklabels([1, 2, 4, 7, 9, 10])
	
	
	# plt.tight_layout()
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

def myplot_map_double_join(map1, map2, style):
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=2, nrows=1)

	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	font_caption = 23+3
	lim = 3

	# map1[5,5] = 1.
	
	# plt.subplots_adjust(wspace=0.33, left=0.08, right=0.95, bottom=0.213, top=0.89)
	# fig.set_size_inches(12.2, 4.0)
	
	plt.subplots_adjust(wspace=0.33, left=0.08, right=0.82, bottom=0.213, top=0.89)
	fig.set_size_inches(12.2, 4.0)
	
	# plt.subplots_adjust(wspace=0.23, left=0.08, right=0.95, bottom=0.213, top=0.89)
	# fig.set_size_inches(13, 4.0)
	
	fig.text(0.05, 0.03, '(a)', fontsize=font_caption)
	# fig.text(0.55, 0.03, '(b)', fontsize=font_caption)
	fig.text(0.47, 0.03, '(b)', fontsize=font_caption)
	
	
	max1 = np.max(map1)
	max2 = np.max(map2)
	if max1 >= max2:
		print('max1')
		idx = 0
		max = max1
	elif max1 < max2:
		print('max2')
		idx = 1
		max = max2
	
	# Contours = ['0', '2']
	# ax = ['0', '2']
	for j in range(2):
		
		# num = '12' + str(j+1)
		# print(int(num))
		# ax[j] = fig.add_subplot(int(num))
		extent_ = style['extent']
		if j == 0:
			# map1[5,5] = 1.
			c1 = ax[j].contourf(map1, extent=extent_, cmap=style['colormap'], vmax=max)
			# cbar1 = fig.colorbar(c1, ax=ax[j])
			# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
			# cbar1.ax.tick_params(labelsize=font_little) 		
			# cbar1.ax.yaxis.get_offset_text().set(size=font_little)
		elif j == 1:
			c2 = ax[j].contourf(map2, extent=extent_, cmap=style['colormap'], vmax=max)
			# cbar2 = fig.colorbar(c2, ax=ax[j])
			# cbar2.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
			# cbar2.ax.tick_params(labelsize=font_little) 		
			# cbar2.ax.yaxis.get_offset_text().set(size=font_little)
		# extent_ = style['extent']
		# # contour_ = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])
		# Contours[j] = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])		
		# cbar = fig.colorbar(contour_, ax=ax[j])
		
		
		# cbar = fig.colorbar(contour_, ax=ax[j])		
		# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
		# cbar.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		# cbar.ax.tick_params(labelsize=font_little) 		
		# cbar.ax.yaxis.get_offset_text().set(size=font_little)
		
		ax[j].set_xlabel(style['xlabel'], fontsize=font_big)
		ax[j].set_ylabel(style['ylabel'], fontsize=font_big)

		
		#Size labels from axis
		ax[j].tick_params(axis='both', labelsize=font_little)
			
		#Scientific notation	
		ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		
		# #Visibility
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
			# ax_it.yaxis.offsetText.set_visible(True)

		#Eliminate line from label
		if style['legend'] != None:
			if style['legend_line'] == 'OFF':
				ax[j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax[j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		
		#Title
		if style['title'] != None:
			# plt.rcParams['mathtext.fontset'] = 'cm'
			# ax.set_title(keys[0], fontsize=font_big)	
			ax[j].set_title(style['title'][j], fontsize=font_big)
			
		#Size from offset text
		ax[j].yaxis.offsetText.set_fontsize(font_offset)
		

		
		
			
		
		
		#Set Limits in Axis X
		if style['ylim'] != None:
			ax[j].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
		
		if style['xlim'] != None:
			ax[j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			
		#Set Vertical Lines

		#Set Ticks in Axis X
		if style['xticklabels'] != None:
			print('+++++++++++++++')
			ax[j].set_xticks(style['xticklabels']) 		
			ax[j].set_xticklabels(style['xticklabels']) 
	
		ax[j].grid(axis='both')
		
	
	# cbar1 = fig.colorbar(Contours[0], ax=ax[0])

	# cbar2 = fig.colorbar(Contours[1], ax=ax[1])
	
	# cbar0 = fig.colorbar(Contours[0], ax=ax[0])		
	# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	# cbar0.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	# cbar0.ax.tick_params(labelsize=font_little) 		
	# cbar0.ax.yaxis.get_offset_text().set(size=font_little)
	
	# cbar1 = fig.colorbar(Contours[1], ax=ax[1])		
	# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	# cbar1.ax.tick_params(labelsize=font_little) 		
	# cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	cbar_ax = fig.add_axes([0.875, 0.2, 0.025, 0.7])
	if idx == 0:	
		cbar1 = fig.colorbar(c1, cax=cbar_ax, orientation='vertical')
		# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		cbar1.set_label(style['cbar_title'], fontsize=font_big)
		cbar1.ax.tick_params(labelsize=font_little) 		
		cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	else:
		cbar1 = fig.colorbar(c2, cax=cbar_ax, orientation='vertical')
		# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		cbar1.set_label(style['cbar_title'], fontsize=font_big)
		cbar1.ax.tick_params(labelsize=font_little) 		
		cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

	
def myplot_map_four_join_varb(map1, map2, style):
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=2, nrows=2)

	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	font_caption = 23+3
	lim = 3

	# map1[5,5] = 1.
	

	
	# plt.subplots_adjust(hspace=0.48, wspace=0.33, left=0.08, right=0.82, bottom=0.13, top=0.94)
	plt.subplots_adjust(hspace=0.45, wspace=0.125, left=0.08, right=0.82, bottom=0.13, top=0.94)
	fig.set_size_inches(12.2, 7.5)
	
	# plt.subplots_adjust(wspace=0.23, left=0.08, right=0.95, bottom=0.213, top=0.89)
	# fig.set_size_inches(13, 4.0)
	
	fig.text(0.05, 0.03, '(b)', fontsize=font_caption)
	# fig.text(0.47, 0.03, '(d)', fontsize=font_caption)
	
	fig.text(0.05, 0.53, '(a)', fontsize=font_caption)
	# fig.text(0.47, 0.53, '(b)', fontsize=font_caption)
	
	
	max1 = np.max(map1)
	max2 = np.max(map2)
	if max1 >= max2:
		print('max1')
		idx = 0
		max = max1
	elif max1 < max2:
		print('max2')
		idx = 1
		max = max2
	
	# Contours = ['0', '2']
	# ax = ['0', '2']
	for j in range(2):
		
		# num = '12' + str(j+1)
		# print(int(num))
		# ax[j] = fig.add_subplot(int(num))
		extent_ = style['extent']
		if j == 0:
			# map1[5,5] = 1.
			c1 = ax[j][0].contourf(map1, extent=extent_, cmap=style['colormap'], vmax=max)
			c1 = ax[j][1].contourf(map1, extent=extent_, cmap=style['colormap'], vmax=max)
			# cbar1 = fig.colorbar(c1, ax=ax[j])
			# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
			# cbar1.ax.tick_params(labelsize=font_little) 		
			# cbar1.ax.yaxis.get_offset_text().set(size=font_little)
		elif j == 1:
			c2 = ax[j][0].contourf(map2, extent=extent_, cmap=style['colormap'], vmax=max)
			c2 = ax[j][1].contourf(map2, extent=extent_, cmap=style['colormap'], vmax=max)
			# cbar2 = fig.colorbar(c2, ax=ax[j])
			# cbar2.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
			# cbar2.ax.tick_params(labelsize=font_little) 		
			# cbar2.ax.yaxis.get_offset_text().set(size=font_little)
		# extent_ = style['extent']
		# # contour_ = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])
		# Contours[j] = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])		
		# cbar = fig.colorbar(contour_, ax=ax[j])
		
		
		# cbar = fig.colorbar(contour_, ax=ax[j])		
		# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
		# cbar.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		# cbar.ax.tick_params(labelsize=font_little) 		
		# cbar.ax.yaxis.get_offset_text().set(size=font_little)
		
		ax[j][0].set_xlabel(style['xlabel'], fontsize=font_big)
		ax[j][1].set_xlabel(style['xlabel'], fontsize=font_big)
		
		ax[j][0].set_ylabel(style['ylabel'], fontsize=font_big)
		# ax[j][1].set_ylabel(style['ylabel'], fontsize=font_big)

		
		#Size labels from axis
		ax[j][0].tick_params(axis='both', labelsize=font_little)
		ax[j][1].tick_params(axis='both', labelsize=font_little)
			
		#Scientific notation	
		ax[j][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		ax[j][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		
		# #Visibility
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
			# ax_it.yaxis.offsetText.set_visible(True)

		#Eliminate line from label
		if style['legend'] != None:
			if style['legend_line'] == 'OFF':
				ax[j][0].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
				ax[j][1].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax[j][0].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
				ax[j][1].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		
		#Title
		if style['title'] != None:
			# plt.rcParams['mathtext.fontset'] = 'cm'
			# ax.set_title(keys[0], fontsize=font_big)	
			ax[j][0].set_title(style['title'][j], fontsize=font_big)
			# ax[j][1].set_title(style['title'][j], fontsize=font_big)
			
		#Size from offset text
		ax[j][0].yaxis.offsetText.set_fontsize(font_offset)
		ax[j][1].yaxis.offsetText.set_fontsize(font_offset)
		

		
		
			
		
		
		#Set Limits in Axis X
		if style['ylim'] != None:
			ax[j][0].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
			ax[j][1].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
		
		if style['xlim'] != None:
			ax[j][0].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			ax[j][1].set_xlim(left=style['xlim'][2], right=style['xlim'][3])
			
		#Set Vertical Lines

		#Set Ticks in Axis X
		if style['xticklabels'] != None:
			print('+++++++++++++++')
			ax[j][0].set_xticks(style['xticklabels']) 		
			ax[j][0].set_xticklabels(style['xticklabels']) 
			
			ax[j][1].set_xticks(style['xticklabels']) 		
			ax[j][1].set_xticklabels(style['xticklabels']) 
	
		ax[j][0].grid(axis='both')
		ax[j][1].grid(axis='both')
		
	
	# cbar1 = fig.colorbar(Contours[0], ax=ax[0])

	# cbar2 = fig.colorbar(Contours[1], ax=ax[1])
	
	# cbar0 = fig.colorbar(Contours[0], ax=ax[0])		
	# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	# cbar0.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	# cbar0.ax.tick_params(labelsize=font_little) 		
	# cbar0.ax.yaxis.get_offset_text().set(size=font_little)
	
	# cbar1 = fig.colorbar(Contours[1], ax=ax[1])		
	# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	# cbar1.ax.tick_params(labelsize=font_little) 		
	# cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	
	plt.setp(ax[1,1].get_yticklabels(), visible=False)
	plt.setp(ax[0,1].get_yticklabels(), visible=False)
	
	# plt.annotate(s='AE-35', xy=[0.8, 0.5], fontsize=font_big)
	# plt.annotate(s='AE-63', xy=[0.1, 0.1], fontsize=font_big)
	# plt.annotate(s='AE-38', xy=[1.5, 1.5], fontsize=font_big)
	# plt.annotate(s='AE-9', xy=[50, 50], fontsize=font_big)
	# ax[0][0].annotate(s='AE-3', xy=[0.8, 0.5], fontsize=font_big)
	# ax[0][0].annotate(s='AE-377', xy=[20, 500], fontsize=font_big)
	
	cbar_ax = fig.add_axes([0.875, 0.2, 0.025, 0.7])
	if idx == 0:	
		cbar1 = fig.colorbar(c1, cax=cbar_ax, orientation='vertical')
		# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		cbar1.set_label(style['cbar_title'], fontsize=font_big)
		cbar1.ax.tick_params(labelsize=font_little) 		
		cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	else:
		cbar1 = fig.colorbar(c2, cax=cbar_ax, orientation='vertical')
		# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		cbar1.set_label(style['cbar_title'], fontsize=font_big)
		cbar1.ax.tick_params(labelsize=font_little) 		
		cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

def myplot_map_four_join(map1, map2, style):
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=2, nrows=2)

	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	font_caption = 23+3
	lim = 3

	# map1[5,5] = 1.
	

	
	plt.subplots_adjust(hspace=0.48, wspace=0.33, left=0.08, right=0.82, bottom=0.13, top=0.94)
	# plt.subplots_adjust(hspace=0.48, wspace=0.15, left=0.08, right=0.82, bottom=0.13, top=0.94)
	fig.set_size_inches(12.2, 7.5)
	
	# plt.subplots_adjust(wspace=0.23, left=0.08, right=0.95, bottom=0.213, top=0.89)
	# fig.set_size_inches(13, 4.0)
	
	fig.text(0.05, 0.03, '(c)', fontsize=font_caption)
	fig.text(0.47, 0.03, '(d)', fontsize=font_caption)
	
	fig.text(0.05, 0.52, '(a)', fontsize=font_caption)
	fig.text(0.47, 0.52, '(b)', fontsize=font_caption)
	
	
	max1 = np.max(map1)
	max2 = np.max(map2)
	if max1 >= max2:
		print('max1')
		idx = 0
		max = max1
	elif max1 < max2:
		print('max2')
		idx = 1
		max = max2
	
	# Contours = ['0', '2']
	# ax = ['0', '2']
	for j in range(2):
		
		# num = '12' + str(j+1)
		# print(int(num))
		# ax[j] = fig.add_subplot(int(num))
		extent_ = style['extent']
		if j == 0:
			# map1[5,5] = 1.
			c1 = ax[j][0].contourf(map1, extent=extent_, cmap=style['colormap'], vmax=max)
			c1 = ax[j][1].contourf(map1, extent=extent_, cmap=style['colormap'], vmax=max)
			# cbar1 = fig.colorbar(c1, ax=ax[j])
			# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
			# cbar1.ax.tick_params(labelsize=font_little) 		
			# cbar1.ax.yaxis.get_offset_text().set(size=font_little)
		elif j == 1:
			c2 = ax[j][0].contourf(map2, extent=extent_, cmap=style['colormap'], vmax=max)
			c2 = ax[j][1].contourf(map2, extent=extent_, cmap=style['colormap'], vmax=max)
			# cbar2 = fig.colorbar(c2, ax=ax[j])
			# cbar2.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
			# cbar2.ax.tick_params(labelsize=font_little) 		
			# cbar2.ax.yaxis.get_offset_text().set(size=font_little)
		# extent_ = style['extent']
		# # contour_ = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])
		# Contours[j] = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])		
		# cbar = fig.colorbar(contour_, ax=ax[j])
		
		
		# cbar = fig.colorbar(contour_, ax=ax[j])		
		# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
		# cbar.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		# cbar.ax.tick_params(labelsize=font_little) 		
		# cbar.ax.yaxis.get_offset_text().set(size=font_little)
		
		ax[j][0].set_xlabel(style['xlabel'], fontsize=font_big)
		ax[j][1].set_xlabel(style['xlabel'], fontsize=font_big)
		
		ax[j][0].set_ylabel(style['ylabel'], fontsize=font_big)
		ax[j][1].set_ylabel(style['ylabel'], fontsize=font_big)

		
		#Size labels from axis
		ax[j][0].tick_params(axis='both', labelsize=font_little)
		ax[j][1].tick_params(axis='both', labelsize=font_little)
			
		#Scientific notation	
		ax[j][0].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		ax[j][1].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		
		# #Visibility
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
			# ax_it.yaxis.offsetText.set_visible(True)

		#Eliminate line from label
		if style['legend'] != None:
			if style['legend_line'] == 'OFF':
				ax[j][0].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
				ax[j][1].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax[j][0].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
				ax[j][1].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		
		#Title
		if style['title'] != None:
			# plt.rcParams['mathtext.fontset'] = 'cm'
			# ax.set_title(keys[0], fontsize=font_big)	
			ax[j][0].set_title(style['title'][j], fontsize=font_big)
			ax[j][1].set_title(style['title'][j], fontsize=font_big)
			
		#Size from offset text
		ax[j][0].yaxis.offsetText.set_fontsize(font_offset)
		ax[j][1].yaxis.offsetText.set_fontsize(font_offset)
		

		
		
			
		
		
		#Set Limits in Axis X
		if style['ylim'] != None:
			ax[j][0].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
			ax[j][1].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
		
		if style['xlim'] != None:
			ax[j][0].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			ax[j][1].set_xlim(left=style['xlim'][2], right=style['xlim'][3])
			
		#Set Vertical Lines

		#Set Ticks in Axis X
		if style['xticklabels'] != None:
			print('+++++++++++++++')
			ax[j][0].set_xticks(style['xticklabels']) 		
			ax[j][0].set_xticklabels(style['xticklabels']) 
			
			ax[j][1].set_xticks(style['xticklabels']) 		
			ax[j][1].set_xticklabels(style['xticklabels'])
		
		if style['yticklabels'] != None:
			print('+++++++++++++++')
			ax[j][0].set_yticks(style['yticklabels']) 		
			ax[j][0].set_yticklabels(style['yticklabels']) 
			
			ax[j][1].set_yticks(style['yticklabels']) 		
			ax[j][1].set_yticklabels(style['yticklabels']) 
	
		ax[j][0].grid(axis='both')
		ax[j][1].grid(axis='both')
		
	
	# cbar1 = fig.colorbar(Contours[0], ax=ax[0])

	# cbar2 = fig.colorbar(Contours[1], ax=ax[1])
	
	# cbar0 = fig.colorbar(Contours[0], ax=ax[0])		
	# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	# cbar0.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	# cbar0.ax.tick_params(labelsize=font_little) 		
	# cbar0.ax.yaxis.get_offset_text().set(size=font_little)
	
	# cbar1 = fig.colorbar(Contours[1], ax=ax[1])		
	# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	# cbar1.ax.tick_params(labelsize=font_little) 		
	# cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	
	# plt.setp(ax[1,1].get_yticklabels(), visible=False)
	# plt.setp(ax[0,1].get_yticklabels(), visible=False)
	
	cbar_ax = fig.add_axes([0.875, 0.2, 0.025, 0.7])
	if idx == 0:	
		cbar1 = fig.colorbar(c1, cax=cbar_ax, orientation='vertical')
		# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		cbar1.set_label(style['cbar_title'], fontsize=font_big)
		cbar1.ax.tick_params(labelsize=font_little) 		
		cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	else:
		cbar1 = fig.colorbar(c2, cax=cbar_ax, orientation='vertical')
		# cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		cbar1.set_label(style['cbar_title'], fontsize=font_big)
		cbar1.ax.tick_params(labelsize=font_little) 		
		cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return
	
def myplot_map_double_ind(map, style):
	# #Modules and global properties
	# from matplotlib import font_manager
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=2, nrows=1)
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	font_caption = 23+3
	lim = 3

	# plt.subplots_adjust(left=0.15, right=0.93, bottom=0.165, top=0.92)
	
	
	plt.subplots_adjust(wspace=0.33, left=0.08, right=0.95, bottom=0.213, top=0.89)
	# fig.set_size_inches(10.2, 4.0)
	fig.set_size_inches(12.2, 4.0)
	fig.text(0.05, 0.03, '(a)', fontsize=font_caption)
	fig.text(0.55, 0.03, '(b)', fontsize=font_caption)
	
	Contours = ['0', '2']
	for j in range(2):
	
		ax[j].set_xlabel(style['xlabel'], fontsize=font_big)
		

		extent_ = style['extent']
		# contour_ = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])
		Contours[j] = ax[j].contourf(map, extent=extent_, cmap=style['colormap'])
		
		
		# cbar = fig.colorbar(contour_, ax=ax[j])		
		# # cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
		# cbar.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
		# cbar.ax.tick_params(labelsize=font_little) 		
		# cbar.ax.yaxis.get_offset_text().set(size=font_little)
		
		
		ax[j].set_ylabel(style['ylabel'], fontsize=font_big)

		
		#Size labels from axis
		ax[j].tick_params(axis='both', labelsize=font_little)
			
		#Scientific notation	
		ax[j].ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
		
		# #Visibility
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
			# ax_it.yaxis.offsetText.set_visible(True)

		#Eliminate line from label
		if style['legend'] != None:
			if style['legend_line'] == 'OFF':
				ax[j].legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
			else:
				ax[j].legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
		
		#Title
		if style['title'] != None:
			# plt.rcParams['mathtext.fontset'] = 'cm'
			# ax.set_title(keys[0], fontsize=font_big)	
			ax[j].set_title(style['title'][j], fontsize=font_big)
			
		#Size from offset text
		ax[j].yaxis.offsetText.set_fontsize(font_offset)
		

		
		
			
		
		
		#Set Limits in Axis X
		if style['ylim'] != None:
			ax[j].set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
		
		if style['xlim'] != None:
			ax[j].set_xlim(left=style['xlim'][0], right=style['xlim'][1])
			
		#Set Vertical Lines

		#Set Ticks in Axis X
		if style['xticklabels'] != None:
			print('+++++++++++++++')
			ax[j].set_xticks(style['xticklabels']) 		
			ax[j].set_xticklabels(style['xticklabels']) 
	
		ax[j].grid(axis='both')
		
		
	
	cbar0 = fig.colorbar(Contours[0], ax=ax[1])		
	# cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	cbar0.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	cbar0.ax.tick_params(labelsize=font_little) 		
	cbar0.ax.yaxis.get_offset_text().set(size=font_little)
	
	cbar1 = fig.colorbar(Contours[1], ax=ax[0])		
	# cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	cbar1.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	cbar1.ax.tick_params(labelsize=font_little) 		
	cbar1.ax.yaxis.get_offset_text().set(size=font_little)
	
	
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return

def myplot_map(map, style):
	# #Modules and global properties
	# from matplotlib import font_manager
	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	# plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	fig, ax = plt.subplots(ncols=1, nrows=1)
	
	#Values Fixed
	amp_factor = 1
	font_big = (17+2+1)*amp_factor
	font_little = (15+2+1)*amp_factor
	font_label = (13+2+1)*amp_factor
	font_offset = (15+2+1)*amp_factor
	lim = 3

	plt.subplots_adjust(left=0.15, right=0.93, bottom=0.165, top=0.92)
	
	
	
	ax.set_xlabel(style['xlabel'], fontsize=font_big)
	

	extent_ = style['extent']
	contour_ = ax.contourf(map, extent=extent_, cmap=style['colormap'])
	
	
	cbar = fig.colorbar(contour_, ax=ax)
	
	# cbar.set_label('SCD [V$^{2}$]', fontsize=font_big)
	# cbar.set_label(r'$S_{xx}(f,\alpha)$ [mV$^{2}$]', fontsize=font_big)
	cbar.set_label(r'$\gamma_{xx}(f,\alpha)$ [-]', fontsize=font_big)
	
	# cbar.set_label('SC [-]', fontsize=font_big)
	# cbar.set_label('Energy [V$^{2}$]', fontsize=font_big)
	# cbar.set_label('Magnitude [V]', fontsize=font_big)
	# cbar.set_ticks([0, 200, 400, 600, 800])
	# cbar.set_ticks([0, 1200, 2400, 3600, 4800])
	cbar.ax.tick_params(labelsize=font_little) 
	
	cbar.ax.yaxis.get_offset_text().set(size=font_little)
	
	
	ax.set_ylabel(style['ylabel'], fontsize=font_big)

	
	#Size labels from axis
	ax.tick_params(axis='both', labelsize=font_little)
		
	#Scientific notation	
	ax.ticklabel_format(axis='y', style='sci', scilimits=(-lim, lim))
	
	# #Visibility
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
		# ax_it.yaxis.offsetText.set_visible(True)

	#Eliminate line from label
	if style['legend'] != None:
		if style['legend_line'] == 'OFF':
			ax.legend(loc=style['loc_legend'], handlelength=0, handletextpad=0, fancybox=True, fontsize=font_label)
		else:
			ax.legend(loc=style['loc_legend'], fontsize=font_label, handletextpad=0.3, labelspacing=.3)
	
	#Title
	if style['title'] != None:
		# plt.rcParams['mathtext.fontset'] = 'cm'
		# ax.set_title(keys[0], fontsize=font_big)	
		ax.set_title(style['title'], fontsize=font_big)
	#Size from offset text
	ax.yaxis.offsetText.set_fontsize(font_offset)
	

	
	
		
	
	
	#Set Limits in Axis X
	if style['ylim'] != None:
		ax.set_ylim(bottom=style['ylim'][0], top=style['ylim'][1])
	
	if style['xlim'] != None:
		ax.set_xlim(left=style['xlim'][0], right=style['xlim'][1])
		
	#Set Vertical Lines

	#Set Ticks in Axis X
	if style['xticklabels'] != None:
		print('+++++++++++++++')
		ax.set_xticks(style['xticklabels']) 		
		ax.set_xticklabels(style['xticklabels']) 
	
	ax.grid(axis='both')
	if style['output'] == 'plot':
		plt.show()
	elif style['output'] == 'save':
		plt.savefig(style['path_1'])
		plt.savefig(style['path_2'])
	return


if __name__ == '__main__':
	main(sys.argv)
