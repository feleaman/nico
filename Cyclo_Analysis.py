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

from mpl_toolkits.mplot3d import Axes3D
#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode', 'channel']
InputsOpt_Defaults = {'power2':'OFF', 'projection':'surface', 'save':'ON'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	if config['mode'] == 'plot_from_mat':
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data = load_signal(filepath, channel=config['channel'])
		
		# data = data*(1000/70.8)*(1000/70.8)*1000*1000
		# data = data*(1000/141.25)*(1000/141.25)*1000*1000
		
		
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
		data[:,0] = 0.
		
		fig = plt.figure()
		
		if config['projection'] == 'surface':
			ax = fig.add_subplot(111, projection='3d')			
			alpha, f = np.meshgrid(alpha, f)			
			ax.plot_surface(alpha, f/1000, data, cmap='plasma')	
			# ax.plot_wireframe(alpha, f/1000, data, color='black')
			# ax.plot_trisurf(alpha, f/1000, data, cmap='viridis')	
			
		# elif config['projection'] == 'mesh':
			# ax = fig.add_subplot(111)	
			# dog = ax.pcolormesh(alpha, f/1000., np.log(data), cmap='inferno', vmax=None)
			# cbar = fig.colorbar(dog, ax=ax)
			# cbar.set_label('SCoh [-]', fontsize=13)
			# cbar.ax.tick_params(axis='both', labelsize=12)
			# # cbar.ax.set_ylim(bottom=0.000001, top=20)
			
		elif config['projection'] == 'mesh':
			ax = fig.add_subplot(111)
			# vmax = max_cspectrum(data)
			# dog = ax.pcolormesh(alpha, f/1000, 10*np.log10(data), cmap='inferno', vmax=None)
			dog = ax.pcolormesh(alpha, f/1000, data, cmap='inferno', vmax=None)

			cbar = fig.colorbar(dog, ax=ax)			
			# cbar.set_label('SCD [mV$^{2}$]', fontsize=13)
			cbar.set_label('SCoh [-]', fontsize=13)
			cbar.ax.tick_params(axis='both', labelsize=12)
			# cbar.ax.set_ylim(bottom=0.000001, top=20)
			
		elif config['projection'] == 'save_map':
			extent_ = [0, np.max(alpha), 0, 500]
			# ax.contourf(cwtmatr, extent=extent_)
			# ax.set_ylim(bottom=0 , top=500000)		
			
			# plt.show()
			
			mydict = {'map':data, 'extent':extent_}
			save_pickle('CycloDensity_simulated_AE_mod_fault_frec.pkl', mydict)
			sys.exit()
			
			
		ax.set_xlabel('Cyclic Frequency [Hz]', fontsize=13)
		ax.set_ylabel('Frequency [kHz]', fontsize=13)
		ax.tick_params(axis='both', labelsize=12)
		ax.set_title('S-1: AE-3', fontsize=12)
		# ax.set_xlim(left=0, right=50)
		# ax.set_ylim(bottom=0, top=500)
		plt.show()
	
	elif config['mode'] == 'plot_from_mat_cyclic':
		
		
		print('Select MAT from DATA') 
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		data = load_signal(filepath, channel=config['channel'])
		
		# data = data*(1000/70.8)*(1000/70.8)*1000*1000
		# data = data*(1000/141.25)*(1000/141.25)*1000*1000
		
		
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
		
		sum = np.zeros(len(data[:,0])) #sum len 100
		for i in range(len(data[0,:])): #i itera de 1 a 100
			sum += data[:,i] #data suma a len 100
		plt.plot(f/1000, sum)
		plt.show()
		
		sum = np.zeros(len(data[0,:])) #sum len 100
		for i in range(len(data[:,0])): #i itera de 1 a 100
			sum += data[i,:] #data suma a len 100
		plt.plot(alpha, sum)
		plt.show()

		sys.exit()
	
	elif config['mode'] == 'avg_plot_from_mat_cyclic':
		
		for i in range(5):
		
			print('Select MAT from DATA') 
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()		
			data = load_signal(filepath, channel=config['channel'])
			
			# data = data*(1000/70.8)*(1000/70.8)*1000*1000
			# data = data*(1000/141.25)*(1000/141.25)*1000*1000
			
			
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
			
			sum = np.zeros(len(data[:,0])) #sum len 100
			for i in range(len(data[0,:])): #i itera de 1 a 100
				sum += data[:,i] #data suma a len 100
			plt.plot(f/1000, sum)
			plt.show()
			
			sum = np.zeros(len(data[0,:])) #sum len 100
			for i in range(len(data[:,0])): #i itera de 1 a 100
				sum += data[i,:] #data suma a len 100
			plt.plot(alpha, sum)
			plt.show()

			sys.exit()
		
		
		
	
	elif config['mode'] == 'comp_plot_from_mat':
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
		
		select_alpha = 4.3 - 2
		da = alpha[1] - alpha[0]
		result_alpha_idx = int(round(select_alpha/da))
		result_alpha = alpha[result_alpha_idx]
		
		plt.plot(f, data[:, result_alpha_idx])
		plt.show()
		
		print(result_alpha)
		print(alpha)
		sys.exit()
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		alpha, f = np.meshgrid(alpha, f)		
		
		ax.plot_surface(alpha, f, data, cmap='plasma')
		
		plt.show()
	
	elif config['mode'] == 'multi_comp_plot_from_mat':
		prom = 1
		for i in range(prom):
			print('Select MAT from DATA')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()		
			data = load_signal(filepath, channel=config['channel'])
			# data = data*(1000/70.8)*(1000/70.8)*1000*1000
			# data = data*(1000/141.25)*(1000/141.25)*1000*1000
			
			# data = 10*np.log(data)
			
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
			
			#++++++++TEST RIG
			# select_alphas = np.array([4.333, 4.333*2, 4.333*3, 4.333*4, 4.333*5, 4.333*6, 4.333*7, 4.333*8, 4.333*9, 4.333*10])
			# select_alphas = np.array([312., 624., 936., 1248., 1560.])
			
			# select_alphas = np.array([307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] +  [619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65] + [931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65] + [1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35] + [1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65])*1.023077
			
			# select_alphas = [4.333, 4.333*2, 4.333*3, 4.333*4, 4.333*5, 4.333*6, 4.333*7, 4.333*8, 4.333*9, 4.333*10]
			
			# select_alphas += [312., 624., 936., 1248., 1560.]
			# select_alphas += [307.67, 316.33, 303.34, 320.66, 294.68, 329.32, 290.35, 333.65] +  [619.67, 628.33, 615.34, 632.66, 606.68, 641.32, 602.35, 645.65] + [931.67, 940.33, 927.34, 944.66, 918.68, 953.32, 914.35, 957.65] + [1243.67, 1252.33, 1239.34, 1256.66, 1230.68, 1265.32, 1226.35, 1269.35] + [1555.67, 1564.33, 1551.34, 1568.66, 1542.68, 1577.32, 1538.35, 1581.65]
			# select_alphas = np.array(select_alphas)
			
			# *1.023077
			
			
			# #++++++++CWD DAMAGE
			# # select_alphas = [35.8875, 71.775, 107.6625, 143.55, 179.4375] #fg
			# select_alphas = [0.3625, 0.725, 1.0875, 1.45, 1.8125, 2.175, 2.5375, 2.9, 3.2625, 3.625] #ffr
			# # select_alphas = [0.3625] #ffr
			
			# #++++++++TURM Schottland DAMAGE
			# select_alphas = [3.872, 7.744, 11.615, 15.487, 19.359, 23.231, 27.103, 30.974, 34.847, 38.718] #ffr
			
			#++++++++BOCHUM NO DAMAGE
			select_alphas = [3.481, 6.961, 10.442, 13.923, 17.403, 20.884, 24.365, 27.485, 31.326, 34.806] #ffr
			
			da = alpha[1] - alpha[0]		
			
			result_alpha_idxes = []
			if i == 0:
				PromsComps = np.zeros(len(data[:, 0]))
			Comps = np.zeros(len(data[:, 0]))
			for select_alpha in select_alphas:		
				result_alpha_idx = int(round(select_alpha/da))
				result_alpha_idxes.append(result_alpha_idx)
				print(alpha[result_alpha_idx])
				Comps = Comps + data[:, result_alpha_idx]
			PromsComps = PromsComps + Comps
			
		PromsComps = PromsComps / prom
		
		fig, ax = plt.subplots()
		ax.plot(f/1000., PromsComps)
		ax.set_xlabel('Frequency [kHz]', fontsize=13)
		# name_ylabel = r'$\Sigma_{i=1}^{10}$ SCD $i f_{c}$ [$\mu$V$^{2}$]'
		name_ylabel = r'$\Sigma_{i=1}^{10}$ SCoh $i f_{c}$ [-]'
		ax.set_ylabel(name_ylabel, fontsize=13)
		ax.tick_params(axis='both', labelsize=12)
		ax.set_title('B-3: AE-3', fontsize=12)
		plt.show()
		
		
		
		
		if config['save'] == 'ON':
			# #TEST-RIG
			# save_pickle('MessQ_MeanDensity_5h_sb_f_g_c.pkl', PromsComps)
			# save_pickle('MessQ_fDensity_5h_sb_f_g_c.pkl', f)
			
			# #CWD-Damage
			# save_pickle('CWD_Damage_MeanCoherence_10fft.pkl', PromsComps)
			# save_pickle('CWD_Damage_fCoherence_10ffr.pkl', f)
			
			#Schottland-Damage
			save_pickle('Schottland_Damage_MeanCoherence_10ffr.pkl', PromsComps)
			save_pickle('Schottland_Damage_fCoherence_10ffr.pkl', f)
		
		
	
	elif config['mode'] == 'plot_mean':
		print('Select FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f = read_pickle(filepath)
		
		print('Select MEAN DATA')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()		
		data = read_pickle(filepath)
		
		names = ['MessA', 'MessG', 'MessI', 'MessL', 'MessN', 'MessQ']
		signals = {}
		for name in names:
			for filepath in Filepaths:
				if filepath.find(name) != -1:
					if name not in signals.keys():
						signals[name] = read_pickle(filepath)
		
		
		font_big = 17
		font_little = 15
		font_label = 13
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.55)
		fig.set_size_inches(14.2, 6.2)		

		ax[0][0].plot(f/1000., signals['MessA'])
		
		ax[0][1].plot(f/1000., signals['MessG'])
		
		ax[0][2].plot(f/1000., signals['MessI'])		
		
		ax[1][0].plot(f/1000., signals['MessL'])
		
		ax[1][1].plot(f/1000., signals['MessN'])
		
		ax[1][2].plot(f/1000., signals['MessQ'])

		
		name_ylabel = r'$\Sigma$ SCD [-]'
		ax[0][0].set_ylabel(name_ylabel, fontsize=font_big)
		ax[0][1].set_ylabel(name_ylabel, fontsize=font_big)
		ax[0][2].set_ylabel(name_ylabel, fontsize=font_big)
		
		ax[1][0].set_ylabel(name_ylabel, fontsize=font_big)
		ax[1][1].set_ylabel(name_ylabel, fontsize=font_big)
		ax[1][2].set_ylabel(name_ylabel, fontsize=font_big)
		
		name_xlabel = 'Frequency [kHz]'
		ax[0][0].set_xlabel(name_xlabel, fontsize=font_big)
		ax[0][1].set_xlabel(name_xlabel, fontsize=font_big)
		ax[0][2].set_xlabel(name_xlabel, fontsize=font_big)
		
		ax[1][0].set_xlabel(name_xlabel, fontsize=font_big)
		ax[1][1].set_xlabel(name_xlabel, fontsize=font_big)
		ax[1][2].set_xlabel(name_xlabel, fontsize=font_big)
		
		# ax[0][0].legend(fontsize=font_label, ncol=2, loc='best')
		# ax[0][1].legend(fontsize=font_label, ncol=2, loc='best')
		# ax[0][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		# ax[1][0].legend(fontsize=font_label, ncol=2, loc='best')
		# ax[1][1].legend(fontsize=font_label, ncol=2, loc='best')
		# ax[1][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$1^{st}$ MC', fontsize=font_big)
		ax[0][1].set_title('$2^{nd}$ MC', fontsize=font_big)
		ax[0][2].set_title('$4^{th}$ MC', fontsize=font_big)
		
		ax[1][0].set_title('$7^{th}$ MC', fontsize=font_big)
		ax[1][1].set_title('$9^{th}$ MC', fontsize=font_big)
		ax[1][2].set_title('$10^{th}$ MC', fontsize=font_big)
		
		
		ax[0][0].set_xlim(left=0, right=500)
		ax[0][1].set_xlim(left=0, right=500)
		ax[0][2].set_xlim(left=0, right=500)
		
		ax[1][0].set_xlim(left=0, right=500)
		ax[1][1].set_xlim(left=0, right=500)
		ax[1][2].set_xlim(left=0, right=500)
		

		valtop = 0.4
		ax[0][0].set_ylim(bottom=0, top=valtop)
		ax[0][1].set_ylim(bottom=0, top=valtop)
		ax[0][2].set_ylim(bottom=0, top=valtop)
		
		ax[1][0].set_ylim(bottom=0, top=valtop)
		ax[1][1].set_ylim(bottom=0, top=valtop)
		ax[1][2].set_ylim(bottom=0, top=valtop)

		
		
		# ax[0][0].set_yticks([0, 100, 200, 300, 400, 500])
		# ax[0][1].set_yticks([0, 100, 200, 300, 400, 500])
		# ax[0][2].set_yticks([0, 100, 200, 300, 400, 500])
		
		# ax[1][0].set_yticks([0, 100, 200, 300, 400, 500])
		# ax[1][1].set_yticks([0, 100, 200, 300, 400, 500])
		# ax[1][2].set_yticks([0, 100, 200, 300, 400, 500])
		

		
		
		ax[0][0].tick_params(axis='both', labelsize=font_little)
		ax[0][1].tick_params(axis='both', labelsize=font_little)
		ax[0][2].tick_params(axis='both', labelsize=font_little)
		
		ax[1][0].tick_params(axis='both', labelsize=font_little)
		ax[1][1].tick_params(axis='both', labelsize=font_little)
		ax[1][2].tick_params(axis='both', labelsize=font_little)
		
		
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
			ax_it.yaxis.offsetText.set_visible(True)
		
		plt.show()
	
	elif config['mode'] == 'plot_mean_multi':
		print('Select FREQ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()		
		f = read_pickle(filepath)
		
		print('Select MEAN DATA')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()		
		data = read_pickle(filepath)
		
		names = ['MessA', 'MessG', 'MessI', 'MessL', 'MessN', 'MessQ']
		harmonics = ['10h_f_f_r', '5h_f_g', '5h_sb_f_g_c']
		signals = {}
		for name in names:
			for harmonic in harmonics:
				for filepath in Filepaths:
					if filepath.find(name) != -1 and filepath.find(harmonic) != -1:
						if name not in signals.keys():
							signals[name + '_' + harmonic] = read_pickle(filepath)
		
		
		font_big = 17 + 3
		font_little = 15 + 3 
		font_label = 13 + 3
		font_caption = 23+3
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)
		# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.55)
		plt.subplots_adjust(wspace=0.32, left=0.07, right=0.985, bottom=0.135, top=0.94, hspace=0.52)
		# fig.set_size_inches(14.2, 6.2)
		fig.set_size_inches(14.2, 7)
		
		fig.text(0.053-0.015, 0.04, '(d)', fontsize=font_caption)
		fig.text(0.385-0.015, 0.04, '(e)', fontsize=font_caption)
		fig.text(0.717-0.015, 0.04, '(f)', fontsize=font_caption)
		
		fig.text(0.053-0.015, 0.528, '(a)', fontsize=font_caption)
		fig.text(0.385-0.015, 0.528, '(b)', fontsize=font_caption)
		fig.text(0.717-0.015, 0.528, '(c)', fontsize=font_caption)
		
		
		
		ax[0][0].plot(f/1000., signals['MessA_10h_f_f_r'], label='$n f_{c}$')
		ax[0][0].plot(f/1000., signals['MessA_5h_f_g'], label='$p f_{m}$')
		ax[0][0].plot(f/1000., signals['MessA_5h_sb_f_g_c'], label='$p f_{m} \pm s f_{c}$')
		
		ax[0][1].plot(f/1000., signals['MessG_10h_f_f_r'], label='$n f_{c}$')
		ax[0][1].plot(f/1000., signals['MessG_5h_f_g'], label='$p f_{m}$')
		ax[0][1].plot(f/1000., signals['MessG_5h_sb_f_g_c'], label='$p f_{m} \pm s f_{c}$')
		
		ax[0][2].plot(f/1000., signals['MessI_10h_f_f_r'], label='$n f_{c}$')
		ax[0][2].plot(f/1000., signals['MessI_5h_f_g'], label='$p f_{m}$')
		ax[0][2].plot(f/1000., signals['MessI_5h_sb_f_g_c'], label='$p f_{m} \pm s f_{c}$')
		
		ax[1][0].plot(f/1000., signals['MessL_10h_f_f_r'], label='$n f_{c}$')
		ax[1][0].plot(f/1000., signals['MessL_5h_f_g'], label='$p f_{m}$')
		ax[1][0].plot(f/1000., signals['MessL_5h_sb_f_g_c'], label='$p f_{m} \pm s f_{c}$')
		
		ax[1][1].plot(f/1000., signals['MessN_10h_f_f_r'], label='$n f_{c}$')
		ax[1][1].plot(f/1000., signals['MessN_5h_f_g'], label='$p f_{m}$')
		ax[1][1].plot(f/1000., signals['MessN_5h_sb_f_g_c'], label='$p f_{m} \pm s f_{c}$')
		
		ax[1][2].plot(f/1000., signals['MessQ_10h_f_f_r'], label='$n f_{c}$')
		ax[1][2].plot(f/1000., signals['MessQ_5h_f_g'], label='$p f_{m}$')
		ax[1][2].plot(f/1000., signals['MessQ_5h_sb_f_g_c'], label='$p f_{m} \pm s f_{c}$')

		
		# name_ylabel = r'$\Sigma$ SCD [mV$^{2}$]'
		name_ylabel = r'$\Sigma$ SC [-]'
		
		ax[0][0].set_ylabel(name_ylabel, fontsize=font_big)
		ax[0][1].set_ylabel(name_ylabel, fontsize=font_big)
		ax[0][2].set_ylabel(name_ylabel, fontsize=font_big)
		
		ax[1][0].set_ylabel(name_ylabel, fontsize=font_big)
		ax[1][1].set_ylabel(name_ylabel, fontsize=font_big)
		ax[1][2].set_ylabel(name_ylabel, fontsize=font_big)
		
		name_xlabel = 'Frequency [kHz]'
		ax[0][0].set_xlabel(name_xlabel, fontsize=font_big)
		ax[0][1].set_xlabel(name_xlabel, fontsize=font_big)
		ax[0][2].set_xlabel(name_xlabel, fontsize=font_big)
		
		ax[1][0].set_xlabel(name_xlabel, fontsize=font_big)
		ax[1][1].set_xlabel(name_xlabel, fontsize=font_big)
		ax[1][2].set_xlabel(name_xlabel, fontsize=font_big)
		
		ax[0][0].legend(fontsize=font_label, ncol=1, loc='best', labelspacing=0.2, handletextpad=0.2)
		ax[0][1].legend(fontsize=font_label, ncol=1, loc='best', labelspacing=0.2, handletextpad=0.2)
		ax[0][2].legend(fontsize=font_label, ncol=1, loc='best', labelspacing=0.2, handletextpad=0.2)
		
		ax[1][0].legend(fontsize=font_label, ncol=1, loc='best', labelspacing=0.2, handletextpad=0.2)
		ax[1][1].legend(fontsize=font_label, ncol=1, loc='best', labelspacing=0.2, handletextpad=0.2)
		ax[1][2].legend(fontsize=font_label, ncol=1, loc='best', labelspacing=0.2, handletextpad=0.2)
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		# ax[0][0].set_title('$1^{st}$ MC', fontsize=font_big)
		# ax[0][1].set_title('$2^{nd}$ MC', fontsize=font_big)
		# ax[0][2].set_title('$4^{th}$ MC', fontsize=font_big)
		
		# ax[1][0].set_title('$7^{th}$ MC', fontsize=font_big)
		# ax[1][1].set_title('$9^{th}$ MC', fontsize=font_big)
		# ax[1][2].set_title('$10^{th}$ MC', fontsize=font_big)
		
		ax[0][0].set_title('MC 1', fontsize=font_big)
		ax[0][1].set_title('MC 2', fontsize=font_big)
		ax[0][2].set_title('MC 4', fontsize=font_big)
		
		ax[1][0].set_title('MC 7', fontsize=font_big)
		ax[1][1].set_title('MC 9', fontsize=font_big)
		ax[1][2].set_title('MC 10', fontsize=font_big)
		
		
		ax[0][0].set_xlim(left=0, right=500)
		ax[0][1].set_xlim(left=0, right=500)
		ax[0][2].set_xlim(left=0, right=500)
		
		ax[1][0].set_xlim(left=0, right=500)
		ax[1][1].set_xlim(left=0, right=500)
		ax[1][2].set_xlim(left=0, right=500)
		

		# valtop = 0.15
		valtop = 1.
		ax[0][0].set_ylim(bottom=0, top=valtop)
		ax[0][1].set_ylim(bottom=0, top=valtop)
		ax[0][2].set_ylim(bottom=0, top=valtop)
		
		ax[1][0].set_ylim(bottom=0, top=valtop)
		ax[1][1].set_ylim(bottom=0, top=valtop)
		ax[1][2].set_ylim(bottom=0, top=valtop)

		
		
		ax[0][0].set_xticks([0, 100, 200, 300, 400, 500])
		ax[0][1].set_xticks([0, 100, 200, 300, 400, 500])
		ax[0][2].set_xticks([0, 100, 200, 300, 400, 500])
		
		ax[1][0].set_xticks([0, 100, 200, 300, 400, 500])
		ax[1][1].set_xticks([0, 100, 200, 300, 400, 500])
		ax[1][2].set_xticks([0, 100, 200, 300, 400, 500])
		

		
		
		ax[0][0].tick_params(axis='both', labelsize=font_little)
		ax[0][1].tick_params(axis='both', labelsize=font_little)
		ax[0][2].tick_params(axis='both', labelsize=font_little)
		
		ax[1][0].tick_params(axis='both', labelsize=font_little)
		ax[1][1].tick_params(axis='both', labelsize=font_little)
		ax[1][2].tick_params(axis='both', labelsize=font_little)
		
		ax[0][0].grid(axis='both')
		ax[0][1].grid(axis='both')
		ax[0][2].grid(axis='both')
		ax[1][0].grid(axis='both')
		ax[1][1].grid(axis='both')
		ax[1][2].grid(axis='both')
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
			ax_it.yaxis.offsetText.set_visible(True)
		
		# plt.show()
		
		
		name = 'TestBench_Cyclo_SC_Each_AE_EnvFftComps'		
		path_1 = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\03_Figures\\'
		path_2 = 'C:\\Felix\\29_THESIS\\MODEL_A\\LATEX_Diss_FLn\\bilder\\Figures_Test_Bench\\'		
		path_1b = path_1 + name + '.svg'
		path_2b = path_2 + name + '.pdf'
		
		plt.savefig(path_1b)
		plt.savefig(path_2b)
		
		
		
	else:
		print('wrong mode')
		
		
		
		
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


if __name__ == '__main__':
	main(sys.argv)
