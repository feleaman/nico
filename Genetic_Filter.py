# Hilbert_Analysis.py
# Created: 25.09.2018 by Felix Leaman
# Description:

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from tkinter import filedialog
from tkinter import Tk
from decimal import Decimal
from argparse import ArgumentParser
from scipy.signal import hilbert
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from os import chdir
plt.rcParams['savefig.directory'] = chdir(os.path.dirname('C:'))
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from sklearn.cluster import KMeans	
from matplotlib.collections import LineCollection
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpeg'

#++++++++++++++++++++++ CONFIG ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['channel', 'fs', 'mode']

# vec = [4.33, ]

InputsOpt_Defaults = {'nlevel':3, 'num_generations':3, 'num_parents_mating':6, 'weight_mutation':0.05, 'freq_value':312., 'width_freq_range':30., 'name':'name'}
Style_Defaults = {'color_map':'copper_r', 'xlabel':'Time [s]', 'ylabel':'Frequency [kHz]', 'zlabel':'Amplitude [mV]', 'kHz':'ON', 'ylim':[80, 240], 'title': 'Channel AE-2, Second IMF, LC N°1', 'linewidth':2, 'step_cbar':75, 'logscale':'ON', 'xlim':[150, 400],'cbarlim':[0, 1050]}
InputsOpt_Defaults.update(Style_Defaults)

#++++++++++++++++++++++ MODES ++++++++++++++++++++++++++++++++++++++++++++++

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	if config['mode'] == 'test_genetic':
		# print(generate_population_filter_clusters(nlevel=7, fs=1.e6, clusters=20))
	
		# sys.exit()
		dt = 0.01
		fs = 1/dt
		t = np.arange(0, 5., dt)
		n = len(t)
		x = 0.21*np.sin(2*np.pi*4*t) + 0.11*np.sin(2*np.pi*30.25*t+2) + 0.16*np.sin(2*np.pi*11.1*t+2) + 0.161*np.sin(2*np.pi*15.1*t+2) + 0.1*np.random.randn(n)
		y = 0.16*np.sin(2*np.pi*11.1*t+2)
		# magX, f, df = mag_fft(x, fs)
		# plt.plot(f, magX)
		# plt.show()

		nlevel = 2		
		
		new_population = generate_population_filter(nlevel, fs)
		print('+++++++++++++++++++++++++++++++++++++initial population')		
		print(new_population)
		print(len(new_population))

		sol_per_pop = len(new_population)
		pop_size = (sol_per_pop, 2)

		num_generations = 15
		num_parents_mating = 4

		print('+++++++++++++++++++++++++++++++++++++best fitness initial population')
		ini_fitness = cal_pop_fitness_corr(x, y, fs, new_population)
		print(np.max(ini_fitness))
		print(select_mating_pool(new_population, ini_fitness, 1))
		
		for generation in range(num_generations):			
			fitness = cal_pop_fitness_corr(x, y, fs, new_population)
			parents = select_mating_pool(new_population, fitness, num_parents_mating)

			offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
			
			offspring_mutation = mutation(offspring_crossover, 0.05)

			new_population[0:parents.shape[0], :] = parents
			new_population[parents.shape[0]:, :] = offspring_mutation
		
			print('+++++++++++++++++++++++++++++++++++++best fitness final population')	
			end_fitness = cal_pop_fitness_corr(x, y, fs, new_population)
			print(np.max(end_fitness))
			print(select_mating_pool(new_population, end_fitness, 1))
	
	
	elif config['mode'] == 'test_ae':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		filename = os.path.basename(filepath)
		root.destroy()
		x = load_signal(filepath, channel=config['channel'])
		fs = config['fs']
		print(filename)

		# nlevel = 2		
		# num_generations = 3
		# num_parents_mating = 4
		# weight_mutation = 0.05
		# freq_value = 160.
		# width_freq_range = 20.
		
		nlevel = config['nlevel']		
		num_generations = config['num_generations']
		num_parents_mating = config['num_parents_mating']
		weight_mutation = config['weight_mutation']
		freq_value = config['freq_value']
		width_freq_range = config['width_freq_range']
		
		option = input('enter existing population? y/n:.... ')
		if option == 'n':
			new_population = generate_population_filter(nlevel, fs)
			print('+++++++++++++++++++++++++++++++++++++initial population')		
			print(new_population)
			print(len(new_population))

			sol_per_pop = len(new_population)
			pop_size = (sol_per_pop, 2)		

			print('+++++++++++++++++++++++++++++++++++++best fitness initial population')
			ini_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
			# print(ini_fitness)
			print(np.nanmax(ini_fitness))
			print(select_mating_pool(new_population, ini_fitness, 1))
			count = 0
		elif option == 'y':
			root = Tk()
			root.withdraw()
			root.update()
			filepath_pop = filedialog.askopenfilename()
			filename_pop = os.path.basename(filepath_pop)
			print(filename_pop)
			root.destroy()
			mydict_pop = read_pickle(filepath_pop)
			new_population = mydict_pop['new_population']
			sol_per_pop = len(new_population)
			pop_size = (sol_per_pop, 2)		
			count = mydict_pop['elapsed_generations']
			
		for generation in range(num_generations):		
			fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
			parents = select_mating_pool(new_population, fitness, num_parents_mating)

			offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
			
			offspring_mutation = mutation(offspring_crossover, weight_mutation)

			new_population[0:parents.shape[0], :] = parents
			new_population[parents.shape[0]:, :] = offspring_mutation
		
			print('+++++++++++++++++++++++++++++++++++++best fitness final population')	
			end_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
			print(np.max(end_fitness))
			best = select_mating_pool(new_population, end_fitness, 1)
			print(best)
			count += 1
		# sys.exit()
		mydict = {'new_population':new_population, 'elapsed_generations':count, 'config':config, 'best':best, 'filename':filename}
		save_pickle('config_generation_' + str(count) + '_' + config['name'] + '.pkl', mydict)

	elif config['mode'] == 'test_plot':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		
		
		
		fig = plt.figure(figsize=(8, 6))
		t = fig.suptitle('Wine Residual Sugar - Alcohol Content - Acidity - Type', fontsize=14)
		ax = fig.add_subplot(111, projection='3d')

		# xs = list(wines['residual sugar'])
		# ys = list(wines['alcohol'])
		# zs = list(wines['fixed acidity'])
		# data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]
		# colors = ['red' if wt == 'red' else 'yellow' for wt in list(wines['wine_type'])]

		# for data, color in zip(data_points, colors):
			# x, y, z = data
			# ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=30)
		x = [1, 2, 3]
		y = [6, 9, 10]
		z = [10, 18, 15]
		ax.scatter(x, y, z)
		
		ax.set_xlabel('Residual Sugar')
		ax.set_ylabel('Alcohol')
		ax.set_zlabel('Fixed Acidity')
		
		plt.show()
	
	elif config['mode'] == 'plot_genetic':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		print('Select PKL bests')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		# print('Select PKL configs')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths_Configs = filedialog.askopenfilenames()
		# root.destroy()
		Filepaths_Configs = Filepaths
		
		
		
		Filters_C = {}
		Filters_W = {}
		Mutations = []
		Generations = []
		for filepath, filepath_config in zip(Filepaths, Filepaths_Configs):
			data = read_pickle(filepath)
			config = read_pickle(filepath_config)
			# print(config)
			# Mutations = Mutations + [config['mutation']] + [config['mutation']] + [config['mutation']]
			# Generations = Generations + [config['generations']] + [config['generations']] + [config['generations']]
			# config['generations']
			# config['parents']
			for key in data.keys():
				idx = key.find('_AE_') + 4
				name_mc = key[idx:idx+8]			
				if name_mc not in Filters_C.keys():
					Filters_C[name_mc] = []
					Filters_W[name_mc] = []
					Filters_C[name_mc].append(data[key][0])
					Filters_W[name_mc].append(data[key][1])
				else:
					Filters_C[name_mc].append(data[key][0])
					Filters_W[name_mc].append(data[key][1])
			
			
		# print(Filters_C)
		# print(Filters_W)
		# print(Mutations)
		
		fig, ax = plt.subplots()
		
		ax.scatter(Filters_C['20181102'], Filters_W['20181102'])
		# ax.scatter(np.array(Filters_C['20180227']) - np.array(Filters_W['20180227'])/2, np.array(Filters_C['20180227']) + np.array(Filters_W['20180227'])/2)
		plt.show()
		
		
		
		
		sys.exit()
		fig, ax = plt.subplots()
		colors = ['red', 'blue', 'green', 'black', 'magenta', 'orange']
		count = 0
		for key_c, key_w in zip(Filters_C.keys(), Filters_W.keys()):
			ax.scatter(Filters_C[key_c], Filters_W[key_w], color=colors[count])
			count += 1
		
		plt.show()
			
		
		
		sys.exit()
	
	elif config['mode'] == 'plot_genetic_scores':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		print('Select XLXs bests')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		D_10h_f_f_r = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		D_5h_f_g = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		D_5h_sb_f_g_c = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		
		# mydict = pd.read_excel(filepath, sheetname=config['sheet'])
		# mydict = mydict.to_dict(orient='list')
		# Feature += mydict[config['feature']]

		for filepath in Filepaths:			
			mydict = pd.read_excel(filepath)
			mydict = mydict.to_dict(orient='list')
			
			# print(type(mydict))
			# print(mydict.keys())
			# print()
			# sys.exit()
			
			D_10h_f_f_r['20180227'].append(np.mean(np.array(mydict['10h_f_f_r'][0:3])))
			D_10h_f_f_r['20180314'].append(np.mean(np.array(mydict['10h_f_f_r'][3:6])))
			D_10h_f_f_r['20180316'].append(np.mean(np.array(mydict['10h_f_f_r'][6:9])))
			D_10h_f_f_r['20180322'].append(np.mean(np.array(mydict['10h_f_f_r'][9:12])))
			D_10h_f_f_r['20180511'].append(np.mean(np.array(mydict['10h_f_f_r'][12:15])))
			D_10h_f_f_r['20181102'].append(np.mean(np.array(mydict['10h_f_f_r'][15:18])))
			
			D_5h_f_g['20180227'].append(np.mean(np.array(mydict['5h_f_g'][0:3])))
			D_5h_f_g['20180314'].append(np.mean(np.array(mydict['5h_f_g'][3:6])))
			D_5h_f_g['20180316'].append(np.mean(np.array(mydict['5h_f_g'][6:9])))
			D_5h_f_g['20180322'].append(np.mean(np.array(mydict['5h_f_g'][9:12])))
			D_5h_f_g['20180511'].append(np.mean(np.array(mydict['5h_f_g'][12:15])))
			D_5h_f_g['20181102'].append(np.mean(np.array(mydict['5h_f_g'][15:18])))
			
			D_5h_sb_f_g_c['20180227'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][0:3])))
			D_5h_sb_f_g_c['20180314'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][3:6])))
			D_5h_sb_f_g_c['20180316'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][6:9])))
			D_5h_sb_f_g_c['20180322'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][9:12])))
			D_5h_sb_f_g_c['20180511'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][12:15])))
			D_5h_sb_f_g_c['20181102'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][15:18])))
			
			
		
		S_10h_f_f_r = np.zeros(len(D_10h_f_f_r['20180227']))
		S2_10h_f_f_r = np.zeros(len(D_10h_f_f_r['20180227']))
		for i in range(len(D_10h_f_f_r['20180227'])):
			score = 0
			if D_10h_f_f_r['20180227'][i] < D_10h_f_f_r['20180314'][i]:
				score += 1
			if D_10h_f_f_r['20180314'][i] < D_10h_f_f_r['20180316'][i]:
				score += 1
			if D_10h_f_f_r['20180316'][i] < D_10h_f_f_r['20180322'][i]:
				score += 1
			if D_10h_f_f_r['20180322'][i] < D_10h_f_f_r['20180511'][i]:
				score += 1
			if D_10h_f_f_r['20180511'][i] < D_10h_f_f_r['20181102'][i]:
				score += 1
			S_10h_f_f_r[i] = score
			S2_10h_f_f_r[i] = D_10h_f_f_r['20181102'][i] - D_10h_f_f_r['20180227'][i]
		
		S_5h_f_g = np.zeros(len(D_10h_f_f_r['20180227']))
		S2_5h_f_g = np.zeros(len(D_10h_f_f_r['20180227']))
		for i in range(len(D_5h_f_g['20180227'])):
			score = 0
			if D_5h_f_g['20180227'][i] < D_5h_f_g['20180314'][i]:
				score += 1
			if D_5h_f_g['20180314'][i] < D_5h_f_g['20180316'][i]:
				score += 1
			if D_5h_f_g['20180316'][i] < D_5h_f_g['20180322'][i]:
				score += 1
			if D_5h_f_g['20180322'][i] < D_5h_f_g['20180511'][i]:
				score += 1
			if D_5h_f_g['20180511'][i] < D_5h_f_g['20181102'][i]:
				score += 1
			S_5h_f_g[i] = score
			S2_5h_f_g[i] = D_5h_f_g['20181102'][i] - D_5h_f_g['20180227'][i]
		
		S_5h_sb_f_g_c = np.zeros(len(D_10h_f_f_r['20180227']))
		S2_5h_sb_f_g_c = np.zeros(len(D_10h_f_f_r['20180227']))
		for i in range(len(D_5h_sb_f_g_c['20180227'])):
			score = 0
			if D_5h_sb_f_g_c['20180227'][i] < D_5h_sb_f_g_c['20180314'][i]:
				score += 1
			if D_5h_sb_f_g_c['20180314'][i] < D_5h_sb_f_g_c['20180316'][i]:
				score += 1
			if D_5h_sb_f_g_c['20180316'][i] < D_5h_sb_f_g_c['20180322'][i]:
				score += 1
			if D_5h_sb_f_g_c['20180322'][i] < D_5h_sb_f_g_c['20180511'][i]:
				score += 1
			if D_5h_sb_f_g_c['20180511'][i] < D_5h_sb_f_g_c['20181102'][i]:
				score += 1
			S_5h_sb_f_g_c[i] = score
			S2_5h_sb_f_g_c[i] = D_5h_sb_f_g_c['20181102'][i] - D_5h_sb_f_g_c['20180227'][i]
		
		
		Scores = S_10h_f_f_r + S_5h_f_g + S_5h_sb_f_g_c
		Scores2 = S2_10h_f_f_r + S2_5h_f_g + S2_5h_sb_f_g_c
		
		
		
		
		
		plt.plot(Scores2, '-o')
		plt.show()
		
		for i in range(len(Scores)):
			if Scores[i] >= 14:
				print(i+14)
		sys.exit()
	
	
	elif config['mode'] == 'plot_filter_generations':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		print('Select PKLs bests')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		generations = 10+1
		# LP_freq = np.zeros(generations)
		# HP_freq = np.zeros(generations)
		dict_LP = {}
		dict_HP = {}
		count = 0
		for filepath in Filepaths:
			filename = os.path.basename(filepath)
			if filename[3:18] not in dict_LP.keys():
				dict_LP[filename[3:18]] = np.zeros(generations)
				dict_HP[filename[3:18]] = np.zeros(generations)
			for i in range(generations):
				text = 'generation_' + str(i)
				if filename.find(text) != -1:
					best = read_pickle(filepath)
					best = best[0]
					# print(best)
					# LP_freq[i] = best[0] - best[1]/2
					# HP_freq[i] = best[0] + best[1]/2
					
					dict_LP[filename[3:18]][i] = best[0] - best[1]/2
					dict_HP[filename[3:18]][i] = best[0] + best[1]/2
		
		LP_freq_mean = np.zeros(generations)
		LP_freq_error = np.zeros(generations)
		
		HP_freq_mean = np.zeros(generations)
		HP_freq_error = np.zeros(generations)
		
		count = 0
		for key in dict_LP.keys():
			LP_freq_mean = LP_freq_mean + dict_LP[key]
			HP_freq_mean = HP_freq_mean + dict_HP[key]
			count += 1
		LP_freq_mean = LP_freq_mean/count
		HP_freq_mean = HP_freq_mean/count
		
		count = 0
		for key in dict_LP.keys():
			LP_freq_error = LP_freq_error + (dict_LP[key] - LP_freq_mean)**2.0
			HP_freq_error = HP_freq_error + (dict_HP[key] - HP_freq_mean)**2.0
			count += 1
		LP_freq_error = (LP_freq_error/count)**0.5
		HP_freq_error = (HP_freq_error/count)**0.5
		
		fig, ax = plt.subplots()
		ax.plot(np.arange(generations), LP_freq_mean)
		ax.plot(np.arange(generations), HP_freq_mean)
		plt.show()
		
		
		fig, ax = plt.subplots()
		ax.fill_between(np.arange(generations), LP_freq_mean-LP_freq_error/2, LP_freq_mean+LP_freq_error/2)
		ax.fill_between(np.arange(generations), HP_freq_mean-HP_freq_error/2, HP_freq_mean+HP_freq_error/2, color='red')
		plt.show()
	
	
	elif config['mode'] == 'plot_filter_generations_2':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		print('Select PKLs bests')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		dict_idx = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		dict_filters = {}
		
		count = 0
		for filepath in Filepaths:
			filename = os.path.basename(filepath)
			for key in dict_idx.keys():
				if filename.find(key) != -1:
					dict_idx[key].append(count)
			count += 1
		
		
		
		generations = 50+1
		# LP_freq = np.zeros(generations)
		# HP_freq = np.zeros(generations)
		
		
		for key, idxes in dict_idx.items():
			# print(key)
			dict_LP = {}
			dict_HP = {}
			for i in idxes:
				# print(i)
				filepath = Filepaths[i]
				filename = os.path.basename(filepath)
				if filename[3:18] not in dict_LP.keys():
					dict_LP[filename[3:18]] = np.zeros(generations)
					dict_HP[filename[3:18]] = np.zeros(generations)
				for i in range(generations):
					print(i)
					text = 'generation_' + str(i)
					if filename.find(text) != -1:
						best = read_pickle(filepath)
						best = best[0]						
						dict_LP[filename[3:18]][i] = best[0] - best[1]/2
						dict_HP[filename[3:18]][i] = best[0] + best[1]/2
			
			# print(dict_HP)
			# print(dict_LP)
			# a = input('pause...')
			
			LP_freq_mean = np.zeros(generations)
			LP_freq_error = np.zeros(generations)
			
			HP_freq_mean = np.zeros(generations)
			HP_freq_error = np.zeros(generations)
			
			count = 0
			for key2 in dict_LP.keys():
				LP_freq_mean = LP_freq_mean + dict_LP[key2]
				HP_freq_mean = HP_freq_mean + dict_HP[key2]
				count += 1
			LP_freq_mean = LP_freq_mean/count
			HP_freq_mean = HP_freq_mean/count
			
			count = 0
			for key3 in dict_LP.keys():
				LP_freq_error = LP_freq_error + (dict_LP[key3] - LP_freq_mean)**2.0
				HP_freq_error = HP_freq_error + (dict_HP[key3] - HP_freq_mean)**2.0
				count += 1
			LP_freq_error = (LP_freq_error/count)**0.5
			HP_freq_error = (HP_freq_error/count)**0.5
			
			# plt.plot(LP_freq_mean)
			# plt.plot(HP_freq_mean)
			# plt.show()
			
		
			dict_filters[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
		
		
		
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)		
		ax[0][0].plot(np.arange(generations), dict_filters['20180227'][0])
		ax[0][0].plot(np.arange(generations), dict_filters['20180227'][1])
		
		ax[0][1].plot(np.arange(generations), dict_filters['20180314'][0])
		ax[0][1].plot(np.arange(generations), dict_filters['20180314'][1])
		
		ax[0][2].plot(np.arange(generations), dict_filters['20180316'][0])
		ax[0][2].plot(np.arange(generations), dict_filters['20180316'][1])
		
		
		ax[1][0].plot(np.arange(generations), dict_filters['20180322'][0])
		ax[1][0].plot(np.arange(generations), dict_filters['20180322'][1])
		
		ax[1][1].plot(np.arange(generations), dict_filters['20180511'][0])
		ax[1][1].plot(np.arange(generations), dict_filters['20180511'][1])
		
		ax[1][2].plot(np.arange(generations), dict_filters['20181102'][0])
		ax[1][2].plot(np.arange(generations), dict_filters['20181102'][1])
		plt.show()
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)		
		ax[0][0].fill_between(np.arange(generations), dict_filters['20180227'][0], dict_filters['20180227'][1], hatch='/', edgecolor='k', facecolor='lightgray')
		
		ax[0][1].fill_between(np.arange(generations), dict_filters['20180314'][0], dict_filters['20180314'][1], hatch='/', edgecolor='k', facecolor='lightgray')
		
		ax[0][2].fill_between(np.arange(generations), dict_filters['20180316'][0], dict_filters['20180316'][1], hatch='/', edgecolor='k', facecolor='lightgray')
		
		
		ax[1][0].fill_between(np.arange(generations), dict_filters['20180322'][0], dict_filters['20180322'][1], hatch='/', edgecolor='k', facecolor='lightgray')
		
		ax[1][1].fill_between(np.arange(generations), dict_filters['20180511'][0], dict_filters['20180511'][1], hatch='/', edgecolor='k', facecolor='lightgray')
		
		ax[1][2].fill_between(np.arange(generations), dict_filters['20181102'][0], dict_filters['20181102'][1], hatch='/', edgecolor='k', facecolor='lightgray')
		plt.show()
		
		
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True)		
		ax[0][0].fill_between(np.arange(generations), dict_filters['20180227'][0]-dict_filters['20180227'][2]/2, dict_filters['20180227'][0]+dict_filters['20180227'][2]/2, edgecolor='k')
		ax[0][0].fill_between(np.arange(generations), dict_filters['20180227'][1]-dict_filters['20180227'][3]/2, dict_filters['20180227'][1]+dict_filters['20180227'][3]/2, edgecolor='k')
		
		ax[0][1].fill_between(np.arange(generations), dict_filters['20180314'][0]-dict_filters['20180314'][2]/2, dict_filters['20180314'][0]+dict_filters['20180314'][2]/2, edgecolor='k')
		ax[0][1].fill_between(np.arange(generations), dict_filters['20180314'][1]-dict_filters['20180314'][3]/2, dict_filters['20180314'][1]+dict_filters['20180314'][3]/2, edgecolor='k')
		
		ax[0][2].fill_between(np.arange(generations), dict_filters['20180316'][0]-dict_filters['20180316'][2]/2, dict_filters['20180316'][0]+dict_filters['20180316'][2]/2, edgecolor='k')
		ax[0][2].fill_between(np.arange(generations), dict_filters['20180316'][1]-dict_filters['20180316'][3]/2, dict_filters['20180316'][1]+dict_filters['20180316'][3]/2, edgecolor='k')
		
		
		
		ax[1][0].fill_between(np.arange(generations), dict_filters['20180322'][0]-dict_filters['20180322'][2]/2, dict_filters['20180322'][0]+dict_filters['20180322'][2]/2, edgecolor='k')
		ax[1][0].fill_between(np.arange(generations), dict_filters['20180322'][1]-dict_filters['20180322'][3]/2, dict_filters['20180322'][1]+dict_filters['20180322'][3]/2, edgecolor='k')
		
		ax[1][1].fill_between(np.arange(generations), dict_filters['20180511'][0]-dict_filters['20180511'][2]/2, dict_filters['20180511'][0]+dict_filters['20180511'][2]/2, edgecolor='k')
		ax[1][1].fill_between(np.arange(generations), dict_filters['20180511'][1]-dict_filters['20180511'][3]/2, dict_filters['20180511'][1]+dict_filters['20180511'][3]/2, edgecolor='k')
		
		ax[1][2].fill_between(np.arange(generations), dict_filters['20181102'][0]-dict_filters['20181102'][2]/2, dict_filters['20181102'][0]+dict_filters['20181102'][2]/2, edgecolor='k')
		ax[1][2].fill_between(np.arange(generations), dict_filters['20181102'][1]-dict_filters['20181102'][3]/2, dict_filters['20181102'][1]+dict_filters['20181102'][3]/2, edgecolor='k')
		# ax[1][2].legend()
		plt.show()		
		
		
		# sys.exit()
		
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
		
		ax[0][0].fill_between(np.arange(generations), dict_filters['20180227'][0]/1.e3-dict_filters['20180227'][2]/2/1.e3, dict_filters['20180227'][1]/1.e3+dict_filters['20180227'][3]/2/1.e3, hatch='/', facecolor='white')
		
		ax[0][1].fill_between(np.arange(generations), dict_filters['20180314'][0]/1.e3-dict_filters['20180314'][2]/2/1.e3, dict_filters['20180314'][1]/1.e3+dict_filters['20180314'][3]/2/1.e3, hatch='/', facecolor='white')
		
		ax[0][2].fill_between(np.arange(generations), dict_filters['20180316'][0]/1.e3-dict_filters['20180316'][2]/2/1.e3, dict_filters['20180316'][1]/1.e3+dict_filters['20180316'][3]/2/1.e3, hatch='/', facecolor='white')
		
				
		ax[1][0].fill_between(np.arange(generations), dict_filters['20180322'][0]/1.e3-dict_filters['20180322'][2]/2/1.e3, dict_filters['20180322'][1]/1.e3+dict_filters['20180322'][3]/2/1.e3, hatch='/', facecolor='white')
		
		ax[1][1].fill_between(np.arange(generations), dict_filters['20180511'][0]/1.e3-dict_filters['20180511'][2]/2/1.e3, dict_filters['20180511'][1]/1.e3+dict_filters['20180511'][3]/2/1.e3, hatch='/', facecolor='white')
		
		ax[1][2].fill_between(np.arange(generations), dict_filters['20181102'][0]/1.e3-dict_filters['20181102'][2]/2/1.e3, dict_filters['20181102'][1]/1.e3+dict_filters['20181102'][3]/2/1.e3, hatch='/', facecolor='white')
		
		

		
		ax[0][0].fill_between(np.arange(generations), dict_filters['20180227'][0]/1.e3-dict_filters['20180227'][2]/2/1.e3, dict_filters['20180227'][0]/1.e3+dict_filters['20180227'][2]/2/1.e3, edgecolor='k', label='LP')
		ax[0][0].fill_between(np.arange(generations), dict_filters['20180227'][1]/1.e3-dict_filters['20180227'][3]/2/1.e3, dict_filters['20180227'][1]/1.e3+dict_filters['20180227'][3]/2/1.e3, edgecolor='k', label='HP')
		
		ax[0][1].fill_between(np.arange(generations), dict_filters['20180314'][0]/1.e3-dict_filters['20180314'][2]/2/1.e3, dict_filters['20180314'][0]/1.e3+dict_filters['20180314'][2]/2/1.e3, edgecolor='k', label='LP')
		ax[0][1].fill_between(np.arange(generations), dict_filters['20180314'][1]/1.e3-dict_filters['20180314'][3]/2/1.e3, dict_filters['20180314'][1]/1.e3+dict_filters['20180314'][3]/2/1.e3, edgecolor='k', label='HP')
		
		ax[0][2].fill_between(np.arange(generations), dict_filters['20180316'][0]/1.e3-dict_filters['20180316'][2]/2/1.e3, dict_filters['20180316'][0]/1.e3+dict_filters['20180316'][2]/2/1.e3, edgecolor='k', label='LP')
		ax[0][2].fill_between(np.arange(generations), dict_filters['20180316'][1]/1.e3-dict_filters['20180316'][3]/2/1.e3, dict_filters['20180316'][1]/1.e3+dict_filters['20180316'][3]/2/1.e3, edgecolor='k', label='HP')
		
		
		
		ax[1][0].fill_between(np.arange(generations), dict_filters['20180322'][0]/1.e3-dict_filters['20180322'][2]/2/1.e3, dict_filters['20180322'][0]/1.e3+dict_filters['20180322'][2]/2/1.e3, edgecolor='k', label='LP')
		ax[1][0].fill_between(np.arange(generations), dict_filters['20180322'][1]/1.e3-dict_filters['20180322'][3]/2/1.e3, dict_filters['20180322'][1]/1.e3+dict_filters['20180322'][3]/2/1.e3, edgecolor='k', label='HP')
		
		ax[1][1].fill_between(np.arange(generations), dict_filters['20180511'][0]/1.e3-dict_filters['20180511'][2]/2/1.e3, dict_filters['20180511'][0]/1.e3+dict_filters['20180511'][2]/2/1.e3, edgecolor='k', label='LP')
		ax[1][1].fill_between(np.arange(generations), dict_filters['20180511'][1]/1.e3-dict_filters['20180511'][3]/2/1.e3, dict_filters['20180511'][1]/1.e3+dict_filters['20180511'][3]/2/1.e3, edgecolor='k', label='HP')
		
		ax[1][2].fill_between(np.arange(generations), dict_filters['20181102'][0]/1.e3-dict_filters['20181102'][2]/2/1.e3, dict_filters['20181102'][0]/1.e3+dict_filters['20181102'][2]/2/1.e3, edgecolor='k', label='LP')
		ax[1][2].fill_between(np.arange(generations), dict_filters['20181102'][1]/1.e3-dict_filters['20181102'][3]/2/1.e3, dict_filters['20181102'][1]/1.e3+dict_filters['20181102'][3]/2/1.e3, edgecolor='k', label='HP')
		

		ax[0][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
		ax[1][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
			
		ax[0][0].set_xlabel('Generations', fontsize=font_big)
		ax[0][1].set_xlabel('Generations', fontsize=font_big)
		ax[0][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[1][0].set_xlabel('Generations', fontsize=font_big)
		ax[1][1].set_xlabel('Generations', fontsize=font_big)
		ax[1][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[0][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		ax[1][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$1^{st}$ MC', fontsize=font_big)
		ax[0][1].set_title('$2^{nd}$ MC', fontsize=font_big)
		ax[0][2].set_title('$4^{th}$ MC', fontsize=font_big)
		
		ax[1][0].set_title('$7^{th}$ MC', fontsize=font_big)
		ax[1][1].set_title('$9^{th}$ MC', fontsize=font_big)
		ax[1][2].set_title('$10^{th}$ MC', fontsize=font_big)
		
		
		# ax[0][0].set_xlim(left=0, right=10)
		# ax[0][1].set_xlim(left=0, right=10)
		# ax[0][2].set_xlim(left=0, right=10)
		
		# ax[1][0].set_xlim(left=0, right=10)
		# ax[1][1].set_xlim(left=0, right=10)
		# ax[1][2].set_xlim(left=0, right=10)
		

		valtop = 500
		ax[0][0].set_ylim(bottom=0, top=valtop)
		ax[0][1].set_ylim(bottom=0, top=valtop)
		ax[0][2].set_ylim(bottom=0, top=valtop)
		
		ax[1][0].set_ylim(bottom=0, top=valtop)
		ax[1][1].set_ylim(bottom=0, top=valtop)
		ax[1][2].set_ylim(bottom=0, top=valtop)

		
		
		ax[0][0].set_yticks([0, 100, 200, 300, 400, 500])
		ax[0][1].set_yticks([0, 100, 200, 300, 400, 500])
		ax[0][2].set_yticks([0, 100, 200, 300, 400, 500])
		
		ax[1][0].set_yticks([0, 100, 200, 300, 400, 500])
		ax[1][1].set_yticks([0, 100, 200, 300, 400, 500])
		ax[1][2].set_yticks([0, 100, 200, 300, 400, 500])
		
		
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
	
	elif config['mode'] == 'plot_filter_generations_3':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		dict_filters3 = {}
		dict_filters5 = {}
		dict_filters7 = {}
		
		for p in range(3):
			print('Select PKLs bests')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
			
			dict_idx = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
			
			
			
			
			count = 0
			for filepath in Filepaths:
				filename = os.path.basename(filepath)
				for key in dict_idx.keys():
					if filename.find(key) != -1:
						dict_idx[key].append(count)
				count += 1
					
			
			generations = 50+1
			
			for key, idxes in dict_idx.items():
				# print(key)
				dict_LP = {}
				dict_HP = {}
				for i in idxes:
					# print(i)
					filepath = Filepaths[i]
					filename = os.path.basename(filepath)
					if filename[3:18] not in dict_LP.keys():
						dict_LP[filename[3:18]] = np.zeros(generations)
						dict_HP[filename[3:18]] = np.zeros(generations)
					for i in range(generations):
						text = 'generation_' + str(i)
						if filename.find(text) != -1:
							best = read_pickle(filepath)
							best = best[0]						
							dict_LP[filename[3:18]][i] = best[0] - best[1]/2
							dict_HP[filename[3:18]][i] = best[0] + best[1]/2
				
				
				LP_freq_mean = np.zeros(generations)
				LP_freq_error = np.zeros(generations)
				
				HP_freq_mean = np.zeros(generations)
				HP_freq_error = np.zeros(generations)
				
				count = 0
				for key2 in dict_LP.keys():
					LP_freq_mean = LP_freq_mean + dict_LP[key2]
					HP_freq_mean = HP_freq_mean + dict_HP[key2]
					count += 1
				LP_freq_mean = LP_freq_mean/count
				HP_freq_mean = HP_freq_mean/count
				
				count = 0
				for key3 in dict_LP.keys():
					LP_freq_error = LP_freq_error + (dict_LP[key3] - LP_freq_mean)**2.0
					HP_freq_error = HP_freq_error + (dict_HP[key3] - HP_freq_mean)**2.0
					count += 1
				LP_freq_error = (LP_freq_error/count)**0.5
				HP_freq_error = (HP_freq_error/count)**0.5
			
				if p == 0:
					dict_filters3[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
					print('caca')
				elif p == 1:
					dict_filters5[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
				elif p == 2:
					dict_filters7[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]

		
		font_big = 17
		font_little = 15
		font_label = 13
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=False, sharey=False)
		
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.55)
		fig.set_size_inches(14.2, 6.2)
		
		# # ax.plot(np.arange(generations), dict_filters3['20180227'][0]/1.e3, color='r', label='level 3')
		# # ax.plot(np.arange(generations), dict_filters3['20180227'][1]/1.e3, color='b', label='level ff')
		
		
		# plt.scatter(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3, c=np.arange(generations), cmap='cool')
		
		# plt.plot(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3)
		
		# # for i in range(len(np.arange(generations))-1):
			# # plt.arrow(dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i]/1.e3, dict_filters3['20180227'][0][i+1]/1.e3 - dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i+1]/1.e3 - dict_filters3['20180227'][1][i]/1.e3, shape='full', lw=0, length_includes_head=True, head_width=.05)
		# plt.colorbar()
		
		# plt.show()
		# sys.exit()
		
		ax[0][0].plot(np.arange(generations), dict_filters3['20180227'][0]/1.e3, color='r')
		ax2 = ax[0][0].twinx()
		ax2.set_ylabel('Temperature', color='b')
		ax2.tick_params('y', colors='b')
		
		ax2.plot(np.arange(generations), dict_filters3['20180227'][1]/1.e3, color='b')
		
		ax[0][0].plot(np.arange(generations), dict_filters5['20180227'][0]/1.e3, color='r')
		ax2.plot(np.arange(generations), dict_filters5['20180227'][1]/1.e3, color='b')
		
		ax[0][0].plot(np.arange(generations), dict_filters7['20180227'][0]/1.e3, color='r')
		ax2.plot(np.arange(generations), dict_filters7['20180227'][1]/1.e3, color='b')
		
		
		
		ax[0][1].plot(np.arange(generations), dict_filters3['20180314'][0]/1.e3, color='r')
		ax[0][1].plot(np.arange(generations), dict_filters3['20180314'][1]/1.e3, color='b')
		
		ax[0][1].plot(np.arange(generations), dict_filters5['20180314'][0]/1.e3, color='r')
		ax[0][1].plot(np.arange(generations), dict_filters5['20180314'][1]/1.e3, color='b')
		
		ax[0][1].plot(np.arange(generations), dict_filters7['20180314'][0]/1.e3, color='r')
		ax[0][1].plot(np.arange(generations), dict_filters7['20180314'][1]/1.e3, color='b')
		
		
		
		
		
		ax[0][2].plot(np.arange(generations), dict_filters3['20180316'][0]/1.e3, color='r')
		ax3 = ax[0][2].twinx()
		ax3.set_ylabel('Temperature', color='b')
		ax3.tick_params('y', colors='b')
		
		ax3.plot(np.arange(generations), dict_filters3['20180316'][1]/1.e3, color='b')
		
		ax[0][2].plot(np.arange(generations), dict_filters5['20180316'][0]/1.e3, color='r')
		ax3.plot(np.arange(generations), dict_filters5['20180316'][1]/1.e3, color='b')
		
		ax[0][2].plot(np.arange(generations), dict_filters7['20180316'][0]/1.e3, color='r')
		ax3.plot(np.arange(generations), dict_filters7['20180316'][1]/1.e3, color='b')
		
		
		
		
		ax[1][0].plot(np.arange(generations), dict_filters3['20180322'][0]/1.e3, color='r')
		ax[1][0].plot(np.arange(generations), dict_filters3['20180322'][1]/1.e3, color='b')
		
		ax[1][0].plot(np.arange(generations), dict_filters5['20180322'][0]/1.e3, color='r')
		ax[1][0].plot(np.arange(generations), dict_filters5['20180322'][1]/1.e3, color='b')
		
		ax[1][0].plot(np.arange(generations), dict_filters7['20180322'][0]/1.e3, color='r')
		ax[1][0].plot(np.arange(generations), dict_filters7['20180322'][1]/1.e3, color='b')
		
		
		
		
		ax[1][1].plot(np.arange(generations), dict_filters3['20180511'][0]/1.e3, color='r')
		ax[1][1].plot(np.arange(generations), dict_filters3['20180511'][1]/1.e3, color='b')
		
		ax[1][1].plot(np.arange(generations), dict_filters5['20180511'][0]/1.e3, color='r')
		ax[1][1].plot(np.arange(generations), dict_filters5['20180511'][1]/1.e3, color='b')
		
		ax[1][1].plot(np.arange(generations), dict_filters7['20180511'][0]/1.e3, color='r')
		ax[1][1].plot(np.arange(generations), dict_filters7['20180511'][1]/1.e3, color='b')#
		
		
		
		
		ax[1][2].plot(np.arange(generations), dict_filters3['20181102'][0]/1.e3, color='r')
		ax[1][2].plot(np.arange(generations), dict_filters3['20181102'][1]/1.e3, color='b')
		
		ax[1][2].plot(np.arange(generations), dict_filters5['20181102'][0]/1.e3, color='r')
		ax[1][2].plot(np.arange(generations), dict_filters5['20181102'][1]/1.e3, color='b')
		
		ax[1][2].plot(np.arange(generations), dict_filters7['20181102'][0]/1.e3, color='r')
		ax[1][2].plot(np.arange(generations), dict_filters7['20181102'][1]/1.e3, color='b')
		
		
		
		
		
		

		

		ax[0][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
		ax[1][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
			
		ax[0][0].set_xlabel('Generations', fontsize=font_big)
		ax[0][1].set_xlabel('Generations', fontsize=font_big)
		ax[0][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[1][0].set_xlabel('Generations', fontsize=font_big)
		ax[1][1].set_xlabel('Generations', fontsize=font_big)
		ax[1][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[0][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		ax[1][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$1^{st}$ MC', fontsize=font_big)
		ax[0][1].set_title('$2^{nd}$ MC', fontsize=font_big)
		ax[0][2].set_title('$4^{th}$ MC', fontsize=font_big)
		
		ax[1][0].set_title('$7^{th}$ MC', fontsize=font_big)
		ax[1][1].set_title('$9^{th}$ MC', fontsize=font_big)
		ax[1][2].set_title('$10^{th}$ MC', fontsize=font_big)
		
		
		# ax[0][0].set_xlim(left=0, right=10)
		# ax[0][1].set_xlim(left=0, right=10)
		# ax[0][2].set_xlim(left=0, right=10)
		
		# ax[1][0].set_xlim(left=0, right=10)
		# ax[1][1].set_xlim(left=0, right=10)
		# ax[1][2].set_xlim(left=0, right=10)
		

		# valtop = 500
		# ax[0][0].set_ylim(bottom=0, top=valtop)
		# ax[0][1].set_ylim(bottom=0, top=valtop)
		# ax[0][2].set_ylim(bottom=0, top=valtop)
		
		# ax[1][0].set_ylim(bottom=0, top=valtop)
		# ax[1][1].set_ylim(bottom=0, top=valtop)
		# ax[1][2].set_ylim(bottom=0, top=valtop)

		
		
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
	
	
	elif config['mode'] == 'plot_filter_generations_4':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		dict_filters3 = {}
		dict_filters5 = {}
		dict_filters7 = {}
		
		for p in range(3):
			print('Select PKLs bests')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
			
			dict_idx = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
			
			
			
			
			count = 0
			for filepath in Filepaths:
				filename = os.path.basename(filepath)
				for key in dict_idx.keys():
					if filename.find(key) != -1:
						dict_idx[key].append(count)
				count += 1
					
			
			generations = 50+1
			
			for key, idxes in dict_idx.items():
				# print(key)
				dict_LP = {}
				dict_HP = {}
				for i in idxes:
					# print(i)
					filepath = Filepaths[i]
					filename = os.path.basename(filepath)
					if filename[3:18] not in dict_LP.keys():
						dict_LP[filename[3:18]] = np.zeros(generations)
						dict_HP[filename[3:18]] = np.zeros(generations)
					for i in range(generations):
						text = 'generation_' + str(i)
						if filename.find(text) != -1:
							best = read_pickle(filepath)
							best = best[0]						
							dict_LP[filename[3:18]][i] = best[0] - best[1]/2
							dict_HP[filename[3:18]][i] = best[0] + best[1]/2
				
				
				LP_freq_mean = np.zeros(generations)
				LP_freq_error = np.zeros(generations)
				
				HP_freq_mean = np.zeros(generations)
				HP_freq_error = np.zeros(generations)
				
				count = 0
				for key2 in dict_LP.keys():
					LP_freq_mean = LP_freq_mean + dict_LP[key2]
					HP_freq_mean = HP_freq_mean + dict_HP[key2]
					count += 1
				LP_freq_mean = LP_freq_mean/count
				HP_freq_mean = HP_freq_mean/count
				
				count = 0
				for key3 in dict_LP.keys():
					LP_freq_error = LP_freq_error + (dict_LP[key3] - LP_freq_mean)**2.0
					HP_freq_error = HP_freq_error + (dict_HP[key3] - HP_freq_mean)**2.0
					count += 1
				LP_freq_error = (LP_freq_error/count)**0.5
				HP_freq_error = (HP_freq_error/count)**0.5
			
				if p == 0:
					dict_filters3[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
					print('caca')
				elif p == 1:
					dict_filters5[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
				elif p == 2:
					dict_filters7[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]

		
		font_big = 17
		font_little = 15
		font_label = 13
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=False, sharey=False)
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.55)
		fig.set_size_inches(14.2, 6.2)
		

		
		
		# plt.scatter(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3, c=np.arange(generations), cmap='cool')
		
		# plt.plot(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3)
		
		# # for i in range(len(np.arange(generations))-1):
			# # plt.arrow(dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i]/1.e3, dict_filters3['20180227'][0][i+1]/1.e3 - dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i+1]/1.e3 - dict_filters3['20180227'][1][i]/1.e3, shape='full', lw=0, length_includes_head=True, head_width=.05)
		# plt.colorbar()
		
		# plt.show()
		# sys.exit()
		
		# Choose colormap
		cmap = plt.cm.Blues

		# Get the colormap colors
		my_cmap = cmap(np.arange(cmap.N))

		# Set alpha
		my_cmap[:,-1] = np.linspace(0, 0.1, cmap.N)

		# Create new colormap
		my_cmap = ListedColormap(my_cmap)
		
		
		ax[0][0].scatter(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3, c=np.arange(generations), cmap='Blues')		
		ax[0][0].scatter(dict_filters5['20180227'][0]/1.e3, dict_filters5['20180227'][1]/1.e3, c=np.arange(generations), cmap='Reds')		
		ax[0][0].scatter(dict_filters7['20180227'][0]/1.e3, dict_filters7['20180227'][1]/1.e3, c=np.arange(generations), cmap='Greens')
		
		# aa = ax[0][0].plot(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3, c='blue')		
		# aa = ax[0][0].plot(dict_filters5['20180227'][0]/1.e3, dict_filters5['20180227'][1]/1.e3, c='red')		
		# aa = ax[0][0].plot(dict_filters7['20180227'][0]/1.e3, dict_filters7['20180227'][1]/1.e3, c='green')
		
		
		
		ax[0][1].scatter(dict_filters3['20180314'][0]/1.e3, dict_filters3['20180314'][1]/1.e3, c=np.arange(generations), cmap='Blues')		
		ax[0][1].scatter(dict_filters5['20180314'][0]/1.e3, dict_filters5['20180314'][1]/1.e3, c=np.arange(generations), cmap='Reds')		
		ax[0][1].scatter(dict_filters7['20180314'][0]/1.e3, dict_filters7['20180314'][1]/1.e3, c=np.arange(generations), cmap='Greens')
		
		
		
		
		
		ax[0][2].scatter(dict_filters3['20180316'][0]/1.e3, dict_filters3['20180316'][1]/1.e3, c=np.arange(generations), cmap='Blues')		
		ax[0][2].scatter(dict_filters5['20180316'][0]/1.e3, dict_filters5['20180316'][1]/1.e3, c=np.arange(generations), cmap='Reds')		
		ax[0][2].scatter(dict_filters7['20180316'][0]/1.e3, dict_filters7['20180316'][1]/1.e3, c=np.arange(generations), cmap='Greens')
		
		
		
		
		ax[1][0].scatter(dict_filters3['20180322'][0]/1.e3, dict_filters3['20180322'][1]/1.e3, c=np.arange(generations), cmap='Blues')		
		ax[1][0].scatter(dict_filters5['20180322'][0]/1.e3, dict_filters5['20180322'][1]/1.e3, c=np.arange(generations), cmap='Reds')		
		ax[1][0].scatter(dict_filters7['20180322'][0]/1.e3, dict_filters7['20180322'][1]/1.e3, c=np.arange(generations), cmap='Greens')
		
		
		
		
		lvl3 = ax[1][1].scatter(dict_filters3['20180511'][0]/1.e3, dict_filters3['20180511'][1]/1.e3, c=np.arange(generations), cmap='Blues')		
		lvl5 = ax[1][1].scatter(dict_filters5['20180511'][0]/1.e3, dict_filters5['20180511'][1]/1.e3, c=np.arange(generations), cmap='Reds')		
		lvl7 = ax[1][1].scatter(dict_filters7['20180511'][0]/1.e3, dict_filters7['20180511'][1]/1.e3, c=np.arange(generations), cmap='Greens')
		
		
		# lc = generate_color_line(dict_filters3['20180511'][0]/1.e3, dict_filters3['20180511'][1]/1.e3, 'spring')	
		# lvl3 = ax[1][1].add_collection(lc)
		# lc = generate_color_line(dict_filters5['20180511'][0]/1.e3, dict_filters5['20180511'][1]/1.e3, 'summer')	
		# lvl5 = ax[1][1].add_collection(lc)
		# lc = generate_color_line(dict_filters7['20180511'][0]/1.e3, dict_filters7['20180511'][1]/1.e3, 'autumn')	
		# lvl7 = ax[1][1].add_collection(lc)
		
		
		
		ax[1][2].scatter(dict_filters3['20181102'][0]/1.e3, dict_filters3['20181102'][1]/1.e3, c=np.arange(generations), cmap='Blues')		
		ax[1][2].scatter(dict_filters5['20181102'][0]/1.e3, dict_filters5['20181102'][1]/1.e3, c=np.arange(generations), cmap='Reds')		
		ax[1][2].scatter(dict_filters7['20181102'][0]/1.e3, dict_filters7['20181102'][1]/1.e3, c=np.arange(generations), cmap='Greens')
		
		
		
		
		
		

		

		ax[0][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
		ax[1][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
			
		ax[0][0].set_xlabel('Generations', fontsize=font_big)
		ax[0][1].set_xlabel('Generations', fontsize=font_big)
		ax[0][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[1][0].set_xlabel('Generations', fontsize=font_big)
		ax[1][1].set_xlabel('Generations', fontsize=font_big)
		ax[1][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[0][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		ax[1][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$1^{st}$ MC', fontsize=font_big)
		ax[0][1].set_title('$2^{nd}$ MC', fontsize=font_big)
		ax[0][2].set_title('$4^{th}$ MC', fontsize=font_big)
		
		ax[1][0].set_title('$7^{th}$ MC', fontsize=font_big)
		ax[1][1].set_title('$9^{th}$ MC', fontsize=font_big)
		ax[1][2].set_title('$10^{th}$ MC', fontsize=font_big)
		
		
		# ax[0][0].set_xlim(left=0, right=10)
		# ax[0][1].set_xlim(left=0, right=10)
		# ax[0][2].set_xlim(left=0, right=10)
		
		# ax[1][0].set_xlim(left=0, right=10)
		# ax[1][1].set_xlim(left=0, right=10)
		# ax[1][2].set_xlim(left=0, right=10)
		

		valtop = 500
		valbottom = 200
		# # ax[0][0].set_ylim(bottom=valbottom, top=valtop)
		# # ax[0][1].set_ylim(bottom=valbottom, top=valtop)
		# # ax[0][2].set_ylim(bottom=valbottom, top=valtop)
		
		# # ax[1][0].set_ylim(bottom=valbottom, top=valtop)
		# # ax[1][1].set_ylim(bottom=valbottom, top=valtop)
		# # ax[1][2].set_ylim(bottom=valbottom, top=valtop)

		
		
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
	
	
	elif config['mode'] == 'plot_filter_generations_5':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		dict_filters3 = {}
		dict_filters5 = {}
		dict_filters7 = {}
		
		for p in range(3):
			print('Select PKLs bests')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
			
			dict_idx = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
			
			
			
			
			count = 0
			for filepath in Filepaths:
				filename = os.path.basename(filepath)
				for key in dict_idx.keys():
					if filename.find(key) != -1:
						dict_idx[key].append(count)
				count += 1
					
			
			generations = 50+1
			
			for key, idxes in dict_idx.items():
				# print(key)
				dict_LP = {}
				dict_HP = {}
				for i in idxes:
					# print(i)
					filepath = Filepaths[i]
					filename = os.path.basename(filepath)
					if filename[3:18] not in dict_LP.keys():
						dict_LP[filename[3:18]] = np.zeros(generations)
						dict_HP[filename[3:18]] = np.zeros(generations)
					for i in range(generations):
						text = 'generation_' + str(i)
						if filename.find(text) != -1:
							best = read_pickle(filepath)
							best = best[0]						
							dict_LP[filename[3:18]][i] = best[0] - best[1]/2
							dict_HP[filename[3:18]][i] = best[0] + best[1]/2
				
				
				LP_freq_mean = np.zeros(generations)
				LP_freq_error = np.zeros(generations)
				
				HP_freq_mean = np.zeros(generations)
				HP_freq_error = np.zeros(generations)
				
				count = 0
				for key2 in dict_LP.keys():
					LP_freq_mean = LP_freq_mean + dict_LP[key2]
					HP_freq_mean = HP_freq_mean + dict_HP[key2]
					count += 1
				LP_freq_mean = LP_freq_mean/count
				HP_freq_mean = HP_freq_mean/count
				
				count = 0
				for key3 in dict_LP.keys():
					LP_freq_error = LP_freq_error + (dict_LP[key3] - LP_freq_mean)**2.0
					HP_freq_error = HP_freq_error + (dict_HP[key3] - HP_freq_mean)**2.0
					count += 1
				LP_freq_error = (LP_freq_error/count)**0.5
				HP_freq_error = (HP_freq_error/count)**0.5
			
				if p == 0:
					dict_filters3[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
					print('caca')
				elif p == 1:
					dict_filters5[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
				elif p == 2:
					dict_filters7[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]

		
		font_big = 17
		font_little = 15
		font_label = 13
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=2, sharex=False, sharey=False)
		
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.55)
		fig.set_size_inches(14.2, 6.2)
		
		# # ax.plot(np.arange(generations), dict_filters3['20180227'][0]/1.e3, color='r', label='level 3')
		# # ax.plot(np.arange(generations), dict_filters3['20180227'][1]/1.e3, color='b', label='level ff')
		
		
		# plt.scatter(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3, c=np.arange(generations), cmap='cool')
		
		# plt.plot(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3)
		
		# # for i in range(len(np.arange(generations))-1):
			# # plt.arrow(dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i]/1.e3, dict_filters3['20180227'][0][i+1]/1.e3 - dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i+1]/1.e3 - dict_filters3['20180227'][1][i]/1.e3, shape='full', lw=0, length_includes_head=True, head_width=.05)
		# plt.colorbar()
		
		# plt.show()
		# sys.exit()
		
		ax[0][0].plot(np.arange(generations), dict_filters3['20180227'][0]/1.e3, color='r', marker='o')
		ax00 = ax[0][0].twinx()
		ax00.set_ylabel('HP Frequency [kHz]', color='b')
		ax00.tick_params('y', colors='b')
		
		ax00.plot(np.arange(generations), dict_filters3['20180227'][1]/1.e3, color='b', marker='o')
		
		ax[0][0].plot(np.arange(generations), dict_filters5['20180227'][0]/1.e3, color='r', marker='s')
		ax00.plot(np.arange(generations), dict_filters5['20180227'][1]/1.e3, color='b', marker='s')
		
		ax[0][0].plot(np.arange(generations), dict_filters7['20180227'][0]/1.e3, color='r', marker='^')
		ax00.plot(np.arange(generations), dict_filters7['20180227'][1]/1.e3, color='b', marker='^')
		
		
		
		ax[0][1].plot(np.arange(generations), dict_filters3['20180314'][0]/1.e3, color='r', marker='o')
		ax01 = ax[0][1].twinx()
		ax01.set_ylabel('HP Frequency [kHz]', color='b')
		ax01.tick_params('y', colors='b')
		ax01.plot(np.arange(generations), dict_filters3['20180314'][1]/1.e3, color='b', marker='o')
		
		ax[0][1].plot(np.arange(generations), dict_filters5['20180314'][0]/1.e3, color='r', marker='s')
		ax01.plot(np.arange(generations), dict_filters5['20180314'][1]/1.e3, color='b', marker='s')
		
		ax[0][1].plot(np.arange(generations), dict_filters7['20180314'][0]/1.e3, color='r', marker='^')
		ax01.plot(np.arange(generations), dict_filters7['20180314'][1]/1.e3, color='b', marker='^')
		
		
		
		
		
		ax[0][2].plot(np.arange(generations), dict_filters3['20180316'][0]/1.e3, color='r', marker='o')
		ax02 = ax[0][2].twinx()
		ax02.set_ylabel('HP Frequency [kHz]', color='b')
		ax02.tick_params('y', colors='b')
		
		ax02.plot(np.arange(generations), dict_filters3['20180316'][1]/1.e3, color='b', marker='o')
		
		ax[0][2].plot(np.arange(generations), dict_filters5['20180316'][0]/1.e3, color='r', marker='s')
		ax02.plot(np.arange(generations), dict_filters5['20180316'][1]/1.e3, color='b', marker='s')
		
		ax[0][2].plot(np.arange(generations), dict_filters7['20180316'][0]/1.e3, color='r', marker='^')
		ax02.plot(np.arange(generations), dict_filters7['20180316'][1]/1.e3, color='b', marker='^')
		
		
		
		
		ax[1][0].plot(np.arange(generations), dict_filters3['20180322'][0]/1.e3, color='r', marker='o')
		ax10 = ax[1][0].twinx()
		ax10.set_ylabel('HP Frequency [kHz]', color='b')
		ax10.tick_params('y', colors='b')
		
		ax10.plot(np.arange(generations), dict_filters3['20180322'][1]/1.e3, color='b', marker='o')
		
		ax[1][0].plot(np.arange(generations), dict_filters5['20180322'][0]/1.e3, color='r', marker='s')
		ax10.plot(np.arange(generations), dict_filters5['20180322'][1]/1.e3, color='b', marker='s')
		
		ax[1][0].plot(np.arange(generations), dict_filters7['20180322'][0]/1.e3, color='r', marker='^')
		ax10.plot(np.arange(generations), dict_filters7['20180322'][1]/1.e3, color='b', marker='^')
		
		
		
		
		ax[1][1].plot(np.arange(generations), dict_filters3['20180511'][0]/1.e3, color='r', marker='o')
		ax11 = ax[1][1].twinx()
		ax11.set_ylabel('HP Frequency [kHz]', color='b')
		ax11.tick_params('y', colors='b')
		
		ax11.plot(np.arange(generations), dict_filters3['20180511'][1]/1.e3, color='b', marker='o')
		
		ax[1][1].plot(np.arange(generations), dict_filters5['20180511'][0]/1.e3, color='r', marker='s')
		ax11.plot(np.arange(generations), dict_filters5['20180511'][1]/1.e3, color='b', marker='s')
		
		ax[1][1].plot(np.arange(generations), dict_filters7['20180511'][0]/1.e3, color='r', marker='^')
		ax11.plot(np.arange(generations), dict_filters7['20180511'][1]/1.e3, color='b', marker='^')#
		
		
		
		
		ax[1][2].plot(np.arange(generations), dict_filters3['20181102'][0]/1.e3, color='r', marker='o')
		ax12 = ax[1][2].twinx()
		ax12.set_ylabel('HP Frequency [kHz]', color='b')
		ax12.tick_params('y', colors='b')
		
		ax12.plot(np.arange(generations), dict_filters3['20181102'][1]/1.e3, color='b', marker='o')
		
		ax[1][2].plot(np.arange(generations), dict_filters5['20181102'][0]/1.e3, color='r', marker='s')
		ax12.plot(np.arange(generations), dict_filters5['20181102'][1]/1.e3, color='b', marker='s')
		
		ax[1][2].plot(np.arange(generations), dict_filters7['20181102'][0]/1.e3, color='r', marker='^')
		ax12.plot(np.arange(generations), dict_filters7['20181102'][1]/1.e3, color='b', marker='^')
		
		
		
		
		
		

		

		ax[0][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
		ax[1][0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1][2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
			
		ax[0][0].set_xlabel('Generations', fontsize=font_big)
		ax[0][1].set_xlabel('Generations', fontsize=font_big)
		ax[0][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[1][0].set_xlabel('Generations', fontsize=font_big)
		ax[1][1].set_xlabel('Generations', fontsize=font_big)
		ax[1][2].set_xlabel('Generations', fontsize=font_big)
		
		ax[0][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[0][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		ax[1][0].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][1].legend(fontsize=font_label, ncol=2, loc='best')
		ax[1][2].legend(fontsize=font_label, ncol=2, loc='best')
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$1^{st}$ MC', fontsize=font_big)
		ax[0][1].set_title('$2^{nd}$ MC', fontsize=font_big)
		ax[0][2].set_title('$4^{th}$ MC', fontsize=font_big)
		
		ax[1][0].set_title('$7^{th}$ MC', fontsize=font_big)
		ax[1][1].set_title('$9^{th}$ MC', fontsize=font_big)
		ax[1][2].set_title('$10^{th}$ MC', fontsize=font_big)
		
		
		# ax[0][0].set_xlim(left=0, right=10)
		# ax[0][1].set_xlim(left=0, right=10)
		# ax[0][2].set_xlim(left=0, right=10)
		
		# ax[1][0].set_xlim(left=0, right=10)
		# ax[1][1].set_xlim(left=0, right=10)
		# ax[1][2].set_xlim(left=0, right=10)
		

		# valtop = 500
		# ax[0][0].set_ylim(bottom=0, top=valtop)
		# ax[0][1].set_ylim(bottom=0, top=valtop)
		# ax[0][2].set_ylim(bottom=0, top=valtop)
		
		# ax[1][0].set_ylim(bottom=0, top=valtop)
		# ax[1][1].set_ylim(bottom=0, top=valtop)
		# ax[1][2].set_ylim(bottom=0, top=valtop)

		
		
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
		
	
	elif config['mode'] == 'plot_filter_generations_6':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		dict_filters3 = {}
		dict_filters5 = {}
		dict_filters7 = {}
		
		for p in range(3):
			print('Select PKLs bests')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			root.destroy()
			
			dict_idx = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
			
			
			
			
			count = 0
			for filepath in Filepaths:
				filename = os.path.basename(filepath)
				for key in dict_idx.keys():
					if filename.find(key) != -1:
						dict_idx[key].append(count)
				count += 1
					
			
			generations = 50+1
			
			for key, idxes in dict_idx.items():
				# print(key)
				dict_LP = {}
				dict_HP = {}
				for i in idxes:
					# print(i)
					filepath = Filepaths[i]
					filename = os.path.basename(filepath)
					if filename[3:18] not in dict_LP.keys():
						dict_LP[filename[3:18]] = np.zeros(generations)
						dict_HP[filename[3:18]] = np.zeros(generations)
					for i in range(generations):
						text = 'generation_' + str(i)
						if filename.find(text) != -1:
							best = read_pickle(filepath)
							best = best[0]						
							dict_LP[filename[3:18]][i] = best[0] - best[1]/2
							dict_HP[filename[3:18]][i] = best[0] + best[1]/2
				
				
				LP_freq_mean = np.zeros(generations)
				LP_freq_error = np.zeros(generations)
				
				HP_freq_mean = np.zeros(generations)
				HP_freq_error = np.zeros(generations)
				
				count = 0
				for key2 in dict_LP.keys():
					LP_freq_mean = LP_freq_mean + dict_LP[key2]
					HP_freq_mean = HP_freq_mean + dict_HP[key2]
					count += 1
				LP_freq_mean = LP_freq_mean/count
				HP_freq_mean = HP_freq_mean/count
				
				count = 0
				for key3 in dict_LP.keys():
					LP_freq_error = LP_freq_error + (dict_LP[key3] - LP_freq_mean)**2.0
					HP_freq_error = HP_freq_error + (dict_HP[key3] - HP_freq_mean)**2.0
					count += 1
				LP_freq_error = (LP_freq_error/count)**0.5
				HP_freq_error = (HP_freq_error/count)**0.5
			
				if p == 0:
					dict_filters3[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
					print('caca')
				elif p == 1:
					dict_filters5[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]
				elif p == 2:
					dict_filters7[key] = [LP_freq_mean, HP_freq_mean, LP_freq_error, HP_freq_error]

		
		font_big = 17
		font_little = 15
		font_label = 13
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False)
		
		
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.175, top=0.92)
		
		# plt.subplots_adjust(wspace=0.275, left=0.065, right=0.9, bottom=0.125, top=0.92, hspace=0.55)
		fig.set_size_inches(14.2, 4.2)
		
		# # ax.plot(np.arange(generations), dict_filters3['20180227'][0]/1.e3, color='r', label='level 3')
		# # ax.plot(np.arange(generations), dict_filters3['20180227'][1]/1.e3, color='b', label='level ff')
		
		
		# plt.scatter(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3, c=np.arange(generations), cmap='cool')
		
		# plt.plot(dict_filters3['20180227'][0]/1.e3, dict_filters3['20180227'][1]/1.e3)
		
		# # for i in range(len(np.arange(generations))-1):
			# # plt.arrow(dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i]/1.e3, dict_filters3['20180227'][0][i+1]/1.e3 - dict_filters3['20180227'][0][i]/1.e3, dict_filters3['20180227'][1][i+1]/1.e3 - dict_filters3['20180227'][1][i]/1.e3, shape='full', lw=0, length_includes_head=True, head_width=.05)
		# plt.colorbar()
		
		# plt.show()
		# sys.exit()
		
		mc = np.arange(6)
		ones = np.ones(generations)
		twos = np.ones(generations)*2.
		threes = np.ones(generations)*3.
		fours = np.ones(generations)*4.
		fives = np.ones(generations)*5.
		sixes = np.ones(generations)*6.
		
		my_edgecolor = 'black'
		my_cmap_lp = 'binary'
		my_cmap_hp = 'binary'
		my_s = 100
		
		# plt.colorbar()
		
		im = ax[0].scatter(ones, dict_filters3['20180227'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)
		ax[0].scatter(ones, dict_filters3['20180227'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[0].scatter(twos, dict_filters3['20180314'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[0].scatter(twos, dict_filters3['20180314'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[0].scatter(threes, dict_filters3['20180316'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[0].scatter(threes, dict_filters3['20180316'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[0].scatter(fours, dict_filters3['20180322'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[0].scatter(fours, dict_filters3['20180322'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[0].scatter(fives, dict_filters3['20180511'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[0].scatter(fives, dict_filters3['20180511'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[0].scatter(sixes, dict_filters3['20181102'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s, label='HP')
		ax[0].scatter(sixes, dict_filters3['20181102'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s, label='LP')
		
		
		
		
		ax[1].scatter(ones, dict_filters5['20180227'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[1].scatter(ones, dict_filters5['20180227'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[1].scatter(twos, dict_filters5['20180314'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[1].scatter(twos, dict_filters5['20180314'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[1].scatter(threes, dict_filters5['20180316'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[1].scatter(threes, dict_filters5['20180316'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[1].scatter(fours, dict_filters5['20180322'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[1].scatter(fours, dict_filters5['20180322'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[1].scatter(fives, dict_filters5['20180511'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[1].scatter(fives, dict_filters5['20180511'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[1].scatter(sixes, dict_filters5['20181102'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s, label='HP')		
		ax[1].scatter(sixes, dict_filters5['20181102'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s, label='LP')
		
		
		
		
		
		ax[2].scatter(ones, dict_filters7['20180227'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[2].scatter(ones, dict_filters7['20180227'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[2].scatter(twos, dict_filters7['20180314'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[2].scatter(twos, dict_filters7['20180314'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[2].scatter(threes, dict_filters7['20180316'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[2].scatter(threes, dict_filters7['20180316'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[2].scatter(fours, dict_filters7['20180322'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[2].scatter(fours, dict_filters7['20180322'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[2].scatter(fives, dict_filters7['20180511'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s)		
		ax[2].scatter(fives, dict_filters7['20180511'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s)
		
		ax[2].scatter(sixes, dict_filters7['20181102'][0]/1.e3, c=np.arange(generations), cmap=my_cmap_lp, marker='^', edgecolor=my_edgecolor, s=my_s, label='HP')		
		ax[2].scatter(sixes, dict_filters7['20181102'][1]/1.e3, c=np.arange(generations), cmap=my_cmap_hp, marker='v', edgecolor=my_edgecolor, s=my_s, label='LP')
		
		
		
		
		
		
		
		

		

		ax[0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[0].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
		ax[1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[1].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
		ax[2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		ax[2].set_ylabel('Frequency [kHz]', fontsize=font_big)
		
			
		ax[0].set_xlabel('MC', fontsize=font_big)
		
		ax[1].set_xlabel('MC', fontsize=font_big)
		
		ax[2].set_xlabel('MC', fontsize=font_big)

		
		
		ax[0].legend(fontsize=font_label, ncol=2, loc='upper center')
		ax[0].legend(fontsize=font_label, ncol=2, loc='upper center')
		ax[0].legend(fontsize=font_label, ncol=2, loc='upper center')
		
		ax[1].legend(fontsize=font_label, ncol=2, loc='upper center')
		ax[1].legend(fontsize=font_label, ncol=2, loc='upper center')
		ax[1].legend(fontsize=font_label, ncol=2, loc='upper center')
		
		ax[2].legend(fontsize=font_label, ncol=2, loc='upper center')
		ax[2].legend(fontsize=font_label, ncol=2, loc='upper center')
		ax[2].legend(fontsize=font_label, ncol=2, loc='upper center')
		
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0].set_title('Level 3', fontsize=font_big)
		
		ax[1].set_title('Level 5', fontsize=font_big)
		
		ax[2].set_title('Level 7', fontsize=font_big)

		
		
		# ax[0][0].set_xlim(left=0, right=10)
		# ax[0][1].set_xlim(left=0, right=10)
		# ax[0][2].set_xlim(left=0, right=10)
		
		# ax[1][0].set_xlim(left=0, right=10)
		# ax[1][1].set_xlim(left=0, right=10)
		# ax[1][2].set_xlim(left=0, right=10)
		

		valtop = 500
		ax[0].set_ylim(bottom=0, top=valtop)
		ax[1].set_ylim(bottom=0, top=valtop)
		ax[2].set_ylim(bottom=0, top=valtop)
		
		# ax[1][0].set_ylim(bottom=0, top=valtop)
		# ax[1][1].set_ylim(bottom=0, top=valtop)
		# ax[1][2].set_ylim(bottom=0, top=valtop)

		
		
		# ax[0][0].set_yticks([0, 100, 200, 300, 400, 500])
		# ax[0][1].set_yticks([0, 100, 200, 300, 400, 500])
		# ax[0][2].set_yticks([0, 100, 200, 300, 400, 500])
		
		ax[0].set_xticklabels([1, 2, 4, 7, 9, 10])
		ax[1].set_xticklabels([1, 2, 4, 7, 9, 10])
		ax[2].set_xticklabels([1, 2, 4, 7, 9, 10])
		
		ax[0].set_xticks([1, 2, 3, 4, 5, 6])
		ax[1].set_xticks([1, 2, 3, 4, 5, 6])
		ax[2].set_xticks([1, 2, 3, 4, 5, 6])
		

		
		
		ax[0].tick_params(axis='both', labelsize=font_little)

		
		ax[1].tick_params(axis='both', labelsize=font_little)

		
		ax[2].tick_params(axis='both', labelsize=font_little)
		
		ax[0].grid(axis='y', alpha=0.75, linestyle='--')
		ax[1].grid(axis='y', alpha=0.75, linestyle='--')
		ax[2].grid(axis='y', alpha=0.75, linestyle='--')
		
		leg2 = ax[2].legend(loc='upper center', fontsize=font_label)
		LH2 = leg2.legendHandles
		LH2[0].set_color('white')
		LH2[0].set_edgecolor('black')
		LH2[1].set_color('white')
		LH2[1].set_edgecolor('black')
		
		leg1 = ax[1].legend(loc='upper center', fontsize=font_label)
		LH1 = leg1.legendHandles
		LH1[0].set_color('white')
		LH1[0].set_edgecolor('black')
		LH1[1].set_color('white')
		LH1[1].set_edgecolor('black')
		
		leg0 = ax[0].legend(loc='upper center', fontsize=font_label)
		LH0 = leg0.legendHandles
		LH0[0].set_color('white')
		LH0[0].set_edgecolor('black')
		LH0[1].set_color('white')
		LH0[1].set_edgecolor('black')
		
		
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
			ax_it.yaxis.offsetText.set_visible(True)
		
		
		# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
		# fig.colorbar(im)
		
		fig.subplots_adjust(right=0.9)
		cbar_ax = fig.add_axes([0.935, 0.17, 0.015, 0.75])
		mycbar = fig.colorbar(im, cax=cbar_ax)
		mycbar.set_label('N° generations', fontsize=font_big)
		mycbar.ax.tick_params(labelsize=font_little)
		
		plt.show()
	
	
	
	elif config['mode'] == 'plot_genetic_distr':
		# Visualizing 4-D mix data using scatter plots
		# leveraging the concepts of hue and depth
		
		print('Select XLXs bests')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		
		D_10h_f_f_r = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		D_5h_f_g = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		D_5h_sb_f_g_c = {'20180227':[], '20180314':[], '20180316':[], '20180322':[], '20180511':[], '20181102':[]}
		
		# mydict = pd.read_excel(filepath, sheetname=config['sheet'])
		# mydict = mydict.to_dict(orient='list')
		# Feature += mydict[config['feature']]
		norm = 15.
		for filepath in Filepaths:			
			mydict = pd.read_excel(filepath)
			mydict = mydict.to_dict(orient='list')
			
			# print(type(mydict))
			# print(mydict.keys())
			# print()
			# sys.exit()
			
			D_10h_f_f_r['20180227'].append(np.mean(np.array(mydict['10h_f_f_r'][0:3])))
			D_10h_f_f_r['20180314'].append(np.mean(np.array(mydict['10h_f_f_r'][3:6])))
			D_10h_f_f_r['20180316'].append(np.mean(np.array(mydict['10h_f_f_r'][6:9])))
			D_10h_f_f_r['20180322'].append(np.mean(np.array(mydict['10h_f_f_r'][9:12])))
			D_10h_f_f_r['20180511'].append(np.mean(np.array(mydict['10h_f_f_r'][12:15])))
			D_10h_f_f_r['20181102'].append(np.mean(np.array(mydict['10h_f_f_r'][15:18])))
			
			D_5h_f_g['20180227'].append(np.mean(np.array(mydict['5h_f_g'][0:3])))
			D_5h_f_g['20180314'].append(np.mean(np.array(mydict['5h_f_g'][3:6])))
			D_5h_f_g['20180316'].append(np.mean(np.array(mydict['5h_f_g'][6:9])))
			D_5h_f_g['20180322'].append(np.mean(np.array(mydict['5h_f_g'][9:12])))
			D_5h_f_g['20180511'].append(np.mean(np.array(mydict['5h_f_g'][12:15])))
			D_5h_f_g['20181102'].append(np.mean(np.array(mydict['5h_f_g'][15:18])))
			
			D_5h_sb_f_g_c['20180227'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][0:3])))
			D_5h_sb_f_g_c['20180314'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][3:6])))
			D_5h_sb_f_g_c['20180316'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][6:9])))
			D_5h_sb_f_g_c['20180322'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][9:12])))
			D_5h_sb_f_g_c['20180511'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][12:15])))
			D_5h_sb_f_g_c['20181102'].append(np.mean(np.array(mydict['5h_sb_f_g_c'][15:18])))
			
			
		
		S_10h_f_f_r = np.zeros(len(D_10h_f_f_r['20180227']))
		S2_10h_f_f_r = np.zeros(len(D_10h_f_f_r['20180227']))
		for i in range(len(D_10h_f_f_r['20180227'])):
			score = 0
			if D_10h_f_f_r['20180227'][i] <= D_10h_f_f_r['20180314'][i]:
				score += 1
			else:
				score -= 1
			if D_10h_f_f_r['20180314'][i] <= D_10h_f_f_r['20180316'][i]:
				score += 1
			else:
				score -= 1
			if D_10h_f_f_r['20180316'][i] <= D_10h_f_f_r['20180322'][i]:
				score += 1
			else:
				score -= 1
			if D_10h_f_f_r['20180322'][i] <= D_10h_f_f_r['20180511'][i]:
				score += 1
			else:
				score -= 1
			if D_10h_f_f_r['20180511'][i] <= D_10h_f_f_r['20181102'][i]:
				score += 1
			else:
				score -= 1
			S_10h_f_f_r[i] = score
			S2_10h_f_f_r[i] = D_10h_f_f_r['20181102'][i] - D_10h_f_f_r['20180227'][i]
		
		S_5h_f_g = np.zeros(len(D_10h_f_f_r['20180227']))
		S2_5h_f_g = np.zeros(len(D_10h_f_f_r['20180227']))
		for i in range(len(D_5h_f_g['20180227'])):
			score = 0
			if D_5h_f_g['20180227'][i] <= D_5h_f_g['20180314'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_f_g['20180314'][i] <= D_5h_f_g['20180316'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_f_g['20180316'][i] <= D_5h_f_g['20180322'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_f_g['20180322'][i] <= D_5h_f_g['20180511'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_f_g['20180511'][i] <= D_5h_f_g['20181102'][i]:
				score += 1
			else:
				score -= 1
			S_5h_f_g[i] = score
			S2_5h_f_g[i] = D_5h_f_g['20181102'][i] - D_5h_f_g['20180227'][i]
		
		S_5h_sb_f_g_c = np.zeros(len(D_10h_f_f_r['20180227']))
		S2_5h_sb_f_g_c = np.zeros(len(D_10h_f_f_r['20180227']))
		for i in range(len(D_5h_sb_f_g_c['20180227'])):
			score = 0
			if D_5h_sb_f_g_c['20180227'][i] <= D_5h_sb_f_g_c['20180314'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_sb_f_g_c['20180314'][i] <= D_5h_sb_f_g_c['20180316'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_sb_f_g_c['20180316'][i] <= D_5h_sb_f_g_c['20180322'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_sb_f_g_c['20180322'][i] <= D_5h_sb_f_g_c['20180511'][i]:
				score += 1
			else:
				score -= 1
			if D_5h_sb_f_g_c['20180511'][i] <= D_5h_sb_f_g_c['20181102'][i]:
				score += 1
			else:
				score -= 1
			S_5h_sb_f_g_c[i] = score
			S2_5h_sb_f_g_c[i] = D_5h_sb_f_g_c['20181102'][i] - D_5h_sb_f_g_c['20180227'][i]
		
		
		Scores = (S_10h_f_f_r + S_5h_f_g + S_5h_sb_f_g_c)/norm
		# Scores2 = S2_10h_f_f_r + S2_5h_f_g + S2_5h_sb_f_g_c
		
		
		# print(D_10h_f_f_r)
		# print(D_5h_f_g)
		# print(D_5h_sb_f_g_c)

		# sys.exit()
		
		# plt.plot(Scores2, '-o')
		# plt.show()
		
		# for i in range(len(Scores)):
			# if Scores[i] >= 14:
				# print(i+14)
		# sys.exit()
		
		distr = {'c10_w005_p2':[], 'c20_w005_p2':[], 'c10_w01_p2':[], 'c20_w01_p2':[], 'c10_w015_p2':[], 'c20_w015_p2':[], 'c10_w005_p4':[], 'c20_w005_p4':[], 'c10_w01_p4':[], 'c20_w01_p4':[], 'c10_w015_p4':[], 'c20_w015_p4':[], 'c10_w005_p7':[], 'c20_w005_p7':[], 'c10_w01_p7':[], 'c20_w01_p7':[], 'c10_w015_p7':[], 'c20_w015_p7':[]}
		
		
		print('Select XLXs Configs')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_Configs = filedialog.askopenfilenames()
		root.destroy()
		
		for filepath_config, score in zip(Filepaths_Configs, Scores):
			gen_config = read_pickle(filepath_config)
			# print(gen_config.keys())
			if gen_config['clusters'] == 10 and gen_config['mutation'] == 0.05 and gen_config['parents'] == 2:
				distr['c10_w005_p2'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.05 and gen_config['parents'] == 2:
				distr['c20_w005_p2'].append(score)
				
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.1 and gen_config['parents'] == 2:
				distr['c10_w01_p2'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.1 and gen_config['parents'] == 2:
				distr['c20_w01_p2'].append(score)
			
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.15 and gen_config['parents'] == 2:
				distr['c10_w015_p2'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.15 and gen_config['parents'] == 2:
				distr['c20_w015_p2'].append(score)
			
			
			
			
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.05 and gen_config['parents'] == 4:
				distr['c10_w005_p4'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.05 and gen_config['parents'] == 4:
				distr['c20_w005_p4'].append(score)
				
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.1 and gen_config['parents'] == 4:
				distr['c10_w01_p4'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.1 and gen_config['parents'] == 4:
				distr['c20_w01_p4'].append(score)
			
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.15 and gen_config['parents'] == 4:
				distr['c10_w015_p4'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.15 and gen_config['parents'] == 4:
				distr['c20_w015_p4'].append(score)
			
			
			
			
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.05 and gen_config['parents'] == 7:
				distr['c10_w005_p7'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.05 and gen_config['parents'] == 7:
				distr['c20_w005_p7'].append(score)
				
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.1 and gen_config['parents'] == 7:
				distr['c10_w01_p7'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.1 and gen_config['parents'] == 7:
				distr['c20_w01_p7'].append(score)
			
			
			elif gen_config['clusters'] == 10 and gen_config['mutation'] == 0.15 and gen_config['parents'] == 7:
				distr['c10_w015_p7'].append(score)
				
			elif gen_config['clusters'] == 20 and gen_config['mutation'] == 0.15 and gen_config['parents'] == 7:
				distr['c20_w015_p7'].append(score)

			else:
				print('error scores')
				sys.exit()
		
		
		
		
		width = 1
		hrange = (0/norm, 15/norm)
		nbins = 15
		font_big = 17
		font_little = 15
		font_label = 13
		
		from matplotlib import font_manager
		del font_manager.weight_dict['roman']
		font_manager._rebuild()
		plt.rcParams['font.family'] = 'Times New Roman'	
		
		fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.75)
		fig.set_size_inches(14.2, 7.2)		
		ax[0][0].hist(distr['c10_w005_p2'], rwidth=width, range=hrange, bins=nbins)
		ax[0][1].hist(distr['c10_w01_p2'], rwidth=width, range=hrange, bins=nbins)
		ax[0][2].hist(distr['c10_w015_p2'], rwidth=width, range=hrange, bins=nbins)
		
		ax[1][0].hist(distr['c10_w005_p4'], rwidth=width, range=hrange, bins=nbins)
		ax[1][1].hist(distr['c10_w01_p4'], rwidth=width, range=hrange, bins=nbins)
		ax[1][2].hist(distr['c10_w015_p4'], rwidth=width, range=hrange, bins=nbins)
		
		ax[2][0].hist(distr['c10_w005_p7'], rwidth=width, range=hrange, bins=nbins)
		ax[2][1].hist(distr['c10_w01_p7'], rwidth=width, range=hrange, bins=nbins)
		ax[2][2].hist(distr['c10_w015_p7'], rwidth=width, range=hrange, bins=nbins)	

		ax[0][0].set_ylabel('Occurrence', fontsize=font_big)
		ax[0][1].set_ylabel('Occurrence', fontsize=font_big)
		ax[0][2].set_ylabel('Occurrence', fontsize=font_big)
		
		ax[1][0].set_ylabel('Occurrence', fontsize=font_big)
		ax[1][1].set_ylabel('Occurrence', fontsize=font_big)
		ax[1][2].set_ylabel('Occurrence', fontsize=font_big)
		
		ax[2][0].set_ylabel('Occurrence', fontsize=font_big)
		ax[2][1].set_ylabel('Occurrence', fontsize=font_big)
		ax[2][2].set_ylabel('Occurrence', fontsize=font_big)
		
		ax[0][0].set_xlabel('Monotonicity', fontsize=font_big)
		ax[0][1].set_xlabel('Monotonicity', fontsize=font_big)
		ax[0][2].set_xlabel('Monotonicity', fontsize=font_big)
		
		ax[1][0].set_xlabel('Monotonicity', fontsize=font_big)
		ax[1][1].set_xlabel('Monotonicity', fontsize=font_big)
		ax[1][2].set_xlabel('Monotonicity', fontsize=font_big)
		
		ax[2][0].set_xlabel('Monotonicity', fontsize=font_big)
		ax[2][1].set_xlabel('Monotonicity', fontsize=font_big)
		ax[2][2].set_xlabel('Monotonicity', fontsize=font_big)
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$C$=10, $W$=0.05, $P$=2', fontsize=font_big)
		ax[0][1].set_title('$C$=10, $W$=0.1, $P$=2', fontsize=font_big)
		ax[0][2].set_title('$C$=10, $W$=0.15, $P$=2', fontsize=font_big)
		
		ax[1][0].set_title('$C$=10, $W$=0.05, $P$=4', fontsize=font_big)
		ax[1][1].set_title('$C$=10, $W$=0.1, $P$=4', fontsize=font_big)
		ax[1][2].set_title('$C$=10, $W$=0.15, $P$=4', fontsize=font_big)
		
		ax[2][0].set_title('$C$=10, $W$=0.05, $P$=7', fontsize=font_big)
		ax[2][1].set_title('$C$=10, $W$=0.1, $P$=7', fontsize=font_big)
		ax[2][2].set_title('$C$=10, $W$=0.15, $P$=7', fontsize=font_big)	
		
		ax[0][0].set_xlim(left=0, right=1)
		ax[0][1].set_xlim(left=0, right=1)
		ax[0][2].set_xlim(left=0, right=1)
		
		ax[1][0].set_xlim(left=0, right=1)
		ax[1][1].set_xlim(left=0, right=1)
		ax[1][2].set_xlim(left=0, right=1)
		
		
		ax[2][0].set_xlim(left=0, right=1)
		ax[2][1].set_xlim(left=0, right=1)
		ax[2][2].set_xlim(left=0, right=1)
		
		valtop = 9
		ax[0][0].set_ylim(bottom=0, top=valtop)
		ax[0][1].set_ylim(bottom=0, top=valtop)
		ax[0][2].set_ylim(bottom=0, top=valtop)
		
		ax[1][0].set_ylim(bottom=0, top=valtop)
		ax[1][1].set_ylim(bottom=0, top=valtop)
		ax[1][2].set_ylim(bottom=0, top=valtop)
		
		ax[2][0].set_ylim(bottom=0, top=valtop)
		ax[2][1].set_ylim(bottom=0, top=valtop)
		ax[2][2].set_ylim(bottom=0, top=valtop)
		
		
		ax[0][0].set_yticks([0, 3, 6, 9])
		ax[0][1].set_yticks([0, 3, 6, 9])
		ax[0][2].set_yticks([0, 3, 6, 9])
		
		ax[1][0].set_yticks([0, 3, 6, 9])
		ax[1][1].set_yticks([0, 3, 6, 9])
		ax[1][2].set_yticks([0, 3, 6, 9])
		
		ax[2][0].set_yticks([0, 3, 6, 9])
		ax[2][1].set_yticks([0, 3, 6, 9])
		ax[2][2].set_yticks([0, 3, 6, 9])
		
		
		
		ax[0][0].tick_params(axis='both', labelsize=font_little)
		ax[0][1].tick_params(axis='both', labelsize=font_little)
		ax[0][2].tick_params(axis='both', labelsize=font_little)
		
		ax[1][0].tick_params(axis='both', labelsize=font_little)
		ax[1][1].tick_params(axis='both', labelsize=font_little)
		ax[1][2].tick_params(axis='both', labelsize=font_little)
		
		ax[2][0].tick_params(axis='both', labelsize=font_little)
		ax[2][1].tick_params(axis='both', labelsize=font_little)
		ax[2][2].tick_params(axis='both', labelsize=font_little)
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
			ax_it.yaxis.offsetText.set_visible(True)
		
		plt.show()
		
		fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
		plt.subplots_adjust(wspace=0.275, left=0.065, right=0.98, bottom=0.125, top=0.92, hspace=0.75)
		fig.set_size_inches(14.2, 7.2)		
		ax[0][0].hist(distr['c20_w005_p2'], rwidth=width, range=hrange, bins=nbins)
		ax[0][1].hist(distr['c20_w01_p2'], rwidth=width, range=hrange, bins=nbins)
		ax[0][2].hist(distr['c20_w015_p2'], rwidth=width, range=hrange, bins=nbins)
		
		ax[1][0].hist(distr['c20_w005_p4'], rwidth=width, range=hrange, bins=nbins)
		ax[1][1].hist(distr['c20_w01_p4'], rwidth=width, range=hrange, bins=nbins)
		ax[1][2].hist(distr['c20_w015_p4'], rwidth=width, range=hrange, bins=nbins)
		
		ax[2][0].hist(distr['c20_w005_p7'], rwidth=width, range=hrange, bins=nbins)
		ax[2][1].hist(distr['c20_w01_p7'], rwidth=width, range=hrange, bins=nbins)
		ax[2][2].hist(distr['c20_w015_p7'], rwidth=width, range=hrange, bins=nbins)	

		ax[0][0].set_ylabel('Occurrence', fontsize=font_big)
		ax[0][1].set_ylabel('Occurrence', fontsize=font_big)
		ax[0][2].set_ylabel('Occurrence', fontsize=font_big)
		
		ax[1][0].set_ylabel('Occurrence', fontsize=font_big)
		ax[1][1].set_ylabel('Occurrence', fontsize=font_big)
		ax[1][2].set_ylabel('Occurrence', fontsize=font_big)
		
		ax[2][0].set_ylabel('Occurrence', fontsize=font_big)
		ax[2][1].set_ylabel('Occurrence', fontsize=font_big)
		ax[2][2].set_ylabel('Occurrence', fontsize=font_big)
		
		ax[0][0].set_xlabel('Monotonicity', fontsize=font_big)
		ax[0][1].set_xlabel('Monotonicity', fontsize=font_big)
		ax[0][2].set_xlabel('Monotonicity', fontsize=font_big)
		
		ax[1][0].set_xlabel('Monotonicity', fontsize=font_big)
		ax[1][1].set_xlabel('Monotonicity', fontsize=font_big)
		ax[1][2].set_xlabel('Monotonicity', fontsize=font_big)
		
		ax[2][0].set_xlabel('Monotonicity', fontsize=font_big)
		ax[2][1].set_xlabel('Monotonicity', fontsize=font_big)
		ax[2][2].set_xlabel('Monotonicity', fontsize=font_big)
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		ax[0][0].set_title('$C$=20, $W$=0.05, $P$=2', fontsize=font_big)
		ax[0][1].set_title('$C$=20, $W$=0.1, $P$=2', fontsize=font_big)
		ax[0][2].set_title('$C$=20, $W$=0.15, $P$=2', fontsize=font_big)
		
		ax[1][0].set_title('$C$=20, $W$=0.05, $P$=4', fontsize=font_big)
		ax[1][1].set_title('$C$=20, $W$=0.1, $P$=4', fontsize=font_big)
		ax[1][2].set_title('$C$=20, $W$=0.15, $P$=4', fontsize=font_big)
		
		ax[2][0].set_title('$C$=20, $W$=0.05, $P$=7', fontsize=font_big)
		ax[2][1].set_title('$C$=20, $W$=0.1, $P$=7', fontsize=font_big)
		ax[2][2].set_title('$C$=20, $W$=0.15, $P$=7', fontsize=font_big)	
		
		ax[0][0].set_xlim(left=0, right=1)
		ax[0][1].set_xlim(left=0, right=1)
		ax[0][2].set_xlim(left=0, right=1)
		
		ax[1][0].set_xlim(left=0, right=1)
		ax[1][1].set_xlim(left=0, right=1)
		ax[1][2].set_xlim(left=0, right=1)
		
		
		ax[2][0].set_xlim(left=0, right=1)
		ax[2][1].set_xlim(left=0, right=1)
		ax[2][2].set_xlim(left=0, right=1)
		
		valtop = 9
		ax[0][0].set_ylim(bottom=0, top=valtop)
		ax[0][1].set_ylim(bottom=0, top=valtop)
		ax[0][2].set_ylim(bottom=0, top=valtop)
		
		ax[1][0].set_ylim(bottom=0, top=valtop)
		ax[1][1].set_ylim(bottom=0, top=valtop)
		ax[1][2].set_ylim(bottom=0, top=valtop)
		
		ax[2][0].set_ylim(bottom=0, top=valtop)
		ax[2][1].set_ylim(bottom=0, top=valtop)
		ax[2][2].set_ylim(bottom=0, top=valtop)
		
		
		ax[0][0].set_yticks([0, 3, 6, 9])
		ax[0][1].set_yticks([0, 3, 6, 9])
		ax[0][2].set_yticks([0, 3, 6, 9])
		
		ax[1][0].set_yticks([0, 3, 6, 9])
		ax[1][1].set_yticks([0, 3, 6, 9])
		ax[1][2].set_yticks([0, 3, 6, 9])
		
		ax[2][0].set_yticks([0, 3, 6, 9])
		ax[2][1].set_yticks([0, 3, 6, 9])
		ax[2][2].set_yticks([0, 3, 6, 9])
		
		
		
		ax[0][0].tick_params(axis='both', labelsize=font_little)
		ax[0][1].tick_params(axis='both', labelsize=font_little)
		ax[0][2].tick_params(axis='both', labelsize=font_little)
		
		ax[1][0].tick_params(axis='both', labelsize=font_little)
		ax[1][1].tick_params(axis='both', labelsize=font_little)
		ax[1][2].tick_params(axis='both', labelsize=font_little)
		
		ax[2][0].tick_params(axis='both', labelsize=font_little)
		ax[2][1].tick_params(axis='both', labelsize=font_little)
		ax[2][2].tick_params(axis='both', labelsize=font_little)
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
			ax_it.yaxis.offsetText.set_visible(True)
		
		plt.show()
		
	
	else:
		print('Mode Unknown 98359')
		
		
		
	return

def genetic_optimal_filter_A(x, fs, levels, num_generations, num_parents_mating, freq_values, freq_range, weight_mutation=None, inter=None):
	if weight_mutation == None:
		weight_mutation = 0.05	
	new_population = generate_population_filter(levels, fs, inter)
	print('+++++++++++++++++++++++++++++++++++++initial population!')		
	print(new_population)
	print(len(new_population))

	sol_per_pop = len(new_population)
	pop_size = (sol_per_pop, 2)		

	print('+++++++++++++++++++++++++++++++++++++best fitness initial population!')
	# ini_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
	ini_fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population)
	
	# print(np.nanmax(ini_fitness))
	# print(select_mating_pool(new_population, ini_fitness, 1))
	count = 0
	
		
	for generation in range(num_generations):		
		# fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population)
		
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		offspring_crossover = crossover_mean(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		# offspring_crossover = crossover_maxmin(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		# offspring_crossover = crossover_minmax(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		
		offspring_mutation = mutation(offspring_crossover, weight_mutation)

		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation
	
		print('+++++++++++++++++++++++++++++++++++++best fitness final population!')	
		# end_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		end_fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population)
		print(np.max(end_fitness))
		best = select_mating_pool(new_population, end_fitness, 1)
		print(best)
		count += 1
	# sys.exit()
	# mydict = {'new_population':new_population, 'elapsed_generations':count, 'config':config, 'best':best, 'filename':filename}
	hp = best[0][0] + best[0][1]/2.
	lp = best[0][0] - best[0][1]/2.
	# save_pickle('config_generation_' + str(count) + '_' + config['name'] + '.pkl', mydict)
	return lp, hp

def genetic_optimal_filter_C(x, fs, levels, num_generations, num_parents_mating, freq_values, freq_range, weight_mutation, clusters, filename=None):
	
	new_population = generate_population_filter_clusters(levels, fs, clusters)
	print('+++++++++++++++++++++++++++++++++++++initial population!')		
	print(new_population)
	print(len(new_population))

	sol_per_pop = len(new_population)
	pop_size = (sol_per_pop, 2)		

	print('+++++++++++++++++++++++++++++++++++++best fitness initial population!')
	# ini_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
	ini_fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population)
	
	# print(np.nanmax(ini_fitness))
	count = 0
	best = select_mating_pool(new_population, ini_fitness, 1)
	save_pickle(filename + '_best_generation_' + str(count) + '_lvl_' + str(levels) + '.pkl', best)
	
		
	for generation in range(num_generations):		
		# fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population)
		
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		# offspring_crossover = crossover_bool(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		offspring_crossover = crossover_bool(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		# offspring_crossover = crossover_maxmin(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		# offspring_crossover = crossover_minmax(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		
		offspring_mutation = mutation2(offspring_crossover, weight_mutation)

		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation
	
		print('+++++++++++++++++++++++++++++++++++++best fitness final population!')	
		# end_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		end_fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population)
		# print(np.max(end_fitness))
		best = select_mating_pool(new_population, end_fitness, 1)
		print(best)
		save_pickle(filename + '_best_generation_' + str(count) + '_lvl_' + str(levels) + '.pkl', best)
		count += 1
	# sys.exit()
	# mydict = {'new_population':new_population, 'elapsed_generations':count, 'config':config, 'best':best, 'filename':filename}
	hp = best[0][0] + best[0][1]/2.
	lp = best[0][0] - best[0][1]/2.
	# save_pickle('config_generation_' + str(count) + '_' + config['name'] + '.pkl', mydict)
	return lp, hp

def genetic_optimal_filter_C_kurt_sen(x, fs, levels, num_generations, num_parents_mating, weight_mutation, clusters, filename=None):
	
	new_population = generate_population_filter_clusters(levels, fs, clusters)
	print('+++++++++++++++++++++++++++++++++++++initial population!')		
	print(new_population)
	print(len(new_population))

	sol_per_pop = len(new_population)
	pop_size = (sol_per_pop, 2)		

	print('+++++++++++++++++++++++++++++++++++++best fitness initial population!')
	# ini_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
	ini_fitness = cal_pop_fitness_kurt_sen(x, freq_values, freq_range, fs, new_population)
	
	# print(np.nanmax(ini_fitness))
	count = 0
	best = select_mating_pool(new_population, ini_fitness, 1)
	save_pickle(filename + '_best_generation_' + str(count) + '_lvl_' + str(levels) + '.pkl', best)
	
		
	for generation in range(num_generations):		
		# fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		fitness = cal_pop_fitness_kurt_sen(x, freq_values, freq_range, fs, new_population)
		
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		# offspring_crossover = crossover_bool(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		offspring_crossover = crossover_bool(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		# offspring_crossover = crossover_maxmin(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		# offspring_crossover = crossover_minmax(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		
		offspring_mutation = mutation2(offspring_crossover, weight_mutation)

		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation
	
		print('+++++++++++++++++++++++++++++++++++++best fitness final population!')	
		# end_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		end_fitness = cal_pop_fitness_kurt_sen(x, freq_values, freq_range, fs, new_population)
		# print(np.max(end_fitness))
		best = select_mating_pool(new_population, end_fitness, 1)
		print(best)
		save_pickle(filename + '_best_generation_' + str(count) + '_lvl_' + str(levels) + '.pkl', best)
		count += 1
	# sys.exit()
	# mydict = {'new_population':new_population, 'elapsed_generations':count, 'config':config, 'best':best, 'filename':filename}
	hp = best[0][0] + best[0][1]/2.
	lp = best[0][0] - best[0][1]/2.
	# save_pickle('config_generation_' + str(count) + '_' + config['name'] + '.pkl', mydict)
	return lp, hp

def genetic_optimal_filter_B(x, fs, levels, num_generations, num_parents_mating, freq_values1, freq_range1, freq_values2, freq_range2, weight_mutation=None, inter=None):
	if weight_mutation == None:
		weight_mutation = 0.05	
	new_population = generate_population_filter(levels, fs, inter)
	print('+++++++++++++++++++++++++++++++++++++initial population!')		
	print(new_population)
	print(len(new_population))

	sol_per_pop = len(new_population)
	pop_size = (sol_per_pop, 2)		

	print('+++++++++++++++++++++++++++++++++++++best fitness initial population!')
	# ini_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
	ini_fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr_2(x, freq_values1, freq_range1, freq_values2, freq_range2, fs, new_population)
	
	# print(np.nanmax(ini_fitness))
	# print(select_mating_pool(new_population, ini_fitness, 1))
	count = 0
	
		
	for generation in range(num_generations):		
		# fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr_2(x, freq_values1, freq_range1, freq_values2, freq_range2, fs, new_population)
		
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		offspring_crossover = crossover_mean(parents, offspring_size=(pop_size[0]-parents.shape[0], 2))
		
		offspring_mutation = mutation(offspring_crossover, weight_mutation)

		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation
	
		print('+++++++++++++++++++++++++++++++++++++best fitness final population!')	
		# end_fitness = cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population)
		end_fitness = cal_pop_fitness_hilbert_env_Ncomp_mpr_2(x, freq_values1, freq_range1, freq_values2, freq_range2, fs, new_population)
		print(np.max(end_fitness))
		best = select_mating_pool(new_population, end_fitness, 1)
		print(best)
		count += 1
	# sys.exit()
	# mydict = {'new_population':new_population, 'elapsed_generations':count, 'config':config, 'best':best, 'filename':filename}
	hp = best[0][0] + best[0][1]/2.
	lp = best[0][0] - best[0][1]/2.
	# save_pickle('config_generation_' + str(count) + '_' + config['name'] + '.pkl', mydict)
	return lp, hp

def generate_population_filter(nlevel, fs, inter=None):
	Level_w = np.arange(1,nlevel+1)
	Level_w = np.array([Level_w, Level_w + np.log2(3.)-1])
	Level_w = sorted(Level_w.ravel())
	Level_w = np.append(0,Level_w[0:2*nlevel-1])
	freq_w = np.array([fs/2**(p+1) for p in Level_w])
	freq_c = fs*(np.arange(0,3*2.0**nlevel-1+1))/(3*2**(nlevel+1)) + 1.0/(3.*2.**(2+nlevel))
	print('centers: ', freq_c)
	print('widths: ', freq_w)
	
	#Generate population from centers and widths from kurtogram
	new_population = []
	for center in freq_c:
		for width in freq_w:
			new_population.append([center, width])
	new_population = np.array(new_population)
	# print(new_population)
	# print('+++++++++++++++++++++++++++++++++++++')
	
	
	#Adapts population to avoid negative frequencies OR above nyquist by eliminating
	new_population = list(new_population)
	index_to_keep = []
	print(len(new_population))
	for k in range(len(new_population)):
		if (new_population[k][0] - new_population[k][1]/2 > 0) and (new_population[k][0] + new_population[k][1]/2 < fs/2.):
			index_to_keep.append(k)
	new_population = np.array(new_population)
	new_population = new_population[index_to_keep]
	
	if inter != None:
		new_population = np.array([new_population[inter*i] for i in range(int(len(new_population)/inter))])
	
	return new_population

def generate_population_filter_clusters(nlevel, fs, clusters):
	Level_w = np.arange(1,nlevel+1)
	Level_w = np.array([Level_w, Level_w + np.log2(3.)-1])
	Level_w = sorted(Level_w.ravel())
	Level_w = np.append(0,Level_w[0:2*nlevel-1])
	freq_w = np.array([fs/2**(p+1) for p in Level_w])
	freq_c = fs*(np.arange(0,3*2.0**nlevel-1+1))/(3*2**(nlevel+1)) + 1.0/(3.*2.**(2+nlevel))
	print('centers: ', freq_c)
	print('widths: ', freq_w)
	
	#Generate population from centers and widths from kurtogram
	new_population = []
	for center in freq_c:
		for width in freq_w:
			new_population.append([center, width])
	new_population = np.array(new_population)
	# print(new_population)
	# print('+++++++++++++++++++++++++++++++++++++')
	
	
	#Adapts population to avoid negative frequencies OR above nyquist by eliminating
	new_population = list(new_population)
	index_to_keep = []
	print(len(new_population))
	for k in range(len(new_population)):
		if (new_population[k][0] - new_population[k][1]/2 > 0) and (new_population[k][0] + new_population[k][1]/2 < fs/2.):
			index_to_keep.append(k)
	new_population = np.array(new_population)
	new_population = new_population[index_to_keep]
	
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(new_population)
	
	return kmeans.cluster_centers_


def cal_pop_fitness_corr(x, y, fs, new_population):
	n = len(new_population)
	fitness = np.zeros(n)
	for k in range(n):
		high_pass = new_population[k][0] + new_population[k][1]/2.
		low_pass = new_population[k][0] - new_population[k][1]/2.
		x_filt = butter_bandpass(x, fs, [low_pass, high_pass], 3)
		fitness[k] = np.corrcoef(x_filt, y)[0][1]
	return fitness

def cal_pop_fitness_hilbert_env_comp_mpr(x, freq_value, width_freq_range, fs, new_population):
	n = len(new_population)
	fitness = np.zeros(n)
	for k in range(n):
		high_pass = new_population[k][0] + new_population[k][1]/2.
		low_pass = new_population[k][0] - new_population[k][1]/2.
		x_filt = butter_bandpass(x, fs, [low_pass, high_pass], 3)
		x_env = hilbert_demodulation(x_filt)
		magENV, f, df = mag_fft(x_env, fs)

		mag_freq_value = amp_component_zone(X=magENV, df=df, freq=freq_value, tol=1.0)
		avg_freq_range = avg_in_band(magENV, df, low=freq_value-width_freq_range/2., high=freq_value+width_freq_range/2.)
		fitness[k] = 20*np.log10((mag_freq_value-avg_freq_range)/avg_freq_range)


	return fitness

def cal_pop_fitness_hilbert_env_Ncomp_mpr(x, freq_values, freq_range, fs, new_population):
	n = len(new_population)
	fitness = np.zeros(n)
	for k in range(n):
		high_pass = new_population[k][0] + new_population[k][1]/2.
		low_pass = new_population[k][0] - new_population[k][1]/2.
		x_filt = butter_bandpass(x, fs, [low_pass, high_pass], 3)
		x_env = hilbert_demodulation(x_filt)
		magENV, f, df = mag_fft(x_env, fs)
		
		mag_freq_value = 0.
		for freq_value in freq_values:
			mag_freq_value += amp_component_zone(X=magENV, df=df, freq=freq_value, tol=2.0)
		
		avg_freq_range = avg_in_band(magENV, df, low=freq_range[0], high=freq_range[1])
		fitness[k] = 20*np.log10((mag_freq_value-avg_freq_range)/avg_freq_range)


	return fitness

def cal_pop_fitness_kurt_sen(x, fs, new_population):
	n = len(new_population)
	fitness = np.zeros(n)
	for k in range(n):
		high_pass = new_population[k][0] + new_population[k][1]/2.
		low_pass = new_population[k][0] - new_population[k][1]/2.
		x_filt = butter_bandpass(x, fs, [low_pass, high_pass], 3)
		
		kurt = scipy.stats.kurtosis(x_filt, fisher=False)
		sen = shannon_entropy(x_filt)
		fitness[k] = kurt/sen



	return fitness

def cal_pop_fitness_hilbert_env_Ncomp_mpr_2(x, freq_values1, freq_range1, freq_values2, freq_range2, fs, new_population):
	n = len(new_population)
	fitness = np.zeros(n)
	for k in range(n):
		high_pass = new_population[k][0] + new_population[k][1]/2.
		low_pass = new_population[k][0] - new_population[k][1]/2.
		x_filt = butter_bandpass(x, fs, [low_pass, high_pass], 3)
		x_env = hilbert_demodulation(x_filt)
		magENV, f, df = mag_fft(x_env, fs)
		
		mag_freq_value1 = 0.
		for freq_value1 in freq_values1:
			mag_freq_value1 += amp_component_zone(X=magENV, df=df, freq=freq_value1, tol=2.0)
		
		mag_freq_value2 = 0.
		for freq_value2 in freq_values2:
			mag_freq_value2 += amp_component_zone(X=magENV, df=df, freq=freq_value2, tol=2.0)
		
		avg_freq_range1 = avg_in_band(magENV, df, low=freq_range1[0], high=freq_range1[1])		
		avg_freq_range2 = avg_in_band(magENV, df, low=freq_range2[0], high=freq_range2[1])
		
		fitness[k] = 20*np.log10((mag_freq_value1-avg_freq_range1)/avg_freq_range1) + 20*np.log10((mag_freq_value2-avg_freq_range2)/avg_freq_range2)


	return fitness

def select_mating_pool(population, fitness, num_parents):
	parents = np.empty((num_parents, population.shape[1]))			
	for parent_num in range(num_parents):
		max_fitness_idx = np.where(fitness == np.nanmax(fitness))
		max_fitness_idx = max_fitness_idx[0][0]
		parents[parent_num, :] = population[max_fitness_idx, :]
		fitness[max_fitness_idx] = -99999999999
	return parents

def crossover_mean(parents, offspring_size):
	offspring = np.empty(offspring_size)		 
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]
		offspring[k, 0] = (parents[parent1_idx, 0] + parents[parent2_idx, 0])/2.
		offspring[k, 1] = (parents[parent1_idx, 1] + parents[parent2_idx, 1])/2.				 
	return offspring

def crossover_bool(parents, offspring_size):
	offspring = np.empty(offspring_size)		 
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]		
		val = np.random.rand() > 0.5
		# print(val)
		if val == True:
			offspring[k, 0] = parents[parent1_idx, 0] 
			offspring[k, 1] = parents[parent2_idx, 1]
		elif val == False:
			offspring[k, 0] = parents[parent2_idx, 0]
			offspring[k, 1] = parents[parent1_idx, 1]
		else:
			print('fatal error 517 val in bool')
	return offspring

def crossover_maxmin(parents, offspring_size):
	offspring = np.empty(offspring_size)		 
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]
		offspring[k, 0] = np.max(np.array([parents[parent1_idx, 0], parents[parent2_idx, 0]]))
		offspring[k, 1] = np.min(np.array([parents[parent1_idx, 1], parents[parent2_idx, 1]]))		 
	return offspring



def crossover_minmax(parents, offspring_size):
	offspring = np.empty(offspring_size)		 
	for k in range(offspring_size[0]):
		parent1_idx = k%parents.shape[0]
		parent2_idx = (k+1)%parents.shape[0]
		offspring[k, 0] = np.min(np.array([parents[parent1_idx, 0], parents[parent2_idx, 0]]))
		offspring[k, 1] = np.max(np.array([parents[parent1_idx, 1], parents[parent2_idx, 1]]))		 
	return offspring
	
def mutation(offspring_crossover, weight):
	for idx in range(offspring_crossover.shape[0]):
		# min_offspring_idx = np.where(offspring_crossover == np.min(offspring_crossover))
		# min_values = offspring_crossover[min_offspring_idx[0], min_offspring_idx[1]]
		# if len(min_values) == 1:
			# min_value = min_values
		# elif len(min_values) > 1:
			# min_value = min_values[0]
		# else:
			# print('fatal error 6305')
			# sys.exit()		
		
		hp = 1.e6
		lp = -10.
		count = 0
		while hp >= 500000. or lp <= 0. or hp <= lp or hp <= 0. or lp >= 500000.:
			random_value = np.random.uniform(-weight, weight, 1)
			offspring_crossover[idx, 0] = offspring_crossover[idx, 0] * (1. + random_value)
			random_value = np.random.uniform(-weight, weight, 1)
			# random_value = np.random.uniform(-min_value*weight, min_value*weight, 1)
			offspring_crossover[idx, 1] = offspring_crossover[idx, 1] * (1. + random_value)
			hp = offspring_crossover[idx, 0] + offspring_crossover[idx, 1]/2.
			lp = offspring_crossover[idx, 0] - offspring_crossover[idx, 1]/2.
			count += 1
			print(count)
			print(hp)
			print(lp)
			if count >= 500000:
				print('max iter----')
				offspring_crossover[idx, 0] = 250000.
				offspring_crossover[idx, 1] = 250000.
				
		
	return offspring_crossover

def mutation2(offspring_crossover, weight):
	for idx in range(offspring_crossover.shape[0]):
		# min_offspring_idx = np.where(offspring_crossover == np.min(offspring_crossover))
		# min_values = offspring_crossover[min_offspring_idx[0], min_offspring_idx[1]]
		# if len(min_values) == 1:
			# min_value = min_values
		# elif len(min_values) > 1:
			# min_value = min_values[0]
		# else:
			# print('fatal error 6305')
			# sys.exit()		
		
		hp = 1.e6
		lp = -10.
		count = 0
		while hp >= 500000. or lp <= 0. or hp <= lp or hp <= 0. or lp >= 500000.:
			random_value = np.random.uniform(-weight, weight, 1)
			center = offspring_crossover[idx, 0] * (1. + random_value)
			
			random_value = np.random.uniform(-weight, weight, 1)
			width = offspring_crossover[idx, 1] * (1. + random_value)
			
			hp = center + width/2.
			lp = center - width/2.
			count += 1
			print(count)
			print(hp)
			print(lp)
			if count >= 100000:
				print('max iter----')
				offspring_crossover[idx, 0] = 250000.
				offspring_crossover[idx, 1] = 250000.
		
		offspring_crossover[idx, 0] = center
		offspring_crossover[idx, 1] = width
		
	return offspring_crossover

# plt.show()
def generate_color_line(x, y, color):				
	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	# lc = LineCollection(segments, cmap=plt.get_cmap('spring'), norm=plt.Normalize(0, 10))
	lc = LineCollection(segments, cmap=plt.get_cmap(color))
	lc.set_array(np.arange(len(x)))
	return lc
		
def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'filter':
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
	config['fs'] = float(config['fs'])
	config['linewidth'] = float(config['linewidth'])
	config['step_cbar'] = float(config['step_cbar'])
	
	
	
	
	config['nlevel'] = int(config['nlevel'])
	config['num_generations'] = int(config['num_generations'])
	config['num_parents_mating'] = int(config['num_parents_mating'])
	config['weight_mutation'] = float(config['weight_mutation'])
	config['freq_value'] = float(config['freq_value'])
	config['width_freq_range'] = float(config['width_freq_range'])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	# print(config['filter'][0])
	# if config['filter'][0] != 'OFF':
		# if config['filter'][0] == 'bandpass':
			# config['filter'] = [config['filter'][0], [float(config['filter'][1]), float(config['filter'][2])], float(config['filter'][3])]
		# elif config['filter'][0] == 'highpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		# elif config['filter'][0] == 'lowpass':
			# config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		# else:
			# print('error filter 87965')
			# sys.exit()
	
	# if config['filter2'][0] != 'OFF':
		# if config['filter2'][0] == 'bandpass':
			# config['filter2'] = [config['filter2'][0], [float(config['filter2'][1]), float(config['filter2'][2])], float(config['filter2'][3])]
		# elif config['filter2'][0] == 'highpass':
			# config['filter2'] = [config['filter2'][0], float(config['filter2'][1]), float(config['filter2'][2])]
		# elif config['filter2'][0] == 'lowpass':
			# config['filter2'] = [config['filter2'][0], float(config['filter2'][1]), float(config['filter2'][2])]
		# else:
			# print('error filter2 87465')
			# sys.exit()
	
	
	return config


if __name__ == '__main__':
	main(sys.argv)
