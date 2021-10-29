# import os
from os import listdir
from os.path import join, isdir, basename
import sys
# from sys import exit
# from sys.path import path.insert
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from m_open_extension import *
from m_det_features import signal_rms

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

Inputs = ['mode']
InputsOpt_Defaults = {'n_batches':1, 'name':'default', 'file':None}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'mode1':
		# RMS_long = [[] for i in range(config['n_batches'])]
		
		# root = Tk()
		# root.withdraw()
		# root.update()
		# Filepaths = filedialog.askopenfilenames()			
		# root.destroy()
		# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		# RMS_long = [read_pickle(filepath) for filepath in Filepaths]
		
		
		# rows = table.axes[0].tolist()		
		# max_V = table['MAX'].values
		# rms_V = table['RMS'].values
		
		Filepaths = []
		for i in range(7):
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			Filepaths.append(filepath)
		
		
		RMS_long = []
		for filepath in Filepaths:
			table = pd.read_excel(filepath)	
			RMS_long.append(table['MAX'].values)
		
		# save_pickle('rms_batch.pkl', RMS)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.boxplot(RMS_long)
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
		
		ax.set_title('AE_4', fontsize=12)
		
		ax.set_xticklabels(['M=25%\nn=100%', 'M=50%\nn=100%', 'M=75%\nn=100%', 'M=100%\nn=25%', 'M=100%\nn=50%', 'M=100%\nn=75%', 'M=100%\nn=100%'])
		# for label in ax.get_xmajorticklabels():
			# label.set_rotation(45)
			# label.set_horizontalalignment("right")

		# ax.set_xticklabels(['14:06', '14:16', '14:30', '15:00', '15:30', '16:00', '16:20', '17:00', '17:30', '18:00', '18:30', '19:00', '23:30', '00:03'])
		# ax.set_xlabel('Time on 20171020')
		ax.set_ylabel('Max. Amplitude (V)')
		# ax.set_ylim(bottom=1.e-4, top=4.e-4)
		
		# plt.boxplot(RMS_long)
		plt.show()
	
	elif config['mode'] == 'mode2':
		# RMS_long = [[] for i in range(config['n_batches'])]
		print('Select table with features: ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		# feature = read_pickle(filepath)
		
		# amp_factor = input('Input amplification factor dB: ')
		# amp_factor = float(amp_factor)
		fig, ax = plt.subplots(nrows=1, ncols=1)
		for filepath in Filepaths:
			table = pd.read_excel(filepath)	
			rows = table.axes[0].tolist()		
			# max_V = table['MAX'].values
			feature = table['MAX'].values
			
			# print(rms_V)
			# print(max_V)
			# sys.exit()
			
			times = [rows[i][25:31] for i in range(len(rows))]
			times = [time[0:2] + ":" + time[2:4] + ":" + time[4:6] for time in times]
			
			filename = basename(filepath)
			index = filename.find('.')
			label = filename[index-4:index]			
			ax.plot(feature, label=label)
			# ax.plot(feature, label=label)
			# ax.plot(feature, label=label)
		
		
		
		ax.legend()
		divisions = 20
		ax.set_xticks( [i*divisions for i in range(int(len(times)/divisions))] + [len(times)-1])
		# ax.set_xticklabels(times)
		ax.set_xticklabels( [times[i*divisions] for i in range(int(len(times)/divisions))] + [times[len(times)-1]])
		ax.set_ylabel('Max. Amplitude (V)')
		
		for label in ax.get_xmajorticklabels():
			label.set_rotation(45)
			label.set_horizontalalignment("right")
		
		
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()		
		root.destroy()
		table = pd.read_excel(filepath)	
		rows = table.axes[0].tolist()		
		# max_V = table['MAX'].values
		rpm = table['n'].values
		
		ax2 = ax.twinx()
		ax2.plot(rpm, 'om')
		# ax2.set_ylabel('RPM', color='r')
		# ax2.tick_params('y', colors='r')
		ax2.set_ylabel('RPM', color='m')
		ax2.tick_params('y', colors='m')
		
		plt.show()

	elif config['mode'] == 'mode3':
	
		print('Select Files Beton 150425 Class 0: ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_0 = filedialog.askopenfilenames()			
		root.destroy()
		
		print('Select Files Kohle 152916 Class 1: ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths_1 = filedialog.askopenfilenames()			
		root.destroy()
		
		print(Filepaths_0)
		print(Filepaths_1)

		# fig, ax = plt.subplots(nrows=1, ncols=1)
		
		feat_rms_0 = []
		feat_max_0 = []
		feat_p2p_0 = []
		feat_freq_0 = []
		feat_rise_0 = []
		feat_dur_0 = []
		feat_count_0 = []
		
		feat_rms_1 = []
		feat_max_1 = []
		feat_p2p_1 = []
		feat_freq_1 = []
		feat_rise_1 = []
		feat_dur_1 = []
		feat_count_1 = []
		
		for filepath_0, filepath_1 in zip(Filepaths_0, Filepaths_1):
		
			features_0 = read_pickle(filepath_0)
			features_1 = read_pickle(filepath_1)
			
			# feat_rms_0.append(features_0['feat_rms'])
			# feat_max_0.append(features_0['feat_max'])
			# feat_p2p_0.append(features_0['feat_p2p'])
			# feat_freq_0.append(features_0['feat_freq'])
			# feat_count_0.append(features_0['feat_count'])
			# feat_rise_0.append(features_0['feat_rise'])
			# feat_dur_0.append(features_0['feat_dur'])
			
			# feat_rms_1.append(features_1['feat_rms'])
			# feat_max_1.append(features_1['feat_max'])
			# feat_p2p_1.append(features_1['feat_p2p'])
			# feat_freq_1.append(features_1['feat_freq'])
			# feat_count_1.append(features_1['feat_count'])
			# feat_rise_1.append(features_1['feat_rise'])
			# feat_dur_1.append(features_1['feat_dur'])
			
			feat_rms_0 = feat_rms_0 + features_0['feat_rms']
			feat_max_0 = feat_max_0 + features_0['feat_max']
			feat_p2p_0 = feat_p2p_0 + features_0['feat_p2p']
			feat_freq_0 = feat_freq_0 + features_0['feat_freq']
			feat_count_0 = feat_count_0 + features_0['feat_count']
			feat_dur_0 = feat_dur_0 + features_0['feat_dur']
			feat_rise_0 = feat_rise_0 + features_0['feat_rise']
			
			
			feat_rms_1 = feat_rms_1 + features_1['feat_rms']
			feat_max_1 = feat_max_1 + features_1['feat_max']
			feat_p2p_1 = feat_p2p_1 + features_1['feat_p2p']
			feat_freq_1 = feat_freq_1 + features_1['feat_freq']
			feat_count_1 = feat_count_1 + features_1['feat_count']
			feat_dur_1 = feat_dur_1 + features_1['feat_dur']
			feat_rise_1 = feat_rise_1 + features_1['feat_rise']


		fig, ax = plt.subplots(nrows=2, ncols=4)
		from pylab import text
		caja = ax[0][0].boxplot([feat_rms_0, feat_rms_1])
		ax[0][0].set_title('RMS')
		ax[0][0].set_xticklabels(['Beton', 'Kohle'])		
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][0].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered

		caja = ax[0][1].boxplot([feat_max_0, feat_max_1])
		ax[0][1].set_title('MAX')
		ax[0][1].set_xticklabels(['Beton', 'Kohle'])
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][1].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[0][2].boxplot([feat_p2p_0, feat_p2p_1])
		ax[0][2].set_title('P2P')
		ax[0][2].set_xticklabels(['Beton', 'Kohle'])
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][2].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		feat_freq_0 = np.array(feat_freq_0)/1000.
		feat_freq_0 = feat_freq_0.tolist()
		
		feat_freq_1 = np.array(feat_freq_1)/1000.
		feat_freq_1 = feat_freq_1.tolist()
		
		caja = ax[0][3].boxplot([feat_freq_0, feat_freq_1])
		ax[0][3].set_title('FREQ')
		ax[0][3].set_xticklabels(['Beton', 'Kohle'])
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][3].text(x, y, '%.1f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][0].boxplot([feat_dur_0, feat_dur_1])
		ax[1][0].set_title('DUR')
		ax[1][0].set_xticklabels(['Beton', 'Kohle'])
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][0].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][1].boxplot([feat_count_0, feat_count_1])
		ax[1][1].set_title('COUNT')
		ax[1][1].set_xticklabels(['Beton', 'Kohle'])
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][1].text(x, y, '%.1f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][2].boxplot([feat_rise_0, feat_rise_1])
		ax[1][2].set_title('RISE')
		ax[1][2].set_xticklabels(['Beton', 'Kohle'])
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][2].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		features = []
		classification = []
		
		n_examples_0 = len(feat_max_0)		
		for u in range(n_examples_0):
			features.append([feat_max_0[u], feat_p2p_0[u], feat_rms_0[u], feat_freq_0[u], feat_dur_0[u], feat_count_0[u], feat_rise_0[u]])
			classification.append(0)
		
		n_examples_1 = len(feat_max_1)		
		for w in range(n_examples_1):
			features.append([feat_max_1[w], feat_p2p_1[w], feat_rms_1[w], feat_freq_1[w], feat_dur_1[w], feat_count_1[w], feat_rise_1[w]])
			classification.append(1)
		
		mydict = {}
		mydict['features'] = features
		mydict['classification'] = classification
		mydict['n_examples_0'] = n_examples_0
		mydict['n_examples_1'] = n_examples_1
		
		
		save_pickle(config['name'] + '.pkl', mydict)
		plt.show()
		
		sys.exit()###########################################################
		
		
		ax.legend()
		divisions = 20
		ax.set_xticks( [i*divisions for i in range(int(len(times)/divisions))] + [len(times)-1])
		# ax.set_xticklabels(times)
		ax.set_xticklabels( [times[i*divisions] for i in range(int(len(times)/divisions))] + [times[len(times)-1]])
		ax.set_ylabel('Max. Amplitude (V)')
		
		for label in ax.get_xmajorticklabels():
			label.set_rotation(45)
			label.set_horizontalalignment("right")
		
		
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()		
		root.destroy()
		table = pd.read_excel(filepath)	
		rows = table.axes[0].tolist()		
		# max_V = table['MAX'].values
		rpm = table['n'].values
		
		ax2 = ax.twinx()
		ax2.plot(rpm, 'om')
		# ax2.set_ylabel('RPM', color='r')
		# ax2.tick_params('y', colors='r')
		ax2.set_ylabel('RPM', color='m')
		ax2.tick_params('y', colors='m')
		
		plt.show()
	

	elif config['mode'] == 'mean_mag_fft':
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		meanFFT = np.zeros(len(Data[0])/2)
		
		for k in range(len(Data)):
			magX, f, df = mag_fft(Data[k], config['fs'])
			meanFFT = meanFFT + magX
		meanFFT = meanFFT / len(Data)
		
		save_pickle('mean_5_fft.pkl', meanFFT)
		
		plt.plot(meanFFT)
		plt.show()
	
	elif config['mode'] == 'oneclass_plot':
		
		if config['file'] == None:
			print('Select Files features from signal: ')
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths_0 = filedialog.askopenfilenames()			
			root.destroy()
		else:
			Filepaths_0 = [config['file']]
		

		
		print(Filepaths_0)

		# fig, ax = plt.subplots(nrows=1, ncols=1)
		
		feat_rms_0 = []
		feat_max_0 = []
		feat_p2p_0 = []
		feat_freq_0 = []
		feat_rise_0 = []
		feat_dur_0 = []
		feat_count_0 = []
		feat_crest_0 = []

		
		for filepath_0 in Filepaths_0:
		
			features_0 = read_pickle(filepath_0)
			
			# feat_rms_0.append(features_0['feat_rms'])
			# feat_max_0.append(features_0['feat_max'])
			# feat_p2p_0.append(features_0['feat_p2p'])
			# feat_freq_0.append(features_0['feat_freq'])
			# feat_count_0.append(features_0['feat_count'])
			# feat_rise_0.append(features_0['feat_rise'])
			# feat_dur_0.append(features_0['feat_dur'])
			
			# feat_rms_1.append(features_1['feat_rms'])
			# feat_max_1.append(features_1['feat_max'])
			# feat_p2p_1.append(features_1['feat_p2p'])
			# feat_freq_1.append(features_1['feat_freq'])
			# feat_count_1.append(features_1['feat_count'])
			# feat_rise_1.append(features_1['feat_rise'])
			# feat_dur_1.append(features_1['feat_dur'])
			
			feat_rms_0 = feat_rms_0 + features_0['feat_rms']
			feat_max_0 = feat_max_0 + features_0['feat_max']
			feat_p2p_0 = feat_p2p_0 + features_0['feat_p2p']
			feat_freq_0 = feat_freq_0 + features_0['feat_freq']
			feat_count_0 = feat_count_0 + features_0['feat_count']
			feat_dur_0 = feat_dur_0 + features_0['feat_dur']
			feat_rise_0 = feat_rise_0 + features_0['feat_rise']
			feat_crest_0 = feat_crest_0 + features_0['feat_crest']
			
			
		n_burst = len(feat_rms_0)
		name_x = [str(n_burst) + ' Bursts']
		fig, ax = plt.subplots(nrows=2, ncols=4)
		from pylab import text
		caja = ax[0][0].boxplot([feat_rms_0])
		ax[0][0].set_title('RMS')
		ax[0][0].set_xticklabels(name_x)		
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][0].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered

		caja = ax[0][1].boxplot([feat_max_0])
		ax[0][1].set_title('MAX')
		ax[0][1].set_xticklabels(name_x)
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][1].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[0][2].boxplot([feat_p2p_0])
		ax[0][2].set_title('P2P')
		ax[0][2].set_xticklabels(name_x)
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][2].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		feat_freq_0 = np.array(feat_freq_0)/1000.
		feat_freq_0 = feat_freq_0.tolist()
		

		
		caja = ax[0][3].boxplot([feat_freq_0])
		ax[0][3].set_title('FREQ')
		ax[0][3].set_xticklabels(name_x)
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[0][3].text(x, y, '%.1f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][0].boxplot([feat_dur_0])
		ax[1][0].set_title('DUR')
		ax[1][0].set_xticklabels(name_x)
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][0].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][1].boxplot([feat_count_0])
		ax[1][1].set_title('COUNT')
		ax[1][1].set_xticklabels(name_x)
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][1].text(x, y, '%.1f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][2].boxplot([feat_rise_0])
		ax[1][2].set_title('RISE')
		ax[1][2].set_xticklabels(name_x)
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][2].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		caja = ax[1][3].boxplot([feat_crest_0])
		ax[1][3].set_title('CREST')
		ax[1][3].set_xticklabels(name_x)
		
		
		for line in caja['medians']:
			# get position data for median line
			x, y = line.get_xydata()[1] # top of median line
			# overlay median value
			ax[1][3].text(x, y, '%.3f' % y, horizontalalignment='left') # draw above, centered
		
		
		
		
		features = []
		features_select = []
		classification = []
		
		n_examples_0 = len(feat_max_0)		
		for u in range(n_examples_0):
			features.append([feat_max_0[u], feat_p2p_0[u], feat_rms_0[u], feat_freq_0[u], feat_dur_0[u], feat_count_0[u], feat_rise_0[u], feat_crest_0[u]])
			features_select.append([feat_rms_0[u], feat_count_0[u]])
			classification.append(0)
		
		
		mydict = {}
		mydict['features'] = features
		mydict['features_select'] = features_select
		mydict['classification'] = classification
		mydict['n_examples_0'] = n_examples_0
		
		if config['name'] == 'auto':
			label = basename(Filepaths_0[0])
			label = label[0:len(label)-4]
			config['name'] = label
		save_pickle(config['name'] + '.pkl', mydict)
		print('!!!!')
		
		fig.set_size_inches(14, 10)
		fig.savefig(config['name'] + '.png')
		# plt.show()
		
		
		

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
	
	# config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	config['n_batches'] = int(config['n_batches'])
	# Variable conversion
	return config

def box_8_plot(caca, names, features):
	print(caca)
	fig, ax = plt.subplots(nrows=2, ncols=4)
	from pylab import text
	caja = ax[0][0].boxplot(features[0])
	ax[0][0].set_title(names[0])
	ax[0][0].set_xticklabels(caca)		
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered

	caja = ax[0][1].boxplot(features[1])
	ax[0][1].set_title(names[1])
	ax[0][1].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[0][2].boxplot(features[2])
	ax[0][2].set_title(names[2])
	ax[0][2].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][2].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	

	
	caja = ax[0][3].boxplot(features[3])
	ax[0][3].set_title(names[3])
	ax[0][3].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][0].boxplot(features[4])
	ax[1][0].set_title(names[4])
	ax[1][0].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][1].boxplot(features[5])
	ax[1][1].set_title(names[5])
	ax[1][1].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][2].boxplot(features[6])
	ax[1][2].set_title(names[6])
	ax[1][2].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][2].text(x, y, ' %.1f' % y, horizontalalignment='left') # draw above, centered
	from matplotlib.ticker import FormatStrFormatter
	yFormatter = FormatStrFormatter('%.1f')
	ax[1][2].yaxis.set_major_formatter(yFormatter)
	
	caja = ax[1][3].boxplot(features[7])
	ax[1][3].set_title(names[7])
	ax[1][3].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	


	plt.show()
	return
	
def box_8_plot_vs(labels, names, features_normal, features_anormal):

	fig, ax = plt.subplots(nrows=2, ncols=4)
	from pylab import text
	caja = ax[0][0].boxplot([features_normal[0], features_anormal[0]])
	ax[0][0].set_title(names[0])
	ax[0][0].set_xticklabels(labels)		
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][0].text(x, y, ' %.1f' % y, horizontalalignment='left') # draw above, centered

	caja = ax[0][1].boxplot([features_normal[1], features_anormal[1]])
	ax[0][1].set_title(names[1])
	ax[0][1].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][1].text(x, y, ' %.1f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[0][2].boxplot([features_normal[2], features_anormal[2]])
	ax[0][2].set_title(names[2])
	ax[0][2].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][2].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	

	
	caja = ax[0][3].boxplot([features_normal[3], features_anormal[3]])
	ax[0][3].set_title(names[3])
	ax[0][3].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][3].text(x, y, ' %.0f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][0].boxplot([features_normal[4], features_anormal[4]])
	ax[1][0].set_title(names[4])
	ax[1][0].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][1].boxplot([features_normal[5], features_anormal[5]])
	ax[1][1].set_title(names[5])
	ax[1][1].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][2].boxplot([features_normal[6], features_anormal[6]])
	ax[1][2].set_title(names[6])
	ax[1][2].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][2].text(x, y, ' %.0f' % y, horizontalalignment='left') # draw above, centered
	from matplotlib.ticker import FormatStrFormatter
	yFormatter = FormatStrFormatter('%.1f')
	ax[1][2].yaxis.set_major_formatter(yFormatter)
	
	caja = ax[1][3].boxplot([features_normal[7], features_anormal[7]])
	ax[1][3].set_title(names[7])
	ax[1][3].set_xticklabels(labels)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	fig.set_size_inches(14, 6)
	plt.tight_layout()
	plt.show()
	return fig

	
def box_8_plot(caca, names, features):
	print(caca)
	fig, ax = plt.subplots(nrows=2, ncols=4)
	from pylab import text
	caja = ax[0][0].boxplot(features[0])
	ax[0][0].set_title(names[0])
	ax[0][0].set_xticklabels(caca)		
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered

	caja = ax[0][1].boxplot(features[1])
	ax[0][1].set_title(names[1])
	ax[0][1].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[0][2].boxplot(features[2])
	ax[0][2].set_title(names[2])
	ax[0][2].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][2].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	

	
	caja = ax[0][3].boxplot(features[3])
	ax[0][3].set_title(names[3])
	ax[0][3].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][0].boxplot(features[4])
	ax[1][0].set_title(names[4])
	ax[1][0].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][1].boxplot(features[5])
	ax[1][1].set_title(names[5])
	ax[1][1].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][2].boxplot(features[6])
	ax[1][2].set_title(names[6])
	ax[1][2].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][2].text(x, y, ' %.1f' % y, horizontalalignment='left') # draw above, centered
	from matplotlib.ticker import FormatStrFormatter
	yFormatter = FormatStrFormatter('%.1f')
	ax[1][2].yaxis.set_major_formatter(yFormatter)
	
	caja = ax[1][3].boxplot(features[7])
	ax[1][3].set_title(names[7])
	ax[1][3].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	


	plt.show()
	return
	
def box_10_plot(caca, names, features):
	print(caca)
	fig, ax = plt.subplots(nrows=2, ncols=5)
	from pylab import text
	caja = ax[0][0].boxplot(features[0])
	ax[0][0].set_title(names[0])
	ax[0][0].set_xticklabels(caca)		
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered

	caja = ax[0][1].boxplot(features[1])
	ax[0][1].set_title(names[1])
	ax[0][1].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[0][2].boxplot(features[2])
	ax[0][2].set_title(names[2])
	ax[0][2].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][2].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	

	
	caja = ax[0][3].boxplot(features[3])
	ax[0][3].set_title(names[3])
	ax[0][3].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[0][4].boxplot(features[4])
	ax[0][4].set_title(names[4])
	ax[0][4].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[0][4].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][0].boxplot(features[5])
	ax[1][0].set_title(names[5])
	ax[1][0].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][0].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][1].boxplot(features[6])
	ax[1][1].set_title(names[6])
	ax[1][1].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][1].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][2].boxplot(features[7])
	ax[1][2].set_title(names[7])
	ax[1][2].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][2].text(x, y, ' %.1f' % y, horizontalalignment='left') # draw above, centered
	from matplotlib.ticker import FormatStrFormatter
	yFormatter = FormatStrFormatter('%.1f')
	ax[1][2].yaxis.set_major_formatter(yFormatter)
	
	caja = ax[1][3].boxplot(features[8])
	ax[1][3].set_title(names[8])
	ax[1][3].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][3].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	
	caja = ax[1][4].boxplot(features[9])
	ax[1][4].set_title(names[9])
	ax[1][4].set_xticklabels(caca)
	
	for line in caja['medians']:
		# get position data for median line
		x, y = line.get_xydata()[1] # top of median line
		# overlay median value
		ax[1][4].text(x, y, ' %.2f' % y, horizontalalignment='left') # draw above, centered
	


	plt.show()
	return
	
if __name__ == '__main__':
	main(sys.argv)
