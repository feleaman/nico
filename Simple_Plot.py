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
plt.rcParams['agg.path.chunksize'] = 20000 #for plotting optimization purposes
plt.rcParams['savefig.dpi'] = 1000
plt.rcParams['savefig.format'] = 'jpeg'


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser









#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {'power2':'OFF'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	if config['mode'] == 'bar_plot':

		
		# recall_means = (29.2, 45.5, 55.3, 74.7)
		# recall_std = (8.16, 15.73, 8.88, 9.41)
		# precision_means = (71.4, 81.6, 81.6, 72.36)
		# precision_std = (10.30, 11.49, 5.34, 6.58)
		# fscore_means = (41.1, 55.3, 65.1, 72.7)
		# fscore_std = (10.08, 14.13, 5.68, 3.53)
		# plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)
		
		recall_means = (0.754, 0.373, 0.792)
		recall_std = (0.087, 0.092, 0.077)
		
		fpr_means = (0.149, 0.026, 0.138)
		fpr_std = (0.074, 0.008, 0.079)
		
		mcc_means = (0.572, 0.469, 0.620)
		mcc_std = (0.164, 0.098, 0.165)
		
		
		# recall_means = (0.754, 0.373, 0.715)
		# recall_std = (0.087, 0.092, 0.089)
		
		# fpr_means = (0.149, 0.026, 0.072)
		# fpr_std = (0.074, 0.008, 0.046)
		
		# mcc_means = (0.572, 0.469, 0.656)
		# mcc_std = (0.164, 0.098, 0.140)
		
		
		
		
		plot_scores_error_4(recall_means, recall_std, fpr_means, fpr_std, mcc_means, mcc_std)
	
	elif config['mode'] == 'bar_plot_ae':
	
		# ohne_means = (287.3, 370.1, 300.4, 352.7, 355.4)		
		# mit_means = (108.9, 138.4, 133.3, 137.1, 140.4)
		
		ohne_means = (2.90, 3.06, 3.17, 3.15, 3.1)		
		mit_means = (2.29, 2.39, 2.46, 2.43, 2.41)
	
		
		N = 5
	

		ind = np.arange(N)  # the x locations for the groups
		width = 0.35       # the width of the bars

		fig1, ax = plt.subplots()
		rects1 = ax.bar(ind, ohne_means, width, color='b')


		rects2 = ax.bar(ind + width, mit_means, width, color='g')

		# add some text for labels, title and axes ticks
		ax.set_ylabel('RMS Wert [mV]', fontsize=12)
		# ax.set_title('Test Results', fontsize=12)
		ax.tick_params(axis='both', labelsize=11)
		ax.set_xticks(ind + width / 2)
		# ax.set_ylim((0, 1))
		ax.set_xticklabels(('Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'), fontsize=12)
		# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

		ax.legend((rects1[0], rects2[0]), ('Ohne Lack', 'Mit Lack'), loc='lower right')
		fig1.set_size_inches(5.3, 4.3)
		
		# def autolabel(rects):
			# """
			# Attach a text label above each bar displaying its height
			# """
			# for rect in rects:
				# height = rect.get_height()
				# ax.text(rect.get_x() + rect.get_width()/2. + 0.165, 1.000*height,
						# '%.3f' % height,
						# ha='center', va='bottom')

		# autolabel(rects1)
		# autolabel(rects2)
		plt.tight_layout()

		plt.show()

		
	elif config['mode'] == 'bar_plot_emd':
	

		
		a_means = (29.33, 23.63, 22.08, 18.19, 25.78)		
		g_means = (28.21, 23.78, 24.27, 22.76, 27.85)
		m_means = (27.23, 26.76, 25.45, 26.44, 25.32)
		
		N = 5
	

		ind = np.arange(N)  # the x locations for the groups
		width = 0.2      # the width of the bars

		fig1, ax = plt.subplots()
		rects1 = ax.bar(ind, a_means, width, color='b')
		rects2 = ax.bar(ind + width, g_means, width, color='g')
		rects3 = ax.bar(ind + 2*width, m_means, width, color='r')

		# add some text for labels, title and axes ticks
		ax.set_ylabel('MPR [dB]', fontsize=13)
		# ax.set_title('Test Results', fontsize=12)
		ax.tick_params(axis='both', labelsize=12)
		ax.set_xticks(ind + width / 2)
		# ax.set_ylim((0, 1))
		ax.set_xticklabels(('Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5'), fontsize=13)
		# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

		ax.legend((rects1[0], rects2[0], rects3[0]), ('No fault', 'Initial fault', 'Developed fault'), loc='lower right')
		# fig1.set_size_inches(6.3, 3.3)
		fig1.set_size_inches(6.5, 4.5)
		# fig1.set_size_inches(9.5, 4.5)
		
		# def autolabel(rects):
			# """
			# Attach a text label above each bar displaying its height
			# """
			# for rect in rects:
				# height = rect.get_height()
				# ax.text(rect.get_x() + rect.get_width()/2. + 0.165, 1.000*height,
						# '%.3f' % height,
						# ha='center', va='bottom')

		# autolabel(rects1)
		# autolabel(rects2)
		plt.tight_layout()

		plt.show()
	
	
	
	elif config['mode'] == 'op_cond_plot':


		var = [400, 700, 1000, 1300]
		mcc_thr = [2.20, 2.46, 3.06, 4.20]
		mcc_win = [2.24, 2.59, 3.50, 4.88]
		mcc_edg = [2.25, 2.65, 3.71, 5.28]
		
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
		plot_3_lines_together(ax, var, mcc_thr, mcc_win, mcc_edg, 'RMS Value [mV]')
		
		plt.show()
		
		
		var = [30, 55, 80]
		# mcc_thr = [   ]
		# mcc_win = [   ]
		# mcc_edg = [   ]
		
		mcc_thr = [2.20, 2.24, 2.25,]
		mcc_win = [2.46, 2.59, 2.65,]
		mcc_edg = [3.06, 3.50, 3.71,]		
		mcc_add = [4.20, 4.88, 5.28]
		
		
		
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
		plot_4_lines_together(ax, var, mcc_thr, mcc_win, mcc_edg, mcc_add, 'RMS Value [mV]')
		
		plt.show()
	
	elif config['mode'] == 'freq_screw':

		cycle1 = [2574, 6930, 11158, 17486, 24682]
		cycle2 = [25522, 26782, 28042, 29330, 30170]
		cycle = cycle1 + cycle2
		
		#AE1
		band1 = [1808E+09, 2.882E+07, 4.512E+08, 7.011E+07, 2.476E+07, 1.889E+08, 8.469E+07, 7.458E+08, 1.267E+09, 2.227E+08]
		band2 = [4.972E+08, 4.264E+06, 7.564E+07, 2.023E+07, 1.159E+07, 4.605E+07, 1.209E+07, 1.129E+08, 2.551E+08, 4.285E+07]
		band3 = [0.000E+00, 0.000E+00, 0.000E+00, 2.928E+05, 1.531E+05, 1.860E+07, 7.189E+06, 6.699E+07, 1.446E+08, 2.277E+07]
		band4 = [0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 5.616E+04, 1.470E+05, 7.417E+06, 1.076E+07, 2.082E+06]
		
		# ##AE2
		# band1 = [1.055E+09, 1.695E+09, 6.336E+08, 1.028E+09, 6.456E+07, 5.290E+09, 4.640E+09, 1.176E+10, 6.957E+09, 2.502E+10]
		# band2 = [1.496E+08, 1.623E+08, 2.896E+07, 9.207E+07, 6.355E+06, 6.378E+08, 5.558E+08, 1.194E+09, 8.612E+08, 2.660E+09]
		# band3 = [0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 3.870E+07, 3.411E+07, 1.475E+08, 1.689E+08, 5.690E+08]
		# band4 = [0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00]
		

		
		# fig, ax = plt.subplots()			
		# ax.plot(cycle, np.log1p(band1), '-o', label='100-140 kHz')
		# ax.plot(cycle, np.log1p(band2), '-v', label='140-180 kHz')
		# ax.plot(cycle, np.log1p(band3), '-s', label='180-220 kHz')
		# ax.plot(cycle, np.log1p(band4), '-^', label='220-260 kHz')
		# ax.set_xlabel('N° Cycle', fontsize=14)
		# ax.set_ylabel('log(1+Energy)', fontsize=14)
		# ax.tick_params(axis='both', labelsize=12.5)
		# ax.legend(fontsize=10.5)	
		# plt.show()

		
		
		fig, ax = plt.subplots()
		
		band1 = np.log1p(band1)
		band2 = np.log1p(band2)
		band3 = np.log1p(band3)
		band4 = np.log1p(band4)		
		
		ax.bar(cycle, band1, width=750, label='100-140 kHz', hatch='...', color='mistyrose', linewidth=0.5, edgecolor='k')
		ax.bar(cycle, band2, width=750, bottom=band1, label='140-180 kHz', hatch='---', color='lightcyan', linewidth=0.5, edgecolor='k')
		ax.bar(cycle, band3, width=750, bottom=list(np.array(band1)+np.array(band2)), label='180-220 kHz', hatch='///', color='lavender', linewidth=0.5, edgecolor='k')
		ax.bar(cycle, band4, width=750, bottom=list(np.array(band1)+np.array(band2)+np.array(band3)), label='220-260 kHz', hatch='|||', color='lightyellow', linewidth=0.5, edgecolor='k')
		
		ax.axvline(25090, color='k', linewidth=1.5, linestyle='--')	
		fig.text(0.68, 0.85, 'LC N°1', ha='center', va='center', fontsize=11.5)
		fig.text(0.780, 0.85, 'LC N°2', ha='center', va='center', fontsize=11.5)			
		
		ax.set_xlabel('N° Cycle', fontsize=14)
		ax.set_ylabel('log(1+Energy)', fontsize=14)
		ax.tick_params(axis='both', labelsize=12.5)
		ax.legend(fontsize=10.5)
		ax.set_title('Channel AE-1, Second IMF', fontsize=14)
		
		plt.show()
		
	
	elif config['mode'] == 'aufgaben':

		tarea = [1, 2, 3, 4, 5, 6]
		
		tiempo = [2, 15, 18, 40, 17, 8]
		
		# ##AE2
		# band1 = [1.055E+09, 1.695E+09, 6.336E+08, 1.028E+09, 6.456E+07, 5.290E+09, 4.640E+09, 1.176E+10, 6.957E+09, 2.502E+10]
		# band2 = [1.496E+08, 1.623E+08, 2.896E+07, 9.207E+07, 6.355E+06, 6.378E+08, 5.558E+08, 1.194E+09, 8.612E+08, 2.660E+09]
		# band3 = [0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 3.870E+07, 3.411E+07, 1.475E+08, 1.689E+08, 5.690E+08]
		# band4 = [0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00]
		

		
		fig, ax = plt.subplots()			
		graf = ax.bar(tarea, tiempo)
		ax.set_xlabel('Aufgaben', fontsize=17)
		ax.set_xticks([1, 2, 3, 4, 5, 6])
		ax.set_xticklabels(['Anträge', 'Projekten', 'Veröffentlichungen', 'Dissertation', 'Lehre', 'AMT Aufgaben'])
		ax.set_ylabel('% Aufwand', fontsize=17)
		ax.tick_params(axis='both', labelsize=15)
		ax.legend(fontsize=10.5)	
		
		
		def autolabel(rects):
			for rect in rects:
				height = rect.get_height()
				ax.text(rect.get_x() + rect.get_width()/2. + 0.0, 1.000*height, '%d' % round(height), ha='center', va='bottom', fontsize=15)

		autolabel(graf)
		
		
		plt.show()
		tiempo = [65, 25, 10]
		fig, ax = plt.subplots(ncols=2)
		# ax[0].set_title('Projekten', labelsize=17)
		# patches, texts, autotexts = ax[0].pie(tiempo, labels=['WEACM', 'BCMS', 'TAEC'])
		ax[0].set_title('Projekten (15%)', fontsize=18)
		ax[1].set_title('Lehre (17%)', fontsize=18)
		
		ax[0].pie(tiempo, labels=['WEACM (10%)', 'BCMS (4%)', 'TAEC (1%)'], textprops={'fontsize': 14})
		# texts[0].set_fontsize(15)
		# ax.set_xlabel('Aufgaben', fontsize=17)
		# ax.set_xticks([1, 2, 3, 4, 5, 6])
		# ax.set_xticklabels(['Anträge', 'Projekten', 'Veröffentlichungen', 'Dissertation', 'Lehre', 'AMT Aufgaben'])
		# ax.set_ylabel('% Aufwand', fontsize=17)
		ax[0].tick_params(labelsize=25)
		# ax[0].legend(fontsize=10.5)	
		
		
		tiempo = [40, 35, 25]
		# ax[1].set_title('Lehre', labelsize=17)
		# patches, texts, autotexts = ax[1].pie(tiempo, labels=['Wiss. Arbeiten', 'GTE', 'V+K'])
		ax[1].pie(tiempo, labels=['Wiss. Arbeiten (7%)', 'GTE (6%)', 'V+K (4%)'], textprops={'fontsize': 14})
		# texts[0].set_fontsize(15)
		# ax.set_xlabel('Aufgaben', fontsize=17)
		# ax.set_xticks([1, 2, 3, 4, 5, 6])
		# ax.set_xticklabels(['Anträge', 'Projekten', 'Veröffentlichungen', 'Dissertation', 'Lehre', 'AMT Aufgaben'])
		# ax.set_ylabel('% Aufwand', fontsize=17)
		# ax[1].tick_params(axis='both', labelsize=15)
		# ax[1].legend(fontsize=10.5)	
		
		plt.show()
		
		
		
	
	elif config['mode'] == 'line_plot_std':
		thr = [0.2, 0.25, 0.3, 0.35, 0.4]
	
		accu1 = [73.1, 77.9, 77.6, 77.3, 74.4]
		std_accu1 = [2.50, 2.25, 2.78, 3.91, 3.42]
		
		
		recall1 = [58.2, 82.6, 81.5, 80.2, 84.6]
		std_recall1 = [5.57, 4.89, 6.22, 7.31, 4.20]
		
		
		
		tnr1 = [79.9, 75.4, 75.4, 75.6, 67.2]
		std_tnr1 = [2.75, 4.32, 4.27, 5.02, 5.90]
		
		
		
		plot_3_lines(thr, accu1, recall1, tnr1, std_accu1, std_recall1, std_tnr1)
		
		
		accu2 = [74.5, 74.1, 71.3, 75.2, 71.9]
		std_accu2 = [2.91, 4.12, 3.75, 5.01, 3.89]
		
		recall2 = [73.6, 76.4, 70.1, 78.4, 86.5]
		std_recall2 = [3.22, 6.44, 5.04, 7.25, 5.62]
		
		tnr2 = [75.0, 72.8, 72.1, 73.1, 60.5]
		std_tnr2 = [4.23, 6.17, 6.91, 7.68, 5.76]
		
		plot_3_lines(thr, accu2, recall2, tnr2, std_accu2, std_recall2, std_tnr2)
		
		accu3 = [72.6, 73.9, 73.8, 72.8, 72.2]
		std_accu3 = [4.33, 2.67, 2.92, 4.38, 3.40]
		
		recall3 = [75.6, 79.8, 79.3, 79.9, 89.5]
		std_recall3 = [4.63, 8.24, 7.82, 9.03, 6.11]
		
		tnr3 = [70.9, 70.0, 70.2, 67.6, 58.3]
		std_tnr3 = [6.93, 4.95, 2.85, 6.27, 7.14]
		
		plot_3_lines(thr, accu3, recall3, tnr3, std_accu3, std_recall3, std_tnr3)
		
	elif config['mode'] == 'training_plot':
		# #bad
		# accu = [67.9, 69.4, 71.2, 71.6, 75.4, 77.8, 77.5, 73.8]
		# recall = [66.8, 59.7, 62.8, 65.3, 75.6, 77.3, 80., 74.4]
		# tnr = [68.5, 74.4, 75.6, 74.9, 75.3, 78.1, 76.2, 73.6]
		
		#w0.5 thr02
		accu = [72.8, 75.4, 79.1, 78.9, 85.0, 85.7, 86.0, 88.2]
		recall = [54.7, 59.1, 63.6, 66.0, 78.2, 76.8, 77.8, 83.3]
		tnr = [79.7, 81.6, 85.0, 83.8, 87.6, 89.1, 89.1, 90.0]
		
		# #w1 thr03
		# accu = [63.4, 61.8, 67.4, 72.1, 77.1, 72.7, 81.5, 81.1]
		# recall = [66.7, 52.0, 55.4, 61.3, 73.0, 72.0, 84.0, 77.8]
		# tnr = [61.8, 66.8, 73.5, 77.6, 79.1, 73.1, 80.3, 82.7]
		
		training = [2, 5, 10, 20, 40, 60, 80, 90]
		
		
		
		
		fig, ax = plt.subplots()
		
		# ax.plot(training, accu, label='Exactitud', marker='o')
		# ax.plot(training, recall, label='Sensibilidad Carbón', marker='^')
		# ax.plot(training, tnr, label='Sensibilidad Concreto', marker='v')
		# ax.set_ylabel('% Puntuación', fontsize=11)
		# ax.set_xlabel('% Datos de Entrenamiento', fontsize=11)
		
		# ax.plot(training, accu, label='Accuracy', marker='o')
		# ax.plot(training, recall, label='Coal Sensitivity', marker='s')
		# ax.plot(training, tnr, label='Concrete Sensitivity', marker='v')		
		# ax.set_ylabel('% Score', fontsize=11)
		# ax.set_xlabel('% Training Data', fontsize=11)
		
		ax.plot(training, accu, label='Exactitud', marker='o')
		ax.plot(training, recall, label='Sensibilidad al Carbón', marker='s')
		ax.plot(training, tnr, label='Sensibilidad al Concreto', marker='v')		
		ax.set_ylabel('% Puntuación', fontsize=12)
		ax.set_xlabel('% Datos de Entrenamiento', fontsize=12)
		
		
		
		
		
		ax.legend()
		ax.tick_params(axis='both', labelsize=11)
		ax.set_xlim(left=0, right=90)
		ax.set_ylim(bottom=50, top=95)
		# ax.set_title('Learning Curve for $T_{amp}=0.2 (V)$ and $\Delta t_{def}=0.5 (ms)$')
		ax.set_title('Curva de Aprendizaje para $T_{amp}=0.2 (V)$ y $\Delta t_{def}=0.5 (ms)$')
		# ax.set_title('Curva de Aprendizaje para $T_{amp}=0.3 (V)$ y $\Delta t_{def}=1 (ms)$')
		# ax.set_title('Learning Curve for $T_{amp}=0.3 (V)$ and $\Delta t_{def}=1 (ms)$')

		# ax.minorticks_on()
		# ax.set_xticks(training)
		plt.show()
		
	elif config['mode'] == 'line_plot_3':
	
		# thr = [0.2, 0.25, 0.3, 0.35, 0.4]	
		# accu1 = [73.1, 77.9, 77.6, 77.3, 74.4]
		# std_accu1 = [2.50, 2.25, 2.78, 3.91, 3.42]		
		# recall1 = [58.2, 82.6, 81.5, 80.2, 84.6]
		# std_recall1 = [5.57, 4.89, 6.22, 7.31, 4.20]		
		# tnr1 = [79.9, 75.4, 75.4, 75.6, 67.2]
		# std_tnr1 = [2.75, 4.32, 4.27, 5.02, 5.90]		
		# accu2 = [74.5, 74.1, 71.3, 75.2, 71.9]
		# std_accu2 = [2.91, 4.12, 3.75, 5.01, 3.89]		
		# recall2 = [73.6, 76.4, 70.1, 78.4, 86.5]
		# std_recall2 = [3.22, 6.44, 5.04, 7.25, 5.62]		
		# tnr2 = [75.0, 72.8, 72.1, 73.1, 60.5]
		# std_tnr2 = [4.23, 6.17, 6.91, 7.68, 5.76]		
		# accu3 = [72.6, 73.9, 73.8, 72.8, 72.2]
		# std_accu3 = [4.33, 2.67, 2.92, 4.38, 3.40]		
		# recall3 = [75.6, 79.8, 79.3, 79.9, 89.5]
		# std_recall3 = [4.63, 8.24, 7.82, 9.03, 6.11]		
		# tnr3 = [70.9, 70.0, 70.2, 67.6, 58.3]
		# std_tnr3 = [6.93, 4.95, 2.85, 6.27, 7.14]
		
		
		
		
		
		
		
		
		
		
		thr = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]	
		
		accu0 = [75.7, 85.2, 78.1, 78.4, 77.1, 68.2]
		std_accu0 = [2.54, 2.21, 3.19, 3.69, 4.52, 5.70]		
		recall0 = [75.2, 79.3, 78.8, 84.1, 81.7, 57.3]
		std_recall0 = [3.84, 4.60, 7.00, 6.37, 10.07, 13.71]		
		tnr0 = [75.9, 87.5, 77.7, 74.9, 74.7, 74.9]
		std_tnr0 = [4.74, 3.01, 5.05, 7.16, 3.54, 7.29]
		


		accu1 = [80.4, 77.7, 77.2, 82.0, 71.6, 69.8]
		std_accu1 = [1.95, 4.60, 4.19, 2.96, 4.89, 7.21]		
		recall1 = [69.0, 76.4, 80.6, 83.1, 63.2, 67.4]
		std_recall1 = [5.09, 7.80, 6.22, 7.86, 13.06, 10.93]		
		tnr1 = [84.7, 78.2, 75.5, 81.4, 76.8, 71.6]
		std_tnr1 = [1.56, 6.67, 6.46, 4.91, 3.93, 10.75]
		
		accu2 = [76.6, 80.0, 78.4, 76.8, 71.2, 71.2]
		std_accu2 = [3.49, 2.90, 3.76, 5.90, 7.51, 7.31]		
		recall2 = [76.3, 79.4, 87.3, 83.5, 67.4, 69.0]
		std_recall2 = [4.72, 6.65, 5.85, 7.75, 10.26, 12.13]		
		tnr2 = [76.8, 80.3, 73.1, 72.6, 73.9, 72.9]
		std_tnr2 = [4.09, 5.39, 5.00, 9.64, 8.59, 7.78]	
		
		accu3 = [72.9, 71.2, 76.0, 72.3, 72.6, 74.3]
		std_accu3 = [3.59, 5.00, 5.02, 6.10, 5.88, 8.40]		
		recall3 = [77.7, 80.4, 83.2, 84.3, 66.9, 77.6]
		std_recall3 = [6.15, 7.44, 6.34, 9.85, 11.78, 12.49]		
		tnr3 = [70.5, 65.9, 71.4, 64.4, 76.6, 71.6]
		std_tnr3 = [5.30, 7.64, 6.49, 9.34, 6.99, 12.28]
		
		
		fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
		plot_3_lines_together(ax[0], thr, accu1, accu2, accu3, '% Exactitud - Promedio')
		plot_3_lines_together(ax[1], thr, recall1, recall2, recall3, '% Sensibilidad Carbón - Promedio')
		plot_3_lines_together(ax[2], thr, tnr1, tnr2, tnr3, '% Sensibilidad Concreto - Promedio')
		fig.tight_layout()
		fig.set_size_inches(16, 4)
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		
		
		fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True)
		plot_3_lines_together(ax1[0], thr, std_accu1, std_accu2, std_accu3, '% Exactitud - DE')
		plot_3_lines_together(ax1[1], thr, std_recall1, std_recall2, std_recall3, '% Sensibilidad Carbón - DE')
		plot_3_lines_together(ax1[2], thr, std_tnr1, std_tnr2, std_tnr3, '% Sensibilidad Concreto - DE')
		fig1.tight_layout()
		fig1.set_size_inches(16, 4)
		
		for ax_it in ax1.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		
		
		
		plt.show()

	elif config['mode'] == 'line_plot_3_burst':
		
		
		var = [0.8, 0.9, 1, 1.1, 1.2]	
		
		mcc_thr = [0.4, 0.519, 0.572, 0.62, 0.625]
		mcc_win = [0.463, 0.469, 0.469, 0.472, 0.469]
		mcc_edg = [0.54, 0.588, 0.62, 0.64, 0.658]

		mcc_thr_std = [0.155, 0.158, 0.164, 0.158, 0.149]
		mcc_win_std = [0.091, 0.095, 0.098, 0.095, 0.084]
		mcc_edg_std = [0.169, 0.169, 0.165, 0.154, 0.151]			

		
		
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
		plot_3_lines_together(ax, var, mcc_thr, mcc_win, mcc_edg, 'MCC - Mean Value')
		# plot_3_lines_together(ax, var, mcc_thr_std, mcc_win_std, mcc_edg_std, 'MCC - Standard Deviation')
		fig.tight_layout()
		# fig.set_size_inches(5, 4)
		fig.set_size_inches(2.5, 1.6)
		# ax.tick_params(axis='both', labelsize=11)
		ax.tick_params(axis='both', labelsize=8)
		
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
		
		
		
		plt.tight_layout()
		# plt.subplots_adjust(wspace=0.23)
		plt.show()
		
		
		
		
		fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True)
		# plot_3_lines_together(ax, var, mcc_thr, mcc_win, mcc_edg, 'MCC - Mean Value')
		plot_3_lines_together(ax, var, mcc_thr_std, mcc_win_std, mcc_edg_std, 'MCC - Standard Deviation')
		fig.tight_layout()
		# fig.set_size_inches(10, 4)
		ax.tick_params(axis='both', labelsize=11)
		fig.set_size_inches(5, 4)
		# for ax_it in ax.flatten():
			# for tk in ax_it.get_yticklabels():
				# tk.set_visible(True)
			# for tk in ax_it.get_xticklabels():
				# tk.set_visible(True)
		
		
		
		
		
		
		
		plt.tight_layout()
		# plt.subplots_adjust(wspace=0.23)
		plt.show()
		
	elif config['mode'] == '2d_simple':
		
		
		log_LP = [2.699, 2.875, 3., 3.176, 3.398, 3.699, 4.301, 4.699, 4.845]
		log_ERR = [3.2, 1.279, 1.079, 1., 0.881, 1.204, 1.155, 2.159, 2.196]
		
		fig, ax = plt.subplots()
		ax.plot(log_LP, log_ERR, marker='o')
		# ax.set_xlabel('1 + log(f$_{cutoff}$)', fontsize=12)
		ax.set_xlabel(r'1 + log(f$_{\rm cutoff}$)', fontsize=14)
		ax.set_ylabel('1 + log(error)', fontsize=14)
		ax.tick_params(axis='both', labelsize=13)
		ax.set_xlim(xmax=5, xmin=2.5)
		ax.axvline(3.458, color='k', linewidth=2)	
		fig.text(0.41, 0.6, 'Wave average duration', ha='center', va='center', rotation='vertical', fontsize=13)		
		fig.set_size_inches(6, 4)
		plt.tight_layout()
		plt.show()
	
	elif config['mode'] == '2d_simple_dop':
		
		x = [120, 1400, 2600, 3900, 4800]
		y1 = [212.66, 215.96, 226.44, 221.69, 227.37]
		y2 = [183.46, 180.97, 184.84, 189.86, 189.84]
		
		fig, ax = plt.subplots()
		ax.plot(x, y1, marker='o', label='AE-1')
		ax.plot(x, y2, marker='v', label='AE-2')
		ax.legend()
		ax.set_title('Progression of Frequency Peaks', fontsize=13)
		ax.set_xlabel('N° Cycle', fontsize=13)
		ax.set_ylabel('Frequency [kHz]', fontsize=13)
		ax.tick_params(axis='both', labelsize=12)
		# ax.set_xlim(xmax=5, xmin=2.5)
		# ax.axvline(3.458, color='k', linewidth=2)	
		# fig.text(0.41, 0.6, 'Wave average duration', ha='center', va='center', rotation='vertical', fontsize=13)		
		fig.set_size_inches(6, 4)
		plt.tight_layout()
		plt.show()
	
	elif config['mode'] == 'barh_comparison':
		
		# scale = np.array([1, 2, 3, 4])
		# tick_label = ['A', 'B', 'C', 'D']
		# x1 = [1.23, 4.5, 8.7, 3.3]		
		# fig, ax = plt.subplots()		
		# rec1 = ax.barh(scale-0.1, width=x1, height=0.1, color='green')		
		# rec2 = ax.barh(scale, width=x1, height=0.1, tick_label=tick_label, color='red')
		# rec3 = ax.barh(scale+0.1, width=x1, height=0.1, color='blue')		
		# plt.show()
		
				
		scale = np.array([1, 2, 3, 4, 5])
		tick_label = ['T-1', 'T-2', 'T-3', 'T-4', 'Mean']
		
		recall_fix = np.array([0.807, 0.095, 0.575, 0.002, 0.375])
		recall_rms = np.array([0.769, 0.833, 0.805, 0.609, 0.754])
		
		
		fpr_fix = np.array([0.084, 0., 0.015, 0.001, 0.029])
		fpr_rms = np.array([0.055, 0.161, 0.122, 0.258, 0.149])
		
		mcc_fix = np.array([0.724, 0.276, 0.669, 0.102, 0.443])
		mcc_rms = np.array([0.739, 0.598, 0.649, 0.302, 0.572])
		
		
		fig, ax = plt.subplots(ncols=2, sharey=True)		
		ax[0].barh(scale-0.1, width=recall_fix, height=0.1, color='r', label='Recall')		
		ax[0].barh(scale, width=fpr_fix, height=0.1, color='y', label='FPR')
		ax[0].barh(scale+0.1, width=mcc_fix, height=0.1, color='deepskyblue', label='MCC')	
		
		ax[0].legend()
		
		ax[1].barh(scale-0.1, width=recall_rms, height=0.1, color='r')		
		ax[1].barh(scale, width=fpr_rms, height=0.1, color='y')
		ax[1].barh(scale+0.1, width=mcc_rms, height=0.1, color='deepskyblue')	

		
		
		ax[0].set(yticks=[], yticklabels=[], xlim=(0, 1), title='Fixed value')
		ax[1].set(yticks=[], yticklabels=[], xlim=(0, 1), title='RMS factor')
		
		ax[0].invert_xaxis()
		for yloc, label in zip(scale, tick_label):
			ax[0].annotate(label, (0.5, yloc), xycoords=('figure fraction', 'data'),
							 ha='center', va='center')
		plt.tight_layout()
		plt.show()
	
	
	elif config['mode'] == 'line_plot_4':
		
		
	
		
		thr = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]	
		
		accu0 = [75.7, 85.2, 78.1, 78.4, 77.1, 68.2]
		std_accu0 = [2.54, 2.21, 3.19, 3.69, 4.52, 5.70]		
		recall0 = [75.2, 79.3, 78.8, 84.1, 81.7, 57.3]
		std_recall0 = [3.84, 4.60, 7.00, 6.37, 10.07, 13.71]		
		tnr0 = [75.9, 87.5, 77.7, 74.9, 74.7, 74.9]
		std_tnr0 = [4.74, 3.01, 5.05, 7.16, 3.54, 7.29]
		


		accu1 = [80.4, 77.7, 77.2, 82.0, 71.6, 69.8]
		std_accu1 = [1.95, 4.60, 4.19, 2.96, 4.89, 7.21]		
		recall1 = [69.0, 76.4, 80.6, 83.1, 63.2, 67.4]
		std_recall1 = [5.09, 7.80, 6.22, 7.86, 13.06, 10.93]		
		tnr1 = [84.7, 78.2, 75.5, 81.4, 76.8, 71.6]
		std_tnr1 = [1.56, 6.67, 6.46, 4.91, 3.93, 10.75]
		
		accu2 = [76.6, 80.0, 78.4, 76.8, 71.2, 71.2]
		std_accu2 = [3.49, 2.90, 3.76, 5.90, 7.51, 7.31]		
		recall2 = [76.3, 79.4, 87.3, 83.5, 67.4, 69.0]
		std_recall2 = [4.72, 6.65, 5.85, 7.75, 10.26, 12.13]		
		tnr2 = [76.8, 80.3, 73.1, 72.6, 73.9, 72.9]
		std_tnr2 = [4.09, 5.39, 5.00, 9.64, 8.59, 7.78]	
		
		accu3 = [72.9, 71.2, 76.0, 72.3, 72.6, 74.3]
		std_accu3 = [3.59, 5.00, 5.02, 6.10, 5.88, 8.40]		
		recall3 = [77.7, 80.4, 83.2, 84.3, 66.9, 77.6]
		std_recall3 = [6.15, 7.44, 6.34, 9.85, 11.78, 12.49]		
		tnr3 = [70.5, 65.9, 71.4, 64.4, 76.6, 71.6]
		std_tnr3 = [5.30, 7.64, 6.49, 9.34, 6.99, 12.28]
		
		
		fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
		# plot_4_lines_together(ax[0], thr, accu0, accu1, accu2, accu3, '% Accuracy - Average')
		# plot_4_lines_together(ax[1], thr, recall0, recall1, recall2, recall3, '% Coal Sensitivity - Average')
		# plot_4_lines_together(ax[2], thr, tnr0, tnr1, tnr2, tnr3, '% Concrete Sensitivity - Average')
		
		plot_4_lines_together(ax[0], thr, accu0, accu1, accu2, accu3, '% Exactitud - Promedio')
		plot_4_lines_together(ax[1], thr, recall0, recall1, recall2, recall3, '% Sensibilidad al Carbón - Promedio')
		plot_4_lines_together(ax[2], thr, tnr0, tnr1, tnr2, tnr3, '% Sensibilidad al Concreto - Promedio')
		
		
		fig.tight_layout()
		fig.set_size_inches(18, 4)
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		
		
		fig1, ax1 = plt.subplots(nrows=1, ncols=3, sharey=True)
		# plot_4_lines_together(ax1[0], thr, std_accu0, std_accu1, std_accu2, std_accu3, '% Accuracy - SD')
		# plot_4_lines_together(ax1[1], thr, std_recall0, std_recall1, std_recall2, std_recall3, '% Coal Sensitivity - SD')
		# plot_4_lines_together(ax1[2], thr, std_tnr0, std_tnr1, std_tnr2, std_tnr3, '% Concrete Sensitivity - SD')
		
		plot_4_lines_together(ax1[0], thr, std_accu0, std_accu1, std_accu2, std_accu3, '% Exactitud - DE')
		plot_4_lines_together(ax1[1], thr, std_recall0, std_recall1, std_recall2, std_recall3, '% Sensibilidad al Carbón - DE')
		plot_4_lines_together(ax1[2], thr, std_tnr0, std_tnr1, std_tnr2, std_tnr3, '% Sensibilidad al Concreto - DE')
		
		
		fig1.tight_layout()
		fig1.set_size_inches(18, 4)
		
		for ax_it in ax1.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		
		
		# fig1.tight_layout()
		plt.show()
	
	
	elif config['mode'] == 'line_plot_4_thesis':
		
		
	
		
		thr = [1, 2, 3]	
		
		fc_env = [29.33, 28.21, 28.23]
		fc_env2 = [25.01, 27.25, 28.82]
		fc_kurt_env2 = [22.61, 28.07, 29.37]
		fc_emd_env2 = [10.27, 22.64, 30.47]
		
		fg_env = [18.31, 29.86, 45.99]
		fg_env2 = [15.20, 27.71, 43.79]
		fg_kurt_env2 = [15.71, 27.09, 42.48]
		fg_emd_env2 = [0.32, 4.44, 32.11]
		
		sb_env = [34.71, 36.39, 42.18]
		sb_env2 = [34.11, 37.38, 44.75]
		sb_kurt_env2 = [33.61, 38.85, 45.59]
		sb_emd_env2 = [29.20, 33.17, 46.28]
		
		
		
		
		
		
		
		sys.exit()
		
		fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
		# plot_4_lines_together(ax[0], thr, accu0, accu1, accu2, accu3, '% Accuracy - Average')
		# plot_4_lines_together(ax[1], thr, recall0, recall1, recall2, recall3, '% Coal Sensitivity - Average')
		# plot_4_lines_together(ax[2], thr, tnr0, tnr1, tnr2, tnr3, '% Concrete Sensitivity - Average')
		
		plot_4_lines_together(ax[0], thr, fc_env, fc_env2, fc_kurt_env2, fc_emd_env2, 'fc')
		plot_4_lines_together(ax[1], thr, fg_env, fg_env2, fg_kurt_env2, fg_emd_env2, 'fg')
		plot_4_lines_together(ax[2], thr, sb_env, sb_env2, sb_kurt_env2, sb_emd_env2, 'sb')
		
		
		fig.tight_layout()
		fig.set_size_inches(18, 4)
		
		for ax_it in ax.flatten():
			for tk in ax_it.get_yticklabels():
				tk.set_visible(True)
			for tk in ax_it.get_xticklabels():
				tk.set_visible(True)
		
		
		
		plt.show()
	
	
	elif config['mode'] == 'bar_plot_2':

		
		accu_means = (77.0, 77.6, 73.7, 77.8)
		accu_std = (3.47, 2.78, 3.90, 3.24)
		
		recall_means = (78.1, 81.5, 74.1, 78.2)
		recall_std = (7.87, 6.22, 8.17, 7.08)
		
		
		tnr_means = (76.4, 75.4, 73.4, 77.6)
		tnr_std = (3.91, 4.27, 5.15, 3.80)

		plot_bar_error(accu_means, accu_std, recall_means, recall_std, tnr_means, tnr_std)
	
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
	# config['fs_tacho'] = float(config['fs_tacho'])
	# config['fs_signal'] = float(config['fs_signal'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config




def plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std):

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig1, ax = plt.subplots()
	rects1 = ax.bar(ind, recall_means, width, color='r', yerr=recall_std, error_kw=dict(lw=2, capsize=2, capthick=2))


	rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Scores %', fontsize=12)
	ax.set_title('Test Results', fontsize=12)
	ax.set_xticks(ind + width / 2)
	ax.set_ylim((0, 100))
	ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'), fontsize=12)
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	ax.legend((rects1[0], rects2[0]), ('Recall', 'Precision'), loc='best')


	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2. + 0.095, 1.000*height,
					'%d' % round(height),
					ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	# plt.show()

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig2, ax = plt.subplots()
	rects3 = ax.bar(ind, fscore_means, width, color='deepskyblue', yerr=fscore_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	ax.set_ylabel('Scores %', fontsize=12)
	ax.set_title('Test Results', fontsize=12)
	ax.set_xticks(ind)
	ax.set_ylim((0, 100))
	# ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))
	ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'), fontsize=12)
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	# ax.legend((rects1), ('F-Score'), loc='best')
	ax.legend(('F-Score',), loc='best')


	autolabel(rects3)

	plt.show()
	return

def plot_scores_error_4(recall_means, recall_std, fpr_means, fpr_std, mcc_means, mcc_std):

	N = 3


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig1, ax = plt.subplots()
	rects1 = ax.bar(ind, recall_means, width, color='r', yerr=recall_std, error_kw=dict(lw=2, capsize=2, capthick=2))


	rects2 = ax.bar(ind + width, fpr_means, width, color='y', yerr=fpr_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Scores %', fontsize=12)
	# ax.set_title('Test Results', fontsize=12)
	ax.tick_params(axis='both', labelsize=11)
	ax.set_xticks(ind + width / 2)
	ax.set_ylim((0, 1))
	ax.set_xticklabels(('THR', 'WIN', 'EDG'), fontsize=12)
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	ax.legend((rects1[0], rects2[0]), ('Recall', 'FPR'), loc='best')
	fig1.set_size_inches(5.3, 4.3)
	
	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2. + 0.165, 1.000*height,
					'%.3f' % height,
					ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)
	plt.tight_layout()

	# plt.show()

	N = 3


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig2, ax = plt.subplots()
	rects3 = ax.bar(ind, mcc_means, width, color='deepskyblue', yerr=mcc_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	ax.set_ylabel('Scores %', fontsize=12)
	# ax.set_title('Test Results', fontsize=12)
	ax.tick_params(axis='both', labelsize=11)
	ax.set_xticks(ind)
	ax.set_ylim((0, 1))
	# ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))
	ax.set_xticklabels(('THR', 'WIN', 'EDG'), fontsize=12)
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	# ax.legend((rects1), ('F-Score'), loc='best')
	ax.legend(('MCC',), loc='best')
	autolabel(rects3)
	fig2.set_size_inches(5.3, 4.3)
	plt.tight_layout()
	plt.show()
	return

def plot_3_lines(thr, accu, recall, tnr, std_accu, std_recall, std_tnr):
	fig, ax = plt.subplots(ncols=3, sharex=False, sharey=True)

	ax[0].errorbar(x=thr, y=accu, yerr=std_accu, lw=2, capsize=2, capthick=2, label='Exactitud')
	ax[0].set_ylabel('% Exactitud')
	ax[0].set_xlabel('Umbral V')
	ax[0].set_xticks(thr)		
	
	ax[1].errorbar(x=thr, y=recall, yerr=std_recall, lw=2, capsize=2, capthick=2, label='Sensibilidad Carbón')
	ax[1].set_ylabel('% Sensibilidad Carbón')
	ax[1].set_xlabel('Umbral V')
	ax[1].set_xticks(thr)
	ax[2].errorbar(x=thr, y=tnr, yerr=std_tnr, lw=2, capsize=2, capthick=2, label='Sensibilidad Concreto')
	ax[2].set_ylabel('% Sensibilidad Concreto')
	ax[2].set_xlabel('Umbral V')
	ax[2].set_xticks(thr)
	
	fig.tight_layout()
	fig.set_size_inches(16, 4)
	ax[0].set_ylim(bottom=50, top=100)
	ax[1].set_ylim(bottom=50, top=100)
	ax[2].set_ylim(bottom=50, top=100)
	
	for ax_it in ax.flatten():
		for tk in ax_it.get_yticklabels():
			tk.set_visible(True)
		for tk in ax_it.get_xticklabels():
			tk.set_visible(True)
	
	
	plt.show()
	
	return



def plot_3_lines_together(ax, thr, line1, line2, line3, name_y):
	# fig, ax = plt.subplots()

	# ax.plot(thr, line1, label='1 ms', marker='o')
	# ax.plot(thr, line2, label='2 ms', marker='v')
	# ax.plot(thr, line3, label='3 ms', marker='s')
	
	# ax.plot(thr, line1, label='THR', marker='o', markersize=4)
	# ax.plot(thr, line2, label='WIN', marker='v')
	# ax.plot(thr, line3, label='EDG', marker='s')
	
	ax.plot(thr, line1, label='30%', marker='o', markersize=4)
	ax.plot(thr, line2, label='55%', marker='v', markersize=4)
	ax.plot(thr, line3, label='80%', marker='s', markersize=4)
	
	ax.set_ylabel(name_y, fontsize=12)
	
	# ax.set_xlabel('Factor of selected threshold value', fontsize=7)
	ax.set_xlabel('Rotational speed [RPM]', fontsize=12)
	
	ax.tick_params(axis='both', labelsize=12)
	ax.set_xticks(thr)		
	ax.legend(fontsize=12)
	
	
	# ax.set_ylabel(name_y, fontsize=12)
	# ax.set_xlabel('Factor of selected threshold value', fontsize=12)
	# ax.tick_params(axis='both', labelsize=12)
	# ax.set_xticks(thr)		
	# ax.legend()
	
	# ax.set_ylim(bottom=2, top=9.5)

	
	
	# fig.set_size_inches(16, 4)
	# ax.set_ylim(bottom=50, top=100)

	
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
	
	
	# plt.show()
	
	return


def plot_3_lines_together_alone(ax, thr, line1, line2, line3, name_y):
	# fig, ax = plt.subplots()

	# ax.plot(thr, line1, label='1 ms', marker='o')
	# ax.plot(thr, line2, label='2 ms', marker='v')
	# ax.plot(thr, line3, label='3 ms', marker='s')
	
	ax.plot(thr, line1, label='THR', marker='o')
	ax.plot(thr, line2, label='WIN', marker='v')
	ax.plot(thr, line3, label='EDG', marker='s')
	
	ax.set_ylabel(name_y, fontsize=12)
	ax.set_xlabel('Factor of selected threshold value', fontsize=12)
	ax.tick_params(axis='both', labelsize=12)
	ax.set_xticks(thr)		
	ax.legend()
	
	# ax.set_ylim(bottom=2, top=9.5)

	
	
	# fig.set_size_inches(16, 4)
	# ax.set_ylim(bottom=50, top=100)

	
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
	
	
	# plt.show()
	
	return

def plot_bar_error(accu_means, accu_std, recall_means, recall_std, tnr_means, tnr_std):

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.25       # the width of the bars

	fig1, ax = plt.subplots()
	
	rects1 = ax.bar(ind + width, accu_means, width, color='y', yerr=accu_std, error_kw=dict(lw=2, capsize=2, capthick=2))
	rects2 = ax.bar(ind, recall_means, width, color='r', yerr=recall_std, error_kw=dict(lw=2, capsize=2, capthick=2))
	
	
	
	rects3 = ax.bar(ind + 2*width, tnr_means, width, color='g', yerr=tnr_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	
	# add some text for labels, title and axes ticks
	ax.set_ylabel('% Puntuación')
	# ax.set_title('Test Results')
	ax.set_xticks(ind + width / 2)
	ax.set_ylim((0, 100))
	ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	ax.legend((rects1[0], rects2[0], rects3[0]), ('Exactitud', 'Sensibilidad Carbón', 'Sensibilidad Concreto'), loc='lower left')


	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2. + 0.095, 1.000*height,
					'%d' % round(height),
					ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	# plt.show()

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	# fig2, ax = plt.subplots()
	# rects3 = ax.bar(ind, fscore_means, width, color='deepskyblue', yerr=fscore_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	# ax.set_ylabel('Scores %')
	# ax.set_title('Test Results')
	# ax.set_xticks(ind)
	# ax.set_ylim((0, 100))
	# # ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))
	# ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# # ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	# # ax.legend((rects1), ('F-Score'), loc='best')
	# ax.legend(('F-Score',), loc='best')


	# autolabel(rects3)

	plt.show()
	return

	
def plot_4_lines_together(ax, thr, line0, line1, line2, line3, name_y):
	# fig, ax = plt.subplots()
	
	# ax.plot(thr, line0, label='0.5 (ms)', marker='^')
	# ax.plot(thr, line1, label='1 (ms)', marker='o')
	# ax.plot(thr, line2, label='2 (ms)', marker='v')
	# ax.plot(thr, line3, label='3 (ms)', marker='s')
	
	ax.plot(thr, line0, label='400 [RPM]', marker='^')
	ax.plot(thr, line1, label='700 [RPM]', marker='o')
	ax.plot(thr, line2, label='1000 [RPM]', marker='v')
	ax.plot(thr, line3, label='1300 [RPM]', marker='s')
	
	ax.set_ylabel(name_y, fontsize=12)
	# ax.set_xlabel('Amplitude Threshold (V)', fontsize=11)
	# ax.set_xlabel('Umbral de Amplitud (V)', fontsize=12)
	ax.set_xlabel('Load [%]', fontsize=12)

	ax.tick_params(axis='both', labelsize=12)
	ax.set_xticks(thr)		
	ax.legend()
	# ax.set_ylim(bottom=55, top=90)
	# ax.set_ylim(bottom=0, top=14)


	
	
	# fig.set_size_inches(16, 4)
	# ax.set_ylim(bottom=50, top=100)

	
	# for ax_it in ax.flatten():
		# for tk in ax_it.get_yticklabels():
			# tk.set_visible(True)
		# for tk in ax_it.get_xticklabels():
			# tk.set_visible(True)
	
	
	# plt.show()
	
	return	

if __name__ == '__main__':
	main(sys.argv)





