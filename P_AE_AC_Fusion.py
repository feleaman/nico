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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {'channel':'OFF', 'fs':'OFF', 'name':'auto', 'color':'red', 'marker':'D', 'save':'OFF', 'rs':None, 'test_size':0.3, 'cv':10, 'penalty':1., 'kernel':'linear'}
# import sklearn
# print(sklearn.metrics.SCORERS.keys())
# sys.exit()
def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)

	if config['mode'] == 'train_test_svm':
		
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()	
		print(os.path.basename(filepath))
		
		mydict = pd.read_excel(filepath)
		mydict = mydict.to_dict(orient='list')
		
		mykeys = ['STD', 'MAX', 'KURT']
		y = np.array(mydict['label'])
		X = []
		for key in mydict.keys():
			if key in mykeys:
				print(key)
				X.append(mydict[key])
		X = np.transpose(np.array(X))
		
		
		
		print('\n++++++++++++++++Results First Split++++++++++++++++\n')
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=None)		
		
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		
		clf_1 = SVC(kernel='linear', C=0.1, gamma='auto', verbose=False, max_iter=100000)
		clf_1.fit(X_train, y_train)
		scores = cross_val_score(clf_1, X_train, y_train, cv=10, scoring='f1')
		print('\n+++Validation Score: ', scores)
		
		print("Validation Score Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		
		Pred_y_test = clf_1.predict(X_test)
		print('\n+++Test Accuracy: ', accuracy_score(y_test, Pred_y_test))
		print('+++Test Recall: ', recall_score(y_test, Pred_y_test))
		print('+++Test Precision: ', precision_score(y_test, Pred_y_test))
		print('+++Test F1: ', f1_score(y_test, Pred_y_test))
		
		if config['save'] == 'ON':
			save_pickle('clf_' + config['name'] + '.pkl', clf_1)
		
		
		sys.exit()
		print('\n++++++++++++++++Results Second Split++++++++++++++++\n')
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=None)
		
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		
		
		kf = StratifiedKFold(n_splits=10)
		kf.get_n_splits(X_train)
		Accus = []
		for valid_train_index, valid_test_index in kf.split(X_train, y_train):
			# print('TRAIN:', valid_train_index, 'VALID:', valid_test_index)
			Valid_X_train, Valid_X_test = X_train[valid_train_index], X_train[valid_test_index]
			Valid_y_train, Valid_y_test = y_train[valid_train_index], y_train[valid_test_index]
	
			clf = SVC(kernel='linear', C=0.5, gamma='auto', verbose=False, max_iter=5000)
			clf.fit(Valid_X_train, Valid_y_train) 
			Pred_Valid_y_test = clf.predict(Valid_X_test)
			Accus.append(accuracy_score(Valid_y_test, Pred_Valid_y_test))
		print('+++Validation Accuracy: ', Accus)
		
		Pred_y_test = clf.predict(X_test)
		print('\n+++Test Accuracy: ', accuracy_score(y_test, Pred_y_test))
		print('+++Test Recall: ', recall_score(y_test, Pred_y_test))
		print('+++Test Precision: ', precision_score(y_test, Pred_y_test))
	
	elif config['mode'] == 'learn_svm':		
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()	
		filename = os.path.basename(filepath)
		
		mydict = pd.read_excel(filepath)
		mydict = mydict.to_dict(orient='list')
		
		mykeys = ['STD', 'MAX', 'KURT']
		# mykeys = list(mydict.keys())
		
		y = np.array(mydict['label'])
		X = []
		for key in mydict.keys():
			if key in mykeys:
				print(key)
				X.append(mydict[key])
		X = np.transpose(np.array(X))
		
		
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['rs'])		
		
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		
		# pca = PCA(n_components=3)
		# pca.fit(X_train)
		# # print(pca.explained_variance_ratio_)
		# # sys.exit()
		# X_train = pca.transform(X_train)
		# X_test = pca.transform(X_test)
		
		
		# Penalizations = [0.1, 1.0, 100., 1000., 10000.]
		Penalizations = [0.1, 1.0]
		# Kernels = ['linear', 'poly', 'rbf']
		Kernels = ['linear']

		results = {}
		results['f1_va_mean'] = []
		results['f1_va_std'] = []
		results['f1_te'] = []
		results['penal'] = []
		results['kernel'] = []
		
		for kernel_ in Kernels:
			for penal_ in Penalizations:
		
				clf_1 = SVC(kernel=kernel_, C=penal_, gamma='auto', verbose=False, max_iter=-1)
				clf_1.fit(X_train, y_train)
				scores = cross_val_score(clf_1, X_train, y_train, cv=config['cv'], scoring='f1')
				# print('\n+++Validation Score: ', scores)
				score_mean = scores.mean()
				score_std = scores.std()
				
				# print("Validation Score Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
				
				Pred_y_test = clf_1.predict(X_test)
				# print('\n+++Test Accuracy: ', accuracy_score(y_test, Pred_y_test))
				# print('+++Test Recall: ', recall_score(y_test, Pred_y_test))
				# print('+++Test Precision: ', precision_score(y_test, Pred_y_test))
				score_test = f1_score(y_test, Pred_y_test)
				print('\n+++Test F1: ', score_test)
		
				results['f1_va_mean'].append(score_mean)
				results['f1_va_std'].append(score_std)
				results['f1_te'].append(score_test)
				results['penal'].append(penal_)
				results['kernel'].append(kernel_)
				
				
		
		
		
		if config['save'] == 'ON':
			config['features'] = mykeys
			config['filename'] = filename
			
			from datetime import datetime
			# print(datetime.now())
			# name = str(datetime.now())
			name = datetime.now().strftime("%Y%m%d_%H%M%S")
			print(name)
			save_pickle('config_' + name + '.pkl', config)			
			DataFr = pd.DataFrame(data=results, index=None)
			writer = pd.ExcelWriter('results_' + name + '.xlsx')			
			DataFr.to_excel(writer, sheet_name='SVM_Learn')	
			print('Result OK')
	
	
	elif config['mode'] == 'generate_svm':
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()	
		filename = os.path.basename(filepath)
		print(filename)
		
		mydict = pd.read_excel(filepath)
		mydict = mydict.to_dict(orient='list')
		
		mykeys = ['STD', 'MAX', 'KURT']
		# mykeys = ['STD', 'MAX', 'KURT', 'SEN']
		y = np.array(mydict['label'])
		X = []
		for key in mydict.keys():
			if key in mykeys:
				print(key)
				X.append(mydict[key])
		X = np.transpose(np.array(X))
		
		
		
		
		scaler = StandardScaler()
		scaler.fit(X)
		X = scaler.transform(X)
		
		# pca = PCA(n_components=12)
		# pca.fit(X)
		# X = pca.transform(X)
		
		clf = SVC(kernel=config['kernel'], C=config['penalty'], gamma='auto', verbose=True, max_iter=-1, random_state=config['rs'])
		
		clf.fit(X, y)
		
		if config['save'] == 'ON':
			config['features'] = mykeys
			config['filename'] = filename
			
			from datetime import datetime
			name = datetime.now().strftime("%Y%m%d_%H%M%S")
			
			save_pickle('config_' + name + '.pkl', config)
			save_pickle('clf_' + name + '.pkl', clf)	
			save_pickle('scaler_' + name + '.pkl', scaler)
			# save_pickle('pca_' + name + '.pkl', pca)	
			
	
	elif config['mode'] == 'test_clf':	
		
		print('Select model...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		clf = read_pickle(filepath)
		
		print('Select scaler...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		scaler = read_pickle(filepath)
		
		# print('Select pca...')
		# root = Tk()
		# root.withdraw()
		# root.update()
		# filepath = filedialog.askopenfilename()
		# root.destroy()
		# pca = read_pickle(filepath)
		
		
		print('Select data...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()	
		print(os.path.basename(filepath))
		
		mydict = pd.read_excel(filepath)
		mydict = mydict.to_dict(orient='list')
		
		
		mykeys = ['STD', 'MAX', 'KURT']
		# mykeys = ['STD', 'MAX', 'KURT', 'SEN']
		y = np.array(mydict['label'])
		X = []
		for key in mydict.keys():
			if key in mykeys:
				print(key)
				X.append(mydict[key])
		X = np.transpose(np.array(X))
		
		X = scaler.transform(X)
		
		# X = pca.transform(X)
		print(X.shape)
		pred_y = clf.predict(X)
		
		print('Predictions: ', pred_y)
		# print('+++Length: ', len(pred_y))
		
		print('\n+++CLF Test Accuracy RAW: ', accuracy_score(y, pred_y, normalize=True))
		# print('+++CLF Test Accuracy: ', accuracy_score(y, pred_y, normalize=True))
		# print('+++CLF Test Recall: ', recall_score(y, pred_y))
		# print('+++CLF Test Precision: ', precision_score(y, pred_y))
		
	
	elif config['mode'] == 'scatter_3d':
		from mpl_toolkits.mplot3d import Axes3D
		
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()	
		filename = os.path.basename(filepath)
		
		mydict = pd.read_excel(filepath)
		mydict = mydict.to_dict(orient='list')
		
		mykeys = ['STD', 'MAX', 'KURT']
		
		y = np.array(mydict['label'])
		X = []
		for key in mydict.keys():
			if key in mykeys:
				print(key)
				X.append(mydict[key])
		X = np.transpose(np.array(X))
		print(X.shape)
		# sys.exit()
		
		# scaler = StandardScaler()
		# scaler.fit(X)
		# X = scaler.transform(X)
		
		X_pos = []
		X_neg = []
		for i in range(len(X[:,0])):
			if y[i] == 1:
				X_pos.append(X[i,:])
			else:
				X_neg.append(X[i,:])
		X_pos = np.array(X_pos)
		X_neg = np.array(X_neg)
		print(X_pos.shape)
		print(X_neg.shape)
		
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		
		# xs = X[:,0]
		# ys = X[:,1]
		# zs = X[:,2]
		m1 = 'o'
		m2 = 'v'
		c1 = 'red'
		c2 = 'blue'
		
		# for i in range(len(X[:,0])):
			# if y[i] == 1:
				# ax.scatter(X[i,0], X[i,1], X[i,2], marker=m1, color=c1)
			# else:
				# ax.scatter(X[i,0], X[i,1], X[i,2], marker=m2, color=c2)
		# plt.show()
		
		plt.rcParams['mathtext.fontset'] = 'cm'
		plt.rcParams['font.family'] = 'Times New Roman'
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
		
		ax.scatter(X_pos[:,0], X_pos[:,1], X_pos[:,2], marker=m1, color=c1, label='Fault')
		ax.scatter(X_neg[:,0], X_neg[:,1], X_neg[:,2], marker=m2, color=c2, label='No fault')
		
		# , handletextpad=0.3, labelspacing=.3
		fsbig = 14
		ax.legend(fontsize=fsbig-2, handletextpad=0.3, labelspacing=.3, loc='upper left')
		ax.set_xlabel(r'$\hat{\rho}_{std}$ [-]', fontsize=fsbig)
		ax.set_ylabel(r'$\hat{\rho}_{max}$ [-]', fontsize=fsbig)
		ax.set_zlabel(r'$\hat{\rho}_{kurt}$ [-]', fontsize=fsbig)
		ax.tick_params(axis='both', labelsize=fsbig-2)
		# ax.set_xlim(left=-5, right=2)
		# ax.set_ylim(-2, 3)
		# ax.set_zlim(-2, 3)
		plt.show()
	
	elif config['mode'] == 'simulate':
		#+++++++++Simulation Data
		fs = 1000000.
		dt = 1./fs
		tr = 0.25
		thr = 0.5
		alpha_bursts = 300.
		dura_bursts = 0.001
		frec_bursts = 92000.
		amort_bursts = 7887. #7887.
		planets = 3.
		carrier = 5.
		
		std_add_noise = 0.15 #0.15
		std_phase_noise = 0. #0.0005
		std_mult_noise = 0.001 #0.001
		
		amp_mod_1 = 0.0 #0.4 sinus
		amp_mod_2 = 0.4 #0.4 local
		
		#+++++++++++Time Array
		t = np.arange(0, tr, dt)
		n = len(t)
		
		#Noise Generator
		Noise = np.random.RandomState(seed=1)		
		phase_noise = 0.5*Noise.normal(1,std_phase_noise,n)
		add_noise = Noise.normal(0,std_add_noise,n)
		mult_noise = Noise.normal(1,std_mult_noise,n)
		
		
		#+++++++++++++Parent Signals
		#Train Pulses
		xc = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)
		rect = signal.square(2*np.pi*alpha_bursts*t, duty=phase_noise)		
		
		index = []
		for i in range(n-1):
			if xc[i] < thr and xc[i+1] >= thr:
				index.append(i)
		index = np.array(index)		
		nidx = len(index)		
		
		#Bursts
		tb = np.arange(0, dura_bursts, dt)
		burst = np.sin(2*np.pi*frec_bursts*tb)*np.exp(-amort_bursts*tb)
		nb = len(burst)
		
		for idx in index:
			xc[idx:idx+nb] += burst
			
		#No modulated signal
		x = xc - rect
		
		
		#+++++++++++++Operational Modulations
		# #Distance Pitch-Sensor Sinusoidal 
		mod_s = np.ones(n) + amp_mod_1*np.cos(2*np.pi*planets*carrier*t)
		
		#Local Fault Ring Gear Pulse
		mod_p = np.ones(n) + amp_mod_2*signal.square(2*np.pi*planets*carrier*t, duty=0.0055)	

		
		x = x*mod_p
		add_noise = add_noise*mod_p
		
		# x = x*mod_s
		add_noise = add_noise*mod_s

			
		#+++++++++++++Noise
		x += add_noise
		x = x*mult_noise
		
		
		# plt.plot(x)
		# plt.show()
		
		# #+++++++++++++Special Burst
		# thr = 1.
		# index = []
		# for i in range(n-1):
			# if x[i] < thr and x[i+1] >= thr:
				# print('special burst!')
				# index.append(i)
		# index = np.array(index)		
		# nidx = len(index)		
		# #Bursts
		# tb = np.arange(0, dura_bursts, dt)
		# burst = np.sin(2*np.pi*frec_bursts*2*tb+np.pi/8)*np.exp(-amort_bursts*1.5*tb)
		# nb = len(burst)
		
		# for idx in index:
			# x[idx-42:idx-42+nb] += 1.3*burst
			
		#+++++++++++++Plot
		plt.plot(t, x, 'r')
		plt.show()
		
		#acceleration
		zp = 27
		zc = 51
		fp = 1.
		fc = zp/zc * fp
		fs = 1.e6
		tr = 10
		dt = 1./fs
		t = np.arange(0, tr, dt)
		n = len(t)
		fm = fp*zp
		ac = np.cos(2*np.pi*fc*t+0.32) + np.cos(2*np.pi*fp*t+0.43) + np.cos(2*np.pi*fm*t+0.57)
		plt.plot(t, ac)
		plt.show()
		
		magAC, f, df = mag_fft(ac, fs)
		plt.plot(f, magAC)
		plt.show()
		
		
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
	config['test_size'] = float(config['test_size'])
	config['penalty'] = float(config['penalty'])
	config['cv'] = int(config['cv'])
	if config['rs'] != None:
		config['rs'] = int(config['rs'])
	#Type conversion to int	
	# Variable conversion
	return config

def myplot_scatter_2h(data, style):
	# from matplotlib import font_manager	
	# del font_manager.weight_dict['roman']
	# font_manager._rebuild()
	plt.rcParams['mathtext.fontset'] = 'cm'
	plt.rcParams['font.family'] = 'Times New Roman'	
	
	
	# fig, ax = plt.subplots(ncols=3, nrows=1, sharey='row')
	fig, ax = plt.subplots(ncols=2, nrows=1)
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
	plt.subplots_adjust(wspace=0.33, left=0.09, right=0.97, bottom=0.213, top=0.89)
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
			ax[j].semilogy(data['x'][count], data_y, label=style['legend'][count])
		else:
			ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker=style['marker'], ls='', color=style['color'])
			# ax[j].plot(data['x'][count], data_y, label=style['legend'][count], marker='o', ls='')
			# ax[j].plot(data['x'][count], data_y, label=style['legend'][count])
			# ax[j].bar(data['x'][count], data_y, label=style['legend'][count])
		
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




	# ax[0][1].set_yticklabels([-15, 0, 15, 30])
	# ax[0][1].set_yticks([-15, 0, 15, 30])
	
	
	#Set Limits in Axis X
	
		
	
		
	# ax[0].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	# ax[1].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	# ax[2].set_yticks([1.e4, 1.e5, 1.e6, 1.e7])
	
	# ax[0].set_yticks([-50, -30, -10, 10, 30])
	# ax[1].set_yticks([-50, -30, -10, 10, 30])
	# ax[2].set_yticks([-50, -30, -10, 10, 30])	
	
	# ax[0].set_xticks([1, 2, 3, 4, 5, 6])
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

if __name__ == '__main__':
	main(sys.argv)
