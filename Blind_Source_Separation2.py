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


sys.path.insert(0, './lib')
from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


from sklearn.decomposition import FastICA, PCA
from scipy import signal

#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {'power2':'OFF'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	if config['mode'] == 'test':
		print('test')
		# f1 = 10.
		# f2 = 12.
		# dt = 10000
		# n_points = 10000
		# x1 = np.array([np.sin(2*np.pi*f1*i*dt) for i in range(n_points)])
		# x2 = np.array([np.cos(2*np.pi*f2*i*dt) for i in range(n_points)])
		# x3 = np.array([1.05*np.sin(2*np.pi*f1*i*dt) for i in range(n_points)])
		# x4 = np.array([1.15*np.cos(2*np.pi*f2*i*dt) for i in range(n_points)])
		
		# X = x1 + x2
		# X = X.tolist()
		# x_temp = x3 + x4
		# x_temp = x_temp.tolist()
		# X.append(x_temp)
		# print(X)
		# sys.exit()
		
		
		
		n_samples = 2000
		time = np.linspace(0, 8, n_samples)		
		
		s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
		s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
		s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

		S = np.c_[s1, s2, s3]
		S += 0.2 * np.random.normal(size=S.shape)  # Add noise

		S /= S.std(axis=0)  # Standardize data
		# Mix data
		A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
		X = np.dot(S, A.T)  # Generate observations
		print(X)
		
		
		# plt.figure(0)
		# plt.plot(X[0])
		
		# plt.figure(1)
		# plt.plot(X[1])
		
		# plt.figure(2)
		# plt.plot(X[2])
		# plt.show()
		
		# print()
		
		pca = PCA(n_components=3)
		# X = [x1, x2]
		H = pca.fit_transform(X)
		
		# A = np.dot(X, H.T)
		# print(len(A))
		# print(len(A[0]))
		
		# plt.figure()
		# plt.plot(np.dot(X, H.T))
		# plt.show()
		
		
		models = [X, H]
		names = ['Observations (mixed signal)',
				 'PCA recovered signals']
		colors = ['red', 'steelblue', 'orange']

		for ii, (model, name) in enumerate(zip(models, names), 1):
			plt.subplot(2, 1, ii)
			plt.title(name)
			for sig, color in zip(model.T, colors):
				plt.plot(sig, color=color)

		plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
		plt.show()

	

	
	else:
		print('mode unknown')
	
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


if __name__ == '__main__':
	main(sys.argv)
