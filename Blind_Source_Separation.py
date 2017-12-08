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
	if config['mode'] == 'example':

		# #############################################################################
		# Generate sample data
		np.random.seed(0)
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
		print(len(X))

		# Compute ICA
		ica = FastICA(n_components=3)
		S_ = ica.fit_transform(X)  # Reconstruct signals
		A_ = ica.mixing_  # Get estimated mixing matrix

		# We can `prove` that the ICA model applies by reverting the unmixing.
		assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

		# For comparison, compute PCA
		pca = PCA(n_components=3)
		# print(type(X))
		# print(len(X))
		# plt.plot(X)
		# plt.show()
		# print(X)
		H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
		# print(H)

		# #############################################################################
		# Plot results

		plt.figure()
		plt.plot(H.T)
		plt.show()

		# models = [X, S, S_, H]
		# names = ['Observations (mixed signal)',
				 # 'True Sources',
				 # 'ICA recovered signals',
				 # 'PCA recovered signals']
		# colors = ['red', 'steelblue', 'orange']

		# for ii, (model, name) in enumerate(zip(models, names), 1):
			# plt.subplot(4, 1, ii)
			# plt.title(name)
			# for sig, color in zip(model.T, colors):
				# plt.plot(sig, color=color)

		# plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
		# plt.show()
	
	elif config['mode'] == 'own_example':
		# #############################################################################
		# Generate sample data
		np.random.seed(0)
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

		# Compute ICA
		ica = FastICA(n_components=3)
		S_ = ica.fit_transform(X)  # Reconstruct signals
		A_ = ica.mixing_  # Get estimated mixing matrix

		# We can `prove` that the ICA model applies by reverting the unmixing.
		assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

		# For comparison, compute PCA
		pca = PCA(n_components=3)
		H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

		# #############################################################################
		# Plot results

		plt.figure()

		models = [X, S, S_, H]
		names = ['Observations (mixed signal)',
				 'True Sources',
				 'ICA recovered signals',
				 'PCA recovered signals']
		colors = ['red', 'steelblue', 'orange']

		for ii, (model, name) in enumerate(zip(models, names), 1):
			plt.subplot(4, 1, ii)
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
