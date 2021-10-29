#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# import os.path
from sys import path, argv, exit


path.insert(0, './lib')
# from m_open_extension import *
# from m_fft import *
# from m_demodulation import *
# from m_denois import *
# from m_det_features import *
# from m_processing import *
# from decimal import Decimal
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
from argparse import ArgumentParser

from pykalman import UnscentedKalmanFilter
from pykalman import KalmanFilter
from pykalman import AdditiveUnscentedKalmanFilter
#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++
Inputs = ['mode']
InputsOpt_Defaults = {}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	if config['mode'] == 'normal':
		print('nothing')
		n = 500
		x_pure = np.arange(n)*0.1
		x = x_pure + np.random.normal(scale=0.25, size=n)
		y = np.cos(x)
		# y = 2*x+ 6

		
		kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, random_state=1)
		# measurements = [[x[i], y[i]] for i in range(int(n))]
		measurements = [[y[i]] for i in range(int(n/2))]
		# measurements_full = [[y[i]] for i in range(int(n))]
		
		
		
		
		kf.em(measurements)	
		# print(kf.initial_state_mean)
		# print(kf.initial_state_covariance)
		
		# print(type(kf.initial_state_mean))
		# print(type(kf.initial_state_covariance))
		# exit()
		
		z = kf.filter(measurements)
		
		
		
		means = z[0]
		covs = z[1]
		
		it_mean = means[len(means)-1]
		it_cov = covs[len(covs)-1]
		
		h = list(means)
		r = h
		
		# kf2 = KalmanFilter(initial_state_mean=it_mean, initial_state_covariance=it_cov, n_dim_obs=1)
		# kf2.em(measurements)
		f = list(kf.sample(int(n/2), initial_state=it_mean, random_state=1)[0])	
		# f = h + list(np.array(f) + it_mean)
		f = h + f

	
		for i in range(int(n/2)):
			
		
			it_z = kf.filter_update(filtered_state_mean=it_mean, filtered_state_covariance=it_cov, observation=[y[i+int(n/2)]])
			it_mean = it_z[0]
			it_cov = it_z[1]
			h.append(it_mean)
			
			
		
		
		plt.plot(x_pure, h, 'r')
		plt.plot(x_pure, y)
		plt.plot(x_pure, f, 'g')
		plt.show()
	
	elif config['mode'] == 'ukalman':
		print('Ukalman')
		n = 1000
		x_pure = np.arange(n)*0.1
		x = x_pure + np.random.normal(scale=0.5, size=n)
		y = np.cos(x)
		# y = 2*x
		
		
		
		kf = AdditiveUnscentedKalmanFilter(n_dim_obs=1)
		
		measurements = [[y[i]] for i in range(int(n/2))]
		
		kf.initial_state_mean = np.array([0.])
		kf.initial_state_covariance = np.array([[0.1]])
		
		it_mean = kf.initial_state_mean
		it_cov = kf.initial_state_covariance
		
		print(kf.initial_state_mean)
		print(kf.initial_state_covariance)
		# exit()
		h = []
		for element in measurements:
			it_z = kf.filter_update(filtered_state_mean=it_mean, filtered_state_covariance=it_cov, observation=element)
			it_mean = it_z[0]
			it_cov = it_z[1]
			h.append(it_mean)
		
		f = list(kf.sample(int(n/2), initial_state=it_mean)[0])	
		# f = h + list(np.array(f) + it_mean)
		f = h + f
		
		plt.plot(x_pure, y)
		plt.plot(x_pure, f, 'g')
		plt.show()
	
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
	# config['fs'] = float(config['fs'])
	# config['interval'] = int(config['interval'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


if __name__ == '__main__':
	main(argv)
