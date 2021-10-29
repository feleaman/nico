# import os
from os import listdir
import matplotlib.pyplot as plt


from os.path import join, isdir, basename, dirname, isfile
import sys
from os import chdir
plt.rcParams['savefig.directory'] = chdir(dirname('C:'))
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

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import RobustScaler  


Inputs = ['mode']
InputsOpt_Defaults = {'feature':'RMS', 'name':'name', 'mypath':None, 'fs':1.e6, 'n_mov_avg':0, 'sheet':0, 'train':0.4, 'n_pre':0.5, 'm_post':0.25, 'alpha':1.e-2, 'tol':1.e-3, 'learning_rate_init':0.001, 'max_iter':500000, 'layers':[10], 'solver':'adam', 'rs':1, 'activation':'identity'}

from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'mode1':
	
		print('hola')
		from filterpy.monte_carlo import systematic_resample
		from numpy.linalg import norm
		from numpy.random import randn
		import scipy.stats
		from numpy.random import uniform 
		def predict(particles, u, std, dt=1.):
			N = len(particles)
			# update heading
			particles[:, 2] += u[0] + (randn(N) * std[0])
			particles[:, 2] %= 2 * np.pi

			# move in the (noisy) commanded direction
			dist = (u[1] * dt) + (randn(N) * std[1])
			particles[:, 0] += np.cos(particles[:, 2]) * dist
			particles[:, 1] += np.sin(particles[:, 2]) * dist
		
		def update(particles, weights, z, R, landmarks):
			weights.fill(1.)
			for i, landmark in enumerate(landmarks):
				distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
				weights *= scipy.stats.norm(distance, R).pdf(z[i])

			weights += 1.e-300      # avoid round-off to zero
			weights /= sum(weights) # normalize
		
		def neff(weights):
			return 1. / np.sum(np.square(weights))
		
		def resample_from_index(particles, weights, indexes):
			particles[:] = particles[indexes]
			weights[:] = weights[indexes]
			weights.fill (1.0 / len(weights))
		
		def estimate(particles, weights):

			pos = particles[:, 0:2]
			mean = np.average(pos, weights=weights, axis=0)
			var  = np.average((pos - mean)**2, weights=weights, axis=0)
			return mean, var
		
		def create_uniform_particles(x_range, y_range, hdg_range, N):
			particles = np.empty((N, 3))
			particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
			particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
			particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
			particles[:, 2] %= 2 * np.pi
			return particles

		def run_pf1(N, iters=18, sensor_std_err=.1, 
					do_plot=True, plot_particles=False,
					xlim=(0, 20), ylim=(0, 20),
					initial_x=None):
			landmarks = np.array([[-1, 2], [5, 10], [12,14], [18,21]])
			NL = len(landmarks)
			
			plt.figure()
		   
			# create particles and weights
			if initial_x is not None:
				particles = create_gaussian_particles(
					mean=initial_x, std=(5, 5, np.pi/4), N=N)
			else:
				particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
			weights = np.zeros(N)

			if plot_particles:
				alpha = .20
				if N > 5000:
					alpha *= np.sqrt(5000)/np.sqrt(N)           
				plt.scatter(particles[:, 0], particles[:, 1], 
							alpha=alpha, color='g')
			
			xs = []
			robot_pos = np.array([0., 0.])
			for x in range(iters):
				robot_pos += (1, 1)

				# distance from robot to each landmark
				zs = (norm(landmarks - robot_pos, axis=1) + 
					  (randn(NL) * sensor_std_err))

				# move diagonally forward to (x+1, x+1)
				predict(particles, u=(0.00, 1.414), std=(.2, .05))
				
				# incorporate measurements
				update(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)
				
				# resample if too few effective particles
				if neff(weights) < N/2:
					indexes = systematic_resample(weights)
					resample_from_index(particles, weights, indexes)

				mu, var = estimate(particles, weights)
				xs.append(mu)

				if plot_particles:
					plt.scatter(particles[:, 0], particles[:, 1], 
								color='k', marker=',', s=1)
				p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
								 color='k', s=180, lw=3)
				p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
			
			xs = np.array(xs)
			#plt.plot(xs[:, 0], xs[:, 1])
			plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
			plt.xlim(*xlim)
			plt.ylim(*ylim)
			print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
			plt.show()

		from numpy.random import seed
		seed(2) 
		run_pf1(N=5000, plot_particles=False)
	
	elif config['mode'] == 'mode2':
		import numpy as np
		import pylab as pl

		from pykalman import KalmanFilter

		rnd = np.random.RandomState(0)

		# generate a noisy sine wave to act as our fake observations
		n_timesteps = 100
		x = np.linspace(0, 3 * np.pi, n_timesteps)
		observations = 20 * (np.sin(x) + 0.5 * rnd.randn(n_timesteps))
		Pobservations = observations[0:int(len(observations)/50)]
		print(len(observations))
		print(len(Pobservations))
		# create a Kalman Filter by hinting at the size of the state and observation
		# space.  If you already have good guesses for the initial parameters, put them
		# in here.  The Kalman Filter will try to learn the values of all variables.
		kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
						  transition_covariance=0.01 * np.eye(2))

		# You can use the Kalman Filter immediately without fitting, but its estimates
		# may not be as good as if you fit first.
		kf = kf.em(Pobservations)
		states_pred = kf.smooth(observations)[0]
		# states_pred = kf.em(Pobservations).smooth(observations)[0]
		print('fitted model: {0}'.format(kf))

		# Plot lines for the observations without noise, the estimated position of the
		# target before fitting, and the estimated position after fitting.
		pl.figure(figsize=(16, 6))
		obs_scatter = pl.scatter(x, observations, marker='x', color='b',
								 label='observations')
		position_line = pl.plot(x, states_pred[:, 0],
								linestyle='-', marker='o', color='r',
								label='position est.')
		velocity_line = pl.plot(x, states_pred[:, 1],
								linestyle='-', marker='o', color='g',
								label='velocity est.')
		pl.legend(loc='lower right')
		pl.xlim(xmin=0, xmax=x.max())
		pl.xlabel('time')
		pl.show()

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
		if element == 'layers':
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

	
	# Variable conversion
	
	# Variable conversion
	if config['sheet'] == 'OFF':
		config['sheet'] = 0
	
	return config


	
if __name__ == '__main__':
	main(sys.argv)
