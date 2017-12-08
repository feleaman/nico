import os
import sys
# import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from m_open_extension import *
from m_fft import *

from openpyxl import load_workbook

import matplotlib
import cmath
from scipy import signal
from Burst_Detection import burst_detector

Inputs = ['mode', 'fs', 'channel', 'method', 'plot']

InputsOpt_Defaults = {'demod_filter':['lowpass', 5000., 3], 'demod_prefilter':['highpass', 80.e3, 3], 'diff':1, 'thr_value':0.0004, 'processing':'butter_demod', 'thr_mode':'fixed_value', 'demod_dc':'without_dc', 'demod_rect':'only_positives', 'clf_check':'OFF', 'data_norm':'per_rms', 'denois':'OFF', 'window_time':0.001}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	print('Selected configuration: ')
	print(config)
	option = input('Continue: y / n: ')
	if option == 'n':
		sys.exit()
	
	if config['mode'] == '11':
		sensor_angles = [78., 328., 188.]
		mydict = {}
		print('Runing Localization Analysis: 1 Source, 1 Repetition')
		count = 1
		print('++++++++Select Signal ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepath = filedialog.askopenfilename()			
		root.destroy()
		if config['channel'] == 'all':
			Channels = ['AE_1', 'AE_2', 'AE_3']		

		Data = [load_signal(Filepath, channel=channel) for channel in Channels]
		
		
		angle_value = calculate_angle(Data[0], Data[1], Data[2], sensor_angles, config)
		if config['plot'] == 'ON':
			fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
			t = [i/config['fs'] for i in range(len(Data[0]))]
			ax[0].plot(t, Data[0])
			ax[1].plot(t, Data[1])
			ax[2].plot(t, Data[2])
			
			plt.plot()
			plt.show()

		print('Estimated Angle: ', angle_value)
	
	elif config['mode'] == '1n':

		mydict = {}
		print('Runing Localization Analysis: 1 Source, N Repetitions')
		
		count = 1
		angle_values = []
		print('++++++++Select Signals ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		if config['channel'] == 'all':
			Channels = ['0', '1', '2']

		for filepath in Filepaths:
			Data = [load_signal(filepath, channel=channel) for channel in Channels]
			angle_values.append(calculate_angle(Data[0], Data[1], Data[2], config))


		
		mydict['Source ' + str(count)] = angle_values
		
		row_names = ['Angle_Rep_' + str(count) for count in range(len(Filepaths))]
		# row_names = ['Angle_Rep_1', 'Angle_Rep_2', 'Angle_Rep_3', 'Angle_Rep_4', 'Angle_Rep_5']
		DataFr = pd.DataFrame(data=mydict, index=row_names)
	
		writer = pd.ExcelWriter('to_use.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Angles')	
		print('Result in Excel table')
		
	else:
		print('unknown mode')
		sys.exit()

		
		

		
		
	return


def compass(u, v, arrowprops=None):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    """

    # angles, radii = cart2pol(u, v)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax

def cart2pol(x, y):
    """Convert from Cartesian to polar coordinates.

    Example
    -------
    >>> theta, radius = pol2cart(x, y)
    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius

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
	# config['value'] = float(config['value'])
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


def calculate_angle(signal1, signal2, signal3, sensor_angles, config):
		
	# config['clf_check'] = 'OFF'
	# config['method'] = 'EDG'
	# # config['rms_change'] = 8.2
	# config['rms_change'] = 4.2
	# config['feat_norm'] = 'standard'
	# config['features'] = 'DataSorted'
	# config['data_norm'] = 'per_rms'
	# config['denois'] = 'OFF'
	# config['processing'] = 'butter_demod'
	# config['diff'] = 1
	# config['window_time'] = 0.001
	# config['overlap'] = 0
	# config['window_delay'] = 0
	# config['EMD'] = 'OFF'
	# config['NN_model'] = 'C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171018_180003.pkl'
	# config['demod_filter'] = ['lowpass', 2000., 3]
	# config['demod_prefilter'] = ['highpass', 70.e3, 3]
	# # config['demod_prefilter'] = 'OFF'
	# config['demod_rect'] = None
	# config['demod_dc'] = None
	# config['class2'] = 0
	# config['classes'] = '3n_2isclass'
	# config['thr_mode'] = 'fixed_value'
	# config['thr_value'] = 0.0004
	
	
	
	
	x1_1, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal1, config, count=None)
	# x1 = x1[10000:]
	t = [i/config['fs'] for i in range(len(x1_1))]
	# plt.plot(x1)
	# plt.show()
	# t1 = t_burst_corr1[0]
	t1 = (np.argmax(x1_1[5000:])+5000)/config['fs']
	x1_max = np.max(x1_1[5000:])
	
	x1_2, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal2, config, count=None)
	# x1 = x1[10000:]
	# plt.plot(x1)
	# plt.show()
	# t2 = t_burst_corr1[0]
	t2 = (np.argmax(x1_2[5000:])+5000)/config['fs']
	x2_max = np.max(x1_2[5000:])
	
	x1_3, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal3, config, count=None)
	# x1 = x1[10000:]
	# plt.plot(x1)
	# plt.show()
	# t3 = t_burst_corr1[0]
	t3 = (np.argmax(x1_3[5000:])+5000)/config['fs']
	x3_max = np.max(x1_3[5000:])
	
	print(t1, t2, t3)
	
	if x1_max > x3_max and x3_max > x2_max:
		print('no nothing')
		print('sensor closest: 1, sensor furthest: 2')
		
		
	elif x1_max > x3_max and x2_max > x3_max:
		t2_temp = t2
		t2 = t3
		t3 = t2_temp
		print('sensor closest: 1, sensor furthest: 3')
	elif x3_max > x1_max and x1_max > x2_max:
		t1_temp = t1
		t1 = t3
		t3 = t1_temp
		print('sensor closest: 3, sensor furthest: 2')
	elif x3_max > x1_max and x2_max > x1_max:		
		t1_temp = t1
		t2_temp = t2		
		t1 = t3		
		t2 = t1_temp		
		t3 = t2_temp
		print('sensor closest: 3, sensor furthest: 1')		
	elif x2_max > x1_max and x3_max > x1_max:		
		t1_temp = t1		
		t1 = t2
		t2 = t1_temp
		print('sensor closest: 2, sensor furthest: 1')	
	elif x2_max > x1_max and x1_max > x3_max:		
		t1_temp = t1		
		t1 = t2
		t2 = t3
		t3 = t1_temp
		print('sensor closest: 2, sensor furthest: 3')	
	else:
		print('warning times')
	
	
	

	
	
	fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
	t = [i/config['fs'] for i in range(len(x1_1))]
	ax[0].plot(t, x1_1)
	ax[1].plot(t, x1_2)
	ax[2].plot(t, x1_3)
	
	plt.show()

	# t1 = np.argmax(Data[0])/config['fs']
	# t2 = np.argmax(Data[1])/config['fs']
	# t3 = np.argmax(Data[2])/config['fs']
	

	
	t12 = t2 - t1
	t13 = t3 - t1
	
	
	# theta1 = 57
	# theta2 = 57 + 240
	# theta3 = 57 + 120
	# theta12 = 120
	
	theta1 = sensor_angles[0]
	theta2 = sensor_angles[1]
	theta3 = sensor_angles[2]
	theta12 = theta1 + np.absolute(360. - theta2)
	
	theta13 = theta3 - theta1
	theta01 = (theta13*t12/theta12 - t13)*theta12/(2*t12)
	theta0 = theta01 + theta1
	# print('Time of Max in Sensor 1: ', t1)
	# print('Time of Max in Sensor 2: ', t2)
	# print('Time of Max in Sensor 3: ', t3)
	# print('dt 1-2: ', t12)
	# print('dt 1-3: ', t13)
	# print('Theta 1: ', theta1)
	# print('Theta 2: ', theta2)
	# print('Theta 3: ', theta3)
	# print('Theta 1-3: ', theta13)
	# print('Theta 1-2: ', theta12)
	# print('Theta 0-1: ', theta01)
	# print('Theta 0: ', theta0)
	return theta0



if __name__ == '__main__':
	main(sys.argv)
