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
from m_processing import *


from openpyxl import load_workbook

import matplotlib
import cmath
from scipy import signal
from Burst_Detection import burst_detector

plt.rcParams['savefig.dpi'] = 1600
plt.rcParams['savefig.format'] = 'jpeg'

Inputs = ['mode', 'fs', 'channel', 'method', 'plot']

InputsOpt_Defaults = {'demod_filter':['lowpass', 2000., 3], 'demod_prefilter':['highpass', 80.e3, 3], 'diff':1, 'thr_value':0.0004, 'processing':'butter_demod', 'thr_mode':'fixed_value', 'demod_dc':'without_dc', 'demod_rect':'only_positives', 'clf_check':'OFF', 'data_norm':'per_rms', 'denois':'OFF', 'window_time':0.001}


def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	print('Selected configuration: ')
	print(config)
	option = input('Continue: y / n: ')
	if option == 'n':
		sys.exit()
	
	if config['mode'] == '11':
		# sensor_angles = [78., 318., 198.] #eickhoff
		# sensor_angles = [57., 297., 177.] #cwd_1
		sensor_angles = [225., 315., 45., 135.] #cwd_2
		mydict = {}
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
			# Channels = ['AE_1', 'AE_2', 'AE_3'] #eickhoff
			# Channels = ['0', '1', '2'] #cwd_1
			Channels = ['AE_1', 'AE_2', 'AE_3', 'AE_4'] #cwd_2

		# Data = [load_signal_eickhoff(Filepath, channel=channel) for channel in Channels]
		Data = [1000*load_signal_2(Filepath, channel=channel) for channel in Channels]		 #cwd_1
		# Data = [1000*butter_highpass(x=load_signal_2(Filepath, channel=channel), freq=80.e3, fs=config['fs'], order=3) for channel in Channels]		 #cwd_1
		
		# Data = [butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)/281.8*1000 for signal in Data]
		
		# angle_value = calculate_angle(Data[0], Data[1], Data[2], sensor_angles, config)
		# angle_value = 1.999

		angle_value = calculate_angle_4(Data[0], Data[1], Data[2], Data[3], sensor_angles, config) #cwd2
		
		print('Estimated Angle: ', angle_value)
		# a = input('pause---')
		if config['plot'] == 'ON':
			fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
			t = [i/config['fs'] for i in range(len(Data[0]))]
			
			# signal1 = Data[0][int(0.723*config['fs']): int(0.730*config['fs'])]
			# signal2 = Data[1][int(0.723*config['fs']): int(0.730*config['fs'])]
			# signal3 = Data[2][int(0.723*config['fs']): int(0.730*config['fs'])]
			
			signal1 = Data[0]
			
			plt.plot(signal1)
			plt.show()
			
			signal2 = Data[1]
			signal3 = Data[2]
			
			signal1 = butter_lowpass(x=signal1, fs=config['fs'], freq=5000., order=3)
			signal2 = butter_lowpass(x=signal2, fs=config['fs'], freq=5000., order=3)
			signal3 = butter_lowpass(x=signal3, fs=config['fs'], freq=5000., order=3)
			
			plt.plot(signal1[int(0.723*config['fs']):], 'r')
			plt.show()
			
			signal1 = diff_signal(signal1, 1)
			signal2 = diff_signal(signal2, 1)
			signal3 = diff_signal(signal3, 1)
			
			plt.plot(signal1[int(0.723*config['fs']):], 'k')
			plt.show()
			
			t_op = [i/config['fs'] for i in range(len(signal1))]
			
			# ax[0].plot(t, Data[1], label='AE-2')
			# ax[0].legend(fontsize=10)
			# ax[1].plot(t, Data[2], label='AE-3', color='navy')
			# ax[1].legend(fontsize=10)
			# ax[2].plot(t, Data[3], label='AE-4', color='b')
			# ax[2].legend(fontsize=10)
			
			ax[0].tick_params(axis='both', labelsize=12)
			ax[1].tick_params(axis='both', labelsize=12)
			ax[2].tick_params(axis='both', labelsize=12)
			
			# ax[0].set_ylim(bottom=-190, top=190)
			# ax[1].set_ylim(bottom=-45, top=45)
			# ax[2].set_ylim(bottom=-75, top=75)
			
			ax[0].set_xlim(left=0.7235, right=0.7295)
			ax[1].set_xlim(left=0.7235, right=0.7295)
			ax[2].set_xlim(left=0.7235, right=0.7295)
			
			
			# ax[0].plot(t, Data[0], label='AE-2')
			# ax[0].legend(fontsize=10)
			# ax[1].plot(t, Data[1], label='AE-3', color='navy')
			# ax[1].legend(fontsize=10)
			# ax[2].plot(t, Data[2], label='AE-4', color='b')
			# ax[2].legend(fontsize=10)
			
			
			fig.text(0.04, 0.5, 'Amplitude [mV]', ha='center', va='center', rotation='vertical', fontsize=13)
			# ax[0].set_ylabel('Amplitude [mV]', fontsize=12)
			# ax[1].set_ylabel('Amplitude [mV]', fontsize=12)
			# ax[2].set_ylabel('Amplitude [mV]', fontsize=12)
			ax[2].set_xlabel('Time [s]', fontsize=13)
			
			# ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))
			# ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))
			# ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))
			
			ax[0].plot(t_op, signal1, label='AE-1')
			ax[0].legend(fontsize=11)
			ax[1].plot(t_op, signal2, label='AE-2', color='navy')
			ax[1].legend(fontsize=11)
			ax[2].plot(t_op, signal3, label='AE-3', color='b')
			ax[2].legend(fontsize=11)
			
			# plt.tight_layout()
			plt.show()
			
		# params = {'mathtext.default': 'regular' }          
		# plt.rcParams.update(params)
		
		# ax[0].set_title('Burst an Zahnflanke', fontsize=12)
		# ax[0].tick_params(axis='both', labelsize=11)
		# ax[1].tick_params(axis='both', labelsize=11)
		# ax[2].tick_params(axis='both', labelsize=11)
		
		
		# plt.show()
		
		print('Estimated Angle: ', angle_value)
	
	elif config['mode'] == 'plot_4':
		print('++++++++Select Signal ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepath = filedialog.askopenfilename()			
		root.destroy()
		if config['channel'] == 'all':
			Channels = ['AE_1', 'AE_2', 'AE_3', 'AE_4']

		# Data = [load_signal(Filepath, channel=channel) for channel in Channels]
		Data = [load_signal_2(Filepath, channel=channel) for channel in Channels]		
		# Data = [butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)/281.8*1000 for signal in Data]
		
		
		if config['plot'] == 'ON':
			fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
			t = [i/config['fs'] for i in range(len(Data[0]))]
			
			# signal1 = Data[0][0.741*config['fs']:]
			# signal2 = Data[1][0.741*config['fs']:]
			# signal3 = Data[2][0.741*config['fs']:]
			# t_op = [i/config['fs'] for i in range(len(signal1))]
			
			ax[0].plot(t, Data[0], label='AE-1')
			ax[0].legend(fontsize=10)
			ax[1].plot(t, Data[1], label='AE-2', color='navy')
			ax[1].legend(fontsize=10)
			ax[2].plot(t, Data[2], label='AE-3', color='b')
			ax[2].legend(fontsize=10)
			ax[3].plot(t, Data[3], label='AE-4', color='k')
			ax[3].legend(fontsize=10)
			
			# ax[0].plot(t_op, signal1, label='AE-1')
			# ax[0].legend(fontsize=10)
			# ax[1].plot(t_op, signal2, label='AE-2', color='navy')
			# ax[1].legend(fontsize=10)
			# ax[2].plot(t_op, signal3, label='AE-3', color='b')
			# ax[2].legend(fontsize=10)
			
			# plt.plot()
			
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax[1].set_ylabel(' A m p l i t u d e   [ m $V_{in}$ ] ', fontsize=12)
		ax[3].set_xlabel('Dauer [s]', fontsize=12)
		ax[0].set_title('Burst an Zahnflanke', fontsize=12)
		ax[0].tick_params(axis='both', labelsize=11)
		ax[1].tick_params(axis='both', labelsize=11)
		ax[2].tick_params(axis='both', labelsize=11)
		ax[3].tick_params(axis='both', labelsize=11)
		
		
		plt.show()
		
	
	elif config['mode'] == '1n':

		mydict = {}
		print('Runing Localization Analysis: 1 Source, N Repetitions')
		# sensor_angles = [78., 318., 198.] #eickhoff
		# sensor_angles = [57., 297., 177.] #cwd_1
		sensor_angles = [225., 315., 45., 135.] #cwd_2
		
		count = 1
		angle_values = []
		print('++++++++Select Signals ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		if config['channel'] == 'all':
			# Channels = ['AE_1', 'AE_2', 'AE_3'] #eickhoff
			# Channels = ['0', '1', '2'] #cwd_1
			Channels = ['AE_1', 'AE_2', 'AE_3', 'AE_4'] #cwd_2

		for filepath in Filepaths:
			Data = [load_signal_2(filepath, channel=channel) for channel in Channels]
			# angle_values.append(calculate_angle(Data[0], Data[1], Data[2], config))
			
			# angle_values.append(calculate_angle(Data[0], Data[1], Data[2], sensor_angles, config))
			angle_values.append(calculate_angle_4(Data[0], Data[1], Data[2], Data[3], sensor_angles, config)) #cwd2


		
		mydict['Source ' + str(count)] = angle_values
		
		row_names = ['Angle_Rep_' + str(count) for count in range(len(Filepaths))]
		# row_names = ['Angle_Rep_1', 'Angle_Rep_2', 'Angle_Rep_3', 'Angle_Rep_4', 'Angle_Rep_5']
		DataFr = pd.DataFrame(data=mydict, index=row_names)
	
		writer = pd.ExcelWriter('localization_to_use.xlsx')

		
		DataFr.to_excel(writer, sheet_name='Angles')	
		print('Result in Excel table')
	
	elif config['mode'] == 'plot_3':
		print('++++++++Select Signal ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepath = filedialog.askopenfilename()			
		root.destroy()
		if config['channel'] == 'all':
			Channels = ['0', '1', '2']
			# Channels = ['AE_1', 'AE_2', 'AE_3']

		# Data = [load_signal_eickhoff(Filepath, channel=channel)/281.8*1000. for channel in Channels]
		Data = [load_signal_2(Filepath, channel=channel) for channel in Channels]		
		# Data = [butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)/281.8*1000 for signal in Data]
		
		# plt.plot(Data[1][5000000:])
		# plt.show()
		# sys.exit()
		
		if config['plot'] == 'ON':
			fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
			t = [i/config['fs'] for i in range(len(Data[0]))]
			
			signal1 = Data[0][int(0.741*config['fs']):]
			signal2 = Data[1][int(0.741*config['fs']):]
			signal3 = Data[2][int(0.741*config['fs']):]
			t_op = [i/config['fs'] for i in range(len(signal1))]
			
			# ax[0].plot(t, Data[0], label='AE-1')
			# ax[0].legend(fontsize=10)
			# ax[1].plot(t, Data[1], label='AE-2', color='navy')
			# ax[1].legend(fontsize=10)
			# ax[2].plot(t, Data[2], label='AE-3', color='b')
			# ax[2].legend(fontsize=10)

			
			ax[0].plot(t_op, signal1, label='AE-1')
			ax[0].legend(fontsize=10)
			ax[1].plot(t_op, signal2, label='AE-2', color='navy')
			ax[1].legend(fontsize=10)
			ax[2].plot(t_op, signal3, label='AE-3', color='b')
			ax[2].legend(fontsize=10)
			
			# plt.plot()
			
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax[1].set_ylabel(' A m p l i t u d e   [ m $V_{in}$ ] ', fontsize=12)
		ax[2].set_xlabel('Dauer [s]', fontsize=12)
		# ax[0].set_title('Burst an Zahnflanke', fontsize=12)
		ax[0].set_title('Hsu-Nielsen-Tests', fontsize=12)
		ax[0].tick_params(axis='both', labelsize=11)
		ax[1].tick_params(axis='both', labelsize=11)
		ax[2].tick_params(axis='both', labelsize=11)
		
		
		plt.show()
	
	elif config['mode'] == 'plot_3_fft':
		print('++++++++Select Signal ')
		root = Tk()
		root.withdraw()
		root.update()
		Filepath = filedialog.askopenfilename()			
		root.destroy()
		if config['channel'] == 'all':
			Channels = ['0', '1', '2']

		Data = [load_signal_2(Filepath, channel=channel) for channel in Channels]		
		Spectrums = []
		for i in range(len(Data)):
			magX, f, df = mag_fft(Data[i], fs=config['fs'])
			Spectrums.append(magX)
		
		
		fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

		
		ax[0].plot(f/1000., Spectrums[0], label='AE-1', color='red')
		ax[0].legend(fontsize=10)
		ax[1].plot(f/1000., Spectrums[1], label='AE-2', color='firebrick')
		ax[1].legend(fontsize=10)
		ax[2].plot(f/1000., Spectrums[2], label='AE-3', color='darkred')
		ax[2].legend(fontsize=10)


		# ax[0].set_xlabel('Frequency [kHz]', fontsize=12)
		# ax[1].set_xlabel('Frequency [kHz]', fontsize=12)
		ax[2].set_xlabel('Frequency [kHz]', fontsize=12)
		
		fig.text(0.05, 0.5, 'Amplitude [mV]', ha='center', va='center', rotation='vertical', fontsize=12)
		
		ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		
		# ax[0].set_ylabel('Amplitude [mV]', fontsize=12)
		# ax[1].set_ylabel('Amplitude [mV]', fontsize=12)
		# ax[2].set_ylabel('Amplitude [mV]', fontsize=12)
		
		
		ax[0].tick_params(axis='both', labelsize=11)
		ax[1].tick_params(axis='both', labelsize=11)
		ax[2].tick_params(axis='both', labelsize=11)
		
		
		plt.show()
	
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

	arriv_1 = arrival_time(signal1, config)
	arriv_2 = arrival_time(signal2, config)
	arriv_3 = arrival_time(signal3, config)	
	if arriv_1 < arriv_3 and arriv_3 < arriv_2:
		mark = '1-2'		
	elif arriv_1 < arriv_2 and arriv_2 < arriv_3:
		mark = '1-3'		
	elif arriv_3 < arriv_1 and arriv_1 < arriv_2:
		mark = '3-2'		
	elif arriv_3 < arriv_2 and arriv_2 < arriv_1:
		mark = '3-1'		
	elif arriv_2 < arriv_3 and arriv_3 < arriv_1:
		mark = '2-1'	
	elif arriv_2 < arriv_1 and arriv_1 < arriv_3:
		mark = '2-3'	
	else:
		print('warning times')
		if arriv_1 <= arriv_3 and arriv_3 <= arriv_2:
			mark = '1-2'		
		elif arriv_1 <= arriv_2 and arriv_2 <= arriv_3:
			mark = '1-3'		
		elif arriv_3 <= arriv_1 and arriv_1 <= arriv_2:
			mark = '3-2'		
		elif arriv_3 <= arriv_2 and arriv_2 <= arriv_1:
			mark = '3-1'		
		elif arriv_2 <= arriv_3 and arriv_3 <= arriv_1:
			mark = '2-1'	
		elif arriv_2 <= arriv_1 and arriv_1 <= arriv_3:
			mark = '2-3'
		else:
			print('error times')
			sys.exit()
	
	

	angle1 = sensor_angles[0]
	angle2 = sensor_angles[1]
	angle3 = sensor_angles[2]
	
	
	x1_1, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal1, config, count=None)
	t = [i/config['fs'] for i in range(len(x1_1))]
	t1 = (np.argmax(x1_1[5000:])+5000)/config['fs']
	x1_max = np.max(x1_1[5000:])
	
	x1_2, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal2, config, count=None)
	t2 = (np.argmax(x1_2[5000:])+5000)/config['fs']
	x2_max = np.max(x1_2[5000:])
	
	x1_3, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal3, config, count=None)
	t3 = (np.argmax(x1_3[5000:])+5000)/config['fs']
	x3_max = np.max(x1_3[5000:])
	
	print(t1, t2, t3)
	
	flag = 'LEFT'
	
	

	
	if mark == '1-2':
		print('no nothing')
		print('sensor closest: 1, sensor furthest: 2')		
		
	elif mark == '1-3':
		t2_temp = t2
		t2 = t3
		t3 = t2_temp
		print('sensor closest: 1, sensor furthest: 3')
		print(t1, t2, t3)		
		angle2_temp = angle2
		angle2 = angle3
		angle3 = angle2_temp
		flag = 'LEFT'
		
	elif mark == '3-2':
		t1_temp = t1
		t1 = t3
		t3 = t1_temp
		print('sensor closest: 3, sensor furthest: 2')
		print(t1, t2, t3)
		
		angle1_temp = angle1
		angle1 = angle3
		angle3 = angle1_temp
		flag = 'LEFT'
		
	elif mark == '3-1':
		t1_temp = t1
		t2_temp = t2		
		t1 = t3		
		t2 = t1_temp		
		t3 = t2_temp
		print('sensor closest: 3, sensor furthest: 1')
		print(t1, t2, t3)		
		angle1_temp = angle1
		angle2_temp = angle2		
		angle1 = angle3		
		angle2 = angle1_temp		
		angle3 = angle2_temp		
		flag = 'RIGHT'
		
	elif mark == '2-1':
		t1_temp = t1		
		t1 = t2
		t2 = t1_temp
		print('sensor closest: 2, sensor furthest: 1')	
		print(t1, t2, t3)
		
		angle1_temp = angle1		
		angle1 = angle2
		angle2 = angle1_temp
		flag = 'LEFT'
		
	
	elif mark == '2-3':
		t1_temp = t1		
		t1 = t2
		t2 = t3
		t3 = t1_temp
		print('sensor closest: 2, sensor furthest: 3')	
		print(t1, t2, t3)
		flag = 'RIGHT'
		
		angle1_temp = angle1		
		angle1 = angle2
		angle2 = angle3
		angle3 = angle1_temp
		
		
	
	else:
		print('warning times')
	

	
	if config['plot'] == 'ON':
		fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
		t = [i/config['fs'] for i in range(len(x1_1))]
		ax[0].plot(t, x1_1, label='AE-2')
		ax[1].plot(t, x1_2, label='AE-3', color='navy')
		ax[2].plot(t, x1_3, label='AE-4', color='b')
		
		
		ax[0].legend(fontsize=11)
		ax[1].legend(fontsize=11)
		ax[2].legend(fontsize=11)
		
		fig.text(0.04, 0.5, 'Amp. Diff. Envelope [mV]', ha='center', va='center', rotation='vertical', fontsize=13)
		
		# ax[0].set_ylabel('Amplitude [mV]', fontsize=10)
		# ax[1].set_ylabel('Amplitude [mV]', fontsize=10)
		# ax[2].set_ylabel('Amplitude [mV]', fontsize=10)
		ax[2].set_xlabel('Time [s]', fontsize=13)
		
		ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
		
		ax[0].tick_params(axis='both', labelsize=12)
		ax[1].tick_params(axis='both', labelsize=12)
		ax[2].tick_params(axis='both', labelsize=12)
		
		# ax[0].set_xlim(left=0.7235, right=0.7295)
		# ax[1].set_xlim(left=0.7235, right=0.7295)
		# ax[2].set_xlim(left=0.7235, right=0.7295)
		
		ax[0].set_xlim(left=0.3645, right=0.3705)
		ax[1].set_xlim(left=0.3645, right=0.3705)
		ax[2].set_xlim(left=0.3645, right=0.3705)
		
		# plt.tight_layout()
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
	
	# theta1 = sensor_angles[0]
	# theta2 = sensor_angles[1]
	# theta3 = sensor_angles[2]
	
	theta1 = angle1
	theta2 = angle2
	theta3 = angle3
	
	# theta12 = theta1 + np.absolute(360. - theta2)	
	# theta13 = theta3 - theta1
	
	# theta12 = 120.
	# theta13 = 120.
	
	# theta12 = 90.
	# theta13 = 90.
	
	theta12 = np.min([np.absolute(theta1 - theta2), np.absolute(theta1 - theta3), np.absolute(theta3 - theta2)])
	theta13 = theta12
	
	theta01 = (theta13*t12/theta12 - t13)*theta12/(2*t12)
	print('theta01: ', theta01)
	
	
	
	
	if flag == 'LEFT':
		theta0 = theta01 + theta1
	elif flag == 'RIGHT':
		theta0 = theta1 - theta01
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

def calculate_angle_4(signal1, signal2, signal3, signal4, sensor_angles, config):

	
	myarray = [arrival_time(signal1, config), arrival_time(signal2, config), arrival_time(signal3, config), arrival_time(signal4, config)]
	if np.argmax(myarray) == 0:
		print('selected 2, 3, 4')
		signal1_corr = signal2
		signal2_corr = signal3
		signal3_corr = signal4
		sensor_angles_corr = [sensor_angles[1], sensor_angles[2], sensor_angles[3]]
		
	elif np.argmax(myarray) == 1:
		print('selected 1, 3, 4')
		signal1_corr = signal1
		signal2_corr = signal3
		signal3_corr = signal4
		sensor_angles_corr = [sensor_angles[0], sensor_angles[2], sensor_angles[3]]
	
	elif np.argmax(myarray) == 2:
		print('selected 1, 2, 4')
		signal1_corr = signal1
		signal2_corr = signal2
		signal3_corr = signal4
		sensor_angles_corr = [sensor_angles[0], sensor_angles[1], sensor_angles[3]]
	
	elif np.argmax(myarray) == 3:
		print('selected 1, 2, 3')
		signal1_corr = signal1
		signal2_corr = signal2
		signal3_corr = signal3
		sensor_angles_corr = [sensor_angles[0], sensor_angles[1], sensor_angles[2]]
	
	else:
		print('error 875')
		sys.exit()
	
	
	
	theta0 = calculate_angle(signal1_corr, signal2_corr, signal3_corr, sensor_angles_corr, config)
	
	
	
	return theta0

def arrival_time(signal, config):	
	
	x1_1, t_burst_corr1, amp_burst_corr1, Results, clf_1 = burst_detector(signal, config, count=None)
	t1 = (np.argmax(x1_1[5000:])+5000)/config['fs']	
	
	return t1

if __name__ == '__main__':
	main(sys.argv)
