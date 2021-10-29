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
Inputs = ['mode', 'fs', 'channel']
InputsOpt_Defaults = {'n_batches':1}
from m_fft import mag_fft
from m_denois import *
import pandas as pd
# import time
# print(time.time())
from datetime import datetime

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	
	if config['mode'] == 'txt_plot':
		flag = '1'
		while flag == '1':
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			data = np.loadtxt(filepath)
			
			# data = data[8000:]
			# data = median_filter(data=data, points=3, same_length=False)
			# data = median_filter(data=data, points=3, same_length=False)
			
			plt.plot(data)
			plt.title(config['channel'])
			# plt.ylim((-10000., 10000.))
			
			plt.show()
			
			flag = input('\nFlag: ')
			
			
	
	elif config['mode'] == 'txt_concatenates':
		flag = '1'
		concatenated_data = []
		while flag == '1':
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			data = np.loadtxt(filepath)
			data = data.tolist()
			concatenated_data = concatenated_data + data
			print('appended: ', filepath)		
			flag = input('\nFlag: ')
		np.savetxt('concatenated_file.txt', concatenated_data)
	
	elif config['mode'] == 'auto_txt_concatenates':
		mypath = 'C:\\Felix\\016_AE_Messung_CWD\\Daten\\0704_1400---0704_1550'
		dir_list = []
		Data_Names = ['rpm.txt', 'torque.txt']
		# Data_Names = ['temperature1.txt', 'temperature2.txt', 'temperature3.txt', 'temperature4.txt', 'temperature5.txt', 'temperature6.txt', 'temperature7.txt', 'temperature8.txt', 'temperature9.txt', 'temperature10.txt', 'temperature11.txt', 'temperature12.txt']
		Concatenated_Data = [[] for i in range(len(Data_Names))]
		for element in listdir(mypath):
			if isdir(join(mypath, element)) == True:
				dir_list.append(join(mypath, element))		
		for dir in dir_list:
			print(dir)
			count = 0
			for data_name in Data_Names:
				data_path = join(dir, data_name)
				data = np.loadtxt(data_path)
				data = np.float32(data)
				data = data.tolist()
				Concatenated_Data[count] = Concatenated_Data[count] + data
				count = count + 1
		count = 0
		for data_name in Data_Names:
			np.savetxt('concatenated_' + data_name, Concatenated_Data[count])
			count = count + 1
	
	elif config['mode'] == 'various_plot':
		flag = '1'
		while flag == '1':
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			data = np.loadtxt(filepath)
			data = data.tolist()
			time = [i/config['fs'] for i in range(len(data))]
			
			path_basename = basename(filepath)
			if path_basename.find('temperature') != -1:
				ind1 = path_basename.find('.') - 1
				ind2 = ind1 - 1
				if path_basename[ind2] == 'e':
					name_label = 'sensor' + path_basename[ind1]
				else:
					name_label = 'sensor' + path_basename[ind2] + path_basename[ind1]
				
				if path_basename[ind1] == '4' or path_basename[ind1] == '5' or path_basename[ind1] == '8' or path_basename[ind1] == '9':
					name_label = name_label + ' in'
				else:
					name_label = name_label + ' out'
					
				plt.ylabel('Temperature °C')
			else:
				if path_basename.find('rpm') != -1:
					name_label = 'rpm'
					plt.ylabel('RPM')
				elif path_basename.find('torque') != -1:
					name_label = 'torque'
					plt.ylabel('Torque kNm')
				else:
					name_label = None
				
			
			
			name_label 
			plt.plot(time, data, label=name_label)			
			flag = input('\nFlag: ')
		plt.xlabel('Time s')
		plt.legend(loc='best')
		plt.show()			
	
	elif config['mode'] == 'two_scale_plot':
		fig, ax1 = plt.subplots()
		ax1.set_xlabel('Time s')
		for k in range(2):
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			data = np.loadtxt(filepath)
			data = data.tolist()
			time = [i/config['fs'] for i in range(len(data))]				
			path_basename = basename(filepath)
			if path_basename.find('torque') != -1:
				ax1.plot(time, data, '-b')
				ax1.set_ylabel('Torque kNm', color='b')
				ax1.tick_params('y', colors='b')				
			elif path_basename.find('rpm') != -1:
				# data = data * 1000
				ax2 = ax1.twinx()
				ax2.plot(time, data, '-r')
				ax2.set_ylabel('RPM', color='r')
				ax2.tick_params('y', colors='r')				
			else:
				name_label = None

		fig.tight_layout()
		plt.show()
	
	elif config['mode'] == 'long_analysis_features':

		for count in range(config['n_batches']):
			print('Batch ', count)
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
			Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
			# RMS = [signal_rms(signal) for signal in Data]
			RMS = [signal_rms(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
			# MAX = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
			save_pickle('rms_filt_batch_' + str(count) + '.pkl', RMS)
			# save_pickle('max_filt_batch_' + str(count) + '.pkl', MAX)
			mydict = {}
			
			row_names = [basename(filepath) for filepath in Filepaths]
			mydict['RMS'] = RMS
			# mydict['MAX'] = MAX
			
			DataFr = pd.DataFrame(data=mydict, index=row_names)
			writer = pd.ExcelWriter('to_use_batch_' + str(count) + '.xlsx')

		
			DataFr.to_excel(writer, sheet_name='Sheet1')	
			print('Result in Excel table')
			
			# mean_mag_fft = read_pickle('mean_5_fft.pkl')
			# corrcoefMAGFFT = [np.corrcoef(mag_fft(signal, config['fs'])[0], mean_mag_fft) for signal in Data]
			# save_pickle('fftcorrcoef_batch_' + str(count) + '.pkl', corrcoefMAGFFT)
			
			
			
			
			# plt.boxplot(RMS)
			# plt.show()
	
	elif config['mode'] == 'new_long_analysis_features':
		MASTER_FILEPATH = []
		# Channels = ['AE_1', 'AE_2', 'AE_3', 'AE_4']
		Channels = ['AE_1', 'AE_2', 'AE_3']
		ref_dbAE = 1.e-6
		amp_factor = 43.
		ini_count = 22
		
		for count in range(config['n_batches']):
			print('Select Batch ', count+ini_count)
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			MASTER_FILEPATH.append(Filepaths)
			root.destroy()
		
		# import time
		from time import time
		start_time = time()

		for count in range(config['n_batches']):
			Filepaths = MASTER_FILEPATH[count]
			
			print('Processing Batch ', count+ini_count)			
			
			for channel in Channels:
				# Data = [load_signal(filepath, channel=channel) for filepath in Filepaths]
				Data = [butter_highpass(x=load_signal(filepath, channel=channel), fs=config['fs'], freq=80.e3, order=3) for filepath in Filepaths]
				# for i in range(len(Data)):
					# Data[i] = butter_highpass(x=Data[i], fs=config['fs'], freq=80.e3, order=3)
				# RMS = [signal_rms(signal) for signal in Data]
				MAX = [np.max(np.absolute(signal)) for signal in Data]
				# save_pickle('rms_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', RMS)
				# save_pickle('max_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', MAX)
				
				
				
				# Data_dBAE = []
				# for signal in Data:
					# signal_dBAE = np.zeros(len(signal))
					# for i in range(len(signal)):
						# current_input_V = np.absolute(signal[i]/(10.**(amp_factor/20.)))
						# signal_dBAE[i] = 20*np.log10(current_input_V/ref_dbAE)

					# Data_dBAE.append(signal_dBAE)
				# RMS_dBAE = [signal_rms(signal) for signal in Data_dBAE]
				# MAX_dBAE = [np.max(signal) for signal in Data_dBAE]
				
				# save_pickle('rms_dBAE_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', RMS_dBAE)
				# save_pickle('max_dBAE_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', MAX_dBAE)
				
				mydict = {}			
				row_names = [basename(filepath) for filepath in Filepaths]
				# mydict['RMS'] = RMS
				mydict['MAX'] = MAX	
				# mydict['RMS_dBAE'] = RMS_dBAE
				# mydict['MAX_dBAE'] = MAX_dBAE			
				DataFr = pd.DataFrame(data=mydict, index=row_names)
				# writer = pd.ExcelWriter('to_use_batch_' + str(count+ini_count) + '_channel_' + channel + '.xlsx')
				writer = pd.ExcelWriter('Batch_' + str(count+ini_count) + '_Features_OK_' + channel + '.xlsx')				
				DataFr.to_excel(writer, sheet_name='Sheet1')
				print('Result in Excel table')
			
			# mean_mag_fft = read_pickle('mean_5_fft.pkl')
			# corrcoefMAGFFT = [np.corrcoef(mag_fft(signal, config['fs'])[0], mean_mag_fft) for signal in Data]
			# save_pickle('fftcorrcoef_batch_' + str(count) + '.pkl', corrcoefMAGFFT)
			
		print("--- %s seconds ---" % (time() - start_time))
	
			
			
			# plt.boxplot(RMS)
			# plt.show()
	
	
	elif config['mode'] == 'long_analysis_plot':
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
		for i in range(2):
			# root = Tk()
			# root.withdraw()
			# root.update()
			# filepath = filedialog.askopenfilename()			
			# root.destroy()
			
			i = i + 22
			base = 'M:\Betriebsmessungen\WEA-Getriebe Eickhoff\Durchführung\Auswertung\Batch_' + str(i)
			filepath = join(base, 'Batch_' + str(i) + '_Features_OK_AE_1.xlsx')
			
			
			Filepaths.append(filepath)
		
		
		FEATURE = []
		for filepath in Filepaths:
			table = pd.read_excel(filepath)	
			FEATURE.append(table['MAX'].values)
		
		FEATURE = np.array(FEATURE)
		FEATURE = (FEATURE / 281.8) * 1000
		FEATURE = FEATURE.tolist()
		# save_pickle('rms_batch.pkl', RMS)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.boxplot(FEATURE)
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
		
		ax.set_title('AE_1', fontsize=12)
		
		# ax.set_xticklabels(['M=25%\nn=100%', 'M=50%\nn=100%', 'M=75%\nn=100%', 'M=100%\nn=25%', 'M=100%\nn=50%', 'M=100%\nn=75%', 'M=100%\nn=100%'])
		# ax.set_xticklabels(['Point A', 'Point B', 'Point C', 'Point D', 'Point E', 'Point F'])
		# ax.set_xticklabels(['Tooth 1', 'Tooth 2'])
		# ax.set_xticklabels(['Max High', 'Min High'])
		ax.set_xticklabels(['Point E', 'Point F'])
		# for label in ax.get_xmajorticklabels():
			# label.set_rotation(45)
			# label.set_horizontalalignment("right")

		# ax.set_xticklabels(['14:06', '14:16', '14:30', '15:00', '15:30', '16:00', '16:20', '17:00', '17:30', '18:00', '18:30', '19:00', '23:30', '00:03'])
		# ax.set_xlabel('Time on 20171020')
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		ax.set_ylabel('Max. Amplitude (m$V_{in}$)')
		# ax.set_ylim(bottom=1.e-4, top=4.e-4)
		
		# plt.boxplot(RMS_long)
		plt.show()
	
	elif config['mode'] == 'new_long_analysis_plot':
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
			feature = table['RMS'].values
			
			feature = (feature / 141.25) * 1000
			# feature = (feature / 281.8) * 1000
			
			# movil_avg = 10
			# for i in range(len(feature)):
				# if i >= movil_avg:
					# count = 0
					# for k in range(movil_avg):
						# count = count + feature[i-k]
					# feature[i] = count / movil_avg
						
					
					
			
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
		divisions = 10
		ax.set_xticks( [i*divisions for i in range(int(len(times)/divisions))] + [len(times)-1])
		# ax.set_xticklabels(times)
		ax.set_xticklabels( [times[i*divisions] for i in range(int(len(times)/divisions))] + [times[len(times)-1]])
		
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)		
		ax.set_ylabel('RMS Value (m$V_{in}$)')
		# ax.set_ylabel('Max. Amplitude (m$V_{in}$)')

		
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
		rpm = table['M'].values
		
		ax2 = ax.twinx()
		ax2.plot(rpm, 'om')
		# ax2.set_ylabel('RPM', color='r')
		# ax2.tick_params('y', colors='r')
		# ax2.set_ylabel('RPM', color='m')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		# ax2.set_ylabel('$RPM_{out}$', color='m')
		ax2.set_ylabel('$Torque_{out}$ (kNm)', color='m')

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
	
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	config['n_batches'] = int(config['n_batches'])
	# Variable conversion
	return config


	
if __name__ == '__main__':
	main(sys.argv)
