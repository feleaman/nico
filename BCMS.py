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
# import pandas as pd
from m_open_extension import *
from m_det_features import signal_rms
Inputs = ['mode', 'fs', 'channel']
InputsOpt_Defaults = {'n_batches':1}
from m_fft import mag_fft
from m_denois import *
import pandas as pd

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
			plt.plot(data)
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
		for element in os.listdir(mypath):
			if os.path.isdir(os.path.join(mypath, element)) == True:
				dir_list.append(os.path.join(mypath, element))		
		for dir in dir_list:
			print(dir)
			count = 0
			for data_name in Data_Names:
				data_path = os.path.join(dir, data_name)
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
			
			path_basename = os.path.basename(filepath)
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
			path_basename = os.path.basename(filepath)
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
			MAX = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
			save_pickle('rms_filt_batch_' + str(count) + '.pkl', RMS)
			save_pickle('max_filt_batch_' + str(count) + '.pkl', MAX)
			mydict = {}
			
			row_names = [os.path.basename(filepath) for filepath in Filepaths]
			mydict['RMS'] = RMS
			mydict['MAX'] = MAX
			
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
		Channels = ['AE_1', 'AE_2', 'AE_3', 'AE_4']
		
		for count in range(config['n_batches']):
			print('Select Batch ', count+7)
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()
			MASTER_FILEPATH.append(Filepaths)
			root.destroy()


		for count in range(config['n_batches']):
			Filepaths = MASTER_FILEPATH[count]
			ref_dbAE = 1.e-6
			amp_factor = 43.
			ini_count = 0
			print('Processing Batch ', count+ini_count)			
			
			for channel in Channels:
				Data = [load_signal(filepath, channel=channel) for filepath in Filepaths]
				for i in range(len(Data)):
					Data[i] = butter_highpass(x=Data[i], fs=config['fs'], freq=80.e3, order=3)
				# RMS = [signal_rms(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
				# MAX = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data]
				RMS = [signal_rms(signal) for signal in Data]
				MAX = [np.max(np.absolute(signal)) for signal in Data]
				save_pickle('rms_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', RMS)
				save_pickle('max_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', MAX)
				
				
				
				Data_dBAE = []
				for signal in Data:
					signal_dBAE = np.zeros(len(signal))
					for i in range(len(signal)):
						current_input_V = np.absolute(signal[i]/(10.**(amp_factor/20.)))
						# print(current_input_V)
						signal_dBAE[i] = 20*np.log10(current_input_V/ref_dbAE)

					Data_dBAE.append(signal_dBAE)
				
				# RMS_dBAE = [signal_rms(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data_dBAE]
				# MAX_dBAE = [np.max(butter_highpass(x=signal, fs=config['fs'], freq=80.e3, order=3)) for signal in Data_dBAE]
				RMS_dBAE = [signal_rms(signal) for signal in Data_dBAE]
				MAX_dBAE = [np.max(signal) for signal in Data_dBAE]
				save_pickle('rms_dBAE_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', RMS_dBAE)
				save_pickle('max_dBAE_filt_batch_' + str(count+ini_count) + '_channel_' + channel + '.pkl', MAX_dBAE)
				
				mydict = {}			
				row_names = [os.path.basename(filepath) for filepath in Filepaths]
				mydict['RMS'] = RMS
				mydict['MAX'] = MAX	
				mydict['RMS_dBAE'] = RMS_dBAE
				mydict['MAX_dBAE'] = MAX_dBAE			
				DataFr = pd.DataFrame(data=mydict, index=row_names)
				writer = pd.ExcelWriter('to_use_batch_' + str(count+ini_count) + '_channel_' + channel + '.xlsx')		
				DataFr.to_excel(writer, sheet_name='Sheet1')
				print('Result in Excel table')
			
			# mean_mag_fft = read_pickle('mean_5_fft.pkl')
			# corrcoefMAGFFT = [np.corrcoef(mag_fft(signal, config['fs'])[0], mean_mag_fft) for signal in Data]
			# save_pickle('fftcorrcoef_batch_' + str(count) + '.pkl', corrcoefMAGFFT)
			
			
			
			
			# plt.boxplot(RMS)
			# plt.show()
	
	
	elif config['mode'] == 'long_analysis_plot':
		# RMS_long = [[] for i in range(config['n_batches'])]
		
		root = Tk()
		root.withdraw()
		root.update()
		Filepaths = filedialog.askopenfilenames()			
		root.destroy()
		# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		RMS_long = [read_pickle(filepath) for filepath in Filepaths]
		# save_pickle('rms_batch.pkl', RMS)
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.boxplot(RMS_long)
		ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
		ax.set_xticklabels(['14:06', '14:16', '14:30', '15:00', '15:30', '16:00', '16:20', '17:00', '17:30', '18:00', '18:30', '19:00', '23:30', '00:03'])
		ax.set_xlabel('Time on 20171020')
		ax.set_ylabel('RMS Amplitude')
		ax.set_ylim(bottom=1.e-4, top=4.e-4)
		
		# plt.boxplot(RMS_long)
		plt.show()
	
	elif config['mode'] == 'new_long_analysis_plot':
		# RMS_long = [[] for i in range(config['n_batches'])]
		print('Select table with features: ')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()			
		root.destroy()
		# Data = [load_signal(filepath, channel=config['channel']) for filepath in Filepaths]
		# feature = read_pickle(filepath)
		
		# amp_factor = input('Input amplification factor dB: ')
		# amp_factor = float(amp_factor)
		
		table = pd.read_excel(filepath)		
		rows = table.axes[0].tolist()		
		max_V = table['MAX'].values
		rms_V = table['RMS'].values		
		
		times = [rows[i][25:31] for i in range(len(rows))]
		times = [time[0:2] + ":" + time[2:4] + ":" + time[4:6] for time in times]
		
		
		
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(max_V)
		divisions = 10
		ax.set_xticks( [i*divisions for i in range(int(len(times)/divisions))] + [len(times)-1])
		# ax.set_xticklabels(times)
		ax.set_xticklabels( [times[i*divisions] for i in range(int(len(times)/divisions))] + [times[len(times)-1]])
		ax.set_ylabel('RMS Amplitude (V)')
		
		for label in ax.get_xmajorticklabels():
			label.set_rotation(45)
			label.set_horizontalalignment("right")
		
		
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
