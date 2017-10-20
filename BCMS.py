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

Inputs = ['mode', 'fs']
InputsOpt_Defaults = {'value':1}

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
	config['value'] = float(config['value'])
	config['fs'] = float(config['fs'])
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


	
if __name__ == '__main__':
	main(sys.argv)
