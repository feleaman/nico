#Function for read one TDMS file and return a selected channel like array
#User must provide group and channel name
#If they are not known, file.groups() and file.group_channels() methods could be used
#channel could be:	'Drehmoment', 'Körperschall', 'AE_Signal'

from nptdms import TdmsFile
# from nptdms import TdmsObject
from tkinter import filedialog
import scipy.io
import sys
import pickle
import numpy as np
import h5py
from os.path import basename

def f_open_tdms_old(filename, channel):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	file = TdmsFile(filename)
	
	group_name = file.groups()	
	group_name = group_name[0]

	# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	# print(group_name)
	# print(file)
	# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	data = file.channel_data(group_name, channel)
	
	return data

def f_open_tdms(filename, channel):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	file = TdmsFile.read(filename)
	all_groups = file.groups()
	group = all_groups[0]
	data_channel = group[channel]
	data = data_channel[:]
	
	return data

def f_open_tdms_2(filename):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	tdms_file = TdmsFile(filename)
	group_names = tdms_file.groups()
	# print('groups')
	# print(group_names)
	channel_object = tdms_file.group_channels(group_names[0])
	# print('channel')
	# print(channel_object)
	channel_name = channel_object[0].channel
	# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	# print(channel_name)
	# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	data = tdms_file.channel_data(group_names[0], channel_name)
	
	# channel_object = tdms_file('Thiele_Versuche', 'AE Signal')
	# group_name = file.groups()
	
	
	# print(group_name[0])
	# channel_name = file.group_channels(group_name[0])	
	# print(channel_name[0])
	# canal = file.object(group_name[0], 'AE Signal')
	
	# group_name = file.groups()	
	# group_name = group_name[0]
	# data = file.object('Thiele_Versuche', 'AE Signal')
	
	# data = file.channel_data(group_name[0], 'AE Signal')
	
	return data

def f_open_mat_2(filename, channel=None):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	file = scipy.io.loadmat(filename)

	# print(file['AE_Signal'])
	# group_name = file.groups()
	# group_name = group_name[0]
	
	try:
		data = file[channel]
		channel = 'AE_Signal'
		print('AE_Signal')
	except:
		pass
	
	
	
	try:
		data = file['AE_Signal']
		channel = 'AE_Signal'
		print('AE_Signal')
	except:
		pass
	
	try:
		data = file['Koerperschall']
		channel = 'Koerperschall'
		print('Koerperschall')
	except:
		pass
	
	try:
		data = file['Drehmoment']
		channel = 'Drehmoment'
		print('Drehmoment')
	except:
		pass
	
	try:
		data = file['a']
		channel = 'a'
		print('a')
	except:
		pass
	
	return data, channel

def f_open_mat(filename, channel):
	# try:
		# file = scipy.io.loadmat(filename)
		# data = file[channel]

	# except:
		# try:
			# print('warning 888')		
			# channel = int(channel)
			# arrays = {}
			# f = h5py.File(filename)
			# for k, v in f.items():
				# arrays[k] = np.array(v)
			# data = arrays['AE_y'][channel]
		# except:
			# print('warning 468')		
			# with h5py.File(filename, 'r') as f:
				# data = list(f['Ch0'])[0]
				
	# file = scipy.io.loadmat(filename)
	# data = file[channel]
	# -*- coding: cp852 -*-
	# -*- coding: iso-8859-15 -*-
	import h5py
	# print(filename)
	with h5py.File(filename, 'r') as f:
		print(list(f.keys()))
		print(channel)
		# print(list(f['Ch0']))
		data = list(f[channel])[0]
		# print(type(data))
		# print(data)

	return data

def f_open_mat_matrix(filename):
	# try:
		# file = scipy.io.loadmat(filename)
		# data = file[channel]

	# except:
		# try:
			# print('warning 888')		
			# channel = int(channel)
			# arrays = {}
			# f = h5py.File(filename)
			# for k, v in f.items():
				# arrays[k] = np.array(v)
			# data = arrays['AE_y'][channel]
		# except:
			# print('warning 468')		
			# with h5py.File(filename, 'r') as f:
				# data = list(f['Ch0'])[0]
				
	# file = scipy.io.loadmat(filename)
	# data = file[channel]
	# -*- coding: cp852 -*-
	# -*- coding: iso-8859-15 -*-
	import h5py
	# print(filename)
	with h5py.File(filename, 'r') as f:
		print(list(f.keys()))
		print(channel)
		# print(list(f['Ch0']))
		data = list(f[channel])[0]
		# print(type(data))
		# print(data)

	return data

def f_open_mat_bcms(filename, channel):
	try:
		file = scipy.io.loadmat(filename)
		data = file[channel]

	except:
		try:
			print('warning 888')		
			channel = int(channel)
			arrays = {}
			f = h5py.File(filename)
			for k, v in f.items():
				arrays[k] = np.array(v)
			data = arrays['AE_y'][channel]
		except:
			print('warning 468')		
			with h5py.File(filename, 'r') as f:
				data = list(f['Ch0'])[0]
				
	# # file = scipy.io.loadmat(filename)
	# # data = file[channel]
	# # -*- coding: cp852 -*-
	# # -*- coding: iso-8859-15 -*-
	# import h5py
	# # print(filename)
	# with h5py.File(filename, 'r') as f:
		# print(list(f.keys()))
		# print(channel)
		# # print(list(f['Ch0']))
		# data = list(f[channel])
		# # print(type(data))
		# # print(data)

	return data

def save_pickle(pickle_name, pickle_data):
	pik = open(pickle_name, 'wb')
	pickle.dump(pickle_data, pik)
	pik.close()

def read_pickle(pickle_name):
	pik = open(pickle_name, 'rb')
	pickle_data = pickle.load(pik)
	return pickle_data

def load_signal(filename, channel=None):
	point_index = filename.find('.')
	extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
	if extension == 'mat':
		# x = f_open_mat(filename, channel)
		x, channel = f_open_mat_2(filename, channel=channel)

		x = np.ndarray.flatten(x)
	elif extension == 'tdm': #tdms
		x = f_open_tdms(filename, channel)
		# x = f_open_tdms_2(filename)
	elif extension == 'txt':
		x = np.loadtxt(filename)
	elif extension == 'pkl':
		x = read_pickle(filename)
	else:
		print('Error extention')
		sys.exit()
	return x

def load_signal_varb(filename, channel=None):
	point_index = filename.find('.')
	extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
	if extension == 'mat':
		x = f_open_mat(filename, channel)
		# x, channel = f_open_mat_2(filename, channel=channel)

		x = np.ndarray.flatten(x)
	elif extension == 'tdm': #tdms
		x = f_open_tdms(filename, channel)
		# x = f_open_tdms_2(filename)
	elif extension == 'txt':
		x = np.loadtxt(filename)
	elif extension == 'pkl':
		x = read_pickle(filename)
	else:
		print('Error extention')
		sys.exit()
	return x

def load_signal_2(filename, channel=None):
	filename = basename(filename)
	point_index = filename.find('.')
	extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
	print(extension)
	if extension == 'mat':
		# x = f_open_mat_bcms(filename, channel)
		# x, channel = f_open_mat_2(filename, channel=channel)
		
		x, channel = f_open_mat_2(filename, channel=config['channel'])
		# x = f_open_mat(filename, channel)
		# x = np.ndarray.flatten(x)

		x = np.ndarray.flatten(x)
	elif extension == 'tdm': #tdms
		# x = f_open_tdms(filename, channel)
		x = f_open_tdms_2(filename)
	elif extension == 'txt':
		x = np.loadtxt(filename)
	
	elif extension == 'pkl':
		x = read_pickle(filename)
	
	else:
		print('Error extention')
		sys.exit()
	return x

def load_signal_eickhoff(filename, channel=None):
	point_index = filename.find('.')
	extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
	if extension == 'mat':
		x = f_open_mat(filename, channel)
		# x, channel = f_open_mat_2(filename, channel=channel)

		x = np.ndarray.flatten(x)
	elif extension == 'tdm': #tdms
		x = f_open_tdms(filename, channel)
		# x = f_open_tdms_2(filename)
	elif extension == 'txt':
		x = np.loadtxt(filename)
	else:
		print('Error extention')
		sys.exit()
	return x