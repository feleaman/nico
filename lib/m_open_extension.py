#Function for read one TDMS file and return a selected channel like array
#User must provide group and channel name
#If they are not known, file.groups() and file.group_channels() methods could be used
#channel could be:	'Drehmoment', 'Körperschall', 'AE_Signal'

from nptdms import TdmsFile
from nptdms import TdmsObject
# from tkinter import filedialog
import scipy.io
import sys
import pickle
import numpy as np


def f_open_tdms(filename, channel):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	file = TdmsFile(filename)
	group_name = file.groups()	
	group_name = group_name[0]
	data = file.channel_data(group_name, channel)
	
	return data

def f_open_tdms_2(filename):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	tdms_file = TdmsFile(filename)
	group_names = tdms_file.groups()
	channel_object = tdms_file.group_channels(group_names[0])	
	channel_name = channel_object[0].channel
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

def f_open_mat_2(filename):
	if filename == 'Input':
		filename = filedialog.askopenfilename()

	file = scipy.io.loadmat(filename)
	print(file)
	# print(file['AE_Signal'])
	# group_name = file.groups()
	# group_name = group_name[0]
	try:
		data = file['AE_Signal']
		channel = 'AE_Signal'
	except:
		pass
	
	try:
		data = file['Koerperschall']
		channel = 'Koerperschall'
	except:
		pass
	
	try:
		data = file['Drehmoment']
		channel = 'Drehmoment'
	except:
		pass
	
	return data, channel

def f_open_mat(filename, channel):
	file = scipy.io.loadmat(filename)
	data = file[channel]
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
		x = f_open_mat(filename, channel)
		x = np.ndarray.flatten(x)
	elif extension == 'tdm': #tdms
		x = f_open_tdms(filename, channel)
	elif extension == 'txt':
		x = np.loadtxt(filename)
	else:
		print('Error extention')
		sys.exit()
	return x