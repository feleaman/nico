#Function for read one TDMS file and return a selected channel like array
#User must provide group and channel name
#If they are not known, file.groups() and file.group_channels() methods could be used
#channel could be:	'Drehmoment', 'Körperschall', 'AE_Signal'

from nptdms import TdmsFile
from nptdms import TdmsObject
# from tkinter import filedialog
import scipy.io
import sys

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