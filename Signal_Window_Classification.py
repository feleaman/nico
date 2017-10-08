# Signal_Window_Classification.py
# Last updated: 23.09.2017 by Felix Leaman
# Description:
# 

#++++++++++++++++++++++ IMPORT MODULES AND FUNCTIONS +++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import matplotlib.cm as cm
# from tkinter import filedialog
# from skimage import img_as_uint
from tkinter import Tk
from tkinter import Button
# import skimage.filters
import os.path
import sys
sys.path.insert(0, './lib') #to open user-defined functions

from m_open_extension import *
from m_fft import *
from m_demodulation import *
from m_denois import *
from m_det_features import *
from m_processing import *
from os.path import isfile, join
import pickle
import argparse
import datetime

plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes


#+++++++++++++++++++++++++++PARSER++++++++++++++++++++++++++++++++++++++++++
parser = argparse.ArgumentParser()
parser.add_argument('--channel', nargs='?')
# channel = 'AE_Signal'
# channel = 'Koerperschall'
# channel = 'Drehmoment'
parser.add_argument('--power2', nargs='?')
parser.add_argument('--showplot', nargs='?')
parser.add_argument('--type', nargs='?')
args = parser.parse_args()

if args.channel != None:
	channel = args.channel
else:
	print('Required: Channel')
	sys.exit()

if args.power2 != None:
	n_points = 2**int(args.power2)

if args.showplot != None:
	showplot = args.showplot




#+++++++++++++++++++++++++++FUNCTIONS++++++++++++++++++++++++++++++++++++++++++
def read_pickle(pickle_name):
	pik = open(pickle_name, 'rb')
	pickle_data = pickle.load(pik)
	return pickle_data

def find_extention(filename):
	point_index = filename.find('.')
	extention = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
	return extention


def save_pickle(pickle_name, pickle_data):
	pik = open(pickle_name, 'wb')
	pickle.dump(pickle_data, pik)
	pik.close()

def plot_windows():
	if config_analysis['Overlap'] == True:
		fig, ax = plt.subplots(nrows=1, ncols=2)
		fig.set_size_inches(14.5, 4.5)
		ax[0].axvline(x=t[count*window_advance]+window_time*0.25, color='red')
		ax[0].axvline(x=t[count*window_advance]+window_time*0.75, color='red')
		ax[0].plot(t[count*window_advance:window_points+window_advance*count], x1[count*window_advance:window_points+window_advance*count])
		
		ax[1].axvline(x=t[count*window_advance]+window_time*0.25, color='red')
		ax[1].axvline(x=t[count*window_advance]+window_time*0.75, color='red')
		ax[1].plot(t, x1)
		ax[1].set_xlim(left=t[count*window_advance]-window_time*7, right=t[count*window_advance]+window_time*7)
			
	else:
		fig, ax = plt.subplots(nrows=1, ncols=2)
		fig.set_size_inches(14.5, 4.5)
		ax[0].plot(t[count*window_points:(count+1)*window_points], x1[count*window_points:(count+1)*window_points])
		ax[0].axvline(x=t[count*window_points], color='magenta')
		ax[0].axvline(x=t[count*window_points]+window_time, color='magenta')
		
		ax[1].axvline(x=t[count*window_points], color='magenta')
		ax[1].axvline(x=t[count*window_points]+window_time, color='magenta')
		ax[1].plot(t, x1)
		ax[1].set_xlim(left=t[count*window_points]-window_time*7, right=t[count*window_points]+window_time*7)
	plt.title('Window N ' + str(count))
	plt.show(block=False)



def Start():
	global count
	global classification
	print("start")
	count = 0
	classification = []

	if config_analysis['Overlap'] == True:
		fig, ax = plt.subplots(nrows=1, ncols=2)
		fig.set_size_inches(14.5, 4.5)
		ax[0].axvline(x=t[0]+window_time*0.25, color='red')
		ax[0].axvline(x=t[0]+window_time*0.75, color='red')
		ax[0].plot(t[0:window_points], x1[0:window_points])

		ax[1].axvline(x=t[count*window_advance]+window_time*0.25, color='red')
		ax[1].axvline(x=t[count*window_advance]+window_time*0.75, color='red')
		ax[1].plot(t, x1)
		ax[1].set_xlim(left=t[count*window_advance]-window_time*7, right=t[count*window_advance]+window_time*7)
	
	else:
		fig, ax = plt.subplots(nrows=1, ncols=2)
		fig.set_size_inches(14.5, 4.5)
		ax[0].plot(t[0:window_points], x1[0:window_points])
		ax[0].axvline(x=t[0], color='magenta')
		ax[0].axvline(x=t[window_points], color='magenta')
		
		ax[1].axvline(x=t[0], color='magenta')
		ax[1].axvline(x=t[0]+window_time, color='magenta')
		ax[1].plot(t, x1)
		ax[1].set_xlim(left=t[0]-window_time*7, right=t[0]+window_time*7)
	plt.title('Window N ' + str(count))
	plt.show(block=False)

def Maybe():
	global count
	global classification
	print("Maybe")
	count = count + 1
	print('Window N ', count)
	classification.append(2)
	plt.close()
	if count < n_windows:
		plot_windows()
	else:
		print("Maybe")
		plt.close()
		Quit()

def Positive():
	global count
	global classification
	print("Positive")
	count = count + 1
	print('Window N ', count)
	classification.append(1)
	plt.close()
	if count < n_windows:
		plot_windows()
	else:
		print("Positive")
		plt.close()
		Quit()
	

def Negative():
	global count
	global classification
	print("Negative")
	count = count + 1
	print('Window N ', count)
	classification.append(0)
	plt.close()
	if count < n_windows:
		plot_windows()
	else:
		print("Negative")
		plt.close()
		Quit()

def ALL_Negative():
	global count
	global classification
	print("ALL Negative")
	count = count + 1
	print('Window N ', count)
	for k in range(n_windows):
		classification.append(0)
	plt.close()
	# print("ALL Negative")
	plt.close()
	Quit()

def Discard():
	global count
	global classification
	print("Discard")
	count = count + 1
	print('Window N ', count)
	classification.append(3)
	plt.close()
	if count < n_windows:
		plot_windows()
	else:
		print("Discard")
		plt.close()
		Quit()

def Quit():
	global count
	global classification
	print("quit")
	plt.close()
	root.destroy()

def Save():
	global count
	global classification
	print("Save")
	stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	np.savetxt('saveclassification_' + stamp + '_' + str(filename1[:-4]) + '.txt', classification)


def StartOld():
	global count
	global classification
	print("start with old file")
	rootsave = Tk()
	rootsave.withdraw()
	rootsave.update()
	filesave = filedialog.askopenfilename()
	rootsave.destroy()
	print(filesave)
	extention = find_extention(filesave)
	print(extention)
	if extention == 'txt':
		classification = np.loadtxt(filesave)
	elif extention == 'pkl':
		dog = read_pickle(filesave)
		classification = dog['classification']
	else:
		print('wrong extention')
	if filename1 != dog['filename']:
		print('wrong filenames starting with old')
	
		
	classification = [int(classification[i]) for i in range(len(classification))]
	count = len(classification)
	print(classification)
	print("Windows already analized: ", count)
	print("Windows to be analized: ", n_windows - count)
	
	
	
	if config_analysis['Overlap'] == True:
		fig, ax = plt.subplots(nrows=1, ncols=2)
		fig.set_size_inches(14.5, 4.5)
		ax[0].axvline(x=t[count*window_advance]+window_time*0.25, color='red')
		ax[0].axvline(x=t[count*window_advance]+window_time*0.75, color='red')
		ax[0].plot(t[count*window_advance:window_points+window_advance*count], x1[count*window_advance:window_points+window_advance*count])
		
		ax[1].axvline(x=t[count*window_advance]+window_time*0.25, color='red')
		ax[1].axvline(x=t[count*window_advance]+window_time*0.75, color='red')
		ax[1].plot(t, x1)
		ax[1].set_xlim(left=t[count*window_advance]-window_time*7, right=t[count*window_advance]+window_time*7)
			
	else:
		fig, ax = plt.subplots(nrows=1, ncols=2)
		fig.set_size_inches(14.5, 4.5)
		ax[0].plot(t[count*window_points:(count+1)*window_points], x1[count*window_points:(count+1)*window_points])
		ax[0].axvline(x=t[count*window_points], color='magenta')
		ax[0].axvline(x=t[count*window_points]+window_time, color='magenta')
		
		ax[1].axvline(x=t[count*window_points], color='magenta')
		ax[1].axvline(x=t[count*window_points]+window_time, color='magenta')
		ax[1].plot(t, x1)
		ax[1].set_xlim(left=t[count*window_points]-window_time*7, right=t[count*window_points]+window_time*7)
	plt.title('Window N ' + str(count))
	plt.show(block=False)
#+++++++++++++++++++++++++++CONFIG++++++++++++++++++++++++++++++++++++++++++
# #TEST
# # mypath = 'C:/Felix/Data/CNs_Getriebe/Paper_Bursts/n1500_M80/'
# # filename1 = join(mypath, 'V3_9_n1500_M80_AE_Signal_20160928_154159.mat')
# # filename1 = join(mypath, 'V3_9_n1500_M80_AE_Signal_20160506_152625.mat')

# #TRAIN
# mypath = 'C:/Felix/Data/CNs_Getriebe/Paper_Bursts/n1500_M80/train'
# # filename1 = join(mypath, 'V1_9_n1500_M80_AE_Signal_20160928_144737.mat')
# filename1 = join(mypath, 'V1_9_n1500_M80_AE_Signal_20160506_142422.mat')

from tkinter import filedialog


rootfile = Tk()
rootfile.withdraw()
rootfile.update()
filename1 = filedialog.askopenfilename()
rootfile.destroy()
print(filename1)












config_analysis = {'WindowTime':0.001, 'Overlap':False, 'WindowAdvance':0, 'savepik':True, 'power2':args.power2,
'channel':args.channel, 'start_in':0.0005}

config_filter = {'analysis':False, 'type':'median', 'mode':'bandpass', 'params':[[70.0e3, 350.0e3], 3]}###

# config_autocorr = {'analysis':False, 'type':'wiener', 'mode':'same'}

config_demod = {'analysis':False, 'mode':'butter', 'prefilter':['bandpass', [70.0e3, 170.0e3] , 3], 
'rectification':'absolute_value', 'dc_value':'without_dc', 'filter':['lowpass', 5000.0, 3], 'warming':False,
'warming_points':20000}
#When hilbert is selected, the other parameters are ignored

config_diff = {'analysis':False, 'length':1, 'same':True}



#++++++++++++++++++++++ DATA LOAD ++++++++++++++++++++++++++++++++++++++++++++++

point_index = filename1.find('.')
extension = filename1[point_index+1] + filename1[point_index+2] + filename1[point_index+3]

if extension == 'mat':
	x1 = f_open_mat(filename1, channel)
	x1 = np.ndarray.flatten(x1)


elif extension == 'tdm': #tdms
	x1 = f_open_tdms(filename1, channel)

elif extension == 'txt':
	x1 = np.loadtxt(filename1)
	
else:
	print('Error extension')
	sys.exit()
filename1 = os.path.basename(filename1) #changes from path to file

#++++++++++++++++++++++ SAMPLING +++++++++++++++++++++++++++++++++++++++++++++++++++++++
if channel == 'Koerperschall':
	fs = 1000.0
elif channel == 'Drehmoment':
	fs = 1000.0
elif channel == 'AE_Signal':
	fs = 1000000.0
else:
	print('Error fs assignment')

if args.power2 == None:
	n_points = 2**(max_2power(len(x1)))
x1 = x1[0:n_points]
x1raw = x1






dt = 1.0/fs
n_points = len(x1)
tr = n_points*dt
t = np.array([i*dt for i in range(n_points)])
traw = t

	

if config_analysis['start_in'] != 0:
	print('with start in')
	x1 = x1[int(config_analysis['start_in']*fs):]
	t = t[int(config_analysis['start_in']*fs):]



#++++++++++++++++++++++SIGNAL PROCESSING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#Filter
if config_filter['analysis'] == True:
	print('+++Filter:')
	if config_filter['type'] == 'bandpass':
		print('Bandpass')
		f_nyq = 0.5*fs
		order = config_filter['params'][1]
		freqs_bandpass = [config_filter['params'][0][0]/f_nyq, config_filter['params'][0][1]/f_nyq]
		b, a = signal.butter(order, freqs_bandpass, btype='bandpass')
		x1 = signal.filtfilt(b, a, x1)
	elif config_filter['type'] == 'median':
		print('Median')
		x1 = scipy.signal.medfilt(x1)

#Autocorrelation
# if config_autocorr['analysis'] == True:
	# print('+++Filter:')
	# if config_autocorr['type'] == 'definition':
		# x1 = np.correlate(x1, x1, mode=config_autocorr['mode'])
		# x2 = np.correlate(x2, x2, mode=config_autocorr['mode'])
	
	# elif config_autocorr['type'] == 'wiener':	
		# fftx1 = np.fft.fft(x1)
		# x1 = np.real(np.fft.ifft(fftx1*np.conjugate(fftx1)))
		
		# fftx2 = np.fft.fft(x2)
		# x2 = np.real(np.fft.ifft(fftx2*np.conjugate(fftx2)))


		
#Demodulation
if config_demod['analysis'] == True:
	print('+++Demodulation:')
	if config_demod['mode'] == 'hilbert':
		x1 = hilbert_demodulation(x1)
	elif config_demod['mode'] == 'butter':
		x1 = butter_demodulation(x=x1, fs=fs, filter=config_demod['filter'], prefilter=config_demod['prefilter'], 
		type_rect=config_demod['rectification'], dc_value=config_demod['dc_value'])
	else:
		print('Error assignment demodulation')


#Differentiation
if config_diff['analysis'] == True:
	print('+++Differentiation:')
	if config_diff['same'] == True:
		x1 = diff_signal_eq(x=x1, length_diff=config_diff['length'])
	elif config_diff['same'] == False:
		x1 = diff_signal(x=x1, length_diff=config_diff['length'])
	else:
		print('Error assignment diff')	


if (config_demod['analysis'] == True or config_filter['analysis'] == True):
	if (config_demod['warming'] == True and config_demod['mode'] == 'butter'):
		print('Warm Warning')
		warm = config_demod['warming_points']
		x1 = x1[warm:]
		t = t[warm:]
		warm = float(warm)


window_time = config_analysis['WindowTime']
window_points = int(window_time*fs)
window_advance = int(window_points*config_analysis['WindowAdvance'])
if config_analysis['Overlap'] == True:
	n_windows = int((n_points - window_points)/window_advance) + 1
else:
	n_windows = int(n_points/window_points)

print('Number of windows to analyze: ', n_windows)



#++++++++++++++++++++++ GUI +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
global root
root = Tk()
root.title('canv')
root.config(bg="white")
w,h=root.maxsize()
print("%dx%d"%(w,h))
root.geometry("%dx%d"%(1450,150))

# Button Positive
botonn=Button(root,text="Positive",font=('arial', 22, 'bold'),command=Positive)
botonn['bg']='black'
botonn['fg']='orange'
botonn.place(x=20,y=40)

# Button Negative
botonn=Button(root,text="Negative",font=('arial', 22, 'bold'),command=Negative)
botonn['bg']='dark blue'
botonn['fg']='orange'
botonn.place(x=180,y=40)

# Button Maybe
botonn=Button(root,text="Maybe",font=('arial', 22, 'bold'),command=Maybe)
botonn['bg']='orange'
botonn['fg']='black'
botonn.place(x=350,y=40)

# Button Discard
botonn=Button(root,text="Discard",font=('arial', 22, 'bold'),command=Discard)
botonn['bg']='red'
botonn['fg']='black'
botonn.place(x=490,y=40)

# Button Start
botonn=Button(root,text="Start",font=('arial', 22, 'bold'),command=Start)
botonn['bg']='white'
botonn['fg']='black'
botonn.place(x=660,y=40)

# Button Quit
botonn=Button(root,text="Quit",font=('arial', 22, 'bold'),command=Quit)
botonn['bg']='black'
botonn['fg']='white'
botonn.place(x=760,y=40)

# Button Save
botonn=Button(root,text="Save",font=('arial', 22, 'bold'),command=Save)
botonn['bg']='blue'
botonn['fg']='magenta'
botonn.place(x=860,y=40)


# Button StartOld
botonn=Button(root,text="StartOld",font=('arial', 22, 'bold'),command=StartOld)
botonn['bg']='magenta'
botonn['fg']='black'
botonn.place(x=960,y=40)

# Button ALL Negative and Quit
botonn=Button(root,text="ALLNegative",font=('arial', 22, 'bold'),command=ALL_Negative)
botonn['bg']='yellow'
botonn['fg']='red'
botonn.place(x=1160,y=40)

root.mainloop()

#++++++++++++++++++++++ PRINT INFO AND SIGNAL +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
info = {'filename':filename1, 'n_windows':n_windows, 'classification':classification, 'config_analysis':config_analysis, 
'config_filter':config_filter, 'config_demod':config_demod, 'config_diff':config_diff}

print(info)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_xticks(np.arange(0, n_points*dt, window_points*dt))
ax.plot(t, x1)
plt.grid()
plt.show()


stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if config_analysis['savepik'] == True:
	save_pickle('classification_' + stamp + '_' + str(filename1[:-4]) + '.pkl', info)








