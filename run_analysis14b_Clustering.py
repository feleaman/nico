import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys


# Mypaths = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180227_Messung_A\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long\\Last_720']

# Names = ['AE_Hist_A_20bin_Thr30', 'AE_Hist_G_20bin_Thr30', 'AE_Hist_H_20bin_Thr30', 'AE_Hist_I_20bin_Thr30', 'AE_Hist_J_20bin_Thr30', 'AE_Hist_K1_20bin_Thr30', 'AE_Hist_K2_20bin_Thr30', 'AE_Hist_L_20bin_Thr30', 'AE_Hist_M_20bin_Thr30']


# Mypaths = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Warm']

# Names = ['AE_Hist_I_20bin_Thr30_Long', 'AE_Hist_I_20bin_Thr30_Warm']



for mypath, name in zip(Mypaths, Names):
	os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_detection_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	# os.system('python Bursts_Clustering.py --channel AE_0 --fs 1.e6 --mode burst_detection_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)	

