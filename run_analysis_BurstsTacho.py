import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys

# Mypaths = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long\\Last_720']

Mypaths = [ 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long']

Names = ['1_Long_MessP']

# Names = ['1_Long_MessI']

# Names = ['0_Warm_MessH', '0_Warm_MessK1']

# Names = ['1_Long_MessK2']

# Names = ['2_Last_MessJ', '2_Last_MessK2', '2_Last_MessL']


for mypath, name in zip(Mypaths, Names):
	# os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	
	os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_per_tacho --save ON --plot OFF --thr_value 60 --name ' + name)
	

