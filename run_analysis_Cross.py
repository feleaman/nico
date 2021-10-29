import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys



Mypaths = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180227_Messung_A\\HSU_Getriebe_H', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180313_Messung_F\\HSU_Getriebe_H', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\HSU_Getriebe_H', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\HSU_Getriebe_H',   'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_O\\HSU_Getriebe_H', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\HSU_Getriebe_H']



for mypath in Mypaths:	
	os.system('python Cross_Correlation_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode multi_bursts_segments --save ON')
	

