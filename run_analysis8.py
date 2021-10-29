import os

import sys

from os import listdir
from os.path import isfile, join


# mypath = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE'
# mypath2 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE1'
# mypath3 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE1\\h2'
# mypath4 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE1\\h3'
# mypath5 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE1\\h4'
# mypath6 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE1\\h5'
# mypath7 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE1\\h6'
# Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
# Filepaths2 = [join(mypath2, f) for f in listdir(mypath2) if isfile(join(mypath2, f))]
# Filepaths3 = [join(mypath3, f) for f in listdir(mypath3) if isfile(join(mypath3, f))]
# Filepaths4 = [join(mypath4, f) for f in listdir(mypath4) if isfile(join(mypath4, f))]
# Filepaths5 = [join(mypath5, f) for f in listdir(mypath5) if isfile(join(mypath5, f))]
# Filepaths6 = [join(mypath6, f) for f in listdir(mypath6) if isfile(join(mypath6, f))]
# Filepaths7 = [join(mypath7, f) for f in listdir(mypath7) if isfile(join(mypath7, f))]




mypath = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\select'
mypath2 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE2\\select'
mypath3 = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\EMD\\AE2\\h2_sd_002\\select'


Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
Filepaths2 = [join(mypath2, f) for f in listdir(mypath2) if isfile(join(mypath2, f))]
Filepaths3 = [join(mypath3, f) for f in listdir(mypath3) if isfile(join(mypath3, f))]



# for filepath, filepath2 in zip(Filepaths, Filepaths2):
	# try:
		# os.system('python EmpMD.py --file ' + filepath + ' --channel AE2 --power2 22 --save ON --min_iter 0 --max_iter 50000 --s_number 2 --tolerance 0.02 --fs 1.e6 --file_h1 ' + filepath2)
	# except:
		# print('error')

for filepath, filepath2, filepath3 in zip(Filepaths, Filepaths2, Filepaths3):
	try:
		os.system('python EmpMD.py --file ' + filepath + ' --channel AE2 --power2 22 --save ON --min_iter 0 --max_iter 50000 --s_number 2 --tolerance 0.02 --fs 1.e6 --file_h1 ' + filepath2 + ' --file_h2 ' + filepath3)
	except:
		print('error')


# for filepath, filepath2, filepath3, filepath4, filepath5, filepath6, filepath7 in zip(Filepaths, Filepaths2, Filepaths3, Filepaths4, Filepaths5, Filepaths6, Filepaths7):

	# try:
		# os.system('python EmpMD.py --file ' + filepath + ' --channel AE1 --power2 22 --save ON --min_iter 0 --max_iter 50000 --s_number 2 --tolerance 0.2 --fs 1.e6 --file_h1 ' + filepath2 + ' --file_h2 ' + filepath3 + ' --file_h3 ' + filepath4 + ' --file_h4 ' + filepath5 + ' --file_h5 ' + filepath6 + ' --file_h6 ' + filepath7)
	# except:
		# print('error')


	
