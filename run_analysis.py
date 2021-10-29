import os



from os import listdir
from os.path import isfile, join
mypath = 'M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\last_ae2_strathisla'
# mypath = 'C:\\Felix\\Schraube_EMD'
Filepaths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


for filepath in Filepaths:
	try:
		os.system('python EmpMD.py --file ' + filepath + ' --channel AE2 --power2 22 --save ON --min_iter 0 --max_iter 50000 --s_number 2 --tolerance 0.2 --fs 1.e6')
	except:
		print('error')


	
# try:
	# os.system('python EmpMD.py --file M:\\Betriebsmessungen\\Messdaten_IEHK_M36_Schrauben\\Messdaten\\Probe_2_36-78\\Versuch\\AE\\ZMB_Schraube_AE_20170515_152951.mat --channel AE1 --power2 22 --save ON --min_iter 0 --max_iter 1000000 --s_number 2 --tolerance 0.2 --fs 1.e6')
# except:
	# print('error')