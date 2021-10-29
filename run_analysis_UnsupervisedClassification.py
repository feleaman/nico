import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys


Mypaths1 = ['C:\\Felix\\29_THESIS\\Chapter_4\\04_Data\\Burst_Detector\\Index1\\A_1_Long_BuDe_Index1_AE_Burst_Features.xlsx']




Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180227_Messung_A\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Long']





Mypaths3 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180227_Messung_A\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180315_Messung_H\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_I\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180316_Messung_J\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Long\\Last_720']

Mypaths = Mypaths1 + Mypaths2 + Mypaths3


Names1 = ['G_0_Warm_BuDe_Index1', 'H_0_Warm_BuDe_Index1','I_0_Warm_BuDe_Index1','J_0_Warm_BuDe_Index1','K_0_Warm_BuDe_Index1','L_0_Warm_BuDe_Index1','M_0_Warm_BuDe_Index1','N_0_Warm_BuDe_Index1', 'P_0_Warm_BuDe_Index1', 'Q_0_Warm_BuDe_Index1']


Names2 = ['A_1_Long_BuDe_Index1', 'G_1_Long_BuDe_Index1', 'H_1_Long_BuDe_Index1','I_1_Long_BuDe_Index1','J_1_Long_BuDe_Index1','K1_1_Long_BuDe_Index1', 'K2_1_Long_BuDe_Index1', 'L_1_Long_BuDe_Index1','M_1_Long_BuDe_Index1','N_1_Long_BuDe_Index1', 'P_0_Long_BuDe_Index1', 'Q_0_Long_BuDe_Index1']




Names3 = ['A_2_Last_BuDe_Index1', 'G_2_Last_BuDe_Index1', 'H_2_Last_BuDe_Index1','I_2_Last_BuDe_Index1','J_2_Last_BuDe_Index1','K1_2_Last_BuDe_Index1', 'K2_2_Last_BuDe_Index1', 'L_2_Last_BuDe_Index1','M_2_Last_BuDe_Index1','N_2_Last_BuDe_Index1','P_2_Last_BuDe_Index1','Q_2_Last_BuDe_Index1']

Names = Names1 + Names2 + Names3


# Mypaths = ['C:\\Felix\\Data\\QuickGearbox']
# Names = ['QuickGearbox']
# Mypaths = ['C:\\Felix\\Data\\QuickGearbox\\A', 'C:\\Felix\\Data\\QuickGearbox\\G', 'C:\\Felix\\Data\\QuickGearbox\\H', 'C:\\Felix\\Data\\QuickGearbox\\I', 'C:\\Felix\\Data\\QuickGearbox\\J', 'C:\\Felix\\Data\\QuickGearbox\\K', 'C:\\Felix\\Data\\QuickGearbox\\L', 'C:\\Felix\\Data\\QuickGearbox\\M', 'C:\\Felix\\Data\\QuickGearbox\\N', 'C:\\Felix\\Data\\QuickGearbox\\P', 'C:\\Felix\\Data\\QuickGearbox\\Q']
# Names = ['QuickGearbox_A', 'QuickGearbox_G', 'QuickGearbox_H', 'QuickGearbox_I', 'QuickGearbox_J', 'QuickGearbox_K', 'QuickGearbox_L', 'QuickGearbox_M', 'QuickGearbox_N', 'QuickGearbox_P', 'QuickGearbox_Q']


count = 0
for mypath, name in zip(Mypaths, Names):

	count += 1
	os.system('python M_Bursts_Detector.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --plot OFF --mode detector_thr --thr_value 3.6 --thr_mode factor_rms --name ' + name)
	

