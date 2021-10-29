import os
from os import listdir
from os.path import join, isdir, basename, dirname, isfile
import sys



Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Bochum\\B3']
Names = ['B3_Tri_9rms']

# Mypaths = ['C:\\Felix\\29_THESIS\\MODEL_A\\Chapter4_WT_Gearboxes\\04_Data\\Schottland\\S1']
# Names = ['S1_Tri_9rms']

Channels = ['AE_1', 'AE_2', 'AE_3']
count = 0
for mypath, name in zip(Mypaths, Names):

	for channel in Channels:
		# os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode mode5 --thr_value 15 --thr_mode factor_rms --name ' + channel + '_' + name + '_' + str(count) + ' --side_points 0 --harm_fc 3 --plot OFF --range 0 11 --db_out 43 --unit mv --time_reg 11')
		
		os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel ' + channel + ' --fs 1.e6 --mode mode5 --thr_value 9 --thr_mode factor_rms --name ' + channel + '_' + name + '_' + str(count) + ' --side_points 0 --harm_fc 3 --plot OFF --range 0 11 --db_out 43 --unit mv --time_reg 11')
	

	
	
	count += 1


sys.exit()






Mypaths1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_H\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_I\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_J\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Warm']

# Mypaths1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Warm']

# Mypaths1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Warm']
# Mypaths1 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Warm', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Warm']


Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_H\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_I\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_J\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long']

# Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long']

# Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181031_Messung_P\\Long']


# Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Long']

# Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20181102_Messung_Q\\Long']

# Mypaths2 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long']


Mypaths3 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180314_Messung_G\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_H\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_I\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180317_Messung_J\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long1\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180320_Messung_K\\Long2\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180404_Messung_M\\Long\\Last_720', 'M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180511_Messung_N\\Long\\Last_720']

# Mypaths3 = ['M:\\Betriebsmessungen\\Getriebepruefstand_FLn\\20180322_Messung_L\\Long\\Last_720']




# Mypaths = ['C:\\Felix\\Measurement_Thesis']
Mypaths = Mypaths1 + Mypaths2 + Mypaths3
# Mypaths = Mypaths2 + Mypaths3


Names1 = ['G_0_Warm_Tri_Index17', 'H_0_Warm_Tri_Index17','I_0_Warm_Tri_Index17','J_0_Warm_Tri_Index17','K_0_Warm_Tri_Index17','L_0_Warm_Tri_Index17','M_0_Warm_Tri_Index17','N_0_Warm_Tri_Index17']

# Names1 = ['M_0_Warm_Tri_Index17']



Names2 = ['G_1_Long_Tri_Index17', 'H_1_Long_Tri_Index17','I_1_Long_Tri_Index17','J_1_Long_Tri_Index17','K1_1_Long_Tri_Index17', 'K2_1_Long_Tri_Index17', 'L_1_Long_Tri_Index17','M_1_Long_Tri_Index17','N_1_Long_Tri_Index17']

# Names2 = ['L_1_Long_Tri_Index17','M_1_Long_Tri_Index17']



Names3 = ['G_2_Last_Tri_Index17', 'H_2_Last_Tri_Index17','I_2_Last_Tri_Index17','J_2_Last_Tri_Index17','K1_2_Last_Tri_Index17', 'K2_2_Last_Tri_Index17', 'L_2_Last_Tri_Index17','M_2_Last_Tri_Index17','N_2_Last_Tri_Index17']

# Names3 = ['L_2_Last_Tri_Index17']



# Names = ['Intento9_3xInterval']
Names = Names1 + Names2 + Names3
# Names = Names2 + Names3




count = 0
for mypath, name in zip(Mypaths, Names):
	# os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	count += 1
	os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel AE_1 --fs 1.e6 --mode mode5 --thr_value 12 --thr_mode factor_rms --name ' + name + ' --side_points 0 --harm_fc 3 --plot ON --range 0 11 --db_out 43 --unit mv')
	

	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
# Names1 = ['N_0_Warm_Tri_Index10', 'Q_0_Warm_Tri_Index10_corr']
# Names2 = ['N_1_Long_Tri_Index10', 'Q_1_Long_Tri_Index10_corr']
# Names3 = ['N_2_Last_Tri_Index10', 'Q_2_Last_Tri_Index10_corr']
# Names = Names1 + Names2 + Names3






















# count = 0
# for mypath, name in zip(Mypaths, Names):
	# # os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	# count += 1
	# os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode mode5 --thr_value 0.6 --thr_mode fixed_value --name ' + name + ' --side_points 1 --harm_fc 3 --plot OFF')

# Names1 = ['N_0_Warm_Tri_Index11', 'Q_0_Warm_Tri_Index11_corr']
# Names2 = ['N_1_Long_Tri_Index11', 'Q_1_Long_Tri_Index11_corr']
# Names3 = ['N_2_Last_Tri_Index11', 'Q_2_Last_Tri_Index11_corr']
# Names = Names1 + Names2 + Names3

# count = 0
# for mypath, name in zip(Mypaths, Names):
	# # os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	# count += 1
	# os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode mode5 --thr_value 0.6 --thr_mode fixed_value --name ' + name + ' --side_points 0 --harm_fc 1 --plot OFF')

# Names1 = ['N_0_Warm_Tri_Index12', 'Q_0_Warm_Tri_Index12_corr']
# Names2 = ['N_1_Long_Tri_Index12', 'Q_1_Long_Tri_Index12_corr']
# Names3 = ['N_2_Last_Tri_Index12', 'Q_2_Last_Tri_Index12_corr']
# Names = Names1 + Names2 + Names3

# count = 0
# for mypath, name in zip(Mypaths, Names):
	# # os.system('python Bursts_Clustering.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode burst_features_multi_intervals_3 --save ON --plot OFF --highpass 20.e3 --thr_value 30 --name ' + name)
	# count += 1
	# os.system('python Tri_Analysis.py --mypath ' + mypath + ' --channel AE_0 --fs 1.e6 --mode mode5 --thr_value 0.6 --thr_mode fixed_value --name ' + name + ' --side_points 0 --harm_fc 3 --plot OFF')