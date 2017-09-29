import os
import sys
# os.system('')
# os.system('python EmpMD.py --path --file_x --channel --power2 --save --file_h1  --min_iter')

os.system('cd C:\\Felix\\Code\\nico')
# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --layers 25 --classifications C:/code/nico1/classification_20170923_061144_ok_v3_n1500_m80.pkl --files C:/code/nico1/ok_v3_n1500_m80.txt --features interval10_stats_nomean')

# #+++++++++++TRAIN
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features interval10_stats_nomean --method THR --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20170921_110533_.pkl --data_norm per_signal --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\OK\\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Analysis_Case_1500_80\\OK\\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --clf_check ON --class2 0 --thr_mode fixed_value --thr_value 7 --data_norm per_rms')


# Layers = ['100 20', '300', '15', '15 2', '100']
# 

# for layer in Layers:



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++MANUAL


# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features i10statsnmnsnk_lrstd --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1 --data_norm per_rms --alpha 1.e-1 --solver adam --layers 100 20 --processing demod_hilbert --eval_features OFF')


# +++++++++++VALID 1500 80
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features i10statsnmnsnk_lrstd --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20170929_163117.pkl --n_files 2 --save OFF --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\V3_9_n1500_M80_AE_Signal_20160506_152625.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 0 --thr_mode fixed_value --thr_value 7.5 --data_norm per_rms --processing demod_hilbert --plot OFF')	

a = input('press enter to continue . . . ')
#+++++++++++VALID 1000 80
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features i10statsnmnsnk_lrstd --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20170929_163117.pkl --n_files 2 --save OFF --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\V1_8_n1000_M80_AE_Signal_20160928_144217.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\V1_8_n1000_M80_AE_Signal_20160506_141822.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 0 --thr_mode fixed_value --thr_value 7.5 --data_norm per_rms --processing demod_hilbert --plot OFF')



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AUTO
# EMDs = ['OFF', 'ON']
# Processings = ['demod_hilbert', 'times_demod_hilbert', 'square']
# Denois = ['OFF', 'median']

# # Alphas = ['1.e-1']
# Data_Norms = ['per_signal', 'per_rms']
# Layers = ['100 20', '300', '15', '15 2', '100']


# count = 0
# for emd in EMDs:
	# for processing in Processings:
		# for denois in Denois:
			# for data_norm in Data_Norms:
				# for layer in Layers:
					# try:
						# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features i10statsnm_lrstd --rs 1 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation tanh --data_norm ' + data_norm + ' --layers ' + layer + ' --alpha 1.e-1 --solver adam --EMD ' + emd + ' --denois ' + denois + ' --processing ' + processing + ' --med_kernel 3 --NN_name ' + str(count))


						# # +++++++++++VALID 1500 80
						# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features i10statsnm_lrstd --method NN --NN_model C:\\Felix\\Code\\nico\\clf_' + str(count) +'.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\V3_9_n1500_M80_AE_Signal_20160506_152625.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 0 --data_norm ' + data_norm + ' --EMD ' + emd + ' --denois ' + denois + ' --processing ' + processing + ' --med_kernel 3 --plot OFF')
						
						# # +++++++++++VALID 1000 80
						# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features i10statsnm_lrstd --method NN --NN_model C:\\Felix\\Code\\nico\\clf_' + str(count) +'.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\V1_8_n1000_M80_AE_Signal_20160928_144217.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\V1_8_n1000_M80_AE_Signal_20160506_141822.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 0 --data_norm ' + data_norm + ' --EMD ' + emd + ' --denois ' + denois + ' --processing ' + processing + ' --med_kernel 3 --plot OFF')
						
						
						
						
					# except:
						# print('exception')
					
					# count = count + 1



	



# #+++++++++++TEST 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features interval10_stats_nomean --method THR --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20170921_110533_.pkl --data_norm per_signal --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\V2_9_n1500_M80_AE_Signal_20160928_151441.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\OK\\V2_9_n1500_M80_AE_Signal_20160506_145215.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\classification_20170922_134842_V2_9_n1500_M80_AE_Signal_20160928_151441.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\OK\\classification_20170925_164839_V2_9_n1500_M80_AE_Signal_20160506_145215.pkl --clf_check ON --class2 0 --thr_mode fixed_value --thr_value 7.5 --data_norm per_rms')


# #+++++++++++TEST 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features interval10_stats_nomean --method THR --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20170921_110533_.pkl --data_norm per_signal --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\V2_8_n1000_M80_AE_Signal_20160928_151108.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\OK\\V2_8_n1000_M80_AE_Signal_20160506_144637.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\classification_20170922_122221_V2_8_n1000_M80_AE_Signal_20160928_151108.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\OK\\classification_20170922_122316_V2_8_n1000_M80_AE_Signal_20160506_144637.pkl --clf_check ON --class2 0 --thr_mode fixed_value --thr_value 7.5 --data_norm per_rms')


# #+++++++++++TEST 1500 40
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features interval10_stats_nomean --method THR --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20170921_110533_.pkl --data_norm per_signal --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\V1_6_n1500_M40_AE_Signal_20160928_143502.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\OK\\V1_6_n1500_M40_AE_Signal_20160506_140849.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\classification_20170921_151654_V1_6_n1500_M40_AE_Signal_20160928_143502.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\OK\\classification_20170925_164917_V1_6_n1500_M40_AE_Signal_20160506_140849.pkl --clf_check ON --class2 0 --thr_mode fixed_value --thr_value 7.5 --data_norm per_rms')




























# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 100 10 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 50 10 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 50 5 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 300 150 80 40 20 10 5 2 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')


# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval5_stats_nomean --layers 50 40 30 25 15 10 5 2 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')




