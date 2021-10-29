import os
import sys
os.system('cd C:\Felix\Code\nico')
from numpy import arange



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++MANUAL

# +++++++++++TRAINING 1500 80

# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --fs 1.e6 --features sortint20_stats_nsnk --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20171007_155215_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20171007_181945_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation tanh --rs 1 --data_norm per_rms --alpha 1.e4 --solver lbfgs --layers 320 240 160 120 80 40 20 15 6 3 --processing butter_demod --eval_features OFF --tol 1.e-4 --diff 1 --classes 3n_2isclass --class2 0 --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3')



# # +++++++++++VALID 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --save_plot OFF --fs 1.e6 --power2 20 --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171013_161658.pkl --n_files 2 --save OFF --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\V3_9_n1500_M80_AE_Signal_20160506_152625.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 0 --classes 3n_2isclass --data_norm per_rms --processing butter_demod --plot ON --diff 1 --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --thr_mode fixed_value --thr_value 0.008 --window_delay 0')

# a = input('press enter to continue . . . ')
# #+++++++++++VALID 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --save_plot OFF --fs 1.e6 --power2 20 --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171013_161658.pkl --n_files 2 --save OFF --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\V1_8_n1000_M80_AE_Signal_20160928_144217.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\V1_8_n1000_M80_AE_Signal_20160506_141822.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 0 --classes 3n_2isclass --data_norm per_rms --processing butter_demod --plot ON --diff 1 --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --thr_mode fixed_value --thr_value 0.008 --window_delay 0')




# #+++++++++++TEST 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171011_122417.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\V2_9_n1500_M80_AE_Signal_20160928_151441.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\OK\\V2_9_n1500_M80_AE_Signal_20160506_145215.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\classification_20170922_134842_V2_9_n1500_M80_AE_Signal_20160928_151441.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\OK\\classification_20170925_164839_V2_9_n1500_M80_AE_Signal_20160506_145215.pkl --clf_check ON --class2 0 --data_norm per_rms --processing butter_demod --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot OFF --diff 1 --classes 3n_2isclass --thr_mode fixed_value --thr_value 5 --save_plot OFF --window_delay 0 --save_name best')
# a = input('press enter to continue . . . ')


# #+++++++++++TEST 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171011_122417.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\V2_8_n1000_M80_AE_Signal_20160928_151108.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\OK\\V2_8_n1000_M80_AE_Signal_20160506_144637.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\classification_20170922_122221_V2_8_n1000_M80_AE_Signal_20160928_151108.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\OK\\classification_20170922_122316_V2_8_n1000_M80_AE_Signal_20160506_144637.pkl --clf_check ON --class2 0 --data_norm per_rms --processing butter_demod --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot OFF --diff 1 --classes 3n_2isclass --thr_mode fixed_value --thr_value 5 --save_plot OFF --window_extend 0 --save_name best')
# a = input('press enter to continue . . . ')


# #+++++++++++TEST 1500 40
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171011_122417.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\V1_6_n1500_M40_AE_Signal_20160928_143502.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\OK\\V1_6_n1500_M40_AE_Signal_20160506_140849.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\classification_20170921_151654_V1_6_n1500_M40_AE_Signal_20160928_143502.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\OK\\classification_20170925_164917_V1_6_n1500_M40_AE_Signal_20160506_140849.pkl --clf_check ON --class2 0 --data_norm per_rms --processing butter_demod --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot OFF --diff 1 --classes 3n_2isclass --thr_mode fixed_value --thr_value 5 --save_plot OFF --window_extend 0 --save_name 40best')



# 0.009755






# sys.exit()

# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features i10statsnmnsnk_lrstd --files C:\\work\\Burst_Detection\\Data\\Train_Case_1500_80\\Fault\\train_1500_80_fault.txt C:\\work\\Burst_Detection\\Data\\Train_Case_1500_80\\Fault\\train_1500_80_fault.txt C:\\work\\Burst_Detection\\Data\\Train_Case_1500_80\\Fault\\train_1500_80_fault.txt C:\\work\\Burst_Detection\\Data\\Train_Case_1500_80\\OK\\train_1500_80_ok.txt --classifications C:\\work\Burst_Detection\\Data\\Train_Case_1500_80\\Fault\\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\\work\Burst_Detection\\Data\\Train_Case_1500_80\\Fault\\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\\work\Burst_Detection\\Data\\Train_Case_1500_80\\Fault\\classification_20171007_155215_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\\work\Burst_Detection\\Data\\Train_Case_1500_80\\OK\\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1 --data_norm per_rms --alpha 1.e-1 --solver adam --layers 100 20 --processing demod_hilbert --eval_features OFF')



# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features i10statsnmnsnk_lrstd --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1 --data_norm per_rms --alpha 1.e-1 --solver adam --layers 100 20 --processing demod_hilbert --eval_features OFF')






#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AUTO ANN
EMDs = ['OFF']
Processings = ['butter_demod']
Denois = ['OFF']

Alphas = ['1.e1']
Data_Norms = ['per_rms']
Layers = ['160', '300', '120 20', '250 40']

Classes = ['3n_2isclass']

count = 0
for emd in EMDs:
	for processing in Processings:
		for denois in Denois:
			for alpha in Alphas:
				for data_norm in Data_Norms:
					for layer in Layers:
						for classe in Classes:
						# try:
							os.system('python Reco_Signal_Training.py --channel AE_Signal --fs 1.e6 --save ON --features sortint20_stats_nsnk --rs 1 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20171007_155215_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20171007_181945_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --tol 1.e-4 --activation tanh --data_norm ' + data_norm + ' --layers ' + layer + ' --alpha 1.e1 --class2 0 --solver lbfgs --EMD ' + emd + ' --denois ' + denois + ' --processing ' + processing + ' --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --NN_name ' + str(count+12) + ' --classes ' + classe)


							# +++++++++++VALID 1500 80
							os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Code\\nico\\clf_' + str(count+12) + '.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\V3_9_n1500_M80_AE_Signal_20160506_152625.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 0 --data_norm ' + data_norm + ' --EMD ' + emd + ' --denois ' + denois + ' --processing ' + processing + ' --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot OFF' + ' --classes ' + classe)
							
							# +++++++++++VALID 1000 80
							os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save ON --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Code\\nico\\clf_' + str(count+12) + '.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\V1_8_n1000_M80_AE_Signal_20160928_144217.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\V1_8_n1000_M80_AE_Signal_20160506_141822.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 0 --data_norm ' + data_norm + ' --EMD ' + emd + ' --denois ' + denois + ' --processing ' + processing + ' --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot OFF' + ' --classes ' + classe)
							
								
								
								
							# except:
								# print('exception')
							
							count = count + 1
# termina



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AUTO EDG

# Demod_Prefilters = ['OFF']
# Demod_Filters = ['lowpass 2000. 3', 'lowpass 5000. 3', 'lowpass 8000. 3']
# Delays = ['0.0002', '0.0001', '0.00005']
# # Thresholds = ['0.004']
# Thresholds = arange(0.001, 0.019, 0.001)
# Thresholds = [str(indi) for indi in Thresholds]
# # print(Thresholds)
# # sys.exit()

# count = 0
# for demod_prefilter in Demod_Prefilters:
	# for demod_filter, delay in zip(Demod_Filters, Delays):
		# for threshold in Thresholds:
			# try:
				# # +++++++++++VALID 1500 80
				# os.system('python Burst_Detection.py --channel AE_Signal --save_plot OFF --fs 1.e6 --power2 20 --features sortint20_stats_nsnk --method EDG --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171012_153112.pkl --n_files 2 --save ON --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\V3_9_n1500_M80_AE_Signal_20160506_152625.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 0 --classes 3n_2isclass --data_norm per_rms --processing butter_demod --plot OFF --diff 1 --demod_filter ' + demod_filter + ' --demod_prefilter ' + demod_prefilter + ' --thr_mode fixed_value --thr_value ' + threshold + ' --window_delay ' + delay + ' --save_name ' + str(count))

				# # a = input('press enter to continue . . . ')
				# #+++++++++++VALID 1000 80
				# os.system('python Burst_Detection.py --channel AE_Signal --save_plot OFF --fs 1.e6 --power2 20 --features sortint20_stats_nsnk --method EDG --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171012_153112.pkl --n_files 2 --save ON --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\V1_8_n1000_M80_AE_Signal_20160928_144217.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\V1_8_n1000_M80_AE_Signal_20160506_141822.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 0 --classes 3n_2isclass --data_norm per_rms --processing butter_demod --plot OFF --diff 1 --demod_filter ' + demod_filter + ' --demod_prefilter ' + demod_prefilter + ' --thr_mode fixed_value --thr_value ' + threshold + ' --window_delay ' + delay + ' --save_name ' + str(count))
				

			# except:
				# print('exception')
				
			# count = count + 1
# # # termina



































#CASA****************************************************************

# # +++++++++++VALID 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features Data --method NN --n_files 2 --save OFF --processing butter_demod --files C:\\work\\Burst_Detection\\Data\\Validation_Case\\1500_80\\Fault\\valid_1500_80_fault.txt C:\\work\\Burst_Detection\\Data\\Validation_Case\\1500_80\\OK\\valid_1500_80_ok.txt --clf_files C:\\work\Burst_Detection\\Data\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\work\\Burst_Detection\\Data\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 1 --data_norm per_rms --plot ON --NN_model C:\\code\\nico\\clf_20171008_143406.pkl --classes 2n_2noclass --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --diff 1')	

# a = input('press enter to continue . . . ')
# #+++++++++++VALID 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features Data --method NN --n_files 2 --save OFF --processing butter_demod --files C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\Fault\\valid_1000_80_fault.txt C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\OK\\valid_1000_80_ok.txt --clf_files C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 1 --data_norm per_rms --plot ON --NN_model C:\\code\\nico\\clf_20171008_143406.pkl --classes 2n_2noclass --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --diff 1')



