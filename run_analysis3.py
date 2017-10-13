import os
import sys


os.system('cd C:\Felix\Code\nico')




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++MANUAL

# +++++++++++TRAINING 1500 80

# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --fs 1.e6 --features sortint20_stats_nsnk --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20171007_155215_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20171007_181945_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation tanh --rs 1 --data_norm per_rms --alpha 1.e2 --solver lbfgs --layers 160 80 20 --processing butter_demod --eval_features OFF --tol 1.e-4 --diff 1 --classes 3n_2isclass --class2 0 --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3')



# # +++++++++++VALID 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --save_plot OFF --fs 1.e6 --power2 20 --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171013_160404.pkl --n_files 2 --save OFF --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\V3_9_n1500_M80_AE_Signal_20160928_154159.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\V3_9_n1500_M80_AE_Signal_20160506_152625.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 0 --classes 3n_2isclass --data_norm per_rms --processing butter_demod --plot ON --diff 1 --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --thr_mode fixed_value --thr_value 0.008 --window_delay 0')

# a = input('press enter to continue . . . ')
# #+++++++++++VALID 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --save_plot OFF --fs 1.e6 --power2 20 --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171013_160404.pkl --n_files 2 --save OFF --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\V1_8_n1000_M80_AE_Signal_20160928_144217.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\V1_8_n1000_M80_AE_Signal_20160506_141822.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 0 --classes 3n_2isclass --data_norm per_rms --processing butter_demod --plot ON --diff 1 --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --thr_mode fixed_value --thr_value 0.008 --window_delay 0')




#+++++++++++TEST 1500 80
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save OFF --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171011_122417.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\V2_9_n1500_M80_AE_Signal_20160928_151441.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\OK\\V2_9_n1500_M80_AE_Signal_20160506_145215.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\Fault\\classification_20170922_134842_V2_9_n1500_M80_AE_Signal_20160928_151441.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_80\\OK\\classification_20170925_164839_V2_9_n1500_M80_AE_Signal_20160506_145215.pkl --clf_check ON --class2 0 --data_norm per_rms --processing butter_demod --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot ON --diff 1 --classes 3n_2isclass --thr_mode fixed_value --thr_value 0.008 --save_plot ON --window_delay 0 --save_name best')
# a = input('press enter to continue . . . ')


#+++++++++++TEST 1000 80
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save OFF --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171011_122417.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\V2_8_n1000_M80_AE_Signal_20160928_151108.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\OK\\V2_8_n1000_M80_AE_Signal_20160506_144637.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\Fault\\classification_20170922_122221_V2_8_n1000_M80_AE_Signal_20160928_151108.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1000_80\\OK\\classification_20170922_122316_V2_8_n1000_M80_AE_Signal_20160506_144637.pkl --clf_check ON --class2 0 --data_norm per_rms --processing butter_demod --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot ON --diff 1 --classes 3n_2isclass --thr_mode fixed_value --thr_value 0.008 --save_plot ON --window_delay 0 --save_name best')
# a = input('press enter to continue . . . ')


#+++++++++++TEST 1500 40
os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --save OFF --features sortint20_stats_nsnk --method NN --NN_model C:\\Felix\\Data\\CNs_Getriebe\\Paper_Bursts\\NN_Models\\clf_20171011_122417.pkl --n_files 2 --files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\V1_6_n1500_M40_AE_Signal_20160928_143502.mat C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\OK\\V1_6_n1500_M40_AE_Signal_20160506_140849.mat --clf_files C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\Fault\\classification_20170921_151654_V1_6_n1500_M40_AE_Signal_20160928_143502.pkl C:\\Felix\Data\\CNs_Getriebe\\Paper_Bursts\\Test_Case\\1500_40\\OK\\classification_20170925_164917_V1_6_n1500_M40_AE_Signal_20160506_140849.pkl --clf_check ON --class2 0 --data_norm per_rms --processing butter_demod --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --plot ON --diff 1 --classes 3n_2isclass --thr_mode fixed_value --thr_value 0.008 --save_plot ON --window_delay 0 --save_name 40best')








#CASA****************************************************************

# # +++++++++++VALID 1500 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features Data --method NN --n_files 2 --save OFF --processing butter_demod --files C:\\work\\Burst_Detection\\Data\\Validation_Case\\1500_80\\Fault\\valid_1500_80_fault.txt C:\\work\\Burst_Detection\\Data\\Validation_Case\\1500_80\\OK\\valid_1500_80_ok.txt --clf_files C:\\work\Burst_Detection\\Data\\Validation_Case\\1500_80\\Fault\\classification_20170921_103023_V3_9_n1500_M80_AE_Signal_20160928_154159.pkl C:\\work\\Burst_Detection\\Data\\Validation_Case\\1500_80\\OK\\classification_20170922_094233_V3_9_n1500_M80_AE_Signal_20160506_152625.pkl --clf_check ON --class2 1 --data_norm per_rms --plot ON --NN_model C:\\code\\nico\\clf_20171008_143406.pkl --classes 2n_2noclass --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --diff 1')	

# a = input('press enter to continue . . . ')
# #+++++++++++VALID 1000 80
# os.system('python Burst_Detection.py --channel AE_Signal --fs 1.e6 --power2 20 --features Data --method NN --n_files 2 --save OFF --processing butter_demod --files C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\Fault\\valid_1000_80_fault.txt C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\OK\\valid_1000_80_ok.txt --clf_files C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\Fault\\classification_20170921_131001_V1_8_n1000_M80_AE_Signal_20160928_144217.pkl C:\\work\\Burst_Detection\\Data\\Validation_Case\\1000_80\\OK\\classification_20170922_094309_V1_8_n1000_M80_AE_Signal_20160506_141822.pkl --clf_check ON --class2 1 --data_norm per_rms --plot ON --NN_model C:\\code\\nico\\clf_20171008_143406.pkl --classes 2n_2noclass --demod_filter lowpass 5000. 3 --demod_prefilter highpass 70.e3 3 --diff 1')




