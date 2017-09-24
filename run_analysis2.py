import os
# os.system('')
# os.system('python EmpMD.py --path --file_x --channel --power2 --save --file_h1  --min_iter')

os.system('cd C:\code\nico')
# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --layers 25 --classifications C:/code/nico1/classification_20170923_061144_ok_v3_n1500_m80.pkl --files C:/code/nico1/ok_v3_n1500_m80.txt --features interval10_stats_nomean')

os.system('python Burst_Detection1.py --channel AE_Signal --fs 1.e6 --power2 20 --method NN --clf_files C:\\code\\nico1\\classification_20170923_061144_ok_v3_n1500_m80.pkl --NN_model C:\\code\\nico1\\clf_20170923_104347_.pkl --data_norm per_signal --n_files 2 --files C:\\code\\nico1\\ok_v3_n1500_m80.txt C:\\code\\nico1\\defect_v3_n1500_m80.txt')


# os.system('python Burst_Detection1.py --channel AE_Signal --fs 1.e6 --power2 20 --method THR --clf_files C:\\code\\nico1\\classification_20170923_061144_ok_v3_n1500_m80.pkl --NN_model C:\\code\\nico1\\clf_20170923_104347_.pkl --data_norm per_signal --n_files 2')



























# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save OFF --features feat --layers 3')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 300 30 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')



# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval5_stats_nomean --layers 200 20 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 100 10 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 50 10 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 50 5 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')

# # os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval10_stats_nomean --layers 300 150 80 40 20 10 5 2 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')


# os.system('python Reco_Signal_Training.py --channel AE_Signal --save ON --features interval5_stats_nomean --layers 50 40 30 25 15 10 5 2 --files C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\V1_9_n1500_M80_AE_Signal_20160928_144737.mat C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\V1_9_n1500_M80_AE_Signal_20160506_142422.mat --classifications C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_111359_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\Fault\classification_20170831_093634_V1_9_n1500_M80_AE_Signal_20160928_144737.pkl C:\Felix\Data\CNs_Getriebe\Paper_Bursts\Analysis_Case_1500_80\OK\classification_20170901_101337_V1_9_n1500_M80_AE_Signal_20160506_142422.pkl --activation relu --rs 1')




