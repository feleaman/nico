import os
import sys
os.system('cd C:\Felix\Code\nico')
from numpy import arange


# mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\Congreso_Lyon\\Relative_Counting_08mV_THR_300_300'
# count = 999

os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.854891255 --m_post 0.1359038 --hidden_layers 1163 --alpha 0.099571681 --epochs 4 --mini_batches 35 --save_model ON --name train70_gen14 --save_plot ON')
sys.exit()

# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.74032954 --m_post 0.160153759 --hidden_layers 1192 --alpha 0.0804273 --epochs 13 --mini_batches 32 --save_model ON --name train70_gen8 --save_plot ON')
# sys.exit()


# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.5 --m_post 0.13 --hidden_layers 1183 --alpha 0.1 --epochs 59 --mini_batches 32 --save_model ON --name train70_ini --save_plot ON')
# sys.exit()

# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.8180189 --m_post 0.157536615 --hidden_layers 1324 --alpha 0.10291208 --epochs 18 --mini_batches 33 --save_model ON --name train70_gen9 --save_plot ON')
# sys.exit()


# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.61350857 --m_post 0.13612566 --hidden_layers 1379 --alpha 0.10616858 --epochs 30 --mini_batches 27 --save_model ON --name train70_gen6 --save_plot ON')
# sys.exit()

# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.535581203 --m_post 0.134615634 --hidden_layers 1731 945 --alpha 0.11996927 --epochs 31 --mini_batches 35 --save_model ON --name train70_gen4 --save_plot ON')
# sys.exit()

# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.6875412 --m_post 0.1331973 --hidden_layers 1506 821 --alpha 0.078932 --epochs 32 --mini_batches 28 --save_model ON --name train70_gen7 --save_plot ON')
# sys.exit()

# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.8255944 --m_post 0.1636741 --hidden_layers 1375 --alpha 0.1019659 --epochs 5 --mini_batches 34 --save_model ON --name train70_gen10 --save_plot ON')
# sys.exit()

# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.7 --n_pre 0.5893027 --m_post 0.1509425 --hidden_layers 1158 --alpha 0.097952 --epochs 10 --mini_batches 30 --save_model ON --name train70_gen5 --save_plot ON')
# sys.exit()


# os.system('python M_Regression7.py --mode ann_test --n_mov_avg 12 --feature CORR --train 0.8 --n_pre 0.6348067 --m_post 0.1505314 --hidden_layers 1858 1014 --alpha 0.13085 --epochs 21 --mini_batches 33 --save_model ON --name test80 --save_plot ON')
# sys.exit()


# os.system('python M_Regression7.py --mode generate_genetic --n_bests 18 --n_children 54 --weight 0.1')
# sys.exit()

# # os.system('python M_Regression7.py --plot ON --rs 0 --hidden_layers 1858 1014 --mode ann_test_with_model --name CORR_Idx14_ANN_Idx_' + str(count) + ' --activation identity --train_test_ref ON --alpha 1.3085e-1 --n_pre 0.63481 --m_post 0.15053 --n_mov_avg 12 --train 0.8 --feature CORR --predict_node last --valid 0. --mini_batches 33 --epochs 21 --mypath ' + mypath + ' --save_plot OFF --save_model OFF')


# sys.exit()

Alphas = ['1.e-3', '1.e-1']
Activations = ['identity']
Trains = ['0.5']
Valids = ['0.2']
N_Mov_Avgs = ['12']
N_Pres = ['0.5', '0.32', '0.16']
M_Posts = ['0', '0.06', '0.13']
Hidden_Layers = ['0 1', '0 2']
Mini_Batches = ['32', '128']
Epochs = ['100']

# mypath = 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Tri_Analysis\\Idx14'

mypath = 'C:\\Felix\\29_THESIS\\MODEL_A\\Chapter3_Test_Bench\\04_Data\\Congreso_Lyon\\Relative_Counting_08mV_THR_300_300'

count = 0
for hidden_layers in Hidden_Layers:
	for alpha in Alphas:
		for activation in Activations:
			for n_pre in N_Pres:
				for m_post in M_Posts:
					for n_mov_avg in N_Mov_Avgs:
						for train in Trains:
							for valid in Valids:
								for mini_batches in Mini_Batches:
									for epochs in Epochs:
										if count >= 0:
											os.system('python M_Regression7.py --plot OFF --rs 0 --hidden_layers ' + hidden_layers + ' --mode ann_validation --name CORR_Idx14_ANN_Idx_' + str(count) + ' --activation ' + activation + ' --train_test_ref ON --alpha ' + alpha + ' --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg ' + n_mov_avg + ' --train ' + train + ' --feature CORR' + ' --predict_node last --valid ' + valid + ' --mini_batches ' + mini_batches + ' --epochs ' + epochs + ' --mypath ' + mypath + ' --save_plot ON --save_model OFF')
					
					
					
										count += 1


sys.exit()




#old
Alphas = ['1.e-3', '1.e-1']
Activations = ['identity']
Trains = ['0.6']
Valids = ['0.2']
N_Mov_Avgs = ['12']
N_Pres = ['0.5', '0.35', '0.2']
M_Posts = ['0.05', '0.1', '0.2']
Auto_Layers = ['1x50', '2x75x25', '3x80x50x20']
Mini_Batches = ['32', '128']
Epochs = ['100']
mypath = 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Tri_Analysis\\Idx14'


count = 0
for auto_layers in Auto_Layers:
	for alpha in Alphas:
		for activation in Activations:
			for n_pre in N_Pres:
				for m_post in M_Posts:
					for n_mov_avg in N_Mov_Avgs:
						for train in Trains:
							for valid in Valids:
								for mini_batches in Mini_Batches:
									for epochs in Epochs:
										os.system('python M_Regression.py --plot OFF --rs 0 --hidden_layers 0 --mode auto_ann_regression_msevalid --name CORR_Idx14_ANN_Idx_' + str(count) + ' --activation ' + activation + ' --auto_layers ' + auto_layers + ' --alpha ' + alpha + ' --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg ' + n_mov_avg + ' --train ' + train + ' --feature CORR' + ' --predict_node last --valid ' + valid + ' --mini_batches ' + mini_batches + ' --epochs ' + epochs + ' --mypath ' + mypath + ' --save_plot ON --save_model OFF')
					
					
					
										count += 1


sys.exit()



Alphas = ['0.009269']
Activations = ['identity']
Trains = ['0.80']
Valids = ['0.0']
N_Mov_Avgs = ['12']
N_Pres = ['0.512842']
M_Posts = ['0.208447']
Hidden_Layers = ['1748 1130']
Mini_Batches = ['172']
Epochs = ['44']
mypath = 'C:\\Felix\\29_THESIS\\MODEL_B\\Chapter_4_Prognostics\\04_Data\\Tri_Analysis\\Idx14'


count = 0
for layers in Hidden_Layers:
	for alpha in Alphas:
		for activation in Activations:
			for n_pre in N_Pres:
				for m_post in M_Posts:
					for n_mov_avg in N_Mov_Avgs:
						for train in Trains:
							for valid in Valids:
								for mini_batches in Mini_Batches:
									for epochs in Epochs:
										os.system('python M_Regression.py --plot OFF --rs 0 --hidden_layers ' + layers + ' --mode auto_ann_regression_novalid --name Best_Gen2s_8_Idx14_ANN_Idx_' + str(count) + ' --activation ' + activation + ' --auto_layers no --alpha ' + alpha + ' --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg ' + n_mov_avg + ' --train ' + train + ' --feature CORR' + ' --predict_node last --valid ' + valid + ' --mini_batches ' + mini_batches + ' --epochs ' + epochs + ' --mypath ' + mypath + ' --save_plot ON --save_model ON')
					
					
					
										count += 1


sys.exit()

# Alphas = ['1.e-3', '1.e-1', '1.e1', '1.e3']
# Solvers = ['adam', 'lbfgs']
# Activations = ['identity']
# # Trains = ['0.7']
# Trains = ['0.2', '0.4', '0.6', '0.8']
# N_Mov_Avgs = ['1']
# Features = ['5h_f_g']
# N_Pres = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4']
# M_Posts = ['0.05', '0.1', '0.15', '0.2']
# Layers_Pros = ['0']
# # RSs = ['1']



# count = 0
# for layers_pro in Layers_Pros:
	# for alpha in Alphas:
		# for solver in Solvers:
			# for activation in Activations:
				# for n_pre in N_Pres:
					# for m_post in M_Posts:
						# for n_mov_avg in N_Mov_Avgs:
							# for train in Trains:
								# for feature in Features:
				
									# os.system('python Full_Prognosis.py --mode mode2 --name ' + str(count) + ' --activation_pro ' + activation + ' --solver_pro ' + solver + ' --layers_pro ' + layers_pro + ' --alpha_pro ' + alpha + ' --max_iter 500000 --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg ' + n_mov_avg + ' --train ' + train + ' --feature ' + feature + ' --source_file AUTO')
				
				
				
									# count += 1






