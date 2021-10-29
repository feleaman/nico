import os
import sys
os.system('cd C:\Felix\Code\nico')
from numpy import arange




Alphas = ['1.e-3', '1.e-1', '1.e1', '1.e3']
Solvers = ['adam', 'lbfgs']
Activations = ['identity']
Trains = ['0.6']
Valids = ['0.2']

N_Mov_Avgs = ['1']
Features = ['5h_f_g']
N_Pres = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4']
M_Posts = ['0.05', '0.1', '0.15', '0.2']
Layers_Pros = ['0']
# RSs = ['1']



count = 0
for layers_pro in Layers_Pros:
	for alpha in Alphas:
		for solver in Solvers:
			for activation in Activations:
				for n_pre in N_Pres:
					for m_post in M_Posts:
						for n_mov_avg in N_Mov_Avgs:
							for train in Trains:
								for valid in Valids:
									for feature in Features:
					
										os.system('python Full_Prognosis.py --mode mode2c --name ' + str(count) + ' --activation_pro ' + activation + ' --solver_pro ' + solver + ' --layers_pro ' + layers_pro + ' --alpha_pro ' + alpha + ' --max_iter 500000 --n_pre ' + n_pre + ' --m_post ' + m_post + ' --n_mov_avg ' + n_mov_avg + ' --train ' + train + ' --feature ' + feature + ' --source_file AUTO --valid ' + valid)
					
					
					
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






