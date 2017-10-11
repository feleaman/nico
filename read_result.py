import os
import sys
import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
from m_open_extension import read_pickle
from argparse import ArgumentParser
from matplotlib.pyplot import plot, title, savefig, figure, legend
from numpy import float64, loadtxt
# def read_pickle(pickle_name):
	# pik = open(pickle_name, 'rb')
	# pickle_data = pickle.load(pik)
	# return pickle_data


Inputs = ['mode']
InputsOpt_Defaults = {'fpr_max':4.0, 'fscore_min':65.0}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	# print(config)
	# print(config['fpr_max'])
	# sys.exit()
	if config['mode'] == 'one':
		flag = '1'
		while flag == '1':
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()			
			root.destroy()
			Filenames, Results, config = read_pickle(filepath)
			print('++++++++ Results ++++++++')
			print('\n+++Result File:')
			print(filepath)
			print('\n+++Config:')
			print(config)
			print('\n+++Filenames:')
			print(Filenames)
			print('\n+++Result on File 1:')
			print(Results[0])
			print('\n+++Result on File 2:')
			print(Results[1])
			print('\n+++Result on both Files:')
			print('\nRecall: ', Results[0]['Recall'])
			precision = (100*Results[0]['TP'])/(Results[0]['TP'] + Results[0]['FP'] + Results[1]['FP'])
			print('\nPrecision: ', precision)
			print('\nF Score: ', 2*precision*Results[0]['Recall']/(precision + Results[0]['Recall']))
			print('\nFPR: ', 100*(Results[0]['FP'] + Results[1]['FP'])/(Results[0]['FP'] + Results[1]['FP'] + Results[0]['TN'] + Results[1]['TN']))
			flag = input('\nFlag: ')
	
	elif config['mode'] == 'multi':
		root = Tk()
		root.withdraw()
		root.update()
		Multi_Filepaths = filedialog.askopenfilenames()
		Multi_Filepaths = list(Multi_Filepaths)			
		root.destroy()
		Recalls = []
		Precisions = []
		F_Scores = []
		FPRs = []
		TNRs = []
		for filepath in Multi_Filepaths:
			Filenames, Results, config = read_pickle(filepath)
			try:
				precision = (100*Results[0]['TP'])/(Results[0]['TP'] + Results[0]['FP'] + Results[1]['FP'])
			except:
				precision = 0
			try:
				f_score = 2*precision*Results[0]['Recall']/(precision + Results[0]['Recall'])
			except:
				f_score = 0
			fpr = 100*(Results[0]['FP'] + Results[1]['FP'])/(Results[0]['FP'] + Results[1]['FP'] + Results[0]['TN'] + Results[1]['TN'])
			Recalls.append(Results[0]['Recall'])
			Precisions.append(precision)
			F_Scores.append(f_score)
			FPRs.append(fpr)
			TNRs.append(100-fpr)
		# figure(0)
		# plot(Recalls, 'ro-')
		# title('Recall')
		# savefig('Recall.png')
		
		# figure(1)
		# plot(Precisions, 'go-')
		# title('Precision')
		# savefig('Precision.png')
		
		# figure(2)
		# plot(F_Scores, 'bo-')
		# title('F Score')
		# savefig('F_Score.png')
		
		figure(0)
		plot(FPRs, 'kp-')
		title('False Positive Rate')
		savefig('FPR.png')
		
		# figure(4)
		# plot(TNRs, 'ko-')
		# title('True Negative Rate')
		# savefig('TNR.png')
		
		figure(1)
		plot(TNRs, 'ko-', label='TNR')
		plot(F_Scores, 'bo-', label='FScore')
		legend(loc='best')
		title('TNR and F Score')
		savefig('TNR_FScore.png')
		
		figure(2)
		plot(Recalls, 'ro-', label='Recall')
		plot(Precisions, 'go-', label='Precisions')
		legend(loc='best')
		title('Recall and Precision')
		savefig('Recall_Precision.png')
	
	elif config['mode'] == 'bests':
		root = Tk()
		root.withdraw()
		root.update()
		Multi_Filepaths = filedialog.askopenfilenames()
		Multi_Filepaths = list(Multi_Filepaths)			
		root.destroy()
		Recalls = []
		Precisions = []
		F_Scores = []
		FPRs = []
		TNRs = []
		for filepath in Multi_Filepaths:
			Filenames, Results, config_pickle = read_pickle(filepath)
			try:
				precision = (100*Results[0]['TP'])/(Results[0]['TP'] + Results[0]['FP'] + Results[1]['FP'])
			except:
				precision = 0
			try:
				f_score = 2*precision*Results[0]['Recall']/(precision + Results[0]['Recall'])
			except:
				f_score = 0
			fpr = 100*(Results[0]['FP'] + Results[1]['FP'])/(Results[0]['FP'] + Results[1]['FP'] + Results[0]['TN'] + Results[1]['TN'])
			Recalls.append(Results[0]['Recall'])
			Precisions.append(precision)
			F_Scores.append(f_score)
			FPRs.append(fpr)
			TNRs.append(100-fpr)
		count = 0
		print('+++Bests+++')
		for fpr, fscore in zip(FPRs, F_Scores):
			if ((fpr < config['fpr_max']) and (fscore > config['fscore_min'])):
				print(count)
			count = count + 1
	
	elif config['mode'] == 'eval_features':
		print('select features')
		root = Tk()
		root.withdraw()
		root.update()
		featurespath = filedialog.askopenfilenames()
		featurespath = list(featurespath)
		root.destroy()
		# Features = read_pickle(featurespath)
		print('select classification')
		root = Tk()
		root.withdraw()
		root.update()
		classpath = filedialog.askopenfilename()
		root.destroy()
		classification = loadtxt(classpath)
		
		for i in range(len(featurespath)):
			features = loadtxt(featurespath[i])
			# x = [j for j in range(len(features))]
			figure(i)
			plot(features, 'bo')
			title('feature ' + str(i))
			
			features_pos = []
			x = []
			for k in range(len(classification)):
				if int(classification[k]) == 1:
					features_pos.append(features[k])
					x.append(k)
			
			# features_pos = [features[k] for k in range(len(classification)) if int(classification[k]) == 1]
			plot(x, features_pos, 'ro')
			savefig('feature' + str(i) + '.png')

		
	
	else:
		print('unknown mode')
		
	return






def read_parser(argv, Inputs, InputsOpt_Defaults):
	Inputs_opt = [key for key in InputsOpt_Defaults]
	Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]

	parser = ArgumentParser()
	for element in (Inputs + Inputs_opt):
		print(element)
		if element == 'no_element':
			parser.add_argument('--' + element, nargs='+')
		else:
			parser.add_argument('--' + element, nargs='?')
	
	args = parser.parse_args()
	config_input = {}
	for element in Inputs:
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Required:', element)
			sys.exit()

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config_input[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config_input[element] = value
	
	#Type conversion to float
	config_input['fpr_max'] = float(config_input['fpr_max'])
	config_input['fscore_min'] = float(config_input['fscore_min'])
	#Type conversion to int	
	# Variable conversion

	return config_input

if __name__ == '__main__':
	main(sys.argv)