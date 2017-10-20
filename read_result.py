import os
import sys
import pickle
from tkinter import filedialog
from tkinter import Tk
sys.path.insert(0, './lib') #to open user-defined functions
from m_open_extension import read_pickle, save_pickle
from argparse import ArgumentParser
from matplotlib.pyplot import plot, title, savefig, figure, legend
from numpy import float64, loadtxt
# def read_pickle(pickle_name):
	# pik = open(pickle_name, 'rb')
	# pickle_data = pickle.load(pik)
	# return pickle_data
import pandas as pd

Inputs = ['mode']
InputsOpt_Defaults = {'fpr_max':4.0, 'fscore_min':65.0, 'table':'params'}

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
	
	elif config['mode'] == 'multi_plot':
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
	
	elif config['mode'] == 'multi_excel':
		writer = pd.ExcelWriter('Results.xlsx')
		
		if config['table'] == 'params':
			print('Select with 1500')
			DF_1500 = obtain_DF_params()		
			DF_1500.to_excel(writer, sheet_name='1500')	
			print('Select with 1000')
			DF_1000 = obtain_DF_params()		
			DF_1000.to_excel(writer, sheet_name='1000')
		elif config['table'] == 'mean':
			print('Select with 1500')
			DF_1500 = obtain_DF()		
			DF_1500.to_excel(writer, sheet_name='1500')	
			print('Select with 1000')
			DF_1000 = obtain_DF()		
			DF_1000.to_excel(writer, sheet_name='1000')
			DF_MEAN = (DF_1000 + DF_1500)/2.
			DF_MEAN.to_excel(writer, sheet_name='MEAN')
		else:
			print('unknown table type')
			sys.exit()
		writer.save()
	
	elif config['mode'] == 'excel_test_3':
		writer = pd.ExcelWriter('Test_Results.xlsx')
		
		print('Select with 1500_80')
		DF_1500_80 = obtain_DF_params()		
		DF_1500_80.to_excel(writer, sheet_name='Test_Results', startrow=0)	
		print('Select with 1000_80')
		DF_1000_80 = obtain_DF_params()
		DF_1000_80.to_excel(writer, sheet_name='Test_Results', startrow=5)	
		print('Select with 1500_40')
		DF_1500_40 = obtain_DF_params()
		DF_1500_40.to_excel(writer, sheet_name='Test_Results', startrow=10)	


		writer.save()
	
	elif config['mode'] == 'excel_test_4':
		writer = pd.ExcelWriter('Test_ResultsExt.xlsx')
		
		print('Select with 1500_80')
		DF_1500_80 = obtain_DF_params()		
		DF_1500_80.to_excel(writer, sheet_name='Test_Results', startrow=0)	
		print('Select with 1000_80')
		DF_1000_80 = obtain_DF_params()
		DF_1000_80.to_excel(writer, sheet_name='Test_Results', startrow=5)	
		print('Select with 1500_40')
		DF_1500_40 = obtain_DF_params()
		DF_1500_40.to_excel(writer, sheet_name='Test_Results', startrow=10)	
		print('Select with 1000_40')
		DF_1000_40 = obtain_DF_params()
		DF_1000_40.to_excel(writer, sheet_name='Test_Results', startrow=15)	


		writer.save()
	
	elif config['mode'] == 'modify_classification':
		root = Tk()
		root.withdraw()
		root.update()
		filename = filedialog.askopenfilename()
		root.destroy()

		pik = read_pickle(filename)
		filename = os.path.basename(filename)
		print(pik['classification'][324])
		print(pik['classification'][325])
		print(pik['classification'][326])
		# pik['classification'][456] = 0
		# pik['classification'][566] = 1
		# pik['classification'][723] = 0
		# pik['classification'][724] = 1
		# pik['classification'][967] = 1
		# pik['classification'][878] = 1
		# pik['classification'][322] = 0
		# pik['classification'][323] = 1
		
		# save_pickle(filename, pik)


	
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

def obtain_DF():
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
	Rs = []
	Params = []
	FP_0 = []
	TP_0 = []
	TN_0 = []
	FN_0 = []
	FP_1 = []
	TP_1 = []
	TN_1 = []
	FN_1 = []
	
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
		try:
			r = Results[0]['TP'] / Results[1]['FP']
		except:
			r = -1
		fpr = 100*(Results[0]['FP'] + Results[1]['FP'])/(Results[0]['FP'] + Results[1]['FP'] + Results[0]['TN'] + Results[1]['TN'])
		Recalls.append(Results[0]['Recall'])
		Precisions.append(precision)
		F_Scores.append(f_score)
		FPRs.append(fpr)
		Rs.append(r)			
		FP_0.append(Results[0]['FP'])
		TP_0.append(Results[0]['TP'])
		TN_0.append(Results[0]['TN'])
		FN_0.append(Results[0]['FN'])
		
		FP_1.append(Results[1]['FP'])
		TP_1.append(Results[1]['TP'])
		TN_1.append(Results[1]['TN'])
		FN_1.append(Results[1]['FN'])
		
		Params.append(str(config['demod_prefilter']) + ' ' + str(config['demod_filter']) + ' ' + str(['delay', config['window_delay']]) + ' ' + str(['threshold', config['thr_value']]))
	
	# row_names = ['FP_0', 'TP_0', 'TN_0', 'FN_0', 'FP_1', 'TP_1', 'TN_1', 'FN_1', 'Recall', 'Precision', 'F_Score', 'FPR', 'R']
	# dict = {'FP_0':FP_0, 'TP_0':TP_0, 'TN_0':TN_0, 'FN_0':FN_0, 'FP_1':FP_1, 'TP_1':TP_1, 'TN_1':TN_1, 'FN_1':FN_1, 'Recall':Recalls, 'Precisions':Precisions, 'F_Score':F_Scores, 'FPR':FPRs, 'R':Rs}
	dict = {'a_FP_0':FP_0, 'b_TP_0':TP_0, 'c_TN_0':TN_0, 'd_FN_0':FN_0, 'e_FP_1':FP_1, 'f_TP_1':TP_1, 'g_TN_1':TN_1, 'h_FN_1':FN_1, 'i_Recall':Recalls, 'j_Precisions':Precisions, 'k_F_Score':F_Scores, 'l_FPR':FPRs, 'm_R':Rs}
	DataF = pd.DataFrame(data=dict, index=Params)
			
	return DataF

def obtain_DF_params():
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
	Rs = []
	Params = []
	FP_0 = []
	TP_0 = []
	TN_0 = []
	FN_0 = []
	FP_1 = []
	TP_1 = []
	TN_1 = []
	FN_1 = []
	PreFilters = []
	Filters = []
	Thresholds = []
	
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
		try:
			r = Results[0]['TP'] / Results[1]['FP']
		except:
			r = -1
		fpr = 100*(Results[0]['FP'] + Results[1]['FP'])/(Results[0]['FP'] + Results[1]['FP'] + Results[0]['TN'] + Results[1]['TN'])
		Recalls.append(Results[0]['Recall'])
		Precisions.append(precision)
		F_Scores.append(f_score)
		FPRs.append(fpr)
		Rs.append(r)			
		FP_0.append(Results[0]['FP'])
		TP_0.append(Results[0]['TP'])
		TN_0.append(Results[0]['TN'])
		FN_0.append(Results[0]['FN'])
		
		FP_1.append(Results[1]['FP'])
		TP_1.append(Results[1]['TP'])
		TN_1.append(Results[1]['TN'])
		FN_1.append(Results[1]['FN'])
		
		PreFilters.append(str(config['demod_prefilter']))
		Filters.append(str(config['demod_filter']))
		Thresholds.append(config['thr_value'])
		
		# Params.append(str(config['demod_prefilter']) + ' ' + str(config['demod_filter']) + ' ' + str(['delay', config['window_delay']]) + ' ' + str(['threshold', config['thr_value']]))
	
	# row_names = ['FP_0', 'TP_0', 'TN_0', 'FN_0', 'FP_1', 'TP_1', 'TN_1', 'FN_1', 'Recall', 'Precision', 'F_Score', 'FPR', 'R']
	# dict = {'FP_0':FP_0, 'TP_0':TP_0, 'TN_0':TN_0, 'FN_0':FN_0, 'FP_1':FP_1, 'TP_1':TP_1, 'TN_1':TN_1, 'FN_1':FN_1, 'Recall':Recalls, 'Precisions':Precisions, 'F_Score':F_Scores, 'FPR':FPRs, 'R':Rs}
	dict = {'0_PreFilter':PreFilters, '1_Filter':Filters, '2_Threshold':Thresholds, 'a_FP_0':FP_0, 'b_TP_0':TP_0, 'c_TN_0':TN_0, 'd_FN_0':FN_0, 'e_FP_1':FP_1, 'f_TP_1':TP_1, 'g_TN_1':TN_1, 'h_FN_1':FN_1, 'i_Recall':Recalls, 'j_Precisions':Precisions, 'k_F_Score':F_Scores, 'l_FPR':FPRs, 'm_R':Rs}
	DataF = pd.DataFrame(data=dict)
			
	return DataF
	
if __name__ == '__main__':
	main(sys.argv)
# header=['FP_0', 'TP_0', 'TN_0', 'FN_0', 'FP_1', 'TP_1', 'TN_1', 'FN_1', 'Recall', 'Precisions', 'F_Score', 'FPR', 'R']