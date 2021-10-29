import numpy as np
from scipy.integrate import odeint
from scipy import signal
from scipy import stats
import scipy

def test_values_limit(ini, fin, step, features, classes):
	test_thr = list(np.arange(ini, fin, step))
	function_error = []			
	for limit in test_thr:
		error = 0
		for array_feat, array_class in zip(features, classes):
			if array_feat[0] < limit:
				prediction = 0
			else:
				prediction = 1
			error = error + (array_class - prediction)**2.0
		function_error.append(error)
	best_index = np.argmin(np.array(function_error))
	best_thr = test_thr[best_index]
	return best_thr


def obtain_bin_scores(classes_prediction, classes_test):
	if len(classes_prediction) != len(classes_test):
		print('error in obtain scores')
		sys.exit()
	mydict = {}
	TP = 0
	FP = 0
	FN = 0
	TN = 0
	for i in range(len(classes_prediction)):
		if classes_test[i] == 1:
			if classes_prediction[i] == 1:
				TP = TP + 1
			else:
				FN = FN + 1
		elif classes_test[i] == 0:
			if classes_prediction[i] == 0:
				TN = TN + 1
			else:
				FP = FP + 1
	
	try:
		recall = TP / (TP+FN)
	except:
		recall = -2
	
	try:
		precision =  TP / (TP+FP)
	except:
		precision = -2
	
	try:
		fpr = FP / (FP+TN)
	except:
		fpr = -2
	
	try:
		accuracy = (TP + TN) / (TP + TN + FP + FN)
	except:
		accuracy = -2
	
	try:
		tnr =  TN / (TN+FP)
	except:
		tnr = -2
	
	try:
		f1score = 2*recall*precision / (recall + precision)
	except:
		f1score = -2
	
	try:
		mcc = (TP*TN - FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
	except:
		mcc = -2
	
	mydict = {'recall':recall, 'precision':precision, 'fpr':fpr, 'accuracy':accuracy, 'tnr':tnr, 'f1score':f1score, 'mcc':mcc}
	return mydict