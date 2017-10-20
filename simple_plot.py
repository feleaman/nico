# import os
# import sys
# import pickle
# from tkinter import filedialog
# from tkinter import Tk
# sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
# from argparse import ArgurecalltParser
# from matplotlib.pyplot import plot, title, savefig, figure, legend
# from numpy import float64, loadtxt


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std):

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, recall_means, width, color='r', yerr=recall_std, error_kw=dict(lw=2, capsize=0, capthick=0))


	rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std, error_kw=dict(lw=2, capsize=0, capthick=0))

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Scores %')
	ax.set_title('Test Results')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))

	ax.legend((rects1[0], rects2[0]), ('Recall', 'Precision'), loc='lower left')


	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2. + 0.09, 1.01*height,
					'%d' % round(height),
					ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	plt.show()

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects3 = ax.bar(ind, fscore_means, width, color='deepskyblue', yerr=fscore_std, error_kw=dict(lw=2, capsize=0, capthick=0))

	# precision_means = (75, 70, 66, 62)
	# precision_std = (9.5, 6.2, 2.8, 2.7)
	# rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Scores %')
	ax.set_title('Test Results')
	ax.set_xticks(ind)
	ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))

	# ax.legend((rects1), ('F-Score'), loc='best')
	ax.legend(('F-Score',), loc='lower left')


	# def autolabel(rects):
		# """
		# Attach a text label above each bar displaying its height
		# """
		# for rect in rects:
			# height = rect.get_height()
			# ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
					# '%d' % int(height),
					# ha='center', va='bottom')

	autolabel(rects3)
	# autolabel(rects2)

	plt.show()
	return

# 'THR', 'EDG', 'WIN', 'ANN'
#SOLO CON 3 TESTS
recall_means = (33.8, 60.4, 54.5, 80.1)
recall_std = (2.73, 0.92, 2.58, 1.43)
precision_means = (74.9, 78.7, 75.5, 69.1)
precision_std = (9.7, 2.04, 5.04, 4.05)
fscore_means = (46.5, 68.4, 63.3, 74.2)
fscore_std = (4.46, 1.19, 3.28, 2.84)
plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)


# #INCLUYENDO 1000 40 
# recall_means = (26.5, 52.1, 41.8, 65.5)
# recall_std = (12.8, 14.47, 22.2, 18.06)
# precision_means = (64.5, 77.6, 81.6, 65.1)
# precision_std = (19.85, 2.63, 11.49, 5.92)
# fscore_means = (36.9, 61.2, 49.2, 62.7)
# fscore_std = (17.01, 12.47, 24.6, 12.9)
# plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)