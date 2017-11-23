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

	fig1, ax = plt.subplots()
	rects1 = ax.bar(ind, recall_means, width, color='r', yerr=recall_std, error_kw=dict(lw=2, capsize=2, capthick=2))


	rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Scores %')
	ax.set_title('Test Results')
	ax.set_xticks(ind + width / 2)
	ax.set_ylim((0, 100))
	ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	ax.legend((rects1[0], rects2[0]), ('Recall', 'Precision'), loc='best')


	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2. + 0.095, 1.000*height,
					'%d' % round(height),
					ha='center', va='bottom')

	autolabel(rects1)
	autolabel(rects2)

	# plt.show()

	N = 4


	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig2, ax = plt.subplots()
	rects3 = ax.bar(ind, fscore_means, width, color='deepskyblue', yerr=fscore_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	# precision_means = (75, 70, 66, 62)
	# precision_std = (9.5, 6.2, 2.8, 2.7)
	# rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std)

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Scores %')
	ax.set_title('Test Results')
	ax.set_xticks(ind)
	ax.set_ylim((0, 100))
	# ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))
	ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'Formerkennung'))

	# ax.legend((rects1), ('F-Score'), loc='best')
	ax.legend(('F-Score',), loc='best')


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
# #SOLO CON 3 TESTS
# recall_means = (33.8, 60.4, 54.5, 80.1)
# recall_std = (2.73, 0.92, 2.58, 1.43)
# precision_means = (74.9, 78.7, 75.5, 69.1)
# precision_std = (9.7, 2.04, 5.04, 4.05)
# fscore_means = (46.5, 68.4, 63.3, 74.2)
# fscore_std = (4.46, 1.19, 3.28, 2.84)
# plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)


#INCLUYENDO 1000 40 
recall_means = (29.2, 45.5, 55.3, 74.7)
recall_std = (8.16, 15.73, 8.88, 9.41)
precision_means = (71.4, 81.6, 81.6, 72.36)
precision_std = (10.30, 11.49, 5.34, 6.58)
fscore_means = (41.1, 55.3, 65.1, 72.7)
fscore_std = (10.08, 14.13, 5.68, 3.53)
plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)






















# def plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std):

	# N = 4


	# ind = np.arange(N)  # the x locations for the groups
	# width = 0.2       # the width of the bars

	# fig1, ax = plt.subplots()
	# rects1 = ax.bar(ind, recall_means, width, color='r')


	# rects2 = ax.bar(ind + width, precision_means, width, color='y')
	
	# rects3 = ax.bar(ind + width + width, precision_means, width, color='g')

	# # add some text for labels, title and axes ticks
	# ax.set_ylabel('Scores %')
	# ax.set_title('Test Results')
	# ax.set_xticks(ind + width / 2)
	# # ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# ax.set_xticklabels(('1500RPM / 80%', '1000RPM / 80%', '1500RPM / 40%', '1000RPM / 40%'))

	# ax.legend((rects1[0], rects2[0] , rects3[0]), ('Recall', 'Precision', 'F-Score'), loc='best')


	# def autolabel(rects):
		# """
		# Attach a text label above each bar displaying its height
		# """
		# for rect in rects:
			# height = rect.get_height()
			# ax.text(rect.get_x() + rect.get_width()/2., 1.000*height,
					# '%d' % round(height),
					# ha='center', va='bottom')

	# autolabel(rects1)
	# autolabel(rects2)
	# autolabel(rects3)

	# plt.show()

	# # N = 4


	# # ind = np.arange(N)  # the x locations for the groups
	# # width = 0.35       # the width of the bars

	# # fig2, ax = plt.subplots()
	# # rects3 = ax.bar(ind, fscore_means, width, color='deepskyblue', yerr=fscore_std, error_kw=dict(lw=2, capsize=2, capthick=2))

	# # # precision_means = (75, 70, 66, 62)
	# # # precision_std = (9.5, 6.2, 2.8, 2.7)
	# # # rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std)

	# # # add some text for labels, title and axes ticks
	# # ax.set_ylabel('Scores %')
	# # ax.set_title('Test Results')
	# # ax.set_xticks(ind)
	# # # ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))
	# # ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# # ax.set_xticklabels(('Grenzwert Amp.', 'Fensterung', 'Kantendetektion', 'NN-Kantenerkennung'))

	# # # ax.legend((rects1), ('F-Score'), loc='best')
	# # ax.legend(('F-Score',), loc='best')


	# # # def autolabel(rects):
		# # # """
		# # # Attach a text label above each bar displaying its height
		# # # """
		# # # for rect in rects:
			# # # height = rect.get_height()
			# # # ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
					# # # '%d' % int(height),
					# # # ha='center', va='bottom')

	# # autolabel(rects3)
	# # # autolabel(rects2)

	# # plt.show()
	# return

# # 'THR', 'EDG', 'WIN', 'ANN'
# # #SOLO CON 3 TESTS
# # recall_means = (33.8, 60.4, 54.5, 80.1)
# # recall_std = (2.73, 0.92, 2.58, 1.43)
# # precision_means = (74.9, 78.7, 75.5, 69.1)
# # precision_std = (9.7, 2.04, 5.04, 4.05)
# # fscore_means = (46.5, 68.4, 63.3, 74.2)
# # fscore_std = (4.46, 1.19, 3.28, 2.84)
# # plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)


# #INCLUYENDO 1000 40 
# recall_means = (29.2, 45.5, 55.3, 74.7)
# recall_std = (8.16, 15.73, 8.88, 9.41)
# precision_means = (71.4, 81.6, 81.6, 72.36)
# precision_std = (10.30, 11.49, 5.34, 6.58)
# fscore_means = (41.1, 55.3, 65.1, 72.7)
# fscore_std = (10.08, 14.13, 5.68, 3.53)
# plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)










# def plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std):

	# N = 4


	# ind = np.arange(N)  # the x locations for the groups
	# width = 0.2      # the width of the bars

	# fig1, ax = plt.subplots()
	# rects1 = ax.bar(ind, recall_means, width, color='r')


	# rects2 = ax.bar(ind + width, precision_means, width, color='y')
	
	# rects3 = ax.bar(ind + 2*width, fscore_means, width, color='deepskyblue')

	# # add some text for labels, title and axes ticks
	# ax.set_ylabel('Scores %')
	# ax.set_title('Test Results')
	# ax.set_ylim((0, 100))
	# ax.set_xticks(ind + width / 2)
	# # ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# ax.set_xticklabels(('1500RPM / 80%', '1000RPM / 80%', '1500RPM / 40%', '1000RPM / 40%'))

	# ax.legend((rects1[0], rects2[0], rects3[0]), ('Recall', 'Precision', 'F-Score'), loc='lower left')


	# def autolabel(rects):
		# """
		# Attach a text label above each bar displaying its height
		# """
		# for rect in rects:
			# height = rect.get_height()
			# ax.text(rect.get_x() + rect.get_width()/2., 1.000*height,
					# '%d' % round(height),
					# ha='center', va='bottom')

	# autolabel(rects1)
	# autolabel(rects2)
	# autolabel(rects3)

	# plt.show()

	# # N = 4


	# # ind = np.arange(N)  # the x locations for the groups
	# # width = 0.35       # the width of the bars

	# # fig2, ax = plt.subplots()
	# # rects3 = ax.bar(ind, fscore_means, width, color='deepskyblue')

	# # # precision_means = (75, 70, 66, 62)
	# # # precision_std = (9.5, 6.2, 2.8, 2.7)
	# # # rects2 = ax.bar(ind + width, precision_means, width, color='y', yerr=precision_std)

	# # # add some text for labels, title and axes ticks
	# # ax.set_ylabel('Scores %')
	# # ax.set_title('Test Results')
	# # ax.set_xticks(ind)
	# # ax.set_ylim((0, 100))
	# # # ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))
	# # ax.set_xticklabels(('THR', 'WIN', 'EDG', 'ENN'))
	# # ax.set_xticklabels(('1500RPM / 80%', '1000RPM / 80%', '1500RPM / 40%', '1000RPM / 40%'))

	# # # ax.legend((rects1), ('F-Score'), loc='best')
	# # ax.legend(('F-Score',), loc='best')


	# # # def autolabel(rects):
		# # # """
		# # # Attach a text label above each bar displaying its height
		# # # """
		# # # for rect in rects:
			# # # height = rect.get_height()
			# # # ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
					# # # '%d' % int(height),
					# # # ha='center', va='bottom')

	# # autolabel(rects3)
	# # # autolabel(rects2)

	# # plt.show()
	# return

# # 'THR', 'EDG', 'WIN', 'ANN'
# # #SOLO CON 3 TESTS
# # recall_means = (33.8, 60.4, 54.5, 80.1)
# recall_std = (2.73, 0.92, 2.58, 1.43)
# # precision_means = (74.9, 78.7, 75.5, 69.1)
# precision_std = (9.7, 2.04, 5.04, 4.05)
# # fscore_means = (46.5, 68.4, 63.3, 74.2)
# fscore_std = (4.46, 1.19, 3.28, 2.84)
# # plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)


# #INCLUYENDO 1000 40 
# # recall_means = (37.6, 31.4, 32.3, 15.7)
# # precision_means = (88.3, 65.9, 70.5, 61.1)
# # fscore_means = (52.7, 42.5, 44.3, 25.0)


# # recall_means = (58.2, 52.3, 53.1, 18.6)
# # precision_means = (81.2, 76.3, 68.9, 100.0)
# # fscore_means = (67.8, 62.1, 60.0, 31.3)


# # recall_means = (60.3, 61.6, 59.4, 40.0)
# # precision_means = (81.0, 79.1, 76.0, 90.3)
# # fscore_means = (69.1, 69.3, 66.7, 55.4)


# recall_means = (80.9, 81.4, 78.1, 58.6)
# precision_means = (74.0, 69.3, 64.1, 82.0)
# fscore_means = (77.3, 74.9, 70.4, 68.3)
# plot_scores_error(recall_means, recall_std, precision_means, precision_std, fscore_means, fscore_std)
