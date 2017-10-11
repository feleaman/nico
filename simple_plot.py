# import os
# import sys
# import pickle
# from tkinter import filedialog
# from tkinter import Tk
# sys.path.insert(0, './lib') #to open user-defined functions
# from m_open_extension import read_pickle
# from argparse import ArgumentParser
# from matplotlib.pyplot import plot, title, savefig, figure, legend
# from numpy import float64, loadtxt


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# recall = [32, 52, 40, 68]

# methods = ['THR', 'CAN', 'WIN', 'ANN']

# recall_error = [4.54, 5., 3.6, 3.27]

# recall_dict = {}
# for key, value in zip(methods, recall):
	# recall_dict[key] = value

# # recall_dict = [recall_dict, recall_dict]

# fig, ax = plt.subplots(nrows=1, ncols=1)

# recall_df = pd.DataFrame.from_dict(recall_dict, orient='index')

# recall_df.plot(use_index=True, kind='bar', capsize=4, rot=0, ax=ax, legend=False)
# N = 4
# width = 0.1
# ind = np.arange(N)

# rect1 = ax.bar(ind, recall)
# rect2 = ax.bar(ind+width, recall, color='y')
# rect2 = ax.bar(ind+width, recall, color='y')


N = 4
men_means = (34, 49, 38, 73)
men_std = (4.54, 5., 3.62, 3.27)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std, )

women_means = (75, 70, 66, 62)
women_std = (9.53, 6.22, 2.76, 2.69)
rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores %')
ax.set_title('Scores in Test Signals')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))

ax.legend((rects1[0], rects2[0]), ('Recall', 'Precision'), loc='lower left')


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()





N = 4
men_means = (47, 57, 48, 67)
men_std = (3.91, 3.55, 3.47, 3.27)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='lightblue', yerr=men_std, )

# women_means = (75, 70, 66, 62)
# women_std = (9.5, 6.2, 2.8, 2.7)
# rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores %')
ax.set_title('F-Score in Test Signals')
ax.set_xticks(ind)
ax.set_xticklabels(('THR', 'EDG', 'WIN', 'ANN'))

# ax.legend((rects1), ('F-Score'), loc='best')


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
# autolabel(rects2)

plt.show()